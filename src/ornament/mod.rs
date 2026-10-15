//! Layer 3 — shape-aware ornamentation: contour stripes, hatching, stippling and clipping
//! generated in the parameter space of a [`Surface`](crate::geom::Surface).
//!
//! Every generator works in the `(u, v)` space of a surface (`u` along the shape, `v`
//! across it), so lines bend, converge and taper with the shape itself. All outputs are
//! finite, free of zero-length segments and deterministic for a given seed.

pub mod contour;
pub mod engrave;
pub mod hatch;
pub mod stipple;

pub use contour::{nested, spaced_stripes, stripes, trimmed_stripes};
pub use engrave::{EngraveOptions, engrave};
pub use hatch::{HatchOptions, Shade, cross_hatch, hatch};
pub use stipple::{stipple, stipple_path};

use crate::geom::validate::{degenerate_segments, path_is_finite, sanitize};
use crate::geom::{Curve, CurveExt, EPS, Surface};
use kurbo::{Affine, BezPath, Point, Vec2};

/// Number of `u` samples used to scan a surface's half-width profile.
const PROFILE_SAMPLES: usize = 128;

/// Fitting accuracy never drops below this fraction of a curve's size: at huge
/// coordinates an absolute tolerance is below `f64` resolution and the fitter never ends.
const REL_ACCURACY: f64 = 1e-9;

/// Accuracy actually used for Bézier fitting: invalid values fall back to `0.1`, tiny
/// ones are raised so the fitter cannot be asked for sub-nanometre precision.
pub(crate) fn fit_accuracy(accuracy: f64) -> f64 {
    if accuracy.is_finite() && accuracy > 0.0 {
        accuracy.max(1e-3)
    } else {
        0.1
    }
}

/// Largest half-width of the surface (canvas units), `0` if it is degenerate.
pub(crate) fn max_half_width<S: Surface + ?Sized>(surface: &S) -> f64 {
    (0..=PROFILE_SAMPLES)
        .map(|i| surface.half_width(i as f64 / PROFILE_SAMPLES as f64))
        .filter(|w| w.is_finite())
        .fold(0.0, f64::max)
}

/// Half-width at `u`, with non-finite values mapped to `0`.
pub(crate) fn half_width_at<S: Surface + ?Sized>(surface: &S, u: f64) -> f64 {
    let w = surface.half_width(u.clamp(0.0, 1.0));
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

/// Maximal sub-intervals of `[0, 1]` on which `pred` holds.
///
/// `pred` is scanned at `n + 1` evenly spaced points; every true/false transition is then
/// refined by bisection, so interval ends are accurate to ~`1e-6` regardless of `n`.
pub(crate) fn runs(pred: impl Fn(f64) -> bool, n: usize) -> Vec<(f64, f64)> {
    let n = n.max(2);
    let at = |i: usize| i as f64 / n as f64;
    // Bisect between a parameter where `pred == inside_at_lo` and one where it differs.
    let refine = |mut lo: f64, mut hi: f64, inside_at_lo: bool| {
        for _ in 0..20 {
            let mid = 0.5 * (lo + hi);
            if pred(mid) == inside_at_lo {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if inside_at_lo { lo } else { hi }
    };
    let flags: Vec<bool> = (0..=n).map(|i| pred(at(i))).collect();
    let mut out = Vec::new();
    let mut start: Option<f64> = None;
    for (i, &flag) in flags.iter().enumerate() {
        match (flag, start) {
            (true, None) => {
                start = Some(if i == 0 {
                    0.0
                } else {
                    refine(at(i - 1), at(i), false)
                });
            }
            (false, Some(s)) => {
                out.push((s, refine(at(i - 1), at(i), true)));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, 1.0));
    }
    out.retain(|(a, b)| b - a > 1e-9);
    out
}

/// Approximate canvas length of a curve from `n` chords (non-finite chords count as 0).
pub(crate) fn chord_length<C: Curve + ?Sized>(curve: &C, n: usize) -> f64 {
    let pts = curve.sample(n.max(2));
    pts.windows(2)
        .map(|w| (w[1] - w[0]).hypot())
        .filter(|d| d.is_finite())
        .sum()
}

/// Fit a curve to a clean path, or `None` if it is non-finite, too short or degenerate.
///
/// The curve is probed before fitting so the fitter never sees NaN; the result is
/// sanitised so it carries no zero-length segments.
pub(crate) fn fit_clean<C: Curve + ?Sized>(
    curve: &C,
    min_len: f64,
    accuracy: f64,
) -> Option<BezPath> {
    const PROBES: usize = 33;
    if !curve.sample(PROBES).iter().all(|p| p.is_finite()) {
        return None;
    }
    let len = chord_length(curve, PROBES);
    if !len.is_finite() || len < min_len.max(EPS * 1e3) {
        return None;
    }
    // Fit at unit size, then map back: an absolute tolerance at huge coordinates is below
    // `f64` resolution, and the fitter then never finishes (or overflows).
    let origin = curve.sample(2)[0];
    let unit = Normalised {
        curve,
        origin,
        scale: len,
    };
    let acc = (fit_accuracy(accuracy) / len).max(REL_ACCURACY);
    let mut fitted = unit.to_bezpath(acc);
    fitted.apply_affine(Affine::translate(origin.to_vec2()) * Affine::scale(len));
    let path = sanitize(&fitted, 1e-6);
    (path.elements().len() >= 2 && path_is_finite(&path) && degenerate_segments(&path, 1e-6) == 0)
        .then_some(path)
}

/// `(curve − origin) / scale`, keeping the curve's own derivative.
struct Normalised<'a, C: ?Sized> {
    curve: &'a C,
    origin: Point,
    scale: f64,
}

impl<C: Curve + ?Sized> Curve for Normalised<'_, C> {
    fn point(&self, t: f64) -> Point {
        ((self.curve.point(t) - self.origin) / self.scale).to_point()
    }
    fn deriv(&self, t: f64) -> Vec2 {
        self.curve.deriv(t) / self.scale
    }
}

/// Golden-ratio low-discrepancy sequence in `[0, 1)`: evenly spread, never periodic, so
/// per-line variations (thresholds, drop-out points) never form a visible regular pattern.
pub(crate) fn golden(i: usize, phase: f64) -> f64 {
    const PHI: f64 = 0.618_033_988_749_894_9;
    (phase + i as f64 * PHI).rem_euclid(1.0)
}

#[cfg(test)]
pub(crate) mod test_shapes {
    //! Shared test surfaces.
    use crate::geom::{RuledSurface, Surface};
    use kurbo::{BezPath, CubicBez, Point, Shape};

    /// A petal between two cubics meeting at base `(0,0)` and tip `(0,160)`.
    pub fn petal() -> RuledSurface<CubicBez, CubicBez> {
        RuledSurface {
            left: CubicBez::new((0.0, 0.0), (-70.0, 40.0), (-55.0, 120.0), (0.0, 160.0)),
            right: CubicBez::new((0.0, 0.0), (70.0, 40.0), (55.0, 120.0), (0.0, 160.0)),
        }
    }

    /// A curved leaf with a blunt base and a sharply pinched tip at `u = 1`, built from a
    /// closure: the centreline bends to the right and the width falls as `(1-u)^1.5`.
    pub struct Leaf;

    impl Surface for Leaf {
        fn point(&self, u: f64, v: f64) -> Point {
            let u = u.clamp(0.0, 1.0);
            let c = Point::new(60.0 * u * u, 200.0 * u);
            // centreline tangent (120u, 200) → left normal (-200, 120u) normalised
            let (tx, ty) = (120.0 * u, 200.0);
            let l = (tx * tx + ty * ty).sqrt();
            let n = (-ty / l, tx / l);
            let hw = 30.0 * (1.0 - u).powf(1.5) * (0.6 + 0.4 * (std::f64::consts::PI * u).sin());
            Point::new(c.x + n.0 * hw * v, c.y + n.1 * hw * v)
        }
    }

    /// A surface with zero width everywhere.
    pub struct Needle;

    impl Surface for Needle {
        fn point(&self, u: f64, _v: f64) -> Point {
            Point::new(0.0, 100.0 * u)
        }
    }

    /// A twisted ribbon: the edges swap sides at `u = 0.5`, so the outline is a
    /// figure-eight whose signed area is zero although the shape is not degenerate.
    pub struct Bow;

    impl Surface for Bow {
        fn point(&self, u: f64, v: f64) -> Point {
            Point::new(100.0 * u, 20.0 * v * (1.0 - 2.0 * u))
        }
    }

    /// A pointed shape at astronomically large coordinates.
    pub struct Huge;

    impl Surface for Huge {
        fn point(&self, u: f64, v: f64) -> Point {
            Point::new(1e150 * u, 1e149 * v * (1.0 - u))
        }
    }

    /// Fraction of `pts` strictly inside `outline` (non-zero winding).
    pub fn inside_fraction(outline: &BezPath, pts: &[Point]) -> f64 {
        if pts.is_empty() {
            return 1.0;
        }
        let n = pts.iter().filter(|p| outline.winding(**p) != 0).count();
        n as f64 / pts.len() as f64
    }

    /// Points densely sampled along every segment of every path.
    pub fn path_samples(paths: &[BezPath]) -> Vec<Point> {
        use kurbo::ParamCurve;
        let mut out = Vec::new();
        for p in paths {
            for seg in p.segments() {
                for i in 0..=8 {
                    out.push(seg.eval(i as f64 / 8.0));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_finds_refined_intervals() {
        let r = runs(|t| (0.2..0.4).contains(&t) || t > 0.9, 16);
        assert_eq!(r.len(), 2);
        assert!((r[0].0 - 0.2).abs() < 1e-5 && (r[0].1 - 0.4).abs() < 1e-5);
        assert!((r[1].0 - 0.9).abs() < 1e-5 && r[1].1 == 1.0);
        assert!(runs(|_| false, 8).is_empty());
        assert_eq!(runs(|_| true, 8), vec![(0.0, 1.0)]);
    }

    #[test]
    fn fit_clean_terminates_at_huge_coordinates() {
        let wave = |k: f64| {
            crate::geom::FnCurve(move |t: f64| kurbo::Point::new(k * t, k * (3.0 * t).sin()))
        };
        for k in [1e12, 1e50, 1e150] {
            let p = fit_clean(&wave(k), 0.0, 0.05).unwrap_or_else(|| panic!("no fit at {k}"));
            assert!(path_is_finite(&p));
        }
        // lengths overflow beyond ~1e154 (`hypot` squares): rejected, but no panic
        assert!(fit_clean(&wave(1e200), 0.0, 0.05).is_none());
    }

    #[test]
    fn fit_clean_rejects_nan_and_points() {
        let nan = crate::geom::FnCurve(|_t: f64| kurbo::Point::new(f64::NAN, 0.0));
        assert!(fit_clean(&nan, 0.0, 0.1).is_none());
        let dot = crate::geom::FnCurve(|_t: f64| kurbo::Point::new(1.0, 2.0));
        assert!(fit_clean(&dot, 0.0, 0.1).is_none());
        let line = kurbo::Line::new((0.0, 0.0), (10.0, 0.0));
        assert!(fit_clean(&line, 0.0, f64::NAN).is_some());
        assert!(fit_clean(&line, 20.0, 0.1).is_none());
    }
}
