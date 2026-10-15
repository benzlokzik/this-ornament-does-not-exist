//! Nested contour stripes: curves of constant `v` that run along a surface parallel to
//! its edges and its midrib, trimmed where the shape pinches so they taper into tips
//! instead of piling up on top of each other.

use super::{PROFILE_SAMPLES, fit_clean, golden, half_width_at, max_half_width, runs};
use crate::geom::{CurveExt, EPS, Surface};
use kurbo::{BezPath, Point};

/// Most stripes a single call produces per side, so absurd spacings stay bounded.
const MAX_PER_SIDE: usize = 64;

/// Stripes along `iso_v(v)` for every `v` in `vs`, trimmed at pinches automatically.
///
/// The trimming gap is derived from the stripe set itself: half of the smallest spacing
/// between neighbouring `|v|` levels (the centre line counts as a level), measured in
/// canvas units at the widest point of the surface. See [`trimmed_stripes`].
pub fn stripes<S: Surface + ?Sized>(surface: &S, vs: &[f64], accuracy: f64) -> Vec<BezPath> {
    let gap = 0.5 * max_half_width(surface) * level_step(vs);
    trimmed_stripes(surface, vs, gap, accuracy)
}

/// Smallest positive spacing between the sorted `|v|` levels of `vs` (the centre line
/// counts as a level), or `1` if there is none.
fn level_step(vs: &[f64]) -> f64 {
    let mut levels: Vec<f64> = vs
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs().min(1.0))
        .chain([0.0])
        .collect();
    levels.sort_by(f64::total_cmp);
    levels
        .windows(2)
        .map(|p| p[1] - p[0])
        .filter(|d| *d > 1e-6)
        .fold(1.0, f64::min)
}

/// Stripes along `iso_v(v)` for every `v` in `vs`, kept only where
/// `half_width(u) · |v| >= min_gap` and where the stripe would not crowd its neighbours.
///
/// `half_width(u)·|v|` is the canvas distance between the stripe and the centre line, so
/// near a pointed tip the inner stripes stop first and the outer ones run further: the
/// stripes taper into the tip as a nested fan. To keep the outer stripes of a dense set
/// from converging into a dark blot, stripes also drop out hierarchically: with `Δv`
/// the set's level step and `k = |v|/Δv`, stripe `k` needs
/// `half_width(u) · Δv · 2^tz(k) >= 0.35 · W · Δv` (`tz` = trailing zero bits, `W` =
/// largest half-width, with a small per-stripe stagger), so odd stripes end first, then
/// every other remaining one, and the local spacing stays above about a third of the
/// spacing at the widest point.
///
/// The centre line (`v = 0`) is treated as a midrib and is kept wherever the full
/// half-width is at least `min_gap`. No stripe comes within `min_gap` of a pinched end
/// (an end narrower than `min_gap`), however blunt the tip. A stripe whose valid `u`-range is split (a waisted
/// shape) yields one path per piece. Non-finite `v` are skipped, others clamped to
/// `[-1, 1]`; a non-finite or negative `min_gap` disables the `|v|` trimming.
pub fn trimmed_stripes<S: Surface + ?Sized>(
    surface: &S,
    vs: &[f64],
    min_gap: f64,
    accuracy: f64,
) -> Vec<BezPath> {
    let w = max_half_width(surface);
    if w < EPS {
        return Vec::new();
    }
    let gap = if min_gap.is_finite() {
        min_gap.max(0.0)
    } else {
        0.0
    };
    let step = level_step(vs);
    // Pinched ends: where the shape is narrower than the gap. The width rule alone lets
    // the centre line run to within `min_gap / tan(half tip angle)` of a blunt tip.
    let tips: Vec<Point> = [0.0, 1.0]
        .into_iter()
        .filter(|&u| half_width_at(surface, u) < gap)
        .map(|u| surface.point(u, 0.0))
        .filter(|p| p.is_finite())
        .collect();
    let mut out = Vec::new();
    for &v in vs.iter().filter(|v| v.is_finite()) {
        let v = v.clamp(-1.0, 1.0);
        let (reach, crowd) = if v.abs() < 1e-6 {
            (1.0, 0.0)
        } else {
            let k = ((v.abs() / step).round() as u64).max(1);
            let level = k.trailing_zeros().min(6);
            let stagger = 0.8 + 0.4 * golden(k as usize, 0.29);
            (v.abs(), 0.35 * w * stagger / f64::from(1u32 << level))
        };
        let keep = |u: f64| {
            let hw = half_width_at(surface, u);
            hw * reach >= gap
                && hw >= crowd
                && tips
                    .iter()
                    .all(|t| (surface.point(u, v) - *t).hypot() >= gap)
        };
        for (a, b) in runs(keep, PROFILE_SAMPLES) {
            let piece = surface.iso_v(v).sub(a..b);
            if let Some(path) = fit_clean(&piece, 0.0, accuracy) {
                out.push(path);
            }
        }
    }
    out
}

/// `n` stripes on each side of the centre line at `v = ±(k/n)·(1 − inset)`, `k = 1..=n`,
/// plus the centre line itself when `centre` is set.
///
/// `inset` (clamped to `[0, 0.95]`) keeps the outermost stripe away from the outline.
/// Stripes are trimmed at pinches as in [`stripes`]. `n` is capped at 64.
pub fn nested<S: Surface + ?Sized>(
    surface: &S,
    n: usize,
    inset: f64,
    centre: bool,
    accuracy: f64,
) -> Vec<BezPath> {
    let n = n.min(MAX_PER_SIDE);
    let inset = if inset.is_finite() {
        inset.clamp(0.0, 0.95)
    } else {
        0.0
    };
    let mut vs = Vec::with_capacity(2 * n + 1);
    if centre {
        vs.push(0.0);
    }
    for k in 1..=n {
        let v = k as f64 / n as f64 * (1.0 - inset);
        vs.extend([v, -v]);
    }
    stripes(surface, &vs, accuracy)
}

/// Stripes spaced `spacing` canvas units apart at the widest point of the surface, from
/// the centre line outwards (the outline itself is not repeated).
///
/// Elsewhere the spacing shrinks in proportion to the local width, as engraved contour
/// lines do; stripes are trimmed where they would come closer than `spacing / 2` to the
/// centre line. A non-finite, zero or negative `spacing` is clamped to the densest
/// allowed value (64 stripes per side).
pub fn spaced_stripes<S: Surface + ?Sized>(
    surface: &S,
    spacing: f64,
    accuracy: f64,
) -> Vec<BezPath> {
    let w = max_half_width(surface);
    if w < EPS {
        return Vec::new();
    }
    let min_spacing = w / MAX_PER_SIDE as f64;
    let spacing = if spacing.is_finite() {
        spacing.max(min_spacing)
    } else {
        min_spacing
    };
    let step = spacing / w;
    let mut vs = vec![0.0];
    let mut k = 1;
    // Stop a quarter step short of the edge so no stripe doubles the outline.
    while k as f64 * step < 1.0 - 0.25 * step {
        let v = k as f64 * step;
        vs.extend([v, -v]);
        k += 1;
    }
    trimmed_stripes(surface, &vs, 0.5 * spacing, accuracy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::ornament::test_shapes::{Huge, Leaf, Needle, inside_fraction, path_samples, petal};
    use kurbo::{Line, ParamCurve, PathSeg};

    fn assert_clean(paths: &[BezPath]) {
        for p in paths {
            assert!(path_is_finite(p));
            assert_eq!(degenerate_segments(p, 1e-6), 0);
            assert!(p.elements().len() >= 2);
        }
    }

    #[test]
    fn nested_stripes_are_clean_and_inside() {
        for s in [&petal() as &dyn Surface, &Leaf] {
            let paths = nested(s, 4, 0.12, true, 0.05);
            assert!(paths.len() >= 9, "{}", paths.len());
            assert_clean(&paths);
            let outline = s.outline(0.02);
            let f = inside_fraction(&outline, &path_samples(&paths));
            assert!(f >= 0.99, "inside fraction {f}");
        }
    }

    #[test]
    fn stripes_taper_before_the_tip() {
        let leaf = Leaf;
        let tip = leaf.point(1.0, 0.0);
        let gap = 4.0;
        let vs = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9];
        let paths = trimmed_stripes(&leaf, &vs, gap, 0.05);
        assert_eq!(paths.len(), vs.len());
        let mut ends = Vec::new();
        for p in &paths {
            let segs: Vec<PathSeg> = p.segments().collect();
            ends.push(segs[0].eval(0.0));
            ends.push(segs[segs.len() - 1].eval(1.0));
        }
        for e in &ends {
            assert!((*e - tip).hypot() >= gap, "endpoint {e:?} too close to tip");
        }
        // the innermost stripe stops first: compare the tip-side end heights
        let tip_end = |i: usize| ends[2 * i + 1].y;
        assert!(tip_end(1) < tip_end(3) && tip_end(1) < tip_end(5));
    }

    #[test]
    fn stripes_stop_short_of_blunt_tips() {
        // the petal's pinches are blunter than 90°, where `hw · |v| >= gap` alone lets
        // the centre line run to within `gap / tan(half angle)` of the tip
        let s = petal();
        let pinches = [s.point(0.0, 0.0), s.point(1.0, 0.0)];
        let vs = [0.0, 0.3, -0.3, 0.6, -0.6, 0.9, -0.9];
        let gap = 4.0;
        let paths = trimmed_stripes(&s, &vs, gap, 0.05);
        assert_eq!(paths.len(), vs.len());
        assert_clean(&paths);
        for p in &paths {
            let segs: Vec<PathSeg> = p.segments().collect();
            for e in [segs[0].eval(0.0), segs[segs.len() - 1].eval(1.0)] {
                for t in pinches {
                    assert!((e - t).hypot() >= gap - 1e-3, "endpoint {e:?} near {t:?}");
                }
            }
        }
    }

    #[test]
    fn spaced_stripes_are_uniform_at_the_widest_point() {
        let s = petal();
        let spacing = 6.0;
        let paths = spaced_stripes(&s, spacing, 0.02);
        assert_clean(&paths);
        // the petal is symmetric, so its widest section is at u = 0.5
        let across = Line::new(s.point(0.5, -1.0), s.point(0.5, 1.0));
        let mut hits: Vec<f64> = Vec::new();
        for p in &paths {
            for seg in p.segments() {
                for h in seg.intersect_line(across) {
                    hits.push(h.line_t);
                }
            }
        }
        hits.sort_by(f64::total_cmp);
        assert!(hits.len() >= 9, "{hits:?}");
        let len = (across.p1 - across.p0).hypot();
        let gaps: Vec<f64> = hits.windows(2).map(|w| (w[1] - w[0]) * len).collect();
        let w_max = max_half_width(&s);
        let expected = spacing * s.half_width(0.5) / w_max;
        for g in gaps {
            assert!(
                (g - expected).abs() < 0.05 * expected,
                "gap {g} vs {expected}"
            );
        }
    }

    #[test]
    fn reproducible() {
        let a = nested(&Leaf, 3, 0.1, true, 0.05);
        let b = nested(&Leaf, 3, 0.1, true, 0.05);
        assert_eq!(a, b);
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert!(nested(&Needle, 4, 0.1, true, 0.05).is_empty());
        assert!(spaced_stripes(&Needle, 3.0, 0.05).is_empty());
        let s = petal();
        for sp in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let p = spaced_stripes(&s, sp, 0.1);
            assert!(p.len() <= 2 * MAX_PER_SIDE + 1);
            assert_clean(&p);
        }
        let p = stripes(&s, &[f64::NAN, 7.0, -0.5], f64::NAN);
        assert_eq!(p.len(), 2);
        assert_clean(&p);
        assert_clean(&trimmed_stripes(&s, &[0.5], f64::NAN, 0.1));
        assert!(trimmed_stripes(&s, &[0.5], 1e9, 0.1).is_empty());
        assert_clean(&nested(&s, 10_000, f64::NAN, false, 0.1));
        let tiny = trimmed_stripes(&s, &[0.5], 0.0, 0.1);
        assert_eq!(tiny.len(), 1);
        // huge coordinates: an absolute accuracy used to make the fitter spin forever
        let huge = nested(&Huge, 2, 0.1, true, 0.05);
        assert!(!huge.is_empty());
        assert_clean(&huge);
    }
}
