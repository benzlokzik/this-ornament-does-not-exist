//! Surface-following hatching that reads as volume.
//!
//! Hatch lines are straight in a *normalised* parameter space and bend with the shape
//! once mapped through the surface. The normalised space has coordinates
//! `X = arc length of the centre line iso_v(0)` and `Y = v · W` (`W` = largest
//! half-width), both in canvas units, so angles and spacings mean the same thing on a
//! long thin leaf and on a round petal.
//!
//! A smooth [`Shade`] field `s(u, v) ∈ [0, 1]` (1 = darkest) drives two mechanisms:
//! - **length** (lines more across than along, `45° ≤ angle ≤ 135°`): each line keeps only
//!   the part where `s` exceeds a per-line threshold. Thresholds are spread with a
//!   golden-ratio sequence, so on an [`Shade::Edge`] field lines start at the dark edge
//!   and reach inward by an amount that follows the shade, ending at staggered points
//!   that fade smoothly instead of forming a hard front.
//! - **density** (lines more along than across): lines are placed with a local spacing
//!   `∝ 1 / (0.2 + s̄)`, where `s̄` is the shade averaged along the line, and are also
//!   length-modulated with low thresholds. Where the shape narrows, lines drop out
//!   hierarchically (every other line first, then every fourth, …) so they never pile up
//!   at a pinched tip.

use super::{fit_clean, golden, half_width_at, max_half_width, runs};
use crate::geom::{ArcLength, EPS, FnCurve, Surface, lerp, smoothstep};
use crate::rng::{Draw, SeedTree};
use kurbo::BezPath;
use std::fmt;
use std::sync::Arc;

/// Lines never reach closer than this to the edges (in `|v|`), so they stay inside the
/// outline even after Bézier fitting; the outline stroke covers the hairline gap.
const V_MAX: f64 = 0.98;

/// Upper bound on the number of lines in one hatch layer.
const MAX_LINES: usize = 600;

/// A smooth shading field `s(u, v) ∈ [0, 1]` over a surface (`1` = darkest).
#[derive(Clone)]
pub enum Shade {
    /// Constant level (clamped to `[0, 1]`).
    Uniform(f64),
    /// Dark along the edge `v = side` (sign of `side`; `+1` or `-1`), fading smoothly
    /// across the shape and reaching zero `width` of the full width away
    /// (`width` is a fraction in `(0, 2]`; `1` fades over the whole width).
    Edge { side: f64, width: f64 },
    /// Dark at both ends (`u = 0` and `u = 1`), fading to zero `width` (in `u`) inward.
    Ends { width: f64 },
    /// Arbitrary field; results are clamped to `[0, 1]` and NaN counts as `0`.
    Fn(Arc<dyn Fn(f64, f64) -> f64 + Send + Sync>),
}

impl Shade {
    /// Wrap a closure as a [`Shade::Fn`].
    pub fn func(f: impl Fn(f64, f64) -> f64 + Send + Sync + 'static) -> Shade {
        Shade::Fn(Arc::new(f))
    }

    /// Shade at `(u, v)`, always finite and in `[0, 1]`.
    pub fn at(&self, u: f64, v: f64) -> f64 {
        let width = |w: f64| {
            if w.is_finite() {
                w.clamp(0.02, 2.0)
            } else {
                0.5
            }
        };
        let s = match self {
            Shade::Uniform(level) => *level,
            Shade::Edge { side, width: w } => {
                let side = if *side < 0.0 { -1.0 } else { 1.0 };
                // distance from the shaded edge as a fraction of the full width
                let d = (v - side).abs() * 0.5;
                1.0 - smoothstep(d / width(*w))
            }
            Shade::Ends { width: w } => 1.0 - smoothstep(u.min(1.0 - u) / width(*w)),
            Shade::Fn(f) => f(u, v),
        };
        if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl fmt::Debug for Shade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shade::Uniform(l) => f.debug_tuple("Uniform").field(l).finish(),
            Shade::Edge { side, width } => f
                .debug_struct("Edge")
                .field("side", side)
                .field("width", width)
                .finish(),
            Shade::Ends { width } => f.debug_struct("Ends").field("width", width).finish(),
            Shade::Fn(_) => f.write_str("Fn(..)"),
        }
    }
}

/// Hatching parameters. Out-of-range values are clamped, never rejected.
#[derive(Clone, Debug)]
pub struct HatchOptions {
    /// Line spacing in canvas units: along the centre line for across-lines, across the
    /// widest section for along-lines (where it is the spacing at full shade).
    /// Clamped so one layer has at most a few hundred lines.
    pub spacing: f64,
    /// Line direction in degrees in the normalised parameter space: `0` runs along the
    /// shape (`u`), `90` across it (`v`).
    pub angle: f64,
    /// Lines (or line pieces) shorter than this, in canvas units, are dropped.
    pub min_len: f64,
    /// Irregularity in `[0, 1]`: position, angle and threshold randomisation.
    pub jitter: f64,
    /// Fraction of the length (in `[0, 0.5]`) at each end of the shape over which the
    /// shading fades out, so hatching dissolves into tips instead of stopping abruptly.
    pub taper: f64,
    /// The shading field.
    pub shade: Shade,
}

impl Default for HatchOptions {
    fn default() -> Self {
        HatchOptions {
            spacing: 3.0,
            angle: 90.0,
            min_len: 1.5,
            jitter: 0.3,
            taper: 0.12,
            shade: Shade::Edge {
                side: -1.0,
                width: 0.8,
            },
        }
    }
}

/// Hatch lines over `surface`; randomness comes from `seed.child("hatch")`.
pub fn hatch<S: Surface + ?Sized>(
    surface: &S,
    opts: &HatchOptions,
    seed: &SeedTree,
    accuracy: f64,
) -> Vec<BezPath> {
    layer(
        surface,
        opts,
        opts.angle,
        0.0,
        seed.child("hatch"),
        accuracy,
    )
}

/// Cross-hatching: the [`hatch`] layer plus a second layer at `second_angle` degrees
/// that only appears where the shade is deep (`s` above ~0.35), as in engraving where
/// the second pass darkens the shadow core. The first layer equals [`hatch`] output.
pub fn cross_hatch<S: Surface + ?Sized>(
    surface: &S,
    opts: &HatchOptions,
    second_angle: f64,
    seed: &SeedTree,
    accuracy: f64,
) -> Vec<BezPath> {
    let mut out = hatch(surface, opts, seed, accuracy);
    out.extend(layer(
        surface,
        opts,
        second_angle,
        0.35,
        seed.child("cross"),
        accuracy,
    ));
    out
}

/// The normalised frame of a surface plus clamped options.
struct Frame<'a, S: ?Sized> {
    surface: &'a S,
    shade: &'a Shade,
    /// Arc-length table of the centre line.
    al: ArcLength,
    /// Centre-line length `L` and largest half-width `W`.
    len: f64,
    w: f64,
    spacing: f64,
    min_len: f64,
    jitter: f64,
    taper: f64,
}

impl<S: Surface + ?Sized> Frame<'_, S> {
    /// `u` at centre-line arc length `x`.
    fn u_at(&self, x: f64) -> f64 {
        self.al.t_at(x)
    }

    /// Shade multiplied by the end taper envelope.
    fn shade(&self, x: f64, u: f64, v: f64) -> f64 {
        let env = if self.taper > 0.0 {
            smoothstep(x.min(self.len - x) / (self.taper * self.len))
        } else {
            1.0
        };
        self.shade.at(u, v) * env
    }

    /// Fit the piece `t ∈ [a, b]` of a line given as `t ↦ (X, Y)`.
    fn piece(
        &self,
        line: &impl Fn(f64) -> (f64, f64),
        a: f64,
        b: f64,
        acc: f64,
    ) -> Option<BezPath> {
        let curve = FnCurve(|tau: f64| {
            let (x, y) = line(lerp(a, b, tau));
            let x = x.clamp(0.0, self.len);
            self.surface
                .point(self.u_at(x), (y / self.w).clamp(-V_MAX, V_MAX))
        });
        fit_clean(&curve, self.min_len, acc)
    }

    /// Per-line threshold in `[lo, hi]`: golden-ratio spread blended with noise by `jitter`.
    fn threshold(&self, i: usize, noise: f64, lo: f64, hi: f64, bias: f64) -> f64 {
        let q = lerp(golden(i, 0.137), noise, self.jitter);
        bias + (1.0 - bias) * lerp(lo, hi, q)
    }
}

fn clamp_or(x: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        fallback
    }
}

/// One layer of hatching at `angle` degrees; `bias` lifts all thresholds so the layer
/// only covers darker regions.
fn layer<S: Surface + ?Sized>(
    surface: &S,
    opts: &HatchOptions,
    angle: f64,
    bias: f64,
    seed: SeedTree,
    accuracy: f64,
) -> Vec<BezPath> {
    let w = max_half_width(surface);
    let centre = surface.iso_v(0.0);
    let al = ArcLength::new(&*centre, 256);
    let len = al.total();
    if w < EPS || !len.is_finite() || len < EPS {
        return Vec::new();
    }
    let floor = len.max(2.0 * w) / MAX_LINES as f64;
    let frame = Frame {
        surface,
        shade: &opts.shade,
        al,
        len,
        w,
        spacing: clamp_or(opts.spacing, floor, f64::MAX, floor),
        min_len: clamp_or(opts.min_len, 0.0, f64::MAX, 0.0),
        jitter: clamp_or(opts.jitter, 0.0, 1.0, 0.0),
        taper: clamp_or(opts.taper, 0.0, 0.5, 0.0),
    };
    let theta = clamp_or(angle, -1e6, 1e6, 90.0)
        .rem_euclid(180.0)
        .to_radians();
    let mut rng = seed.rng();
    let mut out = Vec::new();
    if theta.sin() >= std::f64::consts::FRAC_1_SQRT_2 {
        across(&frame, theta, bias, &mut rng, accuracy, &mut out);
    } else {
        along(&frame, theta, bias, &mut rng, accuracy, &mut out);
    }
    out
}

/// Length-modulated lines crossing the shape (`45° ≤ θ ≤ 135°`).
fn across<S: Surface + ?Sized>(
    f: &Frame<'_, S>,
    theta: f64,
    bias: f64,
    rng: &mut impl Draw,
    acc: f64,
    out: &mut Vec<BezPath>,
) {
    let (sin, cos) = theta.sin_cos();
    let cot = cos / sin;
    let ymax = f.w * V_MAX;
    // Lines are X = c + Y·cot θ; intercepts c are `spacing / sin θ` apart so the
    // perpendicular spacing is `spacing`, and range wide enough to cover the slant.
    let reach = ymax * cot.abs();
    let span = f.len + 2.0 * reach;
    let step = (f.spacing / sin).max(span / MAX_LINES as f64);
    let count = (span / step).floor() as usize;
    let samples = ((4.0 * f.w / f.spacing) as usize).clamp(24, 128);
    for i in 0..count {
        // Draw every random number up front so the stream never depends on geometry.
        let c = -reach + (i as f64 + 0.5) * step + rng.jitter(0.0, 0.35 * f.jitter * step);
        let slant = cot + rng.jitter(0.0, 0.06 * f.jitter);
        let r = f.threshold(i, rng.uniform(0.0, 1.0), 0.1, 0.9, bias);
        let line = move |t: f64| {
            let y = lerp(-ymax, ymax, t);
            (c + y * slant, y)
        };
        let keep = |t: f64| {
            let (x, y) = line(t);
            (0.0..=f.len).contains(&x) && f.shade(x, f.u_at(x), y / f.w) > r
        };
        for (a, b) in runs(keep, samples) {
            out.extend(f.piece(&line, a, b, acc));
        }
    }
}

/// Density-modulated lines running along the shape (`θ < 45°` or `θ > 135°`).
fn along<S: Surface + ?Sized>(
    f: &Frame<'_, S>,
    theta: f64,
    bias: f64,
    rng: &mut impl Draw,
    acc: f64,
    out: &mut Vec<BezPath>,
) {
    let (tan, cos) = (theta.tan(), theta.cos().abs());
    let ymax = f.w * V_MAX;
    let mid = 0.5 * f.len;
    // Lines are Y = c + (X − L/2)·tan θ with intercepts c covering the whole shape.
    let reach = ymax + tan.abs() * mid;
    let line_at = |c: f64| {
        move |t: f64| {
            let x = t * f.len;
            (x, c + (x - mid) * tan)
        }
    };
    // Mean shade along a candidate line (inside part only).
    let mean_shade = |c: f64| {
        let line = line_at(c);
        let (mut sum, mut n) = (0.0, 0usize);
        for k in 0..=24 {
            let (x, y) = line(k as f64 / 24.0);
            if y.abs() <= ymax {
                sum += f.shade(x, f.u_at(x), y / f.w);
                n += 1;
            }
        }
        if n == 0 { 0.0 } else { sum / n as f64 }
    };
    // Line density per unit of c: 1/spacing at full shade, 1/(6·spacing) at none.
    let perp = f.spacing / cos;
    let density = |c: f64| (0.2 + mean_shade(c)) / 1.2 / perp;
    const GRID: usize = 256;
    let cs: Vec<f64> = (0..=GRID)
        .map(|k| lerp(-reach, reach, k as f64 / GRID as f64))
        .collect();
    let dens: Vec<f64> = cs.iter().map(|&c| density(c)).collect();
    let mut cum = vec![0.0; GRID + 1];
    for k in 1..=GRID {
        cum[k] = cum[k - 1] + 0.5 * (dens[k] + dens[k - 1]) * (cs[k] - cs[k - 1]);
    }
    let total = cum[GRID];
    // If the cap bites, thin uniformly by scaling the cumulative count.
    let scale = (MAX_LINES as f64 / total).min(1.0);
    let count = (total * scale).floor() as usize;
    let samples = ((2.0 * f.len / f.spacing) as usize).clamp(32, 256);
    let mut k_seg = 1;
    for i in 0..count {
        let target = (i as f64 + 0.5 + rng.jitter(0.0, 0.3 * f.jitter)) / scale;
        let r = f.threshold(i, rng.uniform(0.0, 1.0), 0.03, 0.45, bias);
        // invert the cumulative count by linear interpolation
        while k_seg < GRID && cum[k_seg] < target {
            k_seg += 1;
        }
        let (c0, c1, m0, m1) = (cs[k_seg - 1], cs[k_seg], cum[k_seg - 1], cum[k_seg]);
        let c = if m1 - m0 > EPS {
            lerp(c0, c1, ((target - m0) / (m1 - m0)).clamp(0.0, 1.0))
        } else {
            c0
        };
        // Canvas gap to the neighbours where the shape is widest, as a fraction of W.
        let dv = cos / (scale * density(c)) / f.w;
        // Hierarchical drop-out: line i survives `2^level` times further into a pinch.
        let level = ((i + 1).trailing_zeros()).min(6);
        let gap = f.spacing * (0.7 + 0.4 * golden(i, 0.71)) / f64::from(1u32 << level);
        let line = line_at(c);
        let keep = |t: f64| {
            let (x, y) = line(t);
            if y.abs() > ymax {
                return false;
            }
            let u = f.u_at(x);
            f.shade(x, u, y / f.w) > r && half_width_at(f.surface, u) * dv >= gap
        };
        for (a, b) in runs(keep, samples) {
            out.extend(f.piece(&line, a, b, acc));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::ornament::test_shapes::{Huge, Leaf, Needle, inside_fraction, path_samples, petal};
    use kurbo::{ParamCurve, Point};

    fn assert_clean(paths: &[BezPath]) {
        for p in paths {
            assert!(path_is_finite(p));
            assert_eq!(degenerate_segments(p, 1e-6), 0);
        }
    }

    fn opts(angle: f64, shade: Shade) -> HatchOptions {
        HatchOptions {
            spacing: 4.0,
            angle,
            shade,
            ..Default::default()
        }
    }

    #[test]
    fn shade_fields_are_bounded() {
        let shades = [
            Shade::Uniform(3.0),
            Shade::Edge {
                side: 1.0,
                width: f64::NAN,
            },
            Shade::Ends { width: 0.0 },
            Shade::func(|u, _| u * 10.0 - 5.0),
            Shade::func(|_, _| f64::NAN),
        ];
        for s in &shades {
            for i in 0..=10 {
                for j in 0..=10 {
                    let x = s.at(i as f64 / 10.0, j as f64 / 5.0 - 1.0);
                    assert!((0.0..=1.0).contains(&x));
                }
            }
        }
        let e = Shade::Edge {
            side: -1.0,
            width: 1.0,
        };
        assert_eq!(e.at(0.5, -1.0), 1.0);
        assert_eq!(e.at(0.5, 1.0), 0.0);
        assert!(e.at(0.5, -0.5) > e.at(0.5, 0.5));
    }

    #[test]
    fn hatching_is_clean_and_inside() {
        let seed = SeedTree::new(5);
        for s in [&petal() as &dyn Surface, &Leaf] {
            let outline = s.outline(0.02);
            for angle in [90.0, 60.0, 0.0, 20.0, 160.0] {
                let paths = hatch(
                    s,
                    &opts(
                        angle,
                        Shade::Edge {
                            side: 1.0,
                            width: 0.9,
                        },
                    ),
                    &seed,
                    0.05,
                );
                assert!(!paths.is_empty(), "angle {angle}");
                assert_clean(&paths);
                let f = inside_fraction(&outline, &path_samples(&paths));
                assert!(f >= 0.99, "angle {angle}: inside fraction {f}");
            }
        }
    }

    /// Ink on one half of the shape vs the other: total line length per side.
    /// The petal is symmetric about `x = 0`, with `v = +1` on the right (`+x`).
    fn ink_by_side(paths: &[BezPath]) -> (f64, f64) {
        let (mut left, mut right) = (0.0, 0.0);
        for p in paths {
            for seg in p.segments() {
                let pts: Vec<Point> = (0..=8).map(|i| seg.eval(i as f64 / 8.0)).collect();
                for w in pts.windows(2) {
                    let d = (w[1] - w[0]).hypot();
                    if w[0].x + w[1].x > 0.0 {
                        right += d;
                    } else {
                        left += d;
                    }
                }
            }
        }
        (left, right)
    }

    #[test]
    fn shaded_side_gets_more_ink() {
        let s = petal();
        let seed = SeedTree::new(9);
        let shade = Shade::Edge {
            side: 1.0,
            width: 1.0,
        };
        for angle in [90.0, 0.0] {
            let paths = hatch(&s, &opts(angle, shade.clone()), &seed, 0.05);
            let (left, right) = ink_by_side(&paths);
            assert!(
                right > 2.0 * left,
                "angle {angle}: right {right} left {left}"
            );
        }
    }

    #[test]
    fn across_lines_start_at_the_dark_edge() {
        let s = petal();
        let o = HatchOptions {
            jitter: 0.0,
            taper: 0.0,
            ..opts(
                90.0,
                Shade::Edge {
                    side: -1.0,
                    width: 1.0,
                },
            )
        };
        let paths = hatch(&s, &o, &SeedTree::new(1), 0.05);
        assert!(paths.len() > 10);
        for p in &paths {
            let start = p.segments().next().unwrap().eval(0.0);
            // v = -1 is the left edge (x < 0); every line starts near it
            assert!(start.x < 0.0, "{start:?}");
        }
    }

    #[test]
    fn along_lines_thin_out_at_the_tip() {
        let leaf = Leaf;
        let o = opts(0.0, Shade::Uniform(1.0));
        let paths = hatch(&leaf, &o, &SeedTree::new(2), 0.05);
        // count lines crossing two sections: near the base and near the tip
        let count_at = |u: f64| {
            let y = leaf.point(u, 0.0).y;
            paths
                .iter()
                .filter(|p| {
                    let b = kurbo::Shape::bounding_box(*p);
                    b.y0 <= y && b.y1 >= y
                })
                .count() as f64
        };
        let (base, tip) = (count_at(0.3), count_at(0.9));
        let (wb, wt) = (leaf.half_width(0.3), leaf.half_width(0.9));
        // lines per unit width stays bounded: no pile-up at the tip
        assert!(
            tip / wt <= 1.5 * base / wb,
            "base {base}/{wb} tip {tip}/{wt}"
        );
        assert!(tip < base);
    }

    #[test]
    fn cross_hatch_adds_lines_in_the_shadow() {
        let s = petal();
        let o = opts(
            90.0,
            Shade::Edge {
                side: 1.0,
                width: 1.0,
            },
        );
        let seed = SeedTree::new(4);
        let one = hatch(&s, &o, &seed, 0.05);
        let two = cross_hatch(&s, &o, 30.0, &seed, 0.05);
        assert_eq!(&two[..one.len()], &one[..]);
        assert!(two.len() > one.len());
        let (l, r) = ink_by_side(&two[one.len()..]);
        assert!(r > 2.0 * l);
    }

    #[test]
    fn reproducible_and_seed_dependent() {
        let s = petal();
        let o = opts(75.0, Shade::Ends { width: 0.4 });
        let a = hatch(&s, &o, &SeedTree::new(3), 0.05);
        let b = hatch(&s, &o, &SeedTree::new(3), 0.05);
        let c = hatch(&s, &o, &SeedTree::new(4), 0.05);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn absurd_options_are_clamped() {
        let s = petal();
        let seed = SeedTree::new(1);
        assert!(hatch(&Needle, &HatchOptions::default(), &seed, 0.1).is_empty());
        for (spacing, angle) in [(0.0, 90.0), (-3.0, 0.0), (f64::NAN, f64::NAN), (1e9, 1e12)] {
            let o = HatchOptions {
                spacing,
                angle,
                min_len: f64::NAN,
                jitter: 50.0,
                taper: -1.0,
                shade: Shade::Uniform(f64::INFINITY),
            };
            let p = hatch(&s, &o, &seed, f64::NAN);
            assert!(p.len() <= 4 * MAX_LINES);
            assert_clean(&p);
        }
        for angle in [90.0, 0.0] {
            let o = opts(angle, Shade::Uniform(1.0));
            let p = hatch(&Huge, &o, &seed, 0.05);
            assert!(!p.is_empty());
            assert_clean(&p);
        }
        let huge_min = HatchOptions {
            min_len: 1e6,
            ..HatchOptions::default()
        };
        assert!(hatch(&s, &huge_min, &seed, 0.1).is_empty());
    }
}
