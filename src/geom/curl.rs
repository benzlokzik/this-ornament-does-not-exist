//! Smoothly deformed organic curves: noise-displaced curves ([`Curl`]), gently waving
//! connections ([`organic_curve`]) and the Art Nouveau whiplash line ([`whiplash`]).
//!
//! Everything is deterministic: randomness comes only from the `rng` / `seed` passed in, and
//! each function draws a fixed number of values regardless of its other inputs.

use super::curve::{Composite, Curve, CurveExt};
use super::fit::{fit_curve, stub, unit_or};
use super::{EPS, smoothstep};
use crate::rng::{Draw, Rng};
use kurbo::{CubicBez, Point, Vec2};
use noise::{NoiseFn, Perlin};
use std::f64::consts::{PI, TAU};

/// Half-width (in `t`) of the secant used for the displacement direction, see [`Curl`].
const SECANT: f64 = 0.004;

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// A base curve displaced along its normal by smooth gradient noise:
///
/// ```text
/// P(t) = B(t) + N̂(t) · A · w(u) · n(u),   w(u) = 16·u²·(1−u)²,   u = s(t)/L
/// ```
///
/// * `u` is the *arc-length* fraction of the base at `t`, so noise features are evenly
///   spread along the curve whatever its parameterisation. It also matters for smoothness:
///   piecewise bases such as [`Composite`] are only G1, their parametric speed `|B'|` jumps
///   at joints, and a displacement driven by `t` would turn every speed jump into a visible
///   kink (`dP/ds` would jump by `A·n'·Δ(1/|B'|)`). With `s(t)` exact, `dP/ds` stays
///   continuous. `s(t)` comes from a 256-cell table plus an adaptive Simpson integral of
///   `|B'|` inside the cell, accurate to ~1e-11 of the length even across speed jumps.
/// * `n(u)` is two octaves of improved Perlin noise (quintic fade, hence C²) sampled along a
///   slanted line through 2-D noise space, so the zero crossings are not locked to the lattice.
/// * `w(u)` vanishes with its derivative at both ends: the endpoints *and* their tangents
///   are those of the base curve, so a curl can be dropped into a G1 chain unchanged.
/// * `N̂(t)` is the left normal of the secant `B(t+δ) − B(t−δ)` (`δ = 0.004`) rather than
///   the exact normal: the exact normal of a piecewise-cubic base turns abruptly where its
///   curvature jumps, which would also put small kinks into the displaced curve; the secant
///   direction smooths those over a width `2δ`. For a C² base the difference is invisible.
///
/// `deriv` uses the trait's numerical default.
#[derive(Clone)]
pub struct Curl<C: Curve> {
    base: C,
    amplitude: f64,
    frequency: f64,
    noise: Perlin,
    /// Cumulative arc length of the base at `t = i / CELLS`.
    cum: Vec<f64>,
    /// Absolute tolerance of the in-cell integral.
    tol: f64,
}

/// Arc-length table resolution of [`Curl`].
const CELLS: usize = 256;
/// Recursion limit of the adaptive Simpson integral.
const SIMPSON_DEPTH: u32 = 48;

impl<C: Curve> Curl<C> {
    /// Displace `base` by up to about `amplitude` canvas units (signed noise, so both sides),
    /// with roughly `frequency` noise features along the curve (2–5 reads hand-drawn; much
    /// higher looks nervous). `seed` selects the noise field, e.g. from
    /// [`crate::rng::SeedTree::noise_seed`]. Non-finite values fall back to `0`.
    pub fn new(base: C, amplitude: f64, frequency: f64, seed: u32) -> Curl<C> {
        let mut curl = Curl {
            base,
            amplitude: finite_or(amplitude, 0.0),
            frequency: finite_or(frequency, 0.0).clamp(0.0, 1000.0),
            noise: Perlin::new(seed),
            cum: Vec::with_capacity(CELLS + 1),
            tol: 0.0,
        };
        let rough = finite_or(curl.base.length(), 0.0);
        curl.tol = 1e-11 * (rough + 1.0);
        let mut acc = 0.0;
        curl.cum.push(0.0);
        for i in 0..CELLS {
            acc += curl.integrate(i as f64 / CELLS as f64, (i + 1) as f64 / CELLS as f64);
            curl.cum.push(acc);
        }
        curl
    }

    /// The undeformed curve.
    pub fn base(&self) -> &C {
        &self.base
    }

    /// Arc length of the base curve.
    pub fn base_length(&self) -> f64 {
        self.cum[CELLS]
    }

    /// Signed normal displacement `A · w(u) · n(u)` at `t` (zero outside `[0,1]` and for a
    /// zero-length base).
    pub fn offset(&self, t: f64) -> f64 {
        let total = self.base_length();
        if !(0.0..=1.0).contains(&t) || total <= EPS || self.amplitude == 0.0 {
            return 0.0;
        }
        let u = (self.arclen_at(t) / total).clamp(0.0, 1.0);
        let w = 16.0 * u * u * (1.0 - u) * (1.0 - u);
        let x = u * self.frequency;
        // Positive, non-integer origin: avoids the lattice, where gradient noise is 0.
        let o1 = self.noise.get([3.71 + x, 1.29 + 0.37 * x]);
        let o2 = self.noise.get([11.13 + 2.07 * x, 7.61 + 0.53 * x]);
        self.amplitude * w * (o1 + 0.35 * o2)
    }

    /// Base arc length from `t = 0` to `t ∈ [0,1]`.
    fn arclen_at(&self, t: f64) -> f64 {
        let x = t.clamp(0.0, 1.0) * CELLS as f64;
        let i = (x.floor() as usize).min(CELLS - 1);
        self.cum[i] + self.integrate(i as f64 / CELLS as f64, t)
    }

    fn speed(&self, t: f64) -> f64 {
        let v = self.base.deriv(t).hypot();
        if v.is_finite() { v } else { 0.0 }
    }

    /// `∫ₐᵇ |B'(τ)| dτ` by adaptive Simpson (refines around speed jumps).
    fn integrate(&self, a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let m = 0.5 * (a + b);
        let (fa, fm, fb) = (self.speed(a), self.speed(m), self.speed(b));
        let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
        self.simpson(a, b, [fa, fm, fb], whole, SIMPSON_DEPTH)
    }

    fn simpson(&self, a: f64, b: f64, f: [f64; 3], whole: f64, depth: u32) -> f64 {
        let m = 0.5 * (a + b);
        let (flm, frm) = (self.speed(0.5 * (a + m)), self.speed(0.5 * (m + b)));
        let left = (m - a) / 6.0 * (f[0] + 4.0 * flm + f[1]);
        let right = (b - m) / 6.0 * (f[1] + 4.0 * frm + f[2]);
        let delta = left + right - whole;
        if depth == 0 || delta.abs() <= 15.0 * self.tol {
            return left + right + delta / 15.0;
        }
        self.simpson(a, m, [f[0], flm, f[1]], left, depth - 1)
            + self.simpson(m, b, [f[1], frm, f[2]], right, depth - 1)
    }
}

impl<C: Curve> Curve for Curl<C> {
    fn point(&self, t: f64) -> Point {
        let p = self.base.point(t);
        let off = self.offset(t);
        if off == 0.0 {
            return p;
        }
        let s = self.base.point(t + SECANT) - self.base.point(t - SECANT);
        let tg = unit_or(s, self.base.tangent(t));
        p + Vec2::new(-tg.y, tg.x) * off
    }
}

/// A gently waving connection from `start` to `end`.
///
/// The profile across the chord (length `L`) is `L·(0.25·bulge·sin πf + 0.2·waviness·sin 2πg)`:
/// `bulge` bows the whole line to one side (a C), `waviness` adds an S with one inflection.
/// Mixing the two gives a controllably asymmetric S (one lobe fuller than the other); the
/// sign of each picks its side. Both are clamped to `[-2, 2]`; `|·| ≤ 1` is the useful
/// range. The rng skews the inflection (`g = f^γ`), varies both amplitudes by ±15% and
/// jitters the 9 interior knots slightly before they are joined with
/// [`Composite::through_points`] and refitted with [`fit_curve`]. Coincident endpoints give a `1e-6` stub.
pub fn organic_curve<R: Rng>(
    rng: &mut R,
    start: Point,
    end: Point,
    waviness: f64,
    bulge: f64,
) -> Composite {
    const N: usize = 10;
    // Draw everything up front so the stream use never depends on the inputs.
    let gamma = rng.uniform(-0.3, 0.3).exp();
    let wave_amp = rng.uniform(0.85, 1.15);
    let bulge_amp = rng.uniform(0.85, 1.15);
    let jit: Vec<(f64, f64)> = (1..N)
        .map(|_| (rng.jitter(0.0, 0.08 / N as f64), rng.jitter(0.0, 1.0)))
        .collect();

    let d = end - start;
    let len = d.hypot();
    if !(start.is_finite() && end.is_finite()) || len < EPS || !len.is_finite() {
        return stub(start, Vec2::new(1.0, 0.0));
    }
    let u = d / len;
    let n = Vec2::new(-u.y, u.x);
    let w = finite_or(waviness, 0.0).clamp(-2.0, 2.0);
    let b = finite_or(bulge, 0.0).clamp(-2.0, 2.0);
    let across_jitter = 0.004 * len * (w.abs() + b.abs() + 0.1);
    let pts: Vec<Point> = (0..=N)
        .map(|i| {
            let mut f = i as f64 / N as f64;
            let mut extra = 0.0;
            if i > 0 && i < N {
                f += jit[i - 1].0;
                extra = jit[i - 1].1 * across_jitter;
            }
            let g = f.clamp(0.0, 1.0).powf(gamma);
            let off = len
                * (0.25 * b * bulge_amp * (PI * f).sin() + 0.2 * w * wave_amp * (TAU * g).sin());
            start + d * f + n * (off + extra)
        })
        .collect();
    let Some(raw) = Composite::through_points(&pts, 0.5) else {
        return stub(start, u);
    };
    let acc = (len * 5e-4).max(1e-4);
    Composite::from_bezpath(&fit_curve(&raw, acc)).unwrap_or(raw)
}

/// The Art Nouveau whiplash ("coup de fouet"): a long sweeping S whose end tightens into a
/// hook. Starts at `start` heading along `direction`, has arc length `length` and ends in a
/// curl of `hook` turns (`0` = plain S; `0.5–1.5` is classic; clamped to `[-3, 3]`). A positive
/// `hook` curls towards the left normal of `direction`, a negative one to the right; the
/// opening counter-bow always goes the other way.
///
/// The curve is defined intrinsically by its curvature along normalised arc length `u`:
///
/// ```text
/// κ(u)·L = β·(u − uᵢ) + H · s((u − u_h)/(1 − u_h)) / (1 − u + ε)
/// ```
///
/// * the linear term is the S: curvature changes sign at `uᵢ`; `β = 2·bow/uᵢ²` makes the
///   opening bow turn by `bow` radians;
/// * the second term is the hook: `1/(1 − u + ε)` is the curvature law of a logarithmic
///   spiral approaching its eye (`κ ∝ 1/remaining length`), faded in by a smoothstep `s`
///   from `u_h`; `H` is solved so the hook adds exactly `|hook|·2π` of turning, and `ε` sets
///   how tight the eye gets.
///
/// Both terms are non-decreasing, so `κ` is monotone along the whole curve: by the
/// Tait–Kneser theorem its osculating circles are nested, which is what makes the line read
/// as a single, effortless gesture and keeps the hook free of self-intersection. Turning
/// angle and position are integrated numerically (trapezoid, 2048 steps), the samples joined
/// by cubic Hermite segments with exact tangents, then refitted with [`fit_curve`]. The rng varies `uᵢ`, `bow`,
/// `ε` and `u_h` within tasteful ranges. Invalid `length` gives a `1e-6` stub.
pub fn whiplash<R: Rng>(
    rng: &mut R,
    start: Point,
    direction: Vec2,
    length: f64,
    hook: f64,
) -> Composite {
    const STEPS: usize = 2048;
    const STRIDE: usize = 8;
    let u_i = rng.uniform(0.4, 0.47);
    let bow = rng.uniform(0.6, 0.95);
    let eps = rng.uniform(0.03, 0.06);
    let u_h = rng.uniform(0.45, 0.6);

    let dir = unit_or(direction, Vec2::new(1.0, 0.0));
    if !start.is_finite() || !length.is_finite() || length <= EPS {
        return stub(start, dir);
    }
    let len = length;
    let h = finite_or(hook, 0.0).clamp(-3.0, 3.0);
    let sigma = if h < 0.0 { -1.0 } else { 1.0 };
    let du = 1.0 / STEPS as f64;
    let beta = 2.0 * bow / (u_i * u_i);
    let hook_shape = |u: f64| smoothstep((u - u_h) / (1.0 - u_h)) / (1.0 - u + eps);
    // Trapezoid integral of the hook shape, so H gives exactly |h|·2π with the same rule
    // used for the turning angle below.
    let mut integral = 0.0;
    for i in 0..STEPS {
        let (a, b) = (i as f64 * du, (i + 1) as f64 * du);
        integral += 0.5 * (hook_shape(a) + hook_shape(b)) * du;
    }
    let big_h = if integral > EPS {
        h.abs() * TAU / integral
    } else {
        0.0
    };
    let kappa = |u: f64| beta * (u - u_i) + big_h * hook_shape(u);

    // Integrate θ(u) and P(u) in a local frame (x along `dir`, y towards the curl side).
    let n = Vec2::new(-dir.y, dir.x) * sigma;
    let to_world = |v: Vec2| dir * v.x + n * v.y;
    let mut theta = 0.0;
    let mut pos = Vec2::ZERO;
    let mut knots: Vec<(Point, Vec2)> = Vec::with_capacity(STEPS / STRIDE + 1);
    knots.push((start, dir));
    for i in 0..STEPS {
        let (a, b) = (i as f64 * du, (i + 1) as f64 * du);
        let th_next = theta + 0.5 * (kappa(a) + kappa(b)) * du;
        let (s0, c0) = f64::sin_cos(theta);
        let (s1, c1) = f64::sin_cos(th_next);
        pos += Vec2::new(c0 + c1, s0 + s1) * (0.5 * len * du);
        theta = th_next;
        if (i + 1) % STRIDE == 0 {
            knots.push((start + to_world(pos), to_world(Vec2::new(c1, s1))));
        }
    }
    // Cubic Hermite with unit tangents scaled by the arc length per knot interval.
    let ds = len * du * STRIDE as f64;
    let segs: Vec<CubicBez> = knots
        .windows(2)
        .map(|w| {
            let ((p0, t0), (p1, t1)) = (w[0], w[1]);
            CubicBez::new(p0, p0 + t0 * (ds / 3.0), p1 - t1 * (ds / 3.0), p1)
        })
        .collect();
    let raw = Composite::new_by_length(segs);
    let acc = (len * 2e-4).max(1e-4);
    Composite::from_bezpath(&fit_curve(&raw, acc)).unwrap_or(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::fit::max_kink;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::rng::SeedTree;
    use kurbo::{BezPath, PathEl};

    fn well_formed(c: &Composite) {
        let p = c.to_bezpath_exact();
        assert!(path_is_finite(&p));
        assert_eq!(degenerate_segments(&p, 1e-9), 0);
    }

    /// Max turning between consecutive dense tangent samples (radians).
    fn max_tangent_jump<C: Curve>(c: &C, n: usize) -> f64 {
        (0..n)
            .map(|i| {
                let a = c.tangent(i as f64 / n as f64);
                let b = c.tangent((i + 1) as f64 / n as f64);
                a.cross(b).atan2(a.dot(b)).abs()
            })
            .fold(0.0, f64::max)
    }

    fn polyline(c: &Composite) -> Vec<Point> {
        let mut pts = Vec::new();
        let path: BezPath = c.to_bezpath_exact();
        kurbo::flatten(path.iter(), 0.05, |el| match el {
            PathEl::MoveTo(p) | PathEl::LineTo(p) => pts.push(p),
            _ => {}
        });
        pts
    }

    fn self_intersects(pts: &[Point]) -> bool {
        let n = pts.len();
        for i in 0..n.saturating_sub(1) {
            for j in i + 2..n - 1 {
                let (a, b, c, d) = (pts[i], pts[i + 1], pts[j], pts[j + 1]);
                let (r, s) = (b - a, d - c);
                let den = r.cross(s);
                if den.abs() < 1e-12 {
                    continue;
                }
                let t = (c - a).cross(s) / den;
                let u = (c - a).cross(r) / den;
                if t > 1e-9 && t < 1.0 - 1e-9 && u > 1e-9 && u < 1.0 - 1e-9 {
                    return true;
                }
            }
        }
        false
    }

    fn wavy_base() -> Composite {
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(60.0, 40.0),
            Point::new(140.0, -10.0),
            Point::new(220.0, 30.0),
            Point::new(300.0, 0.0),
        ];
        Composite::through_points(&pts, 0.5).unwrap()
    }

    #[test]
    fn curl_keeps_endpoints_and_is_smooth() {
        let base = wavy_base();
        let c = Curl::new(base.clone(), 12.0, 4.0, 7);
        assert_eq!(c.point(0.0), base.point(0.0));
        assert_eq!(c.point(1.0), base.point(1.0));
        // end tangents preserved (zero envelope slope)
        assert!(c.tangent(0.0).dot(base.tangent(0.0)) > 0.999);
        assert!(c.tangent(1.0).dot(base.tangent(1.0)) > 0.999);
        // actually displaced, but bounded
        let max_off = (0..=100)
            .map(|i| c.offset(i as f64 / 100.0).abs())
            .fold(0.0, f64::max);
        assert!(max_off > 0.5 && max_off < 12.0 * 1.35 + 1e-9, "{max_off}");
        // C1: no tangent jumps when sampled densely (base joints included)
        assert!(max_tangent_jump(&c, 4000) < 0.02);
        let p = c.to_bezpath(0.02);
        assert!(path_is_finite(&p));
        assert_eq!(degenerate_segments(&p, 1e-9), 0);
        let fitted = Composite::from_bezpath(&p).unwrap();
        assert!(max_kink(&fitted) < 0.02);
    }

    #[test]
    fn curl_is_reproducible_and_seeded() {
        let a = Curl::new(wavy_base(), 10.0, 3.0, 1).to_bezpath(0.05);
        let b = Curl::new(wavy_base(), 10.0, 3.0, 1).to_bezpath(0.05);
        let c = Curl::new(wavy_base(), 10.0, 3.0, 2).to_bezpath(0.05);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn curl_edge_cases() {
        let base = wavy_base();
        let flat = Curl::new(base.clone(), 0.0, 3.0, 1);
        assert_eq!(flat.point(0.37), base.point(0.37));
        let nan = Curl::new(base.clone(), f64::NAN, f64::INFINITY, 1);
        assert!(nan.point(0.5).is_finite());
        let dot = Curl::new(
            CubicBez::new((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)),
            5.0,
            3.0,
            9,
        );
        for i in 0..=10 {
            assert!(dot.point(i as f64 / 10.0).is_finite());
        }
        assert!(path_is_finite(&dot.to_bezpath(0.1)));
        let wild = Curl::new(base, 200.0, 50.0, 3);
        assert!(path_is_finite(&wild.to_bezpath(0.1)));
        assert_eq!(wild.offset(-0.5), 0.0);
    }

    #[test]
    fn organic_curve_connects_smoothly() {
        let tree = SeedTree::new(11);
        let (s, e) = (Point::new(10.0, 20.0), Point::new(310.0, 80.0));
        for i in 0..12 {
            let mut rng = tree.child("organic").index(i).rng();
            let w = [0.0, 0.4, -0.7, 1.0][i as usize % 4];
            let b = [0.3, -0.2, 0.0, 0.6][i as usize % 4];
            let c = organic_curve(&mut rng, s, e, w, b);
            well_formed(&c);
            assert!((c.start() - s).hypot() < 1e-9);
            assert!((c.end() - e).hypot() < 1e-9);
            assert!(max_kink(&c) < 0.01, "kink {}", max_kink(&c));
            assert!(!self_intersects(&polyline(&c)));
            let mut again = tree.child("organic").index(i).rng();
            assert_eq!(c, organic_curve(&mut again, s, e, w, b));
        }
        let mut rng = tree.rng();
        well_formed(&organic_curve(&mut rng, s, s, 1.0, 1.0));
        well_formed(&organic_curve(
            &mut rng,
            s,
            Point::new(f64::NAN, 0.0),
            1.0,
            1.0,
        ));
        well_formed(&organic_curve(&mut rng, s, e, f64::NAN, 1e9));
    }

    #[test]
    fn whiplash_is_smooth_simple_and_hooked() {
        let tree = SeedTree::new(5);
        for i in 0..10 {
            for hook in [0.0, 0.6, 1.2, -1.0, 2.0] {
                let mut rng = tree.child("whip").index(i).rng();
                let start = Point::new(50.0, 400.0);
                let dir = Vec2::new(1.0, -0.4);
                let c = whiplash(&mut rng, start, dir, 500.0, hook);
                well_formed(&c);
                assert!((c.start() - start).hypot() < 1e-6);
                assert!(c.tangent(0.0).dot(dir / dir.hypot()) > 0.9999);
                assert!(max_kink(&c) < 0.01, "kink {}", max_kink(&c));
                assert!((c.length() / 500.0 - 1.0).abs() < 0.01);
                assert!(!self_intersects(&polyline(&c)), "seed {i} hook {hook}");
                // hook side follows its sign; the end turns towards it
                let k_end = c.curvature(0.999);
                if hook > 0.0 {
                    assert!(k_end > 0.0);
                } else if hook < 0.0 {
                    assert!(k_end < 0.0);
                }
                let mut again = tree.child("whip").index(i).rng();
                assert_eq!(c, whiplash(&mut again, start, dir, 500.0, hook));
            }
        }
        // the hook really tightens: end curvature far above the mid-body curvature
        let mut rng = tree.rng();
        let c = whiplash(&mut rng, Point::ZERO, Vec2::new(1.0, 0.0), 400.0, 1.0);
        assert!(c.curvature(0.995).abs() > 8.0 * c.curvature(0.5).abs());
    }

    #[test]
    fn whiplash_degenerate_inputs() {
        let mut rng = SeedTree::new(1).rng();
        for (len, hook) in [
            (0.0, 1.0),
            (-5.0, 1.0),
            (f64::NAN, 1.0),
            (100.0, f64::NAN),
            (1e-6, 3.0),
        ] {
            well_formed(&whiplash(&mut rng, Point::ZERO, Vec2::ZERO, len, hook));
        }
        well_formed(&whiplash(
            &mut rng,
            Point::new(f64::INFINITY, 0.0),
            Vec2::new(0.0, 1.0),
            10.0,
            1.0,
        ));
        well_formed(&whiplash(
            &mut rng,
            Point::ZERO,
            Vec2::new(0.0, 1.0),
            100.0,
            99.0,
        ));
    }
}
