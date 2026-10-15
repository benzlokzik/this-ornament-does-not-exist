//! Spirals with analytic derivatives and curvature, and tangent-continuous spiral tails.
//!
//! Both spirals are written in polar form around a centre `c`:
//!
//! ```text
//! P(θ) = c + r(θ)·e(ψ),   ψ = rotation + σ·θ,   e(ψ) = (cos ψ, sin ψ)
//! ```
//!
//! with `σ = +1` (counter-clockwise in a y-up frame) or `σ = −1` (clockwise). The curve
//! parameter `t ∈ [0,1]` maps linearly onto `θ ∈ [θ₀, θ₁]`, so with `Δ = θ₁ − θ₀`
//!
//! ```text
//! dP/dt = Δ·(r'(θ)·e(ψ) + σ·r(θ)·e⊥(ψ)),   e⊥(ψ) = (−sin ψ, cos ψ)
//! ```
//!
//! and the signed curvature follows from the polar formula
//! `κ = σ·sgn(Δ)·(r² + 2r'² − r·r'') / (r² + r'²)^{3/2}`.
//!
//! Note on orientation: SVG's y axis points down, so "counter-clockwise" here appears
//! clockwise on screen. Only the handedness relative to a given tangent matters for
//! [`LogSpiral::from_start`] and [`spiral_tail`], and that is expressed through the curve's
//! own normal (see their docs).

use super::EPS;
use super::curve::{Composite, Curve, CurveExt};
use super::fit::{fit_curve, join_g1, stub, unit_or};
use kurbo::{Point, Vec2};
use std::f64::consts::TAU;

/// Clamp on `b·θ` so `exp` never overflows.
const MAX_EXP: f64 = 300.0;
/// Upper bound on turns accepted by the constructors.
const MAX_TURNS: f64 = 50.0;
/// Upper bound on the log-spiral tightness `k` (per radian).
const MAX_TIGHTNESS: f64 = 5.0;

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

fn positive_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() && x > 0.0 {
        x
    } else {
        fallback
    }
}

fn sanitize_point(p: Point) -> Point {
    if p.is_finite() { p } else { Point::ZERO }
}

fn rot(v: Vec2, a: f64) -> Vec2 {
    let (s, c) = a.sin_cos();
    Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

/// Shared polar frame: centre, orientation and the linear map `t → θ`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Polar {
    center: Point,
    theta_start: f64,
    theta_end: f64,
    clockwise: bool,
    rotation: f64,
}

impl Polar {
    fn new(
        center: Point,
        theta_start: f64,
        theta_end: f64,
        clockwise: bool,
        rotation: f64,
    ) -> Self {
        Polar {
            center: sanitize_point(center),
            theta_start: finite_or(theta_start, 0.0),
            theta_end: finite_or(theta_end, 0.0),
            clockwise,
            rotation: finite_or(rotation, 0.0),
        }
    }
    fn sigma(&self) -> f64 {
        if self.clockwise { -1.0 } else { 1.0 }
    }
    fn delta(&self) -> f64 {
        self.theta_end - self.theta_start
    }
    /// `sgn(Δ)`, with a zero range treated as forward.
    fn dir(&self) -> f64 {
        if self.delta() < 0.0 { -1.0 } else { 1.0 }
    }
    fn theta(&self, t: f64) -> f64 {
        self.theta_start + self.delta() * t
    }
    fn e(&self, theta: f64) -> Vec2 {
        let (s, c) = (self.rotation + self.sigma() * theta).sin_cos();
        Vec2::new(c, s)
    }
    fn point(&self, r: f64, theta: f64) -> Point {
        self.center + self.e(theta) * r
    }
    /// `dP/dθ` for radius `r` and radial speed `dr = r'(θ)`.
    fn dp_dtheta(&self, r: f64, dr: f64, theta: f64) -> Vec2 {
        let e = self.e(theta);
        let e_perp = Vec2::new(-e.y, e.x);
        e * dr + e_perp * (self.sigma() * r)
    }
    fn turns(&self) -> f64 {
        self.delta().abs() / TAU
    }
}

/// Logarithmic (equiangular) spiral `r(θ) = a·e^{bθ}`.
///
/// The tangent meets the radius at the constant angle `atan(1/b)`, so every turn is a scaled
/// copy of the previous one: the natural growth law of shells, fern croziers and the curled
/// ends of Art Nouveau ribbons. `b < 0` winds inwards as `θ` grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogSpiral {
    polar: Polar,
    a: f64,
    b: f64,
}

impl LogSpiral {
    /// General constructor. `a` (radius at `θ = 0`) must be positive (falls back to `1`),
    /// non-finite angles fall back to `0`. `t ∈ [0,1]` maps onto `θ ∈ [theta_start,
    /// theta_end]`; the point at angle `θ` lies at polar angle `rotation ± θ` (`−` when
    /// `clockwise`) around `center`.
    pub fn new(
        center: Point,
        a: f64,
        b: f64,
        theta_start: f64,
        theta_end: f64,
        clockwise: bool,
        rotation: f64,
    ) -> LogSpiral {
        LogSpiral {
            polar: Polar::new(center, theta_start, theta_end, clockwise, rotation),
            a: positive_or(a, 1.0),
            b: finite_or(b, 0.0),
        }
    }

    /// An inward-winding spiral that starts at `start` heading along `tangent`: the curled
    /// tail of a ribbon or a fern crozier.
    ///
    /// * `radius` — distance from `start` to the spiral's centre (the size of the curl).
    /// * `turns` — number of revolutions travelled (clamped to `[0, 50]`).
    /// * `tightness` — `k ≥ 0` in `r(φ) = radius·e^{−kφ}` where `φ` is the angle travelled;
    ///   the radius shrinks by `e^{−2πk}` per turn. Use [`LogSpiral::tightness_for`] to pick
    ///   it from a desired end/start radius ratio. `k = 0` gives a circle.
    /// * `clockwise` — `false` curls towards the left normal of `tangent` (the side
    ///   [`Curve::normal`] points to), `true` towards the right.
    ///
    /// Guarantees `point(0) == start`, `tangent(0) ∥ tangent` and a strictly decreasing
    /// radius for `k > 0`.
    ///
    /// Derivation: with `b = −k` and travelled angle `θ ≥ 0`, `dP/dθ = r·(−k·e + σ·e⊥)`, so
    /// the unit tangent at the start is `T = (−k·e + σ·e⊥)/√(1+k²)`, which is the radial unit
    /// vector `e` rotated by `σβ` with `β = atan2(1, −k)` (`cos β = −k/√(1+k²)`,
    /// `sin β = 1/√(1+k²)`). Inverting, `e = rot(−σβ)·T`, the centre is
    /// `c = start − radius·e` and the polar rotation is `atan2(e)`. For `k → 0`, `β → 90°`
    /// and the centre sits on the normal, as for an osculating circle.
    pub fn from_start(
        start: Point,
        tangent: Vec2,
        radius: f64,
        turns: f64,
        tightness: f64,
        clockwise: bool,
    ) -> LogSpiral {
        let start = sanitize_point(start);
        let k = finite_or(tightness, 0.0).clamp(0.0, MAX_TIGHTNESS);
        let turns = finite_or(turns, 0.0).clamp(0.0, MAX_TURNS);
        let r0 = positive_or(radius, 1.0);
        let t = unit_or(tangent, Vec2::new(1.0, 0.0));
        let sigma = if clockwise { -1.0 } else { 1.0 };
        let beta = 1f64.atan2(-k);
        let e = rot(t, -sigma * beta);
        let center = start - e * r0;
        LogSpiral::new(center, r0, -k, 0.0, TAU * turns, clockwise, e.y.atan2(e.x))
    }

    /// An outward-growing spiral starting `inner_radius` from `center` at polar angle
    /// `start_angle` and growing by `e^{2πk}` per turn (`k = tightness`, clamped to `[0, 5]`).
    pub fn from_center(
        center: Point,
        inner_radius: f64,
        start_angle: f64,
        turns: f64,
        tightness: f64,
        clockwise: bool,
    ) -> LogSpiral {
        let k = finite_or(tightness, 0.0).clamp(0.0, MAX_TIGHTNESS);
        let turns = finite_or(turns, 0.0).clamp(0.0, MAX_TURNS);
        LogSpiral::new(
            center,
            positive_or(inner_radius, 1.0),
            k,
            0.0,
            TAU * turns,
            clockwise,
            start_angle,
        )
    }

    /// Tightness `k` such that after `turns` revolutions the radius is `end_ratio` times the
    /// start radius: `k = −ln(end_ratio) / (2π·turns)`. This keeps the visual weight of the
    /// curl constant whatever the turn count (e.g. `end_ratio ≈ 0.12` reads well for 1.5–3
    /// turns: few turns get a fast-closing eye, many turns a gentle, even coil).
    pub fn tightness_for(turns: f64, end_ratio: f64) -> f64 {
        let turns = finite_or(turns, 1.0).max(1e-3);
        let ratio = finite_or(end_ratio, 0.5).clamp(1e-9, 1.0);
        (-ratio.ln() / (TAU * turns)).clamp(0.0, MAX_TIGHTNESS)
    }

    /// The spiral's pole.
    pub fn center(&self) -> Point {
        self.polar.center
    }

    /// Distance from the centre at parameter `t`.
    pub fn radius_at(&self, t: f64) -> f64 {
        self.radius(self.polar.theta(t))
    }

    /// Number of revolutions between `t = 0` and `t = 1`.
    pub fn total_turns(&self) -> f64 {
        self.polar.turns()
    }

    /// Growth rate `b` of `r = a·e^{bθ}` (negative winds inwards).
    pub fn growth(&self) -> f64 {
        self.b
    }

    fn radius(&self, theta: f64) -> f64 {
        self.a * (self.b * theta).clamp(-MAX_EXP, MAX_EXP).exp()
    }
}

impl Curve for LogSpiral {
    fn point(&self, t: f64) -> Point {
        let th = self.polar.theta(t);
        self.polar.point(self.radius(th), th)
    }

    fn deriv(&self, t: f64) -> Vec2 {
        let th = self.polar.theta(t);
        let r = self.radius(th);
        self.polar.dp_dtheta(r, self.b * r, th) * self.polar.delta()
    }

    /// Exact: `(b·e + σ·e⊥)/√(1+b²)` (times `sgn Δ`), independent of `r`, so it stays well
    /// defined for zero angle ranges and microscopic radii.
    fn tangent(&self, t: f64) -> Vec2 {
        let th = self.polar.theta(t);
        let v = self.polar.dp_dtheta(1.0, self.b, th) * self.polar.dir();
        v / v.hypot()
    }

    /// `κ = σ·sgn(Δ) / (r·√(1+b²))`.
    fn curvature(&self, t: f64) -> f64 {
        let r = self.radius(self.polar.theta(t)).max(EPS);
        self.polar.sigma() * self.polar.dir() / (r * (1.0 + self.b * self.b).sqrt())
    }
}

/// Archimedean spiral `r(θ) = a + b·θ`: evenly spaced coils (pitch `2πb`), the look of a
/// watch spring or a tightly rolled scroll.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchimedeanSpiral {
    polar: Polar,
    a: f64,
    b: f64,
}

impl ArchimedeanSpiral {
    /// General constructor (same angle conventions as [`LogSpiral::new`]). Non-finite inputs
    /// fall back to `0` (`center` to the origin).
    pub fn new(
        center: Point,
        a: f64,
        b: f64,
        theta_start: f64,
        theta_end: f64,
        clockwise: bool,
        rotation: f64,
    ) -> ArchimedeanSpiral {
        ArchimedeanSpiral {
            polar: Polar::new(center, theta_start, theta_end, clockwise, rotation),
            a: finite_or(a, 0.0),
            b: finite_or(b, 0.0),
        }
    }

    /// Spiral starting at `center + inner_radius·e(start_angle)` and moving outwards by
    /// `pitch` per turn for `turns` revolutions (clamped to `[0, 50]`). With
    /// `inner_radius = 0` it starts exactly at the centre, heading radially outwards.
    pub fn from_center(
        center: Point,
        inner_radius: f64,
        start_angle: f64,
        turns: f64,
        pitch: f64,
        clockwise: bool,
    ) -> ArchimedeanSpiral {
        let turns = finite_or(turns, 0.0).clamp(0.0, MAX_TURNS);
        ArchimedeanSpiral::new(
            center,
            finite_or(inner_radius, 0.0).max(0.0),
            finite_or(pitch, 0.0) / TAU,
            0.0,
            TAU * turns,
            clockwise,
            start_angle,
        )
    }

    /// The spiral's pole.
    pub fn center(&self) -> Point {
        self.polar.center
    }

    /// Distance from the centre at parameter `t` (signed `a + bθ`; negative values lie on
    /// the opposite side of the pole).
    pub fn radius_at(&self, t: f64) -> f64 {
        self.a + self.b * self.polar.theta(t)
    }

    /// Number of revolutions between `t = 0` and `t = 1`.
    pub fn total_turns(&self) -> f64 {
        self.polar.turns()
    }
}

impl Curve for ArchimedeanSpiral {
    fn point(&self, t: f64) -> Point {
        let th = self.polar.theta(t);
        self.polar.point(self.a + self.b * th, th)
    }

    fn deriv(&self, t: f64) -> Vec2 {
        let th = self.polar.theta(t);
        self.polar.dp_dtheta(self.a + self.b * th, self.b, th) * self.polar.delta()
    }

    fn tangent(&self, t: f64) -> Vec2 {
        let th = self.polar.theta(t);
        let v = self.polar.dp_dtheta(self.a + self.b * th, self.b, th) * self.polar.dir();
        // r = b = 0 is a single point: fall back to the circular direction.
        unit_or(v, self.polar.dp_dtheta(1.0, 0.0, th) * self.polar.dir())
    }

    /// `κ = σ·sgn(Δ)·(r² + 2b²) / (r² + b²)^{3/2}` (polar formula with `r'' = 0`).
    fn curvature(&self, t: f64) -> f64 {
        let r = self.radius_at(t);
        let (r2, b2) = (r * r, self.b * self.b);
        let den = (r2 + b2).powf(1.5);
        if den < EPS {
            return 0.0;
        }
        let k = self.polar.sigma() * self.polar.dir() * (r2 + 2.0 * b2) / den;
        if k.is_finite() { k } else { 0.0 }
    }
}

/// `curve` followed by an inward [`LogSpiral`] that continues it tangent-continuously, as a
/// single piecewise-cubic [`Composite`] (both parts fitted with
/// [`fit_curve`](super::fit::fit_curve) to `accuracy`).
///
/// The spiral's size is chosen so the join is also curvature-continuous (G2) where possible:
/// if the curve already bends towards the curl side, the spiral's initial radius is
/// `r₀ = 1/(|κ|·√(1+k²))` (its starting curvature then equals the curve's end curvature),
/// clamped to `[0.04, 0.25]` of the curve length; otherwise (an inflection at the join)
/// `r₀ = 0.12·length`. `clockwise` has the meaning of [`LogSpiral::from_start`]: `false`
/// curls towards the curve's left normal. The join is made exactly G1 by [`join_g1`]; check
/// it with [`super::fit::max_kink`].
pub fn spiral_tail<C: Curve + ?Sized>(
    curve: &C,
    turns: f64,
    tightness: f64,
    clockwise: bool,
    accuracy: f64,
) -> Composite {
    let acc = positive_or(accuracy, 0.1);
    let k = finite_or(tightness, 0.0).clamp(0.0, MAX_TIGHTNESS);
    let end = sanitize_point(curve.point(1.0));
    let tangent = curve.tangent(1.0);
    let kappa = finite_or(curve.curvature(1.0), 0.0);
    let len = finite_or(curve.length(), 0.0);
    let bends_same_way = if clockwise { kappa < 0.0 } else { kappa > 0.0 };
    let r0 = if bends_same_way && kappa.abs() > EPS {
        (1.0 / (kappa.abs() * (1.0 + k * k).sqrt())).clamp(0.04 * len, 0.25 * len)
    } else {
        0.12 * len
    };
    let spiral = LogSpiral::from_start(end, tangent, r0.max(1e-3), turns, k, clockwise);
    let head = fit_curve(curve, acc);
    let tail = fit_curve(&spiral, acc);
    join_g1(&[head, tail]).unwrap_or_else(|| stub(end, tangent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::fit::max_kink;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use kurbo::CubicBez;

    fn check_analytic<C: Curve>(c: &C) {
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let h = 1e-6;
            let num = (c.point(t + h) - c.point(t - h)) / (2.0 * h);
            let d = c.deriv(t);
            assert!(
                (num - d).hypot() <= 1e-4 * d.hypot().max(1.0),
                "t={t} {num:?} vs {d:?}"
            );
            let tg = c.tangent(t);
            assert!((tg.hypot() - 1.0).abs() < 1e-12);
            assert!(tg.dot(d) > 0.0);
            // numerical curvature from analytic derivatives
            let dd = (c.deriv(t + h) - c.deriv(t - h)) / (2.0 * h);
            let kn = d.cross(dd) / d.hypot().powi(3);
            let k = c.curvature(t);
            assert!(
                (kn - k).abs() <= 1e-4 * k.abs().max(1e-3),
                "t={t} {kn} vs {k}"
            );
        }
    }

    #[test]
    fn analytic_derivatives_match_numeric() {
        for cw in [false, true] {
            check_analytic(&LogSpiral::new(
                Point::new(3.0, 4.0),
                50.0,
                -0.15,
                0.0,
                9.0,
                cw,
                0.7,
            ));
            check_analytic(&LogSpiral::new(Point::ZERO, 5.0, 0.2, 2.0, -4.0, cw, -1.0));
            check_analytic(&LogSpiral::from_center(
                Point::ZERO,
                2.0,
                0.3,
                2.5,
                0.12,
                cw,
            ));
            check_analytic(&ArchimedeanSpiral::from_center(
                Point::ZERO,
                3.0,
                1.0,
                3.0,
                8.0,
                cw,
            ));
            check_analytic(&ArchimedeanSpiral::new(
                Point::ZERO,
                10.0,
                -1.0,
                0.0,
                6.0,
                cw,
                0.0,
            ));
        }
    }

    #[test]
    fn from_start_continues_tangent_and_winds_inward() {
        let start = Point::new(100.0, 50.0);
        for (tg, cw, k) in [
            (Vec2::new(1.0, 0.0), false, 0.1),
            (Vec2::new(-0.3, 2.0), true, 0.25),
            (Vec2::new(0.0, -1.0), false, 0.0),
            (Vec2::new(1.0, 1.0), true, 1.5),
        ] {
            let s = LogSpiral::from_start(start, tg, 40.0, 2.5, k, cw);
            assert!((s.point(0.0) - start).hypot() < 1e-9);
            let u = tg / tg.hypot();
            assert!(s.tangent(0.0).dot(u) > 1.0 - 1e-12);
            assert!((s.radius_at(0.0) - 40.0).abs() < 1e-9);
            assert!((s.total_turns() - 2.5).abs() < 1e-12);
            // left normal side for counter-clockwise, right side for clockwise
            let n = Vec2::new(-u.y, u.x);
            let side = (s.center() - start).dot(n);
            assert!(if cw { side < 0.0 } else { side > 0.0 });
            assert_eq!(s.curvature(0.5) < 0.0, cw);
            if k > 0.0 {
                let r: Vec<f64> = (0..=50).map(|i| s.radius_at(i as f64 / 50.0)).collect();
                assert!(r.windows(2).all(|w| w[1] < w[0]));
            }
            let p = s.to_bezpath(0.01);
            assert!(path_is_finite(&p));
            assert_eq!(degenerate_segments(&p, 1e-9), 0);
        }
        let k = LogSpiral::tightness_for(2.0, 0.1);
        let s = LogSpiral::from_start(start, Vec2::new(1.0, 0.0), 30.0, 2.0, k, false);
        assert!((s.radius_at(1.0) / 30.0 - 0.1).abs() < 1e-9);
    }

    #[test]
    fn degenerate_inputs_do_not_panic_or_nan() {
        let s = LogSpiral::from_start(
            Point::new(f64::NAN, 1.0),
            Vec2::ZERO,
            -3.0,
            f64::INFINITY,
            f64::NAN,
            false,
        );
        let z = LogSpiral::new(Point::ZERO, 0.0, 1e9, 0.0, 0.0, true, f64::NAN);
        let a = ArchimedeanSpiral::new(Point::ZERO, 0.0, 0.0, 0.0, 0.0, false, 0.0);
        for t in [-0.1, 0.0, 0.5, 1.0, 1.1] {
            for c in [&s as &dyn Curve, &z, &a] {
                assert!(c.point(t).is_finite());
                assert!(c.deriv(t).is_finite());
                assert!(c.tangent(t).is_finite());
                assert!(c.curvature(t).is_finite());
            }
        }
        // huge exponent is clamped, not infinite
        let h = LogSpiral::new(Point::ZERO, 1.0, 10.0, 0.0, 1000.0, false, 0.0);
        assert!(h.point(1.0).is_finite());
        assert!(LogSpiral::tightness_for(0.0, 0.0).is_finite());
        let p =
            ArchimedeanSpiral::from_center(Point::ZERO, 0.0, 0.0, 3.0, 5.0, false).to_bezpath(0.01);
        assert!(path_is_finite(&p));
        assert_eq!(degenerate_segments(&p, 1e-9), 0);
    }

    #[test]
    fn spiral_tail_is_g1_and_reproducible() {
        let curve = CubicBez::new((0.0, 0.0), (80.0, -60.0), (160.0, 60.0), (240.0, 0.0));
        for cw in [false, true] {
            for turns in [0.5, 1.5, 3.0] {
                let k = LogSpiral::tightness_for(turns, 0.15);
                let c = spiral_tail(&curve, turns, k, cw, 0.02);
                let path = c.to_bezpath_exact();
                assert!(path_is_finite(&path));
                assert_eq!(degenerate_segments(&path, 1e-9), 0);
                assert!(max_kink(&c) < 1e-6, "kink {}", max_kink(&c));
                assert!((c.start() - Point::new(0.0, 0.0)).hypot() < 1e-9);
                // the curve part is preserved
                assert!((c.point(0.0) - curve.point(0.0)).hypot() < 1e-9);
                assert_eq!(c, spiral_tail(&curve, turns, k, cw, 0.02));
                assert!(c.length() > curve.length());
            }
        }
        // Curling to the side the curve already bends to gives a G2 join: the fitted
        // segments on both sides of the join have (nearly) the same curvature there.
        let curve = CubicBez::new((0.0, 0.0), (120.0, 0.0), (220.0, -10.0), (250.0, -60.0));
        let (kc, k) = (curve.curvature(1.0), 0.12_f64);
        let r_g2 = 1.0 / (kc.abs() * (1.0 + k * k).sqrt());
        assert!(r_g2 > 0.04 * curve.length() && r_g2 < 0.25 * curve.length());
        let c = spiral_tail(&curve, 2.0, k, kc < 0.0, 0.002);
        let segs = c.segments();
        let j = segs
            .iter()
            .position(|s| (s.p3 - Point::new(250.0, -60.0)).hypot() < 1e-9)
            .unwrap();
        let k_in = segs[j].curvature(1.0);
        let k_out = segs[j + 1].curvature(0.0);
        assert!((k_in / kc - 1.0).abs() < 0.1, "{k_in} vs {kc}");
        assert!((k_out / kc - 1.0).abs() < 0.1, "{k_out} vs {kc}");
    }

    #[test]
    fn spiral_tail_with_tight_eye_stays_g1() {
        // Eyes smaller than the fit's sample spacing used to read as polyline corners.
        let curve = CubicBez::new((0.0, 0.0), (80.0, -60.0), (160.0, 60.0), (240.0, 0.0));
        for (turns, k, acc) in [
            (3.0, LogSpiral::tightness_for(3.0, 0.01), 0.5),
            (3.0, 0.5, 0.02),
            (10.0, 0.3, 0.02),
        ] {
            for cw in [false, true] {
                let c = spiral_tail(&curve, turns, k, cw, acc);
                let p = c.to_bezpath_exact();
                assert!(path_is_finite(&p));
                assert_eq!(degenerate_segments(&p, 1e-9), 0);
                assert!(
                    max_kink(&c) < 1e-6,
                    "turns={turns} k={k} kink {}",
                    max_kink(&c)
                );
            }
        }
    }

    #[test]
    fn spiral_tail_degenerate_curve() {
        let dot = CubicBez::new((5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0));
        let c = spiral_tail(&dot, 2.0, 0.1, false, 0.01);
        let p = c.to_bezpath_exact();
        assert!(path_is_finite(&p));
        assert_eq!(degenerate_segments(&p, 1e-9), 0);
        let c = spiral_tail(&dot, 0.0, 0.1, false, f64::NAN);
        assert!(path_is_finite(&c.to_bezpath_exact()));
        assert_eq!(degenerate_segments(&c.to_bezpath_exact(), 1e-9), 0);
    }
}
