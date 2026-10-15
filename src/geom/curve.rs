//! Parametric curve abstraction: positions, derivatives, tangents, normals, curvature,
//! arc-length reparameterisation, sampling and Bézier fitting.

use super::{EPS, H};
use kurbo::{
    Affine, BezPath, CubicBez, CurveFitSample, ParamCurve, ParamCurveDeriv, ParamCurveFit, PathEl,
    Point, Vec2,
};
use std::ops::Range;

/// A smooth 2-D parametric curve on `t ∈ [0, 1]`.
///
/// Only [`point`](Curve::point) is required; everything else has robust numerical defaults.
pub trait Curve {
    /// Position at `t`. Implementations should accept `t` slightly outside `[0,1]`.
    fn point(&self, t: f64) -> Point;

    /// First derivative `dP/dt`. Default: central finite difference.
    fn deriv(&self, t: f64) -> Vec2 {
        let h = H;
        let (a, b) = if t < h {
            (t, t + h)
        } else if t > 1.0 - h {
            (t - h, t)
        } else {
            (t - h, t + h)
        };
        (self.point(b) - self.point(a)) / (b - a)
    }

    /// Unit tangent. Robust against near-zero derivatives: falls back to widening secants,
    /// and finally to `(1, 0)`.
    fn tangent(&self, t: f64) -> Vec2 {
        let d = self.deriv(t);
        if d.hypot() > EPS {
            return d / d.hypot();
        }
        let mut h = 1e-3;
        while h < 0.5 {
            let a = (t - h).max(0.0);
            let b = (t + h).min(1.0);
            let s = self.point(b) - self.point(a);
            if s.hypot() > EPS {
                return s / s.hypot();
            }
            h *= 4.0;
        }
        Vec2::new(1.0, 0.0)
    }

    /// Unit left normal (tangent rotated by +90°, i.e. `(-ty, tx)`).
    fn normal(&self, t: f64) -> Vec2 {
        let tg = self.tangent(t);
        Vec2::new(0.0 - tg.y, tg.x)
    }

    /// Signed curvature `κ = (x'y'' − y'x'') / |P'|³`. Default: finite differences on `deriv`.
    fn curvature(&self, t: f64) -> f64 {
        let h = 1e-4;
        let a = (t - h).max(0.0);
        let b = (t + h).min(1.0);
        let d = self.deriv(t);
        let dd = (self.deriv(b) - self.deriv(a)) / (b - a);
        let speed3 = d.hypot().powi(3);
        if speed3 < EPS {
            return 0.0;
        }
        let k = d.cross(dd) / speed3;
        if k.is_finite() { k } else { 0.0 }
    }
}

/// Convenience methods available on every [`Curve`].
pub trait CurveExt: Curve {
    /// Uniformly spaced parameter samples (`n >= 2` points including both ends).
    fn sample(&self, n: usize) -> Vec<Point> {
        let n = n.max(2);
        (0..n)
            .map(|i| self.point(i as f64 / (n - 1) as f64))
            .collect()
    }

    /// Build an arc-length lookup table with `n` segments.
    fn arc_length(&self, n: usize) -> ArcLength {
        ArcLength::new(self, n)
    }

    /// Approximate total length.
    fn length(&self) -> f64 {
        self.arc_length(256).total()
    }

    /// `n` samples spaced uniformly by arc length: `(t, point)`.
    fn sample_by_arclen(&self, n: usize) -> Vec<(f64, Point)> {
        let n = n.max(2);
        let al = self.arc_length((n * 4).max(64));
        (0..n)
            .map(|i| {
                let s = i as f64 / (n - 1) as f64;
                let t = al.t_at_fraction(s);
                (t, self.point(t))
            })
            .collect()
    }

    /// Fit a cubic Bézier path to this curve with the given accuracy (canvas units).
    ///
    /// Uses the crate's Schneider fitter ([`super::fit::fit_curve`]): forward handles of
    /// positive length and G1 joins, so tangents and normals of the result stay usable.
    /// (kurbo's `fit_to_bezpath` is kept behind [`FitAdapter`] for callers that want it,
    /// but it can emit zero-length or reversed handles on smooth input.)
    fn to_bezpath(&self, accuracy: f64) -> BezPath {
        let acc = if accuracy.is_finite() && accuracy > 0.0 {
            accuracy
        } else {
            0.1
        };
        let mut path = super::fit::fit_curve(self, acc);
        if path.elements().is_empty() {
            // Degenerate (e.g. zero-length) curve: emit a tiny line so downstream stays well formed.
            path.move_to(self.point(0.0));
            path.line_to(self.point(1.0));
        }
        path
    }

    /// Restrict to a parameter sub-range (re-parameterised to `[0,1]`).
    fn sub(self, range: Range<f64>) -> SubCurve<Self>
    where
        Self: Sized,
    {
        SubCurve { curve: self, range }
    }

    /// Reverse direction.
    fn reversed(self) -> Reversed<Self>
    where
        Self: Sized,
    {
        Reversed(self)
    }

    /// Apply an affine transform.
    fn transformed(self, affine: Affine) -> Transformed<Self>
    where
        Self: Sized,
    {
        Transformed {
            curve: self,
            affine,
        }
    }
}

impl<C: Curve + ?Sized> CurveExt for C {}

impl<C: Curve + ?Sized> Curve for &C {
    fn point(&self, t: f64) -> Point {
        (**self).point(t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        (**self).deriv(t)
    }
    fn tangent(&self, t: f64) -> Vec2 {
        (**self).tangent(t)
    }
    fn curvature(&self, t: f64) -> f64 {
        (**self).curvature(t)
    }
}

impl<C: Curve + ?Sized> Curve for Box<C> {
    fn point(&self, t: f64) -> Point {
        (**self).point(t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        (**self).deriv(t)
    }
    fn tangent(&self, t: f64) -> Vec2 {
        (**self).tangent(t)
    }
    fn curvature(&self, t: f64) -> f64 {
        (**self).curvature(t)
    }
}

impl Curve for CubicBez {
    fn point(&self, t: f64) -> Point {
        self.eval(t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        ParamCurveDeriv::deriv(self).eval(t).to_vec2()
    }
}

impl Curve for kurbo::Line {
    fn point(&self, t: f64) -> Point {
        self.eval(t)
    }
    fn deriv(&self, _t: f64) -> Vec2 {
        self.p1 - self.p0
    }
}

impl Curve for kurbo::QuadBez {
    fn point(&self, t: f64) -> Point {
        self.eval(t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        ParamCurveDeriv::deriv(self).eval(t).to_vec2()
    }
}

/// A curve defined by a closure.
pub struct FnCurve<F: Fn(f64) -> Point>(pub F);

impl<F: Fn(f64) -> Point> Curve for FnCurve<F> {
    fn point(&self, t: f64) -> Point {
        (self.0)(t)
    }
}

/// A composite curve made of consecutive cubic segments, parameterised so that each
/// segment occupies an equal share of `t` (or a share proportional to its length if
/// built with [`Composite::new_by_length`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Composite {
    segs: Vec<CubicBez>,
    /// Cumulative parameter breakpoints, `breaks[0] = 0`, `breaks[n] = 1`.
    breaks: Vec<f64>,
}

impl Composite {
    /// Equal-parameter segments. Panics if `segs` is empty.
    pub fn new(segs: Vec<CubicBez>) -> Composite {
        assert!(!segs.is_empty(), "Composite needs at least one segment");
        let n = segs.len();
        let breaks = (0..=n).map(|i| i as f64 / n as f64).collect();
        Composite { segs, breaks }
    }

    /// Parameter shares proportional to segment arc length (so uniform `t` is close to uniform speed).
    pub fn new_by_length(segs: Vec<CubicBez>) -> Composite {
        assert!(!segs.is_empty(), "Composite needs at least one segment");
        let lens: Vec<f64> = segs
            .iter()
            .map(|s| kurbo::ParamCurveArclen::arclen(s, 1e-6).max(EPS))
            .collect();
        let total: f64 = lens.iter().sum();
        let mut breaks = Vec::with_capacity(segs.len() + 1);
        let mut acc = 0.0;
        breaks.push(0.0);
        for l in &lens {
            acc += l / total;
            breaks.push(acc);
        }
        *breaks.last_mut().unwrap() = 1.0;
        Composite { segs, breaks }
    }

    /// Build from a `BezPath` (first subpath; lines and quads are promoted to cubics).
    /// Returns `None` if the path has no drawable segments.
    pub fn from_bezpath(path: &BezPath) -> Option<Composite> {
        let mut segs = Vec::new();
        let mut started = false;
        for el in path.elements() {
            match el {
                PathEl::MoveTo(_) => {
                    if started {
                        break;
                    }
                    started = true;
                }
                PathEl::ClosePath => break,
                _ => {}
            }
        }
        for seg in path.segments() {
            segs.push(seg.to_cubic());
            // stop at the end of the first subpath
            if segs.len() >= count_first_subpath_segments(path) {
                break;
            }
        }
        if segs.is_empty() {
            None
        } else {
            Some(Composite::new_by_length(segs))
        }
    }

    /// Catmull-Rom-like smooth interpolation through `pts` (`>= 2` points), with
    /// `tension` in `[0,1]` (0 = straight lines, ~0.5 = classic Catmull-Rom).
    pub fn through_points(pts: &[Point], tension: f64) -> Option<Composite> {
        if pts.len() < 2 {
            return None;
        }
        let n = pts.len();
        let k = tension.clamp(0.0, 1.0) / 3.0 * 2.0;
        let mut segs = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let p0 = if i == 0 { pts[0] } else { pts[i - 1] };
            let p1 = pts[i];
            let p2 = pts[i + 1];
            let p3 = if i + 2 < n { pts[i + 2] } else { pts[n - 1] };
            let c1 = p1 + (p2 - p0) * (k * 0.5);
            let c2 = p2 - (p3 - p1) * (k * 0.5);
            segs.push(CubicBez::new(p1, c1, c2, p2));
        }
        Some(Composite::new_by_length(segs))
    }

    pub fn segments(&self) -> &[CubicBez] {
        &self.segs
    }

    pub fn breaks(&self) -> &[f64] {
        &self.breaks
    }

    /// Locate segment index and local parameter for global `t`.
    pub fn locate(&self, t: f64) -> (usize, f64) {
        let n = self.segs.len();
        let t = if t.is_nan() { 0.0 } else { t };
        if t <= 0.0 {
            let w = self.breaks[1] - self.breaks[0];
            return (0, t / w);
        }
        if t >= 1.0 {
            let w = self.breaks[n] - self.breaks[n - 1];
            return (n - 1, 1.0 + (t - 1.0) / w);
        }
        // binary search on breaks
        let i = match self.breaks.binary_search_by(|b| b.total_cmp(&t)) {
            Ok(i) => i.min(n - 1),
            Err(i) => i - 1,
        };
        let w = (self.breaks[i + 1] - self.breaks[i]).max(EPS);
        (i, (t - self.breaks[i]) / w)
    }

    pub fn to_bezpath_exact(&self) -> BezPath {
        let mut p = BezPath::new();
        p.move_to(self.segs[0].p0);
        for s in &self.segs {
            p.curve_to(s.p1, s.p2, s.p3);
        }
        p
    }

    pub fn start(&self) -> Point {
        self.segs[0].p0
    }

    pub fn end(&self) -> Point {
        self.segs[self.segs.len() - 1].p3
    }
}

fn count_first_subpath_segments(path: &BezPath) -> usize {
    let mut n = 0;
    let mut started = false;
    for el in path.elements() {
        match el {
            PathEl::MoveTo(_) => {
                if started {
                    break;
                }
                started = true;
            }
            PathEl::ClosePath => {
                n += 1;
                break;
            }
            _ => n += 1,
        }
    }
    n
}

impl Curve for Composite {
    fn point(&self, t: f64) -> Point {
        let (i, u) = self.locate(t);
        self.segs[i].eval(u)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        let (i, u) = self.locate(t);
        let w = (self.breaks[i + 1] - self.breaks[i]).max(EPS);
        ParamCurveDeriv::deriv(&self.segs[i]).eval(u).to_vec2() / w
    }
}

/// A parameter sub-range of another curve.
pub struct SubCurve<C: Curve> {
    pub curve: C,
    pub range: Range<f64>,
}

impl<C: Curve> Curve for SubCurve<C> {
    fn point(&self, t: f64) -> Point {
        self.curve
            .point(self.range.start + (self.range.end - self.range.start) * t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        let w = self.range.end - self.range.start;
        self.curve.deriv(self.range.start + w * t) * w
    }
}

/// A curve traversed backwards.
pub struct Reversed<C: Curve>(pub C);

impl<C: Curve> Curve for Reversed<C> {
    fn point(&self, t: f64) -> Point {
        self.0.point(1.0 - t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        -self.0.deriv(1.0 - t)
    }
}

/// A curve under an affine map.
pub struct Transformed<C: Curve> {
    pub curve: C,
    pub affine: Affine,
}

impl<C: Curve> Curve for Transformed<C> {
    fn point(&self, t: f64) -> Point {
        self.affine * self.curve.point(t)
    }
    fn deriv(&self, t: f64) -> Vec2 {
        let d = self.curve.deriv(t);
        let [a, b, c, d2, _, _] = self.affine.as_coeffs();
        Vec2::new(a * d.x + c * d.y, b * d.x + d2 * d.y)
    }
}

/// Arc-length lookup table for a curve: maps between `t` and arc length `s`.
#[derive(Debug, Clone)]
pub struct ArcLength {
    /// `ts[i]` and cumulative `ss[i]`.
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ArcLength {
    pub fn new<C: Curve + ?Sized>(curve: &C, n: usize) -> ArcLength {
        let n = n.max(1);
        let mut ts = Vec::with_capacity(n + 1);
        let mut ss = Vec::with_capacity(n + 1);
        let mut prev = curve.point(0.0);
        let mut acc = 0.0;
        ts.push(0.0);
        ss.push(0.0);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let p = curve.point(t);
            let d = (p - prev).hypot();
            acc += if d.is_finite() { d } else { 0.0 };
            ts.push(t);
            ss.push(acc);
            prev = p;
        }
        ArcLength { ts, ss }
    }

    pub fn total(&self) -> f64 {
        *self.ss.last().unwrap()
    }

    /// Arc length at parameter `t`.
    pub fn s_at(&self, t: f64) -> f64 {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        let n = self.ts.len() - 1;
        let x = t * n as f64;
        let i = (x.floor() as usize).min(n.saturating_sub(1));
        let f = x - i as f64;
        self.ss[i] + (self.ss[i + 1] - self.ss[i]) * f
    }

    /// Parameter `t` at arc length `s`.
    pub fn t_at(&self, s: f64) -> f64 {
        let total = self.total();
        if total.is_nan() || total <= EPS || s.is_nan() {
            return if s.is_nan() { 0.0 } else { s.clamp(0.0, 1.0) };
        }
        let s = s.clamp(0.0, total);
        let i = match self.ss.binary_search_by(|v| v.total_cmp(&s)) {
            Ok(i) => return self.ts[i],
            Err(i) => i.clamp(1, self.ss.len() - 1),
        };
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 - s0 > EPS {
            (s - s0) / (s1 - s0)
        } else {
            0.0
        };
        self.ts[i - 1] + (self.ts[i] - self.ts[i - 1]) * f
    }

    /// Parameter at a fraction `f ∈ [0,1]` of total length.
    pub fn t_at_fraction(&self, f: f64) -> f64 {
        self.t_at(f.clamp(0.0, 1.0) * self.total())
    }
}

/// Adapts a [`Curve`] to kurbo's [`ParamCurveFit`] so [`kurbo::fit_to_bezpath`] can be used.
pub struct FitAdapter<'a, C: Curve + ?Sized> {
    pub curve: &'a C,
}

impl<C: Curve + ?Sized> ParamCurveFit for FitAdapter<'_, C> {
    fn sample_pt_tangent(&self, t: f64, _sign: f64) -> CurveFitSample {
        CurveFitSample {
            p: self.curve.point(t),
            tangent: self.curve.tangent(t),
        }
    }

    fn sample_pt_deriv(&self, t: f64) -> (Point, Vec2) {
        (self.curve.point(t), self.curve.deriv(t))
    }

    fn break_cusp(&self, range: Range<f64>) -> Option<f64> {
        // Look for a sharp direction change strictly inside the range. Tangents are compared
        // two samples apart so a sample landing exactly on the cusp (where the numerical
        // tangent is the average of both sides) cannot hide it.
        const N: usize = 16;
        let w = range.end - range.start;
        if w < 1e-6 {
            return None;
        }
        let tangents: Vec<Vec2> = (0..=N)
            .map(|i| self.curve.tangent(range.start + w * i as f64 / N as f64))
            .collect();
        for i in 2..=N {
            let (ta, tb) = (tangents[i - 2], tangents[i]);
            if ta.dot(tb) < 0.2 {
                let mut lo = range.start + w * (i - 2) as f64 / N as f64;
                let mut hi = range.start + w * i as f64 / N as f64;
                for _ in 0..24 {
                    let mid = 0.5 * (lo + hi);
                    let tm = self.curve.tangent(mid);
                    if tm.dot(ta) > tm.dot(tb) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let c = 0.5 * (lo + hi);
                if c > range.start + 1e-6 && c < range.end - 1e-6 {
                    return Some(c);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn circle() -> FnCurve<impl Fn(f64) -> Point> {
        FnCurve(move |t: f64| Point::new((2.0 * PI * t).cos() * 10.0, (2.0 * PI * t).sin() * 10.0))
    }

    #[test]
    fn numeric_derivative_and_curvature_of_circle() {
        let c = circle();
        let d = c.deriv(0.25);
        assert!((d.x - (-2.0 * PI * 10.0)).abs() < 1e-3, "{d:?}");
        assert!((c.curvature(0.3) - 0.1).abs() < 1e-3);
        let n = c.normal(0.0);
        // tangent at t=0 points up (+y); the left normal points to -x
        assert!((n.x + 1.0).abs() < 1e-4 && n.y.abs() < 1e-4);
    }

    #[test]
    fn arc_length_of_circle() {
        let c = circle();
        assert!((c.length() - 2.0 * PI * 10.0).abs() < 0.05);
        let s = c.sample_by_arclen(5);
        assert_eq!(s.len(), 5);
        assert!((s[2].0 - 0.5).abs() < 1e-3);
    }

    #[test]
    fn composite_is_continuous_and_fits() {
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(50.0, 80.0),
            Point::new(120.0, 20.0),
            Point::new(200.0, 90.0),
        ];
        let c = Composite::through_points(&pts, 0.5).unwrap();
        assert_eq!(c.point(0.0), pts[0]);
        assert_eq!(c.point(1.0), pts[3]);
        for i in 1..c.segments().len() {
            let b = c.breaks()[i];
            let a = c.point(b - 1e-9);
            let bpt = c.point(b + 1e-9);
            assert!((a - bpt).hypot() < 1e-5);
        }
        let path = c.to_bezpath(0.05);
        assert!(path.elements().len() >= 2);
        let fitted = Composite::from_bezpath(&path).unwrap();
        for i in 0..=20 {
            let t = i as f64 / 20.0;
            let p = c.point(t);
            // nearest distance from fitted sample set
            let d = fitted
                .sample(400)
                .iter()
                .map(|q| (*q - p).hypot())
                .fold(f64::MAX, f64::min);
            assert!(d < 0.5, "t={t} d={d}");
        }
    }

    #[test]
    fn zero_length_curve_is_robust() {
        let c = FnCurve(|_t: f64| Point::new(3.0, 4.0));
        let tg = c.tangent(0.5);
        assert!(tg.hypot().is_finite());
        assert_eq!(c.curvature(0.5), 0.0);
        let p = c.to_bezpath(0.1);
        assert!(!p.elements().is_empty());
        let al = c.arc_length(10);
        assert_eq!(al.total(), 0.0);
        assert!(al.t_at_fraction(0.5).is_finite());
    }

    #[test]
    fn nan_parameters_do_not_panic() {
        let c = Composite::through_points(
            &[
                Point::new(0.0, 0.0),
                Point::new(10.0, 5.0),
                Point::new(20.0, 0.0),
            ],
            0.5,
        )
        .unwrap();
        assert!(c.point(f64::NAN).is_finite());
        let al = c.arc_length(16);
        assert!(al.t_at(f64::NAN).is_finite() && al.s_at(f64::NAN).is_finite());
        assert!(al.t_at_fraction(f64::NAN).is_finite());
    }

    #[test]
    fn cusp_is_detected_and_fit_stays_close() {
        // a curve with a cusp at t=0.5: |t-0.5| in y
        let c = FnCurve(|t: f64| Point::new(100.0 * t, 100.0 * (t - 0.5).abs()));
        let adapter = FitAdapter { curve: &c };
        let cusp = adapter.break_cusp(0.0..1.0).unwrap();
        assert!((cusp - 0.5).abs() < 1e-3);
        let path = c.to_bezpath(0.1);
        let fitted = Composite::from_bezpath(&path).unwrap();
        let p = fitted.point(0.5);
        assert!(p.is_finite());
    }
}
