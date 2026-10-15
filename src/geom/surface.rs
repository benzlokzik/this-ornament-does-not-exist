//! Two-parameter shapes: ribbons, petals and leaves all expose a [`Surface`] so that
//! ornamentation (contour stripes, hatching, veins) can be generated in parameter space
//! and follows the shape's own curvature.

use super::curve::{Curve, CurveExt, FnCurve};
use kurbo::{BezPath, Point};

/// A parametric patch `P(u, v)` with `u ∈ [0,1]` running *along* the shape
/// (base → tip for petals and leaves, start → end for ribbons) and `v ∈ [-1, 1]`
/// running *across* it (`v = 0` is the centreline / midrib, `v = ±1` the two edges).
pub trait Surface {
    /// Point on the surface.
    fn point(&self, u: f64, v: f64) -> Point;

    /// Closed outline path of the surface (edge `v = +1` forward, edge `v = -1` backward).
    /// Default: sampled from `point` and fitted with `accuracy`.
    fn outline(&self, accuracy: f64) -> BezPath {
        let a = self.iso_v(1.0).to_bezpath(accuracy);
        let b = self.iso_v(-1.0).reversed().to_bezpath(accuracy);
        join_closed(&a, &b)
    }

    /// Curve of constant `v` (runs along the shape).
    fn iso_v(&self, v: f64) -> Box<dyn Curve + '_> {
        Box::new(FnCurve(move |u| self.point(u, v)))
    }

    /// Curve of constant `u` (runs across the shape from `v = -1` to `v = 1`).
    fn iso_u(&self, u: f64) -> Box<dyn Curve + '_> {
        Box::new(FnCurve(move |t| self.point(u, 2.0 * t - 1.0)))
    }

    /// Optional local half-width at `u` in canvas units (used to scale hatching density).
    /// Default: distance between the two edges / 2.
    fn half_width(&self, u: f64) -> f64 {
        (self.point(u, 1.0) - self.point(u, -1.0)).hypot() * 0.5
    }
}

impl<S: Surface + ?Sized> Surface for &S {
    fn point(&self, u: f64, v: f64) -> Point {
        (**self).point(u, v)
    }
    fn outline(&self, accuracy: f64) -> BezPath {
        (**self).outline(accuracy)
    }
    fn half_width(&self, u: f64) -> f64 {
        (**self).half_width(u)
    }
}

impl<S: Surface + ?Sized> Surface for Box<S> {
    fn point(&self, u: f64, v: f64) -> Point {
        (**self).point(u, v)
    }
    fn outline(&self, accuracy: f64) -> BezPath {
        (**self).outline(accuracy)
    }
    fn half_width(&self, u: f64) -> f64 {
        (**self).half_width(u)
    }
}

/// Concatenate two open paths into one closed path (`b` is expected to start near `a`'s end).
pub fn join_closed(a: &BezPath, b: &BezPath) -> BezPath {
    let mut out = BezPath::new();
    let mut first = true;
    for el in a.elements().iter().chain(b.elements().iter()) {
        let last: Option<Point> = out.elements().last().and_then(|e| e.end_point());
        match el {
            kurbo::PathEl::MoveTo(p) => {
                if first {
                    out.move_to(*p);
                    first = false;
                } else if last.is_none_or(|l| (l - *p).hypot() > 1e-9) {
                    out.line_to(*p);
                }
            }
            kurbo::PathEl::ClosePath => {}
            other => out.push(*other),
        }
    }
    if !first {
        out.close_path();
    }
    out
}

/// A ruled surface between two edge curves: `P(u, v) = lerp(left(u), right(u), (v+1)/2)`.
/// Useful for petals authored as a pair of edge Béziers.
pub struct RuledSurface<L: Curve, R: Curve> {
    /// Edge at `v = -1`.
    pub left: L,
    /// Edge at `v = +1`.
    pub right: R,
}

impl<L: Curve, R: Curve> Surface for RuledSurface<L, R> {
    fn point(&self, u: f64, v: f64) -> Point {
        let a = self.left.point(u);
        let b = self.right.point(u);
        a.lerp(b, (v + 1.0) * 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::{CubicBez, Shape};

    #[test]
    fn ruled_surface_outline_is_closed_and_finite() {
        let left = CubicBez::new((0.0, 0.0), (-40.0, 30.0), (-40.0, 70.0), (0.0, 100.0));
        let right = CubicBez::new((0.0, 0.0), (40.0, 30.0), (40.0, 70.0), (0.0, 100.0));
        let s = RuledSurface { left, right };
        let o = s.outline(0.05);
        assert!(matches!(
            o.elements().last(),
            Some(kurbo::PathEl::ClosePath)
        ));
        assert!(o.area().abs() > 1000.0);
        assert_eq!(s.point(0.5, 0.0).x, 0.0);
        assert!(s.half_width(0.5) > 10.0);
    }
}
