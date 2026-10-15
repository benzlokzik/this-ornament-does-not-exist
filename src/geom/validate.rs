//! Geometry validation helpers: finite coordinates and degenerate segment detection.

use kurbo::{BezPath, PathEl, Point};

/// `true` if all coordinates are finite.
pub fn path_is_finite(path: &BezPath) -> bool {
    path.elements().iter().all(|el| match el {
        PathEl::MoveTo(p) | PathEl::LineTo(p) => p.is_finite(),
        PathEl::QuadTo(a, b) => a.is_finite() && b.is_finite(),
        PathEl::CurveTo(a, b, c) => a.is_finite() && b.is_finite() && c.is_finite(),
        PathEl::ClosePath => true,
    })
}

/// Number of segments whose control polygon is shorter than `eps` (effectively zero length).
pub fn degenerate_segments(path: &BezPath, eps: f64) -> usize {
    let mut last: Option<Point> = None;
    let mut n = 0;
    for el in path.elements() {
        match el {
            PathEl::MoveTo(p) => last = Some(*p),
            PathEl::LineTo(p) => {
                if let Some(l) = last
                    && (*p - l).hypot() < eps
                {
                    n += 1;
                }
                last = Some(*p);
            }
            PathEl::QuadTo(a, p) => {
                if let Some(l) = last
                    && (*a - l).hypot() + (*p - *a).hypot() < eps
                {
                    n += 1;
                }
                last = Some(*p);
            }
            PathEl::CurveTo(a, b, p) => {
                if let Some(l) = last
                    && (*a - l).hypot() + (*b - *a).hypot() + (*p - *b).hypot() < eps
                {
                    n += 1;
                }
                last = Some(*p);
            }
            PathEl::ClosePath => {}
        }
    }
    n
}

/// Remove non-finite elements and zero-length segments. Subpaths that end up with fewer
/// than two distinct points are dropped. Returns a well-formed path (possibly empty).
pub fn sanitize(path: &BezPath, eps: f64) -> BezPath {
    let mut out = BezPath::new();
    let mut last: Option<Point> = None;
    let mut sub_start = out.elements().len();
    let mut sub_len = 0usize;
    let flush = |out: &mut BezPath, sub_start: usize, sub_len: usize| {
        if sub_len < 2 {
            out.truncate(sub_start);
        }
    };
    for el in path.elements() {
        match el {
            PathEl::MoveTo(p) => {
                flush(&mut out, sub_start, sub_len);
                if p.is_finite() {
                    sub_start = out.elements().len();
                    out.move_to(*p);
                    last = Some(*p);
                    sub_len = 1;
                } else {
                    last = None;
                    sub_len = 0;
                    sub_start = out.elements().len();
                }
            }
            PathEl::LineTo(p) => {
                if let Some(l) = last
                    && p.is_finite()
                    && (*p - l).hypot() >= eps
                {
                    out.line_to(*p);
                    last = Some(*p);
                    sub_len += 1;
                }
            }
            PathEl::QuadTo(a, p) => {
                if let Some(l) = last
                    && a.is_finite()
                    && p.is_finite()
                    && (*a - l).hypot() + (*p - *a).hypot() >= eps
                {
                    out.quad_to(*a, *p);
                    last = Some(*p);
                    sub_len += 1;
                }
            }
            PathEl::CurveTo(a, b, p) => {
                if let Some(l) = last
                    && a.is_finite()
                    && b.is_finite()
                    && p.is_finite()
                    && (*a - l).hypot() + (*b - *a).hypot() + (*p - *b).hypot() >= eps
                {
                    out.curve_to(*a, *b, *p);
                    last = Some(*p);
                    sub_len += 1;
                }
            }
            PathEl::ClosePath => {
                if sub_len >= 2 {
                    out.close_path();
                }
            }
        }
    }
    flush(&mut out, sub_start, sub_len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_drops_nan_and_zero_length() {
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.line_to((0.0, 0.0));
        p.line_to((f64::NAN, 1.0));
        p.line_to((10.0, 10.0));
        p.move_to((5.0, 5.0));
        p.close_path();
        assert!(!path_is_finite(&p));
        assert_eq!(degenerate_segments(&p, 1e-9), 1);
        let s = sanitize(&p, 1e-9);
        assert!(path_is_finite(&s));
        assert_eq!(s.elements().len(), 2);
    }
}
