//! Point-list fitting and path joining.
//!
//! * [`fit_points`] / [`fit_points_closed`]: Schneider's least-squares cubic fitting
//!   ("An Algorithm for Automatically Fitting Digitized Curves", Graphics Gems, 1990) with
//!   Newton–Raphson reparameterisation, recursive splitting at the point of maximum error
//!   and corner detection.
//! * [`resample_by_arclen`], [`simplify`] (Douglas–Peucker): polyline utilities.
//! * [`join_g1`], [`max_kink`]: concatenate fitted paths into one tangent-continuous
//!   [`Composite`] and measure how continuous it really is.

use super::EPS;
use super::curve::{Composite, Curve, CurveExt};
use super::validate::sanitize;
use kurbo::{BezPath, CubicBez, ParamCurve, ParamCurveDeriv, Point, Vec2};

/// Interior points whose polyline turning angle exceeds this (radians, 75°) are treated as
/// corners: the fit splits there and does not smooth the tangent across them.
pub const CORNER_ANGLE: f64 = 75.0 * std::f64::consts::PI / 180.0;

/// Maximum Newton–Raphson reparameterisation rounds before splitting.
const MAX_REPARAM: usize = 50;
/// Reparameterisation stops early once a round fails to shrink the squared error below
/// this fraction of the previous one.
const REPARAM_PROGRESS: f64 = 0.98;
/// A fitted cubic's control polygon may be at most this many times the length of the
/// polyline it approximates.
const MAX_POLYGON: f64 = 2.5;
/// Depth after which [`fit_cubic`] splits runs in half instead of at the worst point, which
/// bounds the recursion by `MAX_DEPTH + log₂(n)`.
const MAX_DEPTH: usize = 48;
/// [`fit_curve`] bisects a sample interval while the curve turns more than this (15°) across it.
const SAMPLE_TURN: f64 = 15.0 * std::f64::consts::PI / 180.0;
/// Bisection depth and total sample limits of [`fit_curve`]'s refinement.
const MAX_REFINE: usize = 24;
const MAX_SAMPLES: usize = 100_000;

/// Fit a smooth open path of cubic Béziers through `points`, staying within `tolerance`
/// (canvas units) of every input point.
///
/// Non-finite points are dropped and consecutive near-duplicates (closer than
/// `tolerance / 100`) merged. Fewer than two distinct points give an empty path, exactly two
/// a single line. Interior points with a turning angle above [`CORNER_ANGLE`] become corners.
pub fn fit_points(points: &[Point], tolerance: f64) -> BezPath {
    let tol = sanitize_tolerance(tolerance);
    fit_open(&clean_points(points, tol), None, None, tol, true)
}

/// Fit a cubic Bézier path to any [`Curve`] within `tolerance`: the curve is sampled
/// uniformly by arc length (`8·√(length/tolerance)` samples, clamped to `[64, 2000]`), each
/// interval is bisected in `t` while the curve turns more than 15° across it (so tight
/// coils and spiral eyes are neither skipped nor mistaken for corners), and the samples are
/// fitted with [`fit_points`]' algorithm, using the curve's exact end tangents.
///
/// Unlike [`CurveExt::to_bezpath`](super::CurveExt::to_bezpath) (kurbo's fitter), every
/// handle is guaranteed to point forwards with positive length and all interior joins are G1
/// by construction, so tangents and normals of the result are trustworthy everywhere. There
/// is no corner splitting: after refinement a polyline corner can only remain below
/// `tolerance / 100` (a spiral eye collapsing, or a genuine cusp), where rounding it stays
/// within tolerance. Zero-length curves give an empty path.
pub fn fit_curve<C: Curve + ?Sized>(curve: &C, tolerance: f64) -> BezPath {
    let tol = sanitize_tolerance(tolerance);
    let len = curve.length();
    let n = if len.is_finite() {
        ((len / tol).sqrt() * 8.0).clamp(64.0, 2000.0) as usize
    } else {
        64
    };
    let base = curve.sample_by_arclen(n);
    let mut pts = vec![base[0].1];
    for w in base.windows(2) {
        refine(curve, w[0], w[1], tol, MAX_REFINE, &mut pts);
    }
    let pts = clean_points(&pts, tol);
    fit_open(
        &pts,
        Some(curve.tangent(0.0)),
        Some(-curve.tangent(1.0)),
        tol,
        false,
    )
}

/// Push the samples of `curve` after `a` up to and including `b`, bisecting while the
/// tangents at both ends or the midpoint deviate from the chord by more than [`SAMPLE_TURN`]
/// or the midpoint lies more than `tol` off the chord, as long as the chord is longer than
/// the duplicate distance.
fn refine<C: Curve + ?Sized>(
    curve: &C,
    a: (f64, Point),
    b: (f64, Point),
    tol: f64,
    depth: usize,
    out: &mut Vec<Point>,
) {
    let chord = b.1 - a.1;
    let len = chord.hypot();
    if depth > 0 && out.len() < MAX_SAMPLES && len.is_finite() && len > dedup_distance(tol) {
        let tm = 0.5 * (a.0 + b.0);
        let m = (tm, curve.point(tm));
        let dir = chord / len;
        let cos_max = SAMPLE_TURN.cos();
        if (m.1 - a.1).cross(dir).abs() > tol
            || [a.0, tm, b.0]
                .into_iter()
                .any(|t| curve.tangent(t).dot(dir) < cos_max)
        {
            refine(curve, a, m, tol, depth - 1, out);
            refine(curve, m, b, tol, depth - 1, out);
            return;
        }
    }
    out.push(b.1);
}

/// Shared open fit on cleaned points with optional end tangents (`t_end` points backwards),
/// split at polyline corners when `corners` is set.
fn fit_open(
    pts: &[Point],
    t_start: Option<Vec2>,
    t_end: Option<Vec2>,
    tol: f64,
    corners: bool,
) -> BezPath {
    let mut path = BezPath::new();
    match pts.len() {
        0 | 1 => return path,
        2 => {
            path.move_to(pts[0]);
            path.line_to(pts[1]);
            return path;
        }
        _ => {}
    }
    let valid = |t: Option<Vec2>| {
        t.filter(|v| v.is_finite() && v.hypot() > EPS)
            .map(|v| v / v.hypot())
    };
    let (t_start, t_end) = (valid(t_start), valid(t_end));
    path.move_to(pts[0]);
    let mut cuts = vec![0];
    cuts.extend(
        (1..pts.len() - 1).filter(|&i| corners && is_corner(pts[i - 1], pts[i], pts[i + 1])),
    );
    cuts.push(pts.len() - 1);
    let last = cuts.len() - 2;
    for (k, w) in cuts.windows(2).enumerate() {
        let t1 = if k == 0 { t_start } else { None };
        let t2 = if k == last { t_end } else { None };
        fit_run(&pts[w[0]..=w[1]], t1, t2, tol, &mut path);
    }
    path
}

/// Fit a closed loop through `points` (the last point need not repeat the first).
///
/// Without corners the seam at `points[0]` is tangent-continuous (the tangent there is the
/// central difference of its neighbours). With corners the loop is cut at them. Fewer than
/// two distinct points give an empty path; two give a closed back-and-forth line.
pub fn fit_points_closed(points: &[Point], tolerance: f64) -> BezPath {
    let tol = sanitize_tolerance(tolerance);
    let mut pts = clean_points(points, tol);
    let dedup = dedup_distance(tol);
    while pts.len() > 1 && (pts[pts.len() - 1] - pts[0]).hypot() < dedup {
        pts.pop();
    }
    let mut path = BezPath::new();
    let n = pts.len();
    if n < 2 {
        return path;
    }
    if n == 2 {
        path.move_to(pts[0]);
        path.line_to(pts[1]);
        path.close_path();
        return path;
    }
    let corners: Vec<usize> = (0..n)
        .filter(|&i| is_corner(pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]))
        .collect();
    if corners.is_empty() {
        // Smooth loop: one run around, both ends sharing the central tangent at pts[0].
        let mut seq = pts.clone();
        seq.push(pts[0]);
        let t = unit_or(
            pts[1] - pts[n - 1],
            unit_or(pts[1] - pts[0], Vec2::new(1.0, 0.0)),
        );
        path.move_to(pts[0]);
        fit_run(&seq, Some(t), Some(-t), tol, &mut path);
    } else {
        // Rotate so the loop starts at the first corner, then fit corner to corner.
        let c0 = corners[0];
        let mut seq: Vec<Point> = pts[c0..].iter().chain(pts[..c0].iter()).copied().collect();
        seq.push(pts[c0]);
        let mut cuts: Vec<usize> = corners.iter().map(|&c| (c + n - c0) % n).collect();
        cuts.push(n);
        path.move_to(seq[0]);
        for w in cuts.windows(2) {
            fit_run(&seq[w[0]..=w[1]], None, None, tol, &mut path);
        }
    }
    path.close_path();
    path
}

/// `n` points spaced uniformly by arc length along the polyline `points`, including both
/// ends. Non-finite points are ignored. A zero-length polyline yields `n` copies of its point;
/// `n == 0` or no finite point yields an empty vector.
pub fn resample_by_arclen(points: &[Point], n: usize) -> Vec<Point> {
    let pts: Vec<Point> = points.iter().copied().filter(|p| p.is_finite()).collect();
    if pts.is_empty() || n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![pts[0]];
    }
    let mut cum = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    cum.push(0.0);
    for w in pts.windows(2) {
        acc += (w[1] - w[0]).hypot();
        cum.push(acc);
    }
    if acc <= EPS {
        return vec![pts[0]; n];
    }
    let mut out = Vec::with_capacity(n);
    let mut j = 0;
    for i in 0..n {
        let s = acc * i as f64 / (n - 1) as f64;
        while j + 2 < pts.len() && cum[j + 1] < s {
            j += 1;
        }
        let seg = cum[j + 1] - cum[j];
        let f = if seg > EPS {
            ((s - cum[j]) / seg).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(pts[j].lerp(pts[j + 1], f));
    }
    // Exact end points regardless of rounding.
    out[n - 1] = pts[pts.len() - 1];
    out
}

/// Douglas–Peucker polyline simplification: keeps the end points and every point needed to
/// stay within `tolerance` of the original. Non-finite points are dropped.
pub fn simplify(points: &[Point], tolerance: f64) -> Vec<Point> {
    let pts: Vec<Point> = points.iter().copied().filter(|p| p.is_finite()).collect();
    if pts.len() <= 2 {
        return pts;
    }
    let tol = if tolerance.is_finite() {
        tolerance.max(0.0)
    } else {
        0.0
    };
    let mut keep = vec![false; pts.len()];
    keep[0] = true;
    keep[pts.len() - 1] = true;
    // Explicit stack instead of recursion: long polylines cannot overflow.
    let mut stack = vec![(0usize, pts.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (mut idx, mut dmax) = (a, -1.0);
        for (i, p) in pts.iter().enumerate().take(b).skip(a + 1) {
            let d = dist_to_segment(*p, pts[a], pts[b]);
            if d > dmax {
                dmax = d;
                idx = i;
            }
        }
        if dmax > tol {
            keep[idx] = true;
            stack.push((a, idx));
            stack.push((idx, b));
        }
    }
    pts.iter()
        .zip(keep)
        .filter_map(|(p, k)| k.then_some(*p))
        .collect()
}

/// Concatenate open paths into one [`Composite`] that is G1 at every junction.
///
/// Each part is sanitised (non-finite and zero-length segments dropped; empty parts skipped).
/// At a junction the next part's start is snapped onto the previous end (its first handle
/// moves with it), then both handles adjacent to the junction are rotated onto the bisector
/// of the incoming and outgoing tangents, keeping their lengths. Parts are expected to meet
/// with (nearly) parallel tangents; the bisector rule then only removes fitting noise.
/// Returns `None` when no part has a drawable segment.
pub fn join_g1(parts: &[BezPath]) -> Option<Composite> {
    let mut segs: Vec<CubicBez> = Vec::new();
    for part in parts {
        let clean = sanitize(part, EPS);
        let Some(c) = Composite::from_bezpath(&clean) else {
            continue;
        };
        let mut next: Vec<CubicBez> = c
            .segments()
            .iter()
            .copied()
            .filter(|s| polygon_len(s) > EPS)
            .collect();
        if next.is_empty() {
            continue;
        }
        if let Some(last) = segs.last_mut() {
            let p = last.p3;
            let delta = p - next[0].p0;
            next[0].p0 = p;
            next[0].p1 += delta;
            if let (Some(tin), Some(tout)) = (end_dir(last), start_dir(&next[0])) {
                let bis = tin + tout;
                if bis.hypot() > 1e-6 {
                    let d = bis / bis.hypot();
                    let min_in = polygon_len(last) * 1e-3;
                    let min_out = polygon_len(&next[0]) * 1e-3;
                    last.p2 = p - d * (p - last.p2).hypot().max(min_in);
                    next[0].p1 = p + d * (next[0].p1 - p).hypot().max(min_out);
                }
            }
        }
        segs.extend(next);
    }
    if segs.is_empty() {
        None
    } else {
        Some(Composite::new_by_length(segs))
    }
}

/// Largest tangent-direction change (radians, in `[0, π]`) at any junction between
/// consecutive segments of `c`. `0` means G1 everywhere.
pub fn max_kink(c: &Composite) -> f64 {
    c.segments()
        .windows(2)
        .filter_map(|w| {
            let a = end_dir(&w[0])?;
            let b = start_dir(&w[1])?;
            Some(a.cross(b).atan2(a.dot(b)).abs())
        })
        .fold(0.0, f64::max)
}

/// A minimal well-formed one-segment [`Composite`] (a straight stub of length `1e-6` from `p`
/// along `dir`) used as a fallback for degenerate inputs.
pub fn stub(p: Point, dir: Vec2) -> Composite {
    let p = if p.is_finite() { p } else { Point::ZERO };
    let d = unit_or(dir, Vec2::new(1.0, 0.0)) * 1e-6;
    Composite::new(vec![CubicBez::new(
        p,
        p + d / 3.0,
        p + d * (2.0 / 3.0),
        p + d,
    )])
}

/// Unit vector along `v`, or `fallback` if `v` is (nearly) zero or non-finite.
pub fn unit_or(v: Vec2, fallback: Vec2) -> Vec2 {
    let l = v.hypot();
    if l > EPS && l.is_finite() {
        v / l
    } else {
        fallback
    }
}

/// Outgoing direction at the start of a cubic (first non-coincident control point).
fn start_dir(s: &CubicBez) -> Option<Vec2> {
    [s.p1, s.p2, s.p3]
        .into_iter()
        .map(|q| q - s.p0)
        .find(|v| v.hypot() > EPS)
        .map(|v| v / v.hypot())
}

/// Incoming direction at the end of a cubic.
fn end_dir(s: &CubicBez) -> Option<Vec2> {
    [s.p2, s.p1, s.p0]
        .into_iter()
        .map(|q| s.p3 - q)
        .find(|v| v.hypot() > EPS)
        .map(|v| v / v.hypot())
}

fn polygon_len(s: &CubicBez) -> f64 {
    (s.p1 - s.p0).hypot() + (s.p2 - s.p1).hypot() + (s.p3 - s.p2).hypot()
}

fn sanitize_tolerance(t: f64) -> f64 {
    if t.is_finite() && t > 0.0 { t } else { 0.1 }
}

fn dedup_distance(tol: f64) -> f64 {
    (tol * 0.01).max(EPS)
}

/// Drop non-finite points and merge consecutive points closer than `tol / 100`. The last
/// finite point is kept exactly (it replaces a near-duplicate predecessor).
fn clean_points(points: &[Point], tol: f64) -> Vec<Point> {
    let d = dedup_distance(tol);
    let mut out: Vec<Point> = Vec::with_capacity(points.len());
    let mut last = None;
    for &p in points.iter().filter(|p| p.is_finite()) {
        last = Some(p);
        if out.last().is_none_or(|q| (p - *q).hypot() >= d) {
            out.push(p);
        }
    }
    if let Some(p) = last
        && out.len() >= 2
    {
        let n = out.len();
        if (p - out[n - 2]).hypot() >= d {
            out[n - 1] = p;
        } else {
            out.pop();
        }
    }
    out
}

fn is_corner(a: Point, b: Point, c: Point) -> bool {
    let (u, v) = (b - a, c - b);
    if u.hypot() <= EPS || v.hypot() <= EPS {
        return false;
    }
    u.cross(v).atan2(u.dot(v)).abs() > CORNER_ANGLE
}

fn dist_to_segment(p: Point, a: Point, b: Point) -> f64 {
    let ab = b - a;
    let l2 = ab.hypot2();
    if l2 <= EPS * EPS {
        return (p - a).hypot();
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    (p - (a + ab * t)).hypot()
}

/// Fit one corner-free run. `t1` points from `pts[0]` into the run, `t2` from the last point
/// back into the run (Schneider's convention). `None` estimates them from the neighbours.
fn fit_run(pts: &[Point], t1: Option<Vec2>, t2: Option<Vec2>, tol: f64, path: &mut BezPath) {
    let n = pts.len();
    if n < 2 {
        return;
    }
    if n == 2 && t1.is_none() && t2.is_none() {
        path.line_to(pts[1]);
        return;
    }
    let chord = unit_or(pts[n - 1] - pts[0], Vec2::new(1.0, 0.0));
    let t1 = t1.unwrap_or_else(|| unit_or(pts[1] - pts[0], chord));
    let t2 = t2.unwrap_or_else(|| unit_or(pts[n - 2] - pts[n - 1], -chord));
    fit_cubic(pts, t1, t2, tol, 0, path);
}

/// Schneider's recursive step: least-squares fit with fixed end tangents, improve the
/// parameterisation with Newton–Raphson while that keeps shrinking the error, otherwise split at the
/// worst point with a shared (central-difference) tangent so the halves join G1.
fn fit_cubic(pts: &[Point], t1: Vec2, t2: Vec2, tol: f64, depth: usize, path: &mut BezPath) {
    let n = pts.len();
    let (p0, p3) = (pts[0], pts[n - 1]);
    let seg_len = (p3 - p0).hypot();
    if n == 2 {
        // Two points: handles of a third of the chord along the given tangents.
        let a = seg_len / 3.0;
        path.curve_to(p0 + t1 * a, p3 + t2 * a, p3);
        return;
    }
    let tol2 = tol * tol;
    let mut u = chord_params(pts);
    let mut bez = generate_bezier(pts, &u, t1, t2);
    let (mut err, mut split) = max_error(pts, &bez, &u);
    // A closed run (p0 == p3) cannot be one cubic without degenerating; always split it.
    let closed = seg_len <= dedup_distance(tol);
    if !closed && err < tol2 {
        path.curve_to(bez.p1, bez.p2, bez.p3);
        return;
    }
    // Chord-length parameters are far off on runs whose speed varies (a cubic S-curve), so
    // reparameterise even when the first error is large; stop once it no longer improves.
    if !closed {
        for _ in 0..MAX_REPARAM {
            reparameterize(pts, &mut u, &bez);
            bez = generate_bezier(pts, &u, t1, t2);
            let prev = err;
            (err, split) = max_error(pts, &bez, &u);
            if err < tol2 {
                path.curve_to(bez.p1, bez.p2, bez.p3);
                return;
            }
            if err > REPARAM_PROGRESS * prev {
                break;
            }
        }
    }
    // Past the depth guard split in the middle: lopsided worst-point splits on long, tightly
    // coiled runs would otherwise exhaust it and leave an out-of-tolerance cubic.
    let s = if depth >= MAX_DEPTH { n / 2 } else { split }.clamp(1, n - 2);
    let fallback = unit_or(pts[s] - pts[s - 1], t1 * -1.0);
    let center = unit_or(pts[s - 1] - pts[s + 1], fallback);
    fit_cubic(&pts[..=s], t1, center, tol, depth + 1, path);
    fit_cubic(&pts[s..], -center, t2, tol, depth + 1, path);
}

/// Normalised cumulative chord-length parameters.
fn chord_params(pts: &[Point]) -> Vec<f64> {
    let mut u = Vec::with_capacity(pts.len());
    let mut acc = 0.0;
    u.push(0.0);
    for w in pts.windows(2) {
        acc += (w[1] - w[0]).hypot();
        u.push(acc);
    }
    if acc > EPS {
        u.iter_mut().for_each(|x| *x /= acc);
    } else {
        let n = (pts.len() - 1) as f64;
        u.iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = i as f64 / n);
    }
    u
}

/// Least-squares handle lengths `αl, αr` for `Q = (p0, p0 + αl t1, p3 + αr t2, p3)`
/// minimising `Σ |Q(u_i) − p_i|²`: a 2×2 normal-equation system in `(αl, αr)`. Falls back
/// to Wu/Barsky's heuristic (a third of the chord) when the system is singular or a handle
/// comes out non-positive.
fn generate_bezier(pts: &[Point], u: &[f64], t1: Vec2, t2: Vec2) -> CubicBez {
    let n = pts.len();
    let (p0, p3) = (pts[0], pts[n - 1]);
    let (mut c00, mut c01, mut c11, mut x0, mut x1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (p, &t) in pts.iter().zip(u) {
        let s = 1.0 - t;
        let (b0, b1, b2, b3) = (s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t);
        let a1 = t1 * b1;
        let a2 = t2 * b2;
        c00 += a1.dot(a1);
        c01 += a1.dot(a2);
        c11 += a2.dot(a2);
        let base = p0.to_vec2() * (b0 + b1) + p3.to_vec2() * (b2 + b3);
        let tmp = p.to_vec2() - base;
        x0 += a1.dot(tmp);
        x1 += a2.dot(tmp);
    }
    let det = c00 * c11 - c01 * c01;
    let (mut al, mut ar) = if det.abs() > 1e-12 * (c00 * c11).max(EPS) {
        ((x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det)
    } else {
        (0.0, 0.0)
    };
    let seg_len = (p3 - p0).hypot();
    let eps = 1e-6 * seg_len;
    // A control polygon much longer than the run itself means the solution loops out
    // between the samples (sparse, folded runs); the fallback then fails the error test
    // and the run is split instead.
    let run: f64 = pts.windows(2).map(|w| (w[1] - w[0]).hypot()).sum();
    let polygon = |al: f64, ar: f64| al + ((p3 + t2 * ar) - (p0 + t1 * al)).hypot() + ar;
    if !(al.is_finite() && ar.is_finite())
        || al < eps
        || ar < eps
        || polygon(al, ar) > MAX_POLYGON * run
    {
        al = seg_len / 3.0;
        ar = seg_len / 3.0;
    }
    CubicBez::new(p0, p0 + t1 * al, p3 + t2 * ar, p3)
}

/// Max squared distance between `pts[i]` and `bez(u[i])`, and its index.
fn max_error(pts: &[Point], bez: &CubicBez, u: &[f64]) -> (f64, usize) {
    let mut best = (0.0, pts.len() / 2);
    for i in 1..pts.len() - 1 {
        let d = (bez.eval(u[i]) - pts[i]).hypot2();
        if d > best.0 {
            best = (d, i);
        }
    }
    best
}

/// One Newton–Raphson step per sample on `f(u) = (Q(u) − p)·Q'(u)` (the condition for `Q(u)`
/// to be the closest point to `p`): `u ← u − f / f'`, with `f' = Q'·Q' + (Q − p)·Q''`.
fn reparameterize(pts: &[Point], u: &mut [f64], bez: &CubicBez) {
    let d1 = ParamCurveDeriv::deriv(bez);
    let d2 = ParamCurveDeriv::deriv(&d1);
    for (p, t) in pts.iter().zip(u.iter_mut()) {
        let diff = bez.eval(*t) - *p;
        let q1 = d1.eval(*t).to_vec2();
        let q2 = d2.eval(*t).to_vec2();
        let num = diff.dot(q1);
        let den = q1.dot(q1) + diff.dot(q2);
        if den.abs() > EPS {
            let nt = *t - num / den;
            if nt.is_finite() {
                *t = nt.clamp(0.0, 1.0);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use kurbo::PathEl;

    /// Max distance from each sample to the flattened path.
    pub(crate) fn max_dist_to_path(samples: &[Point], path: &BezPath) -> f64 {
        let mut poly: Vec<Vec<Point>> = Vec::new();
        kurbo::flatten(path.iter(), 1e-4, |el| match el {
            PathEl::MoveTo(p) => poly.push(vec![p]),
            PathEl::LineTo(p) => poly.last_mut().unwrap().push(p),
            PathEl::ClosePath => {
                let first = poly.last().unwrap()[0];
                poly.last_mut().unwrap().push(first);
            }
            _ => unreachable!(),
        });
        samples
            .iter()
            .map(|s| {
                poly.iter()
                    .flat_map(|pl| pl.windows(2).map(|w| dist_to_segment(*s, w[0], w[1])))
                    .fold(f64::MAX, f64::min)
            })
            .fold(0.0, f64::max)
    }

    fn wave(n: usize) -> Vec<Point> {
        (0..n)
            .map(|i| {
                let x = i as f64 / (n - 1) as f64 * 300.0;
                Point::new(x, 40.0 * (x / 35.0).sin() + 0.002 * x * x)
            })
            .collect()
    }

    fn assert_well_formed(p: &BezPath) {
        assert!(path_is_finite(p));
        assert_eq!(degenerate_segments(p, 1e-9), 0);
    }

    #[test]
    fn fit_stays_within_tolerance() {
        let pts = wave(200);
        for tol in [0.05, 0.5, 2.0] {
            let p = fit_points(&pts, tol);
            assert_well_formed(&p);
            let d = max_dist_to_path(&pts, &p);
            assert!(d <= tol + 1e-3, "tol={tol} d={d}");
            let n = p.segments().count();
            assert!(n < 60, "too many segments: {n}");
        }
    }

    #[test]
    fn fit_is_deterministic() {
        let pts = wave(120);
        assert_eq!(fit_points(&pts, 0.2), fit_points(&pts, 0.2));
        let c: Vec<Point> = (0..50)
            .map(|i| {
                let a = i as f64 / 50.0 * std::f64::consts::TAU;
                Point::new(30.0 * a.cos(), 20.0 * a.sin())
            })
            .collect();
        assert_eq!(fit_points_closed(&c, 0.1), fit_points_closed(&c, 0.1));
    }

    #[test]
    fn fit_edge_cases() {
        assert!(fit_points(&[], 0.1).elements().is_empty());
        assert!(
            fit_points(&[Point::new(1.0, 1.0)], 0.1)
                .elements()
                .is_empty()
        );
        let two = fit_points(&[Point::ZERO, Point::new(10.0, 0.0)], 0.1);
        assert_eq!(two.elements().len(), 2);
        // all duplicates
        let dup = [Point::new(2.0, 2.0); 10];
        assert!(fit_points(&dup, 0.1).elements().is_empty());
        // duplicates and NaN mixed in
        let mut pts = wave(60);
        pts.insert(10, pts[10]);
        pts.insert(20, Point::new(f64::NAN, 0.0));
        pts.insert(30, pts[30]);
        let p = fit_points(&pts, 0.1);
        assert_well_formed(&p);
        // collinear
        let line: Vec<Point> = (0..30)
            .map(|i| Point::new(i as f64, 2.0 * i as f64))
            .collect();
        let p = fit_points(&line, 0.01);
        assert_well_formed(&p);
        assert_eq!(p.segments().count(), 1);
        assert!(max_dist_to_path(&line, &p) < 0.01);
        // bad tolerance
        let p = fit_points(&wave(40), f64::NAN);
        assert_well_formed(&p);
        // back-and-forth within tolerance
        let p = fit_points(&[Point::ZERO, Point::new(0.05, 0.0), Point::ZERO], 0.1);
        assert_well_formed(&p);
        assert!(!p.elements().is_empty());
    }

    #[test]
    fn corners_are_preserved() {
        let mut pts: Vec<Point> = (0..=20).map(|i| Point::new(i as f64 * 5.0, 0.0)).collect();
        pts.extend((1..=20).map(|i| Point::new(100.0, i as f64 * 5.0)));
        let p = fit_points(&pts, 0.05);
        assert_well_formed(&p);
        assert!(max_dist_to_path(&pts, &p) < 0.06);
        // the corner point is an on-curve point of the path
        let corner = Point::new(100.0, 0.0);
        assert!(p.segments().any(|s| (s.end() - corner).hypot() < 1e-9));
    }

    #[test]
    fn closed_fit_within_tolerance_and_closed() {
        let c: Vec<Point> = (0..80)
            .map(|i| {
                let a = i as f64 / 80.0 * std::f64::consts::TAU;
                let r = 50.0 + 10.0 * (3.0 * a).sin();
                Point::new(r * a.cos(), r * a.sin())
            })
            .collect();
        let p = fit_points_closed(&c, 0.1);
        assert_well_formed(&p);
        assert!(matches!(p.elements().last(), Some(PathEl::ClosePath)));
        assert!(max_dist_to_path(&c, &p) <= 0.101);
        // seam is G1: incoming and outgoing directions at the start agree
        let comp = Composite::from_bezpath(&p).unwrap();
        let segs = comp.segments();
        let a = end_dir(segs.last().unwrap()).unwrap();
        let b = start_dir(&segs[0]).unwrap();
        assert!(a.dot(b) > 0.9999);
        // square: corners kept
        let sq: Vec<Point> = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ]
        .into_iter()
        .map(Point::from)
        .collect();
        let p = fit_points_closed(&sq, 0.01);
        assert_well_formed(&p);
        assert!(max_dist_to_path(&sq, &p) < 0.011);
        assert!(fit_points_closed(&[Point::ZERO], 0.1).elements().is_empty());
        assert!(path_is_finite(&fit_points_closed(
            &[Point::ZERO, Point::new(1.0, 0.0)],
            0.1
        )));
    }

    #[test]
    fn resample_and_simplify() {
        let pts = [Point::ZERO, Point::new(10.0, 0.0), Point::new(10.0, 30.0)];
        let r = resample_by_arclen(&pts, 5);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], pts[0]);
        assert_eq!(r[4], pts[2]);
        assert!((r[1] - Point::new(10.0, 0.0)).hypot() < 1e-9);
        for w in r.windows(2) {
            assert!(((w[1] - w[0]).hypot() - 10.0).abs() < 1e-9);
        }
        assert!(resample_by_arclen(&[], 4).is_empty());
        assert_eq!(
            resample_by_arclen(&[Point::ZERO; 3], 4),
            vec![Point::ZERO; 4]
        );
        assert_eq!(resample_by_arclen(&pts, 1), vec![pts[0]]);

        let line: Vec<Point> = (0..20).map(|i| Point::new(i as f64, 0.0)).collect();
        assert_eq!(simplify(&line, 0.01), vec![line[0], line[19]]);
        let w = wave(300);
        let s = simplify(&w, 0.5);
        assert!(s.len() < 100 && s.len() > 4);
        assert_eq!(s[0], w[0]);
        assert_eq!(*s.last().unwrap(), w[299]);
        // every original point is within tolerance of the simplified polyline
        for p in &w {
            let d = s
                .windows(2)
                .map(|q| dist_to_segment(*p, q[0], q[1]))
                .fold(f64::MAX, f64::min);
            assert!(d <= 0.5 + 1e-9);
        }
        assert_eq!(simplify(&w, f64::NAN).len(), simplify(&w, 0.0).len());
    }

    #[test]
    fn fit_curve_is_close_g1_and_forward() {
        use crate::geom::FnCurve;
        // a tightening spiral and a cubic with a zero-length start handle
        let spiral = FnCurve(|t: f64| {
            let a = t * 3.0 * std::f64::consts::TAU;
            let r = 80.0 * (-0.12 * a).exp();
            Point::new(r * a.cos(), r * a.sin())
        });
        let cubic = CubicBez::new((0.0, 0.0), (0.0, 0.0), (100.0, 80.0), (200.0, 0.0));
        for tol in [0.01, 0.2] {
            for path in [fit_curve(&spiral, tol), fit_curve(&cubic, tol)] {
                assert_well_formed(&path);
                let c = Composite::from_bezpath(&path).unwrap();
                assert!(max_kink(&c) < 1e-9, "kink {}", max_kink(&c));
                for s in c.segments() {
                    assert!((s.p1 - s.p0).hypot() > 0.0 && (s.p3 - s.p2).hypot() > 0.0);
                }
            }
            let path = fit_curve(&spiral, tol);
            let dense: Vec<Point> = spiral
                .sample_by_arclen(3000)
                .into_iter()
                .map(|x| x.1)
                .collect();
            assert!(max_dist_to_path(&dense, &path) < tol * 1.5);
            let c = Composite::from_bezpath(&path).unwrap();
            assert!(c.tangent(0.0).dot(spiral.tangent(0.0)) > 1.0 - 1e-9);
            assert!(c.tangent(1.0).dot(spiral.tangent(1.0)) > 1.0 - 1e-9);
        }
        let dot = FnCurve(|_t: f64| Point::new(1.0, 2.0));
        assert!(fit_curve(&dot, 0.1).elements().is_empty());
        assert_eq!(fit_curve(&spiral, 0.1), fit_curve(&spiral, 0.1));
    }

    #[test]
    fn sparse_jagged_samples_never_fling_handles() {
        // Few samples of rippled, folded edges (a fringed petal): least squares alone can
        // answer with huge handles that pass the samples but loop far outside them.
        for k in 0..200 {
            let f = 3.0 + (k % 17) as f64 * 0.7;
            let n = 8 + k % 40;
            let pts: Vec<Point> = (0..n)
                .map(|i| {
                    let t = i as f64 / (n - 1) as f64;
                    let r = 10.0 + 4.0 * (f * std::f64::consts::TAU * t + k as f64).sin().abs();
                    let a = 0.4 * k as f64 + 2.5 * t;
                    Point::new(r * a.cos() + 30.0 * t, r * a.sin())
                })
                .collect();
            let tol = 0.02 + 0.01 * (k % 5) as f64;
            let p = fit_points(&pts, tol);
            let bb = pts
                .iter()
                .fold(kurbo::Rect::from_points(pts[0], pts[0]), |b, q| {
                    b.union_pt(*q)
                })
                .inflate(2.0, 2.0);
            let fb = kurbo::Shape::bounding_box(&p);
            assert!(
                bb.contains(fb.origin()) && bb.contains(Point::new(fb.x1, fb.y1)),
                "case {k}: fit {fb:?} leaves samples {bb:?}"
            );
        }
    }

    #[test]
    fn smooth_s_curves_need_few_segments() {
        use crate::geom::FnCurve;
        let cubic = CubicBez::new((0.0, 0.0), (150.0, -120.0), (50.0, 220.0), (200.0, 100.0));
        let sine =
            FnCurve(|t: f64| Point::new(300.0 * t, 60.0 * (std::f64::consts::TAU * t).sin()));
        let dense = |c: &dyn Curve| -> Vec<Point> {
            (0..=2000).map(|i| c.point(i as f64 / 2000.0)).collect()
        };
        for (c, max) in [(&cubic as &dyn Curve, 2), (&sine, 8)] {
            let path = fit_curve(c, 0.1);
            assert!(
                path.segments().count() <= max,
                "{} segments",
                path.segments().count()
            );
            assert!(max_dist_to_path(&dense(c), &path) < 0.15);
        }
    }

    #[test]
    fn tight_coils_stay_within_tolerance_and_g1() {
        use crate::geom::FnCurve;
        use crate::geom::spiral::ArchimedeanSpiral;
        // 50 coils one unit apart: the inner coils are shorter than the base sample spacing,
        // and lopsided worst-point splits used to exhaust the recursion guard.
        let a = ArchimedeanSpiral::from_center(Point::ZERO, 0.0, 0.0, 50.0, 1.0, false);
        let path = fit_curve(&a, 0.01);
        assert_well_formed(&path);
        let dense: Vec<Point> = a.sample_by_arclen(2000).into_iter().map(|x| x.1).collect();
        assert!(max_dist_to_path(&dense, &path) < 0.0105);
        assert!(max_kink(&Composite::from_bezpath(&path).unwrap()) < 1e-9);
        // fit_points on the same samples (slack: flattening error of max_dist_to_path)
        let p = fit_points(&dense, 0.01);
        assert_well_formed(&p);
        assert!(max_dist_to_path(&dense, &p) <= 0.01 + 1e-4);
        // a genuine cusp is rounded within tolerance and stays G1
        let cusp = FnCurve(|t: f64| Point::new(100.0 * t, 100.0 * (t - 0.5).abs()));
        let path = fit_curve(&cusp, 0.05);
        assert_well_formed(&path);
        let dense: Vec<Point> = (0..=2000).map(|i| cusp.point(i as f64 / 2000.0)).collect();
        assert!(max_dist_to_path(&dense, &path) < 0.05);
        assert!(max_kink(&Composite::from_bezpath(&path).unwrap()) < 1e-9);
    }

    #[test]
    fn join_g1_aligns_tangents() {
        let mut a = BezPath::new();
        a.move_to((0.0, 0.0));
        a.curve_to((10.0, 0.0), (20.0, 1.0), (30.0, 0.0));
        let mut b = BezPath::new();
        // starts slightly off and with a slightly rotated tangent
        b.move_to((30.0, 0.01));
        b.curve_to((40.0, -0.5), (50.0, -5.0), (60.0, 0.0));
        let mut empty = BezPath::new();
        empty.move_to((5.0, 5.0));
        empty.line_to((5.0, 5.0));
        let c = join_g1(&[a.clone(), empty, b]).unwrap();
        assert_eq!(c.segments().len(), 2);
        assert!(max_kink(&c) < 1e-9);
        assert_eq!(c.segments()[0].p3, c.segments()[1].p0);
        assert!(join_g1(&[]).is_none());
        assert_eq!(join_g1(&[a]).unwrap().segments().len(), 1);
        let s = stub(Point::new(f64::NAN, 0.0), Vec2::ZERO);
        assert_well_formed(&s.to_bezpath_exact());
    }
}
