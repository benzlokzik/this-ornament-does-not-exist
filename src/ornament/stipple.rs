//! Stippling: dots placed by variable-radius Poisson-disk dart throwing.
//!
//! A candidate is drawn uniformly in *canvas* area and accepted only if no earlier dot
//! lies within `d(p) = 3r / √density(p)`. Dot density therefore follows `density`
//! (∝ `1/d²`), while the minimum-distance rule keeps the pattern even, without the
//! clumps and holes of plain random sampling.

use super::{fit_accuracy, half_width_at, max_half_width};
use crate::geom::{ArcLength, EPS, Surface, lerp};
use crate::rng::{Draw, SeedTree};
use kurbo::{BezPath, Circle, Point, Rect, Shape};

/// Below this density no dot is placed (and `d` stays bounded).
const MIN_DENSITY: f64 = 0.02;
/// Upper bound on dart throws per call.
const MAX_CANDIDATES: usize = 60_000;
/// Candidates thrown per area of one densest-packing cell (`d0²`).
const CANDIDATES_PER_CELL: f64 = 12.0;

/// Dots over `surface`, with `density(u, v)` in `[0, 1]` (values outside are clamped,
/// NaN counts as `0`) and dot radius `dot_radius` in canvas units.
///
/// Candidates are drawn uniformly in canvas area: the surface is split into a grid of
/// small `(u, v)` cells weighted by their canvas area (the Jacobian of the map), so dots
/// are not crowded where the shape narrows. Dots are kept a radius clear of the edges,
/// so each circle lies inside the outline. A non-finite or non-positive radius, or a
/// zero-width surface, yields no dots. Randomness comes from `seed.child("stipple")`.
pub fn stipple<S: Surface + ?Sized>(
    surface: &S,
    density: &dyn Fn(f64, f64) -> f64,
    dot_radius: f64,
    seed: &SeedTree,
    accuracy: f64,
) -> Vec<BezPath> {
    const NU: usize = 96;
    const NV: usize = 24;
    let r = dot_radius;
    if !(r.is_finite() && r > 0.0) || max_half_width(surface) < EPS {
        return Vec::new();
    }
    // Grid corners and per-cell canvas areas (shoelace on the four corners).
    let corner = |i: usize, j: usize| {
        surface.point(i as f64 / NU as f64, lerp(-1.0, 1.0, j as f64 / NV as f64))
    };
    let pts: Vec<Point> = (0..=NU)
        .flat_map(|i| (0..=NV).map(move |j| (i, j)))
        .map(|(i, j)| corner(i, j))
        .collect();
    let at = |i: usize, j: usize| pts[i * (NV + 1) + j];
    let mut cdf = Vec::with_capacity(NU * NV);
    let mut total = 0.0;
    for i in 0..NU {
        for j in 0..NV {
            let q = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
            let a = (0..4)
                .map(|k| q[k].to_vec2().cross(q[(k + 1) % 4].to_vec2()))
                .sum::<f64>()
                .abs()
                * 0.5;
            total += if a.is_finite() { a } else { 0.0 };
            cdf.push(total);
        }
    }
    let Some(bounds) = finite_bounds(&pts) else {
        return Vec::new();
    };
    if total < EPS {
        return Vec::new();
    }
    // Arc length along the centre line, to keep dots clear of blunt ends.
    let centre = surface.iso_v(0.0);
    let al = ArcLength::new(&*centre, 128);
    let len = al.total();

    let d0 = 3.0 * r;
    let mut grid = DiskGrid::new(bounds, d0);
    let mut rng = seed.child("stipple").rng();
    let n = ((CANDIDATES_PER_CELL * total / (d0 * d0)) as usize).min(MAX_CANDIDATES);
    for _ in 0..n {
        let (pick, du, dv) = (
            rng.uniform(0.0, total),
            rng.uniform(0.0, 1.0),
            rng.uniform(0.0, 1.0),
        );
        let cell = cdf.partition_point(|&c| c < pick).min(cdf.len() - 1);
        let u = ((cell / NV) as f64 + du) / NU as f64;
        let v = lerp(-1.0, 1.0, ((cell % NV) as f64 + dv) / NV as f64);
        let dens = clamp_density(density(u, v));
        if dens < MIN_DENSITY {
            continue;
        }
        // Keep the whole dot inside: clear of both edges and of both ends.
        let x = al.s_at(u);
        if half_width_at(surface, u) * (1.0 - v.abs()) < 1.2 * r || x.min(len - x) < 1.2 * r {
            continue;
        }
        let p = surface.point(u, v);
        if p.is_finite() {
            grid.try_insert(p, d0 / dens.sqrt());
        }
    }
    grid.dots(r, accuracy)
}

/// Dots inside an arbitrary closed `outline` (non-zero winding), with density given per
/// canvas point (clamped to `[0, 1]`).
///
/// Candidates are drawn uniformly over the outline's bounding box; a dot is kept only if
/// its centre and its four extreme points are inside, so circles do not cross the
/// outline. Invalid radius or an empty/non-finite outline yields no dots. Randomness
/// comes from `seed.child("stipple")`.
pub fn stipple_path(
    outline: &BezPath,
    density_at_point: &dyn Fn(Point) -> f64,
    dot_radius: f64,
    seed: &SeedTree,
) -> Vec<BezPath> {
    let r = dot_radius;
    if !(r.is_finite() && r > 0.0) || outline.elements().is_empty() {
        return Vec::new();
    }
    let bb = outline.bounding_box();
    if !(bb.is_finite() && bb.area() > EPS) {
        return Vec::new();
    }
    let d0 = 3.0 * r;
    let mut grid = DiskGrid::new(bb, d0);
    let mut rng = seed.child("stipple").rng();
    let n = ((CANDIDATES_PER_CELL * bb.area() / (d0 * d0)) as usize).min(MAX_CANDIDATES);
    let inside = |p: Point| outline.winding(p) != 0;
    for _ in 0..n {
        let p = Point::new(rng.uniform(bb.x0, bb.x1), rng.uniform(bb.y0, bb.y1));
        let dens = clamp_density(density_at_point(p));
        if dens < MIN_DENSITY {
            continue;
        }
        let ring = [(0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)];
        if ring.iter().all(|&(dx, dy)| inside(p + (dx, dy))) {
            grid.try_insert(p, d0 / dens.sqrt());
        }
    }
    grid.dots(r, (r * 0.02).max(1e-3))
}

fn clamp_density(d: f64) -> f64 {
    if d.is_finite() {
        d.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn finite_bounds(pts: &[Point]) -> Option<Rect> {
    let mut it = pts.iter().filter(|p| p.is_finite());
    let first = *it.next()?;
    let r = it.fold(Rect::from_points(first, first), |r, p| r.union_pt(*p));
    (r.area() > 0.0 || r.width() > 0.0 || r.height() > 0.0).then_some(r)
}

/// Uniform bucket grid for minimum-distance queries (deterministic: plain vectors,
/// no hashing).
struct DiskGrid {
    origin: Point,
    cell: f64,
    nx: usize,
    ny: usize,
    buckets: Vec<Vec<u32>>,
    pts: Vec<Point>,
}

impl DiskGrid {
    /// Cells are at least `cell` wide, enlarged if needed to keep the grid ≤ 512².
    fn new(bounds: Rect, cell: f64) -> DiskGrid {
        let cell = cell
            .max(bounds.width() / 512.0)
            .max(bounds.height() / 512.0);
        let nx = ((bounds.width() / cell).ceil() as usize).max(1);
        let ny = ((bounds.height() / cell).ceil() as usize).max(1);
        DiskGrid {
            origin: bounds.origin(),
            cell,
            nx,
            ny,
            buckets: vec![Vec::new(); nx * ny],
            pts: Vec::new(),
        }
    }

    fn index(&self, p: Point) -> (usize, usize) {
        let ix = ((p.x - self.origin.x) / self.cell).floor().max(0.0) as usize;
        let iy = ((p.y - self.origin.y) / self.cell).floor().max(0.0) as usize;
        (ix.min(self.nx - 1), iy.min(self.ny - 1))
    }

    /// Insert `p` unless an existing point lies closer than `dist`.
    fn try_insert(&mut self, p: Point, dist: f64) -> bool {
        let (ix, iy) = self.index(p);
        let reach = (dist / self.cell).ceil() as usize;
        let (x0, x1) = (ix.saturating_sub(reach), (ix + reach).min(self.nx - 1));
        let (y0, y1) = (iy.saturating_sub(reach), (iy + reach).min(self.ny - 1));
        let d2 = dist * dist;
        for y in y0..=y1 {
            for x in x0..=x1 {
                for &k in &self.buckets[y * self.nx + x] {
                    if (self.pts[k as usize] - p).hypot2() < d2 {
                        return false;
                    }
                }
            }
        }
        self.buckets[iy * self.nx + ix].push(self.pts.len() as u32);
        self.pts.push(p);
        true
    }

    /// One circle path per accepted point, in acceptance order.
    fn dots(&self, r: f64, accuracy: f64) -> Vec<BezPath> {
        // Relative floor: an absolute tolerance on a huge circle asks kurbo for ~10^20 arcs.
        let acc = fit_accuracy(accuracy).min(r * 0.25).max(r * 1e-4);
        self.pts
            .iter()
            .map(|&p| Circle::new(p, r).to_path(acc))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::ornament::test_shapes::{Huge, Leaf, Needle, inside_fraction, path_samples, petal};

    fn centres(dots: &[BezPath]) -> Vec<Point> {
        dots.iter().map(|d| d.bounding_box().center()).collect()
    }

    fn assert_clean(dots: &[BezPath]) {
        for d in dots {
            assert!(path_is_finite(d));
            assert_eq!(degenerate_segments(d, 1e-9), 0);
        }
    }

    #[test]
    fn dots_are_inside_spaced_and_reproducible() {
        let seed = SeedTree::new(11);
        for s in [&petal() as &dyn Surface, &Leaf] {
            let dots = stipple(s, &|_, _| 1.0, 0.8, &seed, 0.05);
            assert!(dots.len() > 100, "{}", dots.len());
            assert_clean(&dots);
            let outline = s.outline(0.02);
            let f = inside_fraction(&outline, &path_samples(&dots));
            assert!(f >= 0.99, "inside {f}");
            let c = centres(&dots);
            for (i, a) in c.iter().enumerate() {
                for b in &c[i + 1..] {
                    assert!((*a - *b).hypot() >= 2.4 - 1e-9);
                }
            }
            assert_eq!(dots, stipple(s, &|_, _| 1.0, 0.8, &seed, 0.05));
        }
    }

    #[test]
    fn density_controls_dot_count_uniformly_in_canvas() {
        let s = petal();
        let seed = SeedTree::new(2);
        // shade the right half (v > 0) only
        let dots = stipple(&s, &|_, v| if v > 0.0 { 1.0 } else { 0.0 }, 0.6, &seed, 0.1);
        let c = centres(&dots);
        assert!(c.iter().all(|p| p.x > -1e-6));
        // canvas uniformity: dots per area near the wide middle vs the narrow base
        let in_band = |y0: f64, y1: f64| c.iter().filter(|p| p.y >= y0 && p.y < y1).count();
        // right-half canvas area of a band, by counting grid points inside the outline
        let outline = s.outline(0.05);
        let area = |y0: f64, y1: f64| {
            let h = 0.5;
            let mut n = 0;
            let mut y = y0 + 0.5 * h;
            while y < y1 {
                let mut x = 0.5 * h;
                while x < 60.0 {
                    n += usize::from(outline.winding(Point::new(x, y)) != 0);
                    x += h;
                }
                y += h;
            }
            n as f64 * h * h
        };
        let mid = in_band(60.0, 100.0) as f64 / area(60.0, 100.0);
        let low = in_band(15.0, 40.0) as f64 / area(15.0, 40.0);
        assert!((mid / low - 1.0).abs() < 0.35, "mid {mid} low {low}");
        let sparse = stipple(&s, &|_, _| 0.2, 0.6, &seed, 0.1);
        let dense = stipple(&s, &|_, _| 1.0, 0.6, &seed, 0.1);
        assert!(sparse.len() * 3 < dense.len());
    }

    #[test]
    fn stipple_path_respects_outline() {
        let s = petal();
        let outline = s.outline(0.05);
        let seed = SeedTree::new(3);
        let dots = stipple_path(&outline, &|p| (p.y / 160.0).clamp(0.0, 1.0), 0.7, &seed);
        assert!(dots.len() > 50);
        assert_clean(&dots);
        assert!(inside_fraction(&outline, &path_samples(&dots)) >= 0.99);
        assert_eq!(
            dots,
            stipple_path(&outline, &|p| (p.y / 160.0).clamp(0.0, 1.0), 0.7, &seed)
        );
        let c = centres(&dots);
        let top = c.iter().filter(|p| p.y > 80.0).count();
        // density rises linearly with y: the upper half holds clearly more dots
        assert!(top as f64 > 1.6 * (c.len() - top) as f64);
    }

    #[test]
    fn degenerate_inputs_yield_nothing() {
        let seed = SeedTree::new(1);
        let s = petal();
        assert!(stipple(&Needle, &|_, _| 1.0, 1.0, &seed, 0.1).is_empty());
        for r in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(stipple(&s, &|_, _| 1.0, r, &seed, 0.1).is_empty());
            assert!(stipple_path(&s.outline(0.1), &|_| 1.0, r, &seed).is_empty());
        }
        assert!(stipple(&s, &|_, _| f64::NAN, 1.0, &seed, 0.1).is_empty());
        assert!(stipple_path(&BezPath::new(), &|_| 1.0, 1.0, &seed).is_empty());
        // a huge radius fits no dot; a tiny one is bounded by the candidate cap
        assert!(stipple(&s, &|_, _| 1.0, 1e6, &seed, 0.1).is_empty());
        // a huge dot with an absolute accuracy used to ask kurbo for ~10^20 arcs
        let big = stipple(&Huge, &|_, _| 1.0, 1e147, &seed, 0.05);
        assert!(!big.is_empty());
        assert_clean(&big);
        let tiny = stipple(&s, &|_, _| 1.0, 1e-4, &seed, f64::NAN);
        assert!(tiny.len() <= MAX_CANDIDATES);
        assert_clean(&tiny);
    }
}
