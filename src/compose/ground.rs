//! Background fills of the botanica preset: a dense ornament ground of overlapping scrolls,
//! leaves and flowers, and a field of blooms. Both leave no bare paper behind the
//! foreground.
//!
//! Placement rules (warm-left frame, see [`Canvas`]):
//! - **Keep-outs.** Only the focal flower centres and the moonlit window of `occupied`
//!   are respected (its discs of at least 7.5 % of the canvas unit, and its path regions);
//!   everything else may be overlapped, since this layer is painted below it. No element
//!   is centred in a flower's core; a dark bed fills it instead, the depth its petals open
//!   onto. The top of the cool region, where the landscape's sky and moon sit, is *calm*:
//!   large scrolls and the gap filling avoid it, and a few accent scrolls cover about a
//!   fifth of it, so the landscape painted below stays readable.
//! - **Coverage.** A coarse grid records which cells the outlines placed so far cover.
//!   Gaps are found with a distance transform of the uncovered cells: the largest empty
//!   disc gets the next element, sized to it, so the surface closes up from large shapes
//!   to small ones.
//! - **Dense ground.** Large acanthus scrolls (25–45 % of the canvas height) go first,
//!   anchored on evenly spaced streamlines of the flow field ([`guide_curves`]), tangent
//!   to them and curling the way the streamline bends; then medium scrolls and curls fill
//!   the gaps, oriented by the field; then leaves (ovate, cordate, lanceolate, acanthus)
//!   and a few flat star flowers (daisy, aster, cosmos, primrose) are sprinkled where the
//!   large scrolls left gaps, and the last holes get small curls and leaves. Everything
//!   placed first is farther back: darker, lower in contrast and drawn at a coarser
//!   accuracy without hatching. Elements are sized to reach at most 6 % of the height past
//!   the canvas edges.
//! - **Bloom field.** A sparse backdrop of dark scrolls, then 25–60 flowers of every
//!   species, 4–14 % of the canvas height across, overlapping, the larger ones in front,
//!   with leaves and curls filling the gaps between them (behind the flowers).
//! - **Budget.** At most 14,000 paths at fill 1 (scaled with fill); background flowers
//!   keep only their main veins.
//! - **Colour.** Tones come from the warm ramp on the warm side and the cool ramp on the
//!   cool side, blended across the divider by [`Canvas::warmth`]; each element draws its
//!   own position on the ramp, and flowers rotate their hue by up to ±25°.

use super::field::{Field, Mask, Region, along_curve, guide_curves, place_points};
use super::layout::{Canvas, Occupancy, flower_style, leaf_style, unit_or};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::{Curve, Surface, lerp, smoothstep};
use crate::motif::flower::{self, FlowerParams, FlowerStyle};
use crate::motif::foliage::{LeafStyle, leaf_motif_with_detail};
use crate::motif::scroll::{self, ScrollParams, ScrollStyle};
use crate::ornament::Detail;
use crate::palette::Palette;
use crate::rng::{Draw, SeedTree};
use crate::scene::{Color, Node, Style};
use kurbo::{Affine, BezPath, Circle, Point, Rect, Shape, Vec2};
use rayon::prelude::*;
use std::f64::consts::TAU;

/// Path budget of a background fill at fill 1.
const PATH_BUDGET: f64 = 14_000.0;
/// Even the frontmost ground scrolls and leaves stay this far back (as a depth), so the
/// foreground keeps the strongest contrast; on a dark paper, where light tones advance,
/// further back.
const GROUND_DEPTH: f64 = 0.85;
const GROUND_DEPTH_DARK: f64 = 0.55;
/// How far elements may reach past the canvas edges, as a fraction of its height.
const BLEED: f64 = 0.06;
/// Leaves of the ground with their aspect: broad, simple silhouettes with a few lobed ones.
const GAP_LEAVES: [(LeafShape, f64); 5] = [
    (LeafShape::Ovate, 0.55),
    (LeafShape::Ovate, 0.45),
    (LeafShape::Cordate, 0.6),
    (LeafShape::Lanceolate, 0.3),
    (LeafShape::Acanthus { lobes: 4 }, 0.55),
];
/// Species of the flat star flowers sprinkled over the dense ground.
const STAR_FLOWERS: [&str; 4] = ["daisy", "aster", "cosmos", "primrose"];

/// Overlapping scrolls, leaves and flowers covering all free space so no bare paper shows.
#[allow(clippy::too_many_arguments)]
pub fn dense_ground(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    field: &dyn Field,
    occupied: &Occupancy,
    fill: f64,
    accuracy: f64,
) -> Node {
    let mut g = Ground::new(seed, canvas, palette, field, occupied, fill, accuracy);
    g.dense();
    g.finish()
}

/// A field of flowers of varied species, sizes and hues over the free space.
#[allow(clippy::too_many_arguments)]
pub fn bloom_field(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    field: &dyn Field,
    occupied: &Occupancy,
    fill: f64,
    accuracy: f64,
) -> Node {
    let mut g = Ground::new(seed, canvas, palette, field, occupied, fill, accuracy);
    g.blooms();
    g.finish()
}

// ---------------------------------------------------------------------------------------
// Coverage grid
// ---------------------------------------------------------------------------------------

/// Which cells of the canvas the outlines placed so far cover.
struct Cover {
    origin: Point,
    cell: f64,
    nx: usize,
    ny: usize,
    covered: Vec<bool>,
    /// Cells that need no filling (focal flower cores, the window, the calm sky).
    skip: Vec<bool>,
}

impl Cover {
    fn new(canvas: &Canvas, cell: f64) -> Cover {
        let nx = ((canvas.width / cell).ceil() as usize).max(1);
        let ny = ((canvas.height / cell).ceil() as usize).max(1);
        Cover {
            origin: Point::ORIGIN,
            cell,
            nx,
            ny,
            covered: vec![false; nx * ny],
            skip: vec![false; nx * ny],
        }
    }

    fn center(&self, i: usize) -> Point {
        let (x, y) = (i % self.nx, i / self.nx);
        self.origin + Vec2::new((x as f64 + 0.5) * self.cell, (y as f64 + 0.5) * self.cell)
    }

    /// Cell index range `(x0, x1, y0, y1)` (exclusive ends) overlapping `r`.
    fn range(&self, r: Rect) -> (usize, usize, usize, usize) {
        let ix = |x: f64, n: usize| ((x / self.cell).floor().max(0.0) as usize).min(n);
        (
            ix(r.x0 - self.origin.x, self.nx),
            ix(r.x1 - self.origin.x + self.cell, self.nx),
            ix(r.y0 - self.origin.y, self.ny),
            ix(r.y1 - self.origin.y + self.cell, self.ny),
        )
    }

    /// Mark the cells whose centres lie inside `path` (non-zero winding).
    fn stamp(&mut self, path: &BezPath) {
        let (x0, x1, y0, y1) = self.range(path.bounding_box());
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * self.nx + x;
                if !self.covered[i] && path.contains(self.center(i)) {
                    self.covered[i] = true;
                }
            }
        }
    }

    /// Mark the cells whose centres lie within `r` of `c`.
    fn stamp_disc(&mut self, c: Point, r: f64) {
        let (x0, x1, y0, y1) = self.range(Rect::new(c.x - r, c.y - r, c.x + r, c.y + r));
        for y in y0..y1 {
            for x in x0..x1 {
                let i = y * self.nx + x;
                if (self.center(i) - c).hypot() <= r {
                    self.covered[i] = true;
                }
            }
        }
    }

    /// Fraction of the cells that still want filling.
    fn open_fraction(&self) -> f64 {
        let open = (0..self.covered.len())
            .filter(|&i| !self.covered[i] && !self.skip[i])
            .count();
        open as f64 / self.covered.len() as f64
    }

    /// Distance (canvas units) from every open cell to the nearest covered or skipped cell
    /// (two-pass chamfer transform; 0 for covered and skipped cells).
    fn distances(&self) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let big = (nx + ny) as f64 * 2.0;
        let mut d: Vec<f64> = (0..nx * ny)
            .map(|i| {
                if self.covered[i] || self.skip[i] {
                    0.0
                } else {
                    big
                }
            })
            .collect();
        let s2 = std::f64::consts::SQRT_2;
        for y in 0..ny {
            for x in 0..nx {
                let i = y * nx + x;
                let mut v = d[i];
                if x > 0 {
                    v = v.min(d[i - 1] + 1.0);
                }
                if y > 0 {
                    v = v.min(d[i - nx] + 1.0);
                    if x > 0 {
                        v = v.min(d[i - nx - 1] + s2);
                    }
                    if x + 1 < nx {
                        v = v.min(d[i - nx + 1] + s2);
                    }
                }
                d[i] = v;
            }
        }
        for y in (0..ny).rev() {
            for x in (0..nx).rev() {
                let i = y * nx + x;
                let mut v = d[i];
                if x + 1 < nx {
                    v = v.min(d[i + 1] + 1.0);
                }
                if y + 1 < ny {
                    v = v.min(d[i + nx] + 1.0);
                    if x + 1 < nx {
                        v = v.min(d[i + nx + 1] + s2);
                    }
                    if x > 0 {
                        v = v.min(d[i + nx - 1] + s2);
                    }
                }
                d[i] = v;
            }
        }
        d.into_iter().map(|v| v * self.cell).collect()
    }

    /// Up to `max` gaps `(centre, radius)`, largest first, at least `min_r` in radius and
    /// apart from each other; `jitter(i)` in `[0, 1)` breaks ties between cells.
    fn gaps(&self, max: usize, min_r: f64, jitter: impl Fn(usize) -> f64) -> Vec<(Point, f64)> {
        let d = self.distances();
        let mut cells: Vec<(usize, f64)> = d
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= min_r)
            .map(|(i, v)| (i, v + 0.5 * self.cell * jitter(i)))
            .collect();
        cells.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out: Vec<(Point, f64)> = Vec::new();
        for (i, _) in cells {
            if out.len() >= max {
                break;
            }
            let (c, r) = (self.center(i), d[i]);
            if out.iter().all(|(q, s)| (c - *q).hypot() > 1.2 * (r + s)) {
                out.push((c, r));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------
// Elements
// ---------------------------------------------------------------------------------------

/// What an element is.
#[derive(Clone)]
enum Kind {
    /// A scroll whose widest place lands on `at`, its midrib there along `dir`.
    Scroll(ScrollParams),
    /// A leaf from its base at `at` towards `dir`.
    Leaf {
        shape: LeafShape,
        length: f64,
        aspect: f64,
    },
    /// A flower of outer radius `radius` centred at `at`.
    Flower {
        species: usize,
        radius: f64,
        /// Hue rotation in degrees.
        hue: f64,
    },
}

/// One element to build: where, what, how far back (`depth` 0 = farthest) and its tone.
#[derive(Clone)]
struct Spec {
    at: Point,
    dir: Vec2,
    kind: Kind,
    depth: f64,
    tone: f64,
    seed: SeedTree,
}

/// A built element: its node, an outline for the coverage grid and its centre.
struct Placed {
    node: Node,
    outline: Option<BezPath>,
    disc: Option<(Point, f64)>,
    center: Point,
    /// The species of a flower.
    species: Option<usize>,
}

/// The state of a background fill being composed.
struct Ground<'a> {
    seed: SeedTree,
    canvas: &'a Canvas,
    palette: &'a Palette,
    field: &'a dyn Field,
    fill: f64,
    acc: f64,
    /// Focal flower discs and other keep-outs for element centres.
    keep: Vec<Region>,
    cover: Cover,
    budget: usize,
    spent: usize,
    nodes: Vec<Node>,
    /// Centre of everything placed and the species of the flowers (for the tests).
    placed: Vec<(Point, Option<usize>)>,
    species: Vec<(String, FlowerParams)>,
}

impl<'a> Ground<'a> {
    fn new(
        seed: &SeedTree,
        canvas: &'a Canvas,
        palette: &'a Palette,
        field: &'a dyn Field,
        occupied: &Occupancy,
        fill: f64,
        accuracy: f64,
    ) -> Ground<'a> {
        let u = canvas.unit();
        let fill = if fill.is_finite() {
            fill.clamp(0.1, 4.0)
        } else {
            1.0
        };
        let acc = if accuracy.is_finite() && accuracy > 0.0 {
            accuracy
        } else {
            0.05
        };
        // Focal flowers are the large discs of the occupancy map, the window its path or
        // disc; ribbons, vines, small flowers and plants mark smaller discs and boxes.
        let keep: Vec<Region> = occupied
            .regions()
            .iter()
            .filter(|r| match r {
                Region::Circle { radius, .. } => *radius >= 0.075 * u,
                Region::Path(_) => true,
                Region::Rect(_) => false,
            })
            .map(|r| match r {
                // Only the core of a flower is kept clear; its petals hide the rest.
                Region::Circle { center, radius } => Region::Circle {
                    center: *center,
                    radius: 0.7 * radius,
                },
                other => other.clone(),
            })
            .collect();
        let mut cover = Cover::new(canvas, u / 60.0);
        for i in 0..cover.skip.len() {
            let p = cover.center(i);
            // The window is painted over this layer; flower cores get a dark bed.
            cover.skip[i] = keep
                .iter()
                .any(|r| matches!(r, Region::Path(_)) && r.contains(p));
        }
        Ground {
            seed: *seed,
            canvas,
            palette,
            field,
            fill,
            acc,
            keep,
            cover,
            budget: (PATH_BUDGET * fill) as usize,
            spent: 0,
            nodes: Vec::new(),
            placed: Vec::new(),
            species: flower::variants(&seed.child("species")),
        }
    }

    /// How calm `p` is: 1 in the upper half of the cool region, where the open landscape's
    /// sky, moon, clouds and far hills sit, fading out towards mid-height; 0 elsewhere.
    fn calm(&self, p: Point) -> f64 {
        let h = self.canvas.height;
        (1.0 - self.canvas.warmth(p)) * (1.0 - smoothstep((p.y - 0.44 * h) / (0.14 * h)))
    }

    /// Placement density at `p`: 0 in a keep-out, a fifth in the calm sky, 1 elsewhere.
    fn density(&self, p: Point) -> f64 {
        if self.keep.iter().any(|r| r.contains(p)) {
            return 0.0;
        }
        1.0 - 0.8 * self.calm(p)
    }

    /// A dark bed under each focal flower core: the depth its petals open onto, so the
    /// core needs no ornament of its own.
    fn beds(&mut self) {
        let mut bed = Node::empty();
        for r in &self.keep {
            if let Region::Circle { center, radius } = r {
                let tone = self.deep(self.canvas.warmth(*center));
                bed.push(Node::path(
                    Circle::new(*center, *radius).to_path(self.acc),
                    Style::fill(tone),
                ));
                self.cover.stamp_disc(*center, *radius);
            }
        }
        self.spent += bed.path_count();
        self.nodes.push(bed);
    }

    /// The deep tone elements recede into at warmth `w`.
    fn deep(&self, w: f64) -> Color {
        self.palette
            .warm_at(0.0)
            .mix(self.palette.cool_at(0.0), 1.0 - w)
            .mix(self.palette.ink, 0.15)
    }

    /// Distance from `p` to the edge of the canvas grown by its bleed.
    fn room(&self, p: Point) -> f64 {
        let r = self.canvas.rect().inset(BLEED * self.canvas.height);
        (p.x - r.x0)
            .min(r.x1 - p.x)
            .min(p.y - r.y0)
            .min(r.y1 - p.y)
            .max(0.0)
    }

    /// Unit flow direction at `p` (a fallback for a vanishing field).
    fn flow(&self, p: Point) -> Vec2 {
        unit_or(self.field.at(p), Vec2::new(0.0, -1.0))
    }

    /// `true` if the flow turns clockwise (to the right of travel) around `p`.
    fn turns_right(&self, p: Point, d: Vec2, step: f64) -> bool {
        let ahead = self.flow(p + d * step);
        d.cross(ahead) < 0.0
    }

    /// Build `specs` in parallel and add them in order, within the budget, marking their
    /// coverage. Returns how many were added.
    fn add(&mut self, specs: &[Spec]) -> usize {
        let built = self.build_all(specs);
        self.commit(built)
    }

    /// Build `specs` in parallel, in order.
    fn build_all(&self, specs: &[Spec]) -> Vec<Placed> {
        let kit = Kit {
            canvas: self.canvas,
            palette: self.palette,
            acc: self.acc,
            species: &self.species,
        };
        let built: Vec<Option<Placed>> = specs.par_iter().map(|s| kit.build(s)).collect();
        built.into_iter().flatten().collect()
    }

    /// Add built elements in order, within the budget, marking their coverage.
    fn commit(&mut self, built: Vec<Placed>) -> usize {
        let mut n = 0;
        for p in built {
            let cost = p.node.path_count();
            if cost == 0 || !p.node.is_finite() || self.spent + cost > self.budget {
                continue;
            }
            self.spent += cost;
            if let Some(o) = &p.outline {
                self.cover.stamp(o);
            }
            if let Some((c, r)) = p.disc {
                self.cover.stamp_disc(c, r);
            }
            self.placed.push((p.center, p.species));
            self.nodes.push(p.node);
            n += 1;
        }
        n
    }

    /// Mark the calm sky as needing no filling: it gets its own sparse accents instead.
    fn skip_calm(&mut self) {
        for i in 0..self.cover.skip.len() {
            let p = self.cover.center(i);
            if self.calm(p) > 0.5 {
                self.cover.skip[i] = true;
            }
        }
    }

    /// A few scrolls in the calm sky, covering about a fifth of it.
    fn calm_accents(&mut self, depth: f64) {
        let (h, u) = (self.canvas.height, self.canvas.unit());
        let cell = self.cover.cell;
        let calm_area = (0..self.cover.skip.len())
            .filter(|&i| self.calm(self.cover.center(i)) > 0.5)
            .count() as f64
            * cell
            * cell;
        // A scroll 15 % of the height long covers about 0.0068 u².
        let count = (0.2 * calm_area / (0.0068 * u * u)).round() as usize;
        let mask = super::field::FnMask(|p: Point| {
            if self.calm(p) > 0.5 && self.density(p) > 0.0 {
                1.0
            } else {
                0.0
            }
        });
        let pts = place_points(
            self.seed.child("calm"),
            self.canvas.rect(),
            &mask,
            &|_| 0.18 * u,
            count,
        );
        let specs: Vec<Spec> = pts
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let k = self.seed.child("calm-accent").index(i as u64);
                let mut rng = k.rng();
                let dir = self.flow(p);
                let right = self.turns_right(p, dir, 0.05 * u) != rng.chance(0.3);
                let len = (h * rng.uniform(0.12, 0.18)).min(self.room(p) / 0.6);
                Spec {
                    at: p,
                    dir,
                    kind: Kind::Scroll(
                        ScrollParams {
                            stripes: 3,
                            hatch: false,
                            ..ScrollParams::acanthus(len, right)
                        }
                        .jittered(&mut rng, 0.8),
                    ),
                    depth,
                    tone: rng.uniform(0.3, 0.6),
                    seed: k,
                }
            })
            .collect();
        self.add(&specs);
    }

    /// Large scrolls anchored on the field's evenly spaced streamlines, back to front.
    /// `count` caps them; `(lo, hi)` is their length range as a fraction of the height.
    fn big_scrolls(&mut self, count: usize, lo: f64, hi: f64, depth: (f64, f64)) {
        let (w, h, u) = (self.canvas.width, self.canvas.height, self.canvas.unit());
        let bounds = Rect::new(-0.05 * w, -0.05 * h, 1.05 * w, 1.05 * h);
        let guides = guide_curves(
            self.seed.child("guides"),
            self.field,
            bounds,
            40,
            0.16 * u,
            0.01 * u,
            1.2 * h,
        );
        let mut rng = self.seed.rng_for("big-scrolls");
        let mut cands: Vec<(Point, Vec2, bool, f64)> = Vec::new();
        for (i, g) in guides.iter().enumerate() {
            let k = self.seed.child("guide-points").index(i as u64);
            for (t, p, tg) in along_curve(g, 0.3 * h, 0.6, k) {
                let len = h * rng.uniform(lo, hi);
                let keep = rng.uniform(0.0, 1.0);
                if !self.canvas.rect().inset(-0.03 * u).contains(p)
                    || keep >= self.density(p) * (1.0 - self.calm(p))
                {
                    continue;
                }
                // Curl the way the streamline bends, now and then against it.
                let right = (g.curvature(t) < 0.0) != rng.chance(0.25);
                cands.push((p, tg, right, len));
            }
        }
        // Thin out candidates that sit on top of each other, then shuffle the depth order.
        let mut picked: Vec<(Point, Vec2, bool, f64)> = Vec::new();
        for c in cands {
            if picked
                .iter()
                .all(|q| (c.0 - q.0).hypot() > 0.22 * (c.3 + q.3))
            {
                picked.push(c);
            }
        }
        for i in (1..picked.len()).rev() {
            let j = (rng.uniform(0.0, 1.0) * (i + 1) as f64) as usize;
            picked.swap(i, j.min(i));
        }
        picked.truncate(count);
        let n = picked.len().max(1);
        let specs: Vec<Spec> = picked
            .into_iter()
            .enumerate()
            .map(|(i, (p, tg, right, len))| {
                let d = lerp(depth.0, depth.1, i as f64 / n as f64);
                let mut srng = self.seed.child("big").index(i as u64).rng();
                let params = ScrollParams {
                    lobes: 3 + (srng.uniform(0.0, 4.0) as u32).min(3),
                    stripes: 3 + (srng.uniform(0.0, 4.0) as usize).min(3),
                    hatch: d > 0.25,
                    width: srng.uniform(0.13, 0.19),
                    ..ScrollParams::acanthus(len, right)
                }
                .jittered(&mut srng, 0.8);
                Spec {
                    at: p,
                    dir: tg,
                    kind: Kind::Scroll(params),
                    depth: d,
                    tone: lerp(0.05, 0.35, d) + srng.uniform(0.0, 0.2),
                    seed: self.seed.child("big").index(i as u64),
                }
            })
            .collect();
        self.add(&specs);
    }

    /// Fill the open gaps of the coverage grid, largest first, in batches: scrolls in
    /// large gaps, curls or leaves in smaller ones. Stops when less than `target` of the
    /// grid is open, no gap of `min_r` remains or after `rounds` batches.
    #[allow(clippy::too_many_arguments)]
    fn fill_gaps(
        &mut self,
        label: &str,
        scrolls: bool,
        min_r: f64,
        target: f64,
        rounds: usize,
        depth: (f64, f64),
    ) {
        let u = self.canvas.unit();
        let h = self.canvas.height;
        let jit = self.seed.child(label);
        let mut index = 0u64;
        for round in 0..rounds {
            if self.cover.open_fraction() < target || self.spent >= self.budget {
                break;
            }
            let rj = jit.index(round as u64);
            let gaps = self.cover.gaps(24, min_r, |i| {
                let mut r = rj.index(i as u64).rng();
                r.uniform(0.0, 1.0)
            });
            if gaps.is_empty() {
                break;
            }
            let f = round as f64 / rounds.max(1) as f64;
            let specs: Vec<Spec> = gaps
                .into_iter()
                .map(|(c, r)| {
                    let k = jit.child("element").index(index);
                    index += 1;
                    let mut rng = k.rng();
                    let d = lerp(depth.0, depth.1, f) + rng.uniform(0.0, 0.1);
                    let dir = self.flow(c);
                    let right = self.turns_right(c, dir, r) != rng.chance(0.3);
                    // Near an edge an element is sized to stay within the bleed.
                    let room = self.room(c);
                    let kind = if scrolls && r > 0.045 * u && room > 0.06 * h {
                        let len = (7.0 * r).clamp(0.1 * h, 0.32 * h).min(room / 0.6);
                        Kind::Scroll(
                            ScrollParams {
                                lobes: 3 + (rng.uniform(0.0, 3.0) as u32).min(2),
                                stripes: 3 + (rng.uniform(0.0, 3.0) as usize).min(2),
                                hatch: d > 0.3,
                                ..ScrollParams::acanthus(len, right)
                            }
                            .jittered(&mut rng, 0.8),
                        )
                    } else if r > 0.02 * u && rng.chance(0.45) {
                        Kind::Scroll(
                            ScrollParams {
                                width: rng.uniform(0.07, 0.1),
                                ..ScrollParams::curl((5.5 * r).min(room / 0.5), right)
                            }
                            .jittered(&mut rng, 0.6),
                        )
                    } else {
                        let (shape, aspect) = *rng.pick(&GAP_LEAVES);
                        Kind::Leaf {
                            shape,
                            length: (2.6 * r).min(0.16 * h).min(room / 0.6),
                            aspect,
                        }
                    };
                    // A leaf grows from its base: start it behind the gap centre.
                    let at = match &kind {
                        Kind::Leaf { length, .. } => c - dir * (0.45 * length),
                        _ => c,
                    };
                    Spec {
                        at,
                        dir,
                        kind,
                        depth: d.min(1.0),
                        tone: lerp(0.12, 0.75, d) + rng.uniform(-0.12, 0.12),
                        seed: k,
                    }
                })
                .collect();
            self.add(&specs);
        }
    }

    /// Leaves and a few flat star flowers sprinkled with density `mask` (and the calm sky
    /// thinned out).
    fn sprinkle(&mut self, mask: &dyn Mask, leaves: usize, flowers: usize, depth: (f64, f64)) {
        let (h, u) = (self.canvas.height, self.canvas.unit());
        let area = self.canvas.rect();
        let density = |p: Point| mask.at(p) * self.density(p);
        let dmask = super::field::FnMask(density);
        let pts = place_points(
            self.seed.child("sprinkle"),
            area,
            &dmask,
            &|_| 0.07 * u,
            leaves + flowers,
        );
        let star: Vec<usize> = self
            .species
            .iter()
            .enumerate()
            .filter(|(_, (n, _))| STAR_FLOWERS.contains(&n.as_str()))
            .map(|(i, _)| i)
            .collect();
        let n = pts.len().max(1);
        let mut specs: Vec<Spec> = Vec::new();
        let mut blooms: Vec<Spec> = Vec::new();
        for (i, p) in pts.into_iter().enumerate() {
            let k = self.seed.child("sprinkled").index(i as u64);
            let mut rng = k.rng();
            let d = lerp(depth.0, depth.1, i as f64 / n as f64);
            let dir = self.flow(p);
            // Every few points a flower, while the quota lasts.
            if blooms.len() < flowers && !star.is_empty() && i % 4 == 3 {
                blooms.push(Spec {
                    at: p,
                    dir: Vec2::from_angle(rng.uniform(0.0, TAU)),
                    kind: Kind::Flower {
                        species: *rng.pick(&star),
                        radius: h * rng.uniform(0.018, 0.032),
                        hue: rng.jitter(0.0, 25.0),
                    },
                    depth: 1.0,
                    tone: rng.uniform(0.35, 0.85),
                    seed: k,
                });
                continue;
            }
            let (shape, aspect) = *rng.pick(&GAP_LEAVES);
            let length = h * rng.uniform(0.05, 0.085);
            let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
            let dir = rotate(dir, side * rng.uniform(0.2, 0.7));
            specs.push(Spec {
                at: p - dir * (0.4 * length),
                dir,
                kind: Kind::Leaf {
                    shape,
                    length,
                    aspect,
                },
                depth: d,
                tone: lerp(0.3, 0.8, d) + rng.uniform(-0.1, 0.1),
                seed: k,
            });
        }
        self.add(&specs);
        self.add(&blooms);
    }

    /// The dense ground (see the module docs).
    fn dense(&mut self) {
        let f = self.fill;
        let u = self.canvas.unit();
        let area = self.canvas.width * self.canvas.height / (1.5 * u * u);
        self.beds();
        let big = (20.0 * f * area).round() as usize;
        self.big_scrolls(big, 0.25, 0.45, (0.0, 0.35));
        // Gaps the large scrolls left, as a mask for the sprinkles.
        let after_big = self.cover.covered.clone();
        self.skip_calm();
        self.calm_accents(0.35);
        // The calm sky counts as covered: it only gets its own sparse accents.
        let after_big: Vec<bool> = after_big
            .iter()
            .zip(&self.cover.skip)
            .map(|(a, b)| *a || *b)
            .collect();
        self.fill_gaps(
            "medium",
            true,
            0.025 * u,
            0.06,
            (40.0 * f).ceil() as usize,
            (0.35, 0.65),
        );
        let cover = &self.cover;
        let gap_mask = {
            let (nx, ny, cell) = (cover.nx, cover.ny, cover.cell);
            super::field::FnMask(move |p: Point| {
                let x = ((p.x / cell).floor().max(0.0) as usize).min(nx - 1);
                let y = ((p.y / cell).floor().max(0.0) as usize).min(ny - 1);
                if after_big[y * nx + x] { 0.25 } else { 1.0 }
            })
        };
        let leaves = (22.0 * f * area).round() as usize;
        let flowers = (12.0 * f * area).round() as usize;
        self.sprinkle(&gap_mask, leaves, flowers, (0.6, 0.85));
        self.fill_gaps(
            "holes",
            true,
            0.015 * u,
            0.012,
            (80.0 * f).ceil() as usize,
            (0.7, 0.9),
        );
    }

    /// The bloom field (see the module docs).
    fn blooms(&mut self) {
        let f = self.fill;
        let (w, h, u) = (self.canvas.width, self.canvas.height, self.canvas.unit());
        let area = w * h / (1.5 * u * u);
        self.beds();
        self.big_scrolls((6.0 * f * area).round() as usize, 0.28, 0.42, (0.0, 0.2));
        self.skip_calm();
        self.calm_accents(0.2);
        // Flower sizes first, largest placed first so the small ones fill around them.
        let mut rng = self.seed.rng_for("blooms");
        let count = ((40.0 * f * area).round() as usize).clamp(25, 60);
        let mut radii: Vec<f64> = (0..count)
            .map(|_| {
                let x: f64 = rng.uniform(0.0, 1.0);
                0.5 * h * (0.04 + 0.1 * x.powf(0.8))
            })
            .collect();
        radii.sort_by(|a, b| b.total_cmp(a));
        let cands = place_points(
            self.seed.child("bloom-sites"),
            self.canvas.rect().inset(-0.02 * u),
            &super::field::FnMask(|p: Point| self.density(p)),
            &|_| 0.035 * u,
            1200,
        );
        // Species round-robin over a shuffled list, so every species shows up.
        let mut order: Vec<usize> = (0..self.species.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.uniform(0.0, 1.0) * (i + 1) as f64) as usize;
            order.swap(i, j.min(i));
        }
        let mut placed: Vec<(Point, f64)> = Vec::new();
        let mut used = vec![false; cands.len()];
        let mut flowers: Vec<Spec> = Vec::new();
        for (j, &r) in radii.iter().enumerate() {
            let Some(ci) = (0..cands.len()).find(|&i| {
                !used[i]
                    && placed
                        .iter()
                        .all(|(q, s)| (cands[i] - *q).hypot() > 0.8 * (r + s))
            }) else {
                continue;
            };
            used[ci] = true;
            let p = cands[ci];
            placed.push((p, r));
            let k = self.seed.child("bloom").index(j as u64);
            let mut frng = k.rng();
            flowers.push(Spec {
                at: p,
                dir: Vec2::from_angle(frng.uniform(0.0, TAU)),
                kind: Kind::Flower {
                    species: order[j % order.len().max(1)],
                    radius: r,
                    hue: frng.jitter(0.0, 25.0),
                },
                depth: lerp(0.55, 1.0, 1.0 - j as f64 / count as f64),
                tone: frng.uniform(0.2, 0.85),
                seed: k,
            });
        }
        // Leaves and curls between the flowers go behind them: build the flowers and keep
        // their paths in reserve, mark them covered, fill the gaps, then add the flowers
        // smallest first so the large ones are in front.
        flowers.reverse();
        let built = self.build_all(&flowers);
        let reserve: usize = built.iter().map(|p| p.node.path_count()).sum();
        let reserve = reserve.min(self.budget.saturating_sub(self.spent));
        for &(p, r) in &placed {
            self.cover.stamp_disc(p, 0.5 * r);
        }
        self.budget -= reserve;
        self.fill_gaps(
            "between",
            false,
            0.015 * u,
            0.012,
            (80.0 * f).ceil() as usize,
            (0.25, 0.7),
        );
        self.budget += reserve;
        self.commit(built);
    }

    fn finish(self) -> Node {
        Node::group(self.nodes)
    }
}

/// What building an element needs (shared between threads).
struct Kit<'a> {
    canvas: &'a Canvas,
    palette: &'a Palette,
    acc: f64,
    species: &'a [(String, FlowerParams)],
}

impl Kit<'_> {
    /// The deep tone elements recede into at warmth `w`.
    fn deep(&self, w: f64) -> Color {
        self.palette
            .warm_at(0.0)
            .mix(self.palette.cool_at(0.0), 1.0 - w)
            .mix(self.palette.ink, 0.15)
    }

    fn scroll_style(&self, p: Point, t: f64, depth: f64) -> ScrollStyle {
        let (w, depth) = (self.canvas.warmth(p), self.damp() * depth);
        let a = ScrollStyle::from_palette(self.palette, true, t);
        let b = ScrollStyle::from_palette(self.palette, false, t);
        a.mix(&b, 1.0 - w)
            .receding(lerp(0.4, 1.0, depth), self.deep(w), lerp(0.45, 0.0, depth))
    }

    fn leaf_style(&self, p: Point, t: f64, depth: f64) -> LeafStyle {
        let (w, depth) = (self.canvas.warmth(p), self.damp() * depth);
        let s = leaf_style(self.palette, w, t);
        let (deep, k) = (self.deep(w), lerp(0.4, 0.0, depth));
        let c = 1.0 - lerp(0.45, 1.0, depth);
        LeafStyle {
            fill: s.fill.mix(deep, k),
            fill_light: s.fill_light.mix(s.fill, c).mix(deep, k),
            ink: s.ink.mix(s.fill, c).mix(deep, k),
            vein: s.vein.mix(s.fill, c).mix(deep, k),
            widths: s.widths,
        }
    }

    /// How far forward the ground may come (see [`GROUND_DEPTH`]).
    fn damp(&self) -> f64 {
        if self.palette.background.to_hsl().2 < 0.4 {
            GROUND_DEPTH_DARK
        } else {
            GROUND_DEPTH
        }
    }

    fn flower_style(&self, p: Point, t: f64, hue: f64, depth: f64) -> FlowerStyle {
        let w = self.canvas.warmth(p);
        let s = flower_style(self.palette, w, t);
        let (deep, k) = (self.deep(w), lerp(0.4, 0.12, self.damp() * depth));
        let tint = |c: Color| c.rotate_hue(hue).mix(deep, k);
        FlowerStyle {
            petal: tint(s.petal),
            petal_light: tint(s.petal_light),
            ink: s.ink.mix(s.petal, lerp(0.4, 0.0, depth)),
            vein: tint(s.vein),
            center: s.center.rotate_hue(-0.5 * hue).mix(deep, k),
        }
    }

    /// Accuracy for an element at `depth`: coarser far back.
    fn accuracy(&self, depth: f64) -> f64 {
        self.acc * lerp(4.0, 1.0, depth.clamp(0.0, 1.0))
    }

    /// Build one element.
    /// Build one element, shifted inwards where it would reach further than [`BLEED`]
    /// past the canvas edge (`None` if it is larger than the canvas and its bleed).
    fn build(&self, s: &Spec) -> Option<Placed> {
        let p = self.build_raw(s)?;
        let bb = p.node.bounding_box()?;
        let pad = BLEED * self.canvas.height;
        let allowed = self.canvas.rect().inset(pad);
        let shift = |lo: f64, hi: f64, min: f64, max: f64| {
            if hi - lo > max - min {
                None
            } else if lo < min {
                Some(min - lo)
            } else if hi > max {
                Some(max - hi)
            } else {
                Some(0.0)
            }
        };
        let d = Vec2::new(
            shift(bb.x0, bb.x1, allowed.x0, allowed.x1)?,
            shift(bb.y0, bb.y1, allowed.y0, allowed.y1)?,
        );
        if d == Vec2::ZERO {
            return Some(p);
        }
        let t = Affine::translate(d);
        Some(Placed {
            node: p.node.with_transform(t),
            outline: p.outline.map(|o| t * o),
            disc: p.disc.map(|(c, r)| (c + d, r)),
            center: p.center + d,
            species: p.species,
        })
    }

    fn build_raw(&self, s: &Spec) -> Option<Placed> {
        match &s.kind {
            Kind::Scroll(params) => {
                let style = self.scroll_style(s.at, s.tone, s.depth);
                let b = scroll::build(&s.seed, params, &style, self.accuracy(s.depth), false)?;
                let (a, tg) = b.anchor;
                let angle = s.dir.y.atan2(s.dir.x) - tg.y.atan2(tg.x);
                let t = Affine::translate(s.at.to_vec2())
                    * Affine::rotate(angle)
                    * Affine::translate(-a.to_vec2());
                Some(Placed {
                    node: b.node.with_transform(t),
                    outline: Some(t * b.outline),
                    disc: None,
                    center: s.at,
                    species: None,
                })
            }
            Kind::Leaf {
                shape,
                length,
                aspect,
            } => {
                let mut rng = s.seed.rng_for("leaf");
                let params = LeafParams::new(*shape)
                    .with_tip_curl(rng.jitter(0.0, 0.5))
                    .jittered(&mut rng, 0.6);
                let leaf = Leaf::from_base(s.at, s.dir, *length, aspect * length, params);
                let style = self.leaf_style(s.at, s.tone, s.depth);
                let acc = self.accuracy(s.depth);
                // Background leaves keep their hatching and veinlets only when large.
                let node =
                    leaf_motif_with_detail(&s.seed, &leaf, &style, acc, Detail::at_scale(0.5));
                if node.path_count() == 0 {
                    return None;
                }
                Some(Placed {
                    node,
                    outline: Some(leaf.outline(acc)),
                    disc: None,
                    center: leaf.point(0.5, 0.0),
                    species: None,
                })
            }
            Kind::Flower {
                species,
                radius,
                hue,
            } => {
                let (_, params) = self.species.get(*species)?;
                let style = self.flower_style(s.at, s.tone, *hue, s.depth);
                let scale = radius / params.outer_radius().max(1e-6);
                let acc = self.accuracy(s.depth) / scale;
                let node = flower::flower_with_detail(
                    &s.seed,
                    params,
                    &style,
                    acc,
                    // Background flowers keep their main veins only, the front ones forks too.
                    Detail::at_scale(if s.depth > 0.9 { 0.3 } else { 0.18 }),
                );
                let rot = s.dir.y.atan2(s.dir.x);
                let node = node.with_transform(
                    Affine::translate(s.at.to_vec2()) * Affine::rotate(rot) * Affine::scale(scale),
                );
                Some(Placed {
                    node,
                    outline: None,
                    disc: Some((s.at, 0.5 * radius)),
                    center: s.at,
                    species: Some(*species),
                })
            }
        }
    }
}

/// `v` rotated by `angle`.
fn rotate(v: Vec2, angle: f64) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compose::preset::{Focus, flow_field};
    use crate::geom::validate::path_is_finite;

    const W: f64 = 300.0;
    const H: f64 = 450.0;

    /// Element centres and flower species.
    type Placements = Vec<(Point, Option<usize>)>;

    fn setup(seed: u64) -> (Canvas, super::super::field::Sum, Occupancy) {
        let tree = SeedTree::new(seed);
        let canvas = Canvas::new(&tree.child("canvas"), W, H);
        let u = canvas.unit();
        let foci = [
            Focus {
                center: Point::new(0.25 * W, 0.62 * H),
                radius: 0.16 * u,
                spin: 1.0,
            },
            Focus {
                center: Point::new(0.75 * W, 0.4 * H),
                radius: 0.13 * u,
                spin: -1.0,
            },
        ];
        let field = flow_field(&tree, &canvas, &foci, None);
        let mut occ = Occupancy::new();
        for f in &foci {
            occ.add_circle(f.center, 1.05 * f.radius);
        }
        // Ribbons and plants are marked too, but the ground may overlap them.
        occ.add_circle(Point::new(0.5 * W, 0.5 * H), 0.03 * u);
        occ.add_region(Region::Rect(Rect::new(0.0, 0.8 * H, 0.3 * W, H)));
        (canvas, field, occ)
    }

    fn ground(seed: u64, blooms: bool) -> (Node, Placements, Vec<Region>, Canvas) {
        let (canvas, field, occ) = setup(seed);
        let p = Palette::by_name("botanica").unwrap();
        let tree = SeedTree::new(seed).child("ground");
        let mut g = Ground::new(&tree, &canvas, &p, &field, &occ, 1.0, 0.05);
        if blooms {
            g.blooms();
        } else {
            g.dense();
        }
        let (placed, keep) = (g.placed.clone(), g.keep.clone());
        (g.finish(), placed, keep, canvas)
    }

    /// Fraction of a grid of sample points inside an opaque fill of `node`.
    fn coverage(node: &Node) -> f64 {
        coverage_where(node, &|_| true)
    }

    /// Fraction of the sample points where `only` holds that an opaque fill covers.
    fn coverage_where(node: &Node, only: &dyn Fn(Point) -> bool) -> f64 {
        let mut fills: Vec<(BezPath, Rect)> = Vec::new();
        node.for_each_path(|path, style, t| {
            let alpha = style.fill.map_or(0.0, |c| c.a) * style.opacity;
            if alpha >= 0.5 {
                let p = t * path.clone();
                let bb = p.bounding_box();
                fills.push((p, bb));
            }
        });
        let (nx, ny) = (60, 90);
        let (mut hit, mut all) = (0, 0);
        for j in 0..ny {
            for i in 0..nx {
                let q = Point::new(
                    (i as f64 + 0.5) * W / nx as f64,
                    (j as f64 + 0.5) * H / ny as f64,
                );
                if !only(q) {
                    continue;
                }
                all += 1;
                if fills
                    .iter()
                    .any(|(p, bb)| bb.contains(q) && p.winding(q) != 0)
                {
                    hit += 1;
                }
            }
        }
        hit as f64 / all.max(1) as f64
    }

    fn check(blooms: bool) {
        for seed in [1, 2] {
            let (node, placed, keep, canvas) = ground(seed, blooms);
            assert!(node.is_finite(), "seed {seed}: non-finite");
            node.for_each_path(|p, _, _| assert!(path_is_finite(p)));
            let n = node.path_count();
            assert!(n > 0 && n <= PATH_BUDGET as usize, "seed {seed}: {n} paths");
            let c = coverage(&node);
            eprintln!(
                "seed {seed} blooms {blooms}: {n} paths, {} elements, coverage {c:.3}",
                placed.len()
            );
            // The calm upper half of the cool side is left to the open landscape, so the
            // ground alone covers about three quarters of the canvas.
            assert!(c >= 0.7, "seed {seed}: coverage {c:.3}");
            // Nothing sits in a focal flower's core.
            let inside = placed
                .iter()
                .filter(|(p, _)| keep.iter().any(|r| r.contains(*p)))
                .count();
            assert!(
                inside * 50 <= placed.len(),
                "seed {seed}: {inside} in keep-outs"
            );
            // The calm sky gets at most about a third of the density elsewhere.
            let h = canvas.height;
            let calm = |p: Point| {
                (1.0 - canvas.warmth(p)) * (1.0 - smoothstep((p.y - 0.16 * h) / (0.08 * h))) > 0.5
            };
            let probe = (0..2500)
                .filter(|i| {
                    calm(Point::new(
                        (i % 50) as f64 * W / 50.0,
                        (i / 50) as f64 * H / 50.0,
                    ))
                })
                .count() as f64
                / 2500.0;
            let share =
                placed.iter().filter(|(p, _)| calm(*p)).count() as f64 / placed.len() as f64;
            assert!(
                share <= 0.4 * probe + 0.01,
                "seed {seed}: {share:.3} of elements in {probe:.3} calm area"
            );
            if blooms {
                let flowers: Vec<usize> = placed.iter().filter_map(|(_, s)| *s).collect();
                assert!(
                    (25..=60).contains(&flowers.len()),
                    "{} flowers",
                    flowers.len()
                );
                let mut kinds = flowers.clone();
                kinds.sort_unstable();
                kinds.dedup();
                assert!(kinds.len() >= 6, "{} species", kinds.len());
            }
        }
    }

    #[test]
    fn dense_ground_covers_the_canvas_within_budget() {
        check(false);
    }

    #[test]
    fn bloom_field_covers_the_canvas_with_varied_flowers() {
        check(true);
    }

    #[test]
    fn fills_are_deterministic() {
        for blooms in [false, true] {
            assert_eq!(ground(3, blooms).0, ground(3, blooms).0);
        }
    }

    #[test]
    fn gaps_shrink_as_the_grid_fills() {
        let (canvas, ..) = setup(1);
        let mut c = Cover::new(&canvas, 10.0);
        c.stamp_disc(Point::new(150.0, 225.0), 40.0);
        let g = c.gaps(3, 1.0, |_| 0.0);
        assert!(!g.is_empty());
        let r0 = g[0].1;
        c.stamp_disc(g[0].0, r0);
        assert!(c.gaps(1, 1.0, |_| 0.0)[0].1 <= r0 + 1e-9);
        assert!(c.open_fraction() < 1.0);
    }
}
