//! The botanica preset: a tall Art Nouveau poster split into a warm and a cool region.
//!
//! Composition rules, applied in order (warm-left frame, see [`Canvas`]):
//! 1. **Regions and foci.** An organic S divider splits the canvas. A large warm focal
//!    flower sits low in the warm region, a cool one higher in the cool region, and an
//!    optional third, smaller one near the top of the warm side. The moonlit opening, when
//!    enabled, takes the top of the cool region and pushes the cool focus below it.
//! 2. **Flow.** A field of two counter-rotating vortices winding into the foci, curl noise,
//!    an upward drift and a repeller around the opening. Guide curves are its streamlines,
//!    seeded densest along the margins and the divider, longest first, each trimmed to the
//!    part that keeps its distance from earlier guides and stays out of the opening and the
//!    flower centres.
//! 3. **Ribbons.** The divider becomes the dominant band rolling into a large coil, with a
//!    narrower companion. The longest guides become ribbons, wider the longer they are,
//!    rolling into spiral tails at free ends (ends that run into a focal flower hide under
//!    it); a few sprout compound scrolls.
//! 4. **Plants** grow from the ground line, regional species leaning towards their focus.
//! 5. **Vines.** The remaining guides become thin stems with alternating leaves and small
//!    flowers, placed where the focal density mask is high and shrinking with distance from
//!    the nearest focus.
//! 6. **Focal flowers** on collars of crisp leaves, with compound scrolls behind them.
//! 7. **Background foliage** fills the space left by everything else: leaves along the
//!    field direction, dim and low-contrast, placed where the keep-out mask of all placed
//!    elements is high.

use super::field::{
    Band, Combine, Constant, Field, FnMask, FocalPoint, Mask, along_curve, focal_mask, place_points,
};
use super::landscape::{Opening, WindowShape, opening_node};
use super::layout::{
    Canvas, Layer, Layers, Occupancy, flower_style, leaf_style, muted_leaf, plant_style,
    ribbon_style, unit_or,
};
use super::preset::{
    Focus, background, budget, compound_scroll, divider_band, fit_tail, flow_field, flower_at,
    grow_ribbon, guide_starts, leaf_at, leaf_collar, leaf_shape, lean_towards, mean_warmth, rotate,
    scroll_ribbon, select_guides, with_tails,
};
use crate::config::Config;
use crate::geom::leaf::{LeafParams, LeafShape};
use crate::geom::{Composite, Curve, CurveExt, lerp, smoothstep};
use crate::motif::flower::FlowerStyle;
use crate::motif::plant::{self, PlantParams};
use crate::palette::Palette;
use crate::rng::{Draw, SeedTree};
use crate::scene::Node;
use kurbo::{Affine, Point, Rect, Vec2};
use rayon::prelude::*;
use std::f64::consts::TAU;

/// `x` clamped to `[lo, hi]`, or the midpoint when the range is empty.
fn fit(x: f64, lo: f64, hi: f64) -> f64 {
    if lo > hi {
        0.5 * (lo + hi)
    } else {
        x.clamp(lo, hi)
    }
}

const WARM_FOCAL: [&str; 4] = ["rose", "peony", "poppy", "lotus"];
const COOL_FOCAL: [&str; 4] = ["lotus", "lily", "aster", "peony"];
const WARM_SMALL: [&str; 5] = ["cosmos", "poppy", "primrose", "carnation", "rose"];
const COOL_SMALL: [&str; 5] = ["bellflower", "aster", "daisy", "primrose", "lily"];

/// Total absolute turning of a curve in radians.
fn turning(c: &Composite) -> f64 {
    let s = c.sample_by_arclen(64);
    s.windows(3)
        .map(|w| {
            let (a, b) = (w[1].1 - w[0].1, w[2].1 - w[1].1);
            a.cross(b).atan2(a.dot(b)).abs()
        })
        .sum()
}

/// Length weighted by how much a curve turns (up to a full turn): ribbons want sweep.
fn sweep(c: &Composite) -> f64 {
    c.length() * (0.35 + turning(c).min(TAU) / TAU)
}

/// Leaves of the vines: simple, readable silhouettes at small sizes.
const VINE_LEAVES: [(LeafShape, f64); 4] = [
    (LeafShape::Ovate, 0.5),
    (LeafShape::Cordate, 0.6),
    (LeafShape::Lanceolate, 0.28),
    (LeafShape::Serrated { teeth: 9 }, 0.38),
];

/// A placed focal flower.
struct Focal {
    focus: Focus,
    warm: bool,
}

/// Compose the botanica scene (see the module docs for the rules).
pub(crate) fn compose(config: &Config, palette: &Palette) -> (Canvas, Layers) {
    let seed = SeedTree::new(config.seed).child("botanica");
    let canvas = Canvas::new(&seed.child("canvas"), config.width, config.height);
    let mut layers = Layers::new(budget(config));
    let mut rng = seed.rng_for("layout");
    let (w, h, u) = (canvas.width, canvas.height, canvas.unit());
    let acc = config.accuracy;
    let fill = config.density * config.detail.clamp(0.25, 2.0);
    background(&seed, &canvas, palette, &mut layers);
    let mut occ = Occupancy::new();

    // 1. Opening and foci.
    let opening = config.opening.then(|| {
        let r0 = (0.2 * w).min(0.125 * h);
        let y = 0.045 * h + r0;
        let dx = (0..=8)
            .map(|i| canvas.divider_x(y - r0 + 2.0 * r0 * i as f64 / 8.0))
            .fold(f64::MIN, f64::max);
        let r = r0.min(0.5 * (w - dx) - 0.05 * u).max(0.08 * u);
        Opening {
            center: Point::new(fit(0.5 * (dx + w), dx + r + 0.03 * u, w - r - 0.04 * u), y),
            radius: r,
            shape: if rng.chance(0.5) {
                WindowShape::Circle
            } else {
                WindowShape::Arch
            },
        }
    });
    let window = opening.map(|o| o.region(0.035 * u));
    if let Some(r) = &window {
        occ.add_region(r.clone());
    }

    let mut focals = Vec::new();
    let warm_r = u * rng.uniform(0.15, 0.175);
    let yw = h * rng.uniform(0.58, 0.66);
    let dxw = canvas.divider_x(yw);
    let xw = fit(
        dxw * rng.uniform(0.45, 0.55),
        warm_r + 0.05 * u,
        dxw - warm_r - 0.03 * u,
    );
    focals.push(Focal {
        focus: Focus {
            center: Point::new(xw, yw),
            radius: warm_r,
            spin: 1.0,
        },
        warm: true,
    });
    let cool_r = u * rng.uniform(0.12, 0.145);
    let below_window = opening.map_or(0.0, |o| o.bounds().y1 + cool_r + 0.06 * u);
    let yc = (h * rng.uniform(0.36, 0.43)).max(below_window);
    let dxc = canvas.divider_x(yc);
    let xc = fit(
        0.5 * (dxc + w) + rng.jitter(0.0, 0.04 * w),
        dxc + cool_r + 0.03 * u,
        w - cool_r - 0.05 * u,
    );
    focals.push(Focal {
        focus: Focus {
            center: Point::new(xc, yc),
            radius: cool_r,
            spin: -1.0,
        },
        warm: false,
    });
    if rng.chance(0.8) {
        let r = u * rng.uniform(0.075, 0.095);
        let y = h * rng.uniform(0.12, 0.17);
        let dx = canvas.divider_x(y);
        let x = fit(0.45 * dx, r + 0.04 * u, dx - r - 0.03 * u);
        focals.push(Focal {
            focus: Focus {
                center: Point::new(x, y),
                radius: r,
                spin: -1.0,
            },
            warm: true,
        });
    }
    let foci: Vec<Focus> = focals.iter().map(|f| f.focus).collect();
    for f in &foci {
        occ.add_circle(f.center, 1.05 * f.radius);
    }
    let field = flow_field(&seed, &canvas, &foci, opening.map(|o| (o.center, o.radius)));

    // 2–3. Divider band, guides, ribbons.
    // Ribbons keep clear of the opening and of each other, but may pass under flowers.
    let mut bands = Occupancy::new();
    if let Some(r) = &window {
        bands.add_region(r.clone());
    }
    let divider = divider_band(&seed, &canvas, palette, &mut layers, &mut bands, acc);
    let bounds = canvas.inner(0.03 * u);
    let margin = FnMask(move |p: Point| {
        let d = p.x.min(w - p.x).min(p.y).min(h - p.y);
        1.0 - smoothstep(d / (0.2 * u))
    });
    let seed_mask = Combine::Max(
        Box::new(Combine::Max(
            Box::new(Band::along(&canvas.divider, 0.12 * u, 0.6)),
            Box::new(margin),
        )),
        Box::new(Constant(0.2)),
    );
    let starts = guide_starts(&seed, bounds, &seed_mask, 0.065 * u, 56);
    let window_c = window.clone();
    let foci_c = foci.clone();
    let free = move |p: Point| {
        !window_c.as_ref().is_some_and(|r| r.contains(p))
            && foci_c
                .iter()
                .all(|f| (p - f.center).hypot() > 0.55 * f.radius)
    };
    let taken: Vec<Point> = divider
        .sample_by_arclen(240)
        .into_iter()
        .map(|s| s.1)
        .collect();
    let n_guides = 8 + (rng.uniform(0.0, 5.0) as usize).min(4);
    let mut guides = select_guides(
        &field,
        &starts,
        bounds,
        0.005 * u,
        1.3 * h,
        0.2 * u,
        0.075 * u,
        n_guides,
        &taken,
        &free,
        acc,
    );
    // Long, curving guides make the ribbons; straight ones are left for vines.
    guides.sort_by(|a, b| sweep(b).total_cmp(&sweep(a)));
    let n_ribbons = (guides.len() / 2 + 1).clamp(3, 6).min(guides.len());
    // Ribbons alternate between the regions: the side with fewer ribbons takes its best
    // guide unless that one sweeps less than half as much as the best overall.
    let mut rest: Vec<(bool, f64, Composite)> = guides
        .into_iter()
        .map(|g| (mean_warmth(&canvas, &g) > 0.5, sweep(&g), g))
        .collect();
    let mut guides = Vec::with_capacity(rest.len());
    let mut count = [0usize; 2];
    while guides.len() < n_ribbons && !rest.is_empty() {
        let want_warm = count[1] <= count[0];
        let best = rest[0].1;
        let i = rest
            .iter()
            .position(|(warm, s, _)| *warm == want_warm && *s >= 0.3 * best)
            .unwrap_or(0);
        let (warm, _, g) = rest.remove(i);
        count[usize::from(warm)] += 1;
        guides.push(g);
    }
    guides.extend(rest.into_iter().map(|(_, _, g)| g));
    let longest = guides.first().map_or(1.0, |g| g.length());
    let near_focus = |p: Point| {
        foci.iter()
            .any(|f| (p - f.center).hypot() < 0.85 * f.radius)
    };
    for (i, g) in guides.iter().take(n_ribbons).enumerate() {
        let k = seed.child("ribbon").index(i as u64);
        let scale = if i < 3 { 1.0 } else { 0.7 };
        let hw = (0.022 * u * (g.length() / longest).sqrt() * scale).max(0.007 * u);
        if let Some((node, centerline)) =
            grow_ribbon(&k, &canvas, palette, g, hw, &bands, &near_focus, acc)
            && layers.add(Layer::Ribbons, node)
        {
            bands.add_curve(&centerline, hw + 0.012 * u);
        }
    }
    occ.extend(&bands);

    // 4. Plants on the ground line.
    let n_plants = 4 + (rng.uniform(0.0, 3.0) as usize).min(2);
    let ground = 1.004 * h;
    let div_x = canvas.divider_x(h);
    let mut xs: Vec<f64> = (0..n_plants)
        .map(|i| w * (i as f64 + 0.5 + rng.jitter(0.0, 0.3)) / n_plants as f64)
        .collect();
    // The plant closest to the divider's foot grows from it.
    if let Some(i) =
        (0..n_plants).min_by(|&a, &b| (xs[a] - div_x).abs().total_cmp(&(xs[b] - div_x).abs()))
    {
        xs[i] = div_x + rng.jitter(0.0, 0.03 * w);
    }
    // Rules first (sequential, seeded per plant), growth in parallel, in order.
    let specs: Vec<_> = xs
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let k = seed.child("plant").index(i as u64);
            let mut prng = k.rng();
            let base = Point::new(x.clamp(0.04 * w, 0.96 * w), ground);
            let warm = canvas.is_warm(Point::new(base.x, 0.9 * h));
            let by_divider = (x - div_x).abs() < 0.06 * w;
            let mut params = if warm {
                let mut p = if prng.chance(0.5) {
                    PlantParams::flowering_sprig()
                } else {
                    PlantParams {
                        flower_size: 0.1,
                        ..PlantParams::wild_bouquet()
                    }
                };
                p.flowers = WARM_SMALL.iter().map(|s| s.to_string()).collect();
                p
            } else if prng.chance(0.5) {
                PlantParams::fern_clump()
            } else {
                PlantParams::climbing_vine()
            };
            let target = h * if by_divider {
                prng.uniform(0.34, 0.42)
            } else {
                prng.uniform(0.2, 0.3)
            };
            params.height = target;
            // Plants are the costliest motifs: their branching follows the detail level.
            let d = config.detail.clamp(0.3, 1.0);
            params.branches_per_level =
                ((params.branches_per_level as f64 * d).round() as usize).max(2);
            if d < 0.75 {
                params.branch_levels = params.branch_levels.min(1);
            }
            params.leaf_density *= d.max(0.5);
            let focus = focals
                .iter()
                .filter(|f| f.warm == warm)
                .map(|f| f.focus.center)
                .min_by(|a, b| (*a - base).hypot().total_cmp(&(*b - base).hypot()))
                .unwrap_or(Point::new(0.5 * w, 0.5 * h));
            let up = lean_towards(base, focus, 0.3);
            (k, params, base, up, warm, target)
        })
        .collect();
    let plants: Vec<Node> = specs
        .par_iter()
        .map(|(k, params, base, up, warm, target)| {
            let style = plant_style(palette, *warm);
            let node = plant::plant(k, params, *base, *up, &style, acc);
            // Normalise the drawn size (fern clumps grow far beyond their stem height).
            let s = node.bounding_box().map_or(1.0, |bb| {
                (1.15 * target / bb.height().max(1e-6))
                    .min(0.8 * target / bb.width().max(1e-6))
                    .min(1.0)
            });
            let b = base.to_vec2();
            Node::group(vec![node])
                .with_transform(Affine::translate(b) * Affine::scale(s) * Affine::translate(-b))
        })
        .collect();
    for g in plants {
        let region = g.clone();
        if layers.add(Layer::Plants, g) {
            occ.add_node(&region, 0.2);
        }
    }

    // 5. Vines with leaves and small flowers along the remaining guides.
    let attraction: Vec<FocalPoint> = foci
        .iter()
        .map(|f| FocalPoint {
            center: f.center,
            radius: 4.0 * f.radius,
            weight: 1.0,
        })
        .collect();
    let attraction = focal_mask(&attraction);
    let mut blooms: Vec<(Point, f64)> = Vec::new();
    for (i, g) in guides.iter().enumerate().skip(n_ribbons) {
        // A vine needs some bend; straight streamlines are left to the background.
        if turning(g) < 0.8 {
            continue;
        }
        let k = seed.child("vine").index(i as u64);
        let mut vrng = k.rng();
        let hw = 0.005 * u;
        let blocked = |p: Point| occ.contains(p) || !canvas.rect().contains(p);
        let end = fit_tail(&mut vrng, g, hw * 1.6, bounds, &blocked);
        let centerline = with_tails(g, None, end, acc);
        let warmth = mean_warmth(&canvas, &centerline);
        let warm = warmth > 0.5;
        let mut vine = Node::empty();
        let stem = ribbon_style(palette, warmth, 0.0);
        vine.push(scroll_ribbon(
            &k,
            &centerline,
            hw,
            (0.1, 0.02),
            0,
            &stem,
            acc,
        ));
        let len = g.length();
        let spacing = 0.05 * u * vrng.uniform(0.9, 1.2);
        let mut leaves = Node::empty();
        for (j, (t, p, tg)) in along_curve(g, spacing, 0.4, k.child("leaves"))
            .into_iter()
            .enumerate()
        {
            if !free(p) || !vrng.chance(0.85) {
                continue;
            }
            let side = if j % 2 == 0 { 1.0 } else { -1.0 };
            let s = (g.point(t) - g.start()).hypot() / len.max(1e-6);
            let llen = 0.065 * u * vrng.uniform(0.8, 1.15) * (1.0 - 0.3 * s.min(1.0));
            let d = rotate(tg, side * vrng.uniform(0.55, 0.95));
            let (shape, aspect) = *vrng.pick(&VINE_LEAVES);
            let params = LeafParams::new(shape)
                .with_tip_curl(-side * 0.3)
                .jittered(&mut vrng, 0.5);
            let style = leaf_style(palette, warmth.round(), vrng.uniform(0.15, 0.6));
            leaves.push(leaf_at(
                &k.index(j as u64),
                p,
                d,
                llen,
                aspect,
                params,
                &style,
                acc,
            ));
        }
        vine.push(leaves);
        let mut flowers = Node::empty();
        for (j, (_, p, tg)) in along_curve(g, 0.11 * u, 0.6, k.child("flowers"))
            .into_iter()
            .enumerate()
        {
            let m = attraction.at(p);
            if !free(p)
                || blooms.iter().any(|(q, r)| (p - *q).hypot() < r + 0.07 * u)
                || !vrng.chance((0.25 + 0.75 * m) * config.density.min(1.5))
            {
                continue;
            }
            let dist = foci
                .iter()
                .map(|f| (p - f.center).hypot() - f.radius)
                .fold(f64::INFINITY, f64::min);
            let fr = lerp(0.06, 0.028, smoothstep(dist / (0.55 * h))) * u;
            let species = if warm {
                *vrng.pick(&WARM_SMALL)
            } else {
                *vrng.pick(&COOL_SMALL)
            };
            let style = flower_style(palette, warmth.round(), vrng.uniform(0.3, 0.8));
            let rot = tg.y.atan2(tg.x) + vrng.uniform(0.0, TAU);
            flowers.push(flower_at(
                &k.child("flower").index(j as u64),
                species,
                p,
                fr,
                rot,
                &style,
                acc,
            ));
            blooms.push((p, fr));
        }
        vine.push(flowers);
        if layers.add(Layer::Vines, vine) {
            occ.add_curve(&centerline, 0.03 * u);
        }
    }
    for &(p, r) in &blooms {
        occ.add_circle(p, 1.1 * r);
    }

    // 6. Focal flowers with collars of leaves and scrolls.
    let focal_nodes: Vec<Node> = focals
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let k = seed.child("focal").index(i as u64);
            let mut frng = k.rng();
            let (c, r) = (f.focus.center, f.focus.radius);
            let mut g = Node::empty();
            let count = 5 + (frng.uniform(0.0, 3.0) as usize).min(2);
            g.push(leaf_collar(&k, c, r, count, f.warm, palette, acc));
            for s in 0..2 {
                let a = frng.uniform(0.0, TAU);
                let d = Vec2::from_angle(a);
                let side = if s == 0 { 1.0 } else { -1.0 };
                let start = c + d * (0.8 * r);
                let style = ribbon_style(palette, f64::from(u8::from(f.warm)), 0.3);
                let other = ribbon_style(palette, f64::from(u8::from(!f.warm)), 0.35);
                let node = compound_scroll(
                    &k.child("scroll").index(s),
                    start,
                    rotate(d, side * 0.6),
                    r * frng.uniform(1.0, 1.4),
                    side > 0.0,
                    &style,
                    &other,
                    acc,
                );
                let fits = node.bounding_box().is_some_and(|bb| {
                    canvas.inner(0.015 * u).contains(bb.origin())
                        && canvas.inner(0.015 * u).contains(Point::new(bb.x1, bb.y1))
                        && opening
                            .is_none_or(|o| bb.intersect(o.bounds().inset(0.03 * u)).area() <= 0.0)
                });
                if fits {
                    g.push(node);
                }
            }
            let species = if f.warm {
                *frng.pick(&WARM_FOCAL)
            } else {
                *frng.pick(&COOL_FOCAL)
            };
            let style = if i == 2 {
                FlowerStyle::from_palette(palette, f.warm, frng.uniform(0.4, 0.6))
            } else {
                FlowerStyle::from_palette(palette, f.warm, frng.uniform(0.2, 0.42))
            };
            g.push(flower_at(
                &k.child("flower"),
                species,
                c,
                r,
                frng.uniform(0.0, TAU),
                &style,
                acc,
            ));
            g
        })
        .collect();
    for g in focal_nodes {
        layers.add(Layer::Foreground, g);
    }

    // The opening.
    if let Some(o) = &opening {
        layers.add(
            Layer::Opening,
            opening_node(&seed.child("opening"), o, palette, acc),
        );
    }

    // 7. Background foliage where nothing else sits: sprays of leaves fanning from a common
    // point along the field, then single leaves in the gaps that remain.
    let up = Vec2::new(0.0, -1.0);
    let area = Rect::new(-0.03 * w, -0.03 * h, 1.03 * w, 1.03 * h);
    // (base, direction, length, warm, shape seed)
    let mut specs: Vec<(Point, Vec2, f64, bool, SeedTree)> = Vec::new();
    let mut gaps = occ.clone();
    let centers = place_points(
        seed.child("sprays"),
        area,
        &occ.free_mask(0.05 * u),
        &|_| 0.16 * u,
        (36.0 * fill) as usize,
    );
    for (j, p) in centers.into_iter().enumerate() {
        let k = seed.child("spray").index(j as u64);
        let mut srng = k.rng();
        let flow = unit_or(field.at(p), up);
        let n = 3 + (srng.uniform(0.0, 3.0) as usize).min(2);
        let spread = srng.uniform(0.45, 0.75);
        let len0 = u * srng.uniform(0.09, 0.14);
        let base = p - flow * (0.4 * len0);
        for i in 0..n {
            let f = if n > 1 {
                2.0 * i as f64 / (n - 1) as f64 - 1.0
            } else {
                0.0
            };
            let d = rotate(flow, f * spread + srng.jitter(0.0, 0.1));
            let len = len0 * (1.0 - 0.3 * f.abs()) * srng.uniform(0.9, 1.1);
            specs.push((base, d, len, canvas.is_warm(p), k));
        }
        gaps.add_circle(p, 0.75 * len0);
    }
    let singles = place_points(
        seed.child("back-singles"),
        area,
        &gaps.free_mask(0.03 * u),
        &|_| 0.06 * u,
        (30.0 * fill) as usize,
    );
    for (j, p) in singles.into_iter().enumerate() {
        let k = seed.child("back-single").index(j as u64);
        let mut lrng = k.rng();
        let d = rotate(unit_or(field.at(p), up), lrng.jitter(0.0, 0.5));
        let len = u * lrng.uniform(0.05, 0.08);
        specs.push((p - d * (0.45 * len), d, len, canvas.is_warm(p), k));
    }
    let leaves: Vec<Node> = specs
        .into_par_iter()
        .enumerate()
        .map(|(j, (base, d, len, warm, shape_seed))| {
            // Leaves of one spray share their shape; colour and detail vary per leaf.
            let (shape, aspect) = leaf_shape(&mut shape_seed.rng(), warm);
            let k = seed.child("back-leaf").index(j as u64);
            let mut lrng = k.rng();
            let params = LeafParams::new(shape)
                .with_tip_curl(lrng.jitter(0.0, 0.45))
                .jittered(&mut lrng, 0.6);
            let style = muted_leaf(
                &leaf_style(palette, f64::from(u8::from(warm)), lrng.uniform(0.0, 0.3)),
                palette.background,
            );
            leaf_at(&k, base, d, len, aspect, params, &style, acc)
        })
        .collect();
    let mut back = Node::empty();
    let reserve = layers.budget.remaining();
    let mut spent = 0;
    for leaf in leaves {
        let n = leaf.path_count();
        if spent + n > reserve {
            break;
        }
        spent += n;
        back.push(leaf);
    }
    layers.add(Layer::BackFoliage, back.with_opacity(0.72));
    (canvas, layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_handles_empty_ranges() {
        assert_eq!(fit(5.0, 0.0, 1.0), 1.0);
        assert_eq!(fit(5.0, 2.0, 1.0), 1.5);
    }
}
