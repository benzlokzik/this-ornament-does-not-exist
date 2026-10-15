//! Presets: full-scene compositions built from the motif library by rules.
//!
//! Every preset composes in the warm-left frame of [`Canvas`] (mirrored at the end when
//! the seed puts the warm region on the right) and paints in the order of [`Layer`].
//! - [`Preset::Botanica`]: the full scene, see [`super::botanica`].
//! - [`Preset::Bloom`]: one large focal flower on a collar of leaves between two scrolls.
//! - [`Preset::Ribbons`]: ribbons and spirals only, grown along the streamlines of a field
//!   with two counter-rotating vortices, warm on one side of the divider, cool on the other.
//!
//! The shared building blocks live here: the tonal background, the flow field, guide
//! selection, ribbons with spiral tails grown along guides and compound scrolls.

use super::field::{
    Attractor, CurlNoise, Directional, Field, FieldExt, Mask, Sum, Vortex, along_curve,
    place_points, trace,
};
use super::layout::{
    Canvas, Layer, Layers, Occupancy, dir, leaf_style, place, ribbon_style, unit_or,
};
use crate::config::{Config, Preset};
use crate::geom::curl::organic_curve;
use crate::geom::fit::{fit_curve, join_g1};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::ribbon::WidthProfile;
use crate::geom::spiral::LogSpiral;
use crate::geom::{Composite, Curve, CurveExt, smoothstep};
use crate::motif::flower::{self, FlowerStyle};
use crate::motif::foliage::{LeafStyle, leaf_motif};
use crate::motif::ribbon::{RibbonParams, RibbonStyle, coiled, ribbon_motif};
use crate::motif::spiral::{SpiralParams, spiral_ribbon};
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, Rng, SeedTree};
use crate::scene::{Node, Scene, Style};
use anyhow::anyhow;
use kurbo::{BezPath, CubicBez, Point, Rect, Vec2};
use std::f64::consts::{PI, TAU};

/// Path budget of a full scene at detail 1.0.
pub const PATH_BUDGET: f64 = 55_000.0;

/// Build the scene of `config.preset`.
pub fn build(config: &Config) -> anyhow::Result<Scene> {
    let palette = Palette::by_name(&config.palette)
        .ok_or_else(|| anyhow!("unknown palette {:?}", config.palette))?;
    let (canvas, layers) = match config.preset {
        Preset::Botanica => super::botanica::compose(config, &palette),
        Preset::Bloom => bloom(config, &palette),
        Preset::Ribbons => ribbons(config, &palette),
    };
    let mut scene = Scene::new(config.width, config.height).with_background(palette.background);
    scene.push(canvas.finish(layers.into_node()));
    Ok(scene)
}

/// Path budget for `config`.
pub(crate) fn budget(config: &Config) -> usize {
    (PATH_BUDGET * config.detail.clamp(0.1, 4.0)) as usize
}

// ---------------------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------------------

/// A closed, smooth blob of radius about `r` around `c`.
pub(crate) fn blob(rng: &mut ChaCha8Rng, c: Point, r: f64, wobble: f64) -> BezPath {
    let n = 9;
    let phase = rng.uniform(0.0, TAU);
    let pts: Vec<Point> = (0..n)
        .map(|i| {
            let a = phase + TAU * i as f64 / n as f64;
            c + dir(a) * (r * (1.0 + rng.jitter(0.0, wobble)))
        })
        .collect();
    let mut path = BezPath::new();
    path.move_to(pts[0]);
    let k = 1.0 / 6.0;
    for i in 0..n {
        let (p0, p1, p2, p3) = (
            pts[(i + n - 1) % n],
            pts[i],
            pts[(i + 1) % n],
            pts[(i + 2) % n],
        );
        path.curve_to(p1 + (p2 - p0) * k, p2 - (p3 - p1) * k, p2);
    }
    path.close_path();
    path
}

/// Stack of translucent copies of a blob shrinking inwards: a soft-edged shape.
pub(crate) fn soft_shape(
    rng: &mut ChaCha8Rng,
    c: Point,
    r: f64,
    color: crate::scene::Color,
    alpha: f64,
) -> Node {
    let mut g = Node::empty();
    // Every copy shares one outline shape, scaled down.
    let shape = SeedTree::new(rng.next_u64());
    for i in 0..5 {
        let f = 1.0 - 0.17 * i as f64;
        g.push(Node::path(
            blob(&mut shape.rng(), c, r * f, 0.18),
            Style::fill(color.with_alpha(alpha)),
        ));
    }
    g
}

/// Tonal background: each side of the divider washed with its own hue (a few offset
/// copies soften the edge), then large soft shapes tinted by the region they sit in.
pub(crate) fn background(seed: &SeedTree, canvas: &Canvas, palette: &Palette, layers: &mut Layers) {
    let mut rng = seed.rng_for("background");
    let w = canvas.width;
    let mut g = Node::empty();
    let dark_bg = palette.background.to_hsl().2 < 0.4;
    let wash = if dark_bg { 0.07 } else { 0.06 };
    for (warm, t) in [(true, 0.75), (false, 0.7)] {
        let color = if warm {
            palette.warm_at(t)
        } else {
            palette.cool_at(t)
        };
        for k in 0..4 {
            let offset = (k as f64 - 1.5) * 0.018 * w;
            g.push(Node::path(
                canvas.side_path(warm, offset),
                Style::fill(color.with_alpha(wash)),
            ));
        }
    }
    let n = 7;
    for i in 0..n {
        let c = Point::new(
            rng.uniform(-0.1, 1.1) * w,
            canvas.height * (i as f64 + rng.uniform(0.1, 0.9)) / n as f64,
        );
        let r = canvas.unit() * rng.uniform(0.2, 0.38);
        let warm = canvas.is_warm(c);
        let color = if warm {
            palette.warm_at(rng.uniform(0.4, 0.8))
        } else {
            palette.cool_at(rng.uniform(0.35, 0.8))
        };
        g.push(soft_shape(&mut rng, c, r, color, 0.035));
    }
    layers.add(Layer::Background, g.with_id("tones"));
}

// ---------------------------------------------------------------------------------------
// Field and guides
// ---------------------------------------------------------------------------------------

/// A point the composition revolves around.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Focus {
    pub center: Point,
    pub radius: f64,
    /// Spin of the vortex around it: `+1` or `-1`.
    pub spin: f64,
}

/// The flow field: counter-rotating vortices winding into the foci, curl noise for
/// irregularity, a slight upward drift and a repeller around the opening (if any).
pub(crate) fn flow_field(
    seed: &SeedTree,
    canvas: &Canvas,
    foci: &[Focus],
    avoid: Option<(Point, f64)>,
) -> Sum {
    let u = canvas.unit();
    let mut parts: Vec<Box<dyn Field>> = Vec::new();
    for f in foci {
        parts.push(
            Vortex {
                center: f.center,
                strength: f.spin,
                radius: (2.4 * f.radius).max(0.22 * u),
                swirl: 0.3,
            }
            .boxed(),
        );
    }
    parts.push(CurlNoise::new(seed.noise_seed("curl"), 0.55 * u, 0.4).boxed());
    parts.push(Directional(Vec2::new(0.0, -0.28)).boxed());
    if let Some((c, r)) = avoid {
        parts.push(Attractor::repeller(c, 1.2, 1.15 * r).boxed());
    }
    Sum(parts)
}

/// Sub-curve of `curve` over its longest run of arc-length samples where `free` holds,
/// refitted; `None` if no run is at least `min_len` long.
pub(crate) fn longest_free_run(
    curve: &Composite,
    free: &dyn Fn(Point) -> bool,
    min_len: f64,
    accuracy: f64,
) -> Option<Composite> {
    let len = curve.length();
    if !(len.is_finite() && len >= min_len) {
        return None;
    }
    let n = ((len / (0.25 * min_len.max(1e-6))).ceil() as usize).clamp(16, 600);
    let samples = curve.sample_by_arclen(n);
    let (mut best, mut start) = ((0, 0), None);
    for (i, (_, p)) in samples.iter().enumerate() {
        if free(*p) {
            let s = *start.get_or_insert(i);
            if i - s > best.1 - best.0 {
                best = (s, i);
            }
        } else {
            start = None;
        }
    }
    let (a, b) = best;
    if b <= a || (b - a) as f64 / (n - 1) as f64 * len < min_len {
        return None;
    }
    if a == 0 && b == n - 1 {
        return Some(curve.clone());
    }
    let sub = curve.clone().sub(samples[a].0..samples[b].0);
    Composite::from_bezpath(&fit_curve(&sub, accuracy))
}

/// The longest stretch of `curve` whose total absolute turning stays within `max_turn`
/// radians, refitted; `None` if it is shorter than `min_len`.
pub(crate) fn limit_turning(
    curve: &Composite,
    max_turn: f64,
    min_len: f64,
    accuracy: f64,
) -> Option<Composite> {
    let n = 160;
    let s = curve.sample_by_arclen(n);
    // turn[i]: turning accumulated up to sample i.
    let mut turn = vec![0.0; n];
    for i in 1..n - 1 {
        let (a, b) = (s[i].1 - s[i - 1].1, s[i + 1].1 - s[i].1);
        turn[i + 1] = turn[i] + a.cross(b).atan2(a.dot(b)).abs();
    }
    if turn[n - 1] <= max_turn {
        return Some(curve.clone());
    }
    let (mut best, mut lo) = ((0, 0), 0);
    for hi in 0..n {
        while turn[hi] - turn[lo] > max_turn {
            lo += 1;
        }
        if hi - lo > best.1 - best.0 {
            best = (lo, hi);
        }
    }
    let len = curve.length() * (best.1 - best.0) as f64 / (n - 1) as f64;
    if len < min_len {
        return None;
    }
    let sub = curve.clone().sub(s[best.0].0..s[best.1].0);
    Composite::from_bezpath(&fit_curve(&sub, accuracy))
}

/// Streamlines of `field` from `starts`, longest first, each trimmed to its longest run
/// that keeps `spacing` from the curves accepted before it (and from `taken`) and that
/// `free` allows; at most `n` curves of at least `min_len`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_guides(
    field: &dyn Field,
    starts: &[Point],
    bounds: Rect,
    step: f64,
    max_len: f64,
    min_len: f64,
    spacing: f64,
    n: usize,
    taken: &[Point],
    free: &dyn Fn(Point) -> bool,
    accuracy: f64,
) -> Vec<Composite> {
    let mut traced: Vec<(f64, Composite)> = starts
        .iter()
        .filter_map(|&s| trace(field, s, step, max_len, bounds, true))
        .map(|c| (c.length(), c))
        .filter(|(l, _)| l.is_finite() && *l >= min_len)
        .collect();
    traced.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut occupied: Vec<Point> = taken.to_vec();
    let mut out = Vec::new();
    for (_, c) in traced {
        if out.len() >= n {
            break;
        }
        let ok = |p: Point| free(p) && occupied.iter().all(|q| (p - *q).hypot() >= spacing);
        let Some(g) = longest_free_run(&c, &ok, min_len, accuracy) else {
            continue;
        };
        let m = ((g.length() / (0.3 * spacing)).ceil() as usize).clamp(2, 2000);
        occupied.extend(g.sample_by_arclen(m).into_iter().map(|s| s.1));
        out.push(g);
    }
    out
}

/// Seed points for streamlines, denser where `mask` is high.
pub(crate) fn guide_starts(
    seed: &SeedTree,
    bounds: Rect,
    mask: &dyn Mask,
    min_dist: f64,
    n: usize,
) -> Vec<Point> {
    place_points(seed.child("guide-starts"), bounds, mask, &|_| min_dist, n)
}

// ---------------------------------------------------------------------------------------
// Ribbons and scrolls
// ---------------------------------------------------------------------------------------

/// Width-to-radius ratio keeping the turns of a log spiral apart (60 % of the touching
/// limit, as the spiral motifs do).
pub(crate) fn coil_ratio(turns: f64, end_ratio: f64) -> f64 {
    let q = end_ratio.clamp(1e-3, 0.95).powf(1.0 / turns.max(0.25));
    (0.6 * (1.0 - q) / (1.0 + q)).clamp(0.1, 0.4)
}

/// `c` traversed backwards.
pub(crate) fn reversed(c: &Composite) -> Composite {
    let segs: Vec<CubicBez> = c
        .segments()
        .iter()
        .rev()
        .map(|s| CubicBez::new(s.p3, s.p2, s.p1, s.p0))
        .collect();
    Composite::new_by_length(segs)
}

/// A spiral tail for the end of a curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tail {
    pub turns: f64,
    pub end_ratio: f64,
    /// Starting radius of the coil.
    pub radius: f64,
    pub clockwise: bool,
}

/// A tail at the end of `curve` whose coil starts as wide as `half_width`, curling the way
/// the curve already bends unless that would leave `bounds` or enter `blocked`; it shrinks
/// once if needed, and is `None` if no variant fits.
pub(crate) fn fit_tail(
    rng: &mut ChaCha8Rng,
    curve: &Composite,
    half_width: f64,
    bounds: Rect,
    blocked: &dyn Fn(Point) -> bool,
) -> Option<Tail> {
    let turns = rng.uniform(1.6, 2.6);
    let end_ratio = rng.uniform(0.08, 0.14);
    let c = coil_ratio(turns, end_ratio);
    let r0 = half_width / c * rng.uniform(0.95, 1.35);
    let end = curve.end();
    let n = curve.normal(1.0);
    let kappa = curve.curvature(1.0);
    let preferred = kappa < 0.0;
    let len = curve.length();
    let m = ((len / half_width.max(1e-6)) as usize).clamp(8, 400);
    let body: Vec<(f64, Point)> = curve
        .sample_by_arclen(m)
        .into_iter()
        .enumerate()
        .map(|(i, s)| (len * i as f64 / (m - 1) as f64, s.1))
        .collect();
    for r in [r0, 0.7 * r0, 0.45 * r0] {
        // The coil must also stay clear of the ribbon's own body, except along the last
        // stretch leading into it.
        let own: Vec<Point> = body
            .iter()
            .filter(|(s, _)| *s < len - 2.4 * r)
            .map(|(_, p)| *p)
            .collect();
        let clear =
            |p: Point| !blocked(p) && own.iter().all(|q| (p - *q).hypot() > 1.6 * half_width);
        for clockwise in [preferred, !preferred] {
            let side = if clockwise { -1.0 } else { 1.0 };
            let center = end + n * (side * r);
            let fits = bounds.inset(-1.05 * r).contains(center)
                && clear(center)
                && (0..12).all(|i| clear(center + dir(TAU * i as f64 / 12.0) * (1.1 * r)));
            if fits {
                return Some(Tail {
                    turns,
                    end_ratio,
                    radius: r,
                    clockwise,
                });
            }
        }
    }
    None
}

/// A tail for the end of `curve`, cutting the curve back step by step until one fits:
/// a ribbon should roll up rather than end in a spike. Returns the (possibly shortened)
/// curve and the tail.
pub(crate) fn tail_or_trim(
    rng: &mut ChaCha8Rng,
    curve: &Composite,
    half_width: f64,
    bounds: Rect,
    blocked: &dyn Fn(Point) -> bool,
    accuracy: f64,
) -> (Composite, Option<Tail>) {
    for cut in [1.0, 0.86, 0.72, 0.6] {
        let c = if cut < 1.0 {
            match Composite::from_bezpath(&fit_curve(&curve.clone().sub(0.0..cut), accuracy)) {
                Some(c) => c,
                None => continue,
            }
        } else {
            curve.clone()
        };
        if let Some(t) = fit_tail(rng, &c, half_width, bounds, blocked) {
            return (c, Some(t));
        }
    }
    (curve.clone(), None)
}

/// [`tail_or_trim`] at the start of `curve`.
pub(crate) fn tail_or_trim_start(
    rng: &mut ChaCha8Rng,
    curve: &Composite,
    half_width: f64,
    bounds: Rect,
    blocked: &dyn Fn(Point) -> bool,
    accuracy: f64,
) -> (Composite, Option<Tail>) {
    let (c, t) = tail_or_trim(rng, &reversed(curve), half_width, bounds, blocked, accuracy);
    (reversed(&c), t)
}

fn tail_path(curve: &Composite, tail: &Tail, accuracy: f64) -> BezPath {
    let k = LogSpiral::tightness_for(tail.turns, tail.end_ratio);
    let spiral = LogSpiral::from_start(
        curve.end(),
        curve.tangent(1.0),
        tail.radius,
        tail.turns,
        k,
        tail.clockwise,
    );
    fit_curve(&spiral, accuracy)
}

/// A guide with optional spiral tails at its start and end, as one G1 centreline.
pub(crate) fn with_tails(
    guide: &Composite,
    start: Option<Tail>,
    end: Option<Tail>,
    accuracy: f64,
) -> Composite {
    let mut parts = Vec::new();
    if let Some(t) = start {
        parts.push(tail_path(&reversed(guide), &t, accuracy).reverse_subpaths());
    }
    parts.push(guide.to_bezpath_exact());
    if let Some(t) = end {
        parts.push(tail_path(guide, &t, accuracy));
    }
    join_g1(&parts).unwrap_or_else(|| guide.clone())
}

/// Engraved ribbon of half-width `half_width` along `centerline`, its width capped inside
/// coils so turns keep their gap; ends without a coil taper to a point.
pub(crate) fn scroll_ribbon(
    seed: &SeedTree,
    centerline: &Composite,
    half_width: f64,
    tapers: (f64, f64),
    stripes: usize,
    style: &RibbonStyle,
    accuracy: f64,
) -> Node {
    let mut rng = seed.rng_for("scroll-ribbon");
    let body = WidthProfile::tapered(half_width, tapers.0, tapers.1, 0.7).with_noise(
        seed.noise_seed("width"),
        0.12,
        3.0,
    );
    let profile = coiled(body, centerline, coil_ratio(2.0, 0.11));
    let params = RibbonParams {
        stripes,
        hatch: stripes > 0,
        lit_side: if rng.chance(0.5) { 1.0 } else { -1.0 },
        ..RibbonParams::symmetric(profile)
    };
    ribbon_motif(seed, centerline, &params, style, accuracy)
}

/// A ribbon grown along `guide`: trimmed to its longest stretch clear of `taken`, each end
/// rolled into a spiral tail (shortening the guide until one fits) unless `hidden` says the
/// end disappears under a flower, coloured by the region it mostly runs in. Returns the
/// node and its centreline, or `None` if too little of the guide is free.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_ribbon(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    guide: &Composite,
    half_width: f64,
    taken: &Occupancy,
    hidden: &dyn Fn(Point) -> bool,
    accuracy: f64,
) -> Option<(Node, Composite)> {
    let u = canvas.unit();
    let mut rng = seed.rng_for("grow-ribbon");
    let free = |p: Point| !taken.contains(p);
    let g = longest_free_run(guide, &free, 0.15 * u, accuracy)?;
    // A ribbon sweeps at most about three quarters of a turn: streamlines orbiting a
    // vortex would overlap themselves.
    let g = limit_turning(&g, 1.3 * PI, 0.15 * u, accuracy)?;
    let short = g.length() < 0.35 * u;
    let bounds = canvas.inner(0.03 * u);
    let blocked = |p: Point| taken.contains(p) || !canvas.rect().contains(p);
    let (g, end) = if hidden(g.end()) {
        (g, None)
    } else {
        tail_or_trim(&mut rng, &g, half_width, bounds, &blocked, accuracy)
    };
    // Short ribbons roll up at one end only (two coils on a short stem read as a bar).
    let (g, start) = if hidden(g.start()) || (short && end.is_some()) {
        (g, None)
    } else {
        tail_or_trim_start(&mut rng, &g, half_width, bounds, &blocked, accuracy)
    };
    let centerline = with_tails(&g, start, end, accuracy);
    let warmth = mean_warmth(canvas, &centerline);
    let style = ribbon_style(palette, warmth, rng.uniform(0.05, 0.45));
    let tapers = (
        if start.is_some() { 0.02 } else { 0.14 },
        if end.is_some() { 0.02 } else { 0.14 },
    );
    let node = scroll_ribbon(seed, &centerline, half_width, tapers, 2, &style, accuracy);
    Some((node, centerline))
}

/// Start radius of the coil of [`spiral_ribbon`] for `params` and `length` (mirrors the
/// motif's own construction, so a continuation can match its width).
fn coil_start_radius(p: &SpiralParams, length: f64) -> f64 {
    let k = LogSpiral::tightness_for(p.turns, p.end_ratio);
    let sk = (1.0 + k * k).sqrt();
    let coil_len = (1.0 - p.lead) * length;
    if k > 1e-9 {
        coil_len * k / (sk * (1.0 - p.end_ratio))
    } else {
        coil_len / (TAU * p.turns)
    }
}

/// A compound scroll: a ribbon that leaves `start`, bends through a loose hook, and at the
/// hook's end (its [`MotifEnd`](crate::motif::spiral::MotifEnd)) hands over to a smaller,
/// tighter coil of the opposite hand — an S of two nested spirals.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compound_scroll(
    seed: &SeedTree,
    start: Point,
    direction: Vec2,
    length: f64,
    clockwise: bool,
    style: &RibbonStyle,
    second: &RibbonStyle,
    accuracy: f64,
) -> Node {
    let mut rng = seed.rng_for("compound-scroll");
    let hook = SpiralParams {
        end_ratio: rng.uniform(0.4, 0.5),
        lead: rng.uniform(0.35, 0.5),
        width: rng.uniform(0.14, 0.18),
        stripes: 2,
        ..SpiralParams::new(rng.uniform(0.45, 0.6), clockwise)
    };
    let (first, end) = spiral_ribbon(
        &seed.child("hook"),
        start,
        direction,
        length,
        &hook,
        style,
        accuracy,
    );
    let mut curl = SpiralParams {
        open_start: true,
        lead: 0.15,
        stripes: 1,
        ..SpiralParams::new(rng.uniform(1.8, 2.4), !clockwise)
    };
    curl.width = coil_ratio(curl.turns, curl.end_ratio);
    // Match the open start of the curl to the hook's width near its end.
    let r_end = coil_start_radius(&hook, length) * hook.end_ratio;
    let w_end = hook.width * r_end * 1.15;
    let r0 = w_end / (1.22 * curl.width);
    let k = LogSpiral::tightness_for(curl.turns, curl.end_ratio);
    let sk = (1.0 + k * k).sqrt();
    let l2 = r0 * sk * (1.0 - curl.end_ratio) / (k.max(1e-6) * (1.0 - curl.lead));
    let back = end.point - end.tangent * (0.03 * l2);
    let (second_node, _) = spiral_ribbon(
        &seed.child("curl"),
        back,
        end.tangent,
        l2.min(1.2 * length),
        &curl,
        second,
        accuracy,
    );
    Node::group(vec![first, second_node])
}

// ---------------------------------------------------------------------------------------
// Flowers and leaves
// ---------------------------------------------------------------------------------------

/// Flower `species` (or any species if unknown) of outer radius `radius` centred at `at`.
pub(crate) fn flower_at(
    seed: &SeedTree,
    species: &str,
    at: Point,
    radius: f64,
    rotation: f64,
    style: &FlowerStyle,
    accuracy: f64,
) -> Node {
    let all = flower::variants(&seed.child("species"));
    let params = all
        .iter()
        .find(|(n, _)| n == species)
        .or_else(|| all.first())
        .map(|(_, p)| p.clone());
    let Some(params) = params else {
        return Node::empty();
    };
    let scale = radius / params.outer_radius().max(1e-6);
    let node = flower::flower(seed, &params, style, accuracy / scale);
    place(node, at, rotation, scale)
}

/// An engraved leaf from `base` towards `direction`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn leaf_at(
    seed: &SeedTree,
    base: Point,
    direction: Vec2,
    length: f64,
    aspect: f64,
    params: LeafParams,
    style: &LeafStyle,
    accuracy: f64,
) -> Node {
    let leaf = Leaf::from_base(base, direction, length, aspect * length, params);
    leaf_motif(seed, &leaf, style, accuracy)
}

/// A random leaf shape for a region with its typical aspect.
pub(crate) fn leaf_shape(rng: &mut ChaCha8Rng, warm: bool) -> (LeafShape, f64) {
    let warm_shapes = [
        (LeafShape::Acanthus { lobes: 4 }, 0.55),
        (LeafShape::Acanthus { lobes: 5 }, 0.5),
        (LeafShape::Ovate, 0.5),
        (LeafShape::Lanceolate, 0.26),
    ];
    let cool_shapes = [
        (LeafShape::Serrated { teeth: 12 }, 0.36),
        (LeafShape::Fern { pinnae: 14 }, 0.34),
        (LeafShape::Willow, 0.14),
        (LeafShape::Cordate, 0.6),
        (LeafShape::Lanceolate, 0.26),
    ];
    *rng.pick(if warm { &warm_shapes } else { &cool_shapes })
}

/// `v` rotated by `angle`.
pub(crate) fn rotate(v: Vec2, angle: f64) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// Leaves radiating behind a flower at `center` of radius `r`.
pub(crate) fn leaf_collar(
    seed: &SeedTree,
    center: Point,
    r: f64,
    count: usize,
    warm: bool,
    palette: &Palette,
    accuracy: f64,
) -> Node {
    let mut rng = seed.rng_for("collar");
    let mut g = Node::empty();
    let phase = rng.uniform(0.0, TAU);
    for i in 0..count {
        let a = phase + TAU * (i as f64 + rng.jitter(0.0, 0.18)) / count as f64;
        let d = dir(a);
        // Leaves of the flower's own region, one in three from the other as an accent.
        let own = i % 3 != 1;
        let leaf_warm = warm == own;
        let (shape, aspect) = leaf_shape(&mut rng, leaf_warm);
        let curl = rng.jitter(0.0, 0.5);
        let params = LeafParams::new(shape)
            .with_tip_curl(curl)
            .jittered(&mut rng, 0.5);
        let len = r * rng.uniform(1.0, 1.45);
        let tone = if own {
            rng.uniform(0.0, 0.3)
        } else {
            rng.uniform(0.2, 0.5)
        };
        let style = leaf_style(palette, f64::from(u8::from(leaf_warm)), tone);
        g.push(leaf_at(
            &seed.index(i as u64),
            center + d * (0.3 * r),
            d,
            len,
            aspect,
            params,
            &style,
            accuracy,
        ));
    }
    g
}

// ---------------------------------------------------------------------------------------
// Bloom
// ---------------------------------------------------------------------------------------

const BLOOM_SPECIES: [&str; 7] = ["rose", "peony", "lotus", "poppy", "lily", "aster", "cosmos"];

/// One large focal flower (random species) centred on a collar of 3–5 leaves, with two
/// scroll ribbons rising from below in mirror image, over the tonal background.
fn bloom(config: &Config, palette: &Palette) -> (Canvas, Layers) {
    let seed = SeedTree::new(config.seed).child("bloom");
    let canvas = Canvas::new(&seed.child("canvas"), config.width, config.height);
    let mut layers = Layers::new(budget(config));
    let mut rng = seed.rng_for("layout");
    let (w, h, u) = (canvas.width, canvas.height, canvas.unit());
    let acc = config.accuracy;
    background(&seed, &canvas, palette, &mut layers);

    let warm = rng.chance(0.6);
    let center = Point::new(w * 0.5, h * rng.uniform(0.36, 0.42));
    let r = u * rng.uniform(0.25, 0.3);
    // A soft halo behind the flower.
    let halo = if warm {
        palette.warm_at(0.85)
    } else {
        palette.cool_at(0.8)
    };
    layers.add(
        Layer::Background,
        soft_shape(&mut rng, center, 1.55 * r, halo, 0.06),
    );

    // A stem from the bottom edge carrying a few leaves.
    let foot = Point::new(w * 0.5 + rng.jitter(0.0, 0.03 * w), 1.03 * h);
    let stem_curve = organic_curve(&mut rng, foot, center + Vec2::new(0.0, 0.3 * r), 0.12, 0.05);
    let stem_style = ribbon_style(palette, f64::from(u8::from(!warm)), 0.1);
    layers.add(
        Layer::Plants,
        scroll_ribbon(
            &seed.child("stem"),
            &stem_curve,
            0.011 * u,
            (0.0, 0.1),
            1,
            &stem_style,
            acc,
        ),
    );
    let mut stem_leaves = Node::empty();
    for (j, (_, p, tg)) in along_curve(&stem_curve, 0.16 * u, 0.3, seed.child("stem-leaves"))
        .into_iter()
        .enumerate()
    {
        if (p - center).hypot() < 1.2 * r || p.y > 0.97 * h {
            continue;
        }
        let side = if j % 2 == 0 { 1.0 } else { -1.0 };
        let mut lrng = seed.child("stem-leaf").index(j as u64).rng();
        let (shape, aspect) = leaf_shape(&mut lrng, !warm);
        let params = LeafParams::new(shape)
            .with_tip_curl(-side * 0.35)
            .jittered(&mut lrng, 0.4);
        let style = leaf_style(palette, f64::from(u8::from(!warm)), lrng.uniform(0.1, 0.5));
        stem_leaves.push(leaf_at(
            &seed.child("stem-leaf").index(j as u64),
            p,
            rotate(tg, side * 0.8),
            0.17 * u * lrng.uniform(0.85, 1.1),
            aspect,
            params,
            &style,
            acc,
        ));
    }
    layers.add(Layer::Plants, stem_leaves);

    // Two scrolls rising from the foot on either side, bowing outwards and rolling up
    // beside the flower: warm on the warm side, cool on the cool one.
    let flower_zone = |p: Point| (p - center).hypot() < 1.1 * r;
    let mut taken = Occupancy::new();
    taken.add_circle(center, 1.05 * r);
    for (i, side) in [-1.0, 1.0].into_iter().enumerate() {
        let start = Point::new(w * 0.5 + side * 0.14 * u, 1.0 * h);
        let end = Point::new(
            (w * 0.5 + side * (1.3 * r + 0.08 * u)).clamp(0.12 * w, 0.88 * w),
            center.y + 0.35 * r,
        );
        let mut srng = seed.child("scroll").index(i as u64).rng();
        let curve = organic_curve(&mut srng, start, end, 0.2 * side, -0.45 * side);
        let hw = 0.026 * u;
        let blocked = |p: Point| flower_zone(p) || taken.contains(p);
        let (curve, tail) =
            tail_or_trim(&mut srng, &curve, hw, canvas.inner(0.03 * u), &blocked, acc);
        let centerline = with_tails(&curve, None, tail, acc);
        let style = ribbon_style(palette, f64::from(u8::from(side < 0.0)), 0.25);
        let node = scroll_ribbon(
            &seed.child("scroll").index(i as u64),
            &centerline,
            hw,
            (0.0, 0.02),
            2,
            &style,
            acc,
        );
        if layers.add(Layer::Ribbons, node) {
            taken.add_curve(&centerline, hw);
        }
    }

    let count = 3 + (rng.uniform(0.0, 3.0) as usize);
    layers.add(
        Layer::Plants,
        leaf_collar(&seed.child("collar"), center, r, count, warm, palette, acc),
    );
    let species = *rng.pick(&BLOOM_SPECIES);
    let style = FlowerStyle::from_palette(palette, warm, rng.uniform(0.25, 0.5));
    let rot = rng.uniform(0.0, TAU);
    layers.add(
        Layer::Foreground,
        flower_at(&seed.child("flower"), species, center, r, rot, &style, acc),
    );
    (canvas, layers)
}

// ---------------------------------------------------------------------------------------
// Ribbons
// ---------------------------------------------------------------------------------------

/// Ribbons and spirals only: the divider as a broad whiplash band, then ribbons along the
/// longest streamlines of a two-vortex field, coloured by the side of the divider they
/// mostly run in, with spiral tails at free ends and compound scrolls at a few others.
fn ribbons(config: &Config, palette: &Palette) -> (Canvas, Layers) {
    let seed = SeedTree::new(config.seed).child("ribbons");
    let canvas = Canvas::new(&seed.child("canvas"), config.width, config.height);
    let mut layers = Layers::new(budget(config));
    let mut rng = seed.rng_for("layout");
    let (w, h, u) = (canvas.width, canvas.height, canvas.unit());
    let acc = config.accuracy;
    background(&seed, &canvas, palette, &mut layers);

    let foci = [
        Focus {
            center: Point::new(w * rng.uniform(0.22, 0.32), h * rng.uniform(0.58, 0.68)),
            radius: 0.12 * u,
            spin: 1.0,
        },
        Focus {
            center: Point::new(w * rng.uniform(0.68, 0.78), h * rng.uniform(0.3, 0.4)),
            radius: 0.12 * u,
            spin: -1.0,
        },
    ];
    let field = flow_field(&seed, &canvas, &foci, None);
    let mut occ = Occupancy::new();
    let divider = divider_band(&seed, &canvas, palette, &mut layers, &mut occ, acc);
    let bounds = canvas.inner(0.035 * u);
    let mask = super::field::Constant(1.0);
    let starts = guide_starts(&seed, bounds, &mask, 0.1 * u, 40);
    let n = 6 + (rng.uniform(0.0, 4.0) as usize);
    let taken: Vec<Point> = divider
        .sample_by_arclen(200)
        .into_iter()
        .map(|s| s.1)
        .collect();
    let guides = select_guides(
        &field,
        &starts,
        bounds,
        0.005 * u,
        1.4 * h,
        0.3 * u,
        0.1 * u,
        n,
        &taken,
        &|_| true,
        acc,
    );
    let longest = guides.first().map_or(1.0, |g| g.length());
    for (i, g) in guides.iter().enumerate() {
        let k = seed.child("ribbon").index(i as u64);
        let hw = (0.03 * u * (g.length() / longest).sqrt()).max(0.008 * u);
        if let Some((node, centerline)) =
            grow_ribbon(&k, &canvas, palette, g, hw, &occ, &|_| false, acc)
            && layers.add(Layer::Ribbons, node)
        {
            occ.add_curve(&centerline, hw + 0.012 * u);
        }
    }
    (canvas, layers)
}

/// Mean warmth of a curve, pushed towards 0 or 1 so a ribbon commits to one region.
pub(crate) fn mean_warmth(canvas: &Canvas, curve: &Composite) -> f64 {
    let s = curve.sample_by_arclen(48);
    let m = s.iter().map(|(_, p)| canvas.warmth(*p)).sum::<f64>() / s.len() as f64;
    smoothstep((m - 0.5) * 2.5 + 0.5)
}

/// The divider drawn as the dominant band: a broad warm ribbon from the top edge rolling
/// into a large coil low on the canvas, accompanied by a narrower cool ribbon. Returns the
/// drawn centreline.
pub(crate) fn divider_band(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    layers: &mut Layers,
    occ: &mut Occupancy,
    accuracy: f64,
) -> Composite {
    let mut rng = seed.rng_for("divider-band");
    let u = canvas.unit();
    let cut = rng.uniform(0.78, 0.86);
    let body = Composite::from_bezpath(&fit_curve(&canvas.divider.clone().sub(0.0..cut), accuracy))
        .unwrap_or_else(|| canvas.divider.clone());
    let hw = 0.024 * u;
    let tail = fit_tail(&mut rng, &body, hw, canvas.inner(0.02 * u), &|_| false);
    let centerline = with_tails(&body, None, tail, accuracy);
    let style = ribbon_style(palette, 1.0, rng.uniform(0.05, 0.2));
    let node = scroll_ribbon(
        &seed.child("divider"),
        &centerline,
        hw,
        (0.0, 0.02),
        2,
        &style,
        accuracy,
    );
    layers.add(Layer::Ribbons, node);
    occ.add_curve(&centerline, hw + 0.012 * u);

    // Companion band on the cool side: it leaves the top edge alongside the divider, keeps
    // clear of what is already placed (the opening) and rolls up where it parts from it.
    let b = rng.uniform(0.5, 0.66);
    let off = 3.2 * hw;
    let pts: Vec<Point> = (0..=40)
        .map(|i| {
            let t = b * i as f64 / 40.0;
            let p = body.point(t);
            let n = body.normal(t);
            // Push towards the cool side (+x in the warm-left frame).
            let s = if n.x >= 0.0 { 1.0 } else { -1.0 };
            p + n * (s * off)
        })
        .collect();
    let free = |p: Point| !occ.contains(p);
    let comp = Composite::through_points(&pts, 0.5)
        .and_then(|c| Composite::from_bezpath(&fit_curve(&c, accuracy)))
        .and_then(|c| longest_free_run(&c, &free, 0.2 * u, accuracy));
    if let Some(comp) = comp {
        let chw = 0.45 * hw;
        let blocked = |p: Point| occ.contains(p);
        let (comp, tail) = tail_or_trim(
            &mut rng,
            &comp,
            chw,
            canvas.inner(0.02 * u),
            &blocked,
            accuracy,
        );
        let comp = with_tails(&comp, None, tail, accuracy);
        let style = ribbon_style(palette, 0.0, rng.uniform(0.15, 0.4));
        let node = scroll_ribbon(
            &seed.child("companion"),
            &comp,
            chw,
            (0.0, 0.02),
            1,
            &style,
            accuracy,
        );
        layers.add(Layer::Ribbons, node);
        occ.add_curve(&comp, chw + 0.01 * u);
    }
    centerline
}

/// The upward direction `(0, -1)` leaning towards `target` by `amount` (a vector weight).
pub(crate) fn lean_towards(from: Point, target: Point, amount: f64) -> Vec2 {
    let to = unit_or(target - from, Vec2::new(0.0, -1.0));
    unit_or(Vec2::new(0.0, -1.0) + to * amount, Vec2::new(0.0, -1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(preset: Preset, seed: u64) -> Config {
        Config {
            seed,
            preset,
            width: 400.0,
            height: 600.0,
            detail: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn presets_build_finite_scenes() {
        for preset in [Preset::Bloom, Preset::Ribbons] {
            let s = build(&cfg(preset, 3)).unwrap();
            assert!(s.root.is_finite());
            assert!(s.root.path_count() > 10, "{preset:?}");
        }
    }

    #[test]
    fn free_run_trims_blocked_parts() {
        let c = Composite::new(vec![CubicBez::new(
            (0.0, 0.0),
            (100.0, 0.0),
            (200.0, 0.0),
            (300.0, 0.0),
        )]);
        let r = longest_free_run(&c, &|p| p.x < 120.0 || p.x > 140.0, 20.0, 0.1).unwrap();
        assert!(r.end().x > 290.0 && r.start().x > 135.0);
        assert!(longest_free_run(&c, &|_| false, 20.0, 0.1).is_none());
    }
}
