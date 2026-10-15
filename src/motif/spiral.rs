//! Spiral motifs: ribbons that roll into logarithmic and Archimedean coils, fern croziers,
//! volutes and S-shaped double spirals.
//!
//! A coil is built as a lead-in whose curvature grows smoothly from zero to the curvature of
//! a [`LogSpiral`] that continues it (so the join is G2, not merely G1), and the ribbon's
//! half-width inside the coil is a fixed fraction of the local radius (see
//! [`coiled`](super::ribbon::coiled)): every turn is a scaled copy of the previous one, the
//! width tapers evenly into the eye and the turns keep an even gap. Widths are finally
//! fitted so the outline never self-intersects.
//!
//! The builders [`spiral_ribbon`] and [`crozier`] start at a given point and direction, so
//! compositions can attach them to the end of a guide curve; they return the motif and the
//! [`MotifEnd`] at the spiral's eye. [`variants`] assembles named demonstration spirals.

use super::foliage::{LeafStyle, VeinWidths};
use super::ribbon::{
    Builder, RibbonParams, RibbonStyle, Sheet, build_all, coil_ratio, coiled, coin, reversed,
    to_composite,
};
use crate::geom::curl::whiplash;
use crate::geom::fit::{fit_curve, join_g1, stub, unit_or};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::ribbon::{Ribbon, WidthProfile, ease};
use crate::geom::spiral::{ArchimedeanSpiral, LogSpiral, spiral_tail};
use crate::geom::{Composite, Curve, EPS, Surface, lerp, smoothstep};
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, SeedTree};
use crate::scene::{Node, Style};
use kurbo::{Affine, Circle, CubicBez, Point, Shape, Vec2};
use std::f64::consts::{PI, TAU};

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// Where a motif ends and a continuation could attach: the end of its centreline and the
/// unit direction of travel there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotifEnd {
    pub point: Point,
    pub tangent: Vec2,
}

/// Shape of a spiral ribbon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpiralParams {
    /// Revolutions of the coil (clamped to `[0.25, 12]`).
    pub turns: f64,
    /// Eye radius divided by the radius where the coil starts (clamped to `[0.01, 0.9]`).
    pub end_ratio: f64,
    /// Fraction of the length spent in the lead-in before the coil (clamped to `[0, 0.9]`).
    pub lead: f64,
    /// Exponent of the lead-in's curvature ramp `κ₀·x^p` (larger keeps the lead straight
    /// longer before it bends into the coil; clamped to `[0.5, 8]`).
    pub lead_bend: f64,
    /// Half-width as a fraction of the local coil radius (clamped to `[0.02, 0.6]`).
    pub width: f64,
    /// Curl towards the right of `direction` instead of the left.
    pub clockwise: bool,
    /// Contour stripes per side.
    pub stripes: usize,
    /// Start broad and close in a short, sharp point instead of a long taper.
    pub open_start: bool,
}

impl SpiralParams {
    /// A coil of `turns` revolutions closing to 12 % of its starting radius, a quarter of
    /// the length as lead-in, the widest width that keeps turns comfortably apart.
    pub fn new(turns: f64, clockwise: bool) -> SpiralParams {
        SpiralParams {
            turns,
            end_ratio: 0.12,
            lead: 0.25,
            lead_bend: 2.0,
            width: coil_ratio(turns, 0.12),
            clockwise,
            stripes: 2,
            open_start: false,
        }
    }
}

/// A coil centreline: lead-in plus log spiral, with the spiral's starting radius.
struct Coil {
    curve: Composite,
    spiral: LogSpiral,
    r0: f64,
    /// Arc length of the lead-in.
    lead_len: f64,
}

/// Lead-in of length `lead·length` whose curvature ramps from 0 to that of a log spiral of
/// `turns` revolutions and radius ratio `end_ratio`, followed by that spiral; the total
/// arc length is `length`. The spiral's starting radius follows from its arc length
/// `s = r₀·√(1+k²)/k·(1 − end_ratio)`.
fn coil_centerline(start: Point, dir: Vec2, length: f64, p: &SpiralParams, accuracy: f64) -> Coil {
    let turns = finite_or(p.turns, 1.5).clamp(0.25, 12.0);
    let ratio = finite_or(p.end_ratio, 0.12).clamp(0.01, 0.9);
    let lead = finite_or(p.lead, 0.2).clamp(0.0, 0.9);
    let dir = unit_or(dir, Vec2::new(1.0, 0.0));
    let k = LogSpiral::tightness_for(turns, ratio);
    let sk = (1.0 + k * k).sqrt();
    let lead_len = lead * length;
    let coil_len = length - lead_len;
    let r0 = if k > 1e-9 {
        coil_len * k / (sk * (1.0 - ratio))
    } else {
        coil_len / (TAU * turns)
    };
    let sigma = if p.clockwise { -1.0 } else { 1.0 };
    let kappa0 = sigma / (r0 * sk);
    let mut parts = Vec::new();
    let (end, tangent) = if lead_len > 1e-6 * length {
        let pw = finite_or(p.lead_bend, 2.0).clamp(0.5, 8.0);
        let (lead_curve, tangent) = intrinsic(
            start,
            dir,
            lead_len,
            |s| kappa0 * (s / lead_len).clamp(0.0, 1.0).powf(pw),
            accuracy,
        );
        let end = lead_curve.end();
        parts.push(lead_curve.to_bezpath_exact());
        (end, tangent)
    } else {
        (start, dir)
    };
    let spiral = LogSpiral::from_start(end, tangent, r0, turns, k, p.clockwise);
    parts.push(fit_curve(&spiral, accuracy));
    let curve = join_g1(&parts).unwrap_or_else(|| stub(start, dir));
    Coil {
        curve,
        spiral,
        r0,
        lead_len,
    }
}

/// The curve with signed curvature `kappa(s)` (positive turns towards the left normal) for
/// arc length `s ∈ [0, length]`, starting at `start` along `dir`; also returns the exact
/// end tangent. Integrated with the trapezoid rule in 4096 steps, joined by cubic Hermite
/// segments with exact tangents and refitted to `accuracy`.
fn intrinsic(
    start: Point,
    dir: Vec2,
    length: f64,
    kappa: impl Fn(f64) -> f64,
    accuracy: f64,
) -> (Composite, Vec2) {
    const STEPS: usize = 4096;
    const STRIDE: usize = 16;
    let dir = unit_or(dir, Vec2::new(1.0, 0.0));
    if !(length.is_finite() && length > EPS && start.is_finite()) {
        return (stub(start, dir), dir);
    }
    let ds = length / STEPS as f64;
    let k = |s: f64| finite_or(kappa(s), 0.0);
    let mut theta = dir.y.atan2(dir.x);
    let mut pos = start;
    let mut k_prev = k(0.0);
    let mut knots = vec![(start, dir)];
    for i in 0..STEPS {
        let k1 = k((i + 1) as f64 * ds);
        let next = theta + 0.5 * (k_prev + k1) * ds;
        pos += (Vec2::from_angle(theta) + Vec2::from_angle(next)) * (0.5 * ds);
        theta = next;
        k_prev = k1;
        if (i + 1) % STRIDE == 0 {
            knots.push((pos, Vec2::from_angle(theta)));
        }
    }
    let h = ds * STRIDE as f64 / 3.0;
    let segs = knots
        .windows(2)
        .map(|w| CubicBez::new(w[0].0, w[0].0 + w[0].1 * h, w[1].0 - w[1].1 * h, w[1].0))
        .collect();
    let raw = Composite::new_by_length(segs);
    let fitted = Composite::from_bezpath(&fit_curve(&raw, accuracy)).unwrap_or(raw);
    (fitted, Vec2::from_angle(theta))
}

/// A built spiral ribbon: node, fitted ribbon and coil geometry.
struct Built {
    node: Node,
    ribbon: Ribbon<Composite>,
    coil: Coil,
}

fn end_of(c: &Composite) -> MotifEnd {
    MotifEnd {
        point: c.end(),
        tangent: c.tangent(1.0),
    }
}

fn spiral_ribbon_in(
    sh: &mut Sheet,
    seed: &SeedTree,
    start: Point,
    dir: Vec2,
    length: f64,
    p: &SpiralParams,
    style: &RibbonStyle,
) -> Built {
    let coil = coil_centerline(start, dir, length, p, sh.acc);
    let c = finite_or(p.width, 0.25).clamp(0.02, 0.6);
    let body = if p.open_start {
        // A broad start closing in a short, sharp point.
        WidthProfile::tapered(c * coil.r0, 0.12, 0.05, 0.9)
    } else {
        WidthProfile::tapered(c * coil.r0, 0.12, 0.05, 0.9)
    };
    let profile = coiled(body, &coil.curve, c);
    let params = RibbonParams {
        stripes: p.stripes,
        lit_side: if p.clockwise { -1.0 } else { 1.0 },
        ..RibbonParams::symmetric(profile)
    };
    let (node, ribbon) = sh.ribbon("spiral-ribbon", seed, coil.curve.clone(), &params, style);
    Built { node, ribbon, coil }
}

/// A ribbon that leaves `start` along `direction`, bends smoothly and rolls into a log
/// spiral; `length` is the centreline's arc length. The width is proportional to the coil
/// radius, so it decreases evenly towards the eye. Returns the motif and the eye
/// ([`MotifEnd`]). Invalid `length` gives an empty node.
pub fn spiral_ribbon(
    seed: &SeedTree,
    start: Point,
    direction: Vec2,
    length: f64,
    params: &SpiralParams,
    style: &RibbonStyle,
    accuracy: f64,
) -> (Node, MotifEnd) {
    if !(length.is_finite() && length > 0.0 && start.is_finite()) {
        return (Node::empty(), stub_end(start, direction));
    }
    let mut sh = Sheet::new(length, accuracy, false);
    let b = spiral_ribbon_in(&mut sh, seed, start, direction, length, params, style);
    let end = end_of(&b.coil.curve);
    (b.node, end)
}

fn stub_end(start: Point, direction: Vec2) -> MotifEnd {
    MotifEnd {
        point: if start.is_finite() {
            start
        } else {
            Point::ZERO
        },
        tangent: unit_or(direction, Vec2::new(1.0, 0.0)),
    }
}

/// A fern crozier growing from `start` along `direction`: a banded stem that bends into a
/// tight coil of 2–2.6 turns, with discrete rounded pinnae (each with its own vein) along
/// its outer edge that shrink smoothly with the coil and lean towards the eye. `length` is
/// the centreline's arc length; shape details come from `seed`. Returns the motif and the
/// eye. Invalid `length` gives an empty node.
pub fn crozier(
    seed: &SeedTree,
    start: Point,
    direction: Vec2,
    length: f64,
    clockwise: bool,
    style: &RibbonStyle,
    accuracy: f64,
) -> (Node, MotifEnd) {
    if !(length.is_finite() && length > 0.0 && start.is_finite()) {
        return (Node::empty(), stub_end(start, direction));
    }
    let mut sh = Sheet::new(length, accuracy, false);
    crozier_in(&mut sh, seed, start, direction, length, clockwise, style)
}

fn crozier_in(
    sh: &mut Sheet,
    seed: &SeedTree,
    start: Point,
    dir: Vec2,
    length: f64,
    clockwise: bool,
    style: &RibbonStyle,
) -> (Node, MotifEnd) {
    let mut rng = seed.rng_for("crozier");
    let p = SpiralParams {
        end_ratio: rng.uniform(0.07, 0.1),
        lead: rng.uniform(0.38, 0.48),
        lead_bend: 2.5,
        ..SpiralParams::new(rng.uniform(2.0, 2.6), clockwise)
    };
    let coil = coil_centerline(start, dir, length, &p, sh.acc);
    let c = coil_ratio(p.turns, p.end_ratio);
    let w0 = c * coil.r0;
    let profile = coiled(
        // a rounded foot (power 0.5) at the start
        WidthProfile::tapered(1.1 * w0, 0.0, 0.04, 0.8).map(|u, w| w * ease(u / 0.025, 0.5)),
        &coil.curve,
        c,
    );
    let params = RibbonParams {
        stripes: 1,
        lit_side: if clockwise { -1.0 } else { 1.0 },
        ..RibbonParams::symmetric(profile)
    };
    let end = end_of(&coil.curve);
    let (stipe, ribbon) = sh.ribbon("crozier", seed, coil.curve, &params, style);

    // Pinnae along the outer edge from just before the coil to near the eye, each sized
    // and spaced by the local stem width, so they shrink smoothly into the coil. They grow
    // from under the stem and lean towards the eye.
    let outer = if clockwise { 1.0 } else { -1.0 };
    let pinna_style = LeafStyle {
        fill: style.fill,
        fill_light: style.highlight.mix(style.fill, 0.35),
        ink: style.ink,
        vein: style.stripe,
        widths: VeinWidths {
            outline: 0.035,
            midrib: 0.06,
            lateral: 0.0,
            veinlet: 0.0,
            hatch: 0.02,
        },
    };
    let u_start = (coil.lead_len / length - 0.12).max(0.12);
    let (u_end, size) = (0.86, rng.uniform(2.2, 2.5));
    let lean = rng.uniform(0.7, 0.85);
    let total = ribbon.length();
    let cl = ribbon.centerline();
    let mut pinnae = Node::empty();
    let mut u = u_start;
    let mut i = 0;
    while u < u_end && i < 200 {
        let hw = ribbon.half_width(u);
        if !(hw.is_finite() && hw > EPS && total > EPS) {
            break;
        }
        let fade = lerp(0.55, 1.0, smoothstep((u - u_start) / 0.08))
            * lerp(1.0, 0.5, smoothstep((u - 0.7) / (u_end - 0.7)));
        let t = ribbon.t_at(u);
        let a = lean + rng.jitter(0.0, 0.06);
        let d = cl.normal(t) * (outer * a.cos()) + cl.tangent(t) * a.sin();
        let len = size * hw * fade;
        let leaf = Leaf::from_base(
            ribbon.point(u, 0.55 * outer),
            d,
            len,
            len * rng.uniform(0.7, 0.8),
            LeafParams::new(LeafShape::Ovate)
                .with_points(0.3, 0.0)
                .with_tip_curl(-outer * rng.uniform(0.15, 0.3)),
        );
        pinnae.push(sh.foliage("crozier-pinna", &seed.index(i), leaf, &pinna_style));
        u += 1.05 * hw * fade / total;
        i += 1;
    }
    (Node::group(vec![pinnae, stipe]), end)
}

const BUILDERS: [(&str, Builder); 7] = [
    ("log-ribbon", log_ribbon),
    ("crozier", crozier_variant),
    ("volute", volute),
    ("archimedean", archimedean),
    ("whip-spiral", whip_spiral),
    ("double-spiral", double_spiral),
    ("spiral-with-leaves", spiral_with_leaves),
];

/// Named, visually distinct demonstration motifs, each fitting a `size` × `size` box
/// centred at the origin.
pub fn variants(
    seed: &SeedTree,
    palette: &Palette,
    size: f64,
    accuracy: f64,
) -> Vec<(String, Node)> {
    build_all(
        &BUILDERS,
        seed,
        palette,
        &mut Sheet::new(size, accuracy, false),
    )
}

fn random_dir(rng: &mut ChaCha8Rng) -> Vec2 {
    Vec2::from_angle(rng.uniform(0.0, TAU))
}

/// A horn-like ribbon whose centreline is almost entirely a log spiral, open at the wide end.
fn log_ribbon(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let turns = rng.uniform(2.0, 2.5);
    let end_ratio = rng.uniform(0.06, 0.1);
    let p = SpiralParams {
        end_ratio,
        lead: rng.uniform(0.04, 0.1),
        lead_bend: 1.5,
        width: coil_ratio(turns, end_ratio) * rng.uniform(0.8, 1.0),
        stripes: 2,
        open_start: true,
        ..SpiralParams::new(turns, rng.chance(0.5))
    };
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.15, 0.5));
    let dir = random_dir(&mut rng);
    spiral_ribbon_in(sh, seed, Point::ORIGIN, dir, 2.2 * sh.size, &p, &style).node
}

fn crozier_variant(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let style = RibbonStyle::from_palette(palette, false, rng.uniform(0.25, 0.55));
    let dir = Vec2::from_angle(-0.5 * PI + rng.uniform(-0.3, 0.3));
    let cw = rng.chance(0.5);
    let start = Point::new(0.0, 0.5 * sh.size);
    crozier_in(sh, seed, start, dir, 2.0 * sh.size, cw, &style).0
}

/// An Ionic volute: a broad band with three contour lines per side that rolls into nearly
/// three turns around an open eye; its bands run on into the eye and end in a ball.
fn volute(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let turns = rng.uniform(2.6, 3.0);
    let end_ratio = rng.uniform(0.17, 0.22);
    let c = coil_ratio(turns, end_ratio) * 4.0 / 3.0;
    let p = SpiralParams {
        end_ratio,
        lead: rng.uniform(0.14, 0.2),
        lead_bend: 3.0,
        width: c,
        stripes: 3,
        open_start: true,
        ..SpiralParams::new(turns, rng.chance(0.5))
    };
    let coil = coil_centerline(
        Point::ORIGIN,
        random_dir(&mut rng),
        2.4 * sh.size,
        &p,
        sh.acc,
    );
    let profile = coiled(
        WidthProfile::tapered(c * coil.r0, 0.08, 0.0, 0.8),
        &coil.curve,
        c,
    );
    let params = RibbonParams {
        stripes: 3,
        stripe_inset: 0.12,
        lit_side: if p.clockwise { -1.0 } else { 1.0 },
        ..RibbonParams::symmetric(profile)
    };
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.3, 0.6));
    let (band, ribbon) = sh.ribbon("volute", seed, coil.curve, &params, &style);
    // A ball terminal on the blunt end of the band, lit from outside the coil.
    let tip = ribbon.point(1.0, 0.0);
    let r = 1.45 * ribbon.half_width(1.0);
    let out = unit_or(tip - coil.spiral.center(), Vec2::new(1.0, 0.0));
    let ball = Circle::new(tip, r).to_path(sh.acc);
    let shine = Circle::new(tip + out * (0.3 * r), 0.45 * r).to_path(sh.acc);
    let mut root = Node::group(vec![band]);
    root.push(Node::path(ball.clone(), Style::fill(style.shade)));
    root.push(Node::path(shine, Style::fill(style.highlight)).clipped(ball.clone()));
    root.push(Node::path(
        ball,
        Style::stroke(style.ink, 2.0 * style.outline_width * r),
    ));
    root
}

/// Evenly spaced coils: an Archimedean spiral ribbon with tapered ends.
fn archimedean(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let turns = rng.uniform(2.6, 3.4);
    let pitch = 0.12 * sh.size;
    let spiral = ArchimedeanSpiral::from_center(
        Point::ORIGIN,
        rng.uniform(0.45, 0.6) * pitch,
        rng.uniform(0.0, TAU),
        turns,
        pitch,
        rng.chance(0.5),
    );
    let curve = to_composite(&spiral, sh.acc);
    let profile = WidthProfile::tapered(
        pitch * rng.uniform(0.22, 0.28),
        rng.uniform(0.08, 0.14),
        rng.uniform(0.08, 0.14),
        1.0,
    );
    let params = RibbonParams {
        stripes: 1,
        stripe_inset: 0.3,
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.2, 0.5));
    sh.ribbon("archimedean", seed, curve, &params, &style).0
}

/// A long whiplash that ends in a tight two-and-a-half-turn spiral.
fn whip_spiral(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let dir = random_dir(&mut rng);
    let body = whiplash(&mut rng, Point::ORIGIN, dir, 1.15 * l, 0.0);
    let turns = 2.5;
    let end_ratio = rng.uniform(0.07, 0.1);
    let curve = spiral_tail(
        &body,
        turns,
        LogSpiral::tightness_for(turns, end_ratio),
        body.curvature(1.0) < 0.0,
        sh.acc,
    );
    let base = WidthProfile::tapered(l * rng.uniform(0.046, 0.054), 0.3, 0.04, 1.3);
    let profile = coiled(base, &curve, coil_ratio(turns, end_ratio));
    let params = RibbonParams {
        stripes: 2,
        ..RibbonParams::symmetric(profile)
    };
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.1, 0.4));
    let node = sh.ribbon("whip-spiral", seed, curve, &params, &style).0;
    if rng.chance(0.5) {
        node.with_transform(Affine::FLIP_X)
    } else {
        node
    }
}

/// Two coils of opposite handedness joined by one S-shaped ribbon, point-symmetric about
/// the inflection in the middle, widest there.
fn double_spiral(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let end_ratio = 0.12;
    let half = |turns: f64, dir: Vec2, acc: f64| {
        let p = SpiralParams {
            end_ratio,
            lead: 0.2,
            lead_bend: 1.0,
            ..SpiralParams::new(turns, false)
        };
        coil_centerline(Point::ORIGIN, dir, 1.1 * l, &p, acc)
    };
    let (t1, t2) = (rng.uniform(1.7, 2.2), rng.uniform(1.7, 2.2));
    let d = random_dir(&mut rng);
    let a = half(t1, d, sh.acc);
    let b = half(t2, -d, sh.acc);
    // `b` traversed from its eye back to the middle, then `a` out to its eye.
    let curve = join_g1(&[
        reversed(&b.curve).to_bezpath_exact(),
        a.curve.to_bezpath_exact(),
    ])
    .unwrap_or(a.curve);
    let c = coil_ratio(t1.min(t2), end_ratio);
    let base = WidthProfile::tapered(1.15 * c * a.r0.max(b.r0), 0.05, 0.05, 0.9);
    let profile = coiled(base, &curve, c);
    let params = RibbonParams {
        stripes: 2,
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.2, 0.5));
    sh.ribbon("double-spiral", seed, curve, &params, &style).0
}

/// A spiral ribbon with two or three small leaves sprouting from the outside of its
/// outermost turn, shrinking towards the coil.
fn spiral_with_leaves(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let turns = rng.uniform(1.4, 1.8);
    let p = SpiralParams {
        end_ratio: 0.15,
        lead: rng.uniform(0.35, 0.45),
        lead_bend: 2.0,
        width: 0.65 * coil_ratio(turns, 0.15),
        stripes: 2,
        ..SpiralParams::new(turns, rng.chance(0.5))
    };
    let warm = rng.chance(0.5);
    let style = RibbonStyle::from_palette(palette, warm, rng.uniform(0.2, 0.5));
    let length = 1.9 * sh.size;
    let b = spiral_ribbon_in(
        sh,
        seed,
        Point::ORIGIN,
        random_dir(&mut rng),
        length,
        &p,
        &style,
    );
    // Leaves only where nothing lies outside: the lead-in and the first half turn.
    let k = b.coil.spiral.growth().abs();
    let sk = (1.0 + k * k).sqrt();
    let q = p.end_ratio.powf(1.0 / turns);
    let first_turn = if k > 1e-9 {
        b.coil.r0 * sk / k * (1.0 - q)
    } else {
        TAU * b.coil.r0
    };
    let u_max = ((b.coil.lead_len + 0.45 * first_turn) / length).min(0.6);
    let outer = if p.clockwise { 1.0 } else { -1.0 };
    let count = if rng.chance(0.5) { 2 } else { 3 };
    let leaf_style = LeafStyle::from_palette(palette, !warm, rng.uniform(0.3, 0.55));
    let c = b.ribbon.centerline();
    let mut root = Node::empty();
    for i in 0..count {
        let f = i as f64 / (count - 1) as f64;
        let u = lerp(0.12, u_max, f) + rng.jitter(0.0, 0.02);
        let t = b.ribbon.t_at(u);
        let a = rng.uniform(0.6, 0.85);
        let dir = c.normal(t) * (outer * a.cos()) + c.tangent(t) * a.sin();
        let len = b.coil.r0 * lerp(1.5, 0.95, f) * rng.uniform(0.9, 1.1);
        let shape = if i % 2 == 0 {
            LeafShape::Lanceolate
        } else {
            LeafShape::Ovate
        };
        let leaf = Leaf::from_base(
            b.ribbon.point(u, 0.4 * outer),
            dir,
            len,
            len * rng.uniform(0.38, 0.48),
            LeafParams::new(shape).with_tip_curl(-outer * rng.uniform(0.15, 0.35)),
        );
        root.push(sh.foliage("spiral-leaf", &seed.index(i as u64), leaf, &leaf_style));
    }
    root.push(b.node);
    root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::CurveExt;
    use crate::geom::fit::max_kink;
    use crate::motif::ribbon::chord_curvature;
    use crate::motif::ribbon::tests::assert_well_formed;
    use std::collections::HashSet;

    fn palette() -> Palette {
        Palette::by_name("verdigris").unwrap()
    }

    #[test]
    fn variants_build_cleanly_with_valid_ribbons() {
        for seed in 1..=5 {
            let size = 400.0;
            let mut sheet = Sheet::new(size, 0.05, true);
            let vs = build_all(&BUILDERS, &SeedTree::new(seed), &palette(), &mut sheet);
            assert!(vs.len() >= 6);
            let names: HashSet<_> = vs.iter().map(|(n, _)| n.clone()).collect();
            assert_eq!(names.len(), vs.len());
            for (name, node) in &vs {
                assert_well_formed(name, node, size);
            }
            for (label, r) in sheet.reports() {
                assert!(r.is_valid(), "seed {seed} {label}: {r:?}");
            }
        }
    }

    #[test]
    fn variants_are_reproducible_and_seed_dependent() {
        let p = palette();
        let a = variants(&SeedTree::new(3), &p, 300.0, 0.1);
        assert_eq!(a, variants(&SeedTree::new(3), &p, 300.0, 0.1));
        let b = variants(&SeedTree::new(4), &p, 300.0, 0.1);
        for ((name, x), (_, y)) in a.iter().zip(&b) {
            assert_ne!(x, y, "{name} ignores the seed");
        }
    }

    #[test]
    fn extreme_sizes_do_not_panic() {
        for size in [1.0, 10_000.0] {
            let vs = variants(&SeedTree::new(2), &palette(), size, 0.05);
            assert!(vs.len() >= 6);
            for (name, node) in &vs {
                assert_well_formed(name, node, size);
            }
        }
        assert!(variants(&SeedTree::new(2), &palette(), f64::NAN, 0.05).is_empty());
    }

    #[test]
    fn coil_is_smooth_and_closes_at_the_eye() {
        for cw in [false, true] {
            let p = SpiralParams::new(2.5, cw);
            let coil = coil_centerline(Point::new(5.0, 5.0), Vec2::new(0.0, 1.0), 300.0, &p, 0.02);
            assert!(
                max_kink(&coil.curve) < 1e-6,
                "kink {}",
                max_kink(&coil.curve)
            );
            assert!((coil.curve.length() - 300.0).abs() < 3.0);
            assert!((coil.curve.start() - Point::new(5.0, 5.0)).hypot() < 1e-9);
            // the eye is the spiral's end, one end_ratio of the starting radius from the pole
            let r = (coil.curve.end() - coil.spiral.center()).hypot();
            assert!((r / (p.end_ratio * coil.r0) - 1.0).abs() < 0.02, "{r}");
            // the lead's curvature reaches the spiral's (G2): compare just before and after
            let s0 = coil.lead_len / coil.curve.length();
            let ribbon = Ribbon::symmetric(coil.curve.clone(), WidthProfile::zero());
            let before = chord_curvature(&ribbon, s0 - 0.01, 0.004);
            let after = chord_curvature(&ribbon, s0 + 0.01, 0.004);
            assert!((before / after - 1.0).abs() < 0.15, "{before} {after}");
            assert_eq!(before > 0.0, !cw);
        }
    }

    #[test]
    fn builders_attach_and_report_the_eye() {
        let style = RibbonStyle::from_palette(&palette(), true, 0.4);
        let seed = SeedTree::new(11);
        let start = Point::new(100.0, 50.0);
        let dir = Vec2::new(1.0, 1.0);
        let (node, end) = spiral_ribbon(
            &seed,
            start,
            dir,
            400.0,
            &SpiralParams::new(2.0, false),
            &style,
            0.05,
        );
        assert!(node.is_finite() && node.path_count() > 0);
        assert!((end.tangent.hypot() - 1.0).abs() < 1e-9);
        let bb = node.bounding_box().unwrap();
        assert!(bb.inflate(1.0, 1.0).contains(start) && bb.contains(end.point));
        let (node, end) = crozier(&seed, start, dir, 400.0, true, &style, 0.05);
        assert!(node.is_finite() && node.path_count() > 0);
        assert!(node.bounding_box().unwrap().contains(end.point));
        for bad in [0.0, f64::NAN, -3.0] {
            let (n, e) = crozier(&seed, start, dir, bad, false, &style, 0.05);
            assert_eq!(n.path_count(), 0);
            assert_eq!(e.point, start);
        }
    }
}
