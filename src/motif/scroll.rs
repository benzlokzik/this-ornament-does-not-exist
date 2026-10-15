//! Broad acanthus scrolls: wide, curling leaf-scrolls whose midrib sweeps through a C or an
//! S and rolls into a logarithmic spiral.
//!
//! The midrib is an [`organic_curve`] bowed towards the side the scroll curls to, continued
//! by a [`spiral_tail`]. The blade is a [`Ribbon`] around it: a broad body (widest
//! half-width [`ScrollParams::width`] of the length) with a rounded base, narrowing into the
//! coil, where its width is capped by the local radius of curvature (see [`coiled`]) so the
//! turns keep their gap. The outer (convex) edge of the body is cut into 3–6 rounded lobes
//! that shrink towards the tip; the ribbon's lean tips them forward like acanthus lobes.
//!
//! It is engraved in layers, all clipped by the outline: the region tone, alternate tonal
//! stripes between nested contour lines that follow the midrib, a lighter band along the
//! lit outer side, a darker band and fine hatching on the shadowed inner (concave) side,
//! then the contour lines and an ink outline. Line widths are fractions of the widest
//! half-width, so a scroll reads the same at any size.

use crate::geom::fit::fit_points;
use crate::geom::ribbon::{Ribbon, RibbonReport, WidthProfile, ease};
use crate::geom::spiral::{LogSpiral, spiral_tail};
use crate::geom::{Composite, Curve, CurveExt, EPS, Surface, lerp, smoothstep};
use crate::motif::ribbon::{clean, coil_ratio, coiled, fit_to_box, motif_accuracy};
use crate::ornament::{HatchOptions, Shade, band, hatch, stripes};
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, SeedTree};
use crate::scene::{Color, LineCap, Node, Style};
use kurbo::{Affine, BezPath, CubicBez, Point, Vec2};
use std::f64::consts::PI;

/// Radius ratio between the end and the start of the coil.
const END_RATIO: f64 = 0.13;

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// Shape and ornament of one scroll.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollParams {
    /// Arc length of the midrib, base to spiral eye.
    pub length: f64,
    /// Widest half-width of the blade as a fraction of [`length`](Self::length) (clamped
    /// to `[0.01, 0.3]`; 0.12–0.2 for a broad acanthus scroll, about 0.05 for a curl).
    pub width: f64,
    /// Turns of the spiral the scroll rolls into (clamped to `[0.3, 3]`).
    pub turns: f64,
    /// Lobes on the outer edge of the body: 0 for a smooth blade, else clamped to `3..=7`.
    pub lobes: u32,
    /// Curl direction: `false` curls towards the left normal of the start direction (for
    /// the local start direction `+x` that is `+y`), `true` towards the right normal.
    pub clockwise: bool,
    /// Counter-swing of the body in `[-1, 1]`: 0 is a plain C sweeping into the coil,
    /// larger magnitudes an S whose first bow swings the other way.
    pub sway: f64,
    /// Nested contour lines along the midrib (clamped to `0..=8`); the stripes between
    /// them alternate in tone.
    pub stripes: usize,
    /// Hatch the shadowed inner side.
    pub hatch: bool,
}

impl ScrollParams {
    /// A broad acanthus scroll: four lobes, 1.3 turns, four contour stripes, hatched.
    pub fn acanthus(length: f64, clockwise: bool) -> ScrollParams {
        ScrollParams {
            length,
            width: 0.16,
            turns: 1.3,
            lobes: 4,
            clockwise,
            sway: 0.3,
            stripes: 4,
            hatch: true,
        }
    }

    /// A small smooth curl: narrow, two turns, two contour stripes, no hatching.
    pub fn curl(length: f64, clockwise: bool) -> ScrollParams {
        ScrollParams {
            length,
            width: 0.06,
            turns: 2.2,
            lobes: 0,
            clockwise,
            sway: 0.15,
            stripes: 2,
            hatch: false,
        }
    }

    /// Natural variation drawn from `rng` (`amount`: 0 = unchanged, 1 = typical spread).
    pub fn jittered(mut self, rng: &mut ChaCha8Rng, amount: f64) -> ScrollParams {
        let k = finite_or(amount, 0.0).clamp(0.0, 2.0);
        self.width *= 1.0 + rng.jitter(0.0, 0.18 * k);
        self.turns *= 1.0 + rng.jitter(0.0, 0.25 * k);
        self.sway += rng.jitter(0.0, 0.35 * k);
        self
    }

    fn sanitized(self) -> ScrollParams {
        ScrollParams {
            length: finite_or(self.length, 0.0).max(0.0),
            width: finite_or(self.width, 0.15).clamp(0.01, 0.3),
            turns: finite_or(self.turns, 1.2).clamp(0.3, 3.0),
            lobes: if self.lobes == 0 {
                0
            } else {
                self.lobes.clamp(3, 7)
            },
            clockwise: self.clockwise,
            sway: finite_or(self.sway, 0.0).clamp(-1.0, 1.0),
            stripes: self.stripes.min(8),
            hatch: self.hatch,
        }
    }
}

/// Colours and relative line widths of an engraved scroll.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScrollStyle {
    /// Body tone.
    pub fill: Color,
    /// Tone of every other stripe between the contour lines.
    pub band: Color,
    /// Light band along the lit outer side.
    pub light: Color,
    /// Shadow band on the inner side.
    pub shade: Color,
    /// Contour lines and hatching.
    pub line: Color,
    /// Outline ink.
    pub ink: Color,
    /// Outline width as a fraction of the widest half-width.
    pub outline_width: f64,
    /// Contour line width as a fraction of the widest half-width.
    pub stripe_width: f64,
    /// Hatch line width as a fraction of the widest half-width.
    pub hatch_width: f64,
}

impl ScrollStyle {
    /// Derive a style from the warm (`warm = true`) or cool ramp of `palette` at `t ∈ [0, 1]`
    /// (dark → light): the body from the ramp, the stripes a step lighter, the shadow and
    /// lines darker steps mixed with ink.
    pub fn from_palette(palette: &Palette, warm: bool, t: f64) -> ScrollStyle {
        let t = finite_or(t, 0.5).clamp(0.0, 1.0);
        let ramp = |x: f64| {
            if warm {
                palette.warm_at(x)
            } else {
                palette.cool_at(x)
            }
        };
        ScrollStyle {
            fill: ramp(lerp(0.12, 0.6, t)),
            band: ramp(lerp(0.3, 0.8, t)),
            light: ramp(lerp(0.6, 1.0, t)).mix(palette.light, 0.3),
            shade: ramp(lerp(0.0, 0.2, t)).mix(palette.ink, 0.3),
            line: ramp(lerp(0.0, 0.12, t)).mix(palette.ink, 0.5),
            ink: palette.ink,
            outline_width: 0.035,
            stripe_width: 0.02,
            hatch_width: 0.02,
        }
    }

    /// `self` mixed towards `other` by `k`.
    pub fn mix(&self, other: &ScrollStyle, k: f64) -> ScrollStyle {
        ScrollStyle {
            fill: self.fill.mix(other.fill, k),
            band: self.band.mix(other.band, k),
            light: self.light.mix(other.light, k),
            shade: self.shade.mix(other.shade, k),
            line: self.line.mix(other.line, k),
            ink: self.ink.mix(other.ink, k),
            ..*self
        }
    }

    /// A receding version: every tone pulled towards the body by `1 − contrast` and the
    /// whole style mixed towards `toward` by `amount` (e.g. a dark tone for depth).
    pub fn receding(&self, contrast: f64, toward: Color, amount: f64) -> ScrollStyle {
        let c = 1.0 - finite_or(contrast, 1.0).clamp(0.0, 1.0);
        let f = self.fill;
        let pull = |x: Color| x.mix(f, c).mix(toward, amount);
        ScrollStyle {
            fill: f.mix(toward, amount),
            band: pull(self.band),
            light: pull(self.light),
            shade: pull(self.shade),
            line: pull(self.line),
            ink: pull(self.ink),
            ..*self
        }
    }
}

/// A built scroll in its local frame (base at the origin, starting along `+x`).
pub(crate) struct Built {
    /// The engraved scroll.
    pub(crate) node: Node,
    /// Its outline.
    pub(crate) outline: BezPath,
    /// The midrib point at the widest part of the blade, and the midrib tangent there.
    pub(crate) anchor: (Point, Vec2),
    /// Points on the outer (lobed) edge at `u = 0.1, 0.2, …, 0.9` with the unit outward
    /// direction there.
    pub(crate) outer: Vec<(Point, Vec2)>,
    /// Validity of the outline, if it was checked.
    pub(crate) report: Option<RibbonReport>,
}

/// A broad acanthus scroll starting at the origin along `+x` (see the module docs). An
/// invalid or zero length gives an empty group.
pub fn scroll(seed: &SeedTree, params: &ScrollParams, style: &ScrollStyle, accuracy: f64) -> Node {
    build(seed, params, style, accuracy, false).map_or_else(Node::empty, |b| b.node)
}

/// The midrib in the local frame (unit tangent `+x` at the origin) and the arc-length
/// fraction where its coil begins.
///
/// The body is defined by its curvature along its normalised arc length `x`:
/// `κ·L = c₀ + c₁·x^1.5`, with `c₀ ≤ 0` the counter-swing of an S and `c₀ + c₁` the end
/// curvature (radius about a fifth of the body). Rising monotonically, it sweeps through a
/// C (or an S) into the coil without a kink; [`spiral_tail`] continues it curvature-
/// continuously.
fn midrib(rng: &mut ChaCha8Rng, p: &ScrollParams, accuracy: f64) -> Option<(Composite, f64)> {
    const N: usize = 256;
    let side = if p.clockwise { -1.0 } else { 1.0 };
    let c0 = -2.0 * p.sway * rng.uniform(0.8, 1.2);
    let c1 = rng.uniform(4.2, 6.0) - c0;
    // Built at unit body length, then scaled to the requested total length.
    let ds = 1.0 / N as f64;
    let mut pts = Vec::with_capacity(N + 1);
    let (mut q, mut theta) = (Point::ORIGIN, 0.0_f64);
    pts.push(q);
    for i in 0..N {
        let x = (i as f64 + 0.5) * ds;
        let kappa = side * (c0 + c1 * x.powf(1.5));
        let mid = theta + 0.5 * kappa * ds;
        q += Vec2::from_angle(mid) * ds;
        theta += kappa * ds;
        pts.push(q);
    }
    let acc = (accuracy / p.length.max(EPS)).clamp(1e-5, 2e-3);
    let head = Composite::from_bezpath(&fit_points(&pts, 0.25 * acc))?;
    let k = LogSpiral::tightness_for(p.turns, END_RATIO);
    let curve = spiral_tail(&head, p.turns, k, p.clockwise, acc);
    let (head_len, total) = (head.length(), curve.length());
    if !(total.is_finite() && total > EPS && head_len.is_finite()) {
        return None;
    }
    let t = Affine::scale(p.length / total);
    let segs: Vec<CubicBez> = curve
        .segments()
        .iter()
        .map(|s| CubicBez::new(t * s.p0, t * s.p1, t * s.p2, t * s.p3))
        .collect();
    if segs.is_empty() {
        return None;
    }
    Some((
        Composite::new_by_length(segs),
        (head_len / total).clamp(0.2, 0.95),
    ))
}

/// One rounded lobe on `f ∈ [0, 1)`: a half-ellipse whose crest sits past the middle, so
/// the lobe leans towards the tip; neighbouring lobes meet in a narrow notch.
fn lobe(f: f64) -> f64 {
    let x = 2.0 * f.clamp(0.0, 1.0).powf(1.35) - 1.0;
    (1.0 - x * x).max(0.0).sqrt()
}

/// Width factor of `n` lobes spread over `u ∈ [a, b]`, shorter towards the tip, faded in
/// and out at both ends of the span.
fn lobes(n: u32, a: f64, b: f64, u: f64) -> f64 {
    if n == 0 || b <= a {
        return 1.0;
    }
    let x = (u - a) / (b - a);
    if !(0.0..=1.0).contains(&x) {
        return 1.0;
    }
    let f = (f64::from(n) * x.powf(1.2)).fract();
    let depth = lerp(0.42, 0.28, x);
    let amp = smoothstep(x / 0.05) * smoothstep((1.0 - x) / 0.05);
    1.0 - amp * depth * (1.0 - lobe(f))
}

/// The scroll's blade around its midrib, narrowed until its outline is clean.
fn blade(p: &ScrollParams, curve: Composite, coil: f64, accuracy: f64) -> Ribbon<Composite> {
    let w = p.width * p.length;
    let n = p.lobes;
    let fade = (coil + 0.12).min(0.9);
    let body = WidthProfile::custom(move |u| {
        let base = ease(u / 0.2, 0.75);
        let narrow = 1.0 - 0.4 * smoothstep((u - 0.3) / (fade - 0.3).max(0.05));
        let tip = ease((1.0 - u) / 0.04, 0.9);
        w * base * narrow * tip
    });
    let ratio = coil_ratio(p.turns, END_RATIO);
    let span_end = (coil + 0.04).min(0.85);
    // The coil cap takes over only where the midrib rolls up, so it never flattens lobes.
    let capped = |profile: WidthProfile| {
        let cap = coiled(profile.clone(), &curve, ratio);
        WidthProfile::custom(move |u| {
            let k = smoothstep((u - (coil - 0.12)) / 0.12);
            lerp(profile.eval(u), cap.eval(u), k)
        })
    };
    let outer = capped(
        body.clone()
            .map(move |u, x| x * lobes(n, 0.14, span_end, u)),
    );
    let inner = capped(body.scale(0.78));
    let (left, right) = if p.clockwise {
        (outer, inner)
    } else {
        (inner, outer)
    };
    let lean = if n > 0 { 0.1 } else { 0.0 };
    let mut ribbon = Ribbon::new(curve, left, right).with_lean(lean);
    ribbon.fit_width(accuracy);
    ribbon
}

/// Build a scroll; `check` also returns the validity report of its outline.
pub(crate) fn build(
    seed: &SeedTree,
    params: &ScrollParams,
    style: &ScrollStyle,
    accuracy: f64,
    check: bool,
) -> Option<Built> {
    let p = params.sanitized();
    if p.length <= EPS || !p.length.is_finite() {
        return None;
    }
    let acc = motif_accuracy(accuracy, p.length);
    let mut rng = seed.rng_for("scroll");
    let (curve, coil) = midrib(&mut rng, &p, acc)?;
    let ribbon = blade(&p, curve, coil, acc);
    let outline = clean(&ribbon.outline(acc), acc);
    if outline.elements().len() < 3 {
        return None;
    }
    let report = check.then(|| ribbon.check(acc));
    // Widest place of the blade (the anchor the composition places scrolls by).
    let (mut wmax, mut u_max) = (0.0, 0.3);
    for i in 0..=64 {
        let u = i as f64 / 64.0;
        let (a, b) = ribbon.widths_at(u);
        if a + b > wmax {
            (wmax, u_max) = (a + b, u);
        }
    }
    let half = 0.5 * wmax;
    if !(half.is_finite() && half > EPS) {
        return None;
    }
    let anchor = (
        ribbon.point(u_max, 0.0),
        ribbon.centerline().tangent(ribbon.t_at(u_max)),
    );
    let side = if p.clockwise { 1.0 } else { -1.0 };
    let outer = (1..=9)
        .map(|i| {
            let u = i as f64 / 10.0;
            let (c, e) = (ribbon.point(u, 0.0), ribbon.point(u, side));
            let n = ribbon.centerline().normal(ribbon.t_at(u)) * side;
            (
                e,
                if (e - c).hypot() > EPS {
                    (e - c).normalize()
                } else {
                    n
                },
            )
        })
        .collect();
    let node = engrave(seed, &ribbon, &p, style, half, &outline, acc);
    Some(Built {
        node,
        outline,
        anchor,
        outer,
        report,
    })
}

/// Push `paths` merged into one path node.
fn push_merged(group: &mut Node, paths: Vec<BezPath>, style: Style, accuracy: f64) {
    let mut merged = BezPath::new();
    for p in paths {
        merged.extend(clean(&p, accuracy));
    }
    if !merged.elements().is_empty() {
        group.push(Node::path(merged, style));
    }
}

/// The engraved layers of a scroll blade (see the module docs).
fn engrave(
    seed: &SeedTree,
    ribbon: &Ribbon<Composite>,
    p: &ScrollParams,
    style: &ScrollStyle,
    half: f64,
    outline: &BezPath,
    acc: f64,
) -> Node {
    // The inner (concave) side of the curl is in shadow.
    let inner = if p.clockwise { -1.0 } else { 1.0 };
    let mut root = Node::empty();
    root.push(Node::path(outline.clone(), Style::fill(style.fill)));
    let mut details = Node::empty();
    // Alternate stripes between the nested contour levels.
    let n = p.stripes;
    let levels: Vec<f64> = (0..=n + 1)
        .map(|k| -1.0 + 2.0 * k as f64 / (n + 1) as f64)
        .collect();
    let stripes_fill: Vec<BezPath> = levels
        .windows(2)
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, w)| {
            let (lo, hi) = (w[0], w[1]);
            band(ribbon, 0.0, 1.0, move |_| lo, move |_| hi, acc)
        })
        .collect();
    push_merged(
        &mut details,
        stripes_fill,
        Style::fill(style.band).with_opacity(0.7),
        acc,
    );
    // A light band along the lit outer side.
    let c = -inner * 0.5;
    push_merged(
        &mut details,
        vec![band(ribbon, 0.0, 1.0, |_| c - 0.16, |_| c + 0.16, acc)],
        Style::fill(style.light).with_opacity(0.5),
        acc,
    );
    // A shadow band on the inner side, darkest at the edge.
    for (depth, alpha) in [(0.45, 0.35), (0.2, 0.45)] {
        let (lo, hi) = if inner > 0.0 {
            (1.0 - depth, 1.0)
        } else {
            (-1.0, -1.0 + depth)
        };
        push_merged(
            &mut details,
            vec![band(ribbon, 0.0, 1.0, move |_| lo, move |_| hi, acc)],
            Style::fill(style.shade).with_opacity(alpha),
            acc,
        );
    }
    if p.hatch {
        let opts = HatchOptions {
            spacing: 0.055 * half,
            angle: 72.0,
            min_len: 0.08 * half,
            jitter: 0.35,
            taper: 0.1,
            shade: Shade::Edge {
                side: inner,
                width: 0.45,
            },
        };
        push_merged(
            &mut details,
            hatch(ribbon, &opts, &seed.child("hatch"), acc),
            Style::stroke(style.line, style.hatch_width * half).with_cap(LineCap::Round),
            acc,
        );
    }
    if n > 0 {
        let inner_levels = &levels[1..=n];
        push_merged(
            &mut details,
            stripes(ribbon, inner_levels, acc),
            Style::stroke(style.line, style.stripe_width * half).with_cap(LineCap::Round),
            acc,
        );
    }
    root.push(details.clipped(outline.clone()));
    root.push(Node::path(
        outline.clone(),
        Style::stroke(style.ink, style.outline_width * half),
    ));
    root
}

// ---------------------------------------------------------------------------------------
// Demonstration variants
// ---------------------------------------------------------------------------------------

/// Records validity reports of the scrolls a variant builds (in tests).
struct Sheet {
    acc: f64,
    reports: Reports,
}

impl Sheet {
    fn scroll(
        &mut self,
        label: &str,
        seed: &SeedTree,
        params: &ScrollParams,
        style: &ScrollStyle,
    ) -> Option<Built> {
        let b = build(seed, params, style, self.acc, true)?;
        if let Some(r) = b.report {
            self.reports.push((label.to_string(), r));
        }
        Some(b)
    }
}

type Builder = fn(&SeedTree, &Palette, f64, &mut Sheet) -> Node;

/// Validity reports of the scrolls a sheet built, by label.
type Reports = Vec<(String, RibbonReport)>;

const BUILDERS: [(&str, Builder); 4] = [
    ("acanthus", single),
    ("opposed", opposed),
    ("curled", with_curls),
    ("curl", small_curl),
];

fn coin(rng: &mut ChaCha8Rng) -> bool {
    rng.chance(0.5)
}

/// One broad acanthus scroll.
fn single(seed: &SeedTree, palette: &Palette, size: f64, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let cw = coin(&mut rng);
    let params = ScrollParams {
        lobes: 4 + (rng.uniform(0.0, 3.0) as u32).min(2),
        stripes: 4 + (rng.uniform(0.0, 3.0) as usize).min(2),
        ..ScrollParams::acanthus(size, cw)
    }
    .jittered(&mut rng, 1.0);
    let style = ScrollStyle::from_palette(palette, true, rng.uniform(0.15, 0.5));
    sh.scroll("acanthus", seed, &params, &style)
        .map_or_else(Node::empty, |b| b.node)
}

/// Two scrolls springing from one base and curling away from each other.
fn opposed(seed: &SeedTree, palette: &Palette, size: f64, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let spread = rng.uniform(0.35, 0.6);
    let t = rng.uniform(0.2, 0.5);
    let mut g = Node::empty();
    for (i, s) in [1.0, -1.0].into_iter().enumerate() {
        let params = ScrollParams {
            stripes: 3,
            ..ScrollParams::acanthus(size * rng.uniform(0.75, 0.9), s < 0.0)
        }
        .jittered(&mut rng, 0.6);
        let style = ScrollStyle::from_palette(palette, i == 0, t);
        if let Some(b) = sh.scroll("opposed", &seed.index(i as u64), &params, &style) {
            // Both start upwards from the origin and lean apart.
            let a = -PI / 2.0 + s * spread;
            g.push(b.node.with_transform(Affine::rotate(a)));
        }
    }
    g
}

/// A broad scroll with two small curls sprouting from its outer edge.
fn with_curls(seed: &SeedTree, palette: &Palette, size: f64, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let cw = coin(&mut rng);
    let params = ScrollParams {
        stripes: 5,
        ..ScrollParams::acanthus(size, cw)
    }
    .jittered(&mut rng, 0.7);
    let warm = coin(&mut rng);
    let style = ScrollStyle::from_palette(palette, warm, rng.uniform(0.2, 0.5));
    let other = ScrollStyle::from_palette(palette, !warm, rng.uniform(0.3, 0.6));
    let Some(main) = sh.scroll("curled", seed, &params, &style) else {
        return Node::empty();
    };
    let mut g = Node::empty();
    // Curls sprout from the outer edge, leaning forward, and roll the same way.
    for (k, i) in [2usize, 4].into_iter().enumerate() {
        let (at, out) = main.outer[i];
        let fwd = Vec2::new(out.y, -out.x) * if cw { 1.0 } else { -1.0 };
        let d = (out * 0.8 + fwd * 0.6).normalize();
        let len = size * rng.uniform(0.3, 0.4) * if k == 0 { 1.0 } else { 0.8 };
        let curl = ScrollParams {
            width: 0.08,
            ..ScrollParams::curl(len, cw)
        }
        .jittered(&mut rng, 0.6);
        let start = at - d * (0.04 * size);
        if let Some(b) = sh.scroll("curled-curl", &seed.index(k as u64), &curl, &other) {
            let place = Affine::translate(start.to_vec2()) * Affine::rotate(d.y.atan2(d.x));
            g.push(b.node.with_transform(place));
        }
    }
    g.push(main.node);
    g
}

/// A small, tightly rolled curl.
fn small_curl(seed: &SeedTree, palette: &Palette, size: f64, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let params = ScrollParams::curl(size, coin(&mut rng)).jittered(&mut rng, 0.8);
    let style = ScrollStyle::from_palette(palette, coin(&mut rng), rng.uniform(0.2, 0.6));
    sh.scroll("curl", seed, &params, &style)
        .map_or_else(Node::empty, |b| b.node)
}

fn build_all(
    seed: &SeedTree,
    palette: &Palette,
    size: f64,
    accuracy: f64,
) -> (Vec<(String, Node)>, Reports) {
    if !(size.is_finite() && size > 0.0) {
        return (Vec::new(), Vec::new());
    }
    let mut sheet = Sheet {
        acc: motif_accuracy(accuracy, size),
        reports: Vec::new(),
    };
    let nodes = BUILDERS
        .iter()
        .map(|(name, build)| {
            let node = build(&seed.child(name), palette, size, &mut sheet);
            (name.to_string(), fit_to_box(node, size))
        })
        .collect();
    (nodes, sheet.reports)
}

/// Named demonstration scrolls (a single acanthus scroll, an opposed pair, a scroll with
/// curls and a small curl), each fitting a `size` × `size` box centred at the origin.
pub fn variants(
    seed: &SeedTree,
    palette: &Palette,
    size: f64,
    accuracy: f64,
) -> Vec<(String, Node)> {
    build_all(seed, palette, size, accuracy).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motif::ribbon::tests::assert_well_formed;
    use kurbo::Shape;
    use std::collections::HashSet;

    fn palette() -> Palette {
        Palette::by_name("botanica").unwrap()
    }

    #[test]
    fn variants_build_valid_scrolls_that_fit_their_box() {
        for seed in 1..=5 {
            let size = 400.0;
            let (vs, reports) = build_all(&SeedTree::new(seed), &palette(), size, 0.05);
            assert!(vs.len() >= 4);
            let names: HashSet<_> = vs.iter().map(|(n, _)| n.clone()).collect();
            assert_eq!(names.len(), vs.len());
            for (name, node) in &vs {
                assert_well_formed(name, node, size);
            }
            assert!(reports.len() >= vs.len());
            for (label, r) in &reports {
                assert!(r.is_valid(), "seed {seed} {label}: {r:?}");
            }
        }
    }

    #[test]
    fn scrolls_are_deterministic_broad_and_start_along_x() {
        let p = palette();
        let a = variants(&SeedTree::new(3), &p, 300.0, 0.1);
        assert_eq!(a, variants(&SeedTree::new(3), &p, 300.0, 0.1));
        let b = variants(&SeedTree::new(4), &p, 300.0, 0.1);
        for ((name, x), (_, y)) in a.iter().zip(&b) {
            assert_ne!(x, y, "{name} ignores the seed");
        }
        let style = ScrollStyle::from_palette(&p, true, 0.4);
        for cw in [false, true] {
            let params = ScrollParams::acanthus(500.0, cw);
            let b = build(&SeedTree::new(7), &params, &style, 0.1, true).unwrap();
            assert!(b.report.unwrap().is_valid());
            let (anchor, tg) = b.anchor;
            assert!(anchor.is_finite() && (tg.hypot() - 1.0).abs() < 1e-6);
            // It starts at the origin heading along +x, is broad and rolls up.
            let bb = b.outline.bounding_box();
            assert!(bb.x0 > -0.2 * 500.0 && bb.width() < 500.0, "{bb:?}");
            assert!(bb.height() > 0.3 * 500.0, "{bb:?}");
        }
    }

    #[test]
    fn degenerate_inputs_give_empty_nodes() {
        let style = ScrollStyle::from_palette(&palette(), false, 0.5);
        for len in [0.0, -3.0, f64::NAN, f64::INFINITY] {
            let n = scroll(
                &SeedTree::new(1),
                &ScrollParams::acanthus(len, false),
                &style,
                0.1,
            );
            assert_eq!(n.path_count(), 0);
        }
        assert!(variants(&SeedTree::new(1), &palette(), f64::NAN, 0.1).is_empty());
    }
}
