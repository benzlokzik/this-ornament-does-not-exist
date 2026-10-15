//! Ribbon motifs: tapered Art Nouveau ribbons engraved in layers.
//!
//! [`ribbon_motif`] turns a centreline and a pair of width profiles into a finished
//! [`Node`]: a flat fill, a lighter highlight band along the lit side, nested contour
//! stripes that follow the ribbon's own `(u, v)` parameterisation and taper into the tips,
//! optional surface-following hatching on the shaded side, all clipped by the outline, and
//! an ink outline on top. Stroke widths are proportional to the ribbon's widest half-width,
//! so a motif reads the same at any scale.
//!
//! Where a ribbon rolls into a spiral, its width is capped by a fraction of the local
//! radius of curvature (see [`coiled`]): the width then shrinks with the radius, the turns
//! keep an even gap and the eye stays open. [`fitted_ribbon`] finally narrows the whole
//! ribbon if distant parts would still overlap.
//!
//! [`variants`] assembles named demonstration ribbons: an S-ribbon, a whiplash, a banner,
//! a twisted pair, a scroll, a double curl and a ribbon with an acanthus leaf.

use crate::geom::curl::{organic_curve, whiplash};
use crate::geom::fit::{fit_curve, stub};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::ribbon::{Ribbon, RibbonReport, WidthProfile, ease};
use crate::geom::spiral::{LogSpiral, spiral_tail};
use crate::geom::validate::{path_is_finite, sanitize};
use crate::geom::{Composite, Curve, CurveExt, EPS, FnCurve, Surface, lerp};
use crate::ornament::contour::nested;
use crate::ornament::hatch::{HatchOptions, Shade, hatch};
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, SeedTree};
use crate::scene::{Color, Node, Style};
use kurbo::{Affine, BezPath, Circle, CubicBez, Point, Shape, Vec2};
use std::f64::consts::PI;

/// Free margin around each variant, as a fraction of the box size per side.
pub(crate) const MARGIN: f64 = 0.08;
/// Arc-length samples of the curvature table used by [`coiled`].
const COIL_SAMPLES: usize = 1024;

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// Colours and relative stroke widths of an engraved ribbon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RibbonStyle {
    /// Flat body colour.
    pub fill: Color,
    /// Light band along the lit side.
    pub highlight: Color,
    /// Outline ink.
    pub ink: Color,
    /// Contour stripes and hatching.
    pub stripe: Color,
    /// Outline stroke width as a fraction of the ribbon's widest half-width.
    pub outline_width: f64,
    /// Contour stripe stroke width as a fraction of the widest half-width.
    pub stripe_width: f64,
    /// Hatching stroke width as a fraction of the widest half-width.
    pub hatch_width: f64,
}

impl RibbonStyle {
    /// Derive a style from a palette: the body from the warm (`warm = true`) or cool ramp at
    /// position `t ∈ [0, 1]` (dark → light), the highlight from the light end of the same
    /// ramp mixed with the palette's paper colour, stripes from its dark end.
    pub fn from_palette(palette: &Palette, warm: bool, t: f64) -> RibbonStyle {
        let t = finite_or(t, 0.5).clamp(0.0, 1.0);
        let ramp = |x: f64| {
            if warm {
                palette.warm_at(x)
            } else {
                palette.cool_at(x)
            }
        };
        RibbonStyle {
            fill: ramp(lerp(0.2, 0.6, t)),
            highlight: ramp(lerp(0.65, 1.0, t)).mix(palette.light, 0.45),
            ink: palette.ink,
            stripe: ramp(lerp(0.0, 0.15, t)).mix(palette.ink, 0.35),
            outline_width: 0.075,
            stripe_width: 0.04,
            hatch_width: 0.03,
        }
    }
}

/// Shape and ornament parameters of one ribbon.
#[derive(Debug, Clone)]
pub struct RibbonParams {
    /// Half-width profile of the left side (`v > 0`, along the centreline's left normal).
    pub left: WidthProfile,
    /// Half-width profile of the right side.
    pub right: WidthProfile,
    /// Nested contour stripes per side (0 = none).
    pub stripes: usize,
    /// Also draw the centre line (`v = 0`) as a stripe, like a midrib.
    pub centre_stripe: bool,
    /// Inset of the outermost stripe as a fraction of the half-width.
    pub stripe_inset: f64,
    /// Draw a highlight band along the lit side.
    pub highlight: bool,
    /// Hatch the shaded side.
    pub hatch: bool,
    /// Lit side: `+1` = left (`v > 0`), `-1` = right. The opposite side is hatched.
    pub lit_side: f64,
    /// Lean of width bumps towards the end, see [`Ribbon::with_lean`].
    pub lean: f64,
}

impl RibbonParams {
    /// Two stripes per side, highlight and hatching on, lit on the left.
    pub fn new(left: WidthProfile, right: WidthProfile) -> RibbonParams {
        RibbonParams {
            left,
            right,
            stripes: 2,
            centre_stripe: false,
            stripe_inset: 0.18,
            highlight: true,
            hatch: true,
            lit_side: 1.0,
            lean: 0.0,
        }
    }

    /// Same profile on both sides.
    pub fn symmetric(profile: WidthProfile) -> RibbonParams {
        RibbonParams::new(profile.clone(), profile)
    }
}

/// A ribbon around `centerline` with the profiles of `params`, narrowed by
/// [`Ribbon::fit_width`] so that its outline at `accuracy` does not self-intersect.
pub fn fitted_ribbon<C: Curve>(centerline: C, params: &RibbonParams, accuracy: f64) -> Ribbon<C> {
    let mut ribbon = Ribbon::new(centerline, params.left.clone(), params.right.clone())
        .with_lean(finite_or(params.lean, 0.0));
    ribbon.fit_width(sane_accuracy(accuracy));
    ribbon
}

/// Engraved ribbon around `centerline`: see the module docs for the layers. Hatching draws
/// its randomness from `seed`. A degenerate ribbon gives an empty group.
pub fn ribbon_motif<C: Curve>(
    seed: &SeedTree,
    centerline: &C,
    params: &RibbonParams,
    style: &RibbonStyle,
    accuracy: f64,
) -> Node {
    let ribbon = fitted_ribbon(centerline, params, accuracy);
    ribbon_layers(seed, &ribbon, params, style, accuracy)
}

/// The engraved layers of any surface (a [`Ribbon`], a leaf, a sub-band): fill, then the
/// highlight band, hatching and contour stripes in a group clipped by the outline, then the
/// ink outline. Only the ornament fields of `params` are used. Stroke widths are the
/// style's fractions of the surface's widest half-width.
pub fn ribbon_layers<S: Surface + ?Sized>(
    seed: &SeedTree,
    surface: &S,
    params: &RibbonParams,
    style: &RibbonStyle,
    accuracy: f64,
) -> Node {
    let acc = sane_accuracy(accuracy);
    let mut root = Node::empty();
    let wmax = max_half_width(surface);
    if !(wmax.is_finite() && wmax > EPS) {
        return root;
    }
    let outline = clean(&surface.outline(acc), acc);
    if outline.elements().len() < 3 {
        return root;
    }
    let lit = if params.lit_side < 0.0 { -1.0 } else { 1.0 };
    let width = |k: f64| finite_or(k, 0.0).max(0.0) * wmax;
    root.push(Node::path(outline.clone(), Style::fill(style.fill)));

    let mut details = Node::empty();
    if params.highlight {
        // A soft outer band and a brighter core read as a rounded, satin surface.
        for (a, b, alpha) in [(0.18, 0.8, 0.5), (0.32, 0.68, 0.9)] {
            let band = Band {
                surface,
                v0: lit * a,
                v1: lit * b,
            };
            let path = clean(&band.outline(acc), acc);
            if path.elements().len() >= 3 {
                details.push(Node::path(
                    path,
                    Style::fill(style.highlight).with_opacity(alpha),
                ));
            }
        }
    }
    if params.hatch && width(style.hatch_width) > 0.0 {
        let opts = HatchOptions {
            spacing: 0.26 * wmax,
            angle: 62.0,
            min_len: 0.12 * wmax,
            jitter: 0.25,
            taper: 0.1,
            shade: Shade::Edge {
                side: -lit,
                width: 0.55,
            },
        };
        let lines = hatch(surface, &opts, seed, acc);
        push_merged(
            &mut details,
            lines,
            Style::stroke(style.stripe, width(style.hatch_width)).with_opacity(0.85),
            acc,
        );
    }
    if (params.stripes > 0 || params.centre_stripe) && width(style.stripe_width) > 0.0 {
        let lines = nested(
            surface,
            params.stripes,
            finite_or(params.stripe_inset, 0.18),
            params.centre_stripe,
            acc,
        );
        push_merged(
            &mut details,
            lines,
            Style::stroke(style.stripe, width(style.stripe_width)),
            acc,
        );
    }
    if details.path_count() > 0 {
        root.push(details.clipped(outline.clone()));
    }
    if width(style.outline_width) > 0.0 {
        root.push(Node::path(
            outline,
            Style::stroke(style.ink, width(style.outline_width)),
        ));
    }
    root
}

/// The strip `v ∈ [v0, v1]` of a surface, re-parameterised to `v ∈ [-1, 1]`.
struct Band<'a, S: ?Sized> {
    surface: &'a S,
    v0: f64,
    v1: f64,
}

impl<S: Surface + ?Sized> Surface for Band<'_, S> {
    fn point(&self, u: f64, v: f64) -> Point {
        self.surface
            .point(u, lerp(self.v0, self.v1, 0.5 * (v.clamp(-1.0, 1.0) + 1.0)))
    }
}

/// Largest half-width of a surface over 257 samples (non-finite samples ignored).
fn max_half_width<S: Surface + ?Sized>(surface: &S) -> f64 {
    (0..=256)
        .map(|i| surface.half_width(i as f64 / 256.0))
        .filter(|w| w.is_finite())
        .fold(0.0, f64::max)
}

/// Valid accuracy: non-finite or non-positive values fall back to `0.05`.
fn sane_accuracy(accuracy: f64) -> f64 {
    if accuracy.is_finite() && accuracy > 0.0 {
        accuracy
    } else {
        0.05
    }
}

/// Accuracy used for a demonstration motif of `size`: the requested accuracy, kept between
/// a millionth and a five-hundredth of the size so tiny and huge motifs stay well formed.
pub(crate) fn motif_accuracy(accuracy: f64, size: f64) -> f64 {
    if size.is_finite() && size > 0.0 {
        sane_accuracy(accuracy).clamp(size * 1e-6, size * 2e-3)
    } else {
        sane_accuracy(accuracy)
    }
}

/// Path without non-finite or (relative to `accuracy`) zero-length segments; empty if
/// anything non-finite survives.
pub(crate) fn clean(path: &BezPath, accuracy: f64) -> BezPath {
    let p = sanitize(path, accuracy * 1e-3);
    if path_is_finite(&p) {
        p
    } else {
        BezPath::new()
    }
}

/// Push `paths` merged into one path node (nothing if all are empty).
fn push_merged(group: &mut Node, paths: Vec<BezPath>, style: Style, accuracy: f64) {
    let mut merged = BezPath::new();
    for p in paths {
        merged.extend(clean(&p, accuracy));
    }
    if !merged.elements().is_empty() {
        group.push(Node::path(merged, style));
    }
}

/// Scale and centre `node` so its bounding box fits a `size` × `size` box centred at the
/// origin with a [`MARGIN`] on every side.
pub(crate) fn fit_to_box(node: Node, size: f64) -> Node {
    let Some(bb) = node.bounding_box() else {
        return node;
    };
    let extent = bb.width().max(bb.height());
    if !(extent.is_finite() && extent > 0.0) {
        return node;
    }
    let s = size * (1.0 - 2.0 * MARGIN) / extent;
    node.with_transform(Affine::scale(s) * Affine::translate(-bb.center().to_vec2()))
}

/// A Bézier fit of `curve` as a [`Composite`] (a tiny stub if the curve is degenerate).
pub(crate) fn to_composite<C: Curve + ?Sized>(curve: &C, accuracy: f64) -> Composite {
    Composite::from_bezpath(&fit_curve(curve, sane_accuracy(accuracy)))
        .unwrap_or_else(|| stub(curve.point(0.0), curve.tangent(0.0)))
}

/// `c` traversed backwards.
pub(crate) fn reversed(c: &Composite) -> Composite {
    let segs = c
        .segments()
        .iter()
        .rev()
        .map(|s| CubicBez::new(s.p3, s.p2, s.p1, s.p0))
        .collect();
    Composite::new_by_length(segs)
}

/// `+1` or `-1` with equal probability.
pub(crate) fn coin(rng: &mut ChaCha8Rng) -> f64 {
    if rng.chance(0.5) { 1.0 } else { -1.0 }
}

/// Width-to-radius ratio that keeps the turns of a log spiral apart: with `q` the radius
/// ratio of consecutive turns (`end_ratio^(1/turns)`), a ribbon of half-width `c·r` touches
/// the next turn when `q·(1 + c) = 1 − c`; this returns 60 % of that limit.
pub(crate) fn coil_ratio(turns: f64, end_ratio: f64) -> f64 {
    let turns = finite_or(turns, 1.0).max(0.25);
    let q = finite_or(end_ratio, 0.15)
        .clamp(1e-3, 0.95)
        .powf(1.0 / turns);
    (0.6 * (1.0 - q) / (1.0 + q)).clamp(0.1, 0.4)
}

/// `body` capped by `ratio ×` the local radius of curvature of `centerline` (measured on
/// an arc-length grid and smoothed), through a smooth minimum. Along gentle parts the cap
/// is inactive; inside a spiral the width becomes proportional to the radius, so it tapers
/// evenly into the eye and neighbouring turns keep their gap.
pub fn coiled<C: Curve + ?Sized>(body: WidthProfile, centerline: &C, ratio: f64) -> WidthProfile {
    let ratio = finite_or(ratio, 0.25).clamp(0.01, 0.95);
    let radii = curvature_radii(centerline);
    body.map(move |u, w| {
        let r = sample_table(&radii, u);
        if r.is_finite() {
            smooth_min(w, ratio * r)
        } else {
            w
        }
    })
}

/// Smooth minimum of two non-negative values (blend width a quarter of the smaller one).
fn smooth_min(a: f64, b: f64) -> f64 {
    let k = 0.25 * a.min(b).max(0.0);
    (0.5 * (a + b - ((a - b).powi(2) + k * k).sqrt())).max(0.0)
}

/// Linear interpolation in a table sampled at `i / (len − 1)`.
fn sample_table(table: &[f64], u: f64) -> f64 {
    let n = table.len() - 1;
    let x = finite_or(u, 0.0).clamp(0.0, 1.0) * n as f64;
    let i = (x.floor() as usize).min(n - 1);
    let f = x - i as f64;
    let (a, b) = (table[i], table[i + 1]);
    if a.is_finite() && b.is_finite() {
        a + (b - a) * f
    } else if f < 0.5 {
        a
    } else {
        b
    }
}

/// Radius of curvature at [`COIL_SAMPLES`]` + 1` uniform arc-length fractions (Menger
/// curvature over a 4-sample chord, box-blurred twice); straight parts give `∞`.
fn curvature_radii<C: Curve + ?Sized>(curve: &C) -> Vec<f64> {
    let n = COIL_SAMPLES;
    let al = curve.arc_length(4 * n);
    let pts: Vec<Point> = (0..=n)
        .map(|i| curve.point(al.t_at_fraction(i as f64 / n as f64)))
        .collect();
    const H: usize = 2;
    let mut k: Vec<f64> = (0..=n)
        .map(|i| {
            if i < H || i + H > n {
                return f64::NAN;
            }
            let (a, b, c) = (pts[i - H], pts[i], pts[i + H]);
            let d = (b - a).hypot() * (c - b).hypot() * (c - a).hypot();
            let m = 2.0 * (b - a).cross(c - a).abs() / d;
            if m.is_finite() { m } else { 0.0 }
        })
        .collect();
    for i in 0..H {
        k[i] = k[H];
        k[n - i] = k[n - H];
    }
    for _ in 0..2 {
        k = box_blur(&k, 3);
    }
    k.into_iter()
        .map(|k| if k > 1e-12 { 1.0 / k } else { f64::INFINITY })
        .collect()
}

/// Signed Menger curvature of a ribbon's centreline through the points at arc-length
/// fractions `u − h`, `u`, `u + h`. Unlike [`Curve::curvature`] it does not spike at the
/// joints of a piecewise-cubic centreline, where the parametric speed jumps.
pub(crate) fn chord_curvature<C: Curve>(ribbon: &Ribbon<C>, u: f64, h: f64) -> f64 {
    let p = |u: f64| ribbon.point(u.clamp(0.0, 1.0), 0.0);
    let (a, b, c) = (p(u - h), p(u), p(u + h));
    let d = (b - a).hypot() * (c - b).hypot() * (c - a).hypot();
    let k = 2.0 * (b - a).cross(c - a) / d;
    if k.is_finite() { k } else { 0.0 }
}

fn box_blur(x: &[f64], r: usize) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(r), (i + r).min(n - 1));
            x[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
        })
        .collect()
}

/// Collects validity reports of every ribbon a variant builds (in tests) and carries the
/// size and accuracy of the sheet being drawn.
pub(crate) struct Sheet {
    pub(crate) size: f64,
    pub(crate) acc: f64,
    reports: Option<Vec<(String, RibbonReport)>>,
}

impl Sheet {
    pub(crate) fn new(size: f64, accuracy: f64, record: bool) -> Sheet {
        Sheet {
            size,
            acc: motif_accuracy(accuracy, size),
            reports: record.then(Vec::new),
        }
    }

    /// Fit, record and engrave a ribbon; returns its node and the fitted ribbon.
    pub(crate) fn ribbon<C: Curve>(
        &mut self,
        label: &str,
        seed: &SeedTree,
        centerline: C,
        params: &RibbonParams,
        style: &RibbonStyle,
    ) -> (Node, Ribbon<C>) {
        let ribbon = fitted_ribbon(centerline, params, self.acc);
        if let Some(r) = &mut self.reports {
            r.push((label.to_string(), ribbon.check(self.acc)));
        }
        let node = ribbon_layers(seed, &ribbon, params, style, self.acc);
        (node, ribbon)
    }

    /// Fit, record and engrave a leaf.
    pub(crate) fn leaf<C: Curve>(
        &mut self,
        label: &str,
        seed: &SeedTree,
        mut leaf: Leaf<C>,
        params: &RibbonParams,
        style: &RibbonStyle,
    ) -> Node {
        leaf.fit_width(self.acc);
        if let Some(r) = &mut self.reports {
            r.push((label.to_string(), leaf.check(self.acc)));
        }
        ribbon_layers(seed, &leaf, params, style, self.acc)
    }

    #[cfg(test)]
    pub(crate) fn reports(&self) -> &[(String, RibbonReport)] {
        self.reports.as_deref().unwrap_or(&[])
    }
}

/// A variant builder: draws in world units of the sheet size, roughly centred.
pub(crate) type Builder = fn(&SeedTree, &Palette, &mut Sheet) -> Node;

/// Build every named variant with its own `seed.child(name)` and fit it to the box.
pub(crate) fn build_all(
    builders: &[(&str, Builder)],
    seed: &SeedTree,
    palette: &Palette,
    sheet: &mut Sheet,
) -> Vec<(String, Node)> {
    if !(sheet.size.is_finite() && sheet.size > 0.0) {
        return Vec::new();
    }
    builders
        .iter()
        .map(|(name, build)| {
            let node = build(&seed.child(name), palette, sheet);
            (name.to_string(), fit_to_box(node, sheet.size))
        })
        .collect()
}

/// Leaf-like parameters for engraving a leaf with [`ribbon_layers`]: a midrib, one stripe
/// per side and hatching, no highlight.
pub(crate) fn leaf_params(lit_side: f64) -> RibbonParams {
    RibbonParams {
        stripes: 1,
        centre_stripe: true,
        stripe_inset: 0.35,
        highlight: false,
        hatch: true,
        lit_side,
        ..RibbonParams::symmetric(WidthProfile::zero())
    }
}

/// Mirror a node horizontally with probability ½ (handedness variety).
fn maybe_mirror(node: Node, rng: &mut ChaCha8Rng) -> Node {
    if rng.chance(0.5) {
        node.with_transform(Affine::FLIP_X)
    } else {
        node
    }
}

const BUILDERS: [(&str, Builder); 7] = [
    ("s-ribbon", s_ribbon),
    ("whiplash", whiplash_ribbon),
    ("banner", banner),
    ("twist", twist),
    ("scroll", scroll),
    ("double-curl", double_curl),
    ("leafed-ribbon", leafed_ribbon),
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

/// Tapered S with three nested stripes per side.
fn s_ribbon(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let d = Vec2::from_angle(rng.uniform(-0.6, 0.6));
    let wav = rng.uniform(0.7, 1.0) * coin(&mut rng);
    let bulge = rng.uniform(-0.25, 0.25);
    let curve = organic_curve(
        &mut rng,
        Point::ORIGIN - d * (0.45 * l),
        Point::ORIGIN + d * (0.45 * l),
        wav,
        bulge,
    );
    let profile = WidthProfile::swell(
        l * rng.uniform(0.045, 0.062),
        rng.uniform(0.4, 0.6),
        rng.uniform(1.0, 1.3),
        rng.uniform(1.0, 1.3),
    )
    .with_noise(seed.noise_seed("width"), 0.08, 2.5);
    let style = RibbonStyle::from_palette(palette, true, rng.uniform(0.15, 0.55));
    let params = RibbonParams {
        stripes: 3,
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    sh.ribbon("s-ribbon", seed, curve, &params, &style).0
}

/// Long whiplash with needle tips; the hook narrows with its radius of curvature.
fn whiplash_ribbon(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let dir = Vec2::from_angle(rng.uniform(-0.4, 0.4));
    let hook = rng.uniform(0.9, 1.35);
    let curve = whiplash(&mut rng, Point::ORIGIN, dir, 1.3 * l, hook);
    let body = WidthProfile::swell(
        l * rng.uniform(0.032, 0.045),
        rng.uniform(0.22, 0.34),
        1.4,
        0.9,
    );
    let profile = coiled(body, &curve, 0.3);
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.1, 0.5));
    let params = RibbonParams {
        stripes: 2,
        lit_side: 1.0,
        ..RibbonParams::symmetric(profile)
    };
    let node = sh.ribbon("whiplash", seed, curve, &params, &style).0;
    maybe_mirror(node, &mut rng)
}

/// Wide wavy banner with open rounded ends, a border stripe and a bright highlight.
fn banner(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let d = Vec2::from_angle(rng.uniform(-0.25, 0.25));
    let wav = rng.uniform(0.35, 0.55) * coin(&mut rng);
    let bulge = rng.uniform(-0.15, 0.15);
    let curve = organic_curve(
        &mut rng,
        Point::ORIGIN - d * (0.45 * l),
        Point::ORIGIN + d * (0.45 * l),
        wav,
        bulge,
    );
    let profile = WidthProfile::constant(l * rng.uniform(0.07, 0.085))
        .with_noise(seed.noise_seed("width"), 0.07, 2.0)
        .map(|u, w| w * (0.9 + 0.15 * (PI * u).sin()));
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.3, 0.7));
    let params = RibbonParams {
        stripes: 1,
        stripe_inset: 0.14,
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    sh.ribbon("banner", seed, curve, &params, &style).0
}

/// Two strands winding around a common axis; at each crossing the strand on top
/// alternates and casts a soft shadow on the other one.
fn twist(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let acc = sh.acc;
    let d = Vec2::from_angle(rng.uniform(-0.5, 0.5));
    let wav = rng.uniform(0.15, 0.35) * coin(&mut rng);
    let axis = organic_curve(
        &mut rng,
        Point::ORIGIN - d * (0.45 * l),
        Point::ORIGIN + d * (0.45 * l),
        wav,
        0.0,
    );
    let crossings = if rng.chance(0.5) { 3 } else { 4 };
    let m = crossings as f64;
    let axis_len = axis.length();
    // Crossing angle of about 2·atan(amp·π·m / L) ≈ 85°.
    let amp = 0.29 * axis_len / m;
    let w = l * rng.uniform(0.032, 0.04);
    let al = axis.arc_length(1024);
    let strand = |sign: f64| {
        let f = FnCurve(|u: f64| {
            let t = al.t_at_fraction(u);
            axis.point(t) + axis.normal(t) * (sign * amp * (PI * m * u).cos())
        });
        to_composite(&f, acc)
    };
    // Widest where the strands cross (seen face-on), narrower where they turn.
    let profile = WidthProfile::custom(move |u| {
        w * (0.55 + 0.45 * (PI * m * u).sin().abs())
            * ease(u / 0.08, 0.8)
            * ease((1.0 - u) / 0.08, 0.8)
    });
    let params = RibbonParams {
        stripes: 1,
        stripe_inset: 0.2,
        centre_stripe: true,
        hatch: false,
        ..RibbonParams::symmetric(profile)
    };
    let t = rng.uniform(0.2, 0.5);
    let style_a = RibbonStyle::from_palette(palette, true, t);
    let style_b = RibbonStyle::from_palette(palette, false, t);
    let (node_a, rib_a) = sh.ribbon("twist-a", &seed.child("a"), strand(1.0), &params, &style_a);
    let (node_b, rib_b) = sh.ribbon("twist-b", &seed.child("b"), strand(-1.0), &params, &style_b);
    let out_a = clean(&rib_a.outline(acc), acc);
    let out_b = clean(&rib_b.outline(acc), acc);
    let shade = Vec2::new(0.35, 0.45) * w;
    let radius = 0.47 * axis_len / m;
    let region = |j: usize| {
        let c = axis.point(al.t_at_fraction((j as f64 + 0.5) / m));
        Circle::new(c, radius).to_path(acc)
    };
    let shadow = |top: &BezPath, bottom: &BezPath, j: usize| {
        let mut p = top.clone();
        p.apply_affine(Affine::translate(shade));
        Node::path(p, Style::fill(palette.ink).with_opacity(0.4))
            .clipped(bottom.clone())
            .clipped(region(j))
    };
    let mut root = Node::empty();
    root.push(node_b.clone());
    for j in (0..crossings).step_by(2) {
        root.push(shadow(&out_a, &out_b, j));
    }
    root.push(node_a);
    for j in (1..crossings).step_by(2) {
        root.push(shadow(&out_b, &out_a, j));
        root.push(node_b.clone().clipped(region(j)));
    }
    root
}

/// A broad ribbon that sweeps through an S and rolls into a log-spiral tail.
fn scroll(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let dir = Vec2::from_angle(rng.uniform(-0.4, 0.4));
    let body = whiplash(&mut rng, Point::ORIGIN, dir, 0.95 * l, 0.0);
    let turns = rng.uniform(1.4, 1.9);
    let end_ratio = 0.14;
    let k = LogSpiral::tightness_for(turns, end_ratio);
    let cw = body.curvature(1.0) < 0.0;
    let curve = spiral_tail(&body, turns, k, cw, sh.acc);
    let base = WidthProfile::tapered(
        l * rng.uniform(0.05, 0.065),
        rng.uniform(0.15, 0.25),
        0.05,
        1.1,
    );
    let profile = coiled(base, &curve, coil_ratio(turns, end_ratio));
    let style = RibbonStyle::from_palette(palette, rng.chance(0.6), rng.uniform(0.15, 0.5));
    let params = RibbonParams {
        stripes: 2,
        lit_side: if cw { -1.0 } else { 1.0 },
        ..RibbonParams::symmetric(profile)
    };
    let node = sh.ribbon("scroll", seed, curve, &params, &style).0;
    maybe_mirror(node, &mut rng)
}

/// An S whose two ends roll into spirals of opposite handedness.
fn double_curl(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let dir = Vec2::from_angle(rng.uniform(-0.5, 0.5));
    let body = whiplash(&mut rng, Point::ORIGIN, dir, 0.85 * l, 0.0);
    let end_ratio = 0.15;
    let (t1, t2) = (rng.uniform(1.1, 1.6), rng.uniform(1.1, 1.6));
    let first = spiral_tail(
        &body,
        t1,
        LogSpiral::tightness_for(t1, end_ratio),
        body.curvature(1.0) < 0.0,
        sh.acc,
    );
    let back = reversed(&first);
    let curve = spiral_tail(
        &back,
        t2,
        LogSpiral::tightness_for(t2, end_ratio),
        back.curvature(1.0) < 0.0,
        sh.acc,
    );
    let ratio = coil_ratio(t1.min(t2), end_ratio);
    let base = WidthProfile::tapered(l * rng.uniform(0.04, 0.055), 0.04, 0.04, 0.9);
    let profile = coiled(base, &curve, ratio);
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.2, 0.6));
    let params = RibbonParams {
        stripes: 2,
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    sh.ribbon("double-curl", seed, curve, &params, &style).0
}

/// A C-shaped ribbon with a small acanthus leaf growing from its concave side.
fn leafed_ribbon(seed: &SeedTree, palette: &Palette, sh: &mut Sheet) -> Node {
    let mut rng = seed.rng_for("shape");
    let l = sh.size;
    let d = Vec2::from_angle(rng.uniform(-0.4, 0.4));
    let n = Vec2::new(-d.y, d.x) * coin(&mut rng);
    // An asymmetric C: one end rises steeper than the other.
    let (a, b) = (
        Point::ORIGIN - d * (0.45 * l),
        Point::ORIGIN + d * (0.45 * l),
    );
    let curve = CubicBez::new(
        a,
        a + d * (l * rng.uniform(0.05, 0.2)) + n * (l * rng.uniform(0.4, 0.55)),
        b - d * (l * rng.uniform(0.15, 0.3)) + n * (l * rng.uniform(0.3, 0.45)),
        b,
    );
    let profile = WidthProfile::swell(
        l * rng.uniform(0.04, 0.055),
        rng.uniform(0.4, 0.6),
        1.2,
        1.2,
    );
    let warm = rng.chance(0.5);
    let style = RibbonStyle::from_palette(palette, warm, rng.uniform(0.2, 0.5));
    let params = RibbonParams {
        stripes: 2,
        ..RibbonParams::symmetric(profile)
    };
    let (node, ribbon) = sh.ribbon("leafed-ribbon", seed, curve, &params, &style);

    // The leaf sprouts where the ribbon bends most, from inside the ribbon (its base is
    // hidden under it), leaning forward along the direction of travel.
    let c = ribbon.centerline();
    let (u, kappa) = (0..=40)
        .map(|i| {
            let u = lerp(0.35, 0.65, i as f64 / 40.0);
            (u, chord_curvature(&ribbon, u, 0.03))
        })
        .fold((0.5_f64, 0.0_f64), |a, b| {
            if b.1.abs() > a.1.abs() { b } else { a }
        });
    let side = if kappa < 0.0 { -1.0 } else { 1.0 };
    let t = ribbon.t_at(u);
    let lean = rng.uniform(0.35, 0.6);
    let dir = c.normal(t) * (side * lean.cos()) + c.tangent(t) * lean.sin();
    let rho = 1.0 / kappa.abs().max(EPS);
    let len = (0.9 * rho).min(0.3 * ribbon.length());
    let lobes = if rng.chance(0.5) { 4 } else { 5 };
    let leaf = Leaf::from_base(
        ribbon.point(u, 0.5 * side),
        dir,
        len,
        len * rng.uniform(0.5, 0.62),
        LeafParams::new(LeafShape::Acanthus { lobes })
            .with_tip_curl(-side * rng.uniform(0.2, 0.45))
            .with_asymmetry(rng.uniform(-0.2, 0.2)),
    );
    let leaf_style = RibbonStyle::from_palette(palette, !warm, rng.uniform(0.25, 0.5));
    let leaf_node = sh.leaf(
        "leafed-ribbon-leaf",
        &seed.child("leaf"),
        leaf,
        &leaf_params(side),
        &leaf_style,
    );
    Node::group(vec![leaf_node, node])
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::geom::validate::degenerate_segments;
    use std::collections::HashSet;

    fn palette() -> Palette {
        Palette::by_name("botanica").unwrap()
    }

    /// Variants plus the validity reports of every ribbon they used.
    type Reports = Vec<(String, RibbonReport)>;

    fn build(seed: u64, size: f64, acc: f64) -> (Vec<(String, Node)>, Reports) {
        let mut sheet = Sheet::new(size, acc, true);
        let v = build_all(&BUILDERS, &SeedTree::new(seed), &palette(), &mut sheet);
        (v, sheet.reports().to_vec())
    }

    pub(crate) fn assert_well_formed(name: &str, node: &Node, size: f64) {
        assert!(node.is_finite(), "{name}: non-finite");
        assert!(node.path_count() > 0, "{name}: empty");
        node.for_each_path(|p, _, _| {
            assert!(path_is_finite(p), "{name}: non-finite path");
            assert_eq!(
                degenerate_segments(p, size * 1e-9),
                0,
                "{name}: degenerate segment"
            );
        });
        let bb = node.bounding_box().expect("geometry");
        let h = 0.5 * size * (1.0 + 1e-9);
        assert!(
            bb.x0 >= -h && bb.y0 >= -h && bb.x1 <= h && bb.y1 <= h,
            "{name}: {bb:?} exceeds {size}"
        );
        assert!(
            bb.width().max(bb.height()) > 0.8 * size,
            "{name}: {bb:?} too small"
        );
    }

    #[test]
    fn variants_build_cleanly_with_valid_ribbons() {
        for seed in 1..=5 {
            let size = 400.0;
            let (vs, reports) = build(seed, size, 0.05);
            assert!(vs.len() >= 6);
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
        for size in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            assert!(variants(&SeedTree::new(2), &palette(), size, 0.05).is_empty());
        }
        let vs = variants(&SeedTree::new(2), &palette(), 200.0, f64::NAN);
        assert!(vs.iter().all(|(_, n)| n.is_finite()));
    }

    #[test]
    fn motif_layers_and_proportional_strokes() {
        let curve = CubicBez::new((0.0, 0.0), (60.0, -80.0), (140.0, 80.0), (200.0, 0.0));
        let style = RibbonStyle::from_palette(&palette(), true, 0.4);
        let seed = SeedTree::new(7);
        let widths = |w: f64| {
            let params = RibbonParams::symmetric(WidthProfile::lens(w, 1.2));
            let node = ribbon_motif(&seed, &curve, &params, &style, 0.05);
            let Node::Group { children, .. } = &node else {
                panic!("not a group")
            };
            // fill, clipped details, outline
            assert_eq!(children.len(), 3);
            assert!(matches!(&children[1], Node::Group { clip: Some(_), .. }));
            let Node::Path { style: s, .. } = &children[2] else {
                panic!("outline")
            };
            s.stroke_width
        };
        let (a, b) = (widths(8.0), widths(16.0));
        assert!((b / a - 2.0).abs() < 0.05, "{a} {b}");
        let empty = ribbon_motif(
            &seed,
            &curve,
            &RibbonParams::symmetric(WidthProfile::zero()),
            &style,
            0.05,
        );
        assert_eq!(empty.path_count(), 0);
    }

    #[test]
    fn coil_cap_follows_the_radius() {
        let s = LogSpiral::from_start(Point::ORIGIN, Vec2::new(1.0, 0.0), 100.0, 2.0, 0.15, false);
        let p = coiled(WidthProfile::constant(1e3), &s, 0.3);
        let r = |u: f64| {
            let t = s.arc_length(4096).t_at_fraction(u);
            s.radius_at(t) * (1.0f64 + 0.15 * 0.15).sqrt()
        };
        for u in [0.2, 0.5, 0.8] {
            let w = p.eval(u);
            assert!(
                (w / (0.3 * r(u)) - 1.0).abs() < 0.05,
                "u={u} {w} vs {}",
                r(u)
            );
        }
    }
}
