//! Ribbon motifs: tapered Art Nouveau ribbons engraved in layers.
//!
//! [`ribbon_motif`] turns a centreline and a pair of width profiles into a finished
//! [`Node`] shaded as a flat, folded strip in the manner of engraved Art Nouveau borders:
//! a mid-tone fill; a shadow band hugging the shaded edge, darkest at the edge and fading
//! inwards, hatched along the ribbon; a highlight set off-centre towards the lit edge with
//! a soft inner border; a lit edge slightly lighter than the body; thin contour lines and
//! an ink rim, all clipped by the outline, with a hairline outline on top. Where the
//! ribbon bends the other way the shadow and the highlight swap sides smoothly; at a fold
//! (see [`fold_sites`]) the ribbon pinches, shows its dark back side and swaps them at
//! once. Bands and lines follow the ribbon's own `(u, v)` parameterisation with a thickness
//! proportional to the local width, so they taper into tips and spiral eyes and a motif
//! reads the same at any scale.
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
use crate::geom::surface::join_closed;
use crate::geom::validate::{path_is_finite, sanitize};
use crate::geom::{Composite, Curve, CurveExt, EPS, FnCurve, Surface, lerp, smoothstep};
use crate::motif::foliage::{LeafStyle, leaf_motif};
use crate::ornament::contour::{band, line_strips};
use crate::ornament::fit_clean;
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, SeedTree};
use crate::scene::{Color, LineCap, Node, Style};
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
    /// Mid-tone body colour.
    pub fill: Color,
    /// Light core band along the lit side.
    pub highlight: Color,
    /// Outline ink.
    pub ink: Color,
    /// Contour lines and hatching: a dark tone of the fill.
    pub stripe: Color,
    /// Dark tonal bands along the edges.
    pub shade: Color,
    /// Hairline outline stroke width as a fraction of the ribbon's widest half-width (the
    /// visible outline is thicker: an ink rim proportional to the local width).
    pub outline_width: f64,
    /// Thickness of contour lines as a fraction of the local half-width.
    pub stripe_width: f64,
    /// Thickness of hatch lines as a fraction of the local half-width.
    pub hatch_width: f64,
}

impl RibbonStyle {
    /// Derive a style from a palette: the body from the warm (`warm = true`) or cool ramp at
    /// position `t ∈ [0, 1]` (dark → light), the edge bands from a darker step of the same
    /// ramp, the highlight from its light end mixed with the palette's paper colour, lines
    /// from its dark end mixed with ink.
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
            fill: ramp(lerp(0.25, 0.6, t)),
            highlight: ramp(lerp(0.7, 1.0, t)).mix(palette.light, 0.5),
            ink: palette.ink,
            stripe: ramp(lerp(0.0, 0.12, t)).mix(palette.ink, 0.4),
            shade: ramp(lerp(0.05, 0.3, t)).mix(palette.ink, 0.12),
            outline_width: 0.03,
            stripe_width: 0.035,
            hatch_width: 0.028,
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
    /// Contour lines: `0` = none, `1` = the inner border of the shadow band, `2` (or more)
    /// = also a fine line along the lit edge.
    pub stripes: usize,
    /// Also draw the centre line (`v = 0`) as a contour line, like a midrib.
    pub centre_stripe: bool,
    /// Spacing of the contour lines as a fraction of the half-width (kept for
    /// compatibility; the lines follow the tonal bands).
    pub stripe_inset: f64,
    /// Draw the highlight band near the lit edge.
    pub highlight: bool,
    /// Hatch the shadow band.
    pub hatch: bool,
    /// Lit side where the centreline runs straight: `+1` = left (`v > 0`), `-1` = right.
    /// Where it bends, the convex side is lit, so the light swaps sides across an S-curve.
    pub lit_side: f64,
    /// Lean of width bumps towards the end, see [`Ribbon::with_lean`].
    pub lean: f64,
    /// Depth of the shadow band from the shaded edge, as a fraction of the half-width
    /// (`0` = none; clamped to `[0, 0.6]`).
    pub edge_band: f64,
    /// Swallowtail notch cut into both ends, as a multiple of the end's half-width
    /// (`0` = the ribbon's own caps; only open ends are notched).
    pub notch: f64,
    /// Folds, as `(start, end)` arc-length fractions of the back side shown between two
    /// width pinches (empty = none). [`fold_sites`] places them at the ribbon's turns.
    pub folds: Vec<(f64, f64)>,
}

impl RibbonParams {
    /// Both contour lines, shadow band, highlight and hatching on, lit on the left where
    /// straight, no notch, no fold.
    pub fn new(left: WidthProfile, right: WidthProfile) -> RibbonParams {
        RibbonParams {
            left,
            right,
            stripes: 2,
            centre_stripe: false,
            stripe_inset: 0.14,
            highlight: true,
            hatch: true,
            lit_side: 1.0,
            lean: 0.0,
            edge_band: 0.45,
            notch: 0.0,
            folds: Vec::new(),
        }
    }

    /// Same profile on both sides.
    pub fn symmetric(profile: WidthProfile) -> RibbonParams {
        RibbonParams::new(profile.clone(), profile)
    }
}

/// A ribbon around `centerline` with the profiles of `params`, pinched at both ends of
/// every fold, narrowed by [`Ribbon::fit_width`] so that its outline at `accuracy` does not
/// self-intersect.
pub fn fitted_ribbon<C: Curve>(centerline: C, params: &RibbonParams, accuracy: f64) -> Ribbon<C> {
    let folds = fold_spans(&params.folds);
    let pinch = |p: &WidthProfile| {
        if folds.is_empty() {
            p.clone()
        } else {
            let f = folds.clone();
            p.clone().map(move |u, w| w * fold_pinch(&f, u))
        }
    };
    let mut ribbon = Ribbon::new(centerline, pinch(&params.left), pinch(&params.right))
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

/// Samples of the per-`u` lighting of a surface.
const TONE_SAMPLES: usize = 256;
/// Curvature × widest half-width above which a bend counts as one: its convex side is lit
/// (gentler stretches keep [`RibbonParams::lit_side`]).
const BEND: f64 = 1.0 / 60.0;
/// Box-blur radius (in [`TONE_SAMPLES`]) of the light's side: three passes spread a swap
/// over about 15 % of the length.
const SWAP_BLUR: usize = 10;
/// Width at the pinches of a fold, relative to the unpinched width.
const PINCH: f64 = 0.16;
/// Width in the middle of a fold's back side, relative to the unpinched width.
const BACK_WIDTH: f64 = 0.72;

/// Where a surface is lit: `side(u) ∈ [-1, 1]` (`+1` = the left side `v > 0` is lit) and
/// the relative half-width `hw(u) / W`, both tabulated on a uniform `u` grid.
struct Tone {
    side: Vec<f64>,
    rel: Vec<f64>,
}

impl Tone {
    /// The convex side of every bend is lit, as on a flat strip catching the light; where
    /// the centreline runs straight `lit_side` decides. The lit side is blurred along the
    /// ribbon so it swaps gradually across an inflection, but not across a fold: there it
    /// swaps inside the fold's hidden back side, at once.
    fn new<S: Surface + ?Sized>(
        surface: &S,
        lit_side: f64,
        wmax: f64,
        folds: &[(f64, f64)],
    ) -> Tone {
        let n = TONE_SAMPLES;
        let k = centre_curvature(|u| surface.point(u, 0.0), n);
        let bias = if lit_side < 0.0 { -1.0 } else { 1.0 };
        let target: Vec<f64> = k
            .iter()
            .map(|k| (3.0 * (bias - k * wmax / BEND)).tanh())
            .collect();
        // Blur each stretch between fold centres on its own.
        let mut cuts: Vec<usize> = folds
            .iter()
            .map(|(a, b)| ((0.5 * (a + b) * n as f64).round() as usize).clamp(1, n - 1))
            .collect();
        cuts.dedup();
        let mut side = Vec::with_capacity(n + 1);
        let mut start = 0;
        for end in cuts.into_iter().chain([n + 1]) {
            let mut part = target[start..end].to_vec();
            for _ in 0..3 {
                part = box_blur(&part, SWAP_BLUR);
            }
            side.extend(part);
            start = end;
        }
        // Bridge each back side linearly so the table stays continuous.
        for &(a, b) in folds {
            let i0 = ((a * n as f64).floor() as usize).min(n);
            let i1 = ((b * n as f64).ceil() as usize).min(n);
            if i1 > i0 + 1 {
                let (x0, x1) = (side[i0], side[i1]);
                for (j, x) in side[i0..=i1].iter_mut().enumerate() {
                    *x = lerp(x0, x1, j as f64 / (i1 - i0) as f64);
                }
            }
        }
        let rel = (0..=n)
            .map(|i| half_width_at(surface, i as f64 / n as f64) / wmax)
            .collect();
        Tone { side, rel }
    }

    /// Lit side at `u`, in `[-1, 1]`.
    fn side(&self, u: f64) -> f64 {
        sample_table(&self.side, u)
    }

    fn rel(&self, u: f64) -> f64 {
        sample_table(&self.rel, u)
    }

    /// How deep side `side` (`±1`) lies in shadow at `u`, in `[0, 1]`; the two sides
    /// always add up to 1, so the bands never vanish together.
    fn shadow(&self, u: f64, side: f64) -> f64 {
        smoothstep(0.5 * (1.0 - side * self.side(u)))
    }
}

/// Signed curvature of a centreline `point(u)` at `n + 1` uniform `u` samples (Menger
/// curvature over a 6-sample chord, box-blurred three times).
fn centre_curvature(point: impl Fn(f64) -> Point, n: usize) -> Vec<f64> {
    let pts: Vec<Point> = (0..=n).map(|i| point(i as f64 / n as f64)).collect();
    const H: usize = 3;
    let mut k: Vec<f64> = (0..=n)
        .map(|i| {
            let (a, b, c) = (pts[i.saturating_sub(H)], pts[i], pts[(i + H).min(n)]);
            let d = (b - a).hypot() * (c - b).hypot() * (c - a).hypot();
            let m = 2.0 * (b - a).cross(c - a) / d;
            if m.is_finite() { m } else { 0.0 }
        })
        .collect();
    for _ in 0..3 {
        k = box_blur(&k, 10);
    }
    k
}

/// Valid folds: finite, ordered, inside `[0.02, 0.98]`, `0.004` to `0.3` long and not
/// overlapping (a fold overlapping an earlier one is dropped).
fn fold_spans(folds: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = folds
        .iter()
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|&(a, b)| (a.min(b).max(0.02), a.max(b).min(0.98)))
        .filter(|(a, b)| (0.004..=0.3).contains(&(b - a)))
        .collect();
    spans.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(spans.len());
    for s in spans {
        // Keep the pinch ramps (one span wide on each side) apart.
        if out
            .last()
            .is_none_or(|l| s.0 - (s.1 - s.0) >= l.1 + (l.1 - l.0))
        {
            out.push(s);
        }
    }
    out
}

/// Profile of a fold's back side at `x ∈ [0, 1]` between its pinches: a wedge widening
/// slowly from the first crease and closing quickly into the second.
fn back_shape(x: f64) -> f64 {
    (PI * x.clamp(0.0, 1.0).powf(1.9)).sin()
}

/// Width factor of the folds at `u`: the back side between the pinches is a narrower
/// wedge, each pinch narrows to [`PINCH`] and widens back over 1.4 spans outside.
fn fold_pinch(folds: &[(f64, f64)], u: f64) -> f64 {
    folds
        .iter()
        .map(|&(a, b)| {
            let (c, h) = (0.5 * (a + b), 0.5 * (b - a));
            let d = (u - c).abs() / h.max(EPS);
            if d < 1.0 {
                lerp(PINCH, BACK_WIDTH, back_shape((u - a) / (b - a)))
            } else {
                lerp(PINCH, 1.0, smoothstep((d - 1.0) / 1.4))
            }
        })
        .product()
}

/// Up to `max` folds for a ribbon of `profile` along `centerline`, as
/// [`RibbonParams::folds`]: at the inflections between two real bends (where the light
/// swaps sides anyway), away from the ends, the back side about one ribbon width long.
/// Short or stubby ribbons (shorter than 9 widest half-widths) get none.
pub fn fold_sites<C: Curve + ?Sized>(
    centerline: &C,
    profile: &WidthProfile,
    max: usize,
) -> Vec<(f64, f64)> {
    let n = TONE_SAMPLES;
    let wmax = profile.max_sampled(n + 1);
    let al = centerline.arc_length(4 * n);
    let len = al.total();
    if max == 0 || !(wmax.is_finite() && wmax > EPS && len.is_finite()) || len < 9.0 * wmax {
        return Vec::new();
    }
    let k = centre_curvature(|u| centerline.point(al.t_at_fraction(u)), n);
    let bend = |i0: usize, i1: usize| {
        k[i0.min(n)..=i1.min(n)]
            .iter()
            .map(|k| k * wmax)
            .fold((0.0_f64, 0.0_f64), |(lo, hi), x| (lo.min(x), hi.max(x)))
    };
    let reach = n * 15 / 100;
    // Candidates: sign changes with a bend of each sign within reach, scored by the
    // weaker of the two bends.
    let mut cand: Vec<(f64, usize)> = (n / 5..=n - n / 5)
        .filter(|&i| k[i - 1].signum() != k[i].signum())
        .filter_map(|i| {
            let (lo_b, hi_b) = bend(i - reach, i);
            let (lo_a, hi_a) = bend(i, i + reach);
            let before = if k[i - 1] > 0.0 { hi_b } else { -lo_b };
            let after = if k[i] > 0.0 { hi_a } else { -lo_a };
            let score = before.min(after);
            (score > 2.0 * BEND).then_some((score, i))
        })
        .collect();
    cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut sites: Vec<(f64, f64)> = Vec::new();
    for (_, i) in cand {
        let u = i as f64 / n as f64;
        let h = 0.8 * profile.eval(u).max(0.25 * wmax) / len;
        let span = (u - h, u + h);
        if sites
            .iter()
            .all(|s| (0.5 * (s.0 + s.1) - u).abs() > 6.0 * h.max(0.5 * (s.1 - s.0)))
        {
            sites.push(span);
        }
        if sites.len() >= max {
            break;
        }
    }
    fold_spans(&sites)
}

/// Half-width at `u` with non-finite values mapped to `0`.
fn half_width_at<S: Surface + ?Sized>(surface: &S, u: f64) -> f64 {
    let w = surface.half_width(u.clamp(0.0, 1.0));
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

/// Outline of a surface, with a swallowtail notch `notch ×` the end half-width deep cut
/// into each open end when `notch > 0`.
fn notched_outline<S: Surface + ?Sized>(surface: &S, notch: f64, accuracy: f64) -> BezPath {
    let notch = finite_or(notch, 0.0);
    let length = centre_length(surface, 128);
    let (w0, w1) = (half_width_at(surface, 0.0), half_width_at(surface, 1.0));
    if notch <= 0.0 || length.is_nan() || length <= EPS || w0.min(w1) < 4.0 * accuracy {
        return surface.outline(accuracy);
    }
    let du = |w: f64| (notch * w / length).clamp(0.0, 0.3);
    let (Some(left), Some(right)) = (
        fit_clean(&*surface.iso_v(1.0), 0.0, accuracy),
        fit_clean(
            &FnCurve(|t: f64| surface.point(1.0 - t, -1.0)),
            0.0,
            accuracy,
        ),
    ) else {
        return surface.outline(accuracy);
    };
    let mut path = left;
    path.line_to(surface.point(1.0 - du(w1), 0.0));
    path.line_to(surface.point(1.0, -1.0));
    let mut out = join_closed(&path, &right);
    // `join_closed` closes back to the start's left corner: insert the start notch first.
    out.pop();
    out.line_to(surface.point(du(w0), 0.0));
    out.close_path();
    out
}

/// Approximate length of a surface's centre line from `n` chords.
fn centre_length<S: Surface + ?Sized>(surface: &S, n: usize) -> f64 {
    let pts: Vec<Point> = (0..=n)
        .map(|i| surface.point(i as f64 / n as f64, 0.0))
        .collect();
    pts.windows(2).map(|w| (w[1] - w[0]).hypot()).sum()
}

/// The engraved layers of any surface (a [`Ribbon`], a leaf, a sub-band), bottom to top:
/// the mid-tone fill; then, clipped by the outline, a shadow band along the shaded edge
/// (three nested steps, darkest at the edge), a lit edge a little lighter than the body, a
/// highlight off-centre towards the lit edge with a soft inner border, hatch lines inside
/// the shadow band, contour lines, the dark back side of each fold and an ink rim; then a
/// hairline outline. The convex side of every bend is lit (see
/// [`RibbonParams::lit_side`]), so shadow and highlight swap sides across an S-curve.
///
/// Every band and line lives in the surface's `(u, v)` space with a thickness proportional
/// to the local half-width, so it tapers into tips and spiral eyes instead of piling up,
/// and the visible outline thins where the ribbon does. Only the ornament fields of
/// `params` are used; the hairline width is the style's fraction of the widest half-width.
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
    let outline = clean(&notched_outline(surface, params.notch, acc), acc);
    if outline.elements().len() < 3 {
        return root;
    }
    let frac = |k: f64| finite_or(k, 0.0).max(0.0);
    root.push(Node::path(outline.clone(), Style::fill(style.fill)));

    let folds = fold_spans(&params.folds);
    let tone = Tone::new(surface, params.lit_side, wmax, &folds);
    let mut details = Node::empty();
    // A strip around `v = centre(u)`, `half(u)` wide each way, split where it vanishes.
    let strips = |centre: &dyn Fn(f64) -> f64, half: &dyn Fn(f64) -> f64| {
        line_strips(surface, centre, half, acc)
    };
    // Fades a line out where the surface is narrower than `need` of its widest width.
    let wide = |u: f64, need: f64| smoothstep((tone.rel(u) - need) / 0.08);

    // Shadow band: three nested steps from the shaded edge inwards, each a fraction of the
    // full depth, so the tone is darkest at the edge. Its depth follows the shadow, so the
    // band slides over to the other edge where the light swaps sides.
    let depth = finite_or(params.edge_band, 0.0).clamp(0.0, 0.6);
    let shade_depth = |u: f64, side: f64| depth * tone.shadow(u, side);
    if depth > 0.0 {
        for (f, alpha) in [(1.0, 0.42), (0.62, 0.5), (0.3, 0.6)] {
            let mut bands = Vec::new();
            for side in [1.0, -1.0] {
                let half = move |u: f64| 0.5 * f * shade_depth(u, side);
                bands.extend(strips(&|u| side * (1.0 - half(u)), &half));
            }
            push_merged(
                &mut details,
                bands,
                Style::fill(style.shade).with_opacity(alpha),
                acc,
            );
        }
    }

    // Lit edge: a narrow band a little lighter than the body, no dark band.
    let lit_edge = |u: f64, side: f64| 0.2 * (1.0 - tone.shadow(u, side));
    {
        let mut bands = Vec::new();
        for side in [1.0, -1.0] {
            let half = move |u: f64| 0.5 * lit_edge(u, side);
            bands.extend(strips(&|u| side * (1.0 - half(u)), &half));
        }
        push_merged(
            &mut details,
            bands,
            Style::fill(style.fill.mix(style.highlight, 0.3)),
            acc,
        );
    }

    // Highlight: centred about halfway to the lit edge, crisp on its outer border and soft
    // on its inner one (nested steps that reach further inwards at lower opacity).
    if params.highlight {
        let c = |u: f64| 0.47 * tone.side(u);
        for (h, inner, alpha) in [(0.22, 0.12, 0.35), (0.22, 0.0, 0.45), (0.18, -0.08, 0.7)] {
            let lo = |u: f64| c(u) - h - inner * tone.side(u).max(0.0);
            let hi = |u: f64| c(u) + h + inner * (-tone.side(u)).max(0.0);
            push_merged(
                &mut details,
                vec![band(surface, 0.0, 1.0, lo, hi, acc)],
                Style::fill(style.highlight).with_opacity(alpha),
                acc,
            );
        }
    }

    // Hatching: lines along the ribbon inside the shadow band, at fixed fractions of its
    // depth; the inner ones need a deeper shadow, so they drop out first where the band
    // thins, and all drop out where the ribbon narrows. Line ends taper to points.
    let hatch_t = frac(style.hatch_width);
    if params.hatch && hatch_t > 0.0 && depth > 0.0 {
        let mut rng = seed.rng_for("hatch");
        let mut lines = Vec::new();
        for side in [1.0, -1.0] {
            for k in 0..4 {
                let f = 0.22 + 0.19 * k as f64;
                let thr = 0.3 + 0.12 * k as f64 + rng.uniform(-0.04, 0.04);
                let need = 0.1 + 0.05 * k as f64;
                let v = move |u: f64| side * (1.0 - f * shade_depth(u, side));
                let half = |u: f64| {
                    let s = smoothstep((tone.shadow(u, side) - thr) / 0.15);
                    0.5 * hatch_t * s * wide(u, need)
                };
                lines.extend(strips(&v, &half));
            }
        }
        let ink = style.stripe.mix(style.shade, 0.35);
        push_merged(&mut details, lines, Style::fill(ink), acc);
    }

    // Contour lines: the inner border of the shadow band, then a fine line along the lit
    // edge band; each fades out with its band.
    let stripe_t = frac(style.stripe_width);
    if stripe_t > 0.0 {
        let mut lines = Vec::new();
        for side in [1.0, -1.0] {
            if params.stripes >= 1 && depth > 0.0 {
                let v = move |u: f64| side * (1.0 - shade_depth(u, side));
                let half = |u: f64| {
                    0.5 * stripe_t * smoothstep((tone.shadow(u, side) - 0.2) / 0.3) * wide(u, 0.0)
                };
                lines.extend(strips(&v, &half));
            }
            if params.stripes >= 2 {
                let v = move |u: f64| side * (1.0 - lit_edge(u, side));
                let half = |u: f64| {
                    let lit = 1.0 - tone.shadow(u, side);
                    0.35 * stripe_t * smoothstep((lit - 0.4) / 0.3) * wide(u, 0.12)
                };
                lines.extend(strips(&v, &half));
            }
        }
        if params.centre_stripe {
            lines.extend(strips(&|_| 0.0, &|_| 0.5 * stripe_t));
        }
        push_merged(&mut details, lines, Style::fill(style.stripe), acc);
    }

    // Folds: the back side between the pinches in a darker tone, hiding the bands' swap,
    // with two light lines converging into the pinches.
    if !folds.is_empty() {
        let back = style.shade.mix(style.ink, 0.3);
        let covers = folds
            .iter()
            .map(|&(a, b)| band(surface, a, b, |_| -1.0, |_| 1.0, acc))
            .collect();
        push_merged(&mut details, covers, Style::fill(back), acc);
        let mut lines = Vec::new();
        for &(a, b) in &folds {
            let bump = move |u: f64| {
                let x = (u - a) / (b - a);
                if (0.0..=1.0).contains(&x) {
                    back_shape(x)
                } else {
                    0.0
                }
            };
            let lit = tone.side(b).signum();
            for (v, t) in [(0.45, 0.9), (0.12, 0.6)] {
                let half = move |u: f64| 0.5 * t * stripe_t.max(0.02) * bump(u);
                lines.extend(strips(&move |_| lit * v, &half));
            }
        }
        push_merged(
            &mut details,
            lines,
            Style::fill(style.highlight).with_opacity(0.55),
            acc,
        );
    }

    // Ink rim: the visible outline, as thick as a fixed fraction of the local width.
    let hair = frac(style.outline_width);
    if hair > 0.0 {
        let r = 2.2 * hair;
        let rims = vec![
            band(surface, 0.0, 1.0, |_| 1.0 - r, |_| 1.0, acc),
            band(surface, 0.0, 1.0, |_| -1.0, |_| r - 1.0, acc),
        ];
        push_merged(&mut details, rims, Style::fill(style.ink), acc);
    }

    if details.path_count() > 0 {
        root.push(details.clipped(outline.clone()));
    }
    if hair > 0.0 {
        root.push(Node::path(
            outline,
            Style::stroke(style.ink, hair * wmax).with_cap(LineCap::Round),
        ));
    }
    root
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

    /// Fit, record and draw a leaf in the engraved foliage style (two-tone blade, hatching,
    /// a vein per lobe, ink outline).
    pub(crate) fn foliage<C: Curve>(
        &mut self,
        label: &str,
        seed: &SeedTree,
        mut leaf: Leaf<C>,
        style: &LeafStyle,
    ) -> Node {
        leaf.fit_width(self.acc);
        if let Some(r) = &mut self.reports {
            r.push((label.to_string(), leaf.check(self.acc)));
        }
        leaf_motif(seed, &leaf, style, self.acc)
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
        l * rng.uniform(0.07, 0.085),
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
        l * rng.uniform(0.052, 0.064),
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

/// Wide wavy banner swelling towards the middle like cloth, both ends cut into a
/// swallowtail notch, folded over at its strongest turn.
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
    let swell = rng.uniform(0.18, 0.28);
    let profile = WidthProfile::constant(l * rng.uniform(0.072, 0.085))
        .with_noise(seed.noise_seed("width"), 0.05, 2.0)
        .map(move |u, w| w * (1.0 - swell + swell * (PI * u).sin()));
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.3, 0.7));
    let params = RibbonParams {
        stripes: 2,
        notch: rng.uniform(2.0, 2.6),
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    sh.ribbon("banner", seed, curve, &params, &style).0
}

/// Two strands winding around a common axis, bound together at both ends where they
/// converge into one point; at each crossing the strand on top alternates and the one
/// passing under it darkens in its shadow.
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
    let crossings: usize = if rng.chance(0.5) { 2 } else { 3 };
    // The strands cross at `u = j/m`, `j = 1..m−1`, and meet at both ends `u = 0, 1`.
    let m = (crossings + 1) as f64;
    let axis_len = axis.length();
    // Crossing angle of about 2·atan(amp·π·m / L) ≈ 85°.
    let amp = 0.29 * axis_len / m;
    let w = l * rng.uniform(0.05, 0.058);
    let al = axis.arc_length(1024);
    let strand = |sign: f64| {
        let f = FnCurve(|u: f64| {
            let t = al.t_at_fraction(u);
            axis.point(t) + axis.normal(t) * (sign * amp * (PI * m * u).sin())
        });
        to_composite(&f, acc)
    };
    // Widest where the strands cross (seen face-on), narrower where they turn, tapering
    // into the shared points at the ends.
    let profile = WidthProfile::custom(move |u| {
        w * (0.6 + 0.4 * (PI * m * u).cos().abs())
            * ease(u / 0.12, 0.8)
            * ease((1.0 - u) / 0.12, 0.8)
    });
    let params = RibbonParams {
        stripes: 1,
        ..RibbonParams::symmetric(profile)
    };
    let t = rng.uniform(0.2, 0.5);
    let style_a = RibbonStyle::from_palette(palette, true, t);
    let style_b = RibbonStyle::from_palette(palette, false, t);
    let (node_a, rib_a) = sh.ribbon("twist-a", &seed.child("a"), strand(1.0), &params, &style_a);
    let (node_b, rib_b) = sh.ribbon("twist-b", &seed.child("b"), strand(-1.0), &params, &style_b);
    let radius = 0.47 * axis_len / m;
    let centre = |j: usize| axis.point(al.t_at_fraction((j + 1) as f64 / m));

    let region = |j: usize| Circle::new(centre(j), radius).to_path(acc);
    // Shadow on the strand passing under at crossing `j`: nested full-width bands
    // centred on the crossing, darkest next to the strand on top.
    let shadow = |under: &Ribbon<Composite>, style: &RibbonStyle, j: usize| {
        let c = centre(j);
        let u = (0..=512)
            .map(|i| i as f64 / 512.0)
            .min_by(|a, b| {
                let da = (under.point(*a, 0.0) - c).hypot();
                let db = (under.point(*b, 0.0) - c).hypot();
                da.total_cmp(&db)
            })
            .unwrap_or(0.5);
        let len = under.length().max(EPS);
        let mut g = Node::empty();
        for (k, alpha) in [(2.6, 0.12), (2.0, 0.14), (1.5, 0.16), (1.1, 0.2)] {
            let du = k * w / len;
            let p = clean(&band(under, u - du, u + du, |_| -1.0, |_| 1.0, acc), acc);
            if p.elements().len() >= 3 {
                let dark = style.shade.mix(style.ink, 0.5);
                g.push(Node::path(p, Style::fill(dark).with_opacity(1.6 * alpha)));
            }
        }
        g
    };
    let mut root = Node::empty();
    root.push(node_b.clone());
    for j in (0..crossings).step_by(2) {
        root.push(shadow(&rib_b, &style_b, j));
    }
    root.push(node_a);
    for j in (1..crossings).step_by(2) {
        root.push(shadow(&rib_a, &style_a, j));

        root.push(node_b.clone().clipped(region(j)));
    }
    root
}

/// A broad ribbon that sweeps through an S, folded at its turn, and rolls into a log-spiral
/// tail.
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
        l * rng.uniform(0.07, 0.082),
        rng.uniform(0.15, 0.25),
        0.05,
        1.1,
    );
    let profile = coiled(base, &curve, coil_ratio(turns, end_ratio));
    let style = RibbonStyle::from_palette(palette, rng.chance(0.6), rng.uniform(0.15, 0.5));
    let params = RibbonParams {
        stripes: 2,
        lit_side: if cw { 1.0 } else { -1.0 },
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
    let base = WidthProfile::tapered(l * rng.uniform(0.06, 0.07), 0.04, 0.04, 0.9);
    let profile = coiled(base, &curve, ratio);
    let style = RibbonStyle::from_palette(palette, rng.chance(0.5), rng.uniform(0.2, 0.6));
    let params = RibbonParams {
        stripes: 2,
        lit_side: coin(&mut rng),
        ..RibbonParams::symmetric(profile)
    };
    sh.ribbon("double-curl", seed, curve, &params, &style).0
}

/// A C-shaped ribbon with a large acanthus leaf sprouting from its concave side: the
/// leaf's base is tangent to the ribbon's edge, its midrib follows the bend, swings out
/// and curls back towards the ribbon.
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
        l * rng.uniform(0.058, 0.068),
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

    // The leaf grows along the bend, on the concave side (`side`).
    let (u_bend, kappa) = (0..=40)
        .map(|i| {
            let u = lerp(0.35, 0.65, i as f64 / 40.0);
            (u, chord_curvature(&ribbon, u, 0.03))
        })
        .fold((0.5_f64, 0.0_f64), |a, b| {
            if b.1.abs() > a.1.abs() { b } else { a }
        });
    let side = if kappa < 0.0 { -1.0 } else { 1.0 };
    let span = rng.uniform(0.38, 0.45);
    let u0 = (u_bend - rng.uniform(0.35, 0.5) * span).clamp(0.06, 0.94 - span);
    let len = span * ribbon.length();
    let rho = 1.0 / kappa.abs().max(EPS);
    let hw0 = ribbon.half_width(u0);
    let swing = (rng.uniform(0.3, 0.36) * len).min(0.7 * (rho - hw0).max(0.0));
    let c = ribbon.centerline();
    // Midrib: the concave edge, pushed out smoothly by up to `swing` (tangent to the edge
    // at the base, parallel to it again at the tip, where the tip curl turns it back).
    let midrib = FnCurve(|s: f64| {
        let u = u0 + span * s;
        let out = c.normal(ribbon.t_at(u)) * side;
        let off = -0.4 * hw0 * (1.0 - s).powi(2) + swing * smoothstep(s);
        ribbon.point(u, side) + out * off
    });
    let midrib = to_composite(&midrib, sh.acc);
    let lobes = if rng.chance(0.5) { 3 } else { 4 };
    let leaf = Leaf::new(
        midrib,
        LeafParams::new(LeafShape::Acanthus { lobes })
            .with_aspect(rng.uniform(0.5, 0.58))
            .with_tip_curl(-side * rng.uniform(0.55, 0.8))
            .with_asymmetry(side * rng.uniform(0.25, 0.4)),
    );
    let leaf_style = LeafStyle::from_palette(palette, !warm, rng.uniform(0.25, 0.5));
    let leaf_node = sh.foliage("leafed-ribbon-leaf", &seed.child("leaf"), leaf, &leaf_style);
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
        // render cost: tonal bands and lines are merged into a few paths per layer
        assert!(
            node.path_count() < 1500,
            "{name}: {} paths",
            node.path_count()
        );
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

    /// Bounding box of the first detail layer (the outermost shadow step) of a motif.
    fn shadow_box(node: &Node) -> kurbo::Rect {
        let Node::Group { children, .. } = node else {
            panic!("not a group")
        };
        let Node::Group { children: d, .. } = &children[1] else {
            panic!("details")
        };
        d[0].bounding_box().expect("shadow")
    }

    #[test]
    fn shading_is_asymmetric_and_follows_the_lit_side() {
        let line = CubicBez::new((0.0, 0.0), (60.0, 0.0), (140.0, 0.0), (200.0, 0.0));
        let style = RibbonStyle::from_palette(&palette(), true, 0.4);
        let seed = SeedTree::new(5);
        for lit in [1.0, -1.0] {
            let params = RibbonParams {
                lit_side: lit,
                ..RibbonParams::symmetric(WidthProfile::constant(10.0))
            };
            let bb = shadow_box(&ribbon_motif(&seed, &line, &params, &style, 0.05));
            // the left normal of +x is +y: the shadow hugs the unlit edge only
            if lit > 0.0 {
                assert!(bb.y1 < -4.0 && bb.y0 > -10.5, "{bb:?}");
            } else {
                assert!(bb.y0 > 4.0 && bb.y1 < 10.5, "{bb:?}");
            }
        }
        // across an S the shadow swaps sides, so it reaches both edges
        let s = CubicBez::new((0.0, 0.0), (60.0, -80.0), (140.0, 80.0), (200.0, 0.0));
        let params = RibbonParams::symmetric(WidthProfile::constant(6.0));
        let node = ribbon_motif(&seed, &s, &params, &style, 0.05);
        let Node::Group { children, .. } = &node else {
            panic!("not a group")
        };
        let Node::Group { children: d, .. } = &children[1] else {
            panic!("details")
        };
        let Node::Path { path, .. } = &d[0] else {
            panic!("shadow path")
        };
        let ribbon = fitted_ribbon(s, &params, 0.05);
        let near = |v: f64| {
            (5..=95).any(|i| {
                let p = ribbon.point(i as f64 / 100.0, v);
                path.contains(p)
            })
        };
        assert!(near(0.97) && near(-0.97));
    }

    #[test]
    fn folds_sit_at_turns_pinch_the_width_and_stay_valid() {
        let s = CubicBez::new((0.0, 0.0), (60.0, -80.0), (140.0, 80.0), (200.0, 0.0));
        let profile = WidthProfile::constant(8.0);
        let folds = fold_sites(&s, &profile, 2);
        assert_eq!(folds.len(), 1, "{folds:?}");
        let (a, b) = folds[0];
        assert!((0.5 * (a + b) - 0.5).abs() < 0.08 && b > a, "{folds:?}");
        assert!(fold_sites(&s, &WidthProfile::constant(30.0), 1).is_empty());
        assert!(fold_sites(&s, &profile, 0).is_empty());
        let params = RibbonParams {
            folds: folds.clone(),
            ..RibbonParams::symmetric(profile)
        };
        let ribbon = fitted_ribbon(s, &params, 0.05);
        assert!(ribbon.check(0.05).is_valid());
        let w = |u: f64| ribbon.half_width(u);
        assert!(w(a) < 0.25 * w(0.15) && w(b) < 0.25 * w(0.85));
        assert!(w(0.5 * (a + b)) > 2.0 * w(a));
        let style = RibbonStyle::from_palette(&palette(), false, 0.3);
        let seed = SeedTree::new(9);
        let node = ribbon_motif(&seed, &s, &params, &style, 0.05);
        assert!(node.is_finite() && node.path_count() > 0);
        assert_eq!(node, ribbon_motif(&seed, &s, &params, &style, 0.05));
        // garbage folds are ignored instead of breaking the ribbon
        let bad = RibbonParams {
            folds: vec![(f64::NAN, 0.5), (0.7, 0.7), (-1.0, 3.0)],
            ..params
        };
        let node = ribbon_motif(&seed, &s, &bad, &style, 0.05);
        assert!(node.is_finite() && node.path_count() > 0);
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
