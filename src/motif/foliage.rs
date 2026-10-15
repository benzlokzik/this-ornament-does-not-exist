//! Engraved leaves with vein systems, and small foliage motifs built from them.
//!
//! # Veins
//! Veins are authored in the `(u, v)` parameter space of a [`Leaf`] (`u` along the midrib,
//! `v` across the blade) and mapped through its [`Surface`], so they bend with the midrib,
//! fan out where the blade widens and converge into the tip. Every vein is a quadratic in
//! `(u, v)` drawn as a filled, tapering engraving line:
//! - the midrib along `v = 0`, swelling at the base and vanishing near the tip;
//! - pinnate laterals leaving the midrib at alternating, slightly varying positions and
//!   angles, running from `(uᵢ, 0)` to `(uᵢ + d, ±0.88)` and bending towards the tip;
//! - veinlets branching from the laterals of large leaves towards the margin;
//! - for lobed shapes (acanthus, fern) exactly one vein per lobe or pinna, ending in its tip
//!   (acanthus lobes also get a veinlet into each sub-lobe), and for serrated leaves veins
//!   ending in the teeth.
//!
//! # Shading
//! The blade is filled in two tones (a lit half and a shaded half, the "folded" Art Nouveau
//! leaf), the shaded half darkens towards its margin in a soft tonal band and carries surface
//! hatching ([`hatch`] with [`Shade::Edge`]) for volume, and everything is clipped by the
//! outline with the ink outline on top. All stroke widths are proportional to the leaf
//! length. Where foliage motifs stack leaves, each leaf casts a thin shadow line on the
//! leaves under it.

use crate::geom::fit::{fit_curve, fit_points};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::validate::sanitize;
use crate::geom::{Composite, Curve, CurveExt, FnCurve, Surface, lerp};
use crate::ornament::{Detail, HatchOptions, Shade, hatch};
use crate::palette::Palette;
use crate::rng::{Draw, SeedTree};
use crate::scene::{Color, LineCap, Node, Style};
use kurbo::{Affine, BezPath, CubicBez, ParamCurve, Point, QuadBez, Vec2};
use std::f64::consts::PI;

/// Segments shorter than this are dropped from produced paths.
const MIN_SEG: f64 = 1e-7;
/// Leaves longer than this (canvas units) get secondary veinlets.
const VEINLET_MIN_LENGTH: f64 = 60.0;
/// Fitting tolerance of detail lines (veins, hatching, tonal bands) as a fraction of the
/// leaf length: far below their stroke widths, so the fit is invisible.
const DETAIL_TOLERANCE: f64 = 0.002;
/// Level of detail by the leaf length on the canvas (pixels): shorter leaves have no
/// hatching, then no veinlets, then neither lateral veins nor tonal bands (fill, lit half,
/// outline and midrib stay).
const LOD_HATCH: f64 = 40.0;
const LOD_VEINLETS: f64 = 28.0;
const LOD_LATERALS: f64 = 16.0;
/// Fraction of the box left free on each side by [`fit_to_box`] in [`variants`].
const MARGIN: f64 = 0.08;

/// Stroke widths of a leaf as fractions of its midrib length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VeinWidths {
    /// Ink outline.
    pub outline: f64,
    /// Midrib at its base.
    pub midrib: f64,
    /// Lateral veins at their base.
    pub lateral: f64,
    /// Secondary veinlets at their base.
    pub veinlet: f64,
    /// Surface hatching lines.
    pub hatch: f64,
}

impl Default for VeinWidths {
    fn default() -> Self {
        VeinWidths {
            outline: 0.008,
            midrib: 0.024,
            lateral: 0.013,
            veinlet: 0.006,
            hatch: 0.0042,
        }
    }
}

/// Colours and line widths of an engraved leaf.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeafStyle {
    /// Fill of the shaded half of the blade.
    pub fill: Color,
    /// Fill of the lit half.
    pub fill_light: Color,
    /// Outline and hatching ink.
    pub ink: Color,
    /// Vein colour.
    pub vein: Color,
    /// Stroke widths relative to the leaf length.
    pub widths: VeinWidths,
}

impl LeafStyle {
    /// Derive a style from a palette: the blade from the warm (`warm = true`) or cool ramp
    /// at position `t ∈ [0, 1]` (dark → light), veins a darker shade of the same ramp.
    pub fn from_palette(palette: &Palette, warm: bool, t: f64) -> LeafStyle {
        let t = finite_or(t, 0.5).clamp(0.0, 1.0);
        let ramp = |x: f64| {
            if warm {
                palette.warm_at(x)
            } else {
                palette.cool_at(x)
            }
        };
        let fill = ramp(lerp(0.2, 0.6, t));
        LeafStyle {
            fill,
            fill_light: ramp(lerp(0.5, 0.9, t)).mix(palette.light, 0.12),
            ink: palette.ink,
            vein: ramp(lerp(0.0, 0.15, t)).mix(palette.ink, 0.5),
            widths: VeinWidths::default(),
        }
    }
}

/// One vein in `(u, v)` space with its base stroke width (canvas units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Vein {
    /// The vein as a quadratic in `(u, v)` (x = u, y = v).
    pub(crate) uv: QuadBez,
    /// Width at the start; it tapers towards the end.
    pub(crate) width: f64,
    /// Exponent of the taper `width · (1 − 0.9 t)^taper`.
    pub(crate) taper: f64,
    /// Place in the vein hierarchy, which decides when the vein is left out of a small leaf.
    pub(crate) rank: VeinRank,
}

/// Place of a vein in the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VeinRank {
    Midrib,
    /// Primary veins from the midrib (one per lobe, pinna or tooth).
    Lateral,
    /// Secondary veins from a lateral.
    Veinlet,
}

/// An engraved leaf: two-tone fill, hatching on the shaded half, veins following the blade,
/// all clipped by the outline, with the ink outline on top. Returns an empty group for a
/// degenerate leaf. Randomness (vein placement, hatching) comes from `seed`.
pub fn leaf_motif<C: Curve>(
    seed: &SeedTree,
    leaf: &Leaf<C>,
    style: &LeafStyle,
    accuracy: f64,
) -> Node {
    leaf_motif_with_detail(seed, leaf, style, accuracy, Detail::FULL)
}

/// [`leaf_motif`] at a level of detail: a leaf that is short on the canvas leaves out its
/// hatching, then its veinlets, then its lateral veins and tonal bands.
pub fn leaf_motif_with_detail<C: Curve>(
    seed: &SeedTree,
    leaf: &Leaf<C>,
    style: &LeafStyle,
    accuracy: f64,
    detail: Detail,
) -> Node {
    let acc = sane_accuracy(accuracy);
    let len = leaf.length();
    if !(len.is_finite() && len > 1e-6) {
        return Node::empty();
    }
    let outline = leaf_outline(leaf, acc);
    if outline.elements().len() < 3 {
        return Node::empty();
    }
    let tol = acc.max(DETAIL_TOLERANCE * len);
    let px = detail.canvas_size(len);
    let w = style.widths;
    // A frond's outline runs round many small pinnae; a full-weight line would clog them.
    let outline_width = w.outline
        * len
        * if matches!(leaf.params().shape, LeafShape::Fern { .. }) {
            0.6
        } else {
            1.0
        };
    let mut rng = seed.child("leaf-shading").rng();
    // The shaded half alternates with the seed so a cluster of leaves does not look stamped.
    let dark_side = if rng.chance(0.5) { 1.0 } else { -1.0 };

    let mut detail = Vec::new();
    let lit = half_blade(leaf, -dark_side, tol);
    if !lit.elements().is_empty() {
        detail.push(Node::path(lit, Style::fill(style.fill_light)));
    }

    // A soft tonal band along the shaded edge, darkening towards the margin, under the
    // hatching: the blade turns away from the light there.
    let shade = style.fill.mix(style.ink, 0.45);
    let bands: &[(f64, f64)] = if px < LOD_LATERALS {
        &[]
    } else {
        &[(0.5, 0.3), (0.75, 0.35)]
    };
    for &(v0, opacity) in bands {
        let band = blade_band(leaf, dark_side * v0, dark_side, tol);
        if !band.elements().is_empty() {
            detail.push(Node::path(band, Style::fill(shade).with_opacity(opacity)));
        }
    }

    // Hatching on the shaded half, across the blade and slanted towards the tip.
    let opts = HatchOptions {
        spacing: (0.024 * len).max(2.0 * acc),
        angle: 72.0,
        min_len: 0.015 * len,
        jitter: 0.35,
        taper: 0.1,
        shade: Shade::Edge {
            side: dark_side,
            width: 0.6,
        },
    };
    let lines = if px < LOD_HATCH {
        Vec::new()
    } else {
        hatch(leaf, &opts, &seed.child("leaf-hatch"), tol)
    };
    let merged = merge(&lines);
    if !merged.elements().is_empty() {
        let s = Style::stroke(style.ink.mix(style.fill, 0.25), w.hatch * len)
            .with_cap(LineCap::Round)
            .with_opacity(0.75);
        detail.push(Node::path(merged, s));
    }

    // Veins as filled tapering lines: dark on the lit half, light (an engraved highlight)
    // on the shaded half, where a dark line would vanish into the hatching.
    let (mut dark, mut light) = (Vec::new(), Vec::new());
    let max_rank = if px < LOD_LATERALS {
        VeinRank::Midrib
    } else if px < LOD_VEINLETS {
        VeinRank::Lateral
    } else {
        VeinRank::Veinlet
    };
    for v in leaf_veins(seed, leaf, &w) {
        if v.rank > max_rank {
            continue;
        }
        let p = vein_path(leaf, &v, tol);
        if p.elements().is_empty() {
            continue;
        }
        if v.uv.p2.y * dark_side > 0.0 {
            light.push(p);
        } else {
            dark.push(p);
        }
    }
    for (paths, color) in [
        (light, style.fill_light.mix(style.ink, 0.08)),
        (dark, style.vein),
    ] {
        let merged = merge(&paths);
        if !merged.elements().is_empty() {
            detail.push(Node::path(merged, Style::fill(color)));
        }
    }

    Node::group(vec![
        Node::path(outline.clone(), Style::fill(style.fill)),
        Node::group(detail).clipped(outline.clone()),
        Node::path(
            outline,
            Style::stroke(style.ink, outline_width).with_cap(LineCap::Round),
        ),
    ])
}

/// The vein system of `leaf` in `(u, v)` space (midrib first). Widths are canvas units.
pub(crate) fn leaf_veins<C: Curve>(seed: &SeedTree, leaf: &Leaf<C>, w: &VeinWidths) -> Vec<Vein> {
    let mut rng = seed.child("leaf-veins").rng();
    let len = leaf.length();
    if !(len.is_finite() && len > 1e-6) {
        return Vec::new();
    }
    let p = *leaf.params();
    // Veins never get wider than a fraction of the blade (a willow is narrow).
    let max_half = (0..=32)
        .map(|i| leaf.half_width(i as f64 / 32.0))
        .filter(|x| x.is_finite())
        .fold(0.0, f64::max);
    let lateral = (w.lateral * len).min(0.16 * max_half);
    let veinlet = (w.veinlet * len).min(0.09 * max_half);
    let midrib = (w.midrib * len).min(0.3 * max_half.max(0.02 * len));

    let mut out = vec![Vein {
        uv: QuadBez::new(
            (0.0, 0.0),
            (0.5, rng.jitter(0.0, 0.02)),
            (
                if matches!(p.shape, LeafShape::Fern { .. }) {
                    0.985
                } else {
                    0.95
                },
                0.0,
            ),
        ),
        width: midrib,
        taper: 0.8,
        rank: VeinRank::Midrib,
    }];

    match p.shape {
        LeafShape::Acanthus { .. } => {
            for side in [1.0, -1.0] {
                // One vein into every lobe, bending with it into its crest, and a veinlet
                // into each side sub-lobe.
                for lobe in p.lobes(side) {
                    // The surface already leans the lobe forward, so a vein of nearly
                    // constant `u` runs along the lobe's axis.
                    let ut = lobe.crest;
                    let s = lerp(ut, lobe.start, 0.45).max(0.01);
                    let q =
                        QuadBez::new((s, 0.0), (lerp(s, ut, 0.8), side * 0.35), (ut, side * 0.9));
                    out.push(Vein {
                        uv: q,
                        width: lateral,
                        taper: 1.0,
                        rank: VeinRank::Lateral,
                    });
                    for (k, &su) in lobe.subs.iter().enumerate() {
                        let t0 = if k == 0 { 0.3 } else { 0.45 };
                        let a = q.eval(t0);
                        let e = Point::new(su, side * 0.84);
                        let c = Point::new(lerp(a.x, e.x, 0.6), lerp(a.y, e.y, 0.4));
                        out.push(Vein {
                            uv: QuadBez::new(a, c, e),
                            width: veinlet,
                            taper: 1.0,
                            rank: VeinRank::Veinlet,
                        });
                    }
                }
            }
        }
        LeafShape::Fern { pinnae } => {
            let n = pinnae.clamp(4, 40) as f64;
            for side in [1.0, -1.0] {
                let phase = if side < 0.0 { 0.5 * p.asymmetry } else { 0.0 };
                for k in -1..(n as i32 + 1) {
                    let uc = (k as f64 + 0.5 - phase) / n;
                    if !(0.05..=0.96).contains(&uc) {
                        continue;
                    }
                    let s = (uc - 0.25 / n).max(0.0);
                    let q = QuadBez::new((s, 0.0), (uc, side * 0.35), (uc + 0.08 / n, side * 0.85));
                    // A faint vein along every pinna.
                    out.push(Vein {
                        uv: q,
                        width: veinlet.max(0.6 * lateral),
                        taper: 1.0,
                        rank: VeinRank::Lateral,
                    });
                }
            }
        }
        LeafShape::Serrated { teeth } => {
            let n = teeth.clamp(4, 80) as f64;
            let stride = (n / 10.0).ceil().max(1.0) as i32;
            for side in [1.0, -1.0] {
                let phase = if side < 0.0 { 0.5 * p.asymmetry } else { 0.0 };
                // Craspedodromous veins: each ends in a tooth (claw peak at f = 0.8).
                let mut k = if side < 0.0 { stride / 2 } else { 0 };
                while (k as f64) < n + 1.0 {
                    let ut = (k as f64 + 0.8 - phase) / n;
                    k += stride;
                    if !(0.12..=0.9).contains(&ut) {
                        continue;
                    }
                    let d = run(leaf, ut, 0.92, rng.jitter(42.0, 5.0).to_radians()).min(0.5 * ut);
                    let s = ut - d;
                    let q = QuadBez::new(
                        (s, 0.0),
                        (lerp(s, ut, 0.25), side * 0.55),
                        (ut, side * 0.92),
                    );
                    out.push(Vein {
                        uv: q,
                        width: lateral,
                        taper: 1.0,
                        rank: VeinRank::Lateral,
                    });
                }
            }
        }
        shape => {
            let (pairs, angle) = match shape {
                LeafShape::Lanceolate => (7, 34.0),
                LeafShape::Ovate => (6, 46.0),
                LeafShape::Cordate => (5, 50.0),
                LeafShape::Willow => (11, 22.0),
                _ => (6, 40.0),
            };
            let (u0, u1) = if shape == LeafShape::Cordate {
                (0.16, 0.74)
            } else {
                (0.07, 0.8)
            };
            let step = (u1 - u0) / (pairs as f64 - 1.0);
            let big = len > VEINLET_MIN_LENGTH && p.aspect > 0.25;
            for side in [1.0, -1.0] {
                // Alternate: the right side is offset by half a step.
                let offset = if side < 0.0 { 0.5 * step } else { 0.0 };
                for i in 0..pairs {
                    let us = u0 + offset + i as f64 * step + rng.jitter(0.0, 0.18 * step);
                    if us > 0.88 {
                        continue;
                    }
                    let a = rng.jitter(angle, 5.0).to_radians();
                    let d = run(leaf, us, 0.88, a);
                    let ue = (us + d).min(0.975);
                    let ve = side * 0.88 * rng.uniform(0.94, 1.0);
                    let q = QuadBez::new((us, 0.0), (lerp(us, ue, 0.25), 0.62 * ve), (ue, ve));
                    out.push(Vein {
                        uv: q,
                        width: lateral,
                        taper: 1.0,
                        rank: VeinRank::Lateral,
                    });
                    if big {
                        // Veinlets from the basal flank of the lateral towards the margin,
                        // filling the space before the previous lateral.
                        for t0 in [0.4, 0.68] {
                            if !rng.chance(0.8) {
                                continue;
                            }
                            let a = q.eval(t0 + rng.jitter(0.0, 0.05));
                            let eu = (ue - (0.9 - t0) * step).max(a.x + 0.015);
                            let e = Point::new(eu, side * 0.93);
                            let c = Point::new(lerp(a.x, e.x, 0.6), lerp(a.y, e.y, 0.3));
                            out.push(Vein {
                                uv: QuadBez::new(a, c, e),
                                width: veinlet,
                                taper: 1.0,
                                rank: VeinRank::Veinlet,
                            });
                        }
                    }
                }
            }
            if shape == LeafShape::Cordate {
                // Palmate basal veins sweeping back into the two rounded lobes.
                for side in [1.0, -1.0] {
                    for (eu, ev) in [(0.07, 0.8), (0.13, 0.9)] {
                        let q = QuadBez::new((0.015, 0.0), (0.0, side * 0.6), (eu, side * ev));
                        out.push(Vein {
                            uv: q,
                            width: 0.85 * lateral,
                            taper: 1.0,
                            rank: VeinRank::Lateral,
                        });
                    }
                }
            }
        }
    }
    out.retain(|v| v.width.is_finite() && v.width > 0.0);
    out
}

/// Arc-length fraction a vein starting near `u` must run along the midrib to reach `|v|`
/// at the angle `angle` to the midrib (clamped to sensible bounds).
fn run<C: Curve>(leaf: &Leaf<C>, u: f64, v: f64, angle: f64) -> f64 {
    let len = leaf.length().max(1e-9);
    let mut d = 0.15;
    for _ in 0..2 {
        let w = leaf.half_width((u + 0.5 * d).clamp(0.0, 1.0));
        d = (v * w / angle.tan().max(0.05) / len).clamp(0.03, 0.4);
    }
    d
}

/// The closed region of the blade between the midrib and the edge `v = side`.
fn half_blade<C: Curve>(leaf: &Leaf<C>, side: f64, tolerance: f64) -> BezPath {
    blade_band(leaf, 0.0, side, tolerance)
}

/// The closed region of the blade between the contours `v = a` and `v = b`, both fitted
/// within `tolerance`.
fn blade_band<C: Curve>(leaf: &Leaf<C>, a: f64, b: f64, tolerance: f64) -> BezPath {
    let outer = fit_curve(&FnCurve(|u: f64| leaf.point(u, b)), tolerance);
    let inner = fit_curve(&FnCurve(|u: f64| leaf.point(1.0 - u, a)), tolerance);
    let p = sanitize(&join_closed(&[outer, inner]), 1e-6);
    if crate::geom::validate::path_is_finite(&p) {
        p
    } else {
        BezPath::new()
    }
}

/// The closed outline of `leaf`: both edges fitted within `accuracy` and joined at the base
/// and the tip.
///
/// The same edges as [`Leaf::outline`](Surface::outline), fitted with the crate's Schneider
/// fitter, which needs about a third of the segments of the ribbon fitter's bisection.
pub(crate) fn leaf_outline<C: Curve>(leaf: &Leaf<C>, accuracy: f64) -> BezPath {
    let acc = sane_accuracy(accuracy);
    let left = fit_curve(&FnCurve(|u: f64| leaf.point(u, 1.0)), acc);
    let right = fit_curve(&FnCurve(|u: f64| leaf.point(1.0 - u, -1.0)), acc);
    let p = sanitize(&join_closed(&[left, right]), MIN_SEG);
    if crate::geom::validate::path_is_finite(&p) && p.elements().len() > 3 {
        p
    } else {
        BezPath::new()
    }
}

/// Concatenate open paths into one closed path, bridging gaps with straight lines.
fn join_closed(parts: &[BezPath]) -> BezPath {
    let mut out = BezPath::new();
    for el in parts.iter().flat_map(|p| p.elements()) {
        match *el {
            kurbo::PathEl::MoveTo(p) if !out.elements().is_empty() => out.line_to(p),
            kurbo::PathEl::ClosePath => {}
            other => out.push(other),
        }
    }
    if !out.elements().is_empty() {
        out.close_path();
    }
    out
}

/// A vein mapped through the leaf surface as a filled tapering line.
fn vein_path<C: Curve>(leaf: &Leaf<C>, vein: &Vein, accuracy: f64) -> BezPath {
    let curve = uv_curve(leaf, vein.uv);
    let width = vein.width;
    let taper = vein.taper;
    tapered(
        &curve,
        |t| width * (1.0 - 0.9 * t).max(0.0).powf(taper),
        accuracy,
    )
}

/// A `(u, v)` quadratic mapped through a surface.
pub(crate) fn uv_curve<S: Surface + ?Sized>(surface: &S, q: QuadBez) -> impl Curve + '_ {
    FnCurve(move |t: f64| {
        let p = q.eval(t.clamp(0.0, 1.0));
        surface.point(p.x, p.y)
    })
}

/// Concatenate paths into one (several subpaths) so the scene stays small.
pub(crate) fn merge(paths: &[BezPath]) -> BezPath {
    let mut out = BezPath::new();
    for p in paths {
        for el in p.elements() {
            out.push(*el);
        }
    }
    out
}

pub(crate) fn sane_accuracy(accuracy: f64) -> f64 {
    if accuracy.is_finite() && accuracy > 0.0 {
        accuracy.clamp(1e-3, 10.0)
    } else {
        0.1
    }
}

pub(crate) fn finite_or(x: f64, default: f64) -> f64 {
    if x.is_finite() { x } else { default }
}

/// A filled engraving line along `curve` whose full width at parameter `t` is `width(t)`,
/// its sides fitted within `accuracy`. Empty for a (near) zero-length or non-finite curve.
pub(crate) fn tapered(curve: &dyn Curve, width: impl Fn(f64) -> f64, accuracy: f64) -> BezPath {
    let coarse: Vec<Point> = (0..=8).map(|i| curve.point(i as f64 / 8.0)).collect();
    let len: f64 = coarse.windows(2).map(|w| (w[1] - w[0]).hypot()).sum();
    if !len.is_finite() || len <= 1e-6 {
        return BezPath::new();
    }
    let n = (len / (10.0 * accuracy)).ceil().clamp(6.0, 48.0) as usize;
    let pts: Vec<Point> = (0..n)
        .map(|i| curve.point(i as f64 / (n - 1) as f64))
        .collect();
    tapered_points(&pts, |i| width(i as f64 / (n - 1) as f64), accuracy)
}

/// A filled line through `pts` with full width `width(i)` at point `i`, its two sides
/// fitted within `tolerance` (a smooth line needs a few cubics, not one per sample).
pub(crate) fn tapered_points(
    pts: &[Point],
    width: impl Fn(usize) -> f64,
    tolerance: f64,
) -> BezPath {
    let n = pts.len();
    if n < 2 || pts.iter().any(|p| !p.is_finite()) {
        return BezPath::new();
    }
    let mut normal = Vec2::new(1.0, 0.0);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for (i, &p) in pts.iter().enumerate() {
        let d = pts[(i + 1).min(n - 1)] - pts[i.saturating_sub(1)];
        if d.hypot() > 1e-12 {
            normal = Vec2::new(-d.y, d.x) / d.hypot();
        }
        let half = 0.5 * finite_or(width(i), 0.0).max(0.0);
        a.push(p + normal * half);
        b.push(p - normal * half);
    }
    b.reverse();
    let tol = sane_accuracy(tolerance);
    // Join the two sides with a line across the end (dropped by sanitize if empty).
    sanitize(
        &join_closed(&[fit_points(&a, tol), fit_points(&b, tol)]),
        MIN_SEG,
    )
}

/// Midrib of a leaf growing from `base` along `direction` (bent gently to one side),
/// matching [`Leaf::from_base`] but as a [`Composite`] so leaves can share one type.
pub(crate) fn leaf_midrib(base: Point, direction: Vec2, length: f64, bend: f64) -> Composite {
    let d = if direction.is_finite() && direction.hypot() > 1e-9 {
        direction / direction.hypot()
    } else {
        Vec2::new(1.0, 0.0)
    };
    let base = if base.is_finite() { base } else { Point::ZERO };
    let len = finite_or(length, 0.0).max(1e-6);
    let n = Vec2::new(-d.y, d.x);
    let b = finite_or(bend, 0.0) * len;
    Composite::new(vec![CubicBez::new(
        base,
        base + d * (0.35 * len) + n * (0.8 * b),
        base + d * (0.7 * len) + n * b,
        base + d * len,
    )])
}

/// A leaf of midrib `length` and maximum blade `width` from `base` along `direction`.
pub(crate) fn leaf_at(
    base: Point,
    direction: Vec2,
    length: f64,
    width: f64,
    params: LeafParams,
) -> Leaf<Composite> {
    let len = finite_or(length, 0.0).max(0.0);
    let aspect = if len > 1e-9 {
        finite_or(width, 0.0).max(0.0) / len
    } else {
        params.aspect
    };
    let bend = -0.08 * finite_or(params.asymmetry, 0.0).clamp(-1.0, 1.0);
    Leaf::new(
        leaf_midrib(base, direction, len, bend),
        params.with_aspect(aspect),
    )
}

/// Scale and centre `node` so its bounding box fits a `size` × `size` box centred at the
/// origin with a margin of `margin · size` on each side.
pub(crate) fn fit_to_box(node: Node, size: f64, margin: f64) -> Node {
    let Some(bb) = node.bounding_box() else {
        return node;
    };
    let extent = bb.width().max(bb.height());
    if !(extent.is_finite() && extent > 1e-12 && size.is_finite() && size > 0.0) {
        return node;
    }
    let s = size * (1.0 - 2.0 * margin) / extent;
    node.with_transform(Affine::scale(s) * Affine::translate(-bb.center().to_vec2()))
}

/// A short tapering stalk (petiole or stem) along `curve` drawn in `color` with an ink edge.
pub(crate) fn stalk(
    curve: &dyn Curve,
    w0: f64,
    w1: f64,
    color: Color,
    ink: Color,
    acc: f64,
) -> Node {
    let body = tapered(curve, |t| lerp(w0, w1, t), acc);
    if body.elements().is_empty() {
        return Node::empty();
    }
    Node::path(body, Style::fill_stroke(color, ink, 0.18 * w0.max(w1)))
}

/// Named demonstration foliage, each fitting a `size` × `size` box centred at the origin.
pub fn variants(
    seed: &SeedTree,
    palette: &Palette,
    size: f64,
    accuracy: f64,
) -> Vec<(String, Node)> {
    let seed = seed.child("foliage-variants");
    let size = if size.is_finite() && size > 0.0 {
        size
    } else {
        100.0
    };
    let acc = sane_accuracy(accuracy);
    let l = size;
    let mut out = Vec::new();
    let mut add = |name: &str, node: Node| {
        out.push((name.to_string(), fit_to_box(node, size, MARGIN)));
    };

    // Acanthus: a large lobed leaf on an S-curved midrib whose tip curls over.
    {
        let s = seed.child("acanthus");
        let mut rng = s.rng();
        let lobes = *rng.pick(&[4u32, 5]);
        let turn = if rng.chance(0.5) { 1.0 } else { -1.0 };
        let p = LeafParams::new(LeafShape::Acanthus { lobes })
            .with_asymmetry(rng.jitter(0.0, 0.3))
            .jittered(&mut rng, 0.4)
            .with_tip_curl(turn * rng.uniform(0.75, 1.0));
        let d = Vec2::from_angle(-PI / 2.0 + rng.jitter(0.0, 0.25));
        let n = Vec2::new(-d.y, d.x) * turn;
        // The lower midrib bows against the curl, the upper part with it: an S.
        let midrib = Composite::new(vec![CubicBez::new(
            Point::ZERO,
            Point::ZERO + d * (0.35 * l) - n * (0.1 * l),
            Point::ZERO + d * (0.7 * l) - n * (0.02 * l),
            Point::ZERO + d * l + n * (0.06 * l),
        )]);
        let leaf = Leaf::new(midrib, p.with_aspect(rng.uniform(0.62, 0.72)));
        let style = LeafStyle::from_palette(palette, false, rng.uniform(0.2, 0.5));
        add("acanthus", leaf_motif(&s, &leaf, &style, acc));
    }

    // Fern frond: a long pinnate blade on an arching midrib.
    {
        let s = seed.child("fern-frond");
        let mut rng = s.rng();
        let pinnae = rng.uniform(14.0, 20.0) as u32;
        let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
        let mid = crate::geom::curl::organic_curve(
            &mut rng,
            Point::new(0.0, l),
            Point::new(side * 0.35 * l, 0.0),
            0.15,
            side * 0.55,
        );
        let p = LeafParams::new(LeafShape::Fern { pinnae })
            .with_aspect(0.3)
            .with_tip_curl(side * 0.5)
            .jittered(&mut rng, 0.3);
        let leaf = Leaf::new(mid, p);
        let style = LeafStyle::from_palette(palette, false, rng.uniform(0.3, 0.6));
        let stipe = leaf.midrib().point(0.0);
        let tg = leaf.midrib().tangent(0.0);
        let mut g = Node::group(vec![stalk(
            &CubicBez::new(
                stipe - tg * (0.16 * l) + Vec2::new(0.0, 0.02 * l),
                stipe - tg * (0.1 * l),
                stipe - tg * (0.05 * l),
                stipe + tg * (0.02 * l),
            ),
            0.02 * l,
            0.012 * l,
            style.fill,
            style.ink,
            acc,
        )]);
        g.push(leaf_motif(&s, &leaf, &style, acc));
        add("fern-frond", g);
    }

    // Willow sprig: a slender arching twig with alternate narrow leaves.
    {
        let s = seed.child("willow-sprig");
        add("willow-sprig", willow_sprig(&s, palette, l, acc));
    }

    // Heart leaf on a curved petiole.
    {
        let s = seed.child("heart-leaf");
        let mut rng = s.rng();
        let dir = Vec2::from_angle(-PI / 2.0 + rng.jitter(0.0, 0.35));
        let base = Point::new(0.0, 0.0);
        let p = LeafParams::new(LeafShape::Cordate)
            .with_tip_curl(rng.jitter(0.0, 0.4))
            .jittered(&mut rng, 0.4);
        let leaf = leaf_at(base, dir, 0.8 * l, 0.62 * l, p);
        let style = LeafStyle::from_palette(palette, false, rng.uniform(0.25, 0.55));
        let pet = CubicBez::new(
            base + Vec2::new(-0.12 * l, 0.32 * l),
            base + Vec2::new(-0.1 * l, 0.18 * l),
            base - dir * (0.08 * l),
            base + dir * (0.02 * l),
        );
        let g = Node::group(vec![
            stalk(&pet, 0.026 * l, 0.017 * l, style.fill, style.ink, acc),
            leaf_motif(&s, &leaf, &style, acc),
        ]);
        add("heart-leaf", g);
    }

    // Serrated leaf: an elliptic toothed blade.
    {
        let s = seed.child("serrated");
        let mut rng = s.rng();
        let teeth = rng.uniform(18.0, 28.0) as u32;
        let p = LeafParams::new(LeafShape::Serrated { teeth })
            .with_tip_curl(rng.jitter(0.0, 0.5))
            .jittered(&mut rng, 0.5);
        let leaf = leaf_at(
            Point::ZERO,
            Vec2::from_angle(-PI / 2.0 + rng.jitter(0.0, 0.4)),
            l,
            0.38 * l,
            p,
        );
        let style = LeafStyle::from_palette(palette, false, rng.uniform(0.15, 0.45));
        add("serrated", leaf_motif(&s, &leaf, &style, acc));
    }

    // Ivy cluster: 3–5 overlapping leaves radiating from one node.
    {
        let s = seed.child("ivy-cluster");
        add("ivy-cluster", ivy_cluster(&s, palette, l, acc));
    }

    // Laurel pair: two opposite lanceolate leaves on a short twig.
    {
        let s = seed.child("laurel-pair");
        let mut rng = s.rng();
        let style = LeafStyle::from_palette(palette, false, rng.uniform(0.2, 0.5));
        let twig = CubicBez::new(
            Point::new(0.0, 0.45 * l),
            Point::new(0.03 * l, 0.25 * l),
            Point::new(-0.02 * l, 0.1 * l),
            Point::new(0.0, -0.05 * l),
        );
        let mut g = Node::group(vec![stalk(
            &twig,
            0.03 * l,
            0.018 * l,
            style.fill,
            style.ink,
            acc,
        )]);
        let mut leaves = Vec::new();
        for (i, side) in [1.0, -1.0].into_iter().enumerate() {
            let t = twig.eval(0.55 + 0.25 * i as f64);
            let a = -PI / 2.0 + side * rng.jitter(0.75, 0.15);
            let p = LeafParams::new(LeafShape::Lanceolate)
                .with_tip_curl(-side * rng.uniform(0.2, 0.5))
                .jittered(&mut rng, 0.4);
            let leaf = leaf_at(t, Vec2::from_angle(a), 0.65 * l, 0.17 * l, p);
            leaves.push(Stacked::new(
                &leaf,
                leaf_motif(&s.index(i as u64), &leaf, &style, acc),
                acc,
            ));
        }
        for n in stack_with_shadows(leaves, style.ink) {
            g.push(n);
        }
        add("laurel-pair", g);
    }

    out
}

/// A slender arching twig with alternate willow leaves, sizes decreasing to the tip.
fn willow_sprig(seed: &SeedTree, palette: &Palette, l: f64, acc: f64) -> Node {
    let mut rng = seed.rng();
    let side0 = if rng.chance(0.5) { 1.0 } else { -1.0 };
    let twig = crate::geom::curl::organic_curve(
        &mut rng,
        Point::new(0.0, 0.5 * l),
        Point::new(side0 * 0.25 * l, -0.5 * l),
        0.2,
        side0 * 0.4,
    );
    let style = LeafStyle::from_palette(palette, false, rng.uniform(0.3, 0.6));
    let al = twig.arc_length(128);
    let w0 = 0.014 * l;
    let mut nodes = vec![stalk(
        &twig,
        w0,
        0.35 * w0,
        style.fill.mix(style.ink, 0.3),
        style.ink,
        acc,
    )];
    let n = 15;
    for i in 0..n {
        let u = lerp(0.08, 0.97, i as f64 / (n - 1) as f64) + rng.jitter(0.0, 0.015);
        let t = al.t_at_fraction(u.clamp(0.0, 1.0));
        let side = if i % 2 == 0 { 1.0 } else { -1.0 };
        let tg = twig.tangent(t);
        let nrm = Vec2::new(-tg.y, tg.x);
        let width_here = 0.5 * lerp(w0, 0.35 * w0, u);
        let base = twig.point(t) + nrm * (side * width_here);
        // Alternate leaves at varied angles, the lower ones spreading wider.
        let a = side * rng.uniform(0.3, 0.75) * lerp(1.15, 0.75, u);
        let dir = rotate(tg, a);
        let len = l * lerp(0.42, 0.2, u) * rng.uniform(0.85, 1.12);
        let p = LeafParams::new(LeafShape::Willow)
            .with_tip_curl(
                side * rng.uniform(0.05, 0.55) * if rng.chance(0.8) { 1.0 } else { -1.0 },
            )
            .jittered(&mut rng, 0.4);
        let leaf = leaf_at(base, dir, len, 0.16 * len * rng.uniform(0.9, 1.1), p);
        let st = LeafStyle::from_palette(palette, false, rng.uniform(0.25, 0.65));
        nodes.push(leaf_motif(&seed.index(i), &leaf, &st, acc));
    }
    Node::group(nodes)
}

/// 3–5 broad leaves on petioles radiating from one node, the middle one drawn last.
fn ivy_cluster(seed: &SeedTree, palette: &Palette, l: f64, acc: f64) -> Node {
    let mut rng = seed.rng();
    let k = rng.uniform(3.0, 6.0) as usize;
    let node = Point::new(0.0, 0.25 * l);
    let spread = lerp(1.6, 2.4, (k as f64 - 3.0) / 2.0);
    // Indices ordered outside → in so the central leaf overlaps the others.
    let mut order: Vec<usize> = (0..k).collect();
    let mid = (k as f64 - 1.0) / 2.0;
    order.sort_by(|a, b| {
        let da = (*a as f64 - mid).abs();
        let db = (*b as f64 - mid).abs();
        db.total_cmp(&da)
    });
    let shapes = [
        LeafShape::Cordate,
        LeafShape::Ovate,
        LeafShape::Acanthus { lobes: 3 },
    ];
    let shape = *rng.pick(&shapes);
    let mut stalks = Vec::new();
    let mut leaves = Vec::new();
    for i in order {
        let f = if k > 1 {
            i as f64 / (k - 1) as f64
        } else {
            0.5
        };
        let a = -PI / 2.0 + (f - 0.5) * spread + rng.jitter(0.0, 0.12);
        let dir = Vec2::from_angle(a);
        let central = 1.0 - (f - 0.5).abs() * 2.0;
        let pet_len = l * lerp(0.1, 0.22, central) * rng.uniform(0.85, 1.15);
        let base = node + dir * pet_len;
        let len = l * lerp(0.36, 0.5, central) * rng.uniform(0.9, 1.1);
        let p = LeafParams::new(shape)
            .with_asymmetry((f - 0.5) * 0.8)
            .with_tip_curl(rng.jitter(0.0, 0.3))
            .jittered(&mut rng, 0.3);
        let width = len
            * if shape == LeafShape::Ovate {
                0.62
            } else {
                0.72
            };
        let leaf = leaf_at(base, dir, len, width, p);
        let st = LeafStyle::from_palette(palette, false, rng.uniform(0.2, 0.6));
        let pet = kurbo::Line::new(node, base + dir * (0.03 * len));
        stalks.push(stalk(
            &pet,
            0.022 * l,
            0.014 * l,
            st.fill.mix(st.ink, 0.2),
            st.ink,
            acc,
        ));
        leaves.push(Stacked::new(
            &leaf,
            leaf_motif(&seed.index(i as u64), &leaf, &st, acc),
            acc,
        ));
    }
    stalks.extend(stack_with_shadows(leaves, palette.ink));
    Node::group(stalks)
}

/// A drawn leaf with the outline used to cast and receive overlap shadows.
struct Stacked {
    node: Node,
    outline: BezPath,
    length: f64,
}

impl Stacked {
    fn new<C: Curve>(leaf: &Leaf<C>, node: Node, acc: f64) -> Stacked {
        Stacked {
            node,
            outline: leaf_outline(leaf, acc),
            length: leaf.length(),
        }
    }
}

/// Leaves in drawing order; before each leaf, a thin dark line along its outline (offset
/// down and to the right, away from the light) is drawn on every leaf already under it,
/// clipped to that leaf, so overlapping leaves read as stacked.
fn stack_with_shadows(leaves: Vec<Stacked>, ink: Color) -> Vec<Node> {
    use kurbo::Shape;
    let mut out = Vec::new();
    for (j, top) in leaves.iter().enumerate() {
        let len = top.length;
        if top.outline.elements().len() > 2 && len.is_finite() && len > 1e-6 {
            let mut cast = top.outline.clone();
            cast.apply_affine(Affine::translate(Vec2::new(0.012, 0.016) * len));
            let bb = cast.bounding_box();
            let style = Style::stroke(ink, 0.03 * len)
                .with_cap(LineCap::Round)
                .with_opacity(0.4);
            for under in &leaves[..j] {
                if under.outline.elements().len() < 3
                    || bb.intersect(under.outline.bounding_box()).area() <= 0.0
                {
                    continue;
                }
                out.push(Node::path(cast.clone(), style.clone()).clipped(under.outline.clone()));
            }
        }
        out.push(top.node.clone());
    }
    out
}

/// Rotate `v` by `angle` radians.
pub(crate) fn rotate(v: Vec2, angle: f64) -> Vec2 {
    let (s, c) = angle.sin_cos();
    Vec2::new(c * v.x - s * v.y, s * v.x + c * v.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::palette;
    use kurbo::Shape;

    fn pal() -> Palette {
        palette::all().remove(0)
    }

    fn all_paths(n: &Node) -> Vec<BezPath> {
        let mut v = Vec::new();
        n.for_each_path(|p, _, t| {
            let mut q = p.clone();
            q.apply_affine(t);
            v.push(q);
        });
        v
    }

    #[test]
    fn small_leaves_drop_fine_engraving_first() {
        let leaf = Leaf::from_base(
            Point::ZERO,
            Vec2::new(1.0, 0.2),
            200.0,
            90.0,
            LeafParams::new(LeafShape::Ovate),
        );
        let style = LeafStyle::from_palette(&pal(), true, 0.4);
        let seed = SeedTree::new(7);
        let draw = |d: Detail| leaf_motif_with_detail(&seed, &leaf, &style, 0.05, d);
        let full = leaf_motif(&seed, &leaf, &style, 0.05);
        assert_eq!(draw(Detail::FULL), full);
        assert_eq!(draw(Detail::at_scale(1.0)), full);
        // 200 units at these scales: 36 px (no hatching), 20 px (no veinlets), 10 px.
        let counts: Vec<usize> = [0.18, 0.1, 0.05]
            .into_iter()
            .map(|s| draw(Detail::at_scale(s)).path_count())
            .collect();
        assert!(full.path_count() > counts[0], "{counts:?}");
        assert!(
            counts[0] >= counts[1] && counts[1] > counts[2],
            "{counts:?}"
        );
        // The smallest keeps fill, lit half, midrib (one merged vein path) and outline.
        assert_eq!(counts[2], 4, "{counts:?}");
    }

    #[test]
    fn variants_build_fit_and_are_clean() {
        for s in 1..=5u64 {
            let vs = variants(&SeedTree::new(s), &pal(), 200.0, 0.2);
            assert!(vs.len() >= 6);
            let names: std::collections::BTreeSet<_> = vs.iter().map(|(n, _)| n.clone()).collect();
            assert_eq!(names.len(), vs.len(), "names are distinct");
            for (name, node) in &vs {
                assert!(node.is_finite(), "{name} seed {s} not finite");
                assert!(node.path_count() > 3, "{name} seed {s} is empty");
                let bb = node.bounding_box().expect("has geometry");
                assert!(
                    bb.x0 >= -100.0 - 1e-6 && bb.x1 <= 100.0 + 1e-6,
                    "{name} seed {s} {bb:?}"
                );
                assert!(
                    bb.y0 >= -100.0 - 1e-6 && bb.y1 <= 100.0 + 1e-6,
                    "{name} {bb:?}"
                );
                for p in all_paths(node) {
                    assert!(path_is_finite(&p));
                    assert_eq!(degenerate_segments(&p, 1e-9), 0, "{name} seed {s}");
                }
            }
        }
    }

    #[test]
    fn stacked_leaves_cast_clipped_shadows_on_the_ones_below() {
        let st = LeafStyle::from_palette(&pal(), false, 0.5);
        let mk = |x: f64| {
            let leaf = leaf_at(
                Point::new(x, 0.0),
                Vec2::new(0.0, -1.0),
                100.0,
                50.0,
                LeafParams::new(LeafShape::Ovate),
            );
            Stacked::new(&leaf, leaf_motif(&SeedTree::new(1), &leaf, &st, 0.2), 0.2)
        };
        // Overlapping pair: one shadow before the top leaf; far apart: none.
        assert_eq!(stack_with_shadows(vec![mk(0.0), mk(20.0)], st.ink).len(), 3);
        assert_eq!(
            stack_with_shadows(vec![mk(0.0), mk(500.0)], st.ink).len(),
            2
        );
    }

    #[test]
    fn variants_are_deterministic() {
        let a = variants(&SeedTree::new(3), &pal(), 150.0, 0.2);
        let b = variants(&SeedTree::new(3), &pal(), 150.0, 0.2);
        assert_eq!(a, b);
        let c = variants(&SeedTree::new(4), &pal(), 150.0, 0.2);
        assert_ne!(a, c);
    }

    fn shapes() -> Vec<LeafShape> {
        vec![
            LeafShape::Lanceolate,
            LeafShape::Ovate,
            LeafShape::Cordate,
            LeafShape::Acanthus { lobes: 5 },
            LeafShape::Serrated { teeth: 24 },
            LeafShape::Willow,
            LeafShape::Fern { pinnae: 14 },
        ]
    }

    #[test]
    fn veins_stay_inside_the_outline() {
        for (i, shape) in shapes().into_iter().enumerate() {
            for s in 1..=3u64 {
                let seed = SeedTree::new(s * 10 + i as u64);
                let mut rng = seed.rng();
                let p = LeafParams::new(shape)
                    .with_tip_curl(0.5)
                    .jittered(&mut rng, 0.8);
                let leaf = leaf_at(
                    Point::new(5.0, 7.0),
                    Vec2::new(0.3, -1.0),
                    120.0,
                    120.0 * p.aspect,
                    p,
                );
                let outline = leaf.outline(0.1);
                let veins = leaf_veins(&seed, &leaf, &VeinWidths::default());
                assert!(veins.len() > 4, "{shape:?} has veins");
                let mut total = 0;
                let mut inside = 0;
                for v in &veins {
                    let c = uv_curve(&leaf, v.uv);
                    for k in 0..=20 {
                        let p = c.point(k as f64 / 20.0);
                        total += 1;
                        if outline.winding(p) != 0 {
                            inside += 1;
                        }
                    }
                }
                let frac = inside as f64 / total as f64;
                assert!(frac >= 0.95, "{shape:?} seed {s}: {frac}");
            }
        }
    }

    #[test]
    fn lobed_leaves_get_one_vein_per_lobe() {
        let leaf = leaf_at(
            Point::ZERO,
            Vec2::new(0.0, -1.0),
            100.0,
            60.0,
            LeafParams::new(LeafShape::Acanthus { lobes: 5 }),
        );
        let veins = leaf_veins(&SeedTree::new(1), &leaf, &VeinWidths::default());
        let main = veins
            .iter()
            .filter(|v| v.uv.p2.y.abs() > 0.85 && v.uv.p0.y == 0.0);
        // 5 lobes per side minus those too close to the base or tip.
        let n = main.count();
        assert!((6..=10).contains(&n), "{n}");
    }

    #[test]
    fn degenerate_leaves_do_not_panic() {
        let st = LeafStyle::from_palette(&pal(), false, f64::NAN);
        for (len, w) in [(0.0, 0.0), (f64::NAN, 3.0), (100.0, 0.0), (1e-9, 1e-9)] {
            let leaf = leaf_at(
                Point::ZERO,
                Vec2::new(f64::NAN, 1.0),
                len,
                w,
                LeafParams::new(LeafShape::Ovate),
            );
            let n = leaf_motif(&SeedTree::new(1), &leaf, &st, f64::NAN);
            assert!(n.is_finite());
        }
        let vs = variants(&SeedTree::new(1), &pal(), f64::NAN, -1.0);
        assert!(vs.iter().all(|(_, n)| n.is_finite()));
    }

    #[test]
    fn leaf_motif_scales_widths_with_size() {
        let st = LeafStyle::from_palette(&pal(), true, 0.5);
        let width_of = |len: f64| {
            let leaf = leaf_at(
                Point::ZERO,
                Vec2::new(1.0, 0.0),
                len,
                0.5 * len,
                LeafParams::new(LeafShape::Ovate),
            );
            let n = leaf_motif(&SeedTree::new(2), &leaf, &st, 0.05 * len / 100.0);
            let mut w = 0.0;
            n.for_each_path(|_, s, _| {
                if s.stroke.is_some() && s.fill.is_none() && s.opacity == 1.0 {
                    w = s.stroke_width;
                }
            });
            w
        };
        let (a, b) = (width_of(100.0), width_of(200.0));
        assert!(a > 0.0 && (b / a - 2.0).abs() < 1e-6, "{a} {b}");
    }
}
