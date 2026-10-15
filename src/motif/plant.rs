//! Hierarchical procedural plants: stems that branch, carry leaves, tendrils and buds, and
//! end in flowers, grown by rules rather than placed independently.
//!
//! # Growth rules
//! A [`Plant`] is a tree of [`Stem`]s. Each stem is grown like a turtle from its base: it
//! starts along the direction it was given, turns towards `up` by a tropism term
//! (phototropism, or gravity when negative) and waves by a smooth sinusoidal term. On a
//! stem of level `k`:
//! - branches sprout at evenly spread arc-length positions on alternating sides (a planar
//!   phyllotaxis), leave at the branch angle to the parent tangent, and are shorter and
//!   thinner than their parent; growth stops at the level cap or when the element budget
//!   (shared out between the children) is used up;
//! - leaves sit at alternating nodes along the stem, their base on the stem edge, oriented
//!   by the stem tangent there, with sizes decreasing towards the tip and with the level;
//! - tendrils leave a node opposite its leaf and curl into a spiral;
//! - the tip ends in a flower (one species per plant unless mixed), a bud or a small leaf;
//!   with [`PlantParams::frond`] the upper part of every side stem carries one long blade
//!   (a fern frond), and unopened fronds become fiddleheads.
//!
//! # Drawing
//! [`plant_node`] draws deeper stems first so every joint is covered by its parent, stems as
//! tapered ribbons with an ink outline and a highlight, then tendrils, leaves (deepest level
//! first), buds, and the flowers on top.

use super::flower::{self, FlowerParams, FlowerStyle};
use super::foliage::{
    LeafStyle, finite_or, fit_to_box, leaf_at, leaf_motif, rotate, sane_accuracy, stalk, tapered,
};
use crate::geom::curl::{organic_curve, whiplash};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::ribbon::{Ribbon, WidthProfile};
use crate::geom::spiral::spiral_tail;
use crate::geom::{Composite, Curve, CurveExt, Surface, lerp};
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, SeedTree};
use crate::scene::{Color, LineCap, Node, Style};
use kurbo::{Affine, CubicBez, Point, Vec2};
use std::f64::consts::{PI, TAU};

/// Deepest branching level grown, whatever the parameters ask for.
pub const MAX_LEVELS: usize = 5;
/// Upper bound on the number of stems and attachments of one plant.
pub const MAX_ELEMENTS: usize = 400;
/// Upper bound on the branches sprouting from one stem.
const MAX_BRANCHES: usize = 12;
/// Upper bound on the leaves along one stem.
const MAX_LEAVES: usize = 40;
/// Stem half-width at the tip as a fraction of its base half-width.
const TIP_WIDTH: f64 = 0.4;
/// Fraction of the box left free on each side in [`variants`].
const MARGIN: f64 = 0.08;

/// Growth parameters. Lengths are in canvas units or fractions of [`height`](Self::height);
/// non-finite or out-of-range values are replaced or clamped when growing.
#[derive(Debug, Clone, PartialEq)]
pub struct PlantParams {
    /// Length of the main stem.
    pub height: f64,
    /// Base half-width of the main stem as a fraction of the height.
    pub stem_width: f64,
    /// Amplitude of the stems' wave (radians of turning, `0` = straight).
    pub stem_waviness: f64,
    /// Constant bow of the stems (radians of total turning, signed).
    pub stem_bend: f64,
    /// Whiplash hook of the main stem in turns (`0` = a plain grown stem).
    pub hook: f64,
    /// How strongly stems turn towards `up` (`< 0` droops under gravity), in `[-1, 1]`.
    pub tropism: f64,
    /// Branching depth below the main stem (capped at [`MAX_LEVELS`]).
    pub branch_levels: usize,
    /// Branches sprouting from the main stem; deeper stems get fewer.
    pub branches_per_level: usize,
    /// Arc-length range `(from, to)` of the parent where branches sprout.
    pub branch_span: (f64, f64),
    /// Angle between a branch and its parent's tangent, in radians.
    pub branch_angle: f64,
    /// Uniform jitter of the branch angle, in radians.
    pub branch_angle_jitter: f64,
    /// Branches spread in an ordered fan from `-branch_angle` to `+branch_angle` (a clump
    /// or a bunch) instead of alternating sides.
    pub fan: bool,
    /// Branch length as a fraction of the parent's (stems below the main stem use at most 0.6).
    pub branch_length: f64,
    /// `[0, 1]`: how much shorter the uppermost branches are than the lowest.
    pub branch_taper: f64,
    /// Leaf shape template (jittered per leaf).
    pub leaf: LeafParams,
    /// Length of the largest leaf as a fraction of the height.
    pub leaf_size: f64,
    /// Leaves per stem length, measured in leaf lengths (`0` = leafless).
    pub leaf_density: f64,
    /// Angle between a leaf and the stem tangent, in radians.
    pub leaf_angle: f64,
    /// Petiole length as a fraction of the leaf length (`0` = sessile leaves).
    pub petiole: f64,
    /// Side stems carry one blade along their upper part (fern fronds) instead of leaves.
    pub frond: bool,
    /// Probability that a stem ends in a flower.
    pub flower_chance: f64,
    /// Flower species allowed (names from [`flower::variants`]); empty allows all.
    pub flowers: Vec<String>,
    /// Every flower picks its own species instead of one per plant.
    pub mixed_flowers: bool,
    /// Radius of a flower on the main stem as a fraction of the height.
    pub flower_size: f64,
    /// Probability of a tendril at a leaf node (with `frond`: of a fiddlehead per frond).
    pub tendril_chance: f64,
    /// Probability that a stem without a flower ends in a bud.
    pub bud_chance: f64,
}

impl Default for PlantParams {
    fn default() -> Self {
        Self::flowering_sprig()
    }
}

impl PlantParams {
    /// A leafy stem with a few side shoots, each ending in a flower or a bud.
    pub fn flowering_sprig() -> Self {
        PlantParams {
            height: 100.0,
            stem_width: 0.014,
            stem_waviness: 0.35,
            stem_bend: 0.3,
            hook: 0.0,
            tropism: 0.35,
            branch_levels: 1,
            branches_per_level: 3,
            branch_span: (0.3, 0.75),
            branch_angle: 0.65,
            branch_angle_jitter: 0.15,
            fan: false,
            branch_length: 0.42,
            branch_taper: 0.4,
            leaf: LeafParams::new(LeafShape::Ovate),
            leaf_size: 0.15,
            leaf_density: 0.9,
            leaf_angle: 0.85,
            petiole: 0.12,
            frond: false,
            flower_chance: 0.8,
            flowers: names(&["rose", "peony", "poppy", "cosmos", "primrose"]),
            mixed_flowers: false,
            flower_size: 0.11,
            tendril_chance: 0.0,
            bud_chance: 0.8,
        }
    }

    /// A clump of arching fern fronds and curled fiddleheads from a short rhizome.
    pub fn fern_clump() -> Self {
        PlantParams {
            height: 100.0,
            stem_width: 0.03,
            stem_waviness: 0.15,
            stem_bend: 0.45,
            hook: 0.0,
            tropism: 0.15,
            branch_levels: 1,
            branches_per_level: 7,
            branch_span: (0.05, 1.0),
            branch_angle: 1.1,
            branch_angle_jitter: 0.12,
            fan: true,
            branch_length: 6.0,
            branch_taper: 0.35,
            leaf: LeafParams::new(LeafShape::Fern { pinnae: 18 }).with_aspect(0.32),
            leaf_size: 0.1,
            leaf_density: 0.0,
            leaf_angle: 0.8,
            petiole: 0.0,
            frond: true,
            flower_chance: 0.0,
            flowers: Vec::new(),
            mixed_flowers: false,
            flower_size: 0.0,
            tendril_chance: 0.3,
            bud_chance: 0.0,
        }
    }

    /// A long whiplash vine with heart-shaped leaves on petioles, tendrils and bells.
    pub fn climbing_vine() -> Self {
        PlantParams {
            height: 100.0,
            stem_width: 0.011,
            stem_waviness: 0.5,
            stem_bend: 0.0,
            hook: 0.9,
            tropism: 0.1,
            branch_levels: 1,
            branches_per_level: 2,
            branch_span: (0.25, 0.6),
            branch_angle: 0.8,
            branch_angle_jitter: 0.2,
            fan: false,
            branch_length: 0.3,
            branch_taper: 0.3,
            leaf: LeafParams::new(LeafShape::Cordate),
            leaf_size: 0.13,
            leaf_density: 1.1,
            leaf_angle: 0.95,
            petiole: 0.35,
            frond: false,
            flower_chance: 0.6,
            flowers: names(&["bellflower", "primrose", "carnation"]),
            mixed_flowers: false,
            flower_size: 0.07,
            tendril_chance: 0.45,
            bud_chance: 0.6,
        }
    }

    /// A loose bunch of flowering stems fanning out from one point.
    pub fn wild_bouquet() -> Self {
        PlantParams {
            height: 100.0,
            stem_width: 0.03,
            stem_waviness: 0.3,
            stem_bend: 0.0,
            hook: 0.0,
            tropism: 0.25,
            branch_levels: 2,
            branches_per_level: 5,
            branch_span: (0.7, 1.0),
            branch_angle: 0.55,
            branch_angle_jitter: 0.1,
            fan: true,
            branch_length: 3.2,
            branch_taper: 0.35,
            leaf: LeafParams::new(LeafShape::Lanceolate),
            leaf_size: 0.11,
            leaf_density: 0.8,
            leaf_angle: 0.6,
            petiole: 0.0,
            frond: false,
            flower_chance: 0.75,
            flowers: names(&[
                "daisy",
                "aster",
                "poppy",
                "cosmos",
                "carnation",
                "bellflower",
            ]),
            mixed_flowers: true,
            flower_size: 0.06,
            tendril_chance: 0.0,
            bud_chance: 0.7,
        }
    }

    /// Copy with every field finite and inside its working range.
    fn sanitized(&self) -> PlantParams {
        let f = |x: f64, d: f64, lo: f64, hi: f64| finite_or(x, d).clamp(lo, hi);
        let (a, b) = self.branch_span;
        let (a, b) = (f(a, 0.3, 0.0, 1.0), f(b, 0.8, 0.0, 1.0));
        PlantParams {
            height: f(self.height, 0.0, 0.0, 1e9),
            stem_width: f(self.stem_width, 0.012, 0.0, 0.2),
            stem_waviness: f(self.stem_waviness, 0.0, -3.0, 3.0),
            stem_bend: f(self.stem_bend, 0.0, -3.0, 3.0),
            hook: f(self.hook, 0.0, -3.0, 3.0),
            tropism: f(self.tropism, 0.0, -1.0, 1.0),
            branch_levels: self.branch_levels.min(MAX_LEVELS),
            branches_per_level: self.branches_per_level.min(MAX_BRANCHES),
            branch_span: (a.min(b), a.max(b)),
            branch_angle: f(self.branch_angle, 0.6, -PI, PI),
            branch_angle_jitter: f(self.branch_angle_jitter, 0.0, 0.0, PI),
            fan: self.fan,
            branch_length: f(self.branch_length, 0.5, 0.0, 10.0),
            branch_taper: f(self.branch_taper, 0.0, 0.0, 1.0),
            leaf: self.leaf,
            leaf_size: f(self.leaf_size, 0.1, 0.0, 2.0),
            leaf_density: f(self.leaf_density, 0.0, 0.0, 10.0),
            leaf_angle: f(self.leaf_angle, 0.8, -PI, PI),
            petiole: f(self.petiole, 0.0, 0.0, 2.0),
            frond: self.frond,
            flower_chance: f(self.flower_chance, 0.0, 0.0, 1.0),
            flowers: self.flowers.clone(),
            mixed_flowers: self.mixed_flowers,
            flower_size: f(self.flower_size, 0.0, 0.0, 1.0),
            tendril_chance: f(self.tendril_chance, 0.0, 0.0, 1.0),
            bud_chance: f(self.bud_chance, 0.0, 0.0, 1.0),
        }
    }
}

fn names(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// Colours of a plant.
#[derive(Debug, Clone, PartialEq)]
pub struct PlantStyle {
    /// Darkest leaf style; each leaf mixes between it and [`leaf_light`](Self::leaf_light).
    pub leaf: LeafStyle,
    /// Lightest leaf style.
    pub leaf_light: LeafStyle,
    /// Stem fill.
    pub stem: Color,
    /// Stem highlight line.
    pub stem_light: Color,
    /// Outline ink.
    pub ink: Color,
    /// Flower styles (a flower picks one by its tone); never empty when built from a palette.
    pub flowers: Vec<FlowerStyle>,
}

impl PlantStyle {
    /// Flowers from the warm (`warm = true`) or cool ramp, foliage from the other one.
    pub fn from_palette(palette: &Palette, warm: bool) -> PlantStyle {
        let leaf = LeafStyle::from_palette(palette, !warm, 0.15);
        let leaf_light = LeafStyle::from_palette(palette, !warm, 0.7);
        let ramp = |x: f64| {
            if warm {
                palette.cool_at(x)
            } else {
                palette.warm_at(x)
            }
        };
        PlantStyle {
            leaf,
            leaf_light,
            stem: ramp(0.25).mix(palette.ink, 0.15),
            stem_light: ramp(0.75).mix(palette.light, 0.2),
            ink: palette.ink,
            flowers: [0.15, 0.45, 0.75]
                .iter()
                .map(|&t| FlowerStyle::from_palette(palette, warm, t))
                .collect(),
        }
    }

    fn leaf_at(&self, tone: f64) -> LeafStyle {
        let t = finite_or(tone, 0.5).clamp(0.0, 1.0);
        let (a, b) = (&self.leaf, &self.leaf_light);
        LeafStyle {
            fill: a.fill.mix(b.fill, t),
            fill_light: a.fill_light.mix(b.fill_light, t),
            ink: a.ink,
            vein: a.vein.mix(b.vein, t),
            widths: a.widths,
        }
    }

    fn flower_at(&self, tone: f64) -> FlowerStyle {
        if self.flowers.is_empty() {
            let ink = self.ink;
            return FlowerStyle {
                petal: self.stem_light,
                petal_light: self.stem_light.lighten(0.2),
                ink,
                vein: ink,
                center: self.stem,
            };
        }
        let n = self.flowers.len();
        let i = ((finite_or(tone, 0.0).clamp(0.0, 1.0) * n as f64) as usize).min(n - 1);
        self.flowers[i]
    }
}

/// A grown plant.
pub struct Plant {
    /// The main stem; everything else hangs off it.
    pub root: Stem,
    /// Seed for drawing details (leaf hatching, flower jitter).
    pub seed: SeedTree,
}

/// One stem: a centreline, a base half-width and what grows on it.
pub struct Stem {
    /// Centreline from base to tip.
    pub curve: Composite,
    /// Half-width at the base; it tapers towards the tip.
    pub width: f64,
    /// Branching level (0 = main stem).
    pub level: usize,
    /// Branches, leaves, flowers, tendrils and buds, in growth order.
    pub children: Vec<Attachment>,
}

/// Something attached to a stem at arc-length fraction `u`.
pub enum Attachment {
    /// A side stem sprouting at `u`, leaving at `angle` (signed) to the parent tangent.
    Branch { stem: Stem, u: f64, angle: f64 },
    /// A leaf whose base sits on the stem edge at `u` on `side` (`+1` left, `-1` right,
    /// `0` for a frond blade running along the stem), optionally on a petiole.
    Leaf {
        leaf: Box<Leaf<Composite>>,
        u: f64,
        side: f64,
        petiole: Option<Composite>,
        tone: f64,
        seed: SeedTree,
    },
    /// A flower at `u` (the stem tip), drawn `scale` times its natural size.
    Flower {
        params: FlowerParams,
        u: f64,
        scale: f64,
        rotation: f64,
        tone: f64,
        seed: SeedTree,
    },
    /// A curling tendril (or fiddlehead) of base half-width `width`.
    Tendril { curve: Composite, width: f64 },
    /// A closed bud of length `size` at `u` (the stem tip).
    Bud {
        u: f64,
        size: f64,
        tone: f64,
        seed: SeedTree,
    },
}

impl Stem {
    /// Half-width at arc-length fraction `u`.
    pub fn half_width(&self, u: f64) -> f64 {
        stem_half(self.width, u)
    }

    /// Point and unit tangent at arc-length fraction `u`.
    pub fn frame(&self, u: f64) -> (Point, Vec2) {
        let al = self.curve.arc_length(96);
        let t = al.t_at_fraction(finite_or(u, 0.0).clamp(0.0, 1.0));
        (self.curve.point(t), self.curve.tangent(t))
    }

    /// Number of stems and attachments in this subtree (this stem included).
    pub fn element_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(|a| match a {
                Attachment::Branch { stem, .. } => stem.element_count(),
                _ => 1,
            })
            .sum::<usize>()
    }
}

impl Plant {
    /// Number of stems and attachments.
    pub fn element_count(&self) -> usize {
        self.root.element_count()
    }
}

fn stem_half(w0: f64, u: f64) -> f64 {
    w0 * lerp(1.0, TIP_WIDTH, finite_or(u, 0.0).clamp(0.0, 1.0).powf(0.8))
}

struct Ctx<'a> {
    p: &'a PlantParams,
    up: Vec2,
    accuracy: f64,
    /// Flower species available to this plant.
    species: Vec<(String, FlowerParams)>,
}

/// Grow a plant from `base` towards `up` (any non-zero direction; invalid values fall
/// back to screen-up). Randomness comes only from `seed`; `style` is accepted so callers
/// can size growth with drawing in mind and is currently unused by the rules.
pub fn grow(
    seed: &SeedTree,
    params: &PlantParams,
    base: Point,
    up: Vec2,
    style: &PlantStyle,
    accuracy: f64,
) -> Plant {
    let _ = style;
    let p = params.sanitized();
    let up = if up.is_finite() && up.hypot() > 1e-12 {
        up / up.hypot()
    } else {
        Vec2::new(0.0, -1.0)
    };
    let base = if base.is_finite() { base } else { Point::ZERO };
    let seed = seed.child("plant");
    let mut rng = seed.child("species").rng();
    let all = flower::variants(&seed.child("flowers"));
    let mut species: Vec<(String, FlowerParams)> = all
        .iter()
        .filter(|(n, _)| p.flowers.is_empty() || p.flowers.contains(n))
        .cloned()
        .collect();
    if species.is_empty() {
        species = all;
    }
    if !p.mixed_flowers && !species.is_empty() {
        let i = (rng.uniform(0.0, species.len() as f64) as usize).min(species.len() - 1);
        species = vec![species.swap_remove(i)];
    }
    let ctx = Ctx {
        p: &p,
        up,
        accuracy: sane_accuracy(accuracy),
        species,
    };
    let lean = rng.jitter(0.0, 0.12);
    let width = p.stem_width * p.height;
    let root = grow_stem(
        &ctx,
        &seed.child("root"),
        base,
        rotate(up, lean),
        p.height,
        width,
        0,
        0.0,
        MAX_ELEMENTS,
    );
    Plant { root, seed }
}

/// Grow one stem and, recursively, its branches within `budget` elements.
#[allow(clippy::too_many_arguments)]
fn grow_stem(
    ctx: &Ctx,
    seed: &SeedTree,
    base: Point,
    dir: Vec2,
    length: f64,
    width: f64,
    level: usize,
    outward: f64,
    budget: usize,
) -> Stem {
    let p = ctx.p;
    let mut rng = seed.rng();
    let length = finite_or(length, 0.0).max(0.0);
    let curve = if level == 0 && p.hook != 0.0 {
        whiplash(&mut rng, base, dir, length, p.hook)
    } else {
        grown_curve(&mut rng, ctx, base, dir, length, level, outward)
    };
    let mut stem = Stem {
        curve,
        width: finite_or(width, 0.0).max(0.0),
        level,
        children: Vec::new(),
    };
    if length < 1e-6 || budget <= 1 {
        return stem;
    }
    let mut left = budget - 1;
    let level_scale = 0.8f64.powi(level as i32);
    let leaf_len0 = p.leaf_size * p.height * level_scale;
    let frond = p.frond && level >= 1;

    // Leaves at alternating nodes.
    let mut nodes = Vec::new();
    if !frond && leaf_len0 > 1e-9 && p.leaf_density > 0.0 {
        let n = ((p.leaf_density * length / leaf_len0).floor() as usize).min(MAX_LEAVES);
        let side0 = if rng.chance(0.5) { 1.0 } else { -1.0 };
        for j in 0..n {
            let u = lerp(0.12, 0.9, (j as f64 + 0.5) / n as f64) + rng.jitter(0.0, 0.25 / n as f64);
            let side = if j % 2 == 0 { side0 } else { -side0 };
            nodes.push((u.clamp(0.05, 0.95), side));
        }
    }
    for (j, &(u, side)) in nodes.iter().enumerate() {
        if left < 2 {
            break;
        }
        let len = leaf_len0 * lerp(1.0, 0.55, u) * rng.uniform(0.85, 1.1);
        let s = seed.child("leaf").index(j as u64);
        stem.children
            .push(make_leaf(ctx, &s, &stem, u, side, len, &mut rng));
        left -= 1;
        if p.tendril_chance > 0.0 && rng.chance(p.tendril_chance) && left >= 2 {
            let (c, tg) = stem.frame(u);
            let nrm = Vec2::new(-tg.y, tg.x);
            let start = c - nrm * (side * 0.6 * stem.half_width(u));
            let d = rotate(tg, -side * rng.uniform(0.5, 0.9));
            let tl = len * rng.uniform(0.5, 0.8);
            stem.children.push(tendril(
                &mut rng,
                start,
                d,
                tl,
                0.5 * stem.half_width(u),
                side > 0.0,
                ctx.accuracy,
            ));
            left -= 1;
        }
    }

    // Branches, sharing out the remaining budget.
    let count = if level < p.branch_levels {
        let k = (p.branches_per_level as f64 * 0.45f64.powi(level as i32)).round() as usize;
        k.min(MAX_BRANCHES).min(left.saturating_sub(2) / 3)
    } else {
        0
    };
    // Keep a couple of elements in reserve for the tip.
    let share = left.saturating_sub(2).checked_div(count).unwrap_or(0);
    let side0 = if rng.chance(0.5) { 1.0 } else { -1.0 };
    let (a, b) = p.branch_span;
    for i in 0..count {
        let f = (i as f64 + 0.5) / count as f64;
        let u = (lerp(a, b, f) + rng.jitter(0.0, 0.3 * (b - a) / count as f64)).clamp(0.02, 0.98);
        let side = if i % 2 == 0 { side0 } else { -side0 };
        let angle = if p.fan {
            let x = if count > 1 {
                2.0 * i as f64 / (count - 1) as f64 - 1.0
            } else {
                0.0
            };
            x * p.branch_angle + rng.jitter(0.0, p.branch_angle_jitter)
        } else {
            side * (p.branch_angle + rng.jitter(0.0, p.branch_angle_jitter))
        };
        let (c, tg) = stem.frame(u);
        let ratio = if level == 0 {
            p.branch_length
        } else {
            p.branch_length.min(0.6)
        };
        let blen =
            length * ratio * lerp(1.0, 1.0 - 0.55 * p.branch_taper, f) * rng.uniform(0.85, 1.1);
        let s = seed.child("branch").index(i as u64);
        let fiddle = p.frond && level == 0 && rng.chance(p.tendril_chance);
        if fiddle {
            let d = rotate(tg, angle * 0.6);
            let mut r = s.rng();
            stem.children.push(tendril(
                &mut r,
                c,
                d,
                0.35 * blen,
                0.6 * stem.half_width(u),
                side < 0.0,
                ctx.accuracy,
            ));
            left = left.saturating_sub(1);
            continue;
        }
        let child = grow_stem(
            ctx,
            &s,
            c,
            rotate(tg, angle),
            blen,
            0.72 * stem.half_width(u),
            level + 1,
            angle,
            share,
        );
        left = left.saturating_sub(child.element_count());
        stem.children.push(Attachment::Branch {
            stem: child,
            u,
            angle,
        });
    }

    // Fern frond: one blade along the upper part of a side stem.
    if frond {
        let u0 = rng.uniform(0.15, 0.25);
        let pts: Vec<Point> = (0..=24)
            .map(|i| {
                let (c, _) = stem.frame(lerp(u0, 1.0, i as f64 / 24.0));
                c
            })
            .collect();
        if let Some(mid) = Composite::through_points(&pts, 0.5) {
            let mut lp = p.leaf.jittered(&mut rng, 0.3);
            lp.tip_curl = 0.0;
            let leaf = Box::new(Leaf::new(mid, lp));
            stem.children.push(Attachment::Leaf {
                leaf,
                u: u0,
                side: 0.0,
                petiole: None,
                tone: rng.uniform(0.0, 1.0),
                seed: seed.child("frond"),
            });
        }
        return stem;
    }

    // A stem whose branches sprout up to its tip ends in them (a clump, a bunch).
    if count > 0 && p.branch_span.1 > 0.9 {
        return stem;
    }
    // The tip: a flower, else a bud, else a small terminal leaf.
    let flower_r = p.flower_size * p.height * 0.85f64.powi(level as i32);
    if !ctx.species.is_empty() && flower_r > 1e-9 && rng.chance(p.flower_chance) {
        let i = (rng.uniform(0.0, ctx.species.len() as f64) as usize).min(ctx.species.len() - 1);
        let params = ctx.species[i].1.clone();
        let outer = params.outer_radius();
        if outer.is_finite() && outer > 1e-9 {
            stem.children.push(Attachment::Flower {
                params,
                u: 1.0,
                scale: flower_r / outer,
                rotation: rng.uniform(0.0, TAU),
                tone: rng.uniform(0.0, 1.0),
                seed: seed.child("flower"),
            });
            return stem;
        }
    }
    if flower_r > 1e-9 && rng.chance(p.bud_chance) {
        stem.children.push(Attachment::Bud {
            u: 1.0,
            size: 0.9 * flower_r * rng.uniform(0.8, 1.1),
            tone: rng.uniform(0.0, 1.0),
            seed: seed.child("bud"),
        });
    } else if leaf_len0 > 1e-9 {
        let len = 0.6 * leaf_len0;
        let (c, tg) = stem.frame(1.0);
        let lp = p.leaf.jittered(&mut rng, 0.4);
        let leaf = Box::new(leaf_at(c - tg * (0.05 * len), tg, len, len * lp.aspect, lp));
        stem.children.push(Attachment::Leaf {
            leaf,
            u: 1.0,
            side: 0.0,
            petiole: None,
            tone: rng.uniform(0.0, 1.0),
            seed: seed.child("tip-leaf"),
        });
    }
    stem
}

/// Turtle growth: start along `dir`, turn towards `up` by the tropism, bow and wave.
fn grown_curve(
    rng: &mut ChaCha8Rng,
    ctx: &Ctx,
    base: Point,
    dir: Vec2,
    length: f64,
    level: usize,
    outward: f64,
) -> Composite {
    const N: usize = 16;
    let p = ctx.p;
    let phase = rng.uniform(0.0, TAU);
    let freq = rng.uniform(0.6, 1.3);
    let coin = if rng.chance(0.5) { 1.0 } else { -1.0 };
    // Side stems bow away from their parent (they sag outward); the main stem either way.
    let side = if level > 0 && outward != 0.0 {
        outward.signum()
    } else {
        coin
    };
    // A fan's base stays straight so the fan opens symmetrically around `up`.
    let base_bend = if level == 0 && p.fan {
        0.0
    } else {
        p.stem_bend
    };
    let bend = base_bend * side * rng.uniform(0.7, 1.2);
    let wave = p.stem_waviness * rng.uniform(0.7, 1.2);
    let jit: Vec<f64> = (0..N).map(|_| rng.jitter(0.0, 1.0)).collect();
    let mut d = if dir.is_finite() && dir.hypot() > 1e-12 {
        dir / dir.hypot()
    } else {
        ctx.up
    };
    if length < 1e-6 {
        return Composite::new(vec![CubicBez::new(
            base,
            base + d * 3e-7,
            base + d * 6e-7,
            base + d * 1e-6,
        )]);
    }
    let ds = length / N as f64;
    // Side stems respond more to light/gravity than the main stem.
    let tropism = p.tropism * if level == 0 { 0.6 } else { 1.0 };
    let mut pts = vec![base];
    let mut q = base;
    for (k, j) in jit.iter().enumerate() {
        let f = (k as f64 + 0.5) / N as f64;
        let to_up = d.cross(ctx.up).atan2(d.dot(ctx.up));
        let turn = tropism * to_up * (2.5 / N as f64)
            + wave * (2.0 / N as f64) * (TAU * freq * f + phase).cos()
            + bend / N as f64
            + 0.04 * j;
        d = rotate(d, turn);
        q += d * ds;
        pts.push(q);
    }
    Composite::through_points(&pts, 0.5)
        .unwrap_or_else(|| Composite::new(vec![CubicBez::new(base, base, q, q)]))
}

/// A leaf at node `u` of `stem` on `side`, its base on the stem edge (or on a petiole
/// leaving from there), oriented by the stem tangent.
fn make_leaf(
    ctx: &Ctx,
    seed: &SeedTree,
    stem: &Stem,
    u: f64,
    side: f64,
    len: f64,
    rng: &mut ChaCha8Rng,
) -> Attachment {
    let p = ctx.p;
    let (c, tg) = stem.frame(u);
    let nrm = Vec2::new(-tg.y, tg.x);
    let edge = c + nrm * (side * 0.85 * stem.half_width(u));
    let dir = rotate(tg, side * (p.leaf_angle + rng.jitter(0.0, 0.15)));
    // Most leaves droop outward (curl away from the stem), a few curl back.
    let curl = side * rng.uniform(0.15, 0.55) * if rng.chance(0.8) { 1.0 } else { -1.0 };
    let lp = p.leaf.with_tip_curl(curl).jittered(rng, 0.35);
    let pet_len = p.petiole * len;
    let (base, petiole, leaf_dir) = if pet_len > 1e-6 {
        // The petiole arches out from the node and the blade continues its end tangent.
        let bend = rotate(dir, -side * 0.25);
        let end = edge + bend * pet_len;
        let pc = CubicBez::new(
            edge,
            edge + rotate(tg, side * 0.5 * (p.leaf_angle + 0.6)) * (0.4 * pet_len),
            end - bend * (0.35 * pet_len),
            end,
        );
        (end, Some(Composite::new(vec![pc])), bend)
    } else {
        (edge, None, dir)
    };
    let leaf = Box::new(leaf_at(base, leaf_dir, len, len * lp.aspect, lp));
    Attachment::Leaf {
        leaf,
        u,
        side,
        petiole,
        tone: rng.uniform(0.0, 1.0),
        seed: *seed,
    }
}

/// A tendril: a short waving lead from `start` along `dir` ending in a spiral curl.
fn tendril(
    rng: &mut ChaCha8Rng,
    start: Point,
    dir: Vec2,
    length: f64,
    width: f64,
    clockwise: bool,
    accuracy: f64,
) -> Attachment {
    let len = finite_or(length, 0.0).max(1e-6);
    let bow = if clockwise { -0.4 } else { 0.4 };
    let lead = organic_curve(rng, start, start + dir * (0.6 * len), 0.15, bow);
    let turns = rng.uniform(1.3, 2.3);
    let curve = spiral_tail(&lead, turns, 0.22, clockwise, (accuracy * 0.5).max(1e-3));
    Attachment::Tendril {
        curve,
        width: finite_or(width, 0.0).max(0.0),
    }
}

/// Draw a plant: deeper stems first, then tendrils, leaves, buds and flowers on top.
pub fn plant_node(plant: &Plant, style: &PlantStyle, accuracy: f64) -> Node {
    let acc = sane_accuracy(accuracy);
    let mut layers = Layers::default();
    collect(&plant.root, style, acc, &mut layers);
    let mut root = Node::empty().with_id("plant");
    layers
        .stems
        .sort_by_key(|(level, _)| std::cmp::Reverse(*level));
    layers
        .leaves
        .sort_by_key(|(level, _)| std::cmp::Reverse(*level));
    for (_, n) in layers.stems {
        root.push(n);
    }
    for n in layers.tendrils {
        root.push(n);
    }
    for (_, n) in layers.leaves {
        root.push(n);
    }
    for n in layers.tops {
        root.push(n);
    }
    root
}

/// Grow and draw in one call.
pub fn plant(
    seed: &SeedTree,
    params: &PlantParams,
    base: Point,
    up: Vec2,
    style: &PlantStyle,
    accuracy: f64,
) -> Node {
    plant_node(
        &grow(seed, params, base, up, style, accuracy),
        style,
        accuracy,
    )
}

#[derive(Default)]
struct Layers {
    stems: Vec<(usize, Node)>,
    tendrils: Vec<Node>,
    leaves: Vec<(usize, Node)>,
    tops: Vec<Node>,
}

fn collect(stem: &Stem, style: &PlantStyle, acc: f64, out: &mut Layers) {
    out.stems.push((stem.level, stem_node(stem, style, acc)));
    for a in &stem.children {
        match a {
            Attachment::Branch { stem, .. } => collect(stem, style, acc, out),
            Attachment::Leaf {
                leaf,
                petiole,
                tone,
                seed,
                ..
            } => {
                let ls = style.leaf_at(*tone);
                let mut g = Node::empty();
                if let Some(pc) = petiole {
                    let w = (0.6 * stem.half_width(1.0)).max(0.01 * leaf.length());
                    g.push(stalk(pc, 2.0 * w, 1.4 * w, style.stem, style.ink, acc));
                }
                g.push(leaf_motif(seed, leaf, &ls, acc));
                out.leaves.push((stem.level, g));
            }
            Attachment::Tendril { curve, width } => {
                let w = 2.0 * width;
                let body = tapered(curve, |t| w * (1.0 - 0.85 * t), acc);
                if !body.elements().is_empty() {
                    let s =
                        Style::fill_stroke(style.stem, style.ink, 0.2 * w).with_cap(LineCap::Round);
                    out.tendrils.push(Node::path(body, s));
                }
            }
            Attachment::Flower {
                params,
                u,
                scale,
                rotation,
                tone,
                seed,
            } => {
                let (c, tg) = stem.frame(*u);
                let r = params.outer_radius() * scale;
                let fs = style.flower_at(*tone);
                let mut g = Node::empty();
                // Calyx: sepals spreading from under the flower.
                let ls = style.leaf_at(0.3);
                for (k, a) in [-0.9, 0.0, 0.9].into_iter().enumerate() {
                    let d = rotate(tg, a);
                    let sp = LeafParams::new(LeafShape::Lanceolate).with_tip_curl(0.3 * a);
                    let sepal = leaf_at(c - tg * (0.2 * r), d, 0.75 * r, 0.24 * r, sp);
                    g.push(leaf_motif(&seed.index(k as u64), &sepal, &ls, acc));
                }
                let fl = flower::flower(seed, params, &fs, acc / scale.max(1e-9));
                g.push(fl.with_transform(
                    Affine::translate(c.to_vec2())
                        * Affine::rotate(*rotation)
                        * Affine::scale(*scale),
                ));
                out.tops.push(g);
            }
            Attachment::Bud {
                u,
                size,
                tone,
                seed,
            } => {
                let (c, tg) = stem.frame(*u);
                out.tops
                    .push(bud_node(seed, c, tg, *size, *tone, style, acc));
            }
        }
    }
}

/// A stem as a tapered ribbon with an ink outline and a highlight along one side.
fn stem_node(stem: &Stem, style: &PlantStyle, acc: f64) -> Node {
    let w0 = stem.width;
    if !(w0.is_finite() && w0 > 1e-9) {
        return Node::empty();
    }
    let ribbon = Ribbon::symmetric(
        stem.curve.clone(),
        WidthProfile::custom(move |u| stem_half(w0, u)),
    );
    let outline = ribbon.outline(acc);
    if outline.elements().len() < 3 {
        return Node::empty();
    }
    let mut g = Node::group(vec![Node::path(
        outline.clone(),
        Style::fill_stroke(style.stem, style.ink, 0.35 * w0),
    )]);
    let hl = ribbon.contour(0.4, acc);
    if !hl.elements().is_empty() {
        g.push(Node::path(
            hl,
            Style::stroke(style.stem_light, 0.3 * w0).with_cap(LineCap::Round),
        ));
    }
    g
}

/// A closed bud: a pointed ovate body in flower colours held by two sepals.
fn bud_node(
    seed: &SeedTree,
    c: Point,
    tg: Vec2,
    size: f64,
    tone: f64,
    style: &PlantStyle,
    acc: f64,
) -> Node {
    let fs = style.flower_at(tone);
    let body_style = LeafStyle {
        fill: fs.petal,
        fill_light: fs.petal_light,
        ink: fs.ink,
        vein: fs.vein,
        widths: style.leaf.widths,
    };
    let bp = LeafParams::new(LeafShape::Ovate)
        .with_points(0.2, 0.9)
        .with_tip_curl(0.0);
    let body = leaf_at(c - tg * (0.05 * size), tg, size, 0.55 * size, bp);
    let mut g = Node::group(vec![leaf_motif(seed, &body, &body_style, acc)]);
    let ls = style.leaf_at(0.3);
    for (k, side) in [1.0, -1.0].into_iter().enumerate() {
        let sp = LeafParams::new(LeafShape::Lanceolate).with_tip_curl(-side * 0.25);
        let d = rotate(tg, side * 0.22);
        let sepal = leaf_at(c - tg * (0.08 * size), d, 0.7 * size, 0.28 * size, sp);
        g.push(leaf_motif(&seed.index(k as u64), &sepal, &ls, acc));
    }
    g
}

/// Named demonstration plants, each fitting a `size` × `size` box centred at the origin.
pub fn variants(
    seed: &SeedTree,
    palette: &Palette,
    size: f64,
    accuracy: f64,
) -> Vec<(String, Node)> {
    let seed = seed.child("plant-variants");
    let size = if size.is_finite() && size > 0.0 {
        size
    } else {
        100.0
    };
    let acc = sane_accuracy(accuracy);
    let presets: [(&str, PlantParams, bool, Vec2); 4] = [
        (
            "flowering-sprig",
            PlantParams::flowering_sprig(),
            true,
            Vec2::new(0.0, -1.0),
        ),
        (
            "fern-clump",
            PlantParams::fern_clump(),
            true,
            Vec2::new(0.0, -1.0),
        ),
        (
            "climbing-vine",
            PlantParams::climbing_vine(),
            true,
            Vec2::new(0.35, -1.0),
        ),
        (
            "wild-bouquet",
            PlantParams::wild_bouquet(),
            true,
            Vec2::new(0.0, -1.0),
        ),
    ];
    presets
        .into_iter()
        .map(|(name, mut params, warm, up)| {
            let s = seed.child(name);
            params.height = if name == "fern-clump" || name == "wild-bouquet" {
                0.15 * size
            } else {
                size
            };
            if name == "fern-clump" || name == "wild-bouquet" {
                // Rescale the fractions of the short base stem to the whole plant.
                let k = size / params.height;
                params.leaf_size *= k;
                params.flower_size *= k;
                params.stem_width *= 0.35 * k;
            }
            let style = PlantStyle::from_palette(palette, warm);
            let node = plant(&s, &params, Point::ZERO, up, &style, acc);
            (name.to_string(), fit_to_box(node, size, MARGIN))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::palette;

    fn pal() -> Palette {
        palette::all().remove(0)
    }

    #[test]
    fn variants_build_fit_and_are_clean() {
        for s in 1..=5u64 {
            let vs = variants(&SeedTree::new(s), &pal(), 300.0, 0.3);
            assert!(vs.len() >= 4);
            for (name, node) in &vs {
                assert!(node.is_finite(), "{name} seed {s}");
                assert!(node.path_count() > 10, "{name} seed {s} too sparse");
                let bb = node.bounding_box().expect("geometry");
                let h = 150.0 + 1e-6;
                assert!(
                    bb.x0 >= -h && bb.x1 <= h && bb.y0 >= -h && bb.y1 <= h,
                    "{name} {bb:?}"
                );
                node.for_each_path(|p, _, _| {
                    assert!(path_is_finite(p));
                    assert_eq!(degenerate_segments(p, 1e-9), 0, "{name} seed {s}");
                });
            }
        }
    }

    #[test]
    fn variants_are_deterministic() {
        let a = variants(&SeedTree::new(7), &pal(), 200.0, 0.3);
        let b = variants(&SeedTree::new(7), &pal(), 200.0, 0.3);
        assert_eq!(a, b);
    }

    fn style() -> PlantStyle {
        PlantStyle::from_palette(&pal(), true)
    }

    #[test]
    fn hierarchy_follows_growth_rules() {
        let mut p = PlantParams::flowering_sprig();
        p.branch_levels = 2;
        p.branches_per_level = 4;
        let plant = grow(
            &SeedTree::new(3),
            &p,
            Point::ZERO,
            Vec2::new(0.0, -1.0),
            &style(),
            0.2,
        );
        fn check(stem: &Stem, depth: &mut usize) {
            *depth = (*depth).max(stem.level);
            let len = stem.curve.length();
            for a in &stem.children {
                match a {
                    Attachment::Branch { stem: child, u, .. } => {
                        assert_eq!(child.level, stem.level + 1);
                        assert!(child.width < stem.width, "branches are thinner");
                        assert!(child.curve.length() < len, "branches are shorter");
                        // The branch starts on its parent's centreline.
                        let (c, _) = stem.frame(*u);
                        assert!((child.curve.start() - c).hypot() < 1e-6);
                        check(child, depth);
                    }
                    Attachment::Leaf {
                        leaf,
                        u,
                        side,
                        petiole,
                        ..
                    } if *side != 0.0 => {
                        // The leaf (or its petiole) starts on the stem edge.
                        let (c, _) = stem.frame(*u);
                        let start = petiole
                            .as_ref()
                            .map_or_else(|| leaf.midrib().point(0.0), |p| p.start());
                        let d = (start - c).hypot();
                        assert!(
                            d <= stem.half_width(*u) + 1e-6,
                            "leaf base off the stem: {d}"
                        );
                    }
                    Attachment::Flower { u, .. } | Attachment::Bud { u, .. } => {
                        assert_eq!(*u, 1.0, "flowers and buds terminate stems");
                    }
                    _ => {}
                }
            }
        }
        let mut depth = 0;
        check(&plant.root, &mut depth);
        assert_eq!(depth, 2);
    }

    #[test]
    fn leaves_shrink_towards_the_tip() {
        let mut p = PlantParams::flowering_sprig();
        p.branch_levels = 0;
        p.leaf_density = 2.0;
        let plant = grow(
            &SeedTree::new(5),
            &p,
            Point::ZERO,
            Vec2::new(0.0, -1.0),
            &style(),
            0.2,
        );
        let sizes: Vec<(f64, f64)> = plant
            .root
            .children
            .iter()
            .filter_map(|a| match a {
                Attachment::Leaf { leaf, u, side, .. } if *side != 0.0 => Some((*u, leaf.length())),
                _ => None,
            })
            .collect();
        assert!(sizes.len() >= 4);
        let n = sizes.len();
        let low: f64 = sizes[..n / 2].iter().map(|s| s.1).sum::<f64>() / (n / 2) as f64;
        let high: f64 = sizes[n / 2..].iter().map(|s| s.1).sum::<f64>() / (n - n / 2) as f64;
        assert!(high < low, "{low} {high}");
    }

    #[test]
    fn extreme_params_stay_bounded() {
        let st = style();
        let mut cases = Vec::new();
        let mut p = PlantParams::wild_bouquet();
        p.height = 0.0;
        cases.push(p);
        let mut p = PlantParams::flowering_sprig();
        p.branches_per_level = 0;
        cases.push(p);
        let mut p = PlantParams::flowering_sprig();
        p.branch_levels = 50;
        p.branches_per_level = 50;
        p.leaf_density = 50.0;
        p.tendril_chance = 1.0;
        cases.push(p);
        let mut p = PlantParams::climbing_vine();
        p.height = f64::NAN;
        p.branch_angle = f64::NAN;
        p.stem_waviness = f64::INFINITY;
        p.leaf_size = f64::NAN;
        p.flower_size = f64::NAN;
        cases.push(p);
        let mut p = PlantParams::fern_clump();
        p.branch_length = f64::NAN;
        p.branch_span = (f64::NAN, -3.0);
        cases.push(p);
        for (i, p) in cases.iter().enumerate() {
            let plant = grow(
                &SeedTree::new(1),
                p,
                Point::ZERO,
                Vec2::new(f64::NAN, 0.0),
                &st,
                0.3,
            );
            assert!(
                plant.element_count() <= MAX_ELEMENTS,
                "case {i}: {}",
                plant.element_count()
            );
            let node = plant_node(&plant, &st, f64::NAN);
            assert!(node.is_finite(), "case {i}");
            assert!(node.path_count() < 40 * MAX_ELEMENTS, "case {i}");
        }
    }
}
