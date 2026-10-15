//! Leaf silhouettes: a [`Ribbon`] around a midrib with a leaf-shaped width profile.
//!
//! The blade's half-width on each side is a single swell (see [`WidthProfile::swell`]) whose
//! base and tip powers set how rounded or pointed the ends are. Lobes, pinnae and teeth are
//! smooth multiplicative modulations of that envelope along `u`, so every edge stays a
//! normal graph over the midrib and inherits the ribbon's cusp clamp and self-intersection
//! fitting. Asymmetry widens one half and shifts its widest point; tip curl bends the last
//! part of the midrib, which is what makes a leaf read as a living leaf.
//!
//! `u` runs from the base (petiole, `u = 0`) to the tip (`u = 1`); `v = +1` is the left
//! half of the blade (along the midrib's left normal), `v = -1` the right half.

use super::curve::{Curve, CurveExt};
use super::ribbon::{Ribbon, RibbonReport, WidthProfile, ease};
use super::surface::Surface;
use super::{EPS, lerp, smoothstep};
use crate::rng::{Draw, Rng};
use kurbo::{BezPath, CubicBez, Point, Vec2};
use std::f64::consts::PI;

/// Leaf silhouette family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LeafShape {
    /// Narrow, widest below the middle, tapering to a long point.
    Lanceolate,
    /// Egg-shaped: broad, rounded base, widest near the base.
    Ovate,
    /// Heart-shaped: two rounded lobes sweep back around the petiole.
    Cordate,
    /// Deeply lobed Art Nouveau acanthus: `lobes` (3–7) big rounded lobes per side that lean
    /// towards the leaf tip and shrink towards it, each scalloped into two or three rounded
    /// sub-lobes, with deep rounded sinuses (see [`LeafParams::lobes`]).
    Acanthus {
        /// Number of lobes per side, clamped to `3..=7`.
        lobes: u32,
    },
    /// Fine forward-pointing teeth along an elliptic blade.
    Serrated {
        /// Number of teeth per side, clamped to `4..=80`.
        teeth: u32,
    },
    /// Long and very narrow with a slender point.
    Willow,
    /// Fern-like frond: rounded pinnate lobes cut almost to the midrib.
    Fern {
        /// Number of pinnae per side, clamped to `4..=40`.
        pinnae: u32,
    },
}

impl LeafShape {
    /// `(widest point along u, default aspect, default base pointedness, default tip pointedness)`.
    fn defaults(self) -> (f64, f64, f64, f64) {
        match self {
            LeafShape::Lanceolate => (0.34, 0.24, 0.6, 0.75),
            LeafShape::Ovate => (0.36, 0.55, 0.12, 0.55),
            LeafShape::Cordate => (0.3, 0.62, 0.0, 0.6),
            LeafShape::Acanthus { .. } => (0.42, 0.62, 0.35, 0.7),
            LeafShape::Serrated { .. } => (0.42, 0.36, 0.4, 0.7),
            LeafShape::Willow => (0.42, 0.1, 0.7, 0.85),
            LeafShape::Fern { .. } => (0.4, 0.34, 0.65, 0.45),
        }
    }
}

/// Length-independent leaf parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeafParams {
    /// Silhouette family.
    pub shape: LeafShape,
    /// Maximum blade width (both halves) divided by the midrib length.
    pub aspect: f64,
    /// `[-1, 1]`: positive widens the left half (and moves its widest point towards the
    /// tip) while narrowing the right half; lobes of the two halves shift out of phase.
    pub asymmetry: f64,
    /// Bend of the last 45 % of the midrib in radians; positive turns the tip towards the
    /// left half. Clamped to `[-π, π]`.
    pub tip_curl: f64,
    /// `[0, 1]`: 0 gives a rounded base, 1 a sharp wedge.
    pub base_point: f64,
    /// `[0, 1]`: 0 gives a blunt rounded tip, 1 a long acuminate point.
    pub tip_point: f64,
}

impl LeafParams {
    /// Defaults for a shape: symmetric, a slight tip curl, shape-typical aspect and points.
    pub fn new(shape: LeafShape) -> Self {
        let (_, aspect, base_point, tip_point) = shape.defaults();
        LeafParams {
            shape,
            aspect,
            asymmetry: 0.0,
            tip_curl: 0.12,
            base_point,
            tip_point,
        }
    }

    /// Set [`aspect`](Self::aspect).
    pub fn with_aspect(mut self, aspect: f64) -> Self {
        self.aspect = aspect;
        self
    }

    /// Set [`asymmetry`](Self::asymmetry).
    pub fn with_asymmetry(mut self, asymmetry: f64) -> Self {
        self.asymmetry = asymmetry;
        self
    }

    /// Set [`tip_curl`](Self::tip_curl).
    pub fn with_tip_curl(mut self, radians: f64) -> Self {
        self.tip_curl = radians;
        self
    }

    /// Set [`base_point`](Self::base_point) and [`tip_point`](Self::tip_point).
    pub fn with_points(mut self, base: f64, tip: f64) -> Self {
        self.base_point = base;
        self.tip_point = tip;
        self
    }

    /// Natural variation drawn from `rng`: asymmetry, curl, aspect and pointedness are
    /// perturbed by an amount scaled by `amount` (0 = unchanged, 1 = typical spread).
    pub fn jittered<R: Rng>(mut self, rng: &mut R, amount: f64) -> Self {
        let k = finite(amount, 0.0).clamp(0.0, 4.0);
        self.asymmetry += rng.normal(0.0, 0.25 * k);
        self.tip_curl += rng.normal(0.0, 0.2 * k);
        self.aspect *= 1.0 + rng.jitter(0.0, 0.15 * k);
        self.base_point += rng.jitter(0.0, 0.1 * k);
        self.tip_point += rng.jitter(0.0, 0.1 * k);
        self
    }

    /// Copy with every field finite and inside its documented range.
    fn sanitized(self) -> Self {
        LeafParams {
            shape: self.shape,
            aspect: finite(self.aspect, 0.3).clamp(0.0, 4.0),
            asymmetry: finite(self.asymmetry, 0.0).clamp(-1.0, 1.0),
            tip_curl: finite(self.tip_curl, 0.0).clamp(-PI, PI),
            base_point: finite(self.base_point, 0.5).clamp(0.0, 1.0),
            tip_point: finite(self.tip_point, 0.5).clamp(0.0, 1.0),
        }
    }
}

fn finite(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

/// A curve whose part after `start` is progressively rotated about `curve(start)` by
/// `angle·x²`, `x = (t − start)/(1 − start)`. The quadratic onset keeps the bend tangent
/// continuous (C1) at `start` while the tip turns by the full `angle`.
pub struct TipCurl<C: Curve> {
    /// The straight (uncurled) midrib.
    pub curve: C,
    /// Total rotation at the tip in radians (positive = towards the left normal).
    pub angle: f64,
    /// Parameter where the curl begins.
    pub start: f64,
}

impl<C: Curve> TipCurl<C> {
    /// Curl the last 45 % of `curve` by `angle` radians.
    pub fn new(curve: C, angle: f64) -> Self {
        TipCurl {
            curve,
            angle: finite(angle, 0.0).clamp(-PI, PI),
            start: 0.55,
        }
    }
}

impl<C: Curve> Curve for TipCurl<C> {
    fn point(&self, t: f64) -> Point {
        let p = self.curve.point(t);
        if t <= self.start || self.angle == 0.0 {
            return p;
        }
        let x = (t - self.start) / (1.0 - self.start);
        let (s, c) = (self.angle * x * x).sin_cos();
        let pivot = self.curve.point(self.start);
        let d = p - pivot;
        pivot + Vec2::new(c * d.x - s * d.y, s * d.x + c * d.y)
    }
}

/// A leaf: a [`Ribbon`] around a (curled) midrib with a leaf-shaped profile.
pub struct Leaf<C: Curve> {
    ribbon: Ribbon<TipCurl<C>>,
    params: LeafParams,
}

impl<C: Curve> Leaf<C> {
    /// Build a leaf around `midrib` (base at `t = 0`, tip at `t = 1`). Widths follow from
    /// the midrib length and `params.aspect`; the result is already fitted so that its
    /// outline does not self-intersect (at an accuracy of 1/400 of the length).
    pub fn new(midrib: C, params: LeafParams) -> Self {
        let params = params.sanitized();
        let curled = TipCurl::new(midrib, params.tip_curl);
        let length = curled.length();
        let half = if length.is_finite() {
            0.5 * params.aspect * length
        } else {
            0.0
        };
        let sweep = if params.shape == LeafShape::Cordate {
            0.6
        } else {
            0.0
        };
        // Lobes, teeth and pinnae lean towards the tip.
        let lean = match params.shape {
            LeafShape::Acanthus { .. } => 0.5,
            LeafShape::Serrated { .. } => 0.2,
            LeafShape::Fern { .. } => 0.45,
            _ => 0.0,
        };
        let left = side_profile(&params, 1.0, half);
        let right = side_profile(&params, -1.0, half);
        let mut ribbon = Ribbon::new(curled, left, right)
            .with_sweep(sweep)
            .with_lean(lean);
        if ribbon.length() > EPS {
            ribbon.fit_width((ribbon.length() / 400.0).max(1e-3));
        }
        Leaf { ribbon, params }
    }

    /// The midrib as drawn (including the tip curl).
    pub fn midrib(&self) -> &TipCurl<C> {
        self.ribbon.centerline()
    }

    /// The underlying ribbon (widths, contours, stripes, centreline path).
    pub fn ribbon(&self) -> &Ribbon<TipCurl<C>> {
        &self.ribbon
    }

    /// The (sanitised) parameters this leaf was built with.
    pub fn params(&self) -> &LeafParams {
        &self.params
    }

    /// Midrib arc length.
    pub fn length(&self) -> f64 {
        self.ribbon.length()
    }

    /// Validity report of the outline, see [`Ribbon::check`].
    pub fn check(&self, accuracy: f64) -> RibbonReport {
        self.ribbon.check(accuracy)
    }

    /// Re-fit the widths at a different accuracy, see [`Ribbon::fit_width`].
    pub fn fit_width(&mut self, accuracy: f64) -> f64 {
        self.ribbon.fit_width(accuracy)
    }
}

impl Leaf<CubicBez> {
    /// A leaf growing from `base` towards `direction` with the given midrib `length` and
    /// maximum blade `width` (which overrides `params.aspect`). The midrib bends gently
    /// towards the narrower half, as asymmetric leaves do. Invalid inputs fall back to a
    /// unit +x direction and zero size.
    pub fn from_base(
        base: Point,
        direction: Vec2,
        length: f64,
        width: f64,
        params: LeafParams,
    ) -> Self {
        let base = if base.is_finite() { base } else { Point::ZERO };
        let len = finite(length, 0.0).max(0.0);
        let d = if direction.is_finite() && direction.hypot() > EPS {
            direction / direction.hypot()
        } else {
            Vec2::new(1.0, 0.0)
        };
        let n = Vec2::new(-d.y, d.x);
        let p = params.sanitized();
        let bend = -0.08 * p.asymmetry * len;
        let midrib = CubicBez::new(
            base,
            base + d * (0.35 * len) + n * (0.8 * bend),
            base + d * (0.7 * len) + n * bend,
            base + d * len,
        );
        let aspect = if len > EPS {
            finite(width, 0.0).max(0.0) / len
        } else {
            p.aspect
        };
        Leaf::new(midrib, p.with_aspect(aspect))
    }
}

impl<C: Curve> Surface for Leaf<C> {
    fn point(&self, u: f64, v: f64) -> Point {
        self.ribbon.point(u, v)
    }

    fn outline(&self, accuracy: f64) -> BezPath {
        self.ribbon.outline(accuracy)
    }

    fn half_width(&self, u: f64) -> f64 {
        self.ribbon.half_width(u)
    }
}

/// Half-width profile of one blade half (`sign = +1` left, `-1` right).
fn side_profile(p: &LeafParams, sign: f64, half: f64) -> WidthProfile {
    let (peak, ..) = p.shape.defaults();
    let a = p.asymmetry * sign;
    let max = half * (1.0 + 0.22 * a);
    let peak = (peak + 0.06 * a).clamp(0.1, 0.9);
    // Power 0.45 makes a rounded end (≈ √u), 1.3–1.7 a slender concave point.
    let base_power = lerp(0.45, 1.3, p.base_point);
    let tip_power = lerp(0.6, 1.7, p.tip_point);
    let envelope = WidthProfile::swell(max, peak, base_power, tip_power);
    // The right half's lobes shift out of phase with asymmetry (alternate pinnae).
    let phase = if sign < 0.0 { 0.5 * p.asymmetry } else { 0.0 };
    match p.shape {
        LeafShape::Acanthus { lobes } => {
            let n = lobes.clamp(3, 7) as f64;
            envelope.map(move |u, w| w * acanthus_modulation(n, phase, u))
        }
        LeafShape::Serrated { teeth } => {
            let n = teeth.clamp(4, 80) as f64;
            envelope.map(move |u, w| {
                let f = fract(n * u + phase);
                let amp = smoothstep(u / 0.12) * smoothstep((1.0 - u) / 0.06);
                w * (1.0 - amp * 0.09 * (1.0 - claw(f, 0.8, 0.6)))
            })
        }
        LeafShape::Fern { pinnae } => {
            let n = pinnae.clamp(4, 40) as f64;
            envelope.map(move |u, w| {
                let f = fract(n * u + phase);
                let amp = smoothstep(u / 0.04);
                // Pinnae are longest at about 40 % of the frond and shrink smoothly towards
                // both ends, on top of the rounded envelope.
                let d = if u < 0.4 {
                    (0.4 - u) / 0.4
                } else {
                    (u - 0.4) / 0.6
                };
                let bell = 1.0 - 0.55 * d * d;
                bell * w * (1.0 - amp * (1.0 - 0.1) * (1.0 - pinna(f)))
            })
        }
        _ => envelope,
    }
}

fn fract(x: f64) -> f64 {
    x - x.floor()
}

/// One claw-shaped lobe or tooth on `f ∈ [0, 1)`: a truncated smoothstep rise
/// `smoothstep(c·s)/smoothstep(c)`, `s = f/xp`, to a pointed peak at `xp`, then a quadratic
/// fall `(1 − r)²` back into a smooth, rounded sinus (both sides reach the sinus with zero
/// slope). Cutting the smoothstep at `c < 1` leaves a convex lobe body that still arrives at
/// the peak with a finite slope, so the tip is a deliberate corner; smaller `c` gives a
/// straighter, more saw-like flank.
fn claw(f: f64, xp: f64, c: f64) -> f64 {
    if f < xp {
        smoothstep(c * f / xp) / smoothstep(c)
    } else {
        let r = (f - xp) / (1.0 - xp);
        (1.0 - r) * (1.0 - r)
    }
}

/// Exponent of the acanthus lobe spacing `φ(u) = u^p`: lobes get shorter towards the tip.
const ACANTHUS_SPACING: f64 = 1.3;
/// Fraction of a lobe (from its basal sinus) where its crest sits: the long, gently rising
/// basal flank is the lobe's outer edge and carries the sub-lobes, the short distal flank
/// drops steeply into the next sinus, so the lobe points towards the leaf tip.
const ACANTHUS_CREST: f64 = 0.68;
/// Fraction of a lobe's period at its distal end taken by the rounded sinus.
const ACANTHUS_GAP: f64 = 0.08;

/// Acanthus width factor at `u` for `n` lobes: big round-topped, forward-pointing lobes
/// whose outer flank is scalloped into rounded sub-lobes, separated by rounded sinuses that
/// never cut below ~30 % of the envelope.
fn acanthus_modulation(n: f64, phase: f64, u: f64) -> f64 {
    let f = fract(n * u.max(0.0).powf(ACANTHUS_SPACING) + phase);
    let y = f / (1.0 - ACANTHUS_GAP);
    let body = lobe_bump(y);
    // Sinuses are deepest near the base; the last lobes are shallow so the tip stays whole.
    let floor = lerp(0.3, 0.62, smoothstep((u - 0.5) / 0.45));
    let raw = floor + (1.0 - floor) * body * (1.0 - 0.13 * sub_notches(sub_lobes(u), y));
    // Round the bottom of the sinus (a smooth maximum with a slightly raised floor).
    let k = 0.06;
    let b = floor + 0.02;
    let m = 0.5 * (raw + b + ((raw - b) * (raw - b) + k * k).sqrt());
    lerp(1.0, m.min(1.0), smoothstep(u / 0.12))
}

/// Sub-lobes per acanthus lobe: three on the big lower lobes, two towards the tip.
fn sub_lobes(u: f64) -> f64 {
    if u.max(0.0).powf(ACANTHUS_SPACING) < 0.55 {
        3.0
    } else {
        2.0
    }
}

/// Rounded notches between `m` sub-lobes spread along the basal flank `y ∈ [0, crest]`.
fn sub_notches(m: f64, y: f64) -> f64 {
    let t = y / ACANTHUS_CREST;
    (1..m as usize)
        .map(|k| {
            let d = (t - k as f64 / m) / 0.09;
            (-d * d).exp()
        })
        .sum()
}

/// A round-topped lobe on `y ∈ [0, 1]` (zero outside): a convex rise to the crest at
/// [`ACANTHUS_CREST`], then a quarter ellipse down to the sinus. Both join with zero slope.
fn lobe_bump(y: f64) -> f64 {
    if !(0.0..=1.0).contains(&y) {
        0.0
    } else if y < ACANTHUS_CREST {
        let t = 1.0 - y / ACANTHUS_CREST;
        (1.0 - t * t).powf(0.75)
    } else {
        let t = (y - ACANTHUS_CREST) / (1.0 - ACANTHUS_CREST);
        (1.0 - t * t).max(0.0).sqrt()
    }
}

/// One lobe of a lobed leaf half, in arc-length fractions of the midrib.
#[derive(Debug, Clone, PartialEq)]
pub struct Lobe {
    /// Basal sinus.
    pub start: f64,
    /// Crest (the lobe's tip, where its vein ends).
    pub crest: f64,
    /// Distal sinus (may exceed 1 for the last lobe).
    pub end: f64,
    /// Crests of the sub-lobes on the outer flank, below the crest.
    pub subs: Vec<f64>,
}

impl LeafParams {
    /// The lobes of one blade half (`side = +1` left, `-1` right) of an acanthus leaf, from
    /// the base to the tip; empty for other shapes. Lobes whose crest falls outside the
    /// blade are left out.
    pub fn lobes(&self, side: f64) -> Vec<Lobe> {
        let p = self.sanitized();
        let LeafShape::Acanthus { lobes } = p.shape else {
            return Vec::new();
        };
        let n = lobes.clamp(3, 7) as f64;
        let phase = if side < 0.0 { 0.5 * p.asymmetry } else { 0.0 };
        // u where lobe `k` reaches y (lobe-local position, see `acanthus_modulation`).
        let at = |k: f64, y: f64| {
            let x = (k + y * (1.0 - ACANTHUS_GAP) - phase).max(0.0) / n;
            x.powf(1.0 / ACANTHUS_SPACING)
        };
        let mut out = Vec::new();
        for k in -1..(n as i32 + 2) {
            let k = k as f64;
            let crest = at(k, ACANTHUS_CREST);
            if !(0.08..=0.97).contains(&crest) {
                continue;
            }
            let m = sub_lobes(crest);
            let subs = (0..m as usize - 1)
                .map(|j| at(k, ACANTHUS_CREST * (j as f64 + 0.55) / m))
                .collect();
            out.push(Lobe {
                start: at(k, 0.0),
                crest,
                end: at(k + 1.0, 0.0),
                subs,
            });
        }
        out
    }
}

/// One rounded pinna on `f ∈ [0, 1)`: a near-elliptic `sin(πf)^0.6` bump with V-shaped
/// sinuses between neighbours (the ribbon's lean tilts it towards the tip).
fn pinna(f: f64) -> f64 {
    let f = f.clamp(0.0, 1.0);
    ease(2.0 * f.min(1.0 - f), 0.6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::rng::SeedTree;
    use kurbo::{PathEl, Shape};

    fn all_shapes() -> Vec<LeafShape> {
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

    fn leaf(shape: LeafShape) -> Leaf<CubicBez> {
        let p = LeafParams::new(shape)
            .with_asymmetry(0.3)
            .with_tip_curl(0.35);
        Leaf::from_base(Point::new(10.0, 20.0), Vec2::new(1.0, -0.4), 120.0, 50.0, p)
    }

    fn vertices(p: &BezPath) -> Vec<Point> {
        p.elements()
            .iter()
            .filter_map(|el| match el {
                PathEl::MoveTo(q) | PathEl::LineTo(q) | PathEl::CurveTo(_, _, q) => Some(*q),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn every_shape_has_a_valid_outline() {
        for shape in all_shapes() {
            let l = Leaf::from_base(
                Point::ZERO,
                Vec2::new(0.0, -1.0),
                100.0,
                LeafParams::new(shape).aspect * 100.0,
                LeafParams::new(shape)
                    .with_asymmetry(0.25)
                    .with_tip_curl(0.5),
            );
            let o = l.outline(0.1);
            assert!(path_is_finite(&o), "{shape:?}");
            assert_eq!(degenerate_segments(&o, 1e-9), 0, "{shape:?}");
            assert!(matches!(o.elements().last(), Some(PathEl::ClosePath)));
            let rep = l.check(0.1);
            assert_eq!(rep.self_intersections, 0, "{shape:?} {rep:?}");
            assert_eq!(rep.cusp_count, 0, "{shape:?} {rep:?}");
            assert!(o.area().abs() > 100.0, "{shape:?}");
        }
    }

    #[test]
    fn tip_and_base_are_exact_points_on_the_midrib() {
        for shape in all_shapes() {
            let l = leaf(shape);
            let o = l.outline(0.05);
            let vs = vertices(&o);
            let (base, tip) = (l.midrib().point(0.0), l.midrib().point(1.0));
            assert_eq!(vs[0], base, "{shape:?}");
            assert!(vs.contains(&tip), "{shape:?}");
        }
    }

    #[test]
    fn tip_curl_bends_the_tip_towards_the_left() {
        let straight = Leaf::from_base(
            Point::ZERO,
            Vec2::new(1.0, 0.0),
            100.0,
            25.0,
            LeafParams::new(LeafShape::Lanceolate).with_tip_curl(0.0),
        );
        let curled = Leaf::from_base(
            Point::ZERO,
            Vec2::new(1.0, 0.0),
            100.0,
            25.0,
            LeafParams::new(LeafShape::Lanceolate).with_tip_curl(0.6),
        );
        assert!(straight.midrib().point(1.0).y.abs() < 1e-9);
        assert!(curled.midrib().point(1.0).y > 10.0);
    }

    #[test]
    fn acanthus_has_the_requested_lobes() {
        for lobes in [3, 5, 7] {
            let p = LeafParams::new(LeafShape::Acanthus { lobes }).with_tip_curl(0.0);
            let l = Leaf::from_base(Point::ZERO, Vec2::new(1.0, 0.0), 200.0, 120.0, p);
            let w = |u: f64| l.ribbon().widths_at(u).0;
            let crests: Vec<f64> = p.lobes(1.0).iter().map(|b| b.crest).collect();
            assert!(
                crests.len() + 1 >= lobes as usize && crests.len() <= lobes as usize + 1,
                "{lobes}: {crests:?}"
            );
            // Between two lower crests the edge dips into a deep sinus that keeps the blade
            // whole (lobes get shallower towards the tip).
            for c in crests.windows(2).filter(|c| c[1] < 0.7) {
                let sinus = (0..=200)
                    .map(|i| w(lerp(c[0], c[1], i as f64 / 200.0)))
                    .fold(f64::MAX, f64::min);
                let low = w(c[0]).min(w(c[1]));
                assert!(sinus < 0.65 * low && sinus > 0.2 * low, "{lobes} {c:?}");
            }
            // Each crest is (nearly) the widest point of its own lobe.
            for b in p.lobes(1.0).iter().filter(|b| b.crest < 0.7) {
                let wc = w(b.crest);
                assert!(
                    (0..=50).all(|i| w(lerp(b.start, b.end, i as f64 / 50.0)) <= wc / 0.92),
                    "{lobes} {b:?}"
                );
            }
        }
    }

    #[test]
    fn fern_pinnae_are_longest_below_the_middle() {
        let p = LeafParams::new(LeafShape::Fern { pinnae: 16 }).with_tip_curl(0.0);
        let l = Leaf::from_base(Point::ZERO, Vec2::new(1.0, 0.0), 100.0, 30.0, p);
        // Longest pinna per tenth of the frond.
        let reach: Vec<f64> = (0..10)
            .map(|k| {
                (0..40)
                    .map(|i| l.ribbon().widths_at((k as f64 + i as f64 / 40.0) / 10.0).0)
                    .fold(0.0, f64::max)
            })
            .collect();
        let best = (0..10)
            .max_by(|&a, &b| reach[a].total_cmp(&reach[b]))
            .unwrap();
        assert!((2..=4).contains(&best), "{reach:?}");
        assert!(reach[0] < 0.7 * reach[best] && reach[8] < 0.6 * reach[best]);
    }

    #[test]
    fn cordate_base_lobes_reach_behind_the_petiole() {
        let l = Leaf::from_base(
            Point::ZERO,
            Vec2::new(1.0, 0.0),
            100.0,
            60.0,
            LeafParams::new(LeafShape::Cordate).with_tip_curl(0.0),
        );
        let min_x = (0..=200)
            .map(|i| l.point(i as f64 / 200.0, 1.0).x)
            .fold(f64::MAX, f64::min);
        assert!(min_x < -2.0, "{min_x}");
        let o = l.outline(0.1);
        assert_eq!(l.check(0.1).self_intersections, 0);
        assert!(path_is_finite(&o));
    }

    #[test]
    fn asymmetry_widens_the_left_half() {
        let l = leaf(LeafShape::Ovate);
        let (left, right) = (0..=50)
            .map(|i| l.ribbon().widths_at(i as f64 / 50.0))
            .fold((0.0f64, 0.0f64), |(a, b), (x, y)| (a.max(x), b.max(y)));
        assert!(left > right * 1.1);
    }

    #[test]
    fn stripes_stay_inside_the_leaf() {
        for shape in [LeafShape::Lanceolate, LeafShape::Acanthus { lobes: 4 }] {
            let l = leaf(shape);
            let o = l.outline(0.05);
            for k in 1..8 {
                let v = -1.0 + 2.0 * k as f64 / 8.0;
                for i in 1..60 {
                    let u = 0.02 + 0.96 * i as f64 / 60.0;
                    assert_ne!(o.winding(l.point(u, v)), 0, "{shape:?} u={u} v={v}");
                }
            }
            assert_eq!(l.ribbon().stripes(3, 0.1).len(), 3);
        }
    }

    #[test]
    fn deterministic() {
        let make = || {
            let mut rng = SeedTree::new(9).child("leaf").rng();
            let p = LeafParams::new(LeafShape::Serrated { teeth: 20 }).jittered(&mut rng, 1.0);
            Leaf::from_base(Point::ZERO, Vec2::new(0.0, 1.0), 80.0, 30.0, p).outline(0.1)
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn leaning_fern_with_strong_curl_has_no_cusp() {
        // Found by jittering: the lean of the pinnae made the inner edge stop and turn back
        // where |κ|·w was still below the plain normal-offset limit.
        let p = LeafParams {
            shape: LeafShape::Fern { pinnae: 20 },
            aspect: 0.3768407299307604,
            asymmetry: -0.4313823555313748,
            tip_curl: -1.1742657303324564,
            base_point: 0.2122489070655138,
            tip_point: 0.5751852610857096,
        };
        let l = Leaf::from_base(Point::ZERO, Vec2::new(1.0, 0.3), 100.0, p.aspect * 100.0, p);
        let rep = l.check(0.1);
        assert!(rep.is_valid(), "{rep:?}");
        assert!(rep.max_width_curvature_product < 1.0, "{rep:?}");
    }

    /// Distance from `q` to the nearest point of `path`.
    fn distance(path: &BezPath, q: Point) -> f64 {
        use kurbo::ParamCurveNearest;
        path.segments()
            .filter(|s| s.bounding_box().inflate(1.0, 1.0).contains(q))
            .map(|s| s.nearest(q, 1e-9).distance_sq.sqrt())
            .fold(f64::MAX, f64::min)
    }

    #[test]
    fn outline_reaches_sinuses_and_tooth_tips() {
        // The deepest point of each sinus and each tooth tip are corners of the edge; the
        // fitted outline must pass within the accuracy of them, not round them off.
        let acc = 0.02;
        for (shape, at) in [
            (LeafShape::Fern { pinnae: 40 }, 0.0),
            (LeafShape::Serrated { teeth: 80 }, 0.8),
        ] {
            let n = match shape {
                LeafShape::Fern { pinnae } => pinnae,
                LeafShape::Serrated { teeth } => teeth,
                _ => unreachable!(),
            } as f64;
            let p = LeafParams::new(shape)
                .with_asymmetry(0.4)
                .with_tip_curl(0.8);
            let l = Leaf::from_base(Point::ZERO, Vec2::new(1.0, 0.0), 200.0, 80.0, p);
            let o = l.outline(acc);
            let mut worst: f64 = 0.0;
            for (v, phase) in [(1.0, 0.0), (-1.0, 0.5 * 0.4)] {
                for k in 1..n as usize {
                    let u = (k as f64 + at - phase) / n;
                    worst = worst.max(distance(&o, l.point(u, v)));
                }
            }
            assert!(worst <= 1.5 * acc, "{shape:?}: {worst}");
        }
    }

    #[test]
    fn strongly_curled_leaf_is_fitted() {
        // The curled tip overlaps the blade. The fit used to compare against a ribbon far
        // thinner than its accuracy, whose fitting noise crossed too, and gave up.
        let l = Leaf::from_base(
            Point::ZERO,
            Vec2::new(1.0, 0.0),
            100.0,
            60.0,
            LeafParams::new(LeafShape::Ovate).with_tip_curl(3.0),
        );
        assert!(l.ribbon().width_scale() < 1.0);
        for acc in [0.25, 0.1, 0.02] {
            assert_eq!(l.check(acc).self_intersections, 0, "acc {acc}");
        }
    }

    #[test]
    fn fern_outline_has_no_short_segments() {
        let mut rng = SeedTree::new(0).rng();
        let p = LeafParams::new(LeafShape::Fern { pinnae: 25 }).jittered(&mut rng, 1.5);
        let l = Leaf::from_base(Point::new(3.0, 4.0), Vec2::new(0.3, 1.0), 120.0, 60.0, p);
        for acc in [0.1, 0.02] {
            for seg in l.outline(acc).segments() {
                let c = seg.to_cubic();
                let len = (c.p1 - c.p0).hypot() + (c.p2 - c.p1).hypot() + (c.p3 - c.p2).hypot();
                assert!(len >= 0.5 * acc - 1e-9, "acc {acc}: {c:?}");
            }
        }
    }

    #[test]
    fn degenerate_leaves_do_not_panic() {
        let z = Leaf::from_base(
            Point::new(1.0, 1.0),
            Vec2::ZERO,
            0.0,
            10.0,
            LeafParams::new(LeafShape::Ovate),
        );
        assert!(z.outline(0.1).elements().is_empty());
        assert!(z.point(0.5, 0.5).is_finite());
        let wild = LeafParams {
            shape: LeafShape::Acanthus { lobes: 99 },
            aspect: f64::INFINITY,
            asymmetry: f64::NAN,
            tip_curl: 1e9,
            base_point: -4.0,
            tip_point: 7.0,
        };
        let l = Leaf::from_base(Point::ZERO, Vec2::new(f64::NAN, 1.0), 50.0, f64::NAN, wild);
        assert!(path_is_finite(&l.outline(0.1)));
        let thin = Leaf::new(
            CubicBez::new((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1e-12, 0.0)),
            LeafParams::new(LeafShape::Willow),
        );
        assert!(path_is_finite(&thin.outline(0.1)));
    }
}
