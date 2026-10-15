//! Procedural flowers: engraved petals with vein systems, layered into blossoms.
//!
//! # Petals
//! A [`Petal`] is a [`Surface`] in petal-local coordinates: base at the origin, tip at
//! `(0, length)`. Its two edges are piecewise cubic Béziers chosen per [`PetalProfile`]
//! (never ellipses). `P(u, v)` interpolates linearly between the edges in a *flat* frame
//! and is then pushed through a progressive in-plane bend (the curl) and a foreshortening
//! squash (the tilt), so iso-curves, veins and hatching all follow the deformed shape.
//!
//! # Detail
//! Veins, hatching and base shading are authored in the `(u, v)` parameter space of the
//! surface and mapped through it, so they radiate from the base and bend with the petal.
//! Lines are drawn as filled tapered strokes (swelling engraving lines) and clipped to the
//! petal outline.
//!
//! # Flowers
//! A [`FlowerParams`] is a stack of [`PetalLayer`]s (outermost first) around a
//! [`CenterStyle`]. [`flower`] assembles them with per-petal jitter drawn from a
//! [`SeedTree`], and [`variants`] offers named, seed-randomised species.

use crate::geom::fit::fit_points;
use crate::geom::validate::sanitize;
use crate::geom::{Composite, Curve, CurveExt, FnCurve, Surface, lerp, smoothstep, unlerp};
use crate::ornament::Detail;
use crate::palette::Palette;
use crate::rng::{ChaCha8Rng, Draw, SeedTree};
use crate::scene::{Color, Node, Paint, Style};
use kurbo::{Affine, BezPath, Circle, CubicBez, ParamCurve, PathEl, Point, QuadBez, Shape, Vec2};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Segments shorter than this are dropped from every produced path.
const MIN_SEG: f64 = 1e-7;
/// Fitting tolerance of detail lines (veins, hatching, base shading) as a fraction of the
/// petal size: far below their stroke widths, so the fit is invisible.
const DETAIL_TOLERANCE: f64 = 0.002;
/// Level of detail by the petal size on the canvas (pixels, see [`Petal::size`]): smaller
/// petals have no hatching, then no vein forks, then neither lateral veins nor base
/// shading (fill, outline and midrib stay).
const LOD_HATCH: f64 = 24.0;
const LOD_FORKS: f64 = 16.0;
const LOD_VEINS: f64 = 10.0;
/// Discs smaller than this on the canvas (radius, pixels) are drawn without florets.
const LOD_FLORETS: f64 = 6.0;
/// Exponent of the curl profile `θ(s) = curl · (s/L)^BEND_EXP`: > 1 keeps the base straight
/// and concentrates the bend towards the tip, like a real petal.
const BEND_EXP: f64 = 1.6;
/// The bend table covers `s ∈ [0, BEND_SPAN · L]` (edges may overshoot the nominal tip).
const BEND_SPAN: f64 = 1.3;
/// Number of integration intervals of the bend table.
const BEND_STEPS: usize = 64;
/// Maximum relative half-width of any profile (asymmetry and flares included), used to
/// bound the curl so the inner side of a bent petal never folds over itself.
const MAX_HALF_WIDTH: f64 = 0.7;

// ---------------------------------------------------------------------------------------
// Petal parameters
// ---------------------------------------------------------------------------------------

/// Silhouette family of a petal. Each is a pair of hand-tuned cubic edge curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PetalProfile {
    /// Broad obovate petal, widest past the middle, round top (rose, peony).
    Rounded,
    /// Lanceolate tepal, widest low, long acuminate tip with a concave recurve (lily, lotus).
    Pointed,
    /// Narrow claw at the base and a wide rounded blade (poppy, cosmos).
    Spoon,
    /// Heart-shaped tip with a central notch (primrose, phlox).
    Notched,
    /// Fan-shaped petal whose upper edge is modulated by a smooth sinusoid (carnation).
    Fringed,
    /// Long thin ray with parallel sides and a slightly pinched tip (daisy, aster).
    NarrowRay,
    /// Tubular petal or sepal with a flared, pointed lip (bellflower).
    Bell,
}

impl PetalProfile {
    /// All profiles, in a fixed order.
    pub const ALL: [PetalProfile; 7] = [
        PetalProfile::Rounded,
        PetalProfile::Pointed,
        PetalProfile::Spoon,
        PetalProfile::Notched,
        PetalProfile::Fringed,
        PetalProfile::NarrowRay,
        PetalProfile::Bell,
    ];
}

/// Shape parameters of a single petal (petal-local units).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PetalParams {
    /// Silhouette family.
    pub profile: PetalProfile,
    /// Base-to-tip length.
    pub length: f64,
    /// Maximum width, edge to edge.
    pub width: f64,
    /// Attachment width at the base as a fraction of `width`, in `[0, 0.9]`.
    pub base_width: f64,
    /// Tip sharpness in `[0, 1]`: 0 = round / deep notch, 1 = pointed / shallow notch.
    pub tip: f64,
    /// Total in-plane bend at the tip in radians (positive bends the tip towards `-x`).
    /// Clamped so the inner side of the petal never folds.
    pub curl: f64,
    /// Left/right asymmetry in `[-1, 1]`: one side wider, the tip leaning towards it.
    pub asymmetry: f64,
}

impl PetalParams {
    /// Parameters with neutral defaults for the optional controls.
    pub fn new(profile: PetalProfile, length: f64, width: f64) -> PetalParams {
        PetalParams {
            profile,
            length,
            width,
            base_width: 0.15,
            tip: 0.3,
            curl: 0.0,
            asymmetry: 0.0,
        }
    }

    /// Copy with every field finite and inside its valid range (see field docs).
    pub fn sanitized(&self) -> PetalParams {
        let length = finite_or(self.length, 1.0).max(1e-3);
        let width = finite_or(self.width, 0.4 * length).clamp(0.01 * length, 4.0 * length);
        // κ(s) = dθ/ds peaks at the tip: κ_max = |curl|·BEND_EXP / L. Keeping
        // κ_max · half_width < 0.75 guarantees the offset curves of the bend never cusp.
        let max_curl = (0.75 * length / (BEND_EXP * MAX_HALF_WIDTH * width)).min(2.0);
        PetalParams {
            profile: self.profile,
            length,
            width,
            base_width: finite_or(self.base_width, 0.15).clamp(0.0, 0.9),
            tip: finite_or(self.tip, 0.3).clamp(0.0, 1.0),
            curl: finite_or(self.curl, 0.0).clamp(-max_curl, max_curl),
            asymmetry: finite_or(self.asymmetry, 0.0).clamp(-1.0, 1.0),
        }
    }
}

fn finite_or(x: f64, default: f64) -> f64 {
    if x.is_finite() { x } else { default }
}

// ---------------------------------------------------------------------------------------
// Petal surface
// ---------------------------------------------------------------------------------------

/// A petal surface in petal-local coordinates: base at the origin, tip near `(0, length)`.
///
/// `point(u, v)`: `u ∈ [0,1]` runs base → tip, `v ∈ [-1,1]` across (`v = -1` left edge,
/// `v = +1` right edge, `v = 0` the midrib).
#[derive(Debug, Clone)]
pub struct Petal {
    params: PetalParams,
    right: Edge,
    left: Edge,
    bend: Bend,
}

impl Petal {
    /// Build a petal from (sanitised) parameters, facing the viewer (no tilt).
    pub fn new(params: &PetalParams) -> Petal {
        let p = params.sanitized();
        let l = p.length;
        let w = 0.5 * p.width;
        let a = p.asymmetry;
        // Asymmetry: one edge wider than the other, its widest point a little higher, and the
        // tip leaning towards the wider side so the midrib is a gentle curve, not a mirror axis.
        let (wr, wl) = (w * (1.0 + 0.2 * a), w * (1.0 - 0.2 * a));
        let tx = 0.08 * a * w;
        let dy = 0.05 * a;
        let b = p.base_width * w;
        let wave = |half: f64| {
            (p.profile == PetalProfile::Fringed).then(|| {
                let cycles = ((2.0 * half / l) * 3.0).clamp(0.5, 4.0) + 1.5;
                Wave {
                    amp: 0.06 * (2.0 * half).min(l),
                    cycles,
                    start: 0.5,
                    phase: 7.3 * p.asymmetry + 3.1 * p.curl,
                }
            })
        };
        let right = Edge {
            curve: edge_profile(p.profile, l, wr, b.min(0.9 * wr), p.tip, tx, dy),
            wave: wave(wr),
            mirror: false,
        };
        let left = Edge {
            curve: edge_profile(p.profile, l, wl, b.min(0.9 * wl), p.tip, -tx, -dy),
            wave: wave(wl),
            mirror: true,
        };
        Petal {
            params: p,
            right,
            left,
            // λ = 1 for petals at most ~0.7 as wide as long, down to 0.4 for very wide ones.
            bend: Bend::new(p.curl, l, (0.35 * l / w).clamp(0.4, 1.0)),
        }
    }

    /// Foreshorten the petal as if tilted by `tilt` radians about its base line
    /// (0 = facing the viewer). The squash factor `|cos tilt|` is clamped to `[0.2, 1]`.
    pub fn with_tilt(mut self, tilt: f64) -> Petal {
        self.bend.squash = finite_or(tilt, 0.0).cos().abs().clamp(0.2, 1.0);
        self
    }

    /// The sanitised parameters this petal was built from.
    pub fn params(&self) -> &PetalParams {
        &self.params
    }

    /// Characteristic size `sqrt(length · max(width, length/4))` used to scale stroke widths.
    pub fn size(&self) -> f64 {
        (self.params.length * self.params.width.max(0.25 * self.params.length)).sqrt()
    }

    /// Point on the undeformed (flat, untilted) petal.
    ///
    /// The edge ripple (fringed petals) is not interpolated across the petal: it fades in
    /// only for `|v| > 0.6`, so veins and hatching stay calm while the margin is wavy.
    fn flat(&self, u: f64, v: f64) -> Point {
        let a = self.left.base(u);
        let b = self.right.base(u);
        let mut p = a.lerp(b, (v + 1.0) * 0.5);
        if self.right.wave.is_some() {
            // Near a narrow tip the ripple would exceed the gap between the edges and make
            // them cross, so each side may move at most 0.4 of the local gap.
            let room = 0.4 * (b - a).hypot();
            let fit = |r: Vec2, amp: f64| r * (room / amp).min(1.0);
            let (ar, al) = (self.right.amp(), self.left.amp());
            p += fit(self.right.ripple(u), ar) * smoothstep(unlerp(0.6, 1.0, v));
            p += fit(self.left.ripple(u), al) * smoothstep(unlerp(-0.6, -1.0, v));
        }
        p
    }

    /// Closed region bounded by the base and the arched line `u = ut · (1 - 0.3 v²)`.
    fn base_region(&self, ut: f64, accuracy: f64) -> BezPath {
        let top = |v: f64| ut * (1.0 - 0.3 * v * v);
        let side_l = sample_curve(&FnCurve(|t: f64| self.point(top(-1.0) * t, -1.0)), accuracy);
        let arch = sample_curve(
            &FnCurve(|t: f64| {
                let v = lerp(-1.0, 1.0, t);
                self.point(top(v), v)
            }),
            accuracy,
        );
        let side_r = sample_curve(
            &FnCurve(|t: f64| self.point(top(1.0) * (1.0 - t), 1.0)),
            accuracy,
        );
        let fit = |pts: &[Point]| spline(pts, accuracy);
        let (side_l, arch, side_r) = (fit(&side_l), fit(&arch), fit(&side_r));
        closed_concat(&[side_l, arch, side_r])
    }
}

impl Surface for Petal {
    fn point(&self, u: f64, v: f64) -> Point {
        self.bend.map(self.flat(u, v))
    }

    /// Right edge base → tip, then left edge tip → base. Fitting the edges separately keeps
    /// a pointed tip sharp; the base is closed with a straight line when it has a width.
    fn outline(&self, accuracy: f64) -> BezPath {
        // Relative floor: near the limits of f64 precision (huge petals) the adaptive fit
        // never converges and subdivides without end.
        let acc = sane_accuracy(accuracy).max(1e-7 * (self.params.length + self.params.width));
        let right = FnCurve(|u: f64| self.point(u, 1.0));
        let left = FnCurve(|u: f64| self.point(1.0 - u, -1.0));
        let sample = |c: &dyn Curve, n: usize| -> Vec<Point> {
            (0..n).map(|i| c.point(i as f64 / (n - 1) as f64)).collect()
        };
        if let Some(w) = self.right.wave {
            // Adaptive fitting of a rippled edge is slow (every crest is a split candidate);
            // a spline through ~20 samples per ripple period is just as smooth.
            let n = (w.cycles / (1.0 - w.start) * 20.0).clamp(32.0, 400.0) as usize;
            return closed_concat(&[
                spline(&sample(&right, n), acc),
                spline(&sample(&left, n), acc),
            ]);
        }
        // The fit occasionally returns a cubic that starts by doubling back (a cusp within
        // the tolerance) on bent edges; such an edge is splined from samples instead.
        let edge = |c: &dyn Curve| {
            let fitted = c.to_bezpath(acc);
            if !reverses(&fitted) {
                return fitted;
            }
            let len = self.params.length + self.params.width;
            let n = ((len / acc).sqrt() * 4.0).clamp(32.0, 400.0) as usize;
            spline(&sample(c, n), acc)
        };
        closed_concat(&[edge(&right), edge(&left)])
    }
}

/// Smooth, slightly irregular modulation of an edge along its outward normal: two
/// incommensurate sinusoids under a slowly varying amplitude, so the fringe reads as
/// hand-drawn lobes rather than a sawtooth.
#[derive(Debug, Clone, Copy)]
struct Wave {
    amp: f64,
    /// Periods of the main sinusoid between `start` and the tip.
    cycles: f64,
    start: f64,
    /// Per-petal phase so neighbouring petals never share the same crests.
    phase: f64,
}

/// One petal edge in the flat frame: a profile drawn on the `+x` side, optionally waved and
/// mirrored to the `-x` side.
#[derive(Debug, Clone)]
struct Edge {
    curve: Composite,
    wave: Option<Wave>,
    mirror: bool,
}

impl Edge {
    /// Point on the smooth profile (without ripple).
    fn base(&self, t: f64) -> Point {
        let t = finite_or(t, 0.0).clamp(0.0, 1.0);
        let p = self.curve.point(t);
        if self.mirror {
            Point::new(-p.x, p.y)
        } else {
            p
        }
    }

    /// Ripple amplitude (a tiny positive value without a wave, so it can divide).
    fn amp(&self) -> f64 {
        self.wave.map_or(1e-12, |w| w.amp.max(1e-12))
    }

    /// Ripple displacement along the outward normal; zero without a wave. A smoothstep
    /// envelope fades the wave in after `start` and out again just before the tip, so both
    /// edges still meet there; the wave itself is `sin(x) + 0.35 sin(2.17x + φ)` under a
    /// slow amplitude swell, which breaks the regularity of a plain sinusoid.
    fn ripple(&self, t: f64) -> Vec2 {
        let Some(w) = self.wave else {
            return Vec2::ZERO;
        };
        let t = finite_or(t, 0.0).clamp(0.0, 1.0);
        let phase = ((t - w.start) / (1.0 - w.start)).max(0.0);
        let env = smoothstep(unlerp(w.start, w.start + 0.25 * (1.0 - w.start), t))
            * smoothstep(unlerp(1.0, 0.94, t));
        let x = TAU * w.cycles * phase + w.phase;
        let shape = x.sin() + 0.35 * (2.17 * x + 0.7 * w.phase).sin();
        let swell = 1.0 - 0.3 * (0.37 * x + 1.9 * w.phase).sin();
        let tg = self.curve.tangent(t);
        // Right-hand normal of an edge running base → tip on the +x side points outward.
        let out = Vec2::new(tg.y, -tg.x) * (w.amp * env * shape * swell / 1.35);
        if self.mirror {
            Vec2::new(-out.x, out.y)
        } else {
            out
        }
    }
}

/// `p3 + (p3 - p2)·k`: first control point of a segment continuing `prev` with G1 continuity.
fn smooth_next(prev: &CubicBez, k: f64) -> Point {
    prev.p3 + (prev.p3 - prev.p2) * k
}

/// The right-hand (`+x`) edge of a profile from the base `(b, 0)` to the tip near `(tx, l)`.
///
/// `w` is the half width, `b` the base half width, `s` the tip sharpness, `dy` a shift of
/// the widest point (fraction of `l`). Joins between segments are G1: either both adjacent
/// control points sit on the vertical through the widest point, or [`smooth_next`] is used.
fn edge_profile(
    profile: PetalProfile,
    l: f64,
    w: f64,
    b: f64,
    s: f64,
    tx: f64,
    dy: f64,
) -> Composite {
    // x in absolute units, y as a fraction of the length.
    let p = |x: f64, y: f64| Point::new(x, y * l);
    let segs = match profile {
        PetalProfile::Rounded => {
            let ym = 0.6 + dy;
            vec![
                CubicBez::new(
                    p(b, 0.0),
                    p(b + 0.3 * (w - b), 0.16),
                    p(w, ym - 0.26),
                    p(w, ym),
                ),
                CubicBez::new(
                    p(w, ym),
                    p(w, ym + 0.55 * (1.0 - ym)),
                    p(tx + lerp(0.62, 0.16, s) * w, 1.0 - 0.1 * s),
                    p(tx, 1.0),
                ),
            ]
        }
        PetalProfile::Pointed => {
            let ym = 0.34 + dy;
            vec![
                CubicBez::new(
                    p(b, 0.0),
                    p(b + 0.35 * (w - b), 0.1),
                    p(w, ym - 0.15),
                    p(w, ym),
                ),
                // The second control point hugs the midline well below the tip: the edge
                // turns concave before the tip, giving the acuminate, slightly recurved point.
                CubicBez::new(
                    p(w, ym),
                    p(w, ym + 0.24),
                    p(tx + lerp(0.24, 0.06, s) * w, 0.8),
                    p(tx, 1.0),
                ),
            ]
        }
        PetalProfile::Spoon => {
            let ym = 0.7 + dy;
            vec![
                // Vertical start tangent keeps the claw narrow before the blade flares out.
                CubicBez::new(
                    p(b, 0.0),
                    p(b + 0.12 * (w - b), 0.34),
                    p(w, ym - 0.26),
                    p(w, ym),
                ),
                CubicBez::new(
                    p(w, ym),
                    p(w, ym + 0.17),
                    p(tx + lerp(0.6, 0.2, s) * w, 1.0 - 0.06 * s),
                    p(tx, 1.0),
                ),
            ]
        }
        PetalProfile::Notched => {
            let ym = 0.55 + dy;
            let notch = 0.05 + 0.09 * (1.0 - s);
            vec![
                CubicBez::new(
                    p(b, 0.0),
                    p(b + 0.3 * (w - b), 0.16),
                    p(w, ym - 0.24),
                    p(w, ym),
                ),
                // Ends below its own highest point, heading down and inwards: mirrored, the two
                // edges meet in a V-shaped notch between two rounded lobes.
                CubicBez::new(
                    p(w, ym),
                    p(w, ym + 0.36),
                    p(tx + 0.24 * w, 1.05),
                    p(tx, 1.0 - notch),
                ),
            ]
        }
        PetalProfile::Fringed => {
            let ym = 0.72 + dy;
            vec![
                CubicBez::new(
                    p(b, 0.0),
                    p(b + 0.1 * (w - b), 0.3),
                    p(w, ym - 0.26),
                    p(w, ym),
                ),
                CubicBez::new(
                    p(w, ym),
                    p(w, ym + 0.14),
                    p(tx + lerp(0.7, 0.25, s) * w, 1.0),
                    p(tx, 1.0),
                ),
            ]
        }
        PetalProfile::NarrowRay => {
            let pinch = 0.03 * (1.0 - s);
            let s1 = CubicBez::new(
                p(b, 0.0),
                p(b + 0.7 * (w - b), 0.06),
                p(w, 0.28 + 0.5 * dy),
                p(w, 0.5 + dy),
            );
            let s2 = CubicBez::new(
                p(w, 0.5 + dy),
                p(w, 0.8),
                p(tx + 0.85 * w, 0.965),
                p(tx + 0.5 * w, 0.99),
            );
            // Pinched tip: the edge rounds over and dips slightly into the midline.
            let s3 = CubicBez::new(
                s2.p3,
                smooth_next(&s2, 0.6),
                p(tx + 0.12 * w, 1.0 - 0.5 * pinch),
                p(tx, 1.0 - pinch),
            );
            vec![s1, s2, s3]
        }
        PetalProfile::Bell => {
            // A wide tube (large base, near-vertical start) swelling into a rounded shoulder
            // high up, then an acuminate lobe tip.
            let ym = 0.68 + dy;
            vec![
                CubicBez::new(
                    p(b, 0.0),
                    p(b + 0.1 * (w - b), 0.3),
                    p(w, ym - 0.22),
                    p(w, ym),
                ),
                CubicBez::new(
                    p(w, ym),
                    p(w, ym + 0.17),
                    p(tx + lerp(0.45, 0.06, s) * w, lerp(1.0, 0.86, s)),
                    p(tx, 1.0),
                ),
            ]
        }
    };
    Composite::new_by_length(segs)
}

/// Progressive in-plane bend plus foreshortening, mapping the flat frame to the final one.
///
/// The flat midline `x = 0` becomes a curve `c(s)` with turning angle
/// `θ(s) = curl · (s/L)^BEND_EXP`; a flat point `(x, y)` maps to `c(y) + x · (cos φ, sin φ)`
/// with `φ = λ θ`. Narrow petals bend like ribbons (`λ = 1`, cross-sections perpendicular to
/// the midline); wide ones rotate their cross-sections less (`λ < 1`), otherwise the outer
/// side would be stretched into a fan around the tip.
///
/// The map's Jacobian determinant is `cos(θ - φ) + λ κ x` (κ = dθ/ds), so it stays positive
/// (no fold) as long as `|x| κ < 0.75` (enforced by the curl clamp) and `cos((1-λ)θ) > 0.75 λ`,
/// which holds for the `λ` range used here. `c` has no closed form,
/// so it is integrated once with Simpson's rule and evaluated by cubic Hermite interpolation
/// using the exact tangent `(-sin θ, cos θ)` at the nodes (C1, error far below the fit
/// accuracy). Finally `y` is multiplied by `squash` (the tilt foreshortening).
#[derive(Debug, Clone)]
struct Bend {
    curl: f64,
    len: f64,
    /// Cross-section rotation factor `λ`.
    lambda: f64,
    ds: f64,
    /// `(c(s_i), θ(s_i))` for `s_i = i · ds`; empty when there is no curl.
    table: Vec<(Point, f64)>,
    squash: f64,
}

fn heading(theta: f64) -> Vec2 {
    Vec2::new(-theta.sin(), theta.cos())
}

impl Bend {
    fn new(curl: f64, len: f64, lambda: f64) -> Bend {
        let mut bend = Bend {
            curl,
            len,
            lambda,
            ds: len * BEND_SPAN / BEND_STEPS as f64,
            table: Vec::new(),
            squash: 1.0,
        };
        if curl.abs() > 1e-9 {
            let mut c = Point::ORIGIN;
            bend.table.push((c, 0.0));
            for i in 0..BEND_STEPS {
                let s0 = i as f64 * bend.ds;
                let (t0, tm, t1) = (
                    bend.theta(s0),
                    bend.theta(s0 + 0.5 * bend.ds),
                    bend.theta(s0 + bend.ds),
                );
                c += (heading(t0) + heading(tm) * 4.0 + heading(t1)) * (bend.ds / 6.0);
                bend.table.push((c, t1));
            }
        }
        bend
    }

    fn theta(&self, s: f64) -> f64 {
        let q = (s / self.len).clamp(0.0, BEND_SPAN);
        self.curl * q.powf(BEND_EXP)
    }

    /// Midline point and turning angle at arc length `s` (straight below the base and
    /// beyond the end of the table).
    fn frame(&self, s: f64) -> (Point, f64) {
        if self.table.len() < 2 || s <= 0.0 {
            return (Point::new(0.0, s), 0.0);
        }
        let n = self.table.len() - 1;
        let x = s / self.ds;
        if x >= n as f64 {
            let (c, th) = self.table[n];
            return (c + heading(th) * (s - n as f64 * self.ds), th);
        }
        let i = (x.floor() as usize).min(n - 1);
        let f = x - i as f64;
        let (c0, t0) = self.table[i];
        let (c1, t1) = self.table[i + 1];
        let (f2, f3) = (f * f, f * f * f);
        let h00 = 2.0 * f3 - 3.0 * f2 + 1.0;
        let h10 = f3 - 2.0 * f2 + f;
        let h01 = -2.0 * f3 + 3.0 * f2;
        let h11 = f3 - f2;
        let c = c0.to_vec2() * h00
            + heading(t0) * (self.ds * h10)
            + c1.to_vec2() * h01
            + heading(t1) * (self.ds * h11);
        (c.to_point(), self.theta(s))
    }

    fn map(&self, p: Point) -> Point {
        let (c, th) = self.frame(p.y);
        let phi = self.lambda * th;
        let q = c + Vec2::new(phi.cos(), phi.sin()) * p.x;
        Point::new(q.x, q.y * self.squash)
    }
}

// ---------------------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------------------

fn sane_accuracy(accuracy: f64) -> f64 {
    if accuracy.is_finite() && accuracy > 0.0 {
        accuracy.clamp(1e-3, 10.0)
    } else {
        0.1
    }
}

/// Concatenate open paths into one closed path. A part starting where the previous one
/// ended is joined without a (zero-length) connecting line.
fn closed_concat(parts: &[BezPath]) -> BezPath {
    let mut out = BezPath::new();
    let mut last: Option<Point> = None;
    for part in parts {
        for el in part.elements() {
            match *el {
                PathEl::MoveTo(p) => match last {
                    None => {
                        out.move_to(p);
                        last = Some(p);
                    }
                    Some(l) if (p - l).hypot() > 1e-6 => {
                        out.line_to(p);
                        last = Some(p);
                    }
                    Some(_) => {}
                },
                PathEl::LineTo(p) => {
                    out.line_to(p);
                    last = Some(p);
                }
                PathEl::QuadTo(a, p) => {
                    out.quad_to(a, p);
                    last = Some(p);
                }
                PathEl::CurveTo(a, b, p) => {
                    out.curve_to(a, b, p);
                    last = Some(p);
                }
                PathEl::ClosePath => {}
            }
        }
    }
    if last.is_some() {
        out.close_path();
    }
    sanitize(&out, MIN_SEG)
}

/// `true` if a path meant to be smooth doubles back at a join: the tangents on both sides
/// of a segment boundary point against each other.
fn reverses(path: &BezPath) -> bool {
    let dir = |from: Point, to: &[Point]| {
        to.iter()
            .map(|&q| q - from)
            .find(|d| d.hypot() > 1e-9 * (1.0 + from.to_vec2().hypot()))
    };
    let mut prev_end: Option<Vec2> = None;
    for seg in path.segments() {
        let c = seg.to_cubic();
        let start = dir(c.p0, &[c.p1, c.p2, c.p3]);
        if let (Some(p), Some(s)) = (prev_end, start)
            && p.dot(s) < 0.0
        {
            return true;
        }
        prev_end = dir(c.p3, &[c.p2, c.p1, c.p0]).map(|d| -d).or(prev_end);
    }
    false
}

/// Smooth path through the samples `pts` of a curve, fitted within `tolerance`.
///
/// Detail lines are thin and smooth, so fitting a few dozen samples is visually exact and
/// much cheaper than adaptive fitting of a numerically differentiated curve; a smooth line
/// needs a handful of cubics instead of one per sample. Only petal outlines use
/// [`CurveExt::to_bezpath`].
fn spline(pts: &[Point], tolerance: f64) -> BezPath {
    fit_points(pts, tolerance)
}

/// Number of samples for a detail line of length `len`: about one per `12 · accuracy`.
fn sample_count(len: f64, accuracy: f64) -> usize {
    (len / (12.0 * accuracy)).ceil().clamp(6.0, 40.0) as usize
}

fn sample_curve(curve: &dyn Curve, accuracy: f64) -> Vec<Point> {
    let coarse: Vec<Point> = (0..=8).map(|i| curve.point(i as f64 / 8.0)).collect();
    let len: f64 = coarse.windows(2).map(|w| (w[1] - w[0]).hypot()).sum();
    if !len.is_finite() || len <= 1e-6 {
        return Vec::new();
    }
    let n = sample_count(len, accuracy);
    (0..n)
        .map(|i| curve.point(i as f64 / (n - 1) as f64))
        .collect()
}

/// A filled engraving line along `curve` whose width at parameter `t` is `width(t)`.
/// Returns an empty path for a (near) zero-length or non-finite curve.
///
/// The centreline is sampled once; normals come from central differences of the samples,
/// and both offset sides are splined, so a line costs `n` surface evaluations.
fn tapered(curve: &dyn Curve, width: impl Fn(f64) -> f64, accuracy: f64) -> BezPath {
    let pts = sample_curve(curve, accuracy);
    let n = pts.len();
    if n < 2 || pts.iter().any(|p| !p.is_finite()) {
        return BezPath::new();
    }
    let mut normal = Vec2::new(1.0, 0.0);
    let mut side_a = Vec::with_capacity(n);
    let mut side_b = Vec::with_capacity(n);
    for (i, &p) in pts.iter().enumerate() {
        let d = pts[(i + 1).min(n - 1)] - pts[i.saturating_sub(1)];
        if d.hypot() > 1e-12 {
            normal = Vec2::new(-d.y, d.x) / d.hypot();
        }
        let half = 0.5 * finite_or(width(i as f64 / (n - 1) as f64), 0.0).max(0.0);
        side_a.push(p + normal * half);
        side_b.push(p - normal * half);
    }
    side_b.reverse();
    closed_concat(&[spline(&side_a, accuracy), spline(&side_b, accuracy)])
}

/// A curve given in `(u, v)` parameter space (as a quadratic), mapped through the surface.
fn uv_curve<'a>(surface: &'a impl Surface, q: QuadBez) -> impl Curve + 'a {
    FnCurve(move |t: f64| {
        let p = q.eval(t.clamp(0.0, 1.0));
        surface.point(p.x, p.y)
    })
}

fn uv(u: f64, v: f64) -> Point {
    Point::new(u, v)
}

fn circle(c: Point, r: f64, accuracy: f64) -> Option<BezPath> {
    (r.is_finite() && r > 1e-6 && c.is_finite())
        .then(|| sanitize(&Circle::new(c, r).to_path(accuracy.min(0.25 * r)), MIN_SEG))
}

// ---------------------------------------------------------------------------------------
// Petal detail
// ---------------------------------------------------------------------------------------

/// Role of a detail path, deciding its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ink {
    /// Translucent darkening near the base.
    Shade,
    /// Midrib and lateral veins.
    Vein,
    /// Fine engraving hatches near the base.
    Hatch,
}

/// How the lateral veins of a profile are organised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VeinPlan {
    /// Veins radiating from the base, fanning out and running parallel to the edges.
    Fan,
    /// Veins bowing out from the base and converging again towards the tip.
    Arcuate,
    /// A few veins parallel to the midrib (long rays).
    Parallel,
}

fn vein_plan(profile: PetalProfile) -> VeinPlan {
    match profile {
        PetalProfile::Pointed | PetalProfile::Bell => VeinPlan::Arcuate,
        PetalProfile::NarrowRay => VeinPlan::Parallel,
        _ => VeinPlan::Fan,
    }
}

/// All detail geometry of a petal in petal-local coordinates (unclipped).
///
/// At a reduced `detail` the lines too fine for the petal's canvas size are left out; the
/// random draws stay the same, so the lines that remain do not move.
fn petal_detail(
    seed: &SeedTree,
    petal: &Petal,
    accuracy: f64,
    detail: Detail,
) -> Vec<(BezPath, Ink)> {
    let mut rng = seed.child("veins").rng();
    let p = petal.params();
    let size = petal.size();
    let acc = sane_accuracy(accuracy).max(DETAIL_TOLERANCE * size);
    let px = detail.canvas_size(size);
    let (hatches, forks, veins) = (px >= LOD_HATCH, px >= LOD_FORKS, px >= LOD_VEINS);
    let aspect = p.width / p.length;
    let (w_mid, w_lat, w_fork, w_hatch) = (0.02 * size, 0.011 * size, 0.007 * size, 0.006 * size);
    let mut out = Vec::new();
    let plan = vein_plan(p.profile);

    // Base shading: stacked translucent arches emulate a gradient towards the base.
    if plan != VeinPlan::Parallel && veins {
        for ut in [0.14, 0.24, 0.36] {
            out.push((petal.base_region(ut, acc), Ink::Shade));
        }
    }

    // Engraved hatching near the base: shortened iso-v lines that taper towards the tip,
    // longest near the midrib.
    let m = ((aspect * 14.0).round() as usize).clamp(4, 12);
    for i in 0..=m {
        let v = lerp(-0.85, 0.85, i as f64 / m as f64) + rng.jitter(0.0, 0.03);
        if v.abs() < 0.1 {
            continue;
        }
        let reach = if plan == VeinPlan::Parallel { 0.6 } else { 1.0 };
        let ue = reach * lerp(0.32, 0.12, v.abs()) * rng.uniform(0.85, 1.15);
        if !hatches {
            continue;
        }
        let q = QuadBez::new(uv(0.0, v), uv(0.5 * ue, v), uv(ue, v));
        let path = tapered(&uv_curve(petal, q), |t| w_hatch * (1.0 - t), acc);
        out.push((path, Ink::Hatch));
    }

    // Midrib from the base to ~0.9 of the length, swelling at the base and vanishing at the end.
    let mid_end = if plan == VeinPlan::Arcuate {
        0.92
    } else {
        0.86
    };
    let q = QuadBez::new(
        uv(0.0, 0.0),
        uv(0.5 * mid_end, rng.jitter(0.0, 0.04)),
        uv(mid_end, 0.0),
    );
    let mid = tapered(&uv_curve(petal, q), |t| w_mid * (1.0 - t).powf(0.8), acc);
    out.push((mid, Ink::Vein));

    let vein = |q: QuadBez, w: f64| tapered(&uv_curve(petal, q), |t| w * (1.0 - 0.92 * t), acc);
    match plan {
        VeinPlan::Fan => {
            let k = ((aspect * 6.0).round() as usize).clamp(2, 6);
            for side in [-1.0, 1.0] {
                for j in 1..=k {
                    let f = j as f64 / (k as f64 + 0.6);
                    let (su, sv) = (0.02 + 0.05 * f, side * 0.06 * f);
                    let eu = (lerp(0.94, 0.62, f.powf(1.3)) + rng.jitter(0.0, 0.03)).min(0.97);
                    let ev = side * (0.2 + 0.72 * f + rng.jitter(0.0, 0.03)).min(0.95);
                    // The control point sits almost at the final v: in parameter space the vein
                    // swings out early, which on the narrow base reads as a fan from the base
                    // and then runs parallel to the edge.
                    let cu = lerp(su, eu, 0.4 + rng.jitter(0.0, 0.05));
                    let q = QuadBez::new(uv(su, sv), uv(cu, 0.85 * ev), uv(eu, ev));
                    if veins {
                        out.push((vein(q, w_lat), Ink::Vein));
                    }
                    if j >= 2 && rng.chance(0.7) {
                        // A fork between this vein and the previous one, bending to the tip.
                        let s = q.eval(rng.uniform(0.45, 0.6));
                        let e = uv((s.x + 0.25).min(0.95), ev * rng.uniform(0.55, 0.72));
                        let c = uv(lerp(s.x, e.x, 0.5), lerp(s.y, e.y, 0.25));
                        if forks {
                            out.push((vein(QuadBez::new(s, c, e), w_fork), Ink::Vein));
                        }
                    }
                }
            }
        }
        VeinPlan::Arcuate => {
            let k = ((aspect * 8.0).round() as usize).clamp(2, 5);
            for side in [-1.0, 1.0] {
                for j in 1..=k {
                    let f = j as f64 / (k as f64 + 0.5);
                    // A quadratic peaks at (start + 2·ctrl + end)/4, so the control point sits
                    // beyond the intended bulge; the end is pulled back towards the midrib.
                    let peak = side * (0.25 + 0.6 * f + rng.jitter(0.0, 0.03));
                    let s = uv(0.03, side * 0.04 * f);
                    let e = uv(lerp(0.95, 0.8, f) + rng.jitter(0.0, 0.02), 0.3 * peak);
                    let c = uv(lerp(s.x, e.x, 0.42), (1.6 * peak).clamp(-1.1, 1.1));
                    if veins {
                        out.push((vein(QuadBez::new(s, c, e), w_lat), Ink::Vein));
                    }
                }
                // A faint groove alongside the midrib.
                let v = side * 0.12;
                let q = QuadBez::new(uv(0.04, v), uv(0.35, v * 1.3), uv(0.7, v));
                if forks {
                    out.push((vein(q, w_fork), Ink::Vein));
                }
            }
        }
        VeinPlan::Parallel => {
            for side in [-1.0, 1.0] {
                let v = side * rng.uniform(0.42, 0.55);
                let q = QuadBez::new(uv(0.05, 0.5 * v), uv(0.4, v), uv(0.82, 0.8 * v));
                if forks {
                    out.push((vein(q, w_fork), Ink::Vein));
                }
            }
        }
    }
    out.retain(|(path, _)| !path.elements().is_empty());
    out
}

/// Build one petal: returns its closed outline and a node (both in petal-local coordinates)
/// with the fill, base shading, veins and hatching clipped to the outline, and the ink
/// outline on top. `fill` is the petal colour; veins, hatches and outline come from `style`.
/// Stroke widths are proportional to [`Petal::size`].
pub fn petal(
    seed: &SeedTree,
    petal: &Petal,
    fill: Color,
    style: &FlowerStyle,
    accuracy: f64,
) -> (BezPath, Node) {
    petal_with_detail(seed, petal, fill, style, accuracy, Detail::FULL)
}

/// [`petal`] at a level of detail (see [`petal_detail`]).
fn petal_with_detail(
    seed: &SeedTree,
    petal: &Petal,
    fill: Color,
    style: &FlowerStyle,
    accuracy: f64,
    detail: Detail,
) -> (BezPath, Node) {
    let acc = sane_accuracy(accuracy);
    let outline = petal.outline(acc);
    let size = petal.size();
    let shade = fill.mix(style.ink, 0.25);
    let vein = fill.mix(style.vein, 0.75);
    let hatch = fill.mix(style.ink, 0.5);
    let detail: Vec<Node> = petal_detail(seed, petal, acc, detail)
        .into_iter()
        .map(|(path, ink)| {
            let s = match ink {
                Ink::Shade => Style::fill(shade).with_opacity(0.22),
                Ink::Vein => Style::fill(vein),
                Ink::Hatch => Style::fill(hatch).with_opacity(0.85),
            };
            Node::path(path, s)
        })
        .collect();
    let ink_w = if petal.params().profile == PetalProfile::Fringed {
        0.01
    } else {
        0.014
    };
    // Ink only the two edges: the base is an attachment, not a contour, and a straight base
    // line would show past small flower centres.
    let edges: BezPath = outline
        .elements()
        .iter()
        .filter(|el| !matches!(el, PathEl::ClosePath))
        .copied()
        .collect();
    let node = Node::group(vec![
        Node::path(
            outline.clone(),
            Style::fill(fill).with_fill_paint(Paint::tonal(fill, 90.0)),
        ),
        Node::group(detail).clipped(outline.clone()),
        // A rippled margin already reads as busy; a lighter line keeps it delicate.
        Node::path(edges, Style::stroke(style.ink, ink_w * size)),
    ]);
    (outline, node)
}

// ---------------------------------------------------------------------------------------
// Flower assembly
// ---------------------------------------------------------------------------------------

/// Colours of a flower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowerStyle {
    /// Main (darkest) petal colour, used for the outermost layer.
    pub petal: Color,
    /// Light petal colour, approached by inner layers.
    pub petal_light: Color,
    /// Outline and engraving ink.
    pub ink: Color,
    /// Vein tint.
    pub vein: Color,
    /// Centre (disc, pistil, anthers) colour.
    pub center: Color,
}

impl FlowerStyle {
    /// Derive a style from a palette: petals from the warm (`warm = true`) or cool ramp at
    /// position `t ∈ [0, 1]` (dark → light), the centre from the opposite ramp.
    pub fn from_palette(palette: &Palette, warm: bool, t: f64) -> FlowerStyle {
        let t = finite_or(t, 0.5).clamp(0.0, 1.0);
        let main = |x: f64| {
            if warm {
                palette.warm_at(x)
            } else {
                palette.cool_at(x)
            }
        };
        let other = |x: f64| {
            if warm {
                palette.cool_at(x)
            } else {
                palette.warm_at(x)
            }
        };
        FlowerStyle {
            petal: main(lerp(0.15, 0.6, t)),
            petal_light: main(lerp(0.6, 1.0, t)).mix(palette.light, 0.25),
            ink: palette.ink,
            vein: main(lerp(0.0, 0.3, t)).mix(palette.ink, 0.3),
            center: other(lerp(0.1, 0.35, t)),
        }
    }
}

/// One ring of petals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PetalLayer {
    /// Number of petals (0 draws nothing).
    pub count: usize,
    /// Petal shape before per-petal jitter and `scale`.
    pub petal: PetalParams,
    /// Distance of the petal bases from the flower centre.
    pub radius: f64,
    /// Angle of the first petal in radians.
    pub rotation: f64,
    /// Size multiplier applied to the petal length and width.
    pub scale: f64,
    /// Foreshortening tilt in radians (0 = petals face the viewer; inner layers of cupped
    /// flowers use larger tilts).
    pub tilt: f64,
}

impl PetalLayer {
    /// A layer of `count` petals at the centre, unrotated, unscaled and untilted.
    pub fn new(count: usize, petal: PetalParams) -> PetalLayer {
        PetalLayer {
            count,
            petal,
            radius: 0.0,
            rotation: 0.0,
            scale: 1.0,
            tilt: 0.0,
        }
    }
}

/// Centre of a flower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CenterStyle {
    /// No centre (e.g. a rose, whose inner petals close it).
    None,
    /// A disc stippled with `seeds` florets in a golden-angle (phyllotaxis) spiral.
    Disc {
        /// Disc radius.
        radius: f64,
        /// Number of florets.
        seeds: usize,
    },
    /// A small pistil disc of `radius` with `count` curved stamens of `length` around it.
    Stamens {
        /// Number of stamens.
        count: usize,
        /// Stamen length beyond the pistil.
        length: f64,
        /// Pistil radius.
        radius: f64,
    },
}

/// A complete flower description.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowerParams {
    /// Petal layers, outermost (drawn first) to innermost.
    pub layers: Vec<PetalLayer>,
    /// Centre ornament, drawn last.
    pub center: CenterStyle,
    /// Irregularity in `[0, 1]`: per-petal jitter of angle, size, width, curl, asymmetry,
    /// tilt and colour.
    pub jitter: f64,
}

impl FlowerParams {
    /// Approximate outer radius (largest `radius + length · scale` over the layers).
    pub fn outer_radius(&self) -> f64 {
        let petals = self
            .layers
            .iter()
            .map(|l| finite_or(l.radius, 0.0) + finite_or(l.petal.length * l.scale, 0.0))
            .fold(0.0, f64::max);
        let center = match self.center {
            CenterStyle::None => 0.0,
            CenterStyle::Disc { radius, .. } => finite_or(radius, 0.0),
            CenterStyle::Stamens { length, radius, .. } => {
                finite_or(radius, 0.0) + finite_or(length, 0.0)
            }
        };
        petals.max(center)
    }
}

/// Build a flower centred at the origin (the caller positions it with a transform).
///
/// Layers are drawn outermost first, over a dark receptacle disc and under the centre.
/// Within a layer petal `i` is drawn over petal `i - 1`, and the first petal is redrawn
/// clipped to its overlap with the last one so the ring closes consistently.
pub fn flower(seed: &SeedTree, params: &FlowerParams, style: &FlowerStyle, accuracy: f64) -> Node {
    flower_with_detail(seed, params, style, accuracy, Detail::FULL)
}

/// [`flower`] at a level of detail: petals that are small on the canvas leave out their
/// hatching, then their vein forks, then their lateral veins and base shading, and a small
/// disc its florets.
pub fn flower_with_detail(
    seed: &SeedTree,
    params: &FlowerParams,
    style: &FlowerStyle,
    accuracy: f64,
    detail: Detail,
) -> Node {
    let seed = seed.child("flower");
    let acc = sane_accuracy(accuracy);
    let jitter = finite_or(params.jitter, 0.0).clamp(0.0, 1.0);
    let n = params.layers.len();
    let mut root = Node::empty().with_id("flower");
    // Receptacle: a dark disc under all petals closes the gap their bases leave around the
    // centre (visible in flowers without a centre ornament) and reads as depth.
    let receptacle = params
        .layers
        .iter()
        .filter(|l| l.count > 0)
        .map(|l| {
            finite_or(l.radius, 0.0).max(0.0)
                + 0.15 * finite_or(l.petal.length * l.scale, 0.0).abs()
        })
        .fold(f64::INFINITY, f64::min);
    if let Some(disc) = circle(Point::ORIGIN, receptacle, acc) {
        root.push(Node::path(
            disc,
            Style::fill(style.petal.mix(style.ink, 0.4)),
        ));
    }
    for (li, layer) in params.layers.iter().enumerate() {
        // 0 for the outermost layer, 1 for the innermost: outer layers are darker.
        let depth = if n > 1 {
            li as f64 / (n - 1) as f64
        } else {
            0.3
        };
        let base = style.petal.mix(style.petal_light, 0.1 + 0.55 * depth);
        let lseed = seed.child("layer").index(li as u64);
        root.push(layer_node(&lseed, layer, jitter, base, style, acc, detail));
    }
    root.push(center_node(
        &seed.child("center"),
        &params.center,
        style,
        acc,
        detail,
    ));
    root
}

/// One ring of petals.
///
/// Draw order: petals are drawn in increasing angle, so each petal covers its predecessor
/// and the ring reads as consistently imbricated (like a contorted corolla). The cycle
/// cannot close in plain painter's order: the last petal would cover the first. So the
/// first petal is drawn a second time at the end, clipped to the region
/// `last petal minus second petal`: it then wins exactly where it overlaps the last petal,
/// without hiding the second petal, which must stay on top of it. Only where three petals
/// overlap (very dense rings) no consistent order exists, and the second petal wins.
fn layer_node(
    seed: &SeedTree,
    layer: &PetalLayer,
    jitter: f64,
    base: Color,
    style: &FlowerStyle,
    acc: f64,
    detail: Detail,
) -> Node {
    let count = layer.count.min(4096);
    let mut group = Node::empty();
    if count == 0 {
        return group;
    }
    let step = TAU / count as f64;
    let rotation = finite_or(layer.rotation, 0.0);
    let radius = finite_or(layer.radius, 0.0);
    let scale = finite_or(layer.scale, 1.0).max(0.0);
    let tilt = finite_or(layer.tilt, 0.0);
    let mut placed: Vec<(BezPath, Node, Affine)> = Vec::with_capacity(count);
    for i in 0..count {
        let pseed = seed.index(i as u64);
        let mut rng = pseed.child("jitter").rng();
        let angle = rotation + step * i as f64 + rng.jitter(0.0, jitter * 0.2 * step.min(1.0));
        let k = scale * (1.0 + rng.jitter(0.0, jitter * 0.08));
        let mut pp = layer.petal;
        pp.length *= k;
        pp.width *= k * (1.0 + rng.jitter(0.0, jitter * 0.06));
        pp.curl += rng.jitter(0.0, jitter * 0.15);
        pp.asymmetry += rng.jitter(0.0, jitter * 0.35);
        let shape = Petal::new(&pp).with_tilt(tilt + rng.jitter(0.0, jitter * 0.12));
        let fill = base
            .mix(style.petal_light, rng.uniform(0.0, 0.15) * jitter)
            .lighten(rng.jitter(0.0, 0.03 * jitter));
        let (outline, node) = petal_with_detail(&pseed, &shape, fill, style, acc, detail);
        // Local +y (the petal axis) must point along the radial direction (cos a, sin a).
        let t = Affine::translate(Vec2::new(angle.cos(), angle.sin()) * radius)
            * Affine::rotate(angle - FRAC_PI_2);
        placed.push((outline, node, t));
    }
    if count >= 3 {
        let (outline0, node0, t0) = &placed[0];
        let local = |i: usize| {
            let (outline, _, t) = &placed[i];
            let mut p = outline.clone();
            p.apply_affine(t0.inverse() * *t);
            p
        };
        let over_last = local(count - 1);
        // Complement of the second petal: a rectangle around petal 0 plus the second petal's
        // outline wound the other way, so the non-zero rule excludes its interior.
        let mut next = local(1);
        let frame = outline0
            .bounding_box()
            .union(next.bounding_box())
            .inflate(1.0, 1.0);
        let mut not_next = frame.to_path(0.1);
        if (not_next.area() > 0.0) == (next.area() > 0.0) {
            next = next.reverse_subpaths();
        }
        not_next.extend(next);
        let again = node0
            .clone()
            .clipped(over_last)
            .clipped(not_next)
            .with_transform(*t0);
        for (_, node, t) in placed {
            group.push(node.with_transform(t));
        }
        group.push(again);
    } else {
        for (_, node, t) in placed {
            group.push(node.with_transform(t));
        }
    }
    group
}

/// Golden angle `π (3 - √5)`, the divergence angle of sunflower florets.
const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;

/// A disc with florets on a Vogel spiral: floret `k` at radius `R·sqrt((k+½)/n)` and angle
/// `k·golden_angle`, which packs them evenly. Florets grow slightly towards the rim and get
/// lighter towards the middle to suggest a dome.
fn disc_nodes(
    radius: f64,
    seeds: usize,
    rot: f64,
    style: &FlowerStyle,
    acc: f64,
    detail: Detail,
) -> Vec<Node> {
    let mut out = Vec::new();
    let Some(disc) = circle(Point::ORIGIN, radius, acc) else {
        return out;
    };
    out.push(Node::path(
        disc,
        Style::fill_stroke(style.center, style.ink, 0.04 * radius),
    ));
    let n = if detail.canvas_size(radius) < LOD_FLORETS {
        0
    } else {
        seeds.min(5000)
    };
    let r_in = 0.9 * radius;
    for k in 0..n {
        let f = (k as f64 + 0.5) / n as f64;
        let r = r_in * f.sqrt();
        let a = rot + k as f64 * GOLDEN_ANGLE;
        let dot = r_in * (1.0 / n as f64).sqrt() * 0.5 * (0.75 + 0.35 * f);
        let c = Point::new(r * a.cos(), r * a.sin());
        let col = style
            .center
            .mix(style.ink, 0.45 * f)
            .lighten(0.12 * (1.0 - f));
        if let Some(path) = circle(c, dot, acc) {
            out.push(Node::path(path, Style::fill(col)));
        }
    }
    out
}

fn center_node(
    seed: &SeedTree,
    center: &CenterStyle,
    style: &FlowerStyle,
    acc: f64,
    detail: Detail,
) -> Node {
    let mut rng = seed.rng();
    let mut nodes = Vec::new();
    match *center {
        CenterStyle::None => {}
        CenterStyle::Disc { radius, seeds } => {
            let radius = finite_or(radius, 0.0).max(0.0);
            nodes = disc_nodes(radius, seeds, rng.uniform(0.0, TAU), style, acc, detail);
        }
        CenterStyle::Stamens {
            count,
            length,
            radius,
        } => {
            let radius = finite_or(radius, 0.0).max(0.0);
            let length = finite_or(length, 0.0).max(0.0);
            let count = count.min(2000);
            let mut heads = Vec::new();
            let swirl = rng.jitter(0.0, 0.35);
            for i in 0..count {
                let a = TAU * (i as f64 + rng.uniform(0.0, 0.6)) / count.max(1) as f64;
                let reach = radius + length * rng.uniform(0.75, 1.05);
                let bend = swirl + rng.jitter(0.0, 0.2);
                let at = |r: f64, a: f64| Point::new(r * a.cos(), r * a.sin());
                let q = QuadBez::new(
                    at(0.5 * radius, a),
                    at(radius + 0.5 * length, a + 0.4 * bend),
                    at(reach, a + bend),
                );
                let w = (0.05 * length).max(0.01 * radius);
                let tol = acc.max(DETAIL_TOLERANCE * length);
                let stalk = tapered(&q, |t| w * (1.0 - 0.5 * t), tol);
                if !stalk.elements().is_empty() {
                    nodes.push(Node::path(
                        stalk,
                        Style::fill(style.center.mix(style.ink, 0.35)),
                    ));
                }
                let head_r = (0.07 * length).max(0.02 * radius);
                if let Some(path) = circle(q.p2, head_r, acc) {
                    heads.push(Node::path(
                        path,
                        Style::fill_stroke(style.center.lighten(0.2), style.ink, 0.25 * head_r),
                    ));
                }
            }
            nodes.extend(heads);
            let seeds = ((radius * radius) / (0.04 * length * length).max(1e-9)).clamp(5.0, 60.0);
            nodes.extend(disc_nodes(
                radius,
                seeds as usize,
                PI * 0.1,
                style,
                acc,
                detail,
            ));
        }
    }
    Node::group(nodes)
}

// ---------------------------------------------------------------------------------------
// Variants
// ---------------------------------------------------------------------------------------

/// Named, visually distinct flowers with an outer radius of roughly 100 units. Shapes are
/// fixed per species; counts, proportions, curl and rotation are drawn from `seed`.
pub fn variants(seed: &SeedTree) -> Vec<(String, FlowerParams)> {
    let seed = seed.child("flower-variants");
    let mut out = Vec::new();
    let mut add = |name: &str, build: &dyn Fn(&mut ChaCha8Rng) -> FlowerParams| {
        let mut rng = seed.child(name).rng();
        out.push((name.to_string(), build(&mut rng)));
    };
    use PetalProfile::*;
    let layer =
        |count: usize, petal: PetalParams, radius: f64, rotation: f64, tilt: f64| PetalLayer {
            count,
            petal,
            radius,
            rotation,
            scale: 1.0,
            tilt,
        };
    let pp = |profile, length, width, base_width, tip, curl| PetalParams {
        profile,
        length,
        width,
        base_width,
        tip,
        curl,
        asymmetry: 0.0,
    };

    add("daisy", &|r| {
        let n = *r.pick(&[17usize, 21, 26]);
        let rot = r.uniform(0.0, TAU);
        let len = r.uniform(70.0, 78.0);
        let ray = pp(
            NarrowRay,
            len,
            r.uniform(11.0, 14.0),
            0.3,
            0.2,
            r.jitter(0.0, 0.08),
        );
        let inner = PetalParams {
            length: 0.9 * len,
            ..ray
        };
        FlowerParams {
            layers: vec![
                layer(n, ray, 16.0, rot, 0.0),
                layer(n, inner, 14.0, rot + PI / n as f64, 0.15),
            ],
            center: CenterStyle::Disc {
                radius: 20.0,
                seeds: r.uniform(90.0, 130.0) as usize,
            },
            jitter: 0.6,
        }
    });
    add("aster", &|r| {
        let n = *r.pick(&[30usize, 34, 40]);
        let rot = r.uniform(0.0, TAU);
        let curl = r.jitter(0.0, 0.3);
        let ray = pp(
            NarrowRay,
            r.uniform(66.0, 74.0),
            r.uniform(6.5, 8.0),
            0.3,
            0.5,
            curl,
        );
        let inner = PetalParams {
            length: 0.82 * ray.length,
            width: 0.9 * ray.width,
            ..ray
        };
        FlowerParams {
            layers: vec![
                layer(n, ray, 14.0, rot, 0.0),
                layer(n, inner, 12.0, rot + PI / n as f64, 0.2),
            ],
            center: CenterStyle::Disc {
                radius: 15.0,
                seeds: r.uniform(60.0, 90.0) as usize,
            },
            jitter: 0.8,
        }
    });
    add("rose", &|r| {
        let swirl = if r.chance(0.5) { 1.0 } else { -1.0 } * r.uniform(0.2, 0.35);
        let rot = r.uniform(0.0, TAU);
        let rings = [
            (5, 64.0, 66.0, 6.0, 0.0),
            (5, 54.0, 56.0, 5.0, 0.35),
            (5, 44.0, 46.0, 4.0, 0.55),
            (4, 33.0, 36.0, 3.0, 0.75),
            (3, 22.0, 26.0, 1.5, 0.95),
        ];
        let layers = rings
            .iter()
            .enumerate()
            .map(|(i, &(n, l, w, rad, tilt))| {
                let p = pp(Rounded, l, w, 0.25, r.uniform(0.05, 0.3), swirl);
                layer(n, p, rad, rot + i as f64 * 0.7, tilt)
            })
            .collect();
        FlowerParams {
            layers,
            center: CenterStyle::None,
            jitter: 0.7,
        }
    });
    add("peony", &|r| {
        let rot = r.uniform(0.0, TAU);
        let tip = r.uniform(0.0, 0.3);
        FlowerParams {
            layers: vec![
                layer(
                    6,
                    pp(Rounded, 66.0, 70.0, 0.2, tip, r.jitter(0.0, 0.1)),
                    8.0,
                    rot,
                    0.0,
                ),
                layer(
                    8,
                    pp(Fringed, 54.0, 44.0, 0.2, tip, 0.1),
                    7.0,
                    rot + 0.4,
                    0.3,
                ),
                layer(
                    10,
                    pp(Fringed, 42.0, 34.0, 0.2, tip, -0.1),
                    6.0,
                    rot + 0.9,
                    0.45,
                ),
                layer(
                    12,
                    pp(Fringed, 30.0, 22.0, 0.2, tip, 0.15),
                    5.0,
                    rot + 0.2,
                    0.6,
                ),
            ],
            center: CenterStyle::Stamens {
                count: 28,
                length: 10.0,
                radius: 7.0,
            },
            jitter: 0.8,
        }
    });
    add("lily", &|r| {
        let rot = r.uniform(0.0, TAU);
        let curl = r.uniform(0.05, 0.2) * if r.chance(0.5) { 1.0 } else { -1.0 };
        let outer = pp(
            Pointed,
            r.uniform(88.0, 96.0),
            r.uniform(34.0, 40.0),
            0.25,
            0.7,
            curl,
        );
        let inner = PetalParams {
            length: 0.95 * outer.length,
            width: 0.85 * outer.width,
            ..outer
        };
        FlowerParams {
            layers: vec![
                layer(3, outer, 5.0, rot, 0.0),
                layer(3, inner, 5.0, rot + PI / 3.0, 0.1),
            ],
            center: CenterStyle::Stamens {
                count: 6,
                length: 46.0,
                radius: 5.0,
            },
            jitter: 0.5,
        }
    });
    add("lotus", &|r| {
        let rot = r.uniform(0.0, TAU);
        let n = *r.pick(&[8usize, 9]);
        FlowerParams {
            layers: vec![
                layer(n, pp(Rounded, 80.0, 46.0, 0.25, 0.9, 0.0), 10.0, rot, 0.25),
                layer(
                    n,
                    pp(Rounded, 68.0, 40.0, 0.25, 0.9, 0.0),
                    9.0,
                    rot + PI / n as f64,
                    0.5,
                ),
                layer(
                    6,
                    pp(Rounded, 48.0, 32.0, 0.25, 0.85, 0.0),
                    8.0,
                    rot + 0.2,
                    0.75,
                ),
            ],
            center: CenterStyle::Disc {
                radius: 15.0,
                seeds: r.uniform(9.0, 14.0) as usize,
            },
            jitter: 0.5,
        }
    });
    add("carnation", &|r| {
        let rot = r.uniform(0.0, TAU);
        let curl = r.jitter(0.0, 0.2);
        FlowerParams {
            layers: vec![
                layer(10, pp(Fringed, 60.0, 46.0, 0.15, 0.2, curl), 6.0, rot, 0.0),
                layer(
                    10,
                    pp(Fringed, 48.0, 38.0, 0.15, 0.2, -curl),
                    5.0,
                    rot + PI / 10.0,
                    0.35,
                ),
                layer(
                    8,
                    pp(Fringed, 32.0, 28.0, 0.15, 0.2, curl),
                    3.0,
                    rot + 0.3,
                    0.6,
                ),
            ],
            center: CenterStyle::None,
            jitter: 0.9,
        }
    });
    add("bellflower", &|r| {
        let rot = r.uniform(0.0, TAU);
        let bell = pp(
            Bell,
            r.uniform(68.0, 74.0),
            r.uniform(44.0, 50.0),
            0.6,
            0.8,
            r.jitter(0.0, 0.15),
        );
        FlowerParams {
            layers: vec![layer(5, bell, 3.0, rot, 0.0)],
            center: CenterStyle::Stamens {
                count: 5,
                length: 22.0,
                radius: 9.0,
            },
            jitter: 0.5,
        }
    });
    add("primrose", &|r| {
        let rot = r.uniform(0.0, TAU);
        let p = pp(
            Notched,
            r.uniform(58.0, 64.0),
            r.uniform(52.0, 58.0),
            0.2,
            r.uniform(0.0, 0.5),
            0.0,
        );
        FlowerParams {
            layers: vec![layer(5, p, 9.0, rot, 0.0)],
            center: CenterStyle::Disc {
                radius: 11.0,
                seeds: 34,
            },
            jitter: 0.5,
        }
    });
    add("poppy", &|r| {
        let rot = r.uniform(0.0, TAU);
        let p = pp(
            Rounded,
            r.uniform(70.0, 76.0),
            r.uniform(96.0, 104.0),
            0.3,
            0.0,
            r.jitter(0.0, 0.1),
        );
        let inner = PetalParams {
            length: 0.9 * p.length,
            width: 0.9 * p.width,
            ..p
        };
        FlowerParams {
            layers: vec![
                layer(2, p, 4.0, rot, 0.0),
                layer(2, inner, 4.0, rot + FRAC_PI_2, 0.25),
            ],
            center: CenterStyle::Stamens {
                count: 40,
                length: 16.0,
                radius: 11.0,
            },
            jitter: 0.6,
        }
    });
    add("cosmos", &|r| {
        let rot = r.uniform(0.0, TAU);
        let p = pp(
            Spoon,
            r.uniform(70.0, 78.0),
            r.uniform(40.0, 46.0),
            0.32,
            r.uniform(0.0, 0.3),
            r.jitter(0.0, 0.12),
        );
        FlowerParams {
            layers: vec![layer(8, p, 12.0, rot, 0.0)],
            center: CenterStyle::Disc {
                radius: 15.0,
                seeds: r.uniform(50.0, 70.0) as usize,
            },
            jitter: 0.6,
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};

    fn well_formed(path: &BezPath) -> bool {
        path_is_finite(path) && degenerate_segments(path, 1e-9) == 0
    }

    fn all_paths_ok(node: &Node) -> bool {
        let mut ok = node.is_finite();
        node.for_each_path(|p, s, _| ok &= well_formed(p) && s.stroke_width.is_finite());
        ok
    }

    #[test]
    fn small_flowers_drop_fine_engraving_first() {
        let (_, params) = variants(&SeedTree::new(2))
            .into_iter()
            .find(|(n, _)| n == "rose")
            .unwrap();
        let s = style();
        let seed = SeedTree::new(4);
        let draw = |d: Detail| flower_with_detail(&seed, &params, &s, 0.05, d);
        let full = flower(&seed, &params, &s, 0.05);
        assert_eq!(draw(Detail::FULL), full);
        // Outer radius on the canvas: 200, 60, 20 and 5 px.
        let counts: Vec<usize> = [200.0, 60.0, 20.0, 5.0]
            .into_iter()
            .map(|r| draw(Detail::at_scale(r / params.outer_radius())).path_count())
            .collect();
        assert_eq!(counts[0], full.path_count(), "{counts:?}");
        assert!(counts.windows(2).all(|w| w[0] >= w[1]), "{counts:?}");
        assert!(counts[3] < full.path_count() / 2, "{counts:?}");
    }

    fn style() -> FlowerStyle {
        FlowerStyle::from_palette(&crate::palette::all()[0], true, 0.4)
    }

    /// About one sample per `step` of arc length, so the result does not depend on how
    /// many segments the fit chose.
    fn sample_by_length(path: &BezPath, step: f64) -> Vec<Point> {
        use kurbo::ParamCurveArclen;
        path.segments()
            .flat_map(|seg| {
                let n = (seg.arclen(1e-3) / step).ceil().max(1.0) as usize;
                (0..n).map(move |i| seg.eval(i as f64 / n as f64))
            })
            .collect()
    }

    fn params(profile: PetalProfile) -> PetalParams {
        PetalParams {
            curl: 0.3,
            asymmetry: 0.3,
            ..PetalParams::new(profile, 100.0, 50.0)
        }
    }

    #[test]
    fn petal_outlines_are_closed_finite_and_non_degenerate() {
        for profile in PetalProfile::ALL {
            for p in [PetalParams::new(profile, 100.0, 50.0), params(profile)] {
                let petal = Petal::new(&p).with_tilt(0.4);
                let o = petal.outline(0.1);
                assert!(well_formed(&o), "{profile:?}");
                assert!(matches!(o.elements().last(), Some(PathEl::ClosePath)));
                assert!(o.area().abs() > 500.0, "{profile:?} area {}", o.area());
            }
        }
    }

    #[test]
    fn petal_frame_matches_contract() {
        for profile in PetalProfile::ALL {
            let petal = Petal::new(&PetalParams::new(profile, 100.0, 40.0));
            assert!(petal.point(0.0, 0.0).distance(Point::ORIGIN) < 1e-9);
            let tip = petal.point(1.0, 0.0);
            assert!(
                tip.x.abs() < 1e-9 && (tip.y - 100.0).abs() < 15.0,
                "{profile:?} {tip:?}"
            );
            // Mirror symmetry without asymmetry or curl.
            let (a, b) = (petal.point(0.5, 1.0), petal.point(0.5, -1.0));
            assert!((a.x + b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
        }
        // Rounded petals: narrow base, widest point past the middle.
        let p = Petal::new(&PetalParams::new(PetalProfile::Rounded, 100.0, 60.0));
        let widest = (1..100)
            .map(|i| i as f64 / 100.0)
            .max_by(|a, b| p.half_width(*a).total_cmp(&p.half_width(*b)))
            .unwrap();
        assert!(p.point(widest, 1.0).y > 50.0);
        assert!(p.half_width(0.02) < 0.5 * p.half_width(widest));
    }

    #[test]
    fn curl_bends_the_tip_and_keeps_length() {
        let straight = Petal::new(&PetalParams::new(PetalProfile::Pointed, 100.0, 30.0));
        let bent = Petal::new(&PetalParams {
            curl: 0.8,
            ..PetalParams::new(PetalProfile::Pointed, 100.0, 30.0)
        });
        assert!(bent.point(1.0, 0.0).x < -10.0, "positive curl bends to -x");
        let l0 = straight.iso_v(0.0).length();
        let l1 = bent.iso_v(0.0).length();
        assert!((l0 - l1).abs() < 0.01 * l0, "{l0} vs {l1}");
        // Near the base the bend is negligible.
        assert!(bent.point(0.05, 0.0).x.abs() < 0.1);
    }

    #[test]
    fn veins_lie_inside_the_outline() {
        for profile in PetalProfile::ALL {
            let petal = Petal::new(&params(profile));
            let outline = petal.outline(0.05);
            let detail = petal_detail(&SeedTree::new(3), &petal, 0.05, Detail::FULL);
            assert!(detail.iter().filter(|(_, k)| *k == Ink::Vein).count() >= 3);
            let (mut inside, mut total) = (0usize, 0usize);
            for (path, ink) in &detail {
                assert!(well_formed(path), "{profile:?}");
                if *ink == Ink::Shade {
                    // Shading regions share the petal edges by construction.
                    continue;
                }
                for pt in sample_by_length(path, 0.5) {
                    total += 1;
                    inside += usize::from(outline.winding(pt) != 0);
                }
            }
            let ratio = inside as f64 / total as f64;
            assert!(ratio > 0.95, "{profile:?}: {ratio}");
        }
    }

    #[test]
    fn petal_node_is_clipped_and_deterministic() {
        let s = style();
        let petal_shape = Petal::new(&params(PetalProfile::Rounded));
        let (o1, n1) = petal(&SeedTree::new(1), &petal_shape, s.petal, &s, 0.1);
        let (o2, n2) = petal(&SeedTree::new(1), &petal_shape, s.petal, &s, 0.1);
        assert_eq!(o1, o2);
        assert_eq!(n1, n2);
        assert!(all_paths_ok(&n1));
        let Node::Group { children, .. } = &n1 else {
            panic!("petal node is a group")
        };
        assert!(matches!(&children[1], Node::Group { clip: Some(c), .. } if c.path == o1));
    }

    #[test]
    fn flowers_are_reproducible_and_seed_dependent() {
        let s = style();
        let v = variants(&SeedTree::new(5));
        let (_, p) = &v[0];
        let a = flower(&SeedTree::new(9), p, &s, 0.2);
        let b = flower(&SeedTree::new(9), p, &s, 0.2);
        let c = flower(&SeedTree::new(10), p, &s, 0.2);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(variants(&SeedTree::new(5)), v);
        assert_ne!(variants(&SeedTree::new(6)), v);
    }

    #[test]
    fn all_variants_build_cleanly() {
        let s = style();
        for seed in 0..3 {
            let vs = variants(&SeedTree::new(seed));
            assert!(vs.len() >= 6);
            for (name, p) in &vs {
                let node = flower(&SeedTree::new(seed), p, &s, 0.25);
                assert!(all_paths_ok(&node), "{name}");
                let bb = node.bounding_box().expect("non-empty");
                let r = p.outer_radius();
                assert!(
                    bb.width() > 0.8 * r && bb.width() < 2.6 * r,
                    "{name} {bb:?} r={r}"
                );
            }
        }
    }

    #[test]
    fn extreme_parameters_do_not_panic() {
        let s = style();
        let base = PetalParams::new(PetalProfile::Rounded, 40.0, 20.0);
        let weird = [
            PetalParams { width: 0.0, ..base },
            PetalParams {
                length: 0.0,
                ..base
            },
            PetalParams {
                length: -5.0,
                width: f64::NAN,
                ..base
            },
            PetalParams {
                curl: 50.0,
                asymmetry: 9.0,
                tip: -3.0,
                base_width: 5.0,
                ..base
            },
            PetalParams {
                curl: f64::INFINITY,
                width: 1e6,
                ..base
            },
        ];
        for profile in PetalProfile::ALL {
            for p in weird {
                let p = PetalParams { profile, ..p };
                let shape = Petal::new(&p).with_tilt(f64::NAN);
                let (o, n) = petal(&SeedTree::new(0), &shape, s.petal, &s, f64::NAN);
                assert!(well_formed(&o), "{profile:?} {p:?}");
                assert!(all_paths_ok(&n), "{profile:?} {p:?}");
            }
        }
        for count in [0, 1, 2, 200] {
            let fp = FlowerParams {
                layers: vec![PetalLayer {
                    radius: 3.0,
                    tilt: 1.4,
                    ..PetalLayer::new(count, PetalParams::new(PetalProfile::Spoon, 30.0, 12.0))
                }],
                center: CenterStyle::Stamens {
                    count: 7,
                    length: 0.0,
                    radius: 0.0,
                },
                jitter: 1.0,
            };
            let node = flower(&SeedTree::new(1), &fp, &s, 0.5);
            assert!(all_paths_ok(&node), "count {count}");
        }
        let nan_layer = FlowerParams {
            layers: vec![PetalLayer {
                radius: f64::NAN,
                rotation: f64::INFINITY,
                scale: f64::NAN,
                tilt: f64::NAN,
                ..PetalLayer::new(4, base)
            }],
            center: CenterStyle::Disc {
                radius: f64::NAN,
                seeds: 10,
            },
            jitter: f64::NAN,
        };
        assert!(all_paths_ok(&flower(
            &SeedTree::new(2),
            &nan_layer,
            &s,
            0.0
        )));
    }

    /// Number of proper crossings between non-adjacent segments of a polyline.
    fn crossings(pts: &[Point]) -> usize {
        let side = |p: Point, q: Point, r: Point| (q - p).cross(r - p);
        let m = pts.len();
        let mut k = 0;
        for i in 0..m - 1 {
            for j in i + 2..m - 1 {
                let (a, b, c, d) = (pts[i], pts[i + 1], pts[j], pts[j + 1]);
                if side(a, b, c) * side(a, b, d) < 0.0 && side(c, d, a) * side(c, d, b) < 0.0 {
                    k += 1;
                }
            }
        }
        k
    }

    #[test]
    fn narrow_fringed_edges_do_not_cross_at_the_tip() {
        // Regression: the ripple did not shrink with the gap between the edges, so on narrow
        // pointed fringed petals the two edges crossed just below the tip.
        for width in [3.0, 5.0, 8.0, 12.0] {
            for asymmetry in [-1.0, 0.0, 1.0] {
                for curl in [0.0, 0.3] {
                    let p = PetalParams {
                        tip: 1.0,
                        curl,
                        asymmetry,
                        ..PetalParams::new(PetalProfile::Fringed, 100.0, width)
                    };
                    let petal = Petal::new(&p);
                    let n = 400;
                    let mut pts: Vec<Point> = (0..=n)
                        .map(|i| petal.point(i as f64 / n as f64, 1.0))
                        .collect();
                    pts.extend((0..n).rev().map(|i| petal.point(i as f64 / n as f64, -1.0)));
                    assert_eq!(crossings(&pts), 0, "w={width} a={asymmetry} curl={curl}");
                }
            }
        }
    }

    #[test]
    fn bent_outlines_never_double_back() {
        // Regression: on bent petals the adaptive fit sometimes returned a cubic starting
        // against the previous tangent (a cusp spike in the inked edge).
        let cases = [
            (PetalProfile::Rounded, 15.0, -0.5, 0.0, 0.0, 0.8, 0.1),
            (PetalProfile::Rounded, 15.0, -0.5, 0.3, 1.0, 0.0, 0.1),
            (PetalProfile::Pointed, 15.0, -1.0, 0.0, 1.0, 0.0, 0.01),
            (PetalProfile::Spoon, 15.0, -1.0, 0.0, 1.0, -0.8, 0.1),
            (PetalProfile::Spoon, 70.0, 50.0, 0.9, 0.0, 0.0, 0.01),
            (PetalProfile::Notched, 15.0, -0.5, 0.0, 0.5, 0.0, 0.1),
            (PetalProfile::NarrowRay, 30.0, 0.5, 0.0, 0.5, 0.8, 0.1),
            (PetalProfile::Bell, 15.0, -2.0, 0.0, 1.0, 0.0, 0.5),
        ];
        for (profile, width, curl, base_width, tip, asymmetry, acc) in cases {
            let p = PetalParams {
                curl,
                base_width,
                tip,
                asymmetry,
                ..PetalParams::new(profile, 100.0, width)
            };
            let petal = Petal::new(&p);
            let o = petal.outline(acc);
            assert!(well_formed(&o), "{profile:?}");
            // The two edges, split at the tip (a legitimate corner).
            let tip_pt = petal.point(1.0, 1.0);
            // Edges are all cubics; the only line is the base.
            let cubics: Vec<CubicBez> = o
                .segments()
                .filter_map(|s| match s {
                    kurbo::PathSeg::Cubic(c) => Some(c),
                    _ => None,
                })
                .collect();
            let split = cubics
                .iter()
                .position(|c| c.p3.distance(tip_pt) < 1e-6)
                .expect("tip vertex")
                + 1;
            for half in [&cubics[..split], &cubics[split..]] {
                let mut path = BezPath::new();
                path.move_to(half[0].p0);
                for c in half {
                    path.curve_to(c.p1, c.p2, c.p3);
                }
                assert!(
                    !reverses(&path),
                    "{profile:?} w={width} curl={curl} acc={acc}"
                );
            }
            // Still within the requested accuracy of the true edges.
            let truth: Vec<Point> = (0..=2000)
                .flat_map(|i| {
                    let u = i as f64 / 2000.0;
                    [petal.point(u, 1.0), petal.point(u, -1.0)]
                })
                .collect();
            for c in &cubics {
                for k in 0..8 {
                    let q = c.eval(k as f64 / 8.0);
                    let d = truth
                        .iter()
                        .map(|t| t.distance(q))
                        .fold(f64::INFINITY, f64::min);
                    assert!(d < 2.0 * acc + 0.05, "{profile:?} curl={curl}: off by {d}");
                }
            }
        }
    }

    #[test]
    fn huge_petals_do_not_stall_the_fit() {
        // Regression: at 1e12 units the absolute accuracy sat at the f64 resolution and the
        // adaptive outline fit subdivided without end.
        let s = style();
        for profile in PetalProfile::ALL {
            let p = PetalParams {
                curl: 0.3,
                asymmetry: 0.5,
                ..PetalParams::new(profile, 1e12, 5e11)
            };
            let shape = Petal::new(&p);
            let (o, n) = petal(&SeedTree::new(0), &shape, s.petal, &s, 1e-3);
            assert!(well_formed(&o), "{profile:?}");
            assert!(
                o.elements().len() < 5000,
                "{profile:?}: {}",
                o.elements().len()
            );
            assert!(all_paths_ok(&n), "{profile:?}");
        }
    }

    #[test]
    fn disc_florets_follow_the_golden_angle() {
        let s = style();
        let nodes = disc_nodes(20.0, 50, 0.0, &s, 0.1, Detail::FULL);
        assert_eq!(nodes.len(), 51);
        // Every floret stays inside the disc.
        for n in &nodes[1..] {
            let bb = n.bounding_box().unwrap();
            assert!(bb.center().to_vec2().hypot() + 0.5 * bb.width() <= 20.0 + 1e-6);
        }
    }
}
