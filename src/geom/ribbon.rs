//! Variable-width ribbons: a centreline curve swept by a half-width profile on each side.
//!
//! A [`Ribbon`] is a [`Surface`] parameterised by **arc length** along its centreline
//! (`u` is the fraction of the centreline length) and by `v ∈ [-1, 1]` across it:
//!
//! ```text
//! P(u, v) = C(t(u)) + n(t(u)) · d(u, v),   d = v · w_left(u)  for v ≥ 0 (left, along the normal)
//!                                          d = v · w_right(u) for v < 0 (right, against it)
//! ```
//!
//! where `n` is the centreline's left normal (tangent rotated by +90°). So `v = +1` is the
//! left edge and `v = -1` the right edge, seen while walking from `u = 0` to `u = 1`.
//!
//! # Cusps and self-intersections
//!
//! The normal offset at distance `d` of a curve with curvature `κ` has speed `|C'|·(1 − κd)`
//! along the tangent: on the concave side (`κd > 0`) it stops and reverses where `κd ≥ 1`,
//! producing a cusp and a swallowtail loop. The ribbon therefore limits each side's effective
//! half-width to `safety / |κ|` on the concave side (`safety < 1`, default 0.9), divided by
//! `1 + λ²` where the edges are also shifted along the tangent by `λ·w` (lean, sweep). The limit is
//! tabulated on a uniform arc-length grid (curvature from the Menger formula on three points,
//! which is robust against near-zero tangents and parameter speed), passed through a running
//! minimum and a double box blur so the clamped edge narrows smoothly instead of kinking, and
//! combined with the requested width through a smooth minimum. Global overlaps (two distant
//! parts of a ribbon crossing) are not local; [`Ribbon::check`] detects them on the flattened
//! outline and [`Ribbon::fit_width`] removes them by scaling the widths down.

use super::curve::{ArcLength, Curve, FnCurve};
use super::surface::Surface;
use super::{EPS, smoothstep};
use kurbo::{BezPath, CubicBez, ParamCurve, PathEl, Point, Shape, Vec2};
use noise::{NoiseFn, Perlin};
use std::f64::consts::FRAC_PI_2;
use std::fmt;
use std::sync::Arc;

/// Minimum resolution of the arc-length grid used for curvature limits.
const GRID: usize = 512;
/// Maximum resolution of that grid (long or finely detailed centrelines).
const MAX_GRID: usize = 16384;
/// Resolution of the centreline arc-length table.
const ARC_TABLE: usize = 1024;
/// Resolution of the edge arc-length tables used when fitting contours.
const EDGE_TABLE: usize = 1024;
/// Default fraction of the radius of curvature an offset may reach on the concave side.
pub const DEFAULT_SAFETY: f64 = 0.9;
/// Fraction of the length over which the start sweep (see [`Ribbon::with_sweep`]) fades out.
const SWEEP_SPAN: f64 = 0.32;
/// Extra width reduction applied after [`Ribbon::fit_width`]'s bisection so that parts
/// which would just touch keep a visible gap.
const FIT_MARGIN: f64 = 0.94;
/// `4/3·(√2 − 1)`: cubic control distance approximating a quarter circle.
const KAPPA: f64 = 0.552_284_749_830_793_4;

/// Power ease `sin(πx/2)^p` on `x ∈ [0, 1]`.
///
/// It rises from 0 to 1 with zero slope at `x = 1`, so it joins a plateau or a mirrored ease
/// without a kink. Near `x = 0` it behaves like `x^p`, which sets the shape of a tip: `p = 0.5`
/// gives a rounded (parabolic) end, `p = 1` a straight point, `p > 1` a slender concave needle.
pub fn ease(x: f64, power: f64) -> f64 {
    let x = if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let p = if power.is_finite() {
        power.clamp(0.05, 16.0)
    } else {
        1.0
    };
    (FRAC_PI_2 * x).sin().max(0.0).powf(p)
}

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() { x } else { fallback }
}

fn non_negative(x: f64) -> f64 {
    if x.is_finite() && x > 0.0 { x } else { 0.0 }
}

type ProfileFn = dyn Fn(f64) -> f64 + Send + Sync;

/// A half-width profile `w(u) ≥ 0` for `u ∈ [0, 1]`.
///
/// Built from a few expressive constructors ([`tapered`](Self::tapered),
/// [`swell`](Self::swell), [`lens`](Self::lens), [`teardrop`](Self::teardrop),
/// [`custom`](Self::custom)) and refined with modifiers ([`scale`](Self::scale),
/// [`map`](Self::map), [`with_noise`](Self::with_noise)). Cheap to clone (shared closure).
/// [`eval`](Self::eval) always returns a finite, non-negative value.
#[derive(Clone)]
pub struct WidthProfile {
    f: Arc<ProfileFn>,
}

impl fmt::Debug for WidthProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WidthProfile")
            .field("w(0)", &self.eval(0.0))
            .field("w(0.5)", &self.eval(0.5))
            .field("w(1)", &self.eval(1.0))
            .finish()
    }
}

impl WidthProfile {
    /// Arbitrary profile from a closure (non-finite or negative results are treated as 0).
    pub fn custom(f: impl Fn(f64) -> f64 + Send + Sync + 'static) -> Self {
        WidthProfile { f: Arc::new(f) }
    }

    /// Constant half-width (open, rounded-cap ends).
    pub fn constant(w: f64) -> Self {
        let w = non_negative(w);
        Self::custom(move |_| w)
    }

    /// Zero everywhere (a degenerate ribbon; its outline is empty).
    pub fn zero() -> Self {
        Self::constant(0.0)
    }

    /// Plateau of half-width `max` with eased tapers to exactly 0 over the first `taper_start`
    /// and the last `taper_end` fraction of the length (0 keeps that end open at full width).
    /// `power` shapes the tips, see [`ease`]. Tapers longer than the ribbon are rescaled to meet.
    pub fn tapered(max: f64, taper_start: f64, taper_end: f64, power: f64) -> Self {
        let max = non_negative(max);
        let mut ts = finite_or(taper_start, 0.0).clamp(0.0, 1.0);
        let mut te = finite_or(taper_end, 0.0).clamp(0.0, 1.0);
        if ts + te > 1.0 {
            let k = 1.0 / (ts + te);
            ts *= k;
            te *= k;
        }
        Self::custom(move |u| {
            let a = if ts > 0.0 && u < ts {
                ease(u / ts, power)
            } else {
                1.0
            };
            let b = if te > 0.0 && u > 1.0 - te {
                ease((1.0 - u) / te, power)
            } else {
                1.0
            };
            max * a * b
        })
    }

    /// Single swell: 0 at both ends, maximum `max` at `u = peak`, with independent tip
    /// powers for the start and the end (see [`ease`]). The two halves join with zero slope.
    pub fn swell(max: f64, peak: f64, start_power: f64, end_power: f64) -> Self {
        let max = non_negative(max);
        let m = finite_or(peak, 0.5).clamp(0.02, 0.98);
        Self::custom(move |u| {
            if u < m {
                max * ease(u / m, start_power)
            } else {
                max * ease((1.0 - u) / (1.0 - m), end_power)
            }
        })
    }

    /// Symmetric lens (vesica): widest in the middle, both ends shaped by `power`.
    pub fn lens(max: f64, power: f64) -> Self {
        Self::swell(max, 0.5, power, power)
    }

    /// Teardrop: bulges at `peak` with a rounded end on the near side and a pointed tip on the
    /// far side (`peak < 0.5` puts the round end at the start, `peak > 0.5` at the end).
    pub fn teardrop(max: f64, peak: f64) -> Self {
        let m = finite_or(peak, 0.3).clamp(0.02, 0.98);
        if m <= 0.5 {
            Self::swell(max, m, 0.5, 1.25)
        } else {
            Self::swell(max, m, 1.25, 0.5)
        }
    }

    /// Multiply all widths by `k` (negative or non-finite `k` gives zero).
    pub fn scale(self, k: f64) -> Self {
        let k = non_negative(k);
        self.map(move |_, w| w * k)
    }

    /// General modulation: the new width at `u` is `f(u, w(u))`.
    pub fn map(self, f: impl Fn(f64, f64) -> f64 + Send + Sync + 'static) -> Self {
        let inner = self.f;
        Self::custom(move |u| f(u, non_negative(inner(u))))
    }

    /// Multiplicative smooth noise: `w·(1 + amplitude·perlin(frequency·u))`, seeded with
    /// `seed` (e.g. from [`SeedTree::noise_seed`](crate::rng::SeedTree::noise_seed)).
    /// Being multiplicative it keeps pointed tips at exactly zero. `amplitude` is clamped to
    /// `[0, 0.95]`; `frequency` counts noise features over the whole length.
    pub fn with_noise(self, seed: u32, amplitude: f64, frequency: f64) -> Self {
        let perlin = Perlin::new(seed);
        let amp = finite_or(amplitude, 0.0).clamp(0.0, 0.95);
        let freq = finite_or(frequency, 0.0).clamp(0.0, 1e4);
        // Offset away from the integer lattice, where 1-D Perlin noise is always zero.
        self.map(move |u, w| w * (1.0 + amp * perlin.get([u * freq + 0.371]).clamp(-1.0, 1.0)))
    }

    /// Half-width at `u` (clamped to `[0, 1]`); always finite and `>= 0`.
    pub fn eval(&self, u: f64) -> f64 {
        let u = finite_or(u, 0.0).clamp(0.0, 1.0);
        non_negative((self.f)(u))
    }

    /// Largest value over `n` uniform samples.
    pub fn max_sampled(&self, n: usize) -> f64 {
        let n = n.max(2);
        (0..n)
            .map(|i| self.eval(i as f64 / (n - 1) as f64))
            .fold(0.0, f64::max)
    }
}

/// Validity report of a ribbon outline, see [`Ribbon::check`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RibbonReport {
    /// Maximum of `w·|κ|·(1 + λ²)` over both sides on their concave side, with the
    /// effective (clamped) widths and the lean/sweep shift factor `λ` (0 for a plain
    /// normal offset). Values below 1 mean no local offset cusp is possible.
    pub max_width_curvature_product: f64,
    /// Number of offset cusps (the edge stops and turns back) along the two sampled edges.
    pub cusp_count: usize,
    /// Number of proper crossings between non-adjacent segments of the flattened outline.
    pub self_intersections: usize,
}

impl RibbonReport {
    /// `true` when the outline has neither cusps nor self-intersections.
    pub fn is_valid(&self) -> bool {
        self.cusp_count == 0 && self.self_intersections == 0
    }
}

/// A variable-width ribbon around a centreline; see the module docs for the parameterisation.
pub struct Ribbon<C: Curve> {
    centerline: C,
    left: WidthProfile,
    right: WidthProfile,
    scale: f64,
    safety: f64,
    sweep: f64,
    lean: f64,
    arclen: ArcLength,
    length: f64,
    /// Number of steps of the uniform arc-length curvature grid, see [`grid_size`].
    grid: usize,
    /// Signed curvature on the uniform arc-length grid (`grid + 1` samples).
    kappa: Vec<f64>,
    /// Smoothed concave-side width limits on the same grid.
    limit_left: Vec<f64>,
    limit_right: Vec<f64>,
}

impl<C: Curve> fmt::Debug for Ribbon<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ribbon")
            .field("length", &self.length)
            .field("left", &self.left)
            .field("right", &self.right)
            .field("scale", &self.scale)
            .field("safety", &self.safety)
            .field("sweep", &self.sweep)
            .field("lean", &self.lean)
            .finish_non_exhaustive()
    }
}

/// Which edge of a ribbon.
#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
}

impl<C: Curve> Ribbon<C> {
    /// Ribbon with independent left (`v > 0`, along the normal) and right (`v < 0`) profiles.
    pub fn new(centerline: C, left: WidthProfile, right: WidthProfile) -> Self {
        let mut arclen = ArcLength::new(&centerline, ARC_TABLE);
        let wmax = left.max_sampled(GRID + 1).max(right.max_sampled(GRID + 1));
        let grid = grid_size(&centerline, &arclen, wmax);
        if 2 * grid > ARC_TABLE {
            arclen = ArcLength::new(&centerline, 2 * grid);
        }
        let length = non_negative(arclen.total());
        let mut r = Ribbon {
            centerline,
            left,
            right,
            scale: 1.0,
            safety: DEFAULT_SAFETY,
            sweep: 0.0,
            lean: 0.0,
            arclen,
            length,
            grid,
            kappa: Vec::new(),
            limit_left: Vec::new(),
            limit_right: Vec::new(),
        };
        r.kappa = r.grid_curvature();
        r.refresh_limits();
        r
    }

    /// Ribbon with the same profile on both sides.
    pub fn symmetric(centerline: C, profile: WidthProfile) -> Self {
        Self::new(centerline, profile.clone(), profile)
    }

    /// Like [`new`](Self::new) followed by [`fit_width`](Self::fit_width) at `accuracy`.
    pub fn fitted(centerline: C, left: WidthProfile, right: WidthProfile, accuracy: f64) -> Self {
        let mut r = Self::new(centerline, left, right);
        r.fit_width(accuracy);
        r
    }

    /// Fraction of the local radius of curvature an edge may reach on the concave side
    /// (clamped to `[0.05, 0.99]`; default [`DEFAULT_SAFETY`]).
    pub fn with_safety(mut self, safety: f64) -> Self {
        self.safety = finite_or(safety, DEFAULT_SAFETY).clamp(0.05, 0.99);
        self.refresh_limits();
        self
    }

    /// Sweep the edges backwards near the start by `sweep × local width`, fading out over the
    /// first third of the length. With a zero-width start this turns the base into two
    /// rounded lobes around a notch, as in heart-shaped (cordate) leaves. Clamped to `[0, 2]`.
    pub fn with_sweep(mut self, sweep: f64) -> Self {
        self.sweep = finite_or(sweep, 0.0).clamp(0.0, 2.0);
        self.refresh_limits();
        self
    }

    /// Shift the edges along the direction of travel by `lean × local width` (negative
    /// leans backwards). Wide places move further than narrow ones, so bumps of the width
    /// profile (lobes, pinnae, teeth) lean towards the end like curling acanthus lobes.
    /// Clamped to `[-1.5, 1.5]`.
    pub fn with_lean(mut self, lean: f64) -> Self {
        self.lean = finite_or(lean, 0.0).clamp(-1.5, 1.5);
        self.refresh_limits();
        self
    }

    /// The centreline curve (parameter `t`, not arc length).
    pub fn centerline(&self) -> &C {
        &self.centerline
    }

    /// Requested left profile (before the global scale and curvature clamp).
    pub fn left(&self) -> &WidthProfile {
        &self.left
    }

    /// Requested right profile (before the global scale and curvature clamp).
    pub fn right(&self) -> &WidthProfile {
        &self.right
    }

    /// Centreline arc length.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Global width multiplier (1 unless changed by [`fit_width`](Self::fit_width)).
    pub fn width_scale(&self) -> f64 {
        self.scale
    }

    /// Set the global width multiplier (non-finite or negative values give 0).
    pub fn set_width_scale(&mut self, scale: f64) {
        self.scale = non_negative(scale);
        self.refresh_limits();
    }

    /// Centreline parameter `t` at arc-length fraction `u`.
    pub fn t_at(&self, u: f64) -> f64 {
        self.arclen.t_at_fraction(finite_or(u, 0.0).clamp(0.0, 1.0))
    }

    /// Effective `(left, right)` half-widths at `u`, after scale and curvature clamp.
    pub fn widths_at(&self, u: f64) -> (f64, f64) {
        (self.eff(u, Side::Left), self.eff(u, Side::Right))
    }

    /// Centreline as a fitted Bézier path.
    pub fn centerline_path(&self, accuracy: f64) -> BezPath {
        if self.length <= EPS {
            return BezPath::new();
        }
        let acc = self.sane_accuracy(accuracy);
        let c = |u: f64| self.centerline.point(self.t_at(u));
        cubics_to_path(&merge_short(fit_cubics(&c, acc), 0.5 * acc))
    }

    /// Interior (or edge, `v = ±1`) contour line of constant `v`, following the ribbon's
    /// own parameterisation. Empty for a zero-length ribbon.
    pub fn contour(&self, v: f64, accuracy: f64) -> BezPath {
        if self.length <= EPS {
            return BezPath::new();
        }
        let acc = self.sane_accuracy(accuracy);
        cubics_to_path(&self.fit_iso(finite_or(v, 0.0).clamp(-1.0, 1.0), acc))
    }

    /// `n` contour stripes evenly spaced strictly inside the ribbon
    /// (`v = -1 + 2(i+1)/(n+1)`). Empty if the ribbon is degenerate.
    pub fn stripes(&self, n: usize, accuracy: f64) -> Vec<BezPath> {
        if self.is_degenerate() {
            return Vec::new();
        }
        (0..n)
            .map(|i| self.contour(-1.0 + 2.0 * (i + 1) as f64 / (n + 1) as f64, accuracy))
            .collect()
    }

    /// Validity report computed on the outline at `accuracy`: curvature × width product,
    /// cusps along the edges and self-intersections of the flattened outline.
    pub fn check(&self, accuracy: f64) -> RibbonReport {
        let acc = self.sane_accuracy(accuracy);
        let mut product: f64 = 0.0;
        for (i, &k) in self.kappa.iter().enumerate() {
            let u = i as f64 / self.grid as f64;
            let w = if k > 0.0 {
                self.eff(u, Side::Left)
            } else {
                self.eff(u, Side::Right)
            };
            let lam = self.lean_at(u);
            product = product.max(w * k.abs() * (1.0 + lam * lam));
        }
        let cusp_count = if self.is_degenerate() {
            0
        } else {
            self.edge_cusps(1.0) + self.edge_cusps(-1.0)
        };
        RibbonReport {
            max_width_curvature_product: product,
            cusp_count,
            self_intersections: count_self_intersections(&self.outline(acc), acc),
        }
    }

    /// Scale the widths down until the outline at `accuracy` has no self-intersection, by
    /// bisection on the global width scale. Returns the factor applied (1 if nothing changed).
    ///
    /// Crossings that exist even for a hair-thin ribbon (a centreline that loops over
    /// itself) cannot be removed by narrowing, so the target is the crossing count of such a
    /// thin ribbon rather than zero. "Thin" means a widest half-width of about eight times
    /// the accuracy: any thinner and the two fitted edges, each off by up to `accuracy`,
    /// cross each other and the count measures fitting noise instead of geometry. Only if
    /// a 1000× narrower ribbon has fewer crossings (a sharp centreline corner) is that one
    /// used instead.
    pub fn fit_width(&mut self, accuracy: f64) -> f64 {
        if self.is_degenerate() {
            return 1.0;
        }
        let acc = self.sane_accuracy(accuracy);
        let count = |r: &Self| count_self_intersections(&r.outline(acc), acc);
        let full = count(self);
        if full == 0 {
            return 1.0;
        }
        let original = self.scale;
        let wmax = self
            .left
            .max_sampled(GRID + 1)
            .max(self.right.max_sampled(GRID + 1))
            * original;
        let mut lo = if wmax > 0.0 {
            (8.0 * acc / wmax).clamp(1e-3, 1.0)
        } else {
            1.0
        };
        self.set_width_scale(original * lo);
        let mut baseline = count(self);
        if baseline > 0 && lo > 1e-3 {
            // A sharp centreline corner pinches the concave edge into a tiny swallowtail
            // that only vanishes with the width itself: fall back to a near-zero scale
            // when that does remove the crossings.
            self.set_width_scale(original * 1e-3);
            let tiny = count(self);
            if tiny < baseline {
                baseline = tiny;
                lo = 1e-3;
            }
        }
        if baseline >= full {
            self.set_width_scale(original);
            return 1.0;
        }
        let mut hi = 1.0;
        for _ in 0..14 {
            let mid = 0.5 * (lo + hi);
            self.set_width_scale(original * mid);
            if count(self) <= baseline {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Leave a small gap instead of letting distant parts touch, if that stays valid.
        self.set_width_scale(original * lo * FIT_MARGIN);
        if count(self) <= baseline {
            return lo * FIT_MARGIN;
        }
        self.set_width_scale(original * lo);
        lo
    }

    /// `true` if the ribbon has no area (zero length or zero width everywhere).
    pub fn is_degenerate(&self) -> bool {
        if self.length <= EPS {
            return true;
        }
        let max = (0..=self.grid)
            .map(|i| {
                let u = i as f64 / self.grid as f64;
                self.eff(u, Side::Left) + self.eff(u, Side::Right)
            })
            .fold(0.0, f64::max);
        max <= 1e-12
    }

    // ---- internals -------------------------------------------------------------------

    /// Signed Menger curvature `κ = 2·(b−a)×(c−b) / (|b−a|·|c−b|·|c−a|)` on a uniform
    /// arc-length grid: the curvature of the circle through three neighbouring points.
    fn grid_curvature(&self) -> Vec<f64> {
        let n = self.grid;
        let pts: Vec<Point> = (0..=n)
            .map(|i| self.centerline.point(self.t_at(i as f64 / n as f64)))
            .collect();
        let mut k = vec![0.0; n + 1];
        if self.length <= EPS {
            return k;
        }
        for i in 1..n {
            let (a, b, c) = (pts[i - 1], pts[i], pts[i + 1]);
            let denom = (b - a).hypot() * (c - b).hypot() * (c - a).hypot();
            let v = 2.0 * (b - a).cross(c - b) / denom;
            k[i] = if denom > 1e-30 && v.is_finite() {
                v
            } else {
                0.0
            };
        }
        k[0] = k[1];
        k[n] = k[n - 1];
        k
    }

    /// Recompute the smoothed concave-side limits after a width, scale or safety change.
    ///
    /// Raw limit `L_i = safety/(|κ_i|·(1 + λ_i²))` on the concave side, where `λ` is the
    /// along-shift factor of [`with_lean`](Self::with_lean) and
    /// [`with_sweep`](Self::with_sweep): the edge `C + w·(±n + λT)` has velocity
    /// `T(1 ∓ κw + λw') + n(±w' + λκw)` (plus `λ'w·T ≥ 0` from the fading sweep), which
    /// can only vanish where `|κ|w(1 + λ²) ≥ 1`. A running minimum of radius `2r` followed
    /// by two box blurs of radius `r` gives a smooth table that never exceeds the
    /// raw limit: every blurred value is an average of window minima whose windows all
    /// contain the sample itself. `r` spans about one maximum width of arc length, the
    /// distance over which an offset feels the curvature.
    fn refresh_limits(&mut self) {
        let wmax = self
            .left
            .max_sampled(GRID + 1)
            .max(self.right.max_sampled(GRID + 1))
            * self.scale;
        // Cap stands in for "no limit"; far above any width so the smooth minimum ignores it.
        let cap = 8.0 * wmax.max(EPS);
        let raw = |sign: f64| -> Vec<f64> {
            self.kappa
                .iter()
                .enumerate()
                .map(|(i, &k)| {
                    let k = k * sign;
                    if k > 0.0 {
                        let lam = self.lean_at(i as f64 / self.grid as f64);
                        (self.safety / (k * (1.0 + lam * lam))).min(cap)
                    } else {
                        cap
                    }
                })
                .collect()
        };
        let ds = self.length / self.grid as f64;
        let r = if ds > EPS {
            ((wmax / ds).ceil() as usize).clamp(1, self.grid / 4)
        } else {
            self.grid / 4
        };
        let smooth = |v: Vec<f64>| box_blur(&box_blur(&min_filter(&v, 2 * r), r), r);
        let (left, right) = (smooth(raw(1.0)), smooth(raw(-1.0)));
        self.limit_left = left;
        self.limit_right = right;
    }

    /// `accuracy` made usable: positive and finite (else 0.1), at least `1e-6` and at least
    /// `1e-9` of the ribbon's extent, so that the fits stay bounded for huge geometry.
    fn sane_accuracy(&self, accuracy: f64) -> f64 {
        let wmax = self.left.max_sampled(65).max(self.right.max_sampled(65)) * self.scale;
        let extent = self.length + 2.0 * wmax;
        let acc = sane_accuracy(accuracy);
        if extent.is_finite() {
            acc.max(1e-9 * extent)
        } else {
            acc
        }
    }

    /// Along-shift factor `λ(u)` of the edges: lean minus the start sweep.
    fn lean_at(&self, u: f64) -> f64 {
        self.lean - self.sweep * (1.0 - smoothstep(u / SWEEP_SPAN))
    }

    /// Effective half-width on one side at `u`: smooth minimum of the scaled profile and
    /// the curvature limit.
    fn eff(&self, u: f64, side: Side) -> f64 {
        if self.length <= EPS {
            return 0.0;
        }
        let (profile, limit) = match side {
            Side::Left => (&self.left, &self.limit_left),
            Side::Right => (&self.right, &self.limit_right),
        };
        let w = profile.eval(u) * self.scale;
        if w <= 0.0 {
            return 0.0;
        }
        smooth_min(w, sample_table(limit, u).max(1e-12))
    }

    /// Fit the iso-curve of constant `v`, reparameterised by its own arc length so that the
    /// adaptive fit spends its samples evenly along the edge (stretched convex stretches,
    /// where `κ·w` is large, get proportionally more), then merge segments shorter than half
    /// the accuracy.
    fn fit_iso(&self, v: f64, acc: f64) -> Vec<CubicBez> {
        let iso = FnCurve(|u: f64| self.point(u, v));
        let al = ArcLength::new(&iso, EDGE_TABLE.max(2 * self.grid));
        let cubics = if al.total() > EPS {
            fit_cubics(&|s: f64| iso.point(al.t_at_fraction(s)), acc)
        } else {
            fit_cubics(&|u: f64| iso.point(u), acc)
        };
        merge_short(cubics, 0.5 * acc)
    }

    /// Count sharp reversals along one sampled edge.
    ///
    /// An offset cusp is a point where the edge velocity vanishes and the edge turns back on
    /// itself: a near-180° turn between two very short chords (below a tenth of the
    /// centreline step). Deliberate corners such as lobe tips or the V between pinnae turn
    /// sharply too, but at full speed, so they are not counted.
    fn edge_cusps(&self, v: f64) -> usize {
        let n = EDGE_TABLE.max(2 * self.grid);
        let slow = 0.1 * self.length / n as f64;
        let pts: Vec<Point> = (0..=n)
            .map(|i| self.point(i as f64 / n as f64, v))
            .collect();
        let chords: Vec<Vec2> = pts
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| d.hypot() > 1e-12)
            .collect();
        chords
            .windows(2)
            .filter(|w| {
                let (a, b) = (w[0], w[1]);
                a.dot(b) < -0.9 * a.hypot() * b.hypot() && a.hypot().min(b.hypot()) < slow
            })
            .count()
    }

    /// Two quarter-ellipse cubics closing the ribbon around one end, from edge point `p`
    /// via the tip `m + dir·e` to edge point `q`, where `dir` points out of the ribbon, `m`
    /// is the midpoint of `p` and `q` and `e = |p − q|/2`. Centring the half-ellipse between
    /// the two edges keeps it smooth at the tip when the sides have unequal (or one zero)
    /// widths.
    fn cap(dir: Vec2, p: Point, q: Point) -> [CubicBez; 2] {
        let m = p.midpoint(q);
        let a = p - m;
        let e = a.hypot();
        let b = m + dir * e;
        [
            CubicBez::new(p, p + dir * (e * KAPPA), b + a * KAPPA, b),
            CubicBez::new(b, b - a * KAPPA, q + dir * (e * KAPPA), q),
        ]
    }

    fn build_outline(&self, accuracy: f64) -> BezPath {
        if self.is_degenerate() {
            return BezPath::new();
        }
        let acc = self.sane_accuracy(accuracy);
        let mut left = self.fit_iso(1.0, acc);
        let mut right = self.fit_iso(-1.0, acc);
        if left.is_empty() || right.is_empty() {
            return BezPath::new();
        }
        let (wl0, wr0) = self.widths_at(0.0);
        let (wl1, wr1) = self.widths_at(1.0);
        let (c0, c1) = (self.point(0.0, 0.0), self.point(1.0, 0.0));
        let start_open = wl0 + wr0 > acc;
        let end_open = wl1 + wr1 > acc;
        if !start_open {
            snap_start(&mut left[0], c0);
            snap_start(&mut right[0], c0);
        }
        if !end_open {
            snap_end(left.last_mut().unwrap(), c1);
            snap_end(right.last_mut().unwrap(), c1);
        }
        let mut segs = left;
        if end_open {
            let t1 = self.centerline.tangent(self.t_at(1.0));
            let (p, q) = (segs.last().unwrap().p3, right.last().unwrap().p3);
            segs.extend(Self::cap(t1, p, q));
        }
        segs.extend(
            right
                .iter()
                .rev()
                .map(|s| CubicBez::new(s.p3, s.p2, s.p1, s.p0)),
        );
        if start_open {
            let t0 = self.centerline.tangent(self.t_at(0.0));
            let (p, q) = (segs.last().unwrap().p3, segs[0].p0);
            segs.extend(Self::cap(-t0, p, q));
        }
        let mut path = cubics_to_path(&segs);
        path.close_path();
        path
    }
}

impl<C: Curve> Surface for Ribbon<C> {
    /// `C(u) + n(u)·d` with `d = v·w_left(u)` for `v ≥ 0` (left of the direction of travel,
    /// along the left normal) and `d = v·w_right(u)` for `v < 0`. `u` is the arc-length
    /// fraction; out-of-range or non-finite inputs are clamped. Never returns NaN.
    fn point(&self, u: f64, v: f64) -> Point {
        let u = finite_or(u, 0.0).clamp(0.0, 1.0);
        let v = finite_or(v, 0.0).clamp(-1.0, 1.0);
        let t = self.t_at(u);
        let c = self.centerline.point(t);
        if self.length <= EPS {
            return if c.is_finite() { c } else { Point::ZERO };
        }
        let n = self.centerline.normal(t);
        let tg = Vec2::new(n.y, -n.x);
        let w = if v >= 0.0 {
            self.eff(u, Side::Left)
        } else {
            self.eff(u, Side::Right)
        };
        let along = v.abs() * w * self.lean_at(u);
        let p = c + n * (v * w) + tg * along;
        if p.is_finite() {
            p
        } else if c.is_finite() {
            c
        } else {
            Point::ZERO
        }
    }

    /// Closed outline: left edge forward, end cap, right edge backward, start cap. An end
    /// whose total width exceeds `accuracy` gets a smooth half-ellipse cap built from the
    /// tangent and normal there; a narrower end becomes a clean point placed exactly on the
    /// centreline end. Empty for a degenerate ribbon. Segments shorter than half the
    /// accuracy are merged into their neighbours.
    fn outline(&self, accuracy: f64) -> BezPath {
        self.build_outline(accuracy)
    }

    fn half_width(&self, u: f64) -> f64 {
        let (l, r) = self.widths_at(u);
        0.5 * (l + r)
    }
}

/// Size of the curvature grid for a centreline: at least [`GRID`] steps and about four per
/// maximum half-width `wmax`, doubled while consecutive grid chords still turn by more than
/// 0.25 rad (centreline detail finer than the grid) and the step exceeds `wmax/32`, up to
/// [`MAX_GRID`]. A fixed grid on a long spiral or a finely wiggling centreline would miss
/// the tight turns, and with them the curvature limit.
fn grid_size<C: Curve + ?Sized>(centerline: &C, arclen: &ArcLength, wmax: f64) -> usize {
    let length = arclen.total();
    if !(length > EPS && wmax > EPS && length.is_finite() && wmax.is_finite()) {
        return GRID;
    }
    let mut n = ((4.0 * length / wmax).ceil().min(MAX_GRID as f64) as usize).max(GRID);
    while n < MAX_GRID && length / n as f64 > wmax / 32.0 {
        let pts: Vec<Point> = (0..=n)
            .map(|i| centerline.point(arclen.t_at_fraction(i as f64 / n as f64)))
            .collect();
        let detailed = pts.windows(3).any(|w| {
            let (a, b) = (w[1] - w[0], w[2] - w[1]);
            a.cross(b).atan2(a.dot(b)).abs() > 0.25
        });
        if !detailed {
            break;
        }
        n = (2 * n).min(MAX_GRID);
    }
    n
}

/// Number of proper crossings between non-adjacent segments of `path` flattened with
/// `tolerance` (each subpath treated as a closed polygon if it ends with `ClosePath`).
/// Touching at a shared point is not counted. Sort-and-sweep on `x`, so it stays fast for a
/// few thousand segments.
pub fn count_self_intersections(path: &BezPath, tolerance: f64) -> usize {
    // Relative floor: flattening a huge path at a tiny tolerance would never finish.
    let extent = path.bounding_box().size();
    let extent = extent.width.max(extent.height);
    let tol = if extent.is_finite() {
        sane_accuracy(tolerance).max(1e-9 * extent)
    } else {
        sane_accuracy(tolerance)
    };
    // Collect polylines; consecutive duplicate points are dropped.
    let mut polys: Vec<(Vec<Point>, bool)> = Vec::new();
    kurbo::flatten(path.iter(), tol, |el| match el {
        PathEl::MoveTo(p) => polys.push((vec![p], false)),
        PathEl::LineTo(p) => {
            if let Some((pts, _)) = polys.last_mut()
                && pts.last().is_none_or(|l| (*l - p).hypot() > 1e-12)
            {
                pts.push(p);
            }
        }
        PathEl::ClosePath => {
            if let Some((_, closed)) = polys.last_mut() {
                *closed = true;
            }
        }
        _ => {}
    });
    struct Seg {
        a: Point,
        b: Point,
        poly: usize,
        idx: usize,
        len: usize,
        closed: bool,
    }
    let mut segs = Vec::new();
    for (pi, (pts, closed)) in polys.iter_mut().enumerate() {
        if *closed && pts.len() > 2 && (pts[0] - *pts.last().unwrap()).hypot() <= 1e-12 {
            pts.pop();
        }
        let n = pts.len();
        if n < 2 {
            continue;
        }
        let count = if *closed && n > 2 { n } else { n - 1 };
        for i in 0..count {
            segs.push(Seg {
                a: pts[i],
                b: pts[(i + 1) % n],
                poly: pi,
                idx: i,
                len: count,
                closed: *closed && n > 2,
            });
        }
    }
    let adjacent = |s: &Seg, t: &Seg| {
        if s.poly != t.poly {
            return false;
        }
        let d = s.idx.abs_diff(t.idx);
        d <= 1 || (s.closed && d == s.len - 1)
    };
    let minx = |s: &Seg| s.a.x.min(s.b.x);
    segs.sort_by(|s, t| minx(s).total_cmp(&minx(t)));
    let mut hits = 0;
    for i in 0..segs.len() {
        let s = &segs[i];
        let sx1 = s.a.x.max(s.b.x);
        let (sy0, sy1) = (s.a.y.min(s.b.y), s.a.y.max(s.b.y));
        for t in &segs[i + 1..] {
            if minx(t) > sx1 {
                break;
            }
            if t.a.y.max(t.b.y) < sy0 || t.a.y.min(t.b.y) > sy1 || adjacent(s, t) {
                continue;
            }
            if segments_cross(s.a, s.b, t.a, t.b) {
                hits += 1;
            }
        }
    }
    hits
}

/// Proper crossing test with strict orientation signs (shared endpoints do not count).
fn segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool {
    let o = |p: Point, q: Point, r: Point| (q - p).cross(r - p);
    let (d1, d2) = (o(a, b, c), o(a, b, d));
    let (d3, d4) = (o(c, d, a), o(c, d, b));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Polynomial smooth minimum `min(a, b) − k·h²/4` with `h = max(k − |a − b|, 0)/k` and a
/// blend width `k` of half the larger value: C1, never above the true minimum, never
/// negative for non-negative inputs, and exactly the minimum once the two differ by `k`
/// (so an unconstrained width is reproduced exactly).
fn smooth_min(a: f64, b: f64) -> f64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let k = 0.5 * hi;
    if k <= 0.0 {
        return lo.max(0.0);
    }
    let h = ((k - (hi - lo)) / k).max(0.0);
    (lo - k * h * h * 0.25).max(0.0)
}

/// Running minimum over a window of radius `r` (truncated at the ends).
fn min_filter(v: &[f64], r: usize) -> Vec<f64> {
    let n = v.len();
    (0..n)
        .map(|i| {
            v[i.saturating_sub(r)..(i + r + 1).min(n)]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// Box blur of radius `r` (truncated, renormalised windows at the ends).
fn box_blur(v: &[f64], r: usize) -> Vec<f64> {
    let n = v.len();
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0.0);
    for x in v {
        prefix.push(prefix.last().unwrap() + x);
    }
    (0..n)
        .map(|i| {
            let (a, b) = (i.saturating_sub(r), (i + r + 1).min(n));
            (prefix[b] - prefix[a]) / (b - a) as f64
        })
        .collect()
}

/// Catmull-Rom interpolation of a table sampled uniformly on `[0, 1]`.
fn sample_table(table: &[f64], u: f64) -> f64 {
    let n = table.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return table[0];
    }
    let x = u.clamp(0.0, 1.0) * (n - 1) as f64;
    let i = (x.floor() as usize).min(n - 2);
    let f = x - i as f64;
    let at = |k: isize| table[k.clamp(0, n as isize - 1) as usize];
    let (p0, p1, p2, p3) = (
        at(i as isize - 1),
        at(i as isize),
        at(i as isize + 1),
        at(i as isize + 2),
    );
    let v = p1
        + 0.5
            * f
            * (p2 - p0
                + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + f * (3.0 * (p1 - p2) + p3 - p0)));
    // Keep within the neighbouring values so the interpolant never undershoots the limits.
    v.clamp(p1.min(p2), p1.max(p2))
}

fn sane_accuracy(accuracy: f64) -> f64 {
    if accuracy.is_finite() && accuracy > 0.0 {
        accuracy.max(1e-6)
    } else {
        0.1
    }
}

/// Adaptive cubic Hermite fit of `f` on `[0, 1]` (best when `f` is close to arc-length
/// parameterised). Each piece takes its end points and one-sided end derivatives
/// (`P0, P0 + D0·Δ/3, P1 − D1·Δ/3, P1`) and is split in half while the distance to the curve
/// at seven interior samples exceeds `acc` (up to depth 16), or while it contains a sharp
/// turn and is longer than `acc` (up to depth 24, which only the few pieces around a corner
/// reach). One-sided derivatives keep genuine corners (lobe tips, pointed ends) sharp
/// instead of rounding them. Bounded depth, never panics.
fn fit_cubics(f: &impl Fn(f64) -> Point, acc: f64) -> Vec<CubicBez> {
    const START: usize = 8;
    const ERR_DEPTH: u32 = 16;
    const MAX_DEPTH: u32 = 24;
    fn piece(f: &impl Fn(f64) -> Point, s0: f64, s1: f64) -> CubicBez {
        let span = s1 - s0;
        let h = span * 1e-4;
        let (p0, p3) = (f(s0), f(s1));
        // Control arms `D·Δ/3`, capped at the chord length: across a jump or a cusp the
        // one-sided derivative explodes, and an uncapped arm would fling the handle away.
        let chord = (p3 - p0).hypot();
        let arm = |d: Vec2| {
            let a = d * (span / 3.0);
            let len = a.hypot();
            if len > chord { a * (chord / len) } else { a }
        };
        let a0 = arm((f(s0 + h) - p0) / h);
        let a1 = arm((p3 - f(s1 - h)) / h);
        let c = CubicBez::new(p0, p0 + a0, p3 - a1, p3);
        if c.p1.is_finite() && c.p2.is_finite() {
            c
        } else {
            CubicBez::new(p0, p0.lerp(p3, 1.0 / 3.0), p0.lerp(p3, 2.0 / 3.0), p3)
        }
    }
    fn rec(
        f: &impl Fn(f64) -> Point,
        s0: f64,
        s1: f64,
        acc: f64,
        depth: u32,
        out: &mut Vec<CubicBez>,
    ) {
        let c = piece(f, s0, s1);
        let pts: [Point; 9] = std::array::from_fn(|k| f(s0 + (s1 - s0) * k as f64 / 8.0));
        let err = (1..8)
            .map(|k| (c.eval(k as f64 / 8.0) - pts[k]).hypot())
            .fold(0.0, f64::max);
        // A corner or cusp lying between two samples is invisible to `err`: the cubic can
        // round it off by up to half a sample spacing. Keep splitting where the sampled
        // polyline turns by more than 60° until the piece is no longer than `acc`.
        let len: f64 = pts.windows(2).map(|w| (w[1] - w[0]).hypot()).sum();
        let sharp = pts.windows(3).any(|w| {
            let (a, b) = (w[1] - w[0], w[2] - w[1]);
            a.dot(b) < 0.5 * a.hypot() * b.hypot()
        });
        let split = (err > acc && depth < ERR_DEPTH) || (sharp && len > acc && depth < MAX_DEPTH);
        if !split || !err.is_finite() {
            out.push(c);
        } else {
            let mid = 0.5 * (s0 + s1);
            rec(f, s0, mid, acc, depth + 1, out);
            rec(f, mid, s1, acc, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    for i in 0..START {
        let (s0, s1) = (i as f64 / START as f64, (i + 1) as f64 / START as f64);
        rec(f, s0, s1, acc, 0, &mut out);
    }
    out.retain(|c| c.p0.is_finite() && c.p1.is_finite() && c.p2.is_finite() && c.p3.is_finite());
    out
}

fn poly_len(c: &CubicBez) -> f64 {
    (c.p1 - c.p0).hypot() + (c.p2 - c.p1).hypot() + (c.p3 - c.p2).hypot()
}

/// Merge segments whose control polygon is shorter than `min_len` into a neighbour, moving
/// the neighbour's end point (and adjacent control point, to keep its tangent) onto the
/// dropped segment's far end. Snapping can shorten the neighbour itself (where the edge
/// turns back, as in a sinus), so a neighbour that drops below `min_len` is merged in turn.
/// Keeps at least one segment.
fn merge_short(segs: Vec<CubicBez>, min_len: f64) -> Vec<CubicBez> {
    let mut out: Vec<CubicBez> = Vec::with_capacity(segs.len());
    for s in segs {
        match out.last_mut() {
            Some(last) if poly_len(&s) < min_len => snap_end(last, s.p3),
            _ => out.push(s),
        }
        while out.len() >= 2 && poly_len(&out[out.len() - 1]) < min_len {
            let dropped = out.pop().unwrap();
            snap_end(out.last_mut().unwrap(), dropped.p3);
        }
    }
    if out.len() >= 2 && poly_len(&out[0]) < min_len {
        let first = out.remove(0);
        snap_start(&mut out[0], first.p0);
    }
    out
}

/// Move a cubic's start point to `p`, translating its first control point along.
fn snap_start(c: &mut CubicBez, p: Point) {
    let d = p - c.p0;
    c.p0 = p;
    c.p1 += d;
}

/// Move a cubic's end point to `p`, translating its last control point along.
fn snap_end(c: &mut CubicBez, p: Point) {
    let d = p - c.p3;
    c.p3 = p;
    c.p2 += d;
}

fn cubics_to_path(segs: &[CubicBez]) -> BezPath {
    let mut path = BezPath::new();
    if let Some(first) = segs.first() {
        path.move_to(first.p0);
        for s in segs {
            path.curve_to(s.p1, s.p2, s.p3);
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Composite;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use kurbo::{Line, Shape};
    use std::f64::consts::PI;

    fn straight() -> Line {
        Line::new((0.0, 0.0), (200.0, 0.0))
    }

    /// Tight S-curve: two half circles of radius 20.
    fn s_curve() -> FnCurve<impl Fn(f64) -> Point> {
        FnCurve(|t: f64| {
            let r = 20.0;
            if t < 0.5 {
                let a = PI * (1.0 - 2.0 * t);
                Point::new(r + r * a.cos(), r * a.sin())
            } else {
                let a = PI * (2.0 * t - 1.0);
                Point::new(3.0 * r - r * a.cos(), -r * a.sin())
            }
        })
    }

    /// Serpentine with 100-long parallel legs 40 apart joined by radius-20 half turns,
    /// parameterised by arc length.
    fn serpentine() -> FnCurve<impl Fn(f64) -> Point> {
        FnCurve(|t: f64| {
            let arc = PI * 20.0;
            let s = t.clamp(0.0, 1.0) * (300.0 + 2.0 * arc);
            if s < 100.0 {
                Point::new(s, 0.0)
            } else if s < 100.0 + arc {
                let a = -PI / 2.0 + (s - 100.0) / 20.0;
                Point::new(100.0 + 20.0 * a.cos(), 20.0 + 20.0 * a.sin())
            } else if s < 200.0 + arc {
                Point::new(100.0 - (s - 100.0 - arc), 40.0)
            } else if s < 200.0 + 2.0 * arc {
                let a = -PI / 2.0 - (s - 200.0 - arc) / 20.0;
                Point::new(20.0 * a.cos(), 60.0 + 20.0 * a.sin())
            } else {
                Point::new(s - 200.0 - 2.0 * arc, 80.0)
            }
        })
    }

    fn assert_well_formed(p: &BezPath) {
        assert!(path_is_finite(p));
        assert_eq!(degenerate_segments(p, 1e-9), 0);
        assert!(matches!(p.elements().last(), Some(PathEl::ClosePath)));
        assert!(matches!(p.elements().first(), Some(PathEl::MoveTo(_))));
    }

    fn end_point(p: &BezPath, index: usize) -> Point {
        match p.elements()[index] {
            PathEl::MoveTo(q) | PathEl::LineTo(q) | PathEl::CurveTo(_, _, q) => q,
            PathEl::QuadTo(_, q) => q,
            PathEl::ClosePath => panic!("close"),
        }
    }

    #[test]
    fn profiles_are_finite_and_shaped() {
        let t = WidthProfile::tapered(10.0, 0.2, 0.3, 1.0);
        assert_eq!(t.eval(0.0), 0.0);
        assert_eq!(t.eval(1.0), 0.0);
        assert!((t.eval(0.5) - 10.0).abs() < 1e-12);
        let open = WidthProfile::tapered(10.0, 0.0, 0.0, 1.0);
        assert_eq!(open.eval(0.0), 10.0);
        let td = WidthProfile::teardrop(5.0, 0.25);
        assert!((td.eval(0.25) - 5.0).abs() < 1e-12);
        assert!(td.eval(0.1) > td.eval(0.9));
        let lens = WidthProfile::lens(3.0, 1.0);
        assert!((lens.eval(0.3) - lens.eval(0.7)).abs() < 1e-12);
        let bad = WidthProfile::custom(|u| if u > 0.5 { f64::NAN } else { -1.0 });
        assert_eq!(bad.eval(0.2), 0.0);
        assert_eq!(bad.eval(0.8), 0.0);
        assert_eq!(bad.eval(f64::NAN), 0.0);
        let n1 = WidthProfile::constant(4.0).with_noise(7, 0.4, 6.0);
        let n2 = WidthProfile::constant(4.0).with_noise(7, 0.4, 6.0);
        let mut differs = false;
        for i in 0..=50 {
            let u = i as f64 / 50.0;
            assert_eq!(n1.eval(u), n2.eval(u));
            assert!(n1.eval(u) >= 4.0 * 0.6 - 1e-9 && n1.eval(u) <= 4.0 * 1.4 + 1e-9);
            differs |= (n1.eval(u) - 4.0).abs() > 1e-3;
        }
        assert!(differs);
        assert_eq!(WidthProfile::constant(2.0).scale(f64::NAN).eval(0.5), 0.0);
        assert_eq!(ease(0.0, 0.5), 0.0);
        assert_eq!(ease(1.0, 3.0), 1.0);
    }

    #[test]
    fn straight_ribbon_outline_with_round_caps() {
        let r = Ribbon::symmetric(straight(), WidthProfile::constant(10.0));
        let o = r.outline(0.05);
        assert_well_formed(&o);
        // area of a 200×20 stadium with radius-10 caps
        let expected = 200.0 * 20.0 + PI * 100.0;
        assert!((o.area().abs() - expected).abs() < 5.0, "{}", o.area());
        assert!(o.winding(Point::new(-9.0, 0.0)) != 0);
        assert!(o.winding(Point::new(209.0, 0.0)) != 0);
        assert_eq!(o.winding(Point::new(100.0, 11.0)), 0);
        // v > 0 is the left side: along +y for a +x centreline (left normal = (-ty, tx))
        assert!((r.point(0.5, 1.0) - Point::new(100.0, 10.0)).hypot() < 1e-9);
        assert!((r.point(0.5, -0.5) - Point::new(100.0, -5.0)).hypot() < 1e-9);
        assert!((r.half_width(0.3) - 10.0).abs() < 1e-6);
    }

    /// Largest turning angle (degrees) between the end tangent of each segment and the start
    /// tangent of the next, including the closing join.
    fn max_join_angle(p: &BezPath) -> f64 {
        let segs: Vec<CubicBez> = p.segments().map(|s| s.to_cubic()).collect();
        let dir = |c: &CubicBez, t: f64| {
            let d = kurbo::ParamCurveDeriv::deriv(c).eval(t).to_vec2();
            d / d.hypot()
        };
        (0..segs.len())
            .map(|i| {
                let a = dir(&segs[i], 1.0 - 1e-6);
                let b = dir(&segs[(i + 1) % segs.len()], 1e-6);
                a.cross(b).atan2(a.dot(b)).abs().to_degrees()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn one_sided_ribbon_has_smooth_caps() {
        // Only the left side has width: the caps must still be smooth half-ellipses.
        for right in [WidthProfile::zero(), WidthProfile::constant(0.5)] {
            let r = Ribbon::new(straight(), WidthProfile::constant(10.0), right);
            let o = r.outline(0.05);
            assert_well_formed(&o);
            assert!(max_join_angle(&o) < 1.0, "{}", max_join_angle(&o));
            assert_eq!(
                o.winding(Point::new(100.0, 5.0)),
                o.winding(Point::new(-4.0, 5.0))
            );
        }
    }

    #[test]
    fn pointed_tips_end_exactly_on_centreline() {
        let r = Ribbon::new(
            straight(),
            WidthProfile::tapered(12.0, 0.3, 0.3, 1.0),
            WidthProfile::tapered(6.0, 0.3, 0.3, 0.5),
        );
        let o = r.outline(0.05);
        assert_well_formed(&o);
        assert_eq!(end_point(&o, 0), Point::new(0.0, 0.0));
        let tip = Point::new(200.0, 0.0);
        let hits = (0..o.elements().len() - 1)
            .filter(|&i| end_point(&o, i) == tip)
            .count();
        assert_eq!(hits, 1, "exactly one vertex on the far tip");
        // closing point coincides with the start
        assert_eq!(end_point(&o, o.elements().len() - 2), Point::new(0.0, 0.0));
    }

    #[test]
    fn outline_has_no_short_segments() {
        let r = Ribbon::symmetric(s_curve(), WidthProfile::lens(8.0, 0.8));
        let acc = 0.1;
        let o = r.outline(acc);
        assert_well_formed(&o);
        for seg in o.segments() {
            let c = seg.to_cubic();
            assert!(poly_len(&c) >= 0.5 * acc - 1e-9, "{c:?}");
        }
    }

    #[test]
    fn curvature_clamp_prevents_cusps() {
        // Width 35 on a radius-20 S-curve: the raw offset would form swallowtails.
        let r = Ribbon::symmetric(s_curve(), WidthProfile::constant(35.0));
        let rep = r.check(0.1);
        assert!(rep.max_width_curvature_product < 1.0, "{rep:?}");
        assert_eq!(rep.cusp_count, 0);
        for i in 0..=100 {
            let (l, rr) = r.widths_at(i as f64 / 100.0);
            assert!(l.is_finite() && rr.is_finite() && l >= 0.0 && rr >= 0.0);
        }
    }

    #[test]
    fn fit_width_removes_self_intersections_on_tight_s_curve() {
        // Half-width 30 on radius-20 turns with legs 40 apart: the legs overlap.
        let mut r = Ribbon::symmetric(serpentine(), WidthProfile::tapered(30.0, 0.1, 0.1, 1.0));
        let before = r.check(0.1);
        assert!(before.self_intersections > 0, "{before:?}");
        let k = r.fit_width(0.1);
        assert!(k > 0.3 && k < 1.0, "{k}");
        let after = r.check(0.1);
        assert!(after.is_valid(), "{after:?}");
        assert!(after.max_width_curvature_product < 1.0);
        assert_well_formed(&r.outline(0.1));
        let fitted = Ribbon::fitted(
            serpentine(),
            WidthProfile::tapered(30.0, 0.1, 0.1, 1.0),
            WidthProfile::tapered(30.0, 0.1, 0.1, 1.0),
            0.1,
        );
        assert_eq!(fitted.width_scale(), r.width_scale());
    }

    #[test]
    fn stripes_stay_inside_outline() {
        let mut r = Ribbon::new(
            s_curve(),
            WidthProfile::tapered(14.0, 0.2, 0.25, 0.7),
            WidthProfile::tapered(9.0, 0.2, 0.25, 0.7),
        );
        r.fit_width(0.05);
        let o = r.outline(0.05);
        let stripes = r.stripes(5, 0.05);
        assert_eq!(stripes.len(), 5);
        for s in &stripes {
            assert!(path_is_finite(s));
            let c = Composite::from_bezpath(s).unwrap();
            for i in 1..40 {
                let u = 0.03 + 0.94 * i as f64 / 40.0;
                assert_ne!(o.winding(c.point(u)), 0, "u={u}");
            }
        }
        assert!(path_is_finite(&r.centerline_path(0.05)));
    }

    #[test]
    fn deterministic_output() {
        let make = || {
            let p = WidthProfile::lens(9.0, 0.7).with_noise(3, 0.3, 5.0);
            Ribbon::fitted(s_curve(), p.clone(), p, 0.1).outline(0.1)
        };
        assert_eq!(make(), make());
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let dot = FnCurve(|_t: f64| Point::new(5.0, 5.0));
        let mut r = Ribbon::symmetric(dot, WidthProfile::constant(10.0));
        assert!(r.outline(0.1).elements().is_empty());
        assert!(r.stripes(3, 0.1).is_empty());
        assert!(r.point(0.5, 1.0).is_finite());
        assert_eq!(r.fit_width(0.1), 1.0);
        assert_eq!(r.check(0.1).self_intersections, 0);

        let zero = Ribbon::symmetric(straight(), WidthProfile::zero());
        assert!(zero.outline(0.1).elements().is_empty());
        assert!(zero.is_degenerate());

        // Cubic with coincident control points at both ends: near-zero end tangents.
        let c = CubicBez::new((0.0, 0.0), (0.0, 0.0), (100.0, 50.0), (100.0, 50.0));
        let r = Ribbon::symmetric(c, WidthProfile::constant(5.0));
        let o = r.outline(f64::NAN);
        assert_well_formed(&o);
        for (u, v) in [(f64::NAN, 0.0), (-3.0, 7.0), (2.0, f64::INFINITY)] {
            assert!(r.point(u, v).is_finite());
        }

        // Centreline with a right-angle corner (zero-tension polyline): the concave side
        // pinches to the corner, the convex side sweeps around it.
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(50.0, 0.0),
            Point::new(50.0, 50.0),
        ];
        let corner = Composite::through_points(&pts, 0.0).unwrap();
        let mut r = Ribbon::symmetric(corner, WidthProfile::tapered(8.0, 0.1, 0.1, 1.0));
        r.fit_width(0.1);
        assert_well_formed(&r.outline(0.1));
        let rep = r.check(0.1);
        assert!(rep.is_valid(), "{rep:?}");

        // A hairpin corner (170°) is a centreline cusp: it must stay finite and well formed.
        let pts = [
            Point::new(0.0, 0.0),
            Point::new(50.0, 0.0),
            Point::new(0.0, 10.0),
        ];
        let hairpin = Composite::through_points(&pts, 0.0).unwrap();
        let mut r = Ribbon::symmetric(hairpin, WidthProfile::tapered(8.0, 0.1, 0.1, 1.0));
        r.fit_width(0.1);
        assert_well_formed(&r.outline(0.1));
    }

    #[test]
    fn lean_and_sweep_shift_edges_along_the_centreline() {
        let bumps = WidthProfile::lens(10.0, 1.0).map(|u, w| w * (0.6 + 0.4 * (20.0 * u).sin()));
        let r = Ribbon::symmetric(straight(), bumps)
            .with_lean(0.5)
            .with_sweep(0.6);
        let (w, _) = r.widths_at(0.5);
        let p = r.point(0.5, 1.0);
        assert!((p.x - (100.0 + 0.5 * w)).abs() < 1e-9 && (p.y - w).abs() < 1e-9);
        assert!(r.point(0.02, 1.0).x < r.point(0.0, 0.0).x + 0.02 * 200.0);
        let o = r.outline(0.05);
        assert_well_formed(&o);
        assert!(r.check(0.05).is_valid());
        assert!(r.point(0.3, f64::NAN).is_finite());
    }

    #[test]
    fn curvature_clamp_accounts_for_lean() {
        // Shifting the edges along the tangent by λ·w moves the offset cusp to
        // |κ|·w·(1 + λ²) = 1, so the clamp must tighten accordingly.
        let r = Ribbon::symmetric(s_curve(), WidthProfile::constant(35.0)).with_lean(1.5);
        let factor = 1.0 + 1.5 * 1.5;
        for i in (5..=45).chain(55..=95) {
            let u = i as f64 / 100.0;
            let (l, rr) = r.widths_at(u);
            // radius-20 half circles: right turn first (concave right), then left turn
            let concave = if u < 0.5 { rr } else { l };
            assert!(concave * factor / 20.0 < 1.0, "u={u} w={l},{rr}");
        }
        let rep = r.check(0.1);
        assert!(rep.max_width_curvature_product < 1.0, "{rep:?}");
        assert_eq!(rep.cusp_count, 0);
    }

    #[test]
    fn curvature_clamp_resolves_long_spirals() {
        // Eight turns, radius 5 → 200, about 5000 long: a fixed 512-step grid misses the
        // tight inner turns, so the clamp failed there and the outline crossed itself.
        let spiral = FnCurve(|t: f64| {
            let a = t * 2.0 * PI * 8.0;
            let r = 5.0 + 195.0 * t;
            Point::new(r * a.cos(), r * a.sin())
        });
        let r = Ribbon::symmetric(&spiral, WidthProfile::constant(9.0));
        let rep = r.check(0.1);
        assert!(rep.is_valid(), "{rep:?}");
        let mut product: f64 = 0.0;
        for i in 0..=4000 {
            let u = i as f64 / 4000.0;
            let k = spiral.curvature(r.t_at(u));
            let (l, rr) = r.widths_at(u);
            product = product.max(if k > 0.0 { l } else { rr } * k.abs());
        }
        assert!(product < 1.0, "{product}");
    }

    #[test]
    fn huge_geometry_stays_bounded() {
        // Widths far beyond the requested accuracy used to recurse the fit to its depth
        // limit everywhere and to make the crossing count's flattening overflow.
        for w in [1e12, 1e150] {
            let r = Ribbon::symmetric(straight(), WidthProfile::lens(w, 0.7));
            let o = r.outline(0.1);
            assert_well_formed(&o);
            assert!(o.segments().count() < 50_000, "{}", o.segments().count());
            assert_eq!(r.check(0.1).self_intersections, 0);
        }
    }

    #[test]
    fn self_intersection_counter() {
        let mut bow = BezPath::new();
        bow.move_to((0.0, 0.0));
        bow.line_to((10.0, 10.0));
        bow.line_to((10.0, 0.0));
        bow.line_to((0.0, 10.0));
        bow.close_path();
        assert_eq!(count_self_intersections(&bow, 0.1), 1);
        let sq = kurbo::Rect::new(0.0, 0.0, 10.0, 10.0).to_path(0.1);
        assert_eq!(count_self_intersections(&sq, 0.1), 0);
    }
}
