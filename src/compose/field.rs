//! Composition primitives: vector fields, streamline guide curves, density masks,
//! focal points, keep-out regions and deterministic weighted placement.
//!
//! Everything here is pure and deterministic: randomness only enters through a
//! [`SeedTree`] passed by the caller, and every query (`Field::at`, `Mask::at`) is total —
//! it returns a finite value for any input, including NaN points and degenerate parameters.

use crate::geom::validate::sanitize;
use crate::geom::{Composite, Curve, CurveExt, EPS, lerp, smoothstep, unlerp};
use crate::rng::{ChaCha8Rng, Draw, Rng, SeedTree};
use kurbo::{BezPath, CubicBez, ParamCurve, PathEl, Point, Rect, Shape, Vec2};
use rand::SeedableRng;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------------------

/// Replace a non-finite vector by zero.
#[inline]
fn fin(v: Vec2) -> Vec2 {
    if v.x.is_finite() && v.y.is_finite() {
        v
    } else {
        Vec2::ZERO
    }
}

/// Clamp a mask value to `[0,1]`, mapping NaN to 0.
#[inline]
fn unit(x: f64) -> f64 {
    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
}

/// A finite, positive number or `None`.
#[inline]
fn positive(x: f64) -> Option<f64> {
    (x.is_finite() && x > 0.0).then_some(x)
}

/// Normalise a rectangle (`x0 <= x1`, `y0 <= y1`); `None` if its corners or extent are
/// non-finite or it has zero area.
fn norm_rect(r: Rect) -> Option<Rect> {
    let (x0, x1) = (r.x0.min(r.x1), r.x0.max(r.x1));
    let (y0, y1) = (r.y0.min(r.y1), r.y0.max(r.y1));
    let (w, h) = (x1 - x0, y1 - y0);
    // The extent must be finite too: random draws and grid sizes are computed from it.
    let ok = w.is_finite() && h.is_finite() && w > EPS && h > EPS;
    ok.then(|| Rect::new(x0, y0, x1, y1))
}

/// Smooth radial falloff: 1 for `d <= radius·(1−softness)`, 0 for `d >= radius`, a
/// smoothstep in between. `softness = 0` is a hard edge, `1` fades from the centre.
fn falloff(d: f64, radius: f64, softness: f64) -> f64 {
    let Some(radius) = positive(radius) else {
        return 0.0;
    };
    if !d.is_finite() {
        return 0.0;
    }
    let soft = unit(softness);
    let inner = radius * (1.0 - soft);
    if d <= inner {
        1.0
    } else if d >= radius {
        0.0
    } else {
        1.0 - smoothstep((d - inner) / (radius - inner))
    }
}

/// Distance from `p` to segment `ab`.
fn seg_dist(p: Point, a: Point, b: Point) -> f64 {
    let ab = b - a;
    let l2 = ab.hypot2();
    let t = if l2 > 0.0 {
        ((p - a).dot(ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).hypot()
}

/// Distance from `p` to an open polyline (single point allowed).
fn polyline_dist(p: Point, pts: &[Point]) -> f64 {
    match pts {
        [] => f64::INFINITY,
        [a] => (p - *a).hypot(),
        _ => pts
            .windows(2)
            .map(|w| seg_dist(p, w[0], w[1]))
            .fold(f64::INFINITY, f64::min),
    }
}

/// 2-D Perlin gradient noise with an analytic gradient.
///
/// The `noise` crate's `Perlin` clamps its output to `[-1,1]`, which leaves flat plateaus
/// where the gradient (and hence a curl field) is exactly zero; this version never clamps.
/// Gradients are 16 unit vectors at odd multiples of π/16 (no axis alignment), the fade is
/// the quintic `6t⁵ − 15t⁴ + 10t³` (C², so streamlines of its curl are smooth), and the
/// lattice is periodic with period 256, so coordinates are wrapped to stay exact.
#[derive(Debug, Clone)]
struct Gradient2 {
    perm: [u8; 256],
}

impl Gradient2 {
    /// Max |value| of 2-D Perlin with unit gradients is √½; rescale to about `[-1,1]`.
    const NORM: f64 = std::f64::consts::SQRT_2;

    fn new(seed: u32) -> Gradient2 {
        let mut rng = ChaCha8Rng::seed_from_u64(u64::from(seed) ^ 0x5EED_F0C0_FFEE);
        let mut perm = [0u8; 256];
        for (i, v) in perm.iter_mut().enumerate() {
            *v = i as u8;
        }
        for i in (1..256).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        Gradient2 { perm }
    }

    fn grad(&self, ix: usize, iy: usize) -> Vec2 {
        let h = self.perm[(self.perm[ix & 255] as usize ^ (iy & 255)) & 255];
        let a = (f64::from(h & 15) + 0.5) * std::f64::consts::TAU / 16.0;
        Vec2::new(a.cos(), a.sin())
    }

    /// Noise value (about `[-1,1]`) and its gradient at `(x, y)` in lattice units.
    fn eval(&self, x: f64, y: f64) -> (f64, Vec2) {
        if !(x.is_finite() && y.is_finite()) {
            return (0.0, Vec2::ZERO);
        }
        let (x, y) = (x.rem_euclid(256.0), y.rem_euclid(256.0));
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = (x - fx, y - fy);
        let (ix, iy) = (fx as usize, fy as usize);
        let g00 = self.grad(ix, iy);
        let g10 = self.grad(ix + 1, iy);
        let g01 = self.grad(ix, iy + 1);
        let g11 = self.grad(ix + 1, iy + 1);
        let n00 = g00.dot(Vec2::new(tx, ty));
        let n10 = g10.dot(Vec2::new(tx - 1.0, ty));
        let n01 = g01.dot(Vec2::new(tx, ty - 1.0));
        let n11 = g11.dot(Vec2::new(tx - 1.0, ty - 1.0));
        let fade = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let dfade = |t: f64| 30.0 * t * t * (t * (t - 2.0) + 1.0);
        let (u, v) = (fade(tx), fade(ty));
        let (du, dv) = (dfade(tx), dfade(ty));
        // n = n00 + u(n10−n00) + v(n01−n00) + uv(n00−n10−n01+n11); each corner term n_ij
        // also depends on (x, y) through its own gradient g_ij.
        let k = n00 - n10 - n01 + n11;
        let n = n00 + u * (n10 - n00) + v * (n01 - n00) + u * v * k;
        let g = g00 + (g10 - g00) * u + (g01 - g00) * v + (g00 - g10 - g01 + g11) * (u * v);
        let d = Vec2::new(du * (n10 - n00 + v * k), dv * (n01 - n00 + u * k));
        let (n, grad) = (n * Self::NORM, (g + d) * Self::NORM);
        if n.is_finite() && grad.x.is_finite() && grad.y.is_finite() {
            (n, grad)
        } else {
            (0.0, Vec2::ZERO)
        }
    }
}

// ---------------------------------------------------------------------------------------
// Vector fields
// ---------------------------------------------------------------------------------------

/// A 2-D vector field. [`at`](Field::at) must return a finite vector for every input.
pub trait Field {
    /// The field vector at `p`.
    fn at(&self, p: Point) -> Vec2;
}

impl<F: Field + ?Sized> Field for &F {
    fn at(&self, p: Point) -> Vec2 {
        (**self).at(p)
    }
}

impl<F: Field + ?Sized> Field for Box<F> {
    fn at(&self, p: Point) -> Vec2 {
        (**self).at(p)
    }
}

/// A field defined by a closure (output is sanitised to finite values).
pub struct FnField<F: Fn(Point) -> Vec2>(pub F);

impl<F: Fn(Point) -> Vec2> Field for FnField<F> {
    fn at(&self, p: Point) -> Vec2 {
        fin((self.0)(p))
    }
}

/// A constant vector everywhere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Directional(pub Vec2);

impl Field for Directional {
    fn at(&self, _p: Point) -> Vec2 {
        fin(self.0)
    }
}

/// Radial profile shared by [`Vortex`] and [`Attractor`]: returns `g(d)` such that the
/// field magnitude `d·g(d) = (d/R)·exp((1 − d²/R²)/2)` rises linearly from 0 at the
/// centre (no singularity), peaks at exactly 1 for `d = R` and decays like a Gaussian.
#[inline]
fn ring_profile(d: f64, radius: f64) -> f64 {
    let q = d / radius;
    (0.5 * (1.0 - q * q)).exp() / radius
}

/// A swirl around `center` (counter-clockwise for positive `strength` in a y-up frame,
/// clockwise on screen with y down).
///
/// The speed is `strength` at distance `radius`, zero at the centre and fades smoothly
/// beyond. `swirl = 0` gives closed circular orbits; `swirl > 0` adds an inward radial
/// component so streamlines become logarithmic-like spirals winding into the centre,
/// `swirl < 0` spirals outward. `swirl` is the ratio radial/tangential speed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vortex {
    /// Centre of rotation.
    pub center: Point,
    /// Peak speed (reached at `radius`); negative reverses the rotation.
    pub strength: f64,
    /// Falloff radius.
    pub radius: f64,
    /// Radial/tangential ratio: 0 = pure rotation, > 0 inward, < 0 outward.
    pub swirl: f64,
}

impl Field for Vortex {
    fn at(&self, p: Point) -> Vec2 {
        let Some(radius) = positive(self.radius) else {
            return Vec2::ZERO;
        };
        let r = p - self.center;
        let g = ring_profile(r.hypot(), radius);
        // Using the un-normalised offset `r` times g(d) avoids dividing by d at the centre.
        let perp = Vec2::new(-r.y, r.x);
        let swirl = if self.swirl.is_finite() {
            self.swirl
        } else {
            0.0
        };
        fin((perp - r * swirl) * (self.strength * g))
    }
}

/// Pulls towards (positive `strength`) or pushes away from (negative) `center`, with the
/// same smooth profile as [`Vortex`]: zero at the centre, peak `|strength|` at `radius`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attractor {
    /// Centre of attraction.
    pub center: Point,
    /// Peak speed; negative makes it a repeller.
    pub strength: f64,
    /// Falloff radius.
    pub radius: f64,
}

impl Attractor {
    /// A repeller: an [`Attractor`] with negative strength.
    pub fn repeller(center: Point, strength: f64, radius: f64) -> Attractor {
        Attractor {
            center,
            strength: -strength.abs(),
            radius,
        }
    }
}

impl Field for Attractor {
    fn at(&self, p: Point) -> Vec2 {
        let Some(radius) = positive(self.radius) else {
            return Vec2::ZERO;
        };
        let r = self.center - p;
        fin(r * (self.strength * ring_profile(r.hypot(), radius)))
    }
}

/// Divergence-free turbulence: the curl of a Perlin potential `ψ`,
/// `v = (∂ψ/∂y, −∂ψ/∂x)` (the gradient is analytic, so the field is exactly
/// divergence-free). Streamlines follow the level sets of `ψ`: they never converge or
/// diverge and fill space with evenly flowing, non-crossing curls.
#[derive(Debug, Clone)]
pub struct CurlNoise {
    /// Noise seed (e.g. from [`SeedTree::noise_seed`]).
    pub seed: u32,
    /// Feature size in canvas units.
    pub scale: f64,
    /// Speed multiplier; magnitudes are typically between 0 and `2·strength`.
    pub strength: f64,
    noise: Gradient2,
}

impl CurlNoise {
    /// Curl noise with the given seed, feature size and strength.
    pub fn new(seed: u32, scale: f64, strength: f64) -> CurlNoise {
        CurlNoise {
            seed,
            scale,
            strength,
            noise: Gradient2::new(seed),
        }
    }
}

impl Field for CurlNoise {
    fn at(&self, p: Point) -> Vec2 {
        let Some(scale) = positive(self.scale) else {
            return Vec2::ZERO;
        };
        if !p.is_finite() {
            return Vec2::ZERO;
        }
        // Gradient in noise units, so the speed does not depend on `scale`.
        let (_, g) = self.noise.eval(p.x / scale, p.y / scale);
        fin(Vec2::new(g.y, -g.x) * self.strength)
    }
}

/// Sum of several fields.
pub struct Sum(pub Vec<Box<dyn Field>>);

impl Field for Sum {
    fn at(&self, p: Point) -> Vec2 {
        fin(self.0.iter().fold(Vec2::ZERO, |acc, f| acc + f.at(p)))
    }
}

/// Linear blend `a·(1−m) + b·m` with a spatial weight mask `m`.
pub struct Blend {
    /// Field where the mask is 0.
    pub a: Box<dyn Field>,
    /// Field where the mask is 1.
    pub b: Box<dyn Field>,
    /// Blend weight.
    pub mask: Box<dyn Mask>,
}

impl Field for Blend {
    fn at(&self, p: Point) -> Vec2 {
        let m = unit(self.mask.at(p));
        fin(self.a.at(p) * (1.0 - m) + self.b.at(p) * m)
    }
}

/// See [`FieldExt::normalized`].
pub struct Normalized<F>(pub F);

impl<F: Field> Field for Normalized<F> {
    fn at(&self, p: Point) -> Vec2 {
        let v = self.0.at(p);
        let m = v.hypot();
        if m > EPS && m.is_finite() {
            v / m
        } else {
            Vec2::ZERO
        }
    }
}

/// See [`FieldExt::scaled`].
pub struct Scaled<F>(pub F, pub f64);

impl<F: Field> Field for Scaled<F> {
    fn at(&self, p: Point) -> Vec2 {
        fin(self.0.at(p) * self.1)
    }
}

/// See [`FieldExt::with_mask`].
pub struct Masked<F, M>(pub F, pub M);

impl<F: Field, M: Mask> Field for Masked<F, M> {
    fn at(&self, p: Point) -> Vec2 {
        fin(self.0.at(p) * unit(self.1.at(p)))
    }
}

/// Combinators available on every [`Field`].
pub trait FieldExt: Field + Sized {
    /// Unit-length vectors (zero where the field vanishes).
    fn normalized(self) -> Normalized<Self> {
        Normalized(self)
    }
    /// Multiply by a constant.
    fn scaled(self, k: f64) -> Scaled<Self> {
        Scaled(self, k)
    }
    /// Multiply by a mask in `[0,1]` (fades the field out where the mask is 0).
    fn with_mask<M: Mask>(self, mask: M) -> Masked<Self, M> {
        Masked(self, mask)
    }
    /// Box as a trait object (for [`Sum`] and [`Blend`]).
    fn boxed(self) -> Box<dyn Field>
    where
        Self: 'static,
    {
        Box::new(self)
    }
}

impl<F: Field> FieldExt for F {}

// ---------------------------------------------------------------------------------------
// Masks
// ---------------------------------------------------------------------------------------

/// A scalar density / weight in `[0,1]`. [`at`](Mask::at) must stay in range for every
/// input, including NaN points.
pub trait Mask {
    /// The mask value at `p`, in `[0,1]`.
    fn at(&self, p: Point) -> f64;
}

impl<M: Mask + ?Sized> Mask for &M {
    fn at(&self, p: Point) -> f64 {
        (**self).at(p)
    }
}

impl<M: Mask + ?Sized> Mask for Box<M> {
    fn at(&self, p: Point) -> f64 {
        (**self).at(p)
    }
}

/// A mask defined by a closure (output clamped to `[0,1]`).
pub struct FnMask<F: Fn(Point) -> f64>(pub F);

impl<F: Fn(Point) -> f64> Mask for FnMask<F> {
    fn at(&self, p: Point) -> f64 {
        unit((self.0)(p))
    }
}

/// The same value everywhere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constant(pub f64);

impl Mask for Constant {
    fn at(&self, _p: Point) -> f64 {
        unit(self.0)
    }
}

/// 1 inside a disc, fading to 0 at `radius` with a smoothstep. `softness` in `[0,1]` is
/// the fraction of the radius over which it fades (0 = hard edge, 1 = from the centre).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Radial {
    /// Centre of the disc.
    pub center: Point,
    /// Outer radius (value 0 beyond).
    pub radius: f64,
    /// Fraction of the radius used for the fade.
    pub softness: f64,
}

impl Mask for Radial {
    fn at(&self, p: Point) -> f64 {
        falloff((p - self.center).hypot(), self.radius, self.softness)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Spine {
    Line { origin: Point, dir: Vec2 },
    Polyline(Vec<Point>),
}

/// A band of half-width `half_width` around a straight axis or a guide curve, 1 on the
/// spine and fading to 0 at the band's edge (same `softness` convention as [`Radial`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    spine: Spine,
    /// Distance from the spine at which the value reaches 0.
    pub half_width: f64,
    /// Fraction of the half-width used for the fade.
    pub softness: f64,
}

impl Band {
    /// A band around the infinite line through `origin` with `direction` (a zero
    /// direction falls back to the x axis).
    pub fn axis(origin: Point, direction: Vec2, half_width: f64, softness: f64) -> Band {
        let m = direction.hypot();
        let dir = if m > EPS && m.is_finite() {
            direction / m
        } else {
            Vec2::new(1.0, 0.0)
        };
        Band {
            spine: Spine::Line { origin, dir },
            half_width,
            softness,
        }
    }

    /// A band following a guide curve. The curve is sampled by arc length finely enough
    /// that the polyline stays well within the band's precision.
    pub fn along<C: Curve + ?Sized>(curve: &C, half_width: f64, softness: f64) -> Band {
        let len = curve.length();
        let hw = positive(half_width).unwrap_or(1.0);
        let n = if len.is_finite() {
            (len / (hw * 0.25)).clamp(32.0, 2048.0) as usize
        } else {
            32
        };
        let pts = curve
            .sample_by_arclen(n)
            .into_iter()
            .map(|(_, p)| p)
            .filter(|p| p.is_finite())
            .collect();
        Band {
            spine: Spine::Polyline(pts),
            half_width,
            softness,
        }
    }

    /// Distance from `p` to the spine.
    pub fn distance(&self, p: Point) -> f64 {
        match &self.spine {
            Spine::Line { origin, dir } => (p - *origin).cross(*dir).abs(),
            Spine::Polyline(pts) => polyline_dist(p, pts),
        }
    }
}

impl Mask for Band {
    fn at(&self, p: Point) -> f64 {
        falloff(self.distance(p), self.half_width, self.softness)
    }
}

/// Perlin noise remapped to `[0,1]`: values below `lo` give 0, above `hi` give 1, with a
/// linear ramp in between (on the noise's own `[0,1]` scale).
#[derive(Debug, Clone)]
pub struct NoiseMask {
    /// Noise seed (e.g. from [`SeedTree::noise_seed`]).
    pub seed: u32,
    /// Feature size in canvas units.
    pub scale: f64,
    /// Noise level mapped to 0.
    pub lo: f64,
    /// Noise level mapped to 1.
    pub hi: f64,
    noise: Gradient2,
}

impl NoiseMask {
    /// A noise mask; `lo`/`hi` are thresholds on the noise remapped to `[0,1]`.
    pub fn new(seed: u32, scale: f64, lo: f64, hi: f64) -> NoiseMask {
        NoiseMask {
            seed,
            scale,
            lo,
            hi,
            noise: Gradient2::new(seed),
        }
    }
}

impl Mask for NoiseMask {
    fn at(&self, p: Point) -> f64 {
        let Some(scale) = positive(self.scale) else {
            return 0.0;
        };
        if !p.is_finite() {
            return 0.0;
        }
        let n = 0.5 + 0.5 * self.noise.eval(p.x / scale, p.y / scale).0;
        unit(unlerp(self.lo, self.hi, n))
    }
}

/// Arithmetic on masks; every result is clamped back to `[0,1]`.
pub enum Combine {
    /// `a · b` (intersection-like).
    Multiply(Box<dyn Mask>, Box<dyn Mask>),
    /// `a + b` (clamped).
    Add(Box<dyn Mask>, Box<dyn Mask>),
    /// `max(a, b)` (union).
    Max(Box<dyn Mask>, Box<dyn Mask>),
    /// `min(a, b)` (intersection).
    Min(Box<dyn Mask>, Box<dyn Mask>),
    /// `a − b` (clamped; carve `b` out of `a`).
    Subtract(Box<dyn Mask>, Box<dyn Mask>),
    /// `1 − a`.
    Invert(Box<dyn Mask>),
}

impl Mask for Combine {
    fn at(&self, p: Point) -> f64 {
        let u = |m: &dyn Mask| unit(m.at(p));
        unit(match self {
            Combine::Multiply(a, b) => u(a.as_ref()) * u(b.as_ref()),
            Combine::Add(a, b) => u(a.as_ref()) + u(b.as_ref()),
            Combine::Max(a, b) => u(a.as_ref()).max(u(b.as_ref())),
            Combine::Min(a, b) => u(a.as_ref()).min(u(b.as_ref())),
            Combine::Subtract(a, b) => u(a.as_ref()) - u(b.as_ref()),
            Combine::Invert(a) => 1.0 - u(a.as_ref()),
        })
    }
}

/// A closed outline prepared for inside tests (non-zero winding) and boundary distance
/// (against a flattened copy).
#[derive(Debug, Clone)]
struct Outline {
    path: BezPath,
    rings: Vec<Vec<Point>>,
    bbox: Rect,
}

impl Outline {
    fn new(path: &BezPath) -> Outline {
        // Drop non-finite / zero-length pieces, then make sure every subpath is closed so
        // the winding number is well defined.
        let clean = sanitize(path, EPS);
        let mut closed = BezPath::new();
        let mut open = false;
        for el in clean.elements() {
            match el {
                PathEl::MoveTo(_) => {
                    if open {
                        closed.close_path();
                    }
                    open = true;
                }
                PathEl::ClosePath => open = false,
                _ => {}
            }
            closed.push(*el);
        }
        if open {
            closed.close_path();
        }
        let bbox = if closed.elements().is_empty() {
            Rect::ZERO
        } else {
            closed.bounding_box()
        };
        let tol = ((bbox.width().hypot(bbox.height())) * 1e-3).clamp(1e-4, 0.25);
        let mut rings: Vec<Vec<Point>> = Vec::new();
        kurbo::flatten(closed.elements().iter().copied(), tol, |el| match el {
            PathEl::MoveTo(p) => rings.push(vec![p]),
            PathEl::LineTo(p) => {
                if let Some(r) = rings.last_mut() {
                    r.push(p);
                }
            }
            _ => {}
        });
        // Close each ring explicitly so the distance includes the closing edge.
        for r in &mut rings {
            if let (Some(&a), Some(&b)) = (r.first(), r.last())
                && a != b
            {
                r.push(a);
            }
        }
        Outline {
            path: closed,
            rings,
            bbox,
        }
    }

    fn contains(&self, p: Point) -> bool {
        p.is_finite()
            && !self.rings.is_empty()
            && p.x >= self.bbox.x0
            && p.x <= self.bbox.x1
            && p.y >= self.bbox.y0
            && p.y <= self.bbox.y1
            && self.path.winding(p) != 0
    }

    fn distance(&self, p: Point) -> f64 {
        self.rings
            .iter()
            .map(|r| polyline_dist(p, r))
            .fold(f64::INFINITY, f64::min)
    }
}

/// `inside` within a closed path, `outside` elsewhere, cross-fading over a band of width
/// `feather` centred on the outline (0 = hard edge). Inside means non-zero winding.
#[derive(Debug, Clone)]
pub struct PathMask {
    outline: Outline,
    /// Value inside the path.
    pub inside: f64,
    /// Value outside the path.
    pub outside: f64,
    /// Width of the cross-fade band.
    pub feather: f64,
}

impl PathMask {
    /// Prepare a path mask (open subpaths are closed implicitly).
    pub fn new(path: &BezPath, inside: f64, outside: f64, feather: f64) -> PathMask {
        PathMask {
            outline: Outline::new(path),
            inside,
            outside,
            feather,
        }
    }
}

impl Mask for PathMask {
    fn at(&self, p: Point) -> f64 {
        if !p.is_finite() {
            return unit(self.outside);
        }
        let inside = self.outline.contains(p);
        let w = match positive(self.feather) {
            None => f64::from(u8::from(inside)),
            Some(f) => {
                // Cheap early-out far from the outline: skip the distance query.
                if !inside && !self.outline.bbox.inflate(f, f).contains(p) {
                    0.0
                } else {
                    let d = self.outline.distance(p);
                    let s = if inside { d } else { -d };
                    smoothstep(0.5 + s / f)
                }
            }
        };
        unit(lerp(unit(self.outside), unit(self.inside), w))
    }
}

// ---------------------------------------------------------------------------------------
// Negative space and focal points
// ---------------------------------------------------------------------------------------

/// A keep-out area for negative space.
#[derive(Debug, Clone, PartialEq)]
pub enum Region {
    /// A disc.
    Circle {
        /// Centre.
        center: Point,
        /// Radius.
        radius: f64,
    },
    /// An axis-aligned rectangle (corners in any order).
    Rect(Rect),
    /// A closed path (non-zero winding).
    Path(BezPath),
}

impl Region {
    /// `true` if `p` lies inside (boundary included for circles and rectangles).
    pub fn contains(&self, p: Point) -> bool {
        if !p.is_finite() {
            return false;
        }
        match self {
            Region::Circle { center, radius } => (p - *center).hypot() <= *radius,
            Region::Rect(r) => {
                p.x >= r.x0.min(r.x1)
                    && p.x <= r.x0.max(r.x1)
                    && p.y >= r.y0.min(r.y1)
                    && p.y <= r.y0.max(r.y1)
            }
            Region::Path(path) => Outline::new(path).contains(p),
        }
    }
}

enum Prepared {
    Circle(Point, f64),
    Rect(Rect),
    Path(Outline),
}

impl Prepared {
    /// Distance to the region, 0 inside.
    fn distance(&self, p: Point) -> f64 {
        match self {
            Prepared::Circle(c, r) => ((p - *c).hypot() - r).max(0.0),
            Prepared::Rect(r) => {
                let dx = (r.x0 - p.x).max(p.x - r.x1).max(0.0);
                let dy = (r.y0 - p.y).max(p.y - r.y1).max(0.0);
                dx.hypot(dy)
            }
            Prepared::Path(o) => {
                if o.contains(p) {
                    0.0
                } else {
                    o.distance(p)
                }
            }
        }
    }
}

/// Mask that is 0 inside every keep-out region and 1 away from them, rising smoothly over
/// `feather` outside each region's edge. Built by [`keep_out`].
pub struct KeepOut {
    regions: Vec<Prepared>,
    /// Width of the fade outside each region.
    pub feather: f64,
}

/// Turn keep-out regions into a [`KeepOut`] mask (0 inside the regions). Regions with
/// non-finite or negative parameters are ignored.
pub fn keep_out(regions: &[Region], feather: f64) -> KeepOut {
    let prepared = regions
        .iter()
        .filter_map(|r| match r {
            Region::Circle { center, radius } => {
                (center.is_finite() && radius.is_finite() && *radius >= 0.0)
                    .then_some(Prepared::Circle(*center, *radius))
            }
            Region::Rect(r) => {
                let n = Rect::new(
                    r.x0.min(r.x1),
                    r.y0.min(r.y1),
                    r.x0.max(r.x1),
                    r.y0.max(r.y1),
                );
                [n.x0, n.y0, n.x1, n.y1]
                    .iter()
                    .all(|v| v.is_finite())
                    .then_some(Prepared::Rect(n))
            }
            Region::Path(p) => {
                let o = Outline::new(p);
                (!o.rings.is_empty()).then_some(Prepared::Path(o))
            }
        })
        .collect();
    KeepOut {
        regions: prepared,
        feather,
    }
}

impl Mask for KeepOut {
    fn at(&self, p: Point) -> f64 {
        if !p.is_finite() {
            return 0.0;
        }
        let d = self
            .regions
            .iter()
            .map(|r| r.distance(p))
            .fold(f64::INFINITY, f64::min);
        if d.is_infinite() {
            return 1.0;
        }
        match positive(self.feather) {
            Some(f) => smoothstep(d / f),
            None => f64::from(u8::from(d > 0.0)),
        }
    }
}

/// A point of visual interest: the composition gathers density and movement around it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FocalPoint {
    /// Centre.
    pub center: Point,
    /// Radius of influence.
    pub radius: f64,
    /// Importance in `[0,1]` (peak mask value).
    pub weight: f64,
}

/// Mask built from focal points: `max_i weight_i · (1 − smoothstep(d_i / radius_i))`.
/// Built by [`focal_mask`].
#[derive(Debug, Clone, PartialEq)]
pub struct FocalMask(pub Vec<FocalPoint>);

impl Mask for FocalMask {
    fn at(&self, p: Point) -> f64 {
        unit(
            self.0
                .iter()
                .map(|f| unit(f.weight) * falloff((p - f.center).hypot(), f.radius, 1.0))
                .fold(0.0, f64::max),
        )
    }
}

/// Density mask peaking at each focal point (max of their radial falloffs).
pub fn focal_mask(points: &[FocalPoint]) -> FocalMask {
    FocalMask(points.to_vec())
}

/// The focal point with the largest weight (the first one on ties); `None` if the slice is
/// empty or no weight is finite.
pub fn strongest_focal(points: &[FocalPoint]) -> Option<&FocalPoint> {
    points
        .iter()
        .filter(|f| f.weight.is_finite() && f.center.is_finite())
        .fold(None, |best: Option<&FocalPoint>, f| match best {
            Some(b) if b.weight >= f.weight => Some(b),
            _ => Some(f),
        })
}

/// A spiral [`Vortex`] around the strongest focal point, with its radius as falloff.
/// `swirl > 0` makes streamlines wind inward towards the focal point.
pub fn focal_spiral(points: &[FocalPoint], strength: f64, swirl: f64) -> Option<Vortex> {
    strongest_focal(points).map(|f| Vortex {
        center: f.center,
        strength,
        radius: f.radius,
        swirl,
    })
}

// ---------------------------------------------------------------------------------------
// Spatial grid (shared by streamline separation and placement)
// ---------------------------------------------------------------------------------------

/// Uniform bucket grid over `bounds`; points outside are clamped into edge cells, which
/// keeps queries exact because clamping never increases the distance between cells.
struct Grid {
    origin: Point,
    cell: f64,
    cols: usize,
    rows: usize,
    cells: Vec<Vec<(Point, f64, u32)>>,
}

impl Grid {
    const MAX_DIM: usize = 512;

    fn new(bounds: Rect, cell: f64) -> Grid {
        let (w, h) = (bounds.width(), bounds.height());
        let min_cell = (w.max(h) / Self::MAX_DIM as f64).max(EPS);
        let cell = positive(cell).unwrap_or(min_cell).max(min_cell);
        let cols = ((w / cell).ceil() as usize).clamp(1, Self::MAX_DIM);
        let rows = ((h / cell).ceil() as usize).clamp(1, Self::MAX_DIM);
        Grid {
            origin: Point::new(bounds.x0, bounds.y0),
            cell,
            cols,
            rows,
            cells: vec![Vec::new(); cols * rows],
        }
    }

    fn idx(&self, p: Point) -> (usize, usize) {
        let ix = ((p.x - self.origin.x) / self.cell).floor();
        let iy = ((p.y - self.origin.y) / self.cell).floor();
        (
            ix.clamp(0.0, (self.cols - 1) as f64) as usize,
            iy.clamp(0.0, (self.rows - 1) as f64) as usize,
        )
    }

    fn insert(&mut self, p: Point, r: f64, tag: u32) {
        if p.is_finite() {
            let (ix, iy) = self.idx(p);
            self.cells[iy * self.cols + ix].push((p, r, tag));
        }
    }

    /// `true` if any stored item within `radius` of `p` satisfies `pred(item, distance)`.
    fn any_within(
        &self,
        p: Point,
        radius: f64,
        pred: impl Fn(&(Point, f64, u32), f64) -> bool,
    ) -> bool {
        if !p.is_finite() || radius.is_nan() || radius <= 0.0 {
            return false;
        }
        let (ix, iy) = self.idx(p);
        let k = (radius / self.cell).ceil().min(Self::MAX_DIM as f64) as usize;
        let (x0, x1) = (ix.saturating_sub(k), (ix + k).min(self.cols - 1));
        let (y0, y1) = (iy.saturating_sub(k), (iy + k).min(self.rows - 1));
        for y in y0..=y1 {
            for x in x0..=x1 {
                for item in &self.cells[y * self.cols + x] {
                    let d = (item.0 - p).hypot();
                    if d < radius && pred(item, d) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------------------
// Streamlines and guide curves
// ---------------------------------------------------------------------------------------

/// Hard cap on integration steps per direction, whatever `max_len / step` says.
const MAX_STEPS: usize = 20_000;
/// Field magnitude below which the flow is considered stagnant.
const MIN_SPEED: f64 = 1e-9;
/// Largest direction change allowed between consecutive steps (cos 30°). A sharper turn
/// means the step is too coarse for the local curvature, or a singularity: stop there.
const MAX_TURN_COS: f64 = 0.866;
/// Own recent samples ignored by the loop test (they are close by construction).
const LOOP_SKIP: u32 = 3;
const HISTORY: u32 = u32::MAX;

/// Unit direction of `sign · field` at `p`, or `None` where the field vanishes.
fn unit_dir(field: &dyn Field, p: Point, sign: f64) -> Option<Vec2> {
    let v = field.at(p) * sign;
    let m = v.hypot();
    (m > MIN_SPEED && m.is_finite()).then(|| v / m)
}

/// One classic RK4 step of length `h` on the normalised field (arc-length
/// parameterisation, so samples come out evenly spaced).
fn rk4(field: &dyn Field, p: Point, h: f64, sign: f64) -> Option<Vec2> {
    let k1 = unit_dir(field, p, sign)?;
    let k2 = unit_dir(field, p + k1 * (0.5 * h), sign)?;
    let k3 = unit_dir(field, p + k2 * (0.5 * h), sign)?;
    let k4 = unit_dir(field, p + k3 * h, sign)?;
    let d = (k1 + (k2 + k3) * 2.0 + k4) * (h / 6.0);
    // The stages nearly cancel only at a singularity (source, sink, saddle).
    let m = d.hypot();
    (m > 0.5 * h).then(|| d * (h / m))
}

/// Fixed settings of one streamline integration.
struct Tracer<'a> {
    field: &'a dyn Field,
    step: f64,
    /// Integration stops before leaving this rectangle.
    domain: Rect,
    /// Extra stop condition (used for streamline separation).
    blocked: &'a dyn Fn(Point) -> bool,
}

impl Tracer<'_> {
    /// Integrate one direction from `start` (not included in the output). Stops at the
    /// domain edge, on stagnation, on a sharp turn, when `blocked(p)` holds, when the curve
    /// loops back within `step/2` of an earlier sample (own or in `grid`), or at `max_len`.
    /// `prev_dir` is the direction of travel arriving at `start`, if any, so the turn test
    /// also covers the first step.
    fn integrate(
        &self,
        start: Point,
        sign: f64,
        max_len: f64,
        grid: &mut Grid,
        mut prev_dir: Option<Vec2>,
    ) -> Vec<Point> {
        let Tracer {
            field,
            step,
            domain,
            blocked,
        } = *self;
        let mut pts = Vec::new();
        let mut prev = start;
        let max_steps = ((max_len / step).floor() as usize).min(MAX_STEPS);
        for i in 0..max_steps {
            let Some(d) = rk4(field, prev, step, sign) else {
                break;
            };
            let dir = d / step;
            if prev_dir.is_some_and(|pd| pd.dot(dir) < MAX_TURN_COS) {
                break;
            }
            let next = prev + d;
            if !next.is_finite() || !domain.contains(next) || blocked(next) {
                break;
            }
            let idx = i as u32;
            // Loop test only after a minimum length so the start does not trip it.
            if i >= 3
                && grid.any_within(next, 0.5 * step, |&(_, _, tag), _| {
                    tag == HISTORY || tag + LOOP_SKIP < idx
                })
            {
                break;
            }
            grid.insert(next, 0.0, idx);
            pts.push(next);
            prev = next;
            prev_dir = Some(dir);
        }
        pts
    }
}

/// Points of a streamline through `start` (backward part reversed, then start, then
/// forward), or `None` for invalid input.
fn streamline_points(
    field: &dyn Field,
    start: Point,
    step: f64,
    max_len: f64,
    bounds: Rect,
    both_directions: bool,
    blocked: &dyn Fn(Point) -> bool,
) -> Option<Vec<Point>> {
    let bounds = norm_rect(bounds)?;
    let step = positive(step)?;
    let max_len = positive(max_len)?;
    if !start.is_finite() || !bounds.contains(start) {
        return None;
    }
    // Integrate inside a slightly shrunken domain: the interpolating spline may bulge a
    // fraction of a step beyond its samples, and this keeps it inside `bounds`.
    let margin = (0.5 * step).min(0.25 * bounds.width().min(bounds.height()));
    let domain = bounds.inset(-margin);
    let mut grid = Grid::new(bounds, step);
    grid.insert(start, 0.0, HISTORY);
    let fwd_len = if both_directions {
        0.5 * max_len
    } else {
        max_len
    };
    let tracer = Tracer {
        field,
        step,
        domain,
        blocked,
    };
    let fwd = tracer.integrate(start, 1.0, fwd_len, &mut grid, None);
    let mut pts = Vec::new();
    if both_directions {
        // Forward samples now count as history for the loop test of the backward pass,
        // so closed orbits are not traced twice.
        let mut back_grid = Grid::new(bounds, step);
        back_grid.insert(start, 0.0, HISTORY);
        for p in &fwd {
            back_grid.insert(*p, 0.0, HISTORY);
        }
        let used = fwd.len() as f64 * step;
        // The backward pass must leave `start` opposite to the forward one, otherwise the
        // joined streamline has a kink there (seen near tight curls and singularities).
        let arrive = fwd.first().map(|p| (start - *p) / step);
        let back = tracer.integrate(start, -1.0, max_len - used, &mut back_grid, arrive);
        pts.extend(back.into_iter().rev());
    }
    pts.push(start);
    pts.extend(fwd);
    (pts.len() >= 2).then_some(pts)
}

/// Fit a smooth composite through evenly spaced streamline samples.
///
/// The samples are thinned adaptively — a knot is kept whenever the accumulated turn since
/// the previous knot reaches 20° or after 6 steps — so straight runs become long segments
/// and bends keep dense knots. Each pair of knots is joined by a cubic Hermite segment
/// whose handles follow the streamline's own unit tangent (central difference of the
/// samples) with length chord/3: G1 at every knot, close to the arc, and immune to the
/// loops that uniform Catmull-Rom makes when knot spacing is uneven.
fn fit_points(pts: &[Point]) -> Option<Composite> {
    const MAX_TURN: f64 = 20.0 * std::f64::consts::PI / 180.0;
    const MAX_GAP: usize = 6;
    let n = pts.len();
    if n < 2 {
        return None;
    }
    let mut knots = vec![0usize];
    let (mut turn, mut gap) = (0.0, 0usize);
    for i in 1..n - 1 {
        let (a, b) = (pts[i] - pts[i - 1], pts[i + 1] - pts[i]);
        turn += a.cross(b).atan2(a.dot(b)).abs();
        gap += 1;
        if turn >= MAX_TURN || gap >= MAX_GAP {
            knots.push(i);
            turn = 0.0;
            gap = 0;
        }
    }
    // Avoid a stub final segment: merge it into the previous one when nearly straight.
    if knots.len() >= 2 && gap < 2 && turn < 0.5 * MAX_TURN {
        knots.pop();
    }
    knots.push(n - 1);
    let tangent = |i: usize| {
        let d = pts[(i + 1).min(n - 1)] - pts[i.saturating_sub(1)];
        let m = d.hypot();
        (m > EPS).then(|| d / m)
    };
    let hermite = |i: usize, j: usize| {
        let (a, b) = (pts[i], pts[j]);
        let chord = (b - a).hypot();
        if chord.is_nan() || chord <= EPS {
            return None;
        }
        let fallback = (b - a) / chord;
        let ta = tangent(i).unwrap_or(fallback);
        let tb = tangent(j).unwrap_or(fallback);
        let h = chord / 3.0;
        Some(CubicBez::new(a, a + ta * h, b - tb * h, b))
    };
    // A thinned segment must stay within `tol` of the samples it skips (in both
    // directions), else it is split at its middle sample. Without this, coarse steps on a
    // tight field let the fit bulge by a sizeable fraction of a step, which eats into the
    // separation that `guide_curves` guarantees.
    let tol = 0.05 * (pts[1] - pts[0]).hypot();
    let fits = |c: &CubicBez, i: usize, j: usize| {
        let poly: Vec<Point> = (0..=16).map(|k| c.eval(f64::from(k) / 16.0)).collect();
        poly.iter().all(|q| polyline_dist(*q, &pts[i..=j]) <= tol)
            && pts[i..=j].iter().all(|q| polyline_dist(*q, &poly) <= tol)
    };
    let mut segs: Vec<CubicBez> = Vec::new();
    for w in knots.windows(2) {
        let mut stack = vec![(w[0], w[1])];
        while let Some((i, j)) = stack.pop() {
            let Some(c) = hermite(i, j) else { continue };
            if j - i >= 2 && !fits(&c, i, j) {
                let m = (i + j) / 2;
                stack.push((m, j));
                stack.push((i, m));
            } else {
                segs.push(c);
            }
        }
    }
    (!segs.is_empty()).then(|| Composite::new_by_length(segs))
}

/// Trace a streamline of `field` through `start` with RK4 steps of arc length `step`.
///
/// Integration stops at `bounds`, where the field vanishes, at a turn sharper than 30°
/// per step (a singularity), when the curve loops back within `step/2` of an earlier
/// sample, or after `max_len` (split evenly between both directions when
/// `both_directions`). The samples are fitted into a smooth [`Composite`] that stays
/// inside `bounds`. Returns `None` for invalid input or when no step could be taken.
pub fn trace(
    field: &dyn Field,
    start: Point,
    step: f64,
    max_len: f64,
    bounds: Rect,
    both_directions: bool,
) -> Option<Composite> {
    let pts = streamline_points(
        field,
        start,
        step,
        max_len,
        bounds,
        both_directions,
        &|_| false,
    )?;
    fit_points(&pts)
}

/// Up to `n` evenly spaced streamlines covering `bounds` (simplified Jobard–Lefer).
///
/// Seeds are taken from a queue of candidates offset by `spacing` along the normal of
/// already accepted curves (falling back to random seeds from `seed`). Each streamline is
/// integrated in both directions until it comes within `0.9·spacing` of an existing curve
/// (looked up in a grid); streamlines shorter than `max(2·spacing, 3·step)` are rejected.
/// `step` is clamped to at most `spacing/2` so the separation test is never skipped over.
/// Deterministic for a given seed; invalid input yields an empty vector.
pub fn guide_curves(
    seed: SeedTree,
    field: &dyn Field,
    bounds: Rect,
    n: usize,
    spacing: f64,
    step: f64,
    max_len: f64,
) -> Vec<Composite> {
    let mut out = Vec::new();
    let (Some(bounds), Some(spacing), Some(step), Some(max_len)) = (
        norm_rect(bounds),
        positive(spacing),
        positive(step),
        positive(max_len),
    ) else {
        return out;
    };
    if n == 0 {
        return out;
    }
    let step = step.min(0.5 * spacing);
    let d_test = 0.9 * spacing;
    let min_len = (2.0 * spacing).max(3.0 * step);
    let mut rng = seed.rng_for("guide_curves");
    let mut grid = Grid::new(bounds, spacing);
    let mut queue: VecDeque<Point> = VecDeque::new();
    let max_random = 20 + 10 * n.min(10_000);
    let mut random_tries = 0;
    while out.len() < n {
        let cand = if let Some(c) = queue.pop_front() {
            c
        } else if random_tries < max_random {
            random_tries += 1;
            Point::new(
                rng.uniform(bounds.x0, bounds.x1),
                rng.uniform(bounds.y0, bounds.y1),
            )
        } else {
            break;
        };
        if !bounds.contains(cand) || grid.any_within(cand, spacing, |_, _| true) {
            continue;
        }
        let blocked = |p: Point| grid.any_within(p, d_test, |_, _| true);
        let Some(pts) = streamline_points(field, cand, step, max_len, bounds, true, &blocked)
        else {
            continue;
        };
        if (pts.len() - 1) as f64 * step < min_len {
            continue;
        }
        let Some(curve) = fit_points(&pts) else {
            continue;
        };
        let id = out.len() as u32;
        let len = curve.length();
        // Register dense samples of the fitted curve (what is actually drawn).
        let m = ((len / (0.5 * step)).ceil() as usize).clamp(2, 4 * MAX_STEPS);
        for (_, p) in curve.sample_by_arclen(m) {
            grid.insert(p, 0.0, id);
        }
        // New seed candidates on both sides, one per `spacing` of arc length.
        let k = ((len / spacing).floor() as usize).clamp(1, MAX_STEPS) + 1;
        for (t, p) in curve.sample_by_arclen(k) {
            let nrm = curve.normal(t);
            queue.push_back(p + nrm * (1.05 * spacing));
            queue.push_back(p - nrm * (1.05 * spacing));
        }
        out.push(curve);
    }
    out
}

// ---------------------------------------------------------------------------------------
// Placement
// ---------------------------------------------------------------------------------------

/// Weighted Poisson-disk placement by dart throwing.
///
/// Candidates are drawn uniformly in `bounds` and accepted with probability
/// `density.at(p)`; a candidate is rejected if an accepted point `q` lies closer than
/// `max(min_dist(p), min_dist(q))`, so spacing can vary with position (smaller
/// `min_dist` = closer packing). Keep-out areas are expressed as zero density (see
/// [`keep_out`]). The output is in acceptance order and fully deterministic; at most
/// `max_count` points are returned (fewer if the attempt budget runs out first).
pub fn place_points(
    seed: SeedTree,
    bounds: Rect,
    density: &dyn Mask,
    min_dist: &dyn Fn(Point) -> f64,
    max_count: usize,
) -> Vec<Point> {
    let mut out = Vec::new();
    let Some(bounds) = norm_rect(bounds) else {
        return out;
    };
    if max_count == 0 {
        return out;
    }
    let diag = bounds.width().hypot(bounds.height());
    // NaN / non-positive spacing means no constraint; a spacing beyond the diagonal (even
    // +inf) already excludes every other point, so clamp it there to keep the grid small.
    let radius = |p: Point| {
        let r = min_dist(p);
        if r > 0.0 { r.min(2.0 * diag) } else { 0.0 }
    };
    // Grid cell from the median spacing over a coarse probe lattice: dense regions get
    // small buckets, sparse ones simply search a few more cells.
    let mut probe: Vec<f64> = (0..81)
        .map(|i| {
            let (u, v) = ((i % 9) as f64 / 8.0, (i / 9) as f64 / 8.0);
            radius(Point::new(
                lerp(bounds.x0, bounds.x1, u),
                lerp(bounds.y0, bounds.y1, v),
            ))
        })
        .filter(|r| *r > 0.0)
        .collect();
    probe.sort_by(f64::total_cmp);
    let cell = probe.get(probe.len() / 2).copied().unwrap_or(diag / 64.0);
    let mut grid = Grid::new(bounds, cell);
    let mut r_max: f64 = 0.0;
    let mut rng = seed.rng_for("place_points");
    let budget = max_count.saturating_mul(40).clamp(2_000, 2_000_000);
    for _ in 0..budget {
        if out.len() >= max_count {
            break;
        }
        let p = Point::new(
            rng.uniform(bounds.x0, bounds.x1),
            rng.uniform(bounds.y0, bounds.y1),
        );
        // Always draw the acceptance number so the stream does not depend on the mask.
        let u = rng.uniform(0.0, 1.0);
        if u >= unit(density.at(p)) {
            continue;
        }
        let r = radius(p);
        if grid.any_within(p, r.max(r_max), |&(_, rq, _), d| d < r.max(rq)) {
            continue;
        }
        grid.insert(p, r, 0);
        r_max = r_max.max(r);
        out.push(p);
    }
    out
}

/// Attachment points along a guide curve, evenly spaced by arc length: `(t, point, unit
/// tangent)` in order along the curve.
///
/// `floor(length / spacing)` points (at least one) are centred on the curve so both ends
/// keep the same margin. `jitter` in `[0,1]` displaces each point by up to
/// `±jitter·spacing/2` along the curve (order is preserved). Zero-length curves and
/// invalid spacing give an empty vector.
pub fn along_curve<C: Curve + ?Sized>(
    curve: &C,
    spacing: f64,
    jitter: f64,
    seed: SeedTree,
) -> Vec<(f64, Point, Vec2)> {
    let mut out = Vec::new();
    let Some(spacing) = positive(spacing) else {
        return out;
    };
    let est = curve.length();
    if !(est.is_finite() && est > EPS) {
        return out;
    }
    let table = curve.arc_length(((est / spacing) * 16.0).clamp(256.0, 16_384.0) as usize);
    let total = table.total();
    if !(total.is_finite() && total > EPS) {
        return out;
    }
    let count = ((total / spacing).floor() as usize).clamp(1, 100_000);
    let margin = 0.5 * (total - (count - 1) as f64 * spacing);
    let jitter = unit(jitter) * 0.5 * spacing;
    let mut rng = seed.rng_for("along_curve");
    for i in 0..count {
        let j = rng.uniform(-1.0, 1.0) * jitter;
        let s = (margin + i as f64 * spacing + j).clamp(0.0, total);
        let t = table.t_at(s);
        if !t.is_finite() {
            continue;
        }
        let p = curve.point(t);
        let tg = curve.tangent(t);
        if p.is_finite() && tg.x.is_finite() && tg.y.is_finite() {
            out.push((t, p, tg));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use kurbo::Line;
    use std::f64::consts::PI;

    fn canvas() -> Rect {
        Rect::new(0.0, 0.0, 400.0, 300.0)
    }

    fn probe_points() -> Vec<Point> {
        let mut v: Vec<Point> = (0..21)
            .flat_map(|i| (0..21).map(move |j| Point::new(i as f64 * 20.0, j as f64 * 15.0)))
            .collect();
        v.extend([
            Point::new(200.0, 150.0),
            Point::new(f64::NAN, 0.0),
            Point::new(f64::INFINITY, 3.0),
            Point::new(1e300, -1e300),
            Point::new(-1e12, 1e12),
        ]);
        v
    }

    fn all_fields() -> Vec<Box<dyn Field>> {
        let c = Point::new(200.0, 150.0);
        vec![
            Directional(Vec2::new(1.0, 0.5)).boxed(),
            Directional(Vec2::new(f64::NAN, 1.0)).boxed(),
            Vortex {
                center: c,
                strength: 2.0,
                radius: 80.0,
                swirl: 0.3,
            }
            .boxed(),
            Vortex {
                center: c,
                strength: 1.0,
                radius: 0.0,
                swirl: 0.0,
            }
            .boxed(),
            Vortex {
                center: c,
                strength: f64::INFINITY,
                radius: 1e-300,
                swirl: f64::NAN,
            }
            .boxed(),
            Attractor {
                center: c,
                strength: 1.0,
                radius: 50.0,
            }
            .boxed(),
            Attractor::repeller(c, 1.0, 50.0).boxed(),
            CurlNoise::new(7, 90.0, 1.0).boxed(),
            CurlNoise::new(7, 0.0, 1.0).boxed(),
            Sum(vec![
                Directional(Vec2::new(0.2, 0.0)).boxed(),
                CurlNoise::new(1, 60.0, 1.0).boxed(),
            ])
            .boxed(),
            Blend {
                a: Directional(Vec2::new(1.0, 0.0)).boxed(),
                b: Directional(Vec2::new(0.0, 1.0)).boxed(),
                mask: Box::new(Radial {
                    center: c,
                    radius: 100.0,
                    softness: 0.5,
                }),
            }
            .boxed(),
            CurlNoise::new(3, 50.0, 1.0)
                .normalized()
                .scaled(2.0)
                .with_mask(Constant(0.5))
                .boxed(),
            Directional(Vec2::new(1.0, 1.0)).scaled(f64::NAN).boxed(),
            FnField(|_p: Point| Vec2::new(f64::NAN, 0.0)).boxed(),
        ]
    }

    #[test]
    fn gradient_noise_derivative_matches_finite_differences() {
        let n = Gradient2::new(9);
        let h = 1e-6;
        let mut seen_hi = false;
        for i in 0..200 {
            let (x, y) = (i as f64 * 0.137 - 3.0, i as f64 * 0.291 + 255.5);
            let (v, g) = n.eval(x, y);
            assert!(v.abs() < 1.0 + 1e-9);
            seen_hi |= v.abs() > 0.4;
            let gx = (n.eval(x + h, y).0 - n.eval(x - h, y).0) / (2.0 * h);
            let gy = (n.eval(x, y + h).0 - n.eval(x, y - h).0) / (2.0 * h);
            assert!(
                (gx - g.x).abs() < 1e-5 && (gy - g.y).abs() < 1e-5,
                "{x},{y}"
            );
        }
        assert!(seen_hi);
        // Periodic: wrapping huge coordinates is exact.
        assert_eq!(
            n.eval(3.25, 1.5).0,
            n.eval(3.25 + 256.0 * 4.0, 1.5 - 256.0).0
        );
    }

    #[test]
    fn fields_are_finite_everywhere() {
        for f in all_fields() {
            for p in probe_points() {
                let v = f.at(p);
                assert!(v.x.is_finite() && v.y.is_finite(), "{p:?} -> {v:?}");
            }
        }
    }

    #[test]
    fn vortex_rotates_and_vanishes_at_centre() {
        let c = Point::new(10.0, 20.0);
        let v = Vortex {
            center: c,
            strength: 3.0,
            radius: 40.0,
            swirl: 0.0,
        };
        assert_eq!(v.at(c), Vec2::ZERO);
        for a in 0..12 {
            let ang = a as f64 / 12.0 * 2.0 * PI;
            let p = c + Vec2::new(ang.cos(), ang.sin()) * 40.0;
            let f = v.at(p);
            // Pure rotation is perpendicular to the radius, peak speed at `radius`.
            assert!(f.dot(p - c).abs() < 1e-9);
            assert!((f.hypot() - 3.0).abs() < 1e-9);
        }
        let inward = Vortex { swirl: 0.5, ..v };
        let p = c + Vec2::new(30.0, 0.0);
        assert!(inward.at(p).dot(p - c) < 0.0);
        let a = Attractor {
            center: c,
            strength: 1.0,
            radius: 10.0,
        };
        assert!(a.at(p).dot(p - c) < 0.0);
        assert!(Attractor::repeller(c, 1.0, 10.0).at(p).dot(p - c) > 0.0);
    }

    #[test]
    fn curl_noise_is_divergence_free_and_deterministic() {
        let f = CurlNoise::new(42, 80.0, 1.0);
        let g = CurlNoise::new(42, 80.0, 1.0);
        let h = 1e-3;
        let mut mean_mag = 0.0;
        for p in probe_points().into_iter().filter(|p| p.x.abs() < 1e6) {
            assert_eq!(f.at(p), g.at(p));
            let div = (f.at(p + Vec2::new(h, 0.0)).x - f.at(p - Vec2::new(h, 0.0)).x
                + f.at(p + Vec2::new(0.0, h)).y
                - f.at(p - Vec2::new(0.0, h)).y)
                / (2.0 * h);
            assert!(div.abs() < 1e-3, "div {div} at {p:?}");
            mean_mag += f.at(p).hypot();
        }
        assert!(mean_mag > 0.0);
        assert_ne!(
            CurlNoise::new(1, 80.0, 1.0).at(Point::new(33.3, 77.7)),
            f.at(Point::new(33.3, 77.7))
        );
    }

    fn max_turn_deg(c: &Composite) -> f64 {
        let tg: Vec<Vec2> = (0..=400).map(|i| c.tangent(i as f64 / 400.0)).collect();
        tg.windows(2)
            .map(|w| w[0].dot(w[1]).clamp(-1.0, 1.0).acos().to_degrees())
            .fold(0.0, f64::max)
    }

    fn assert_well_formed(c: &Composite) {
        let path = c.to_bezpath_exact();
        assert!(path_is_finite(&path));
        assert_eq!(degenerate_segments(&path, 1e-9), 0);
    }

    #[test]
    fn trace_stays_in_bounds_and_is_smooth() {
        let b = canvas();
        let fields: Vec<Box<dyn Field>> = vec![
            CurlNoise::new(9, 120.0, 1.0).boxed(),
            Vortex {
                center: Point::new(200.0, 150.0),
                strength: 1.0,
                radius: 120.0,
                swirl: 0.25,
            }
            .boxed(),
            Sum(vec![
                Directional(Vec2::new(1.0, 0.0)).boxed(),
                CurlNoise::new(2, 70.0, 1.5).boxed(),
            ])
            .boxed(),
        ];
        for f in &fields {
            for s in [Point::new(50.0, 60.0), Point::new(310.0, 220.0)] {
                let c = trace(f.as_ref(), s, 3.0, 2000.0, b, true).unwrap();
                assert_well_formed(&c);
                for p in c.sample(800) {
                    assert!(
                        p.x >= b.x0 && p.x <= b.x1 && p.y >= b.y0 && p.y <= b.y1,
                        "{p:?}"
                    );
                }
                assert!(max_turn_deg(&c) < 60.0);
                let again = trace(f.as_ref(), s, 3.0, 2000.0, b, true).unwrap();
                assert_eq!(c, again);
            }
        }
    }

    #[test]
    fn trace_stops_on_loops_and_stagnation() {
        // Closed circular orbits: the loop test must stop after about one revolution.
        let c = Point::new(200.0, 150.0);
        let v = Vortex {
            center: c,
            strength: 1.0,
            radius: 60.0,
            swirl: 0.0,
        };
        let curve = trace(&v, Point::new(250.0, 150.0), 2.0, 5000.0, canvas(), true).unwrap();
        let len = curve.length();
        assert!(
            len < 2.0 * PI * 50.0 + 6.0 && len > 2.0 * PI * 50.0 * 0.8,
            "{len}"
        );
        // Inward spiral reaches the stagnant centre and stops well before max_len.
        let sp = Vortex { swirl: 0.6, ..v };
        let curve = trace(
            &sp,
            Point::new(260.0, 150.0),
            1.0,
            10_000.0,
            canvas(),
            false,
        )
        .unwrap();
        assert!(curve.length() < 2000.0);
        assert!((curve.end() - c).hypot() < 20.0);
        // Zero field: nothing to trace.
        assert!(trace(&Directional(Vec2::ZERO), c, 1.0, 100.0, canvas(), true).is_none());
    }

    #[test]
    fn trace_rejects_degenerate_input() {
        let f = Directional(Vec2::new(1.0, 0.0));
        let p = Point::new(10.0, 10.0);
        assert!(trace(&f, p, 0.0, 100.0, canvas(), true).is_none());
        assert!(trace(&f, p, f64::NAN, 100.0, canvas(), true).is_none());
        assert!(trace(&f, p, 1.0, 0.0, canvas(), true).is_none());
        assert!(trace(&f, p, 1.0, 100.0, Rect::ZERO, true).is_none());
        assert!(trace(&f, Point::new(f64::NAN, 1.0), 1.0, 100.0, canvas(), true).is_none());
        assert!(trace(&f, Point::new(-5.0, 1.0), 1.0, 100.0, canvas(), true).is_none());
        // Straight flow: length is capped by max_len.
        let c = trace(&f, Point::new(100.0, 100.0), 1.0, 50.0, canvas(), true).unwrap();
        assert!((c.length() - 50.0).abs() < 1.5);
        // A step much larger than the bounds still behaves.
        let tiny = Rect::new(0.0, 0.0, 1.0, 1.0);
        let _ = trace(&f, Point::new(0.5, 0.5), 10.0, 100.0, tiny, true);
    }

    #[test]
    fn guide_curves_respect_spacing() {
        let b = canvas();
        let f = Sum(vec![
            Directional(Vec2::new(0.6, 0.2)).boxed(),
            CurlNoise::new(5, 110.0, 1.0).boxed(),
        ]);
        let spacing = 18.0;
        let curves = guide_curves(SeedTree::new(3), &f, b, 40, spacing, 3.0, 800.0);
        assert!(curves.len() >= 8, "{}", curves.len());
        assert!(curves.len() <= 40);
        let samples: Vec<Vec<Point>> = curves
            .iter()
            .map(|c| {
                assert_well_formed(c);
                assert!(max_turn_deg(c) < 60.0);
                c.sample_by_arclen((c.length() / 1.5) as usize + 2)
                    .into_iter()
                    .map(|(_, p)| p)
                    .collect()
            })
            .collect();
        for i in 0..samples.len() {
            for j in i + 1..samples.len() {
                for p in &samples[i] {
                    for q in &samples[j] {
                        let d = (*p - *q).hypot();
                        assert!(d >= spacing * 0.8, "curves {i},{j}: {d}");
                    }
                }
            }
        }
        let again = guide_curves(SeedTree::new(3), &f, b, 40, spacing, 3.0, 800.0);
        assert_eq!(curves, again);
        let other = guide_curves(SeedTree::new(4), &f, b, 40, spacing, 3.0, 800.0);
        assert_ne!(curves, other);
    }

    #[test]
    fn guide_curves_degenerate_input() {
        let f = CurlNoise::new(1, 50.0, 1.0);
        let s = SeedTree::new(1);
        assert!(guide_curves(s, &f, Rect::ZERO, 10, 10.0, 2.0, 100.0).is_empty());
        assert!(guide_curves(s, &f, canvas(), 10, 0.0, 2.0, 100.0).is_empty());
        assert!(guide_curves(s, &f, canvas(), 10, f64::NAN, 2.0, 100.0).is_empty());
        assert!(guide_curves(s, &f, canvas(), 0, 10.0, 2.0, 100.0).is_empty());
        assert!(guide_curves(s, &f, canvas(), 10, 10.0, 0.0, 100.0).is_empty());
        assert!(
            guide_curves(s, &Directional(Vec2::ZERO), canvas(), 10, 10.0, 2.0, 100.0).is_empty()
        );
    }

    fn all_masks() -> Vec<Box<dyn Mask>> {
        let c = Point::new(200.0, 150.0);
        let curve = CubicBez::new((20.0, 250.0), (120.0, 0.0), (280.0, 300.0), (380.0, 40.0));
        let mut blob = BezPath::new();
        blob.move_to((100.0, 100.0));
        blob.curve_to((200.0, 40.0), (320.0, 120.0), (300.0, 200.0));
        blob.curve_to((260.0, 280.0), (120.0, 260.0), (100.0, 100.0));
        blob.close_path();
        vec![
            Box::new(Constant(0.4)),
            Box::new(Constant(7.0)),
            Box::new(Constant(f64::NAN)),
            Box::new(Radial {
                center: c,
                radius: 90.0,
                softness: 0.6,
            }),
            Box::new(Radial {
                center: c,
                radius: 0.0,
                softness: f64::NAN,
            }),
            Box::new(Band::axis(c, Vec2::new(1.0, 1.0), 30.0, 0.5)),
            Box::new(Band::axis(c, Vec2::ZERO, 30.0, 0.5)),
            Box::new(Band::along(&curve, 25.0, 1.0)),
            Box::new(Band::along(
                &kurbo::Line::new((5.0, 5.0), (5.0, 5.0)),
                f64::NAN,
                0.0,
            )),
            Box::new(NoiseMask::new(3, 60.0, 0.3, 0.7)),
            Box::new(NoiseMask::new(3, 60.0, f64::NAN, 0.7)),
            Box::new(Combine::Subtract(
                Box::new(Constant(1.0)),
                Box::new(NoiseMask::new(4, 30.0, 0.4, 0.6)),
            )),
            Box::new(Combine::Add(
                Box::new(Constant(0.8)),
                Box::new(Constant(0.8)),
            )),
            Box::new(Combine::Invert(Box::new(Constant(0.25)))),
            Box::new(PathMask::new(&blob, 1.0, 0.1, 12.0)),
            Box::new(PathMask::new(&blob, 0.0, 1.0, 0.0)),
            Box::new(PathMask::new(&BezPath::new(), 1.0, 0.0, 5.0)),
            Box::new(keep_out(
                &[
                    Region::Circle {
                        center: c,
                        radius: 40.0,
                    },
                    Region::Rect(Rect::new(300.0, 0.0, 250.0, 50.0)),
                    Region::Path(blob.clone()),
                ],
                10.0,
            )),
            Box::new(focal_mask(&[
                FocalPoint {
                    center: c,
                    radius: 100.0,
                    weight: 1.0,
                },
                FocalPoint {
                    center: Point::new(50.0, 50.0),
                    radius: 0.0,
                    weight: f64::NAN,
                },
            ])),
            Box::new(FnMask(|p: Point| p.x / 100.0)),
        ]
    }

    #[test]
    fn masks_stay_in_unit_range() {
        for m in all_masks() {
            for p in probe_points() {
                let v = m.at(p);
                assert!((0.0..=1.0).contains(&v), "{p:?} -> {v}");
            }
        }
    }

    #[test]
    fn mask_shapes() {
        let c = Point::new(0.0, 0.0);
        let r = Radial {
            center: c,
            radius: 10.0,
            softness: 0.5,
        };
        assert_eq!(r.at(Point::new(4.0, 0.0)), 1.0);
        assert_eq!(r.at(Point::new(10.0, 0.0)), 0.0);
        assert!((r.at(Point::new(7.5, 0.0)) - 0.5).abs() < 1e-12);
        let line = Line::new((0.0, 0.0), (100.0, 0.0));
        let b = Band::along(&line, 10.0, 0.0);
        assert_eq!(b.at(Point::new(50.0, 9.0)), 1.0);
        assert_eq!(b.at(Point::new(50.0, 11.0)), 0.0);
        assert!((b.distance(Point::new(50.0, -4.0)) - 4.0).abs() < 1e-9);
        let a = Band::axis(c, Vec2::new(0.0, 2.0), 10.0, 0.0);
        assert!((a.distance(Point::new(3.0, 99.0)) - 3.0).abs() < 1e-12);

        let sq = Rect::new(0.0, 0.0, 100.0, 100.0).to_path(0.1);
        let pm = PathMask::new(&sq, 1.0, 0.0, 10.0);
        assert_eq!(pm.at(Point::new(50.0, 50.0)), 1.0);
        assert_eq!(pm.at(Point::new(150.0, 50.0)), 0.0);
        assert!((pm.at(Point::new(100.0, 50.0)) - 0.5).abs() < 1e-6);
        assert!(pm.at(Point::new(97.0, 50.0)) > 0.5 && pm.at(Point::new(97.0, 50.0)) < 1.0);
        // Open subpaths are closed implicitly.
        let mut open = BezPath::new();
        open.move_to((0.0, 0.0));
        open.line_to((100.0, 0.0));
        open.line_to((100.0, 100.0));
        open.line_to((0.0, 100.0));
        assert_eq!(
            PathMask::new(&open, 1.0, 0.0, 0.0).at(Point::new(50.0, 50.0)),
            1.0
        );
    }

    #[test]
    fn regions_and_focal_points() {
        let circle = Region::Circle {
            center: Point::new(50.0, 50.0),
            radius: 20.0,
        };
        assert!(circle.contains(Point::new(60.0, 50.0)));
        assert!(!circle.contains(Point::new(80.0, 50.0)));
        let rect = Region::Rect(Rect::new(10.0, 10.0, 0.0, 0.0));
        assert!(rect.contains(Point::new(5.0, 5.0)));
        let path = Region::Path(Rect::new(0.0, 0.0, 10.0, 10.0).to_path(0.1));
        assert!(path.contains(Point::new(5.0, 5.0)));
        assert!(!path.contains(Point::new(f64::NAN, 5.0)));
        let ko = keep_out(&[circle, rect, path], 5.0);
        assert_eq!(ko.at(Point::new(50.0, 50.0)), 0.0);
        assert_eq!(ko.at(Point::new(5.0, 5.0)), 0.0);
        assert_eq!(ko.at(Point::new(300.0, 300.0)), 1.0);
        assert!(ko.at(Point::new(72.0, 50.0)) > 0.0 && ko.at(Point::new(72.0, 50.0)) < 1.0);
        assert_eq!(keep_out(&[], 0.0).at(Point::new(1.0, 1.0)), 1.0);

        let fps = [
            FocalPoint {
                center: Point::new(0.0, 0.0),
                radius: 10.0,
                weight: 0.5,
            },
            FocalPoint {
                center: Point::new(100.0, 0.0),
                radius: 40.0,
                weight: 0.9,
            },
            FocalPoint {
                center: Point::new(200.0, 0.0),
                radius: 40.0,
                weight: 0.9,
            },
        ];
        let m = focal_mask(&fps);
        assert!((m.at(Point::new(0.0, 0.0)) - 0.5).abs() < 1e-12);
        assert!((m.at(Point::new(100.0, 0.0)) - 0.9).abs() < 1e-12);
        assert_eq!(m.at(Point::new(50.0, 200.0)), 0.0);
        let v = focal_spiral(&fps, 1.0, 0.3).unwrap();
        assert_eq!(v.center, Point::new(100.0, 0.0));
        assert_eq!(v.radius, 40.0);
        assert!(focal_spiral(&[], 1.0, 0.3).is_none());
    }

    #[test]
    fn place_points_respects_spacing_density_and_keep_out() {
        let b = canvas();
        let holes = [
            Region::Circle {
                center: Point::new(200.0, 150.0),
                radius: 60.0,
            },
            Region::Rect(Rect::new(0.0, 0.0, 80.0, 60.0)),
        ];
        let density = Combine::Multiply(Box::new(keep_out(&holes, 0.0)), Box::new(Constant(0.9)));
        // Tighter packing on the left.
        let md = |p: Point| 6.0 + 10.0 * p.x / 400.0;
        let pts = place_points(SeedTree::new(11), b, &density, &md, 600);
        assert!(pts.len() > 150, "{}", pts.len());
        assert!(pts.len() <= 600);
        for (i, p) in pts.iter().enumerate() {
            assert!(b.contains(*p));
            assert!(!holes.iter().any(|h| h.contains(*p)), "{p:?} in keep-out");
            for q in &pts[i + 1..] {
                assert!((*p - *q).hypot() >= md(*p).max(md(*q)) - 1e-9);
            }
        }
        let left = pts.iter().filter(|p| p.x < 150.0).count();
        let right = pts.iter().filter(|p| p.x > 250.0).count();
        assert!(left > right, "{left} vs {right}");
        assert_eq!(pts, place_points(SeedTree::new(11), b, &density, &md, 600));
        assert_ne!(pts, place_points(SeedTree::new(12), b, &density, &md, 600));
        // max_count is honoured.
        assert_eq!(
            place_points(SeedTree::new(1), b, &Constant(1.0), &|_| 1.0, 25).len(),
            25
        );
    }

    #[test]
    fn place_points_degenerate_input() {
        let s = SeedTree::new(1);
        let one = Constant(1.0);
        assert!(place_points(s, Rect::ZERO, &one, &|_| 5.0, 10).is_empty());
        assert!(place_points(s, canvas(), &one, &|_| 5.0, 0).is_empty());
        assert!(place_points(s, canvas(), &Constant(0.0), &|_| 5.0, 10).is_empty());
        let nan_rect = Rect::new(f64::NAN, 0.0, 1.0, 1.0);
        assert!(place_points(s, nan_rect, &one, &|_| 5.0, 10).is_empty());
        // NaN / negative spacing degrade to "no spacing constraint".
        let pts = place_points(s, canvas(), &one, &|_| f64::NAN, 50);
        assert_eq!(pts.len(), 50);
        let pts = place_points(s, canvas(), &one, &|_| -3.0, 50);
        assert_eq!(pts.len(), 50);
        // Spacing larger than the canvas: exactly one point fits.
        assert_eq!(place_points(s, canvas(), &one, &|_| 1e6, 50).len(), 1);
    }

    #[test]
    fn along_curve_is_uniform_in_arc_length() {
        let c = Composite::through_points(
            &[
                Point::new(0.0, 0.0),
                Point::new(80.0, 120.0),
                Point::new(200.0, 30.0),
                Point::new(330.0, 160.0),
            ],
            0.5,
        )
        .unwrap();
        let table = c.arc_length(8192);
        let total = table.total();
        let pts = along_curve(&c, 20.0, 0.0, SeedTree::new(1));
        assert_eq!(pts.len(), (total / 20.0).floor() as usize);
        let ss: Vec<f64> = pts.iter().map(|(t, _, _)| table.s_at(*t)).collect();
        for w in ss.windows(2) {
            assert!((w[1] - w[0] - 20.0).abs() < 0.05, "{}", w[1] - w[0]);
        }
        // Centred: equal margins at both ends.
        assert!((ss[0] - (total - ss[ss.len() - 1])).abs() < 0.05);
        for (t, p, tg) in &pts {
            assert!((c.point(*t) - *p).hypot() < 1e-9);
            assert!((tg.hypot() - 1.0).abs() < 1e-9);
        }
        // Jitter stays within half a spacing and keeps order; deterministic.
        let j = along_curve(&c, 20.0, 1.0, SeedTree::new(5));
        assert_eq!(j, along_curve(&c, 20.0, 1.0, SeedTree::new(5)));
        assert_eq!(j.len(), pts.len());
        for (a, b) in j.iter().zip(&pts) {
            assert!((table.s_at(a.0) - table.s_at(b.0)).abs() <= 10.0 + 0.05);
        }
        assert!(j.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[test]
    fn along_curve_degenerate_input() {
        let s = SeedTree::new(1);
        let line = Line::new((0.0, 0.0), (10.0, 0.0));
        assert!(along_curve(&line, 0.0, 0.0, s).is_empty());
        assert!(along_curve(&line, f64::NAN, 0.0, s).is_empty());
        assert!(along_curve(&Line::new((3.0, 3.0), (3.0, 3.0)), 1.0, 0.0, s).is_empty());
        // Spacing longer than the curve: a single point in the middle.
        let one = along_curve(&line, 100.0, f64::NAN, s);
        assert_eq!(one.len(), 1);
        assert!((one[0].1.x - 5.0).abs() < 1e-6);
        // Works through a trait object too.
        let dyn_curve: Box<dyn Curve> = Box::new(line);
        assert_eq!(along_curve(&dyn_curve, 2.0, 0.0, s).len(), 5);
    }

    #[test]
    fn streamline_has_no_kink_where_both_directions_meet() {
        // Regression: the backward pass skipped the turn test against the forward pass, so
        // on a tight curl the two halves met at ~90° and the G1 fit bulged into a neighbour.
        let f = CurlNoise::new(0, 37.5, 1.0);
        let start = Point::new(208.2557552933511, 180.45544993965692);
        let step = 9.0;
        let pts = streamline_points(&f, start, step, 1e6, canvas(), true, &|_| false).unwrap();
        for w in pts.windows(3) {
            let (a, b) = (w[1] - w[0], w[2] - w[1]);
            assert!(a.dot(b) / (a.hypot() * b.hypot()) >= MAX_TURN_COS - 1e-9);
        }
        // The thinned fit stays within 5% of a step of the samples (one-sample segments,
        // which cannot be split, are the only ones allowed slightly more).
        let c = fit_points(&pts).unwrap();
        for p in c.sample(2000) {
            assert!(polyline_dist(p, &pts) <= 0.06 * step, "{p:?}");
        }
    }

    #[test]
    fn guide_curves_keep_spacing_on_tight_curl_with_coarse_step() {
        // Regression: with step = spacing/2 on a field whose features are ~2 spacings wide,
        // curves came within 0.69·spacing of each other.
        let f = CurlNoise::new(0, 37.5, 1.0);
        let spacing = 18.0;
        let curves = guide_curves(SeedTree::new(0), &f, canvas(), 300, spacing, 9.0, 1e6);
        assert!(curves.len() >= 10, "{}", curves.len());
        let samples: Vec<Vec<Point>> = curves
            .iter()
            .map(|c| {
                c.sample_by_arclen((c.length() / 1.0) as usize + 2)
                    .into_iter()
                    .map(|(_, p)| p)
                    .collect()
            })
            .collect();
        for i in 0..samples.len() {
            for j in i + 1..samples.len() {
                for p in &samples[i] {
                    for q in &samples[j] {
                        let d = (*p - *q).hypot();
                        assert!(d >= spacing * 0.8, "curves {i},{j}: {}", d / spacing);
                    }
                }
            }
        }
    }

    #[test]
    fn place_points_handles_overflowing_bounds_and_infinite_spacing() {
        let s = SeedTree::new(1);
        let one = Constant(1.0);
        // Finite corners whose extent overflows: rejected instead of yielding inf points.
        let huge = Rect::new(-1e308, -1e308, 1e308, 1e308);
        assert!(place_points(s, huge, &one, &|_| 5.0, 10).is_empty());
        assert!(
            trace(
                &Directional(Vec2::new(1.0, 0.0)),
                Point::ORIGIN,
                1.0,
                9.0,
                huge,
                true
            )
            .is_none()
        );
        // Infinite spacing excludes every other point (it used to mean "no constraint").
        assert_eq!(
            place_points(s, canvas(), &one, &|_| f64::INFINITY, 10).len(),
            1
        );
    }
}
