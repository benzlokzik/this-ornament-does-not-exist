//! Layer 1 — mathematical geometry.
//!
//! All coordinates are `f64`. Parametric curves are exposed through the [`Curve`] trait,
//! two-parameter shapes (ribbons, petals, leaves) through the [`Surface`] trait.

pub mod curl;
pub mod curve;
pub mod fit;
pub mod leaf;
pub mod ribbon;
pub mod spiral;
pub mod surface;
pub mod validate;

pub use curve::{ArcLength, Composite, Curve, CurveExt, FnCurve, Reversed, SubCurve, Transformed};
pub use surface::{RuledSurface, Surface};

/// Tolerance below which a vector is treated as zero.
pub const EPS: f64 = 1e-9;

/// Numerical derivative step.
pub const H: f64 = 1e-5;

/// Linear interpolation.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Hermite smoothstep on `[0,1]`.
#[inline]
pub fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Map `x` from `[a,b]` to `[0,1]`, clamped.
#[inline]
pub fn unlerp(a: f64, b: f64, x: f64) -> f64 {
    if (b - a).abs() < EPS {
        0.0
    } else {
        ((x - a) / (b - a)).clamp(0.0, 1.0)
    }
}
