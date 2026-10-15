//! `orna` — a purely procedural, seed-based Art Nouveau botanical illustration engine.
//!
//! Layers (each depends only on the ones above it):
//! 1. [`geom`] — mathematical geometry: parametric curves, spirals, ribbons, leaves, fitting.
//! 2. [`motif`] — drawable motifs built from geometry (ribbons, spirals, flowers, foliage).
//! 3. [`ornament`] — shape-aware ornamentation (hatching, contours, stippling, clipping).
//! 4. [`compose`] — composition and procedural growth (guide curves, fields, presets, gallery).
//! 5. [`render`] — SVG export and PNG rasterisation of a [`scene`] graph.
//!
//! Everything is deterministic: the same [`config::Config`] yields byte-identical SVG.

pub mod compose;
pub mod config;
pub mod geom;
pub mod motif;
pub mod ornament;
pub mod palette;
pub mod render;
pub mod rng;
pub mod scene;

pub use config::Config;
pub use scene::{Node, Scene, Style};
