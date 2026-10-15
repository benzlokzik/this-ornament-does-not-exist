//! Layer 4 — composition and procedural growth: guide curves, vector fields, density
//! masks, focal points, presets and the demonstration gallery.

pub mod botanica;
pub mod field;
pub mod gallery;
pub mod landscape;
pub mod layout;
pub mod preset;

use crate::config::Config;
use crate::scene::Scene;

/// Generate the artwork described by `config`.
pub fn generate(config: &Config) -> anyhow::Result<Scene> {
    config.validate()?;
    preset::build(config)
}
