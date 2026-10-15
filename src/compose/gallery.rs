//! Demonstration gallery: ribbons, spirals, flowers and palettes on one sheet.
//! TODO: implemented by a parallel agent.

use crate::config::Config;
use crate::scene::Scene;

/// Options for the gallery sheet.
#[derive(Debug, Clone)]
pub struct GalleryOptions {
    pub seed: u64,
    pub width: f64,
    pub height: f64,
    pub palette: String,
}

impl Default for GalleryOptions {
    fn default() -> Self {
        let c = Config::default();
        GalleryOptions {
            seed: c.seed,
            width: 1600.0,
            height: 2000.0,
            palette: c.palette,
        }
    }
}

pub fn build(opts: &GalleryOptions) -> anyhow::Result<Scene> {
    let _ = opts;
    anyhow::bail!("gallery not implemented yet")
}
