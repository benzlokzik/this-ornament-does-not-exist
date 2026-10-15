//! SVG → PNG via resvg. TODO: implemented by a parallel agent.

use crate::scene::Scene;
use std::path::Path;

/// Rasterise to PNG bytes at `scale` × the scene size.
pub fn render_png(scene: &Scene, scale: f64) -> anyhow::Result<Vec<u8>> {
    let _ = (scene, scale);
    anyhow::bail!("png not implemented yet")
}

pub fn write_png(scene: &Scene, path: impl AsRef<Path>, scale: f64) -> anyhow::Result<()> {
    std::fs::write(path, render_png(scene, scale)?)?;
    Ok(())
}
