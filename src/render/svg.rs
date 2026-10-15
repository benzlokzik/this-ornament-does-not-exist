//! Scene → SVG. TODO: implemented by a parallel agent.

use crate::scene::Scene;
use std::path::Path;

pub fn to_svg_document(scene: &Scene) -> svg::Document {
    let _ = scene;
    svg::Document::new()
}

pub fn to_svg_string(scene: &Scene) -> String {
    to_svg_document(scene).to_string()
}

pub fn write_svg(scene: &Scene, path: impl AsRef<Path>) -> anyhow::Result<()> {
    std::fs::write(path, to_svg_string(scene))?;
    Ok(())
}
