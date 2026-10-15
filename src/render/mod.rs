//! Layer 5 — rendering: scene graph → SVG document → PNG.

pub mod png;
pub mod svg;

pub use png::{render_png, write_png};
pub use svg::{to_svg_document, to_svg_string, write_svg};
