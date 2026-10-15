//! Output size budget: fitted detail lines, level of detail and compact path data keep the
//! SVG small (before them botanica at 600 × 900 wrote 28–50 MB; now 3–4 MB).

use orna::compose::generate;
use orna::config::{Config, Preset};
use orna::render::to_svg_string;

/// Largest accepted SVG for botanica at 600 × 900, detail 1, default accuracy (bytes).
const BUDGET: usize = 6_000_000;

#[test]
fn botanica_svg_stays_within_budget() {
    let config = Config {
        seed: 1234,
        preset: Preset::Botanica,
        width: 600.0,
        height: 900.0,
        detail: 1.0,
        ..Default::default()
    };
    let svg = to_svg_string(&generate(&config).expect("botanica builds"));
    assert!(
        svg.len() < BUDGET,
        "botanica 600x900 wrote {} bytes (budget {BUDGET})",
        svg.len()
    );
}
