//! TODO: implemented by a parallel agent.

use crate::palette::Palette;
use crate::rng::SeedTree;
use crate::scene::Node;

/// Named, visually distinct demonstration motifs, each fitting a `size` × `size` box
/// centred at the origin.
pub fn variants(
    seed: &SeedTree,
    palette: &Palette,
    size: f64,
    accuracy: f64,
) -> Vec<(String, Node)> {
    let _ = (seed, palette, size, accuracy);
    Vec::new()
}
