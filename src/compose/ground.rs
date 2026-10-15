//! Background fills of the botanica preset (TODO: implemented by a parallel agent):
//! a dense ornament ground of overlapping scrolls and leaves, and a field of blooms.

use super::field::Field;
use super::layout::{Canvas, Occupancy};
use crate::palette::Palette;
use crate::rng::SeedTree;
use crate::scene::Node;

/// Overlapping scrolls, leaves and flowers covering all free space so no bare paper shows.
#[allow(clippy::too_many_arguments)]
pub fn dense_ground(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    field: &dyn Field,
    occupied: &Occupancy,
    fill: f64,
    accuracy: f64,
) -> Node {
    let _ = (seed, canvas, palette, field, occupied, fill, accuracy);
    Node::empty()
}

/// A field of flowers of varied species, sizes and hues over the free space.
#[allow(clippy::too_many_arguments)]
pub fn bloom_field(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    field: &dyn Field,
    occupied: &Occupancy,
    fill: f64,
    accuracy: f64,
) -> Node {
    let _ = (seed, canvas, palette, field, occupied, fill, accuracy);
    Node::empty()
}
