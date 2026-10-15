//! One-call engraving: turn a [`Surface`] into a finished, layered scene [`Node`].
//!
//! Layer order (bottom to top): flat fill, then all detail layers (contour stripes,
//! hatching, stippling) in one group clipped by the outline through the scene clip, then
//! the outline stroke. Clipping is done by the renderer, not by polygon Booleans, so the
//! detail lines stay single smooth curves.

use super::contour::nested;
use super::hatch::{HatchOptions, Shade, hatch};
use super::stipple::stipple;
use super::{REL_ACCURACY, chord_length, fit_accuracy, max_half_width};
use crate::geom::validate::{path_is_finite, sanitize};
use crate::geom::{EPS, Surface};
use crate::rng::SeedTree;
use crate::scene::{Color, Node, Style};
use kurbo::BezPath;

/// What to draw; every layer is optional. Widths are stroke widths in canvas units.
#[derive(Clone, Debug)]
pub struct EngraveOptions {
    /// Flat fill under everything.
    pub fill: Option<Color>,
    /// Outline stroke `(colour, width)` drawn on top.
    pub outline: Option<(Color, f64)>,
    /// `n` nested contour stripes per side `(n, colour, width)` (see [`nested`]).
    pub stripes: Option<(usize, Color, f64)>,
    /// Hatching `(options, colour, width)` (see [`hatch`]).
    pub hatch: Option<(HatchOptions, Color, f64)>,
    /// Stippling `(density field, dot radius, colour)` (see [`stipple`]).
    pub stipple: Option<(Shade, f64, Color)>,
}

impl Default for EngraveOptions {
    /// A black outline and nothing else.
    fn default() -> Self {
        EngraveOptions {
            fill: None,
            outline: Some((Color::BLACK, 1.0)),
            stripes: None,
            hatch: None,
            stipple: None,
        }
    }
}

/// Inset of the outermost contour stripe, as a fraction of the half-width.
const STRIPE_INSET: f64 = 0.15;

/// Engrave `surface`: returns a group `[fill] + [details clipped by the outline] +
/// [outline]`, omitting empty layers.
///
/// Each detail layer is merged into a single path so the scene stays small. Hatching and
/// stippling draw from `seed.child("hatch")` and `seed.child("stipple")`. Layers with a
/// non-finite or non-positive width/radius are skipped; a degenerate surface yields an
/// empty group.
pub fn engrave<S: Surface + ?Sized>(
    surface: &S,
    opts: &EngraveOptions,
    seed: &SeedTree,
    accuracy: f64,
) -> Node {
    let mut root = Node::empty();
    // Signed area cannot detect a zero-width surface: a twisted ribbon's figure-eight
    // outline has lobes that cancel. Test the width itself.
    let extent = [1.0, -1.0]
        .map(|v| chord_length(&*surface.iso_v(v), 64))
        .iter()
        .sum::<f64>();
    if max_half_width(surface) < EPS || !extent.is_finite() {
        return root;
    }
    let acc = fit_accuracy(accuracy).max(extent * REL_ACCURACY);
    let outline = sanitize(&surface.outline(acc), 1e-6);
    if outline.elements().len() < 3 || !path_is_finite(&outline) {
        return root;
    }
    let positive = |w: f64| w.is_finite() && w > 0.0;

    if let Some(c) = opts.fill {
        root.push(Node::path(outline.clone(), Style::fill(c)));
    }

    let mut details = Node::empty();
    let mut add = |paths: Vec<BezPath>, style: Style| {
        let merged = merge(paths);
        if !merged.elements().is_empty() {
            details.push(Node::path(merged, style));
        }
    };
    if let Some((n, c, w)) = opts.stripes
        && positive(w)
    {
        add(
            nested(surface, n, STRIPE_INSET, false, accuracy),
            Style::stroke(c, w),
        );
    }
    if let Some((h, c, w)) = &opts.hatch
        && positive(*w)
    {
        add(hatch(surface, h, seed, accuracy), Style::stroke(*c, *w));
    }
    if let Some((shade, r, c)) = &opts.stipple
        && positive(*r)
    {
        let density = |u: f64, v: f64| shade.at(u, v);
        add(
            stipple(surface, &density, *r, seed, accuracy),
            Style::fill(*c),
        );
    }
    if details.path_count() > 0 {
        root.push(details.clipped(outline.clone()));
    }

    if let Some((c, w)) = opts.outline
        && positive(w)
    {
        root.push(Node::path(outline, Style::stroke(c, w)));
    }
    root
}

/// Concatenate paths as subpaths of one path.
fn merge(paths: Vec<BezPath>) -> BezPath {
    let mut out = BezPath::new();
    for p in paths {
        out.extend(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::validate::{degenerate_segments, path_is_finite};
    use crate::ornament::test_shapes::{Bow, Huge, Leaf, Needle, petal};
    use crate::scene::Clip;

    fn full() -> EngraveOptions {
        let ink = Color::hex("#2b1d14").unwrap();
        EngraveOptions {
            fill: Some(Color::hex("#e9d8b4").unwrap()),
            outline: Some((ink, 1.2)),
            stripes: Some((3, ink, 0.5)),
            hatch: Some((HatchOptions::default(), ink, 0.4)),
            stipple: Some((Shade::Ends { width: 0.3 }, 0.5, ink)),
        }
    }

    #[test]
    fn layers_are_ordered_and_clipped() {
        let seed = SeedTree::new(8);
        for s in [&petal() as &dyn Surface, &Leaf] {
            let node = engrave(s, &full(), &seed, 0.05);
            assert!(node.is_finite());
            let Node::Group { children, .. } = &node else {
                panic!("not a group")
            };
            assert_eq!(children.len(), 3);
            assert!(matches!(&children[0], Node::Path { style, .. } if style.fill.is_some()));
            let Node::Group {
                clip: Some(Clip { path: clip }),
                children: details,
                ..
            } = &children[1]
            else {
                panic!("details not clipped")
            };
            assert_eq!(details.len(), 3);
            assert!(
                matches!(&children[2], Node::Path { path, style } if style.stroke.is_some() && path == clip)
            );
            node.for_each_path(|p, _, _| {
                assert!(path_is_finite(p));
                assert_eq!(degenerate_segments(p, 1e-6), 0);
            });
        }
    }

    #[test]
    fn reproducible() {
        let seed = SeedTree::new(21);
        let a = engrave(&Leaf, &full(), &seed, 0.05);
        assert_eq!(a, engrave(&Leaf, &full(), &seed, 0.05));
        assert_ne!(a, engrave(&Leaf, &full(), &SeedTree::new(22), 0.05));
    }

    #[test]
    fn degenerate_inputs() {
        let seed = SeedTree::new(1);
        assert_eq!(engrave(&Needle, &full(), &seed, 0.05).path_count(), 0);
        let mut o = full();
        o.outline = Some((Color::BLACK, f64::NAN));
        o.stripes = Some((usize::MAX, Color::BLACK, -1.0));
        o.stipple = Some((Shade::Uniform(1.0), 0.0, Color::BLACK));
        let n = engrave(&petal(), &o, &seed, f64::NAN);
        assert!(n.is_finite());
        // fill + clipped hatching only
        assert_eq!(n.path_count(), 2);
        let bare = EngraveOptions {
            outline: None,
            ..EngraveOptions::default()
        };
        assert_eq!(engrave(&petal(), &bare, &seed, 0.05).path_count(), 0);
    }

    #[test]
    fn twisted_and_huge_surfaces_are_engraved() {
        let seed = SeedTree::new(3);
        // the bow-tie outline has zero signed area but two real lobes
        let bow = engrave(&Bow, &full(), &seed, 0.05);
        assert_eq!(bow.path_count(), 5);
        let huge = engrave(&Huge, &full(), &seed, 0.05);
        assert!(huge.is_finite());
        assert!(huge.path_count() >= 3);
    }
}
