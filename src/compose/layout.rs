//! Composition engine shared by the presets: the canvas split into a warm and a cool
//! region by a flowing divider, the layered painting order, a path budget, an occupancy
//! map of keep-out regions and helpers to place and colour motifs by region.
//!
//! All presets compose in a *warm-left* frame: the warm region lies left of the divider.
//! [`Canvas::finish`] mirrors the whole artwork when the seed puts the warm side on the
//! right, so the composition rules only need to be written once.

use super::field::{KeepOut, Region, keep_out};
use crate::geom::curl::organic_curve;
use crate::geom::{Composite, Curve, CurveExt, smoothstep};
use crate::motif::flower::FlowerStyle;
use crate::motif::foliage::LeafStyle;
use crate::motif::plant::PlantStyle;
use crate::motif::ribbon::RibbonStyle;
use crate::palette::Palette;
use crate::rng::{Draw, SeedTree};
use crate::scene::{Color, Node, Paint};
use kurbo::{Affine, BezPath, Point, Rect, Vec2};

/// The canvas of a full-scene composition and its warm/cool split.
#[derive(Debug, Clone)]
pub struct Canvas {
    /// Canvas width.
    pub width: f64,
    /// Canvas height.
    pub height: f64,
    /// The divider, from above the top edge to below the bottom edge.
    pub divider: Composite,
    /// Divider samples ordered by `y` (for [`divider_x`](Self::divider_x)).
    samples: Vec<Point>,
    /// Half-width of the band over which colours blend across the divider.
    pub blend: f64,
    /// Whether the warm region ends up on the left of the final artwork.
    pub warm_left: bool,
}

impl Canvas {
    /// A canvas whose divider is an organic S running from top to bottom: in the upper
    /// half it bows into the warm (left) side, in the lower half into the cool side, which
    /// leaves room for a cool focal area above and a warm one below.
    pub fn new(seed: &SeedTree, width: f64, height: f64) -> Canvas {
        let mut rng = seed.rng_for("divider");
        let x0 = width * rng.uniform(0.46, 0.56);
        let x1 = width * rng.uniform(0.44, 0.54);
        let wave = rng.uniform(0.38, 0.55) * (width / height).clamp(0.4, 1.2) / 0.667;
        let bulge = rng.uniform(-0.12, 0.12);
        let start = Point::new(x0, -0.05 * height);
        let end = Point::new(x1, 1.05 * height);
        let divider = organic_curve(&mut rng, start, end, wave.min(0.7), bulge);
        let mut samples: Vec<Point> = divider
            .sample_by_arclen(256)
            .into_iter()
            .map(|s| s.1)
            .collect();
        samples.sort_by(|a, b| a.y.total_cmp(&b.y));
        Canvas {
            width,
            height,
            divider,
            samples,
            blend: 0.07 * width,
            warm_left: rng.chance(0.5),
        }
    }

    /// The canvas rectangle.
    pub fn rect(&self) -> Rect {
        Rect::new(0.0, 0.0, self.width, self.height)
    }

    /// The smaller canvas side: the unit most sizes are measured in.
    pub fn unit(&self) -> f64 {
        self.width.min(self.height)
    }

    /// The canvas shrunk by `margin` on every side.
    pub fn inner(&self, margin: f64) -> Rect {
        self.rect().inset(-margin)
    }

    /// Horizontal position of the divider at height `y` (clamped to its ends).
    pub fn divider_x(&self, y: f64) -> f64 {
        let s = &self.samples;
        let i = s.partition_point(|p| p.y < y);
        if i == 0 {
            return s[0].x;
        }
        if i >= s.len() {
            return s[s.len() - 1].x;
        }
        let (a, b) = (s[i - 1], s[i]);
        let f = if b.y - a.y > 1e-9 {
            (y - a.y) / (b.y - a.y)
        } else {
            0.0
        };
        a.x + (b.x - a.x) * f
    }

    /// Signed horizontal distance from the divider (negative on the warm side).
    pub fn side(&self, p: Point) -> f64 {
        p.x - self.divider_x(p.y)
    }

    /// Warmth in `[0, 1]`: 1 deep in the warm region, 0 in the cool one, blending smoothly
    /// across the divider over [`blend`](Self::blend) on each side.
    pub fn warmth(&self, p: Point) -> f64 {
        1.0 - smoothstep((self.side(p) / self.blend + 1.0) * 0.5)
    }

    /// `true` if `p` lies on the warm side.
    pub fn is_warm(&self, p: Point) -> bool {
        self.side(p) < 0.0
    }

    /// Closed outline of one side, the divider shifted by `offset` towards that side.
    pub fn side_path(&self, warm: bool, offset: f64) -> BezPath {
        let mut path = BezPath::new();
        let edge_x = if warm { -self.width } else { 2.0 * self.width };
        let shift = if warm { -offset } else { offset };
        let top = -0.1 * self.height;
        let bottom = 1.1 * self.height;
        path.move_to(Point::new(edge_x, top));
        let n = 64;
        for i in 0..=n {
            let y = top + (bottom - top) * i as f64 / n as f64;
            path.line_to(Point::new(self.divider_x(y) + shift, y));
        }
        path.line_to(Point::new(edge_x, bottom));
        path.close_path();
        path
    }

    /// The finished artwork from layers composed in the warm-left frame, mirrored when the
    /// warm region belongs on the right.
    pub fn finish(&self, root: Node) -> Node {
        if self.warm_left {
            root
        } else {
            root.with_transform(Affine::translate((self.width, 0.0)) * Affine::FLIP_X)
        }
    }
}

/// An upper bound on the number of paths, shared by all layers of an artwork so the
/// output stays renderable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    limit: usize,
    used: usize,
}

impl Budget {
    /// A budget of `limit` paths.
    pub fn new(limit: usize) -> Budget {
        Budget { limit, used: 0 }
    }

    /// Paths still available.
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.used)
    }

    /// Paths spent so far.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Spend the paths of `node` if they fit; `false` (and nothing spent) otherwise.
    pub fn admit(&mut self, node: &Node) -> bool {
        let n = node.path_count();
        if self.used + n > self.limit {
            return false;
        }
        self.used += n;
        true
    }
}

/// Painting order of a composition, back to front.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// Tonal background shapes.
    Background,
    /// The open landscape (sky, moon, hills) painted across the cool region.
    Landscape,
    /// Dim, low-contrast foliage filling the gaps.
    BackFoliage,
    /// The moonlit opening and its frame.
    Opening,
    /// Large ribbons and spirals along the guide curves.
    Ribbons,
    /// Plants growing from the ground line.
    Plants,
    /// Vines and small flowers along the secondary guides.
    Vines,
    /// Focal flowers, their foliage and foreground scrolls.
    Foreground,
}

const LAYERS: [(Layer, &str); 8] = [
    (Layer::Background, "background"),
    (Layer::Landscape, "landscape"),
    (Layer::BackFoliage, "back-foliage"),
    (Layer::Opening, "opening"),
    (Layer::Ribbons, "ribbons"),
    (Layer::Plants, "plants"),
    (Layer::Vines, "vines"),
    (Layer::Foreground, "foreground"),
];

/// Nodes collected per [`Layer`] and the budget they are charged to.
#[derive(Debug, Clone)]
pub struct Layers {
    nodes: Vec<Vec<Node>>,
    /// The path budget every [`add`](Self::add) is charged to.
    pub budget: Budget,
}

impl Layers {
    /// Empty layers sharing a budget of `limit` paths.
    pub fn new(limit: usize) -> Layers {
        Layers {
            nodes: vec![Vec::new(); LAYERS.len()],
            budget: Budget::new(limit),
        }
    }

    /// Add `node` to `layer` if it is finite, non-empty and fits the budget.
    pub fn add(&mut self, layer: Layer, node: Node) -> bool {
        if node.path_count() == 0 || !node.is_finite() || !self.budget.admit(&node) {
            return false;
        }
        let i = LAYERS.iter().position(|(l, _)| *l == layer).unwrap_or(0);
        self.nodes[i].push(node);
        true
    }

    /// All layers as one group, back to front, each with its layer name as id.
    pub fn into_node(self) -> Node {
        let mut root = Node::empty().with_id("artwork");
        for ((_, name), nodes) in LAYERS.iter().zip(self.nodes) {
            if !nodes.is_empty() {
                root.push(Node::group(nodes).with_id(*name));
            }
        }
        root
    }
}

/// `node` (drawn around the origin, pointing along +x) rotated by `angle`, scaled by
/// `scale` and moved to `at`.
pub fn place(node: Node, at: Point, angle: f64, scale: f64) -> Node {
    node.with_transform(
        Affine::translate(at.to_vec2()) * Affine::rotate(angle) * Affine::scale(scale),
    )
}

/// Keep-out regions of everything placed so far; negative space for later layers.
#[derive(Debug, Clone, Default)]
pub struct Occupancy {
    regions: Vec<Region>,
}

impl Occupancy {
    /// An empty map.
    pub fn new() -> Occupancy {
        Occupancy::default()
    }

    /// The regions collected so far.
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    /// Mark a disc.
    pub fn add_circle(&mut self, center: Point, radius: f64) {
        if center.is_finite() && radius.is_finite() && radius > 0.0 {
            self.regions.push(Region::Circle { center, radius });
        }
    }

    /// Mark any region.
    pub fn add_region(&mut self, region: Region) {
        self.regions.push(region);
    }

    /// Mark every region of `other`.
    pub fn extend(&mut self, other: &Occupancy) {
        self.regions.extend(other.regions.iter().cloned());
    }

    /// Mark a band of half-width `radius` along `curve` as discs spaced `radius` apart.
    pub fn add_curve<C: Curve + ?Sized>(&mut self, curve: &C, radius: f64) {
        let len = curve.length();
        if !(len.is_finite() && radius > 0.0) {
            return;
        }
        let n = ((len / radius).ceil() as usize + 1).clamp(2, 400);
        for (_, p) in curve.sample_by_arclen(n) {
            self.add_circle(p, radius);
        }
    }

    /// Mark the bounding box of `node`, shrunk by `shrink` (a fraction of its smaller side;
    /// negative values grow it).
    pub fn add_node(&mut self, node: &Node, shrink: f64) {
        if let Some(bb) = node.bounding_box() {
            let d = shrink * bb.width().min(bb.height());
            let r = bb.inset(-d);
            if r.width() > 0.0 && r.height() > 0.0 {
                self.regions.push(Region::Rect(r));
            }
        }
    }

    /// `true` if `p` lies inside a marked region.
    pub fn contains(&self, p: Point) -> bool {
        self.regions.iter().any(|r| r.contains(p))
    }

    /// Density mask that is 0 inside every region and 1 at `feather` from all of them.
    pub fn free_mask(&self, feather: f64) -> KeepOut {
        keep_out(&self.regions, feather)
    }
}

/// Unit vector at `angle` radians.
pub fn dir(angle: f64) -> Vec2 {
    Vec2::from_angle(angle)
}

/// Unit `v`, or `fallback` when `v` vanishes.
pub fn unit_or(v: Vec2, fallback: Vec2) -> Vec2 {
    let l = v.hypot();
    if l > 1e-12 && l.is_finite() {
        v / l
    } else {
        fallback
    }
}

// ---------------------------------------------------------------------------------------
// Region colours
// ---------------------------------------------------------------------------------------

/// The colour of the warm or cool ramp at `t`.
pub fn ramp(palette: &Palette, warm: bool, t: f64) -> Color {
    if warm {
        palette.warm_at(t)
    } else {
        palette.cool_at(t)
    }
}

/// Flower colours for a point of warmth `w`: the warm style, the cool style or a blend.
pub fn flower_style(palette: &Palette, w: f64, t: f64) -> FlowerStyle {
    let a = FlowerStyle::from_palette(palette, true, t);
    let b = FlowerStyle::from_palette(palette, false, t);
    let k = 1.0 - w.clamp(0.0, 1.0);
    FlowerStyle {
        petal: a.petal.mix(b.petal, k),
        petal_light: a.petal_light.mix(b.petal_light, k),
        ink: a.ink,
        vein: a.vein.mix(b.vein, k),
        center: a.center.mix(b.center, k),
    }
}

/// Leaf colours for warmth `w` (see [`flower_style`]).
pub fn leaf_style(palette: &Palette, w: f64, t: f64) -> LeafStyle {
    let a = LeafStyle::from_palette(palette, true, t);
    let b = LeafStyle::from_palette(palette, false, t);
    mix_leaf(&a, &b, 1.0 - w.clamp(0.0, 1.0))
}

/// `a` mixed towards `b` by `k`.
pub fn mix_leaf(a: &LeafStyle, b: &LeafStyle, k: f64) -> LeafStyle {
    LeafStyle {
        fill: a.fill.mix(b.fill, k),
        fill_light: a.fill_light.mix(b.fill_light, k),
        ink: a.ink.mix(b.ink, k),
        vein: a.vein.mix(b.vein, k),
        widths: a.widths,
    }
}

/// A dim, low-contrast version of a leaf style for the background: desaturated, its
/// light half and ink pulled towards the fill.
pub fn muted_leaf(style: &LeafStyle, background: Color) -> LeafStyle {
    let fill = style.fill.saturate(0.7).mix(background, 0.12);
    LeafStyle {
        fill,
        fill_light: style.fill_light.saturate(0.7).mix(fill, 0.45),
        ink: style.ink.mix(fill, 0.45),
        vein: style.vein.mix(fill, 0.35),
        widths: style.widths,
    }
}

/// Ribbon colours for warmth `w` (see [`flower_style`]).
pub fn ribbon_style(palette: &Palette, w: f64, t: f64) -> RibbonStyle {
    let a = RibbonStyle::from_palette(palette, true, t);
    let b = RibbonStyle::from_palette(palette, false, t);
    let k = 1.0 - w.clamp(0.0, 1.0);
    RibbonStyle {
        fill: a.fill.mix(b.fill, k),
        highlight: a.highlight.mix(b.highlight, k),
        stripe: a.stripe.mix(b.stripe, k),
        ..a
    }
}

/// Plant colours by region: warm plants carry warm flowers on sage foliage; cool plants
/// keep cool leaves and stems with pale lavender-blue flowers.
pub fn plant_style(palette: &Palette, warm: bool) -> PlantStyle {
    if warm {
        return PlantStyle::from_palette(palette, true);
    }
    PlantStyle {
        leaf: LeafStyle::from_palette(palette, false, 0.1),
        leaf_light: LeafStyle::from_palette(palette, false, 0.6),
        stem: palette.cool_at(0.2).mix(palette.ink, 0.2),
        stem_light: palette.cool_at(0.7).mix(palette.light, 0.2),
        ink: palette.ink,
        flowers: [0.45, 0.7, 0.9]
            .iter()
            .map(|&t| FlowerStyle::from_palette(palette, false, t))
            .collect(),
    }
}

/// `c` with its hue rotated by `deg` degrees (Oklch); exactly `c` for 0 or a non-finite
/// shift, so unshifted colours keep their last bits.
fn hue(c: Color, deg: f64) -> Color {
    if deg == 0.0 || !deg.is_finite() {
        c
    } else {
        c.rotate_hue(deg)
    }
}

/// [`ramp`] with the hue rotated by `shift` degrees, for several hues per region.
pub fn ramp_shifted(palette: &Palette, warm: bool, t: f64, shift: f64) -> Color {
    hue(ramp(palette, warm, t), shift)
}

/// A tone for warmth `warm ∈ [0, 1]` (0 = cool ramp, 1 = warm ramp, blended between) at
/// ramp position `t` (dark → light), its hue rotated by `shift` degrees.
pub fn region_tone(palette: &Palette, warm: f64, t: f64, shift: f64) -> Color {
    let k = 1.0
        - if warm.is_finite() {
            warm.clamp(0.0, 1.0)
        } else {
            0.5
        };
    hue(palette.warm_at(t).mix(palette.cool_at(t), k), shift)
}

/// [`flower_style`] with the petal and vein hues rotated by `shift` degrees (the centre
/// keeps its contrasting colour).
pub fn flower_style_shifted(palette: &Palette, w: f64, t: f64, shift: f64) -> FlowerStyle {
    let s = flower_style(palette, w, t);
    FlowerStyle {
        petal: hue(s.petal, shift),
        petal_light: hue(s.petal_light, shift),
        vein: hue(s.vein, shift),
        ..s
    }
}

/// [`leaf_style`] with the blade and vein hues rotated by `shift` degrees.
pub fn leaf_style_shifted(palette: &Palette, w: f64, t: f64, shift: f64) -> LeafStyle {
    let s = leaf_style(palette, w, t);
    LeafStyle {
        fill: hue(s.fill, shift),
        fill_light: hue(s.fill_light, shift),
        vein: hue(s.vein, shift),
        ..s
    }
}

/// [`ribbon_style`] with the body, highlight, stripe and shade hues rotated by `shift`
/// degrees.
pub fn ribbon_style_shifted(palette: &Palette, w: f64, t: f64, shift: f64) -> RibbonStyle {
    let s = ribbon_style(palette, w, t);
    RibbonStyle {
        fill: hue(s.fill, shift),
        highlight: hue(s.highlight, shift),
        stripe: hue(s.stripe, shift),
        shade: hue(s.shade, shift),
        ..s
    }
}

/// The tonal gradient of a motif filled with `base`, after the reference: dark at the
/// bottom of its bounding box, light at the top, the hue drifting along the way — warm
/// shapes from wine to rose, cool ones from indigo towards sage. Equal bases give equal
/// paints, which the SVG export writes once.
pub fn shade_paint(base: Color, warm: bool) -> Paint {
    let (dark_hue, light_hue) = if warm { (-6.0, 8.0) } else { (8.0, -24.0) };
    Paint::linear(
        (0.5, 1.0),
        (0.5, 0.0),
        [
            (0.0, base.lighten(-0.12).rotate_hue(dark_hue)),
            (1.0, base.lighten(0.12).rotate_hue(light_hue)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Style;
    use kurbo::Circle;
    use kurbo::Shape;

    #[test]
    fn divider_splits_warm_and_cool() {
        let c = Canvas::new(&SeedTree::new(3), 1200.0, 1800.0);
        for y in [0.0, 600.0, 1200.0, 1800.0] {
            let x = c.divider_x(y);
            assert!(x > 150.0 && x < 1050.0, "divider at {x}");
            assert!(c.warmth(Point::new(10.0, y)) > 0.99);
            assert!(c.warmth(Point::new(1190.0, y)) < 0.01);
            let mid = c.warmth(Point::new(x, y));
            assert!((mid - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn budget_and_layers() {
        let dot = || {
            Node::path(
                Circle::new((0.0, 0.0), 1.0).to_path(0.1),
                Style::fill(Color::BLACK),
            )
        };
        let mut layers = Layers::new(2);
        assert!(layers.add(Layer::Foreground, dot()));
        assert!(layers.add(Layer::Background, dot()));
        assert!(!layers.add(Layer::Ribbons, dot()));
        assert_eq!(layers.budget.used(), 2);
        let root = layers.into_node();
        assert_eq!(root.path_count(), 2);
    }

    #[test]
    fn occupancy_masks_regions() {
        let mut o = Occupancy::new();
        o.add_circle(Point::new(50.0, 50.0), 10.0);
        use super::super::field::Mask;
        let m = o.free_mask(5.0);
        assert_eq!(m.at(Point::new(50.0, 50.0)), 0.0);
        assert_eq!(m.at(Point::new(90.0, 90.0)), 1.0);
        assert!(o.contains(Point::new(55.0, 50.0)));
    }

    #[test]
    fn hue_shifts_and_region_tones() {
        let p = Palette::by_name("nocturne").unwrap();
        // Zero shift is exactly the unshifted helpers.
        assert_eq!(
            flower_style_shifted(&p, 0.7, 0.4, 0.0),
            flower_style(&p, 0.7, 0.4)
        );
        assert_eq!(
            leaf_style_shifted(&p, 0.2, 0.4, 0.0),
            leaf_style(&p, 0.2, 0.4)
        );
        let r = ribbon_style(&p, 1.0, 0.5);
        let rs = ribbon_style_shifted(&p, 1.0, 0.5, 0.0);
        assert_eq!((rs.fill, rs.shade), (r.fill, r.shade));
        assert_eq!(ramp_shifted(&p, true, 0.3, 0.0), ramp(&p, true, 0.3));
        assert_eq!(region_tone(&p, 1.0, 0.3, 0.0), p.warm_at(0.3));
        assert_eq!(region_tone(&p, 0.0, 0.3, 0.0), p.cool_at(0.3));
        // A shift moves the petal hue and keeps the centre and ink.
        let a = flower_style(&p, 1.0, 0.5);
        let b = flower_style_shifted(&p, 1.0, 0.5, 25.0);
        assert_ne!(a.petal, b.petal);
        assert_eq!((a.center, a.ink), (b.center, b.ink));
        // Only the hue moves: the tone stays (up to sRGB gamut clipping).
        assert!((b.petal.lightness() - a.petal.lightness()).abs() < 0.05);
    }

    #[test]
    fn shade_paint_is_dark_below_and_light_above() {
        let p = Palette::by_name("nocturne").unwrap();
        for warm in [true, false] {
            let base = ramp(&p, warm, 0.5);
            let Paint::Linear(g) = shade_paint(base, warm) else {
                panic!("shade_paint is linear");
            };
            assert!(g.start.y > g.end.y, "dark end at the bottom");
            let (d, l) = (g.stops[0].1, g.stops[1].1);
            assert!(d.lightness() < base.lightness() && base.lightness() < l.lightness());
            assert_eq!(shade_paint(base, warm), shade_paint(base, warm));
        }
    }
}
