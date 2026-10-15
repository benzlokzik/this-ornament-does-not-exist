//! Demonstration gallery: every motif family and every palette on one sheet.
//!
//! The sheet is a grid of square tiles grouped into sections (ribbons, spirals, flowers,
//! foliage, plants, palettes), each introduced by a small ornamental divider. The column
//! count is the largest that keeps tiles at least [`MIN_TILE`] wide; each section starts a
//! new row and a short last row is centred. Empty sections are left out. The sheet keeps the
//! requested width and grows taller when the content needs it; when it is taller than the
//! content, the content is centred vertically.
//!
//! Every tile has a faint rounded frame and holds its motif centred and scaled so its
//! bounding box fills [`FILL`] of the tile side. Tile groups carry the id
//! `<section>-<item>` and section groups the section name, for inspection of the SVG.

use crate::config::Config;
use crate::geom::validate::path_is_finite;
use crate::motif::flower::{self, FlowerStyle};
use crate::motif::{foliage, plant, ribbon, spiral};
use crate::palette::{self, Palette};
use crate::rng::{Draw, SeedTree};
use crate::scene::{Color, Node, Scene, Style};
use anyhow::{anyhow, ensure};
use kurbo::{Affine, BezPath, Circle, Point, Rect, RoundedRect, Shape};

/// Smallest tile side the column count aims for, in sheet units.
pub const MIN_TILE: f64 = 240.0;
/// Fraction of the tile side that a motif's bounding box is scaled to fill.
pub const FILL: f64 = 0.84;

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

/// Grid metrics derived from the sheet width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    /// Outer margin on every side.
    pub margin: f64,
    /// Gap between tiles, horizontally and vertically.
    pub gap: f64,
    /// Number of tile columns.
    pub columns: usize,
    /// Tile side.
    pub tile: f64,
    /// Height of the band holding a section divider.
    pub divider: f64,
}

impl Layout {
    /// Metrics for a sheet of `width`: as many columns as fit with tiles of at least
    /// [`MIN_TILE`] (at least one column).
    pub fn for_width(width: f64) -> Layout {
        let margin = (width * 0.04).clamp(8.0, 64.0).min(width * 0.1);
        let inner = width - 2.0 * margin;
        let gap = (inner * 0.012).clamp(4.0, 20.0);
        let columns = (((inner + gap) / (MIN_TILE + gap)).floor() as usize).max(1);
        let tile = (inner - (columns - 1) as f64 * gap) / columns as f64;
        let divider = (tile * 0.16).clamp(16.0, 44.0);
        Layout {
            margin,
            gap,
            columns,
            tile,
            divider,
        }
    }

    /// Number of tile rows needed for `items` tiles.
    pub fn rows(&self, items: usize) -> usize {
        items.div_ceil(self.columns)
    }

    /// Height of a section of `items` tiles, divider included.
    fn section_height(&self, items: usize) -> f64 {
        let rows = self.rows(items) as f64;
        self.divider + rows * self.tile + (rows - 1.0).max(0.0) * self.gap
    }

    /// Vertical space between two sections.
    fn section_gap(&self) -> f64 {
        2.0 * self.gap
    }
}

/// Build the gallery sheet described by `opts`.
///
/// Fails on a non-finite or too small (< 16) size and on an unknown palette name.
pub fn build(opts: &GalleryOptions) -> anyhow::Result<Scene> {
    ensure!(
        opts.width.is_finite()
            && opts.height.is_finite()
            && opts.width >= 16.0
            && opts.height >= 16.0,
        "invalid gallery size {}x{} (need finite values >= 16)",
        opts.width,
        opts.height
    );
    let pal = Palette::by_name(&opts.palette).ok_or_else(|| {
        let names: Vec<String> = palette::all().into_iter().map(|p| p.name).collect();
        anyhow!(
            "unknown palette {:?} (available: {})",
            opts.palette,
            names.join(", ")
        )
    })?;

    let layout = Layout::for_width(opts.width);
    let seed = SeedTree::new(opts.seed);
    let accuracy = Config::default().accuracy;
    let size = layout.tile * FILL;
    let sections: Vec<(&str, Vec<(String, Node)>)> = vec![
        (
            "ribbons",
            ribbon::variants(&seed.child("ribbons"), &pal, size, accuracy),
        ),
        (
            "spirals",
            spiral::variants(&seed.child("spirals"), &pal, size, accuracy),
        ),
        (
            "scrolls",
            crate::motif::scroll::variants(&seed.child("scrolls"), &pal, size, accuracy),
        ),
        (
            "flowers",
            flowers(&seed.child("flowers"), &pal, size, accuracy),
        ),
        (
            "foliage",
            foliage::variants(&seed.child("foliage"), &pal, size, accuracy),
        ),
        (
            "plants",
            plant::variants(&seed.child("plants"), &pal, size, accuracy),
        ),
        ("palettes", swatches(size)),
    ];
    let sections: Vec<_> = sections
        .into_iter()
        .filter(|(_, items)| !items.is_empty())
        .collect();

    let content = 2.0 * layout.margin
        + sections
            .iter()
            .map(|(_, items)| layout.section_height(items.len()))
            .sum::<f64>()
        + sections.len().saturating_sub(1) as f64 * layout.section_gap();
    let height = opts.height.max(content);
    let chrome = chrome_color(&pal);

    let mut scene = Scene::new(opts.width, height).with_background(pal.background);
    let mut y = layout.margin + (height - content) / 2.0;
    for (name, items) in sections {
        let h = layout.section_height(items.len());
        scene.push(section(name, items, &layout, y, opts.width, chrome));
        y += h + layout.section_gap();
    }
    Ok(scene)
}

/// One section: a divider band at `top`, then its tiles row by row.
fn section(
    name: &str,
    items: Vec<(String, Node)>,
    layout: &Layout,
    top: f64,
    width: f64,
    chrome: Color,
) -> Node {
    let x0 = layout.margin;
    let x1 = width - layout.margin;
    let mut children = vec![
        divider(x0, x1, top + layout.divider / 2.0, layout.divider, chrome)
            .with_id(format!("{name}-divider")),
    ];
    let n = items.len();
    let pitch = layout.tile + layout.gap;
    for (i, (item, node)) in items.into_iter().enumerate() {
        let (row, col) = (i / layout.columns, i % layout.columns);
        let in_row = (n - row * layout.columns).min(layout.columns);
        let indent = (layout.columns - in_row) as f64 * pitch / 2.0;
        let x = x0 + indent + col as f64 * pitch;
        let y = top + layout.divider + row as f64 * pitch;
        children.push(
            tile(node, layout.tile, chrome)
                .with_id(format!("{name}-{}", slug(&item)))
                .with_transform(Affine::translate((x, y))),
        );
    }
    Node::group(children).with_id(name)
}

/// A tile at the origin: a faint rounded frame of side `side` with `content` fitted inside.
fn tile(content: Node, side: f64, chrome: Color) -> Node {
    let frame = RoundedRect::new(0.0, 0.0, side, side, side * 0.05).to_path(0.05);
    let width = (side * 0.004).max(0.6);
    let mut children = vec![Node::path(
        frame,
        Style::stroke(chrome, width).with_opacity(0.22),
    )];
    if let Some(fitted) = fit(content, side) {
        children.push(fitted);
    }
    Node::group(children)
}

/// Centre `node` in a `side` × `side` square at the origin, scaled so its bounding box
/// fills [`FILL`] of the side. `None` when the node has no finite, non-degenerate extent.
fn fit(node: Node, side: f64) -> Option<Node> {
    let bb = finite_bbox(&node)?;
    let extent = bb.width().max(bb.height());
    if !(extent.is_finite() && extent > 1e-9) {
        return None;
    }
    let scale = FILL * side / extent;
    let t = Affine::translate((side / 2.0, side / 2.0))
        * Affine::scale(scale)
        * Affine::translate(-bb.center().to_vec2());
    Some(Node::group(vec![node]).with_transform(t))
}

/// Bounding box of the finite paths of `node` in its own space (the SVG exporter drops
/// non-finite paths, so they must not distort the fit either).
fn finite_bbox(node: &Node) -> Option<Rect> {
    let mut acc: Option<Rect> = None;
    node.for_each_path(|p, _, t| {
        if p.is_empty() || !path_is_finite(p) {
            return;
        }
        let bb = (t * p.clone()).bounding_box();
        if [bb.x0, bb.y0, bb.x1, bb.y1].iter().all(|v| v.is_finite()) {
            acc = Some(acc.map_or(bb, |a| a.union(bb)));
        }
    });
    acc
}

/// A horizontal ornamental divider from `x0` to `x1` centred on `y`, `size` high: two thin
/// rules ending in dots, around a central lens flanked by two small beads.
fn divider(x0: f64, x1: f64, y: f64, size: f64, color: Color) -> Node {
    let cx = (x0 + x1) / 2.0;
    let w = (size * 0.035).max(0.6);
    let half = size * 0.7;
    let line = Style::stroke(color, w).with_opacity(0.45);
    let dot = Style::fill(color).with_opacity(0.45);
    let r = size * 0.06;
    let mut nodes = Vec::new();
    for (a, b) in [
        (x0 + 2.0 * r, cx - 1.8 * half),
        (cx + 1.8 * half, x1 - 2.0 * r),
    ] {
        if b > a {
            let mut p = BezPath::new();
            p.move_to((a, y));
            p.line_to((b, y));
            nodes.push(Node::path(p, line.clone()));
        }
    }
    for x in [x0 + r, x1 - r, cx - 1.45 * half, cx + 1.45 * half] {
        nodes.push(Node::path(
            Circle::new((x, y), r).to_path(0.01),
            dot.clone(),
        ));
    }
    let lens = lens(Point::new(cx, y), 2.0 * half, size * 0.42);
    nodes.push(Node::path(
        lens.clone(),
        Style::fill(color).with_opacity(0.18),
    ));
    nodes.push(Node::path(lens, line));
    Node::group(nodes)
}

/// A horizontal vesica (pointed oval) centred on `c`.
fn lens(c: Point, length: f64, height: f64) -> BezPath {
    // A cubic whose two control points sit at height `k` peaks at `0.75 k`.
    let (hl, k) = (length / 2.0, height / 2.0 / 0.75);
    let mut p = BezPath::new();
    p.move_to((c.x - hl, c.y));
    p.curve_to(
        (c.x - hl * 0.4, c.y - k),
        (c.x + hl * 0.4, c.y - k),
        (c.x + hl, c.y),
    );
    p.curve_to(
        (c.x + hl * 0.4, c.y + k),
        (c.x - hl * 0.4, c.y + k),
        (c.x - hl, c.y),
    );
    p.close_path();
    p
}

/// Every flower variant, alternating warm and cool petals with seeded tones, built so that
/// curve accuracy is `accuracy` at the displayed size `size`.
fn flowers(seed: &SeedTree, pal: &Palette, size: f64, accuracy: f64) -> Vec<(String, Node)> {
    let mut tones = seed.rng_for("tones");
    flower::variants(seed)
        .into_iter()
        .enumerate()
        .map(|(i, (name, params))| {
            let t = tones.uniform(0.1, 0.9);
            let style = FlowerStyle::from_palette(pal, i % 2 == 0, t);
            let scale = size / (2.0 * params.outer_radius().max(1.0));
            let node = flower::flower(&seed.index(i as u64), &params, &style, accuracy / scale);
            (name, node)
        })
        .collect()
}

/// One swatch per built-in palette, each `size` wide and centred at the origin.
fn swatches(size: f64) -> Vec<(String, Node)> {
    palette::all()
        .into_iter()
        .map(|p| {
            let node = swatch(&p, size);
            (p.name, node)
        })
        .collect()
}

/// A palette swatch: a panel in the palette background with a row of warm petals pointing
/// up and a row of cool petals pointing down, separated by a band of the light colour, all
/// outlined in the palette ink.
fn swatch(p: &Palette, s: f64) -> Node {
    let h = s / 2.0;
    let lw = s * 0.008;
    let mut nodes = vec![Node::path(
        RoundedRect::new(-h, -h, h, h, s * 0.06).to_path(0.05),
        Style::fill_stroke(p.background, p.ink, lw),
    )];
    let span = s * 0.84;
    for (colors, up) in [(&p.warm, true), (&p.cool, false)] {
        let n = colors.len().max(1);
        let pitch = span / n as f64;
        let y = if up { -s * 0.2 } else { s * 0.2 };
        for (i, c) in colors.iter().enumerate() {
            let x = -span / 2.0 + pitch * (i as f64 + 0.5);
            let path = petal(Point::new(x, y), pitch * 0.8, s * 0.3, up);
            nodes.push(Node::path(path, Style::fill_stroke(*c, p.ink, lw * 0.8)));
        }
    }
    nodes.push(Node::path(
        RoundedRect::new(-span / 2.0, -s * 0.03, span / 2.0, s * 0.03, s * 0.03).to_path(0.05),
        Style::fill_stroke(p.light, p.ink, lw * 0.8),
    ));
    Node::group(nodes)
}

/// A teardrop petal centred on `c`, `width` wide and `height` tall, with its tip up (`up`)
/// or down.
fn petal(c: Point, width: f64, height: f64, up: bool) -> BezPath {
    let d = if up { 1.0 } else { -1.0 };
    let (hw, hh) = (width / 2.0 / 0.75, height / 2.0);
    let tip = (c.x, c.y - d * hh);
    let base = (c.x, c.y + d * hh);
    let mut p = BezPath::new();
    p.move_to(tip);
    p.curve_to(
        (c.x + hw * 0.6, c.y - d * hh * 0.3),
        (c.x + hw, c.y + d * hh * 1.05),
        base,
    );
    p.curve_to(
        (c.x - hw, c.y + d * hh * 1.05),
        (c.x - hw * 0.6, c.y - d * hh * 0.3),
        tip,
    );
    p.close_path();
    p
}

/// Frame and divider colour: the ink on light sheets, the light colour on dark ones.
fn chrome_color(p: &Palette) -> Color {
    if p.background.to_hsl().2 >= 0.5 {
        p.ink
    } else {
        p.light
    }
}

/// An SVG-id-friendly version of an item name.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::to_svg_string;

    fn small() -> GalleryOptions {
        GalleryOptions {
            width: 800.0,
            height: 300.0,
            ..Default::default()
        }
    }

    /// The tile groups (children of the sections, dividers excluded) with their ids.
    fn tiles(scene: &Scene) -> Vec<(String, Node)> {
        let mut out = Vec::new();
        let Node::Group { children, .. } = &scene.root else {
            panic!("root is a group");
        };
        for sec in children {
            let Node::Group { children, .. } = sec else {
                panic!("section is a group");
            };
            for t in children {
                if let Node::Group { id: Some(id), .. } = t
                    && !id.ends_with("-divider")
                {
                    out.push((id.clone(), t.clone()));
                }
            }
        }
        out
    }

    #[test]
    fn layout_adapts_to_width() {
        assert_eq!(Layout::for_width(400.0).columns, 1);
        for w in [400.0, 800.0, 1600.0, 3000.0] {
            let l = Layout::for_width(w);
            assert!(l.tile >= MIN_TILE, "width {w}: tile {}", l.tile);
            let used = 2.0 * l.margin + l.columns as f64 * l.tile + (l.columns - 1) as f64 * l.gap;
            assert!((used - w).abs() < 1e-9, "width {w}: used {used}");
        }
        assert!(Layout::for_width(1600.0).columns > Layout::for_width(800.0).columns);
        let tiny = Layout::for_width(16.0);
        assert_eq!(tiny.columns, 1);
        assert!(tiny.tile > 0.0);
    }

    #[test]
    fn scene_is_finite_and_fits_the_sheet() {
        let opts = small();
        let scene = build(&opts).unwrap();
        assert_eq!(scene.width, 800.0);
        assert!(scene.height >= opts.height);
        assert!(scene.root.is_finite());
        assert!(scene.root.path_count() > 0);
        let sheet = scene.canvas();
        let tiles = tiles(&scene);
        assert!(
            tiles.len() >= 6 + palette::all().len(),
            "{} tiles",
            tiles.len()
        );
        assert!(tiles.iter().any(|(id, _)| id == "flowers-daisy"));
        assert!(tiles.iter().any(|(id, _)| id == "palettes-botanica"));
        let side = Layout::for_width(opts.width).tile;
        for (id, t) in &tiles {
            let bb = t.bounding_box().unwrap();
            assert!(
                bb.x0 >= sheet.x0 && bb.y0 >= sheet.y0 && bb.x1 <= sheet.x1 && bb.y1 <= sheet.y1,
                "{id}: {bb:?} outside {sheet:?}"
            );
            assert!(
                bb.width() <= side + 1e-6 && bb.height() <= side + 1e-6,
                "{id}: {bb:?} larger than its tile"
            );
        }
        let mut ids: Vec<_> = tiles.iter().map(|(id, _)| id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), tiles.len(), "tile ids are unique");
    }

    #[test]
    fn deterministic() {
        let a = to_svg_string(&build(&small()).unwrap());
        let b = to_svg_string(&build(&small()).unwrap());
        assert_eq!(a, b);
        let other = GalleryOptions {
            seed: small().seed + 1,
            ..small()
        };
        assert_ne!(a, to_svg_string(&build(&other).unwrap()));
    }

    #[test]
    fn rejects_bad_options() {
        let err = build(&GalleryOptions {
            palette: "nope".into(),
            ..small()
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("nope") && err.contains("botanica"), "{err}");
        for (w, h) in [(f64::NAN, 100.0), (100.0, 0.0), (8.0, 100.0)] {
            let opts = GalleryOptions {
                width: w,
                height: h,
                ..small()
            };
            assert!(build(&opts).is_err(), "{w}x{h}");
        }
    }

    #[test]
    fn tall_sheet_centres_content() {
        let opts = GalleryOptions {
            height: 20000.0,
            ..small()
        };
        let scene = build(&opts).unwrap();
        assert_eq!(scene.height, 20000.0);
        let bb = scene.root.bounding_box().unwrap();
        let (top, bottom) = (bb.y0, scene.height - bb.y1);
        // the divider band's ornament sits slightly below the top of the content box
        assert!((top - bottom).abs() < 20.0, "{top} vs {bottom}");
    }
}
