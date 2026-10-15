//! The moonlit opening: a circular or arched window onto a night landscape — a dark sky
//! built from stacked translucent discs, a moon with a thin halo ring, a scatter of stars
//! and soft hill silhouettes that lighten with distance — clipped to the window and framed
//! by an ink rim and a wreath of small leaves and petals.

use super::field::{Region, along_curve};
use super::layout::{leaf_style, ramp, unit_or};
use crate::geom::Composite;
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::motif::foliage::leaf_motif;
use crate::palette::Palette;
use crate::rng::{Draw, SeedTree};
use crate::scene::{Node, Style};
use kurbo::{BezPath, Circle, Point, Rect, Shape, Vec2};

/// Outline of the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowShape {
    /// A full circle.
    Circle,
    /// A round-headed arch: a half circle on top of a rectangle.
    Arch,
}

/// A window onto the night landscape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opening {
    /// Centre of the circle (of the arch's head for an arch).
    pub center: Point,
    /// Radius of the circle (half-width of an arch).
    pub radius: f64,
    /// Window outline.
    pub shape: WindowShape,
}

impl Opening {
    /// Depth of an arch's straight part below its centre, as a fraction of the radius.
    const ARCH_DROP: f64 = 0.9;

    /// The window outline.
    pub fn outline(&self) -> BezPath {
        let (c, r) = (self.center, self.radius);
        match self.shape {
            WindowShape::Circle => Circle::new(c, r).to_path(1e-3 * r),
            WindowShape::Arch => {
                let mut p = BezPath::new();
                let bottom = c.y + Self::ARCH_DROP * r;
                p.move_to(Point::new(c.x - r, bottom));
                p.line_to(Point::new(c.x - r, c.y));
                // Upper half circle from the left to the right, through the top.
                let k = 0.552_284_75 * r;
                p.curve_to(
                    Point::new(c.x - r, c.y - k),
                    Point::new(c.x - k, c.y - r),
                    Point::new(c.x, c.y - r),
                );
                p.curve_to(
                    Point::new(c.x + k, c.y - r),
                    Point::new(c.x + r, c.y - k),
                    Point::new(c.x + r, c.y),
                );
                p.line_to(Point::new(c.x + r, bottom));
                p.close_path();
                p
            }
        }
    }

    /// Bounding rectangle of the window.
    pub fn bounds(&self) -> Rect {
        let (c, r) = (self.center, self.radius);
        let bottom = match self.shape {
            WindowShape::Circle => c.y + r,
            WindowShape::Arch => c.y + Self::ARCH_DROP * r,
        };
        Rect::new(c.x - r, c.y - r, c.x + r, bottom)
    }

    /// Keep-out region of the window grown by `pad`.
    pub fn region(&self, pad: f64) -> Region {
        match self.shape {
            WindowShape::Circle => Region::Circle {
                center: self.center,
                radius: self.radius + pad,
            },
            WindowShape::Arch => {
                let grown = Opening {
                    radius: self.radius + pad,
                    center: self.center,
                    shape: self.shape,
                };
                let p = grown.outline();
                Region::Path(p)
            }
        }
    }
}

fn disc(c: Point, r: f64) -> BezPath {
    Circle::new(c, r).to_path(1e-3 * r.max(1e-3))
}

/// The opening: the clipped night landscape, then the rim and the wreath on top.
pub fn opening_node(seed: &SeedTree, o: &Opening, palette: &Palette, accuracy: f64) -> Node {
    let mut rng = seed.rng_for("landscape");
    let (c, r) = (o.center, o.radius);
    let b = o.bounds();
    let night = palette.cool_at(0.0).mix(palette.ink, 0.35);
    let glow = palette.cool_at(0.55);
    let moon_color = palette.light.mix(palette.warm_at(1.0), 0.15);
    let mut scene = Node::empty();

    // Sky: a dark fill, then translucent discs stacked around the moon, lighter inwards.
    scene.push(Node::path(b.inset(2.0).to_path(1e-3), Style::fill(night)));
    let side = if rng.chance(0.5) { 1.0 } else { -1.0 };
    let moon = Point::new(
        c.x + side * r * rng.uniform(0.18, 0.34),
        c.y - r * rng.uniform(0.28, 0.42),
    );
    let moon_r = r * rng.uniform(0.12, 0.16);
    let rings = 9;
    for i in 0..rings {
        let f = i as f64 / (rings - 1) as f64;
        let rad = moon_r * (1.4 + (1.0 - f) * 9.0);
        scene.push(Node::path(
            disc(moon, rad),
            Style::fill(glow.with_alpha(0.07 + 0.03 * f)),
        ));
    }

    // Stars above the hills, away from the moon's glow.
    let n_stars = 15 + (rng.uniform(0.0, 26.0) as usize);
    let mut stars = BezPath::new();
    let mut placed = 0;
    for _ in 0..n_stars * 6 {
        if placed >= n_stars {
            break;
        }
        let p = Point::new(rng.uniform(b.x0, b.x1), rng.uniform(b.y0, c.y + 0.15 * r));
        let inside = match o.shape {
            WindowShape::Circle => (p - c).hypot() < 0.94 * r,
            WindowShape::Arch => p.y > c.y || (p - c).hypot() < 0.94 * r,
        };
        if !inside || (p - moon).hypot() < 2.6 * moon_r {
            continue;
        }
        let sr = r * rng.uniform(0.005, 0.013);
        stars.extend(disc(p, sr).iter());
        placed += 1;
    }
    scene.push(Node::path(
        stars,
        Style::fill(palette.light.with_alpha(0.9)),
    ));

    // Moon with a thin halo ring.
    scene.push(Node::path(disc(moon, moon_r), Style::fill(moon_color)));
    scene.push(Node::path(
        disc(moon + Vec2::new(0.3, -0.25) * moon_r, 0.55 * moon_r),
        Style::fill(moon_color.lighten(0.04).with_alpha(0.6)),
    ));
    scene.push(Node::path(
        disc(moon, 1.28 * moon_r),
        Style::stroke(palette.light.with_alpha(0.7), 0.012 * r),
    ));

    // Hills, back to front: higher, lighter and hazier at the back.
    let hills = 2 + usize::from(rng.chance(0.6));
    let bottom = b.y1 + 0.1 * r;
    for i in 0..hills {
        let f = i as f64 / (hills.max(2) - 1) as f64;
        let base = c.y + r * (0.08 + 0.45 * f);
        let amp = r * rng.uniform(0.3, 0.48) * (1.0 - 0.4 * f);
        // One or two rounded humps (Gaussian bumps) per range.
        let humps: Vec<(f64, f64, f64)> = (0..1 + usize::from(rng.chance(0.6)))
            .map(|_| {
                (
                    rng.uniform(0.15, 0.85),
                    rng.uniform(0.13, 0.24),
                    rng.uniform(0.6, 1.0),
                )
            })
            .collect();
        let n = 18;
        let pts: Vec<Point> = (0..=n)
            .map(|k| {
                let u = k as f64 / n as f64;
                let x = b.x0 - 0.1 * r + u * (b.width() + 0.2 * r);
                let bump = humps
                    .iter()
                    .map(|&(m, s, a)| a * (-0.5 * ((u - m) / s).powi(2)).exp())
                    .fold(0.0, f64::max);
                Point::new(x, base - amp * bump)
            })
            .collect();
        let Some(ridge) = Composite::through_points(&pts, 0.6) else {
            continue;
        };
        let mut hill = ridge.to_bezpath_exact();
        hill.line_to(Point::new(b.x1 + 0.1 * r, bottom));
        hill.line_to(Point::new(b.x0 - 0.1 * r, bottom));
        hill.close_path();
        let color = palette
            .cool_at(0.55 - 0.45 * f)
            .mix(night, 0.25 + 0.35 * f)
            .mix(glow, 0.25 * (1.0 - f));
        scene.push(Node::path(hill, Style::fill(color)));
    }

    let outline = o.outline();
    let mut root = Node::empty().with_id("moonlit-opening");
    root.push(scene.clipped(outline.clone()));
    // Rim: a light inner line under a dark ink line.
    root.push(Node::path(
        outline.clone(),
        Style::stroke(palette.light.with_alpha(0.8), 0.03 * r),
    ));
    root.push(Node::path(
        outline.clone(),
        Style::stroke(palette.ink, 0.012 * r),
    ));
    root.push(wreath(seed, &outline, r, palette, accuracy));
    root
}

/// Small leaves and a few warm petals around the rim, leaning along it.
fn wreath(seed: &SeedTree, outline: &BezPath, r: f64, palette: &Palette, accuracy: f64) -> Node {
    let Some(rim) = Composite::from_bezpath(outline) else {
        return Node::empty();
    };
    let mut rng = seed.rng_for("wreath");
    let center = outline.bounding_box().center();
    let spacing = r * rng.uniform(0.17, 0.22);
    let mut leaves = Node::empty();
    let mut petals = Node::empty();
    let flip = if rng.chance(0.5) { 1.0 } else { -1.0 };
    for (i, (_, p, tg)) in along_curve(&rim, spacing, 0.3, seed.child("rim"))
        .into_iter()
        .enumerate()
    {
        let out = unit_or(p - center, Vec2::new(0.0, -1.0));
        let along = tg * flip;
        let petal = i % 4 == 2;
        let k = seed.index(i as u64);
        let mut lr = k.rng();
        let len = r * lr.uniform(0.2, 0.3) * if petal { 0.8 } else { 1.0 };
        let d = unit_or(along * 0.8 + out * lr.uniform(0.25, 0.6), out);
        let base = p - out * (0.02 * r);
        if petal {
            let params = LeafParams::new(LeafShape::Lanceolate)
                .with_tip_curl(-0.3 * flip)
                .jittered(&mut lr, 0.5);
            let leaf = Leaf::from_base(base, d, len, 0.36 * len, params);
            let mut style = leaf_style(palette, 1.0, lr.uniform(0.45, 0.7));
            style.fill = ramp(palette, true, lr.uniform(0.35, 0.55));
            petals.push(leaf_motif(&k, &leaf, &style, accuracy));
        } else {
            let shape = *lr.pick(&[
                LeafShape::Ovate,
                LeafShape::Lanceolate,
                LeafShape::Serrated { teeth: 9 },
            ]);
            let params = LeafParams::new(shape)
                .with_tip_curl(0.35 * flip)
                .jittered(&mut lr, 0.6);
            let leaf = Leaf::from_base(base, d, len, 0.4 * len, params);
            let style = leaf_style(palette, 0.0, lr.uniform(0.2, 0.6));
            leaves.push(leaf_motif(&k, &leaf, &style, accuracy));
        }
    }
    Node::group(vec![leaves, petals])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_is_finite_and_inside_its_frame() {
        let p = Palette::by_name("botanica").unwrap();
        for shape in [WindowShape::Circle, WindowShape::Arch] {
            let o = Opening {
                center: Point::new(300.0, 300.0),
                radius: 120.0,
                shape,
            };
            let n = opening_node(&SeedTree::new(5), &o, &p, 0.1);
            assert!(n.is_finite());
            assert!(n.path_count() > 20);
            // The landscape is clipped to the window; rim and wreath stay close around it.
            let Node::Group { children, .. } = &n else {
                panic!("opening is a group");
            };
            let frame = o.bounds().inset(0.45 * o.radius);
            for c in &children[1..] {
                let bb = c.bounding_box().unwrap();
                assert!(frame.contains(bb.origin()) && frame.contains(Point::new(bb.x1, bb.y1)));
            }
            assert!(o.region(10.0).contains(Point::new(300.0, 300.0)));
        }
    }
}
