//! The moonlit opening: a circular or arched window onto a night landscape — a dark sky
//! built from stacked translucent discs, a moon with a thin halo ring, a scatter of stars
//! and soft hill silhouettes that lighten with distance — clipped to the window and framed
//! by an ink rim and a wreath of small leaves and petals.

use super::field::{Region, along_curve};
use super::layout::{Canvas, leaf_style, ramp, unit_or};
use crate::geom::leaf::{Leaf, LeafParams, LeafShape};
use crate::geom::{Composite, lerp, smoothstep};
use crate::motif::foliage::leaf_motif;
use crate::palette::Palette;
use crate::rng::{Draw, SeedTree};
use crate::scene::{Color, Node, Style};
use kurbo::{Affine, Arc, BezPath, Circle, Point, Rect, Shape, Vec2};

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

// ---------------------------------------------------------------------------------------
// Open landscape
// ---------------------------------------------------------------------------------------

/// The open landscape, painted without a frame in the warm-left frame of `canvas`:
///
/// * a night **sky** over the cool region: stacked translucent bands of the darkest cool
///   tone, each reaching less far down and starting further from the divider, so the sky is
///   darkest at the top and fades downwards and towards the warm side; a mauve glow over
///   the horizon; stars (dots and a few four-point sparkles) denser away from the moon;
/// * a **moon** with a thin rim and a soft halo of concentric translucent discs;
/// * engraved **cloud banks**: rows of rounded lobes drawn back to front, each outlined and
///   hatched with fine lines along its lower part;
/// * layered **hills**, far (light, dusty violet) to near (deeper mauve), each hatched on
///   the slopes facing away from the moon, the nearest one fringed with grass;
/// * on the warm side, a cluster of faceted **crystal** prisms fading into mist at the foot;
/// * a few shaded **orbs** floating over the upper third of both regions.
///
/// One node in canvas coordinates; `fill` (about 0.5–2) scales the number of stars,
/// clouds and hatch lines.
pub fn open_landscape(
    seed: &SeedTree,
    canvas: &Canvas,
    palette: &Palette,
    fill: f64,
    accuracy: f64,
) -> Node {
    let f = Frame::new(canvas, fill, accuracy);
    let tones = Tones::new(palette);
    let mut rng = seed.rng_for("open");
    let (w, h, u) = (f.w, f.h, f.u);

    // Vertical plan: moon high up, clouds below it, hills from mid-height down.
    let moon_r = u * rng.uniform(0.04, 0.055);
    let moon_y = (h * rng.uniform(0.07, 0.15)).max(3.2 * moon_r);
    let moon_x = {
        let lo = canvas.divider_x(moon_y) + 3.0 * moon_r + 0.04 * u;
        let hi = w - 2.0 * moon_r - 0.04 * u;
        fit(lerp(lo, hi, rng.uniform(0.35, 0.85)), lo, hi)
    };
    let moon = Point::new(moon_x, moon_y);
    let far_base = h * rng.uniform(0.5, 0.54);
    let near_base = h * rng.uniform(0.79, 0.83);
    let cloud_top = (moon_y + 2.2 * moon_r).max(0.2 * h);
    let cloud_bottom = (far_base - 0.02 * h).max(cloud_top + 0.05 * h);

    let cool_clip = f.cool_band(-f.overlap, -f.bleed, &|_| h + f.bleed);
    let mut root = Node::empty().with_id("open-landscape");
    root.push(sky(&f, &seed.child("sky"), &tones, far_base).with_id("sky"));
    root.push(stars(&f, &seed.child("stars"), &tones, moon, moon_r, cloud_top).with_id("stars"));
    root.push(moon_node(&f, &tones, moon, moon_r).with_id("moon"));
    root.push(
        clouds(&f, &seed.child("clouds"), &tones, cloud_top, cloud_bottom)
            .clipped(cool_clip.clone())
            .with_id("clouds"),
    );
    root.push(
        hills(&f, &seed.child("hills"), &tones, moon, far_base, near_base)
            .clipped(cool_clip)
            .with_id("hills"),
    );
    if rng.chance(0.85) {
        let warm_clip = f.warm_band(f.overlap);
        root.push(
            crystals(&f, &seed.child("crystals"), &tones)
                .clipped(warm_clip)
                .with_id("crystals"),
        );
    }
    root.push(orbs(&f, &seed.child("orbs"), &tones, moon, moon_r).with_id("orbs"));
    root
}

/// Measures shared by the parts of the open landscape.
struct Frame<'a> {
    canvas: &'a Canvas,
    w: f64,
    h: f64,
    u: f64,
    /// How far shapes may run past the canvas edges.
    bleed: f64,
    /// How far the cool side reaches past the divider (the ornament hides the seam).
    overlap: f64,
    accuracy: f64,
    fill: f64,
}

impl Frame<'_> {
    fn new(canvas: &Canvas, fill: f64, accuracy: f64) -> Frame<'_> {
        let u = canvas.unit();
        Frame {
            canvas,
            w: canvas.width,
            h: canvas.height,
            u,
            bleed: 0.01 * u,
            overlap: 0.02 * u,
            accuracy: if accuracy.is_finite() && accuracy > 0.0 {
                accuracy
            } else {
                0.05
            },
            fill: if fill.is_finite() {
                fill.clamp(0.25, 3.0)
            } else {
                1.0
            },
        }
    }

    /// `n` scaled by the fill, at least `min`.
    fn count(&self, n: f64, min: usize) -> usize {
        ((n * self.fill).round() as usize).max(min)
    }

    /// Flattening tolerance for a circle of radius `r`.
    fn tol(&self, r: f64) -> f64 {
        (0.5 * self.accuracy).min(0.05 * r).max(1e-4)
    }

    fn disc(&self, c: Point, r: f64) -> BezPath {
        Circle::new(c, r).to_path(self.tol(r))
    }

    /// The part of the cool side between `top` and the curve `bottom(x)`, its left edge
    /// the divider moved by `off` towards the cool side, bounded by the canvas bleed.
    fn cool_band(&self, off: f64, top: f64, bottom: &dyn Fn(f64) -> f64) -> BezPath {
        let right = self.w + self.bleed;
        let left =
            |y: f64| (self.canvas.divider_x(y) + off).clamp(-self.bleed, right - 1e-3 * self.u);
        let n = 48;
        let mut path = BezPath::new();
        path.move_to(Point::new(left(top), top));
        path.line_to(Point::new(right, top));
        let xl = left(bottom(0.5 * (left(top) + right)).max(top));
        for i in 0..=n {
            let x = right + (xl - right) * i as f64 / n as f64;
            path.line_to(Point::new(x, bottom(x).max(top)));
        }
        let y0 = bottom(xl).max(top);
        for i in 1..n {
            let y = y0 + (top - y0) * i as f64 / n as f64;
            path.line_to(Point::new(left(y), y));
        }
        path.close_path();
        path
    }

    /// The warm side with the divider moved by `off` into the cool side.
    fn warm_band(&self, off: f64) -> BezPath {
        let (top, bottom) = (-self.bleed, self.h + self.bleed);
        let n = 64;
        let mut path = BezPath::new();
        path.move_to(Point::new(-self.bleed, top));
        for i in 0..=n {
            let y = top + (bottom - top) * i as f64 / n as f64;
            let x = (self.canvas.divider_x(y) + off).clamp(0.0, self.w + self.bleed);
            path.line_to(Point::new(x, y));
        }
        path.line_to(Point::new(-self.bleed, bottom));
        path.close_path();
        path
    }

    /// Signed distance from the divider (positive on the cool side).
    fn side(&self, p: Point) -> f64 {
        self.canvas.side(p)
    }
}

/// `x` clamped to `[lo, hi]`, or the midpoint when the range is empty.
fn fit(x: f64, lo: f64, hi: f64) -> f64 {
    if lo > hi {
        0.5 * (lo + hi)
    } else {
        x.clamp(lo, hi)
    }
}

/// The colours of the open landscape, derived from the palette.
struct Tones {
    night: Color,
    dusk: Color,
    star: Color,
    moon: Color,
    halo: Color,
    cloud: Color,
    cloud_line: Color,
    hill_far: Color,
    hill_near: Color,
    crystal: Color,
    crystal_ink: Color,
    mist: Color,
    orb_cool: Color,
    orb_warm: Color,
    orb_ink: Color,
}

impl Tones {
    fn new(p: &Palette) -> Tones {
        let night = p.cool_at(0.0).lighten(-0.03);
        let dusk = p.cool_at(0.35).mix(p.warm_at(0.55), 0.45);
        let crystal = p
            .cool_at(0.4)
            .mix(p.warm_at(0.55), 0.35)
            .saturate(0.85)
            .mix(p.light, 0.35);
        Tones {
            night,
            dusk,
            star: p.light,
            moon: p.light.mix(p.warm_at(1.0), 0.18),
            halo: p.light.mix(p.cool_at(1.0), 0.4),
            cloud: night.mix(dusk, 0.55).mix(p.light, 0.1),
            cloud_line: p.light.mix(p.cool_at(0.8), 0.3),
            hill_far: dusk.mix(p.warm_at(0.75), 0.25).mix(p.light, 0.18),
            hill_near: p.warm_at(0.3).mix(p.cool_at(0.15), 0.45).saturate(0.8),
            crystal,
            crystal_ink: p.ink.mix(crystal, 0.35),
            mist: p.background,
            orb_cool: p.cool_at(0.55).mix(p.light, 0.3).saturate(0.6),
            orb_warm: p.warm_at(0.7).mix(p.light, 0.35).saturate(0.7),
            orb_ink: p.ink,
        }
    }
}

/// Darkest at the top, fading downwards and towards the divider, with a dusky glow over
/// the horizon at `horizon`.
fn sky(f: &Frame, seed: &SeedTree, t: &Tones, horizon: f64) -> Node {
    let mut rng = seed.rng();
    let mut g = Node::empty();
    let bottom = horizon + 0.06 * f.h;
    let n = 14;
    let fade = 0.15 * f.u;
    for k in 0..n {
        let s = k as f64 / (n - 1) as f64;
        let off = -f.overlap + s * fade;
        let base = lerp(bottom, 0.06 * f.h, s.powf(1.4));
        let (amp, freq, phase) = (
            0.012 * f.h,
            rng.uniform(1.0, 2.2),
            rng.uniform(0.0, std::f64::consts::TAU),
        );
        let w = f.w;
        let edge = move |x: f64| base + amp * (phase + freq * std::f64::consts::TAU * x / w).sin();
        g.push(Node::path(
            f.cool_band(off, -f.bleed, &edge),
            Style::fill(t.night.with_alpha(0.115)),
        ));
    }
    // Glow over the horizon.
    for k in 0..4 {
        let top = horizon - (0.03 + 0.035 * k as f64) * f.h;
        g.push(Node::path(
            f.cool_band(0.0, top, &|_| bottom),
            Style::fill(t.dusk.with_alpha(0.09)),
        ));
    }
    g
}

/// A four-point sparkle of radius `r`.
fn sparkle(c: Point, r: f64) -> BezPath {
    let k = 0.14 * r;
    let mut p = BezPath::new();
    p.move_to(c + Vec2::new(0.0, -r));
    p.quad_to(c + Vec2::new(k, -k), c + Vec2::new(r, 0.0));
    p.quad_to(c + Vec2::new(k, k), c + Vec2::new(0.0, r));
    p.quad_to(c + Vec2::new(-k, k), c + Vec2::new(-r, 0.0));
    p.quad_to(c + Vec2::new(-k, -k), c + Vec2::new(0.0, -r));
    p.close_path();
    p
}

/// Stars over the cool sky above `bottom`, sparse near the moon and low down.
fn stars(f: &Frame, seed: &SeedTree, t: &Tones, moon: Point, moon_r: f64, bottom: f64) -> Node {
    let mut rng = seed.rng();
    let n = f.count(rng.uniform(20.0, 45.0), 8);
    let n_sparkles = 2 + (rng.uniform(0.0, 4.0) as usize).min(3);
    let mut dots = BezPath::new();
    let mut glints = BezPath::new();
    let mut placed = 0;
    for _ in 0..n * 12 {
        if placed >= n {
            break;
        }
        let y = rng.uniform(0.01 * f.h, bottom);
        let p = Point::new(rng.uniform(0.0, f.w), y);
        let keep = smoothstep(((p - moon).hypot() - 2.4 * moon_r) / (5.0 * moon_r))
            * (1.0 - 0.7 * y / bottom.max(1e-9))
            * smoothstep(f.side(p) / (0.12 * f.u));
        if f.side(p) < 0.02 * f.u || !rng.chance(keep) {
            continue;
        }
        if placed < n_sparkles {
            glints.extend(sparkle(p, f.u * rng.uniform(0.008, 0.014)).iter());
        } else {
            let r = f.u * rng.uniform(0.0015, 0.0035);
            dots.extend(f.disc(p, r).iter());
        }
        placed += 1;
    }
    let mut g = Node::empty();
    if !dots.is_empty() {
        g.push(Node::path(dots, Style::fill(t.star.with_alpha(0.85))));
    }
    if !glints.is_empty() {
        g.push(Node::path(glints, Style::fill(t.star.with_alpha(0.95))));
    }
    g
}

/// The moon: a soft halo of concentric discs, the disc, a lighter cap and a thin rim.
fn moon_node(f: &Frame, t: &Tones, c: Point, r: f64) -> Node {
    let mut g = Node::empty();
    for (k, a) in [(3.4, 0.06), (2.4, 0.08), (1.6, 0.12)] {
        g.push(Node::path(
            f.disc(c, k * r),
            Style::fill(t.halo.with_alpha(a)),
        ));
    }
    g.push(Node::path(f.disc(c, r), Style::fill(t.moon)));
    g.push(Node::path(
        f.disc(c + Vec2::new(0.22, -0.2) * r, 0.62 * r),
        Style::fill(t.star.with_alpha(0.45)),
    ));
    g.push(Node::path(
        f.disc(c, 1.12 * r),
        Style::stroke(t.star.with_alpha(0.75), 0.006 * f.u),
    ));
    g
}

/// The part of the circle `(c, r)` above the horizontal line `y = base` as a closed
/// lobe, and its arc alone (for the outline).
fn lobe(f: &Frame, c: Point, r: f64, base: f64) -> (BezPath, BezPath) {
    let s = ((base - c.y) / r).clamp(-0.95, 0.95);
    let a = s.asin();
    let arc = Arc::new(
        c,
        Vec2::new(r, r),
        std::f64::consts::PI - a,
        std::f64::consts::PI + 2.0 * a,
        0.0,
    );
    let outline = arc.to_path(f.tol(r));
    let mut closed = outline.clone();
    closed.close_path();
    (closed, outline)
}

/// Two to four banks of engraved clouds between `top` and `bottom`.
fn clouds(f: &Frame, seed: &SeedTree, t: &Tones, top: f64, bottom: f64) -> Node {
    let mut rng = seed.rng();
    let n = f.count(2.0 + rng.uniform(0.0, 2.99).floor(), 1).min(6);
    let mut g = Node::empty();
    for i in 0..n {
        let k = seed.index(i as u64);
        let mut r = k.rng();
        let y = lerp(top, bottom, (i as f64 + r.uniform(0.2, 0.8)) / n as f64);
        let left = f.canvas.divider_x(y) - f.overlap;
        let span = (f.w - left).max(0.05 * f.u);
        let width = span * r.uniform(0.55, 0.9);
        let cx = fit(r.uniform(left + 0.3 * width, f.w - 0.25 * width), left, f.w);
        // Lower banks are nearer: larger lobes, a little lighter.
        let depth = (y - top) / (bottom - top).max(1e-9);
        let lobe_r = f.u * lerp(0.035, 0.06, depth) * r.uniform(0.85, 1.15);
        let fill = t.cloud.mix(t.dusk, 0.15 + 0.3 * depth);
        g.push(cloud_bank(f, &k, t, fill, cx, y, width, lobe_r));
    }
    g
}

/// One bank: a back row and a front row of lobes, each filled, outlined and hatched.
#[allow(clippy::too_many_arguments)]
fn cloud_bank(
    f: &Frame,
    seed: &SeedTree,
    t: &Tones,
    fill: Color,
    cx: f64,
    base: f64,
    width: f64,
    lobe_r: f64,
) -> Node {
    let mut rng = seed.rng_for("lobes");
    let mut g = Node::empty();
    let spacing = 0.0065 * f.u / f.fill.sqrt();
    let line_w = 0.0016 * f.u;
    let rows = [(1.0, 0.7, 0.55), (0.5, 0.85, 0.8), (0.0, 1.0, 1.0)];
    for (row, (rise, shrink, cover)) in rows.iter().enumerate() {
        let row_base = base - rise * lobe_r;
        let row_w = width * cover;
        let n = ((row_w / (1.25 * lobe_r)).round() as usize).clamp(2, 14);
        let tone = fill.lighten(0.025 * row as f64);
        for j in 0..n {
            let s = (j as f64 + 0.5) / n as f64;
            let bell = (std::f64::consts::PI * s).sin();
            let r = lobe_r * shrink * (0.6 + 0.55 * bell) * rng.uniform(0.85, 1.15);
            let x = (cx + (s - 0.5) * row_w + rng.jitter(0.0, 0.2 * lobe_r))
                .clamp(0.2 * r, (f.w - 0.2 * r).max(0.2 * r));
            let c = Point::new(x, row_base - r * rng.uniform(0.25, 0.5));
            let (shape, arc) = lobe(f, c, r, row_base);
            g.push(Node::path(shape.clone(), Style::fill(tone)));
            // Hatching along the lower part of the lobe, curving up at the ends.
            let mut hatch = BezPath::new();
            let mut y = row_base - 0.6 * spacing;
            while y > c.y - 0.15 * r {
                let half = (r * r - (y - c.y).powi(2)).max(0.0).sqrt() * 0.92;
                let a = rng.uniform(0.0, 0.35);
                let x0 = c.x - half + a * half;
                let x1 = c.x + half * rng.uniform(0.75, 1.0);
                if x1 - x0 > 2.0 * spacing {
                    hatch.move_to(Point::new(x0, y - 0.4 * spacing));
                    hatch.quad_to(
                        Point::new(0.5 * (x0 + x1), y + 0.4 * spacing),
                        Point::new(x1, y - 0.4 * spacing),
                    );
                }
                y -= spacing;
            }
            if !hatch.is_empty() {
                g.push(
                    Node::path(
                        hatch,
                        Style::stroke(t.cloud_line.with_alpha(0.3), 0.6 * line_w),
                    )
                    .clipped(shape),
                );
            }
            g.push(Node::path(
                arc,
                Style::stroke(t.cloud_line.with_alpha(0.6), line_w),
            ));
        }
    }
    g
}

/// The profile of one hill: a base height, rounded humps and a tilt.
struct Ridge {
    x0: f64,
    x1: f64,
    base: f64,
    amp: f64,
    tilt: f64,
    humps: Vec<(f64, f64, f64)>,
}

impl Ridge {
    fn y(&self, x: f64) -> f64 {
        let s = (x - self.x0) / (self.x1 - self.x0).max(1e-9);
        let bump = self
            .humps
            .iter()
            .map(|&(m, w, a)| a * (-0.5 * ((s - m) / w).powi(2)).exp())
            .fold(0.0, f64::max);
        self.base - self.amp * bump + self.tilt * (s - 0.5)
    }

    fn points(&self, n: usize) -> Vec<Point> {
        (0..=n)
            .map(|i| {
                let x = lerp(self.x0, self.x1, i as f64 / n as f64);
                Point::new(x, self.y(x))
            })
            .collect()
    }
}

/// Three to five hills from `far` down to `near`, back to front, the nearest fringed with
/// grass.
fn hills(f: &Frame, seed: &SeedTree, t: &Tones, moon: Point, far: f64, near: f64) -> Node {
    let mut rng = seed.rng();
    let n = 3 + (rng.uniform(0.0, 3.0) as usize).min(2);
    let flip = usize::from(rng.chance(0.5));
    let x0 = (0..=16)
        .map(|i| {
            f.canvas
                .divider_x(lerp(far - 0.15 * f.h, f.h, i as f64 / 16.0))
        })
        .fold(f64::MAX, f64::min)
        .max(0.0)
        - f.overlap
        - f.bleed;
    let x1 = f.w + f.bleed;
    let mut g = Node::empty();
    let line_w = 0.0014 * f.u;
    for i in 0..n {
        let s = i as f64 / (n - 1) as f64;
        // One broad hump per hill, alternating sides so the ranges overlap in layers,
        // sometimes with a smaller shoulder.
        let (lo, hi) = if (i + flip) % 2 == 0 {
            (0.55, 1.05)
        } else {
            (-0.05, 0.45)
        };
        let mut humps = vec![(rng.uniform(lo, hi), rng.uniform(0.22, 0.4), 1.0)];
        if rng.chance(0.4) {
            humps.push((
                rng.uniform(0.2, 0.8),
                rng.uniform(0.1, 0.18),
                rng.uniform(0.4, 0.7),
            ));
        }
        let ridge = Ridge {
            x0,
            x1,
            base: lerp(far, near, s) + rng.jitter(0.0, 0.015 * f.h),
            amp: f.h * rng.uniform(0.09, 0.15) * (1.0 - 0.4 * s),
            tilt: rng.jitter(0.0, 0.06 * f.h),
            humps,
        };
        let pts = ridge.points(24);
        let Some(edge) = Composite::through_points(&pts, 0.6) else {
            continue;
        };
        let top = edge.to_bezpath_exact();
        let mut shape = top.clone();
        shape.line_to(Point::new(x1, f.h + f.bleed));
        shape.line_to(Point::new(x0, f.h + f.bleed));
        shape.close_path();
        let color = t.hill_far.mix(t.hill_near, s);
        // Translucent copies starting further from the divider: the hill fades out
        // towards the warm side.
        for (off, a) in [
            (-f.overlap, 0.2),
            (0.02, 0.25),
            (0.04, 0.3),
            (0.06, 0.35),
            (0.08, 0.45),
            (0.11, 0.8),
        ] {
            let off = if off < 0.0 { off } else { off * f.u };
            let band = f.cool_band(off, -f.bleed, &|_| f.h + f.bleed);
            g.push(Node::path(shape.clone(), Style::fill(color.with_alpha(a))).clipped(band));
        }

        // Hatching parallel to the ridge on the slopes turned away from the moon, steeper
        // slopes taking more lines.
        let k = seed.index(i as u64);
        let mut hr = k.rng();
        let lines = f.count(6.0, 2).min(14);
        let sp = 0.0075 * f.u;
        let samples = 120;
        let dx = (x1 - x0) / samples as f64;
        let mut hatch = BezPath::new();
        for j in 1..=lines {
            let need = 0.04 + 0.03 * j as f64;
            let mut run: Vec<Point> = Vec::new();
            for q in 0..=samples {
                let x = x0 + q as f64 * dx;
                let slope = (ridge.y(x + 0.5 * dx) - ridge.y(x - 0.5 * dx)) / dx;
                let away = -slope * (moon.x - x).signum();
                let p = Point::new(x, ridge.y(x) + j as f64 * sp * (1.0 + 0.5 * away.max(0.0)));
                if away > need && !hr.chance(0.06) {
                    run.push(p);
                } else {
                    flush_run(&mut hatch, &mut run);
                }
            }
            flush_run(&mut hatch, &mut run);
        }
        if !hatch.is_empty() {
            let ink = color.lighten(-0.18).with_alpha(0.75);
            g.push(Node::path(hatch, Style::stroke(ink, line_w)).clipped(shape.clone()));
        }
        g.push(Node::path(
            top,
            Style::stroke(color.lighten(-0.12).with_alpha(0.8), line_w),
        ));
        if i == n - 1 {
            g.push(grass(f, &k.child("grass"), color.lighten(-0.16), t, &ridge));
        }
    }
    g
}

/// Append `run` as a polyline to `path` if it is long enough, and clear it.
fn flush_run(path: &mut BezPath, run: &mut Vec<Point>) {
    if run.len() >= 4 {
        path.move_to(run[0]);
        for p in &run[1..] {
            path.line_to(*p);
        }
    }
    run.clear();
}

/// Grass blades, small leaf sprigs and a few seed-head stalks along a ridge.
fn grass(f: &Frame, seed: &SeedTree, color: Color, t: &Tones, ridge: &Ridge) -> Node {
    let mut rng = seed.rng();
    let span = ridge.x1 - ridge.x0;
    let n = f.count(span / (0.007 * f.u), 6).min(400);
    let mut blades = BezPath::new();
    let mut stalks = BezPath::new();
    let mut heads = BezPath::new();
    for _ in 0..n {
        let x = rng.uniform(ridge.x0, ridge.x1);
        let base = Point::new(x, ridge.y(x) + 0.003 * f.u);
        let lean = rng.jitter(0.0, 0.35);
        let up = Vec2::new(lean.sin(), -lean.cos());
        let across = Vec2::new(-up.y, up.x);
        if rng.chance(0.06) {
            // A thin stalk with a round seed head.
            let len = f.u * rng.uniform(0.03, 0.055);
            let tip = base + up * len;
            stalks.move_to(base);
            stalks.quad_to(base + up * (0.5 * len) + across * (0.1 * len), tip);
            heads.extend(f.disc(tip, f.u * rng.uniform(0.003, 0.005)).iter());
        } else if rng.chance(0.15) {
            // A leaf: two arcs meeting at the tip.
            let len = f.u * rng.uniform(0.012, 0.022);
            let wid = 0.35 * len;
            let tip = base + up * len;
            let mid = base + up * (0.5 * len);
            blades.move_to(base);
            blades.quad_to(mid + across * wid, tip);
            blades.quad_to(mid - across * wid, base);
            blades.close_path();
        } else {
            let len = f.u * rng.uniform(0.008, 0.026);
            let bw = 0.0022 * f.u;
            let bend = across * (rng.jitter(0.0, 0.25) * len);
            let tip = base + up * len + bend;
            blades.move_to(base - across * bw);
            blades.quad_to(base + up * (0.55 * len) + bend * 0.3, tip);
            blades.quad_to(base + up * (0.45 * len) + bend * 0.3, base + across * bw);
            blades.close_path();
        }
    }
    let mut g = Node::empty();
    g.push(Node::path(blades, Style::fill(color)));
    if !stalks.is_empty() {
        g.push(Node::path(stalks, Style::stroke(color, 0.0018 * f.u)));
        g.push(Node::path(
            heads,
            Style::fill(t.cloud_line.with_alpha(0.85)),
        ));
    }
    g
}

/// A cluster of five to twelve faceted prisms rising along the left of the warm region,
/// tallest at the back, each leaning slightly, with a mist over their feet.
fn crystals(f: &Frame, seed: &SeedTree, t: &Tones) -> Node {
    let mut rng = seed.rng();
    let n = 7 + (rng.uniform(0.0, 6.0) as usize).min(5);
    let foot = 0.62 * f.h;
    let x_right = (0.62 * f.canvas.divider_x(0.4 * f.h)).max(0.15 * f.w);
    let mut prisms: Vec<(f64, f64, f64, f64, f64)> = (0..n)
        .map(|i| {
            let x = lerp(
                0.03 * f.w,
                x_right,
                (i as f64 + rng.uniform(0.1, 0.9)) / n as f64,
            );
            // Taller to the left (the outer edge), lower towards the centre.
            let reach = (x - 0.03 * f.w) / (x_right - 0.03 * f.w).max(1e-9);
            let top = f.h * (rng.uniform(0.18, 0.3) + 0.14 * reach);
            let base = foot + rng.jitter(0.0, 0.03 * f.h);
            let width = f.u * rng.uniform(0.028, 0.055);
            // Lean a little, the tip moving no more than a few widths.
            let height = base - top;
            let lean =
                rng.jitter(0.0, 0.07).clamp(-1.0, 1.0) * (0.04 * f.u / height).atan().min(1.0);
            (x, base, height, width, lean)
        })
        .collect();
    // Back to front: tallest first.
    prisms.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut g = Node::empty();
    for (i, &(x, base, height, width, lean)) in prisms.iter().enumerate() {
        let depth = i as f64 / (n - 1).max(1) as f64;
        let tone = t.crystal.mix(t.mist, 0.3 * (1.0 - depth));
        let k = seed.index(i as u64);
        g.push(prism(
            f,
            &k,
            t,
            tone,
            Point::new(x, base),
            height,
            width,
            lean,
        ));
    }
    g
}

/// One prism standing at `base`: three visible side facets and a pointed top, each a
/// different tone, outlined, with a few hatch lines along the shaded facet.
#[allow(clippy::too_many_arguments)]
fn prism(
    f: &Frame,
    seed: &SeedTree,
    t: &Tones,
    tone: Color,
    base: Point,
    height: f64,
    width: f64,
    lean: f64,
) -> Node {
    let mut rng = seed.rng();
    let hw = 0.5 * width;
    let e = [
        -hw,
        -hw + width * rng.uniform(0.18, 0.3),
        hw - width * rng.uniform(0.15, 0.28),
        hw,
    ];
    let apex = Point::new(lerp(e[1], e[2], rng.uniform(0.3, 0.7)), -height);
    let drop = |k: f64| -height + width * k;
    let s = [
        drop(rng.uniform(0.45, 0.8)),
        drop(rng.uniform(0.2, 0.4)),
        drop(rng.uniform(0.2, 0.4)),
        drop(rng.uniform(0.45, 0.8)),
    ];
    let to_canvas = Affine::translate(base.to_vec2()) * Affine::rotate(lean);
    let poly = |pts: &[Point]| {
        let mut p = BezPath::new();
        p.move_to(to_canvas * pts[0]);
        for q in &pts[1..] {
            p.line_to(to_canvas * *q);
        }
        p.close_path();
        p
    };
    let pt = Point::new;
    // Lit from the cool side (the moon): the right facets are lighter.
    let shades = [-0.12, 0.0, 0.08];
    let line_w = 0.0014 * f.u;
    let mut g = Node::empty();
    for k in 0..3 {
        let side = poly(&[
            pt(e[k], 0.0),
            pt(e[k], s[k]),
            pt(e[k + 1], s[k + 1]),
            pt(e[k + 1], 0.0),
        ]);
        let cap = poly(&[pt(e[k], s[k]), apex, pt(e[k + 1], s[k + 1])]);
        g.push(Node::path(
            side,
            Style::fill_stroke(tone.lighten(shades[k]), t.crystal_ink, line_w),
        ));
        g.push(Node::path(
            cap,
            Style::fill_stroke(tone.lighten(shades[k] + 0.06), t.crystal_ink, line_w),
        ));
    }
    // Vertical hatching on the shaded facet and a few lines on the front.
    let mut hatch = BezPath::new();
    let lines = f.count(3.0, 1).min(8);
    for j in 0..lines {
        let a = (j as f64 + 0.5) / lines as f64;
        let x = lerp(e[0], e[1], a);
        let y_top = lerp(s[0], s[1], a) + 0.02 * height;
        hatch.move_to(to_canvas * pt(x, 0.0));
        hatch.line_to(to_canvas * pt(x, y_top + rng.uniform(0.0, 0.3) * height));
    }
    for _ in 0..f.count(1.5, 1).min(4) {
        let a = rng.uniform(0.1, 0.9);
        let x = lerp(e[1], e[2], a);
        let y_top = lerp(s[1], s[2], a);
        hatch.move_to(to_canvas * pt(x, 0.0));
        hatch.line_to(to_canvas * pt(x, lerp(y_top, 0.0, rng.uniform(0.3, 0.6))));
    }
    g.push(Node::path(
        hatch,
        Style::stroke(t.crystal_ink.with_alpha(0.35), 0.7 * line_w),
    ));
    // Mist over the foot: stacked translucent bands of the paper colour.
    let outline = poly(&[
        pt(e[0], 0.0),
        pt(e[0], s[0]),
        apex,
        pt(e[3], s[3]),
        pt(e[3], 0.0),
    ]);
    let mut mist = Node::empty();
    for j in 0..14 {
        let top = -(0.02 + 0.011 * j as f64) * f.h;
        let band = poly(&[
            pt(-hw - 1.0, top),
            pt(hw + 1.0, top),
            pt(hw + 1.0, 1.0),
            pt(-hw - 1.0, 1.0),
        ]);
        mist.push(Node::path(band, Style::fill(t.mist.with_alpha(0.11))));
    }
    g.push(mist.clipped(outline));
    g
}

/// One to three shaded orbs in the upper third, the first on the cool side, the second
/// on the warm side, clear of the moon and of each other.
fn orbs(f: &Frame, seed: &SeedTree, t: &Tones, moon: Point, moon_r: f64) -> Node {
    let mut rng = seed.rng();
    let n = 1 + (rng.uniform(0.0, 3.0) as usize).min(2);
    let mut placed: Vec<(Point, f64)> = vec![(moon, 2.4 * moon_r)];
    let mut g = Node::empty();
    for i in 0..n {
        let warm = match i {
            0 => false,
            1 => true,
            _ => rng.chance(0.5),
        };
        let r = f.u * rng.uniform(0.025, 0.06);
        let spot = (0..30).find_map(|_| {
            let y = rng.uniform(0.03 * f.h + r, 0.32 * f.h);
            let dx = f.canvas.divider_x(y);
            let (lo, hi) = if warm {
                (r + 0.03 * f.u, dx - r - 0.04 * f.u)
            } else {
                (dx + r + 0.04 * f.u, f.w - r - 0.03 * f.u)
            };
            if lo >= hi {
                return None;
            }
            let p = Point::new(rng.uniform(lo, hi), y);
            placed
                .iter()
                .all(|&(q, qr)| (p - q).hypot() > qr + r + 0.03 * f.u)
                .then_some(p)
        });
        let Some(c) = spot else {
            continue;
        };
        placed.push((c, r));
        // Light from the moon on the cool side, from above on the warm side.
        let light = if warm {
            Vec2::new(0.45, -0.9)
        } else {
            unit_or(moon - c, Vec2::new(0.0, -1.0))
        };
        let base = if warm { t.orb_warm } else { t.orb_cool };
        g.push(orb(f, t, c, r, light, base));
    }
    g
}

/// A sphere: a mid-tone disc, a lighter disc offset towards `light` clipped to it, fine
/// hatching across the dark crescent and a thin rim.
fn orb(f: &Frame, t: &Tones, c: Point, r: f64, light: Vec2, base: Color) -> Node {
    let outline = f.disc(c, r);
    let lit_c = c + light * (0.32 * r);
    let lit_r = 0.95 * r;
    let mut g = Node::empty();
    g.push(Node::path(outline.clone(), Style::fill(base.lighten(-0.1))));
    g.push(
        Node::path(f.disc(lit_c, lit_r), Style::fill(base.lighten(0.06))).clipped(outline.clone()),
    );
    // Hatch lines perpendicular to the light, each the chord of the orb minus the chord of
    // the lit disc.
    let along = Vec2::new(-light.y, light.x);
    let sp = 0.06 * r / f.fill.sqrt().max(0.5);
    let mut hatch = BezPath::new();
    let mut d = -r + 0.5 * sp;
    while d < r {
        let p = c + light * d;
        let half = (r * r - d * d).max(0.0).sqrt();
        // Chord of the lit disc on this line.
        let dl = d - 0.32 * r;
        let lit_half = (lit_r * lit_r - dl * dl).max(0.0).sqrt();
        for sgn in [-1.0, 1.0] {
            let (a, b) = (lit_half.min(half), half);
            if b - a > 0.08 * r {
                hatch.move_to(p + along * (sgn * a));
                hatch.line_to(p + along * (sgn * (b - 0.04 * r)));
            }
        }
        d += sp;
    }
    if !hatch.is_empty() {
        g.push(Node::path(
            hatch,
            Style::stroke(t.orb_ink.with_alpha(0.45), 0.014 * r),
        ));
    }
    g.push(Node::path(
        outline,
        Style::stroke(t.orb_ink.with_alpha(0.55), 0.0014 * f.u),
    ));
    g
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

    fn find<'a>(n: &'a Node, name: &str) -> Option<&'a Node> {
        match n {
            Node::Group { id, children, .. } => {
                if id.as_deref() == Some(name) {
                    return Some(n);
                }
                children.iter().find_map(|c| find(c, name))
            }
            Node::Path { .. } => None,
        }
    }

    #[test]
    fn open_landscape_is_finite_bounded_and_within_budget() {
        let sizes = [
            (1200.0, 1800.0),
            (200.0, 300.0),
            (2400.0, 300.0),
            (300.0, 2400.0),
        ];
        for name in ["botanica", "moonlit"] {
            let p = Palette::by_name(name).unwrap();
            for (w, h) in sizes {
                for s in 0..6 {
                    let seed = SeedTree::new(s);
                    let canvas = Canvas::new(&seed.child("canvas"), w, h);
                    for fill in [0.5, 1.0, 2.0] {
                        let n = open_landscape(&seed, &canvas, &p, fill, 0.05);
                        assert!(n.is_finite(), "{name} {w}x{h} seed {s}");
                        let count = n.path_count();
                        assert!(count > 30, "{count} paths");
                        if fill <= 1.0 {
                            assert!(count <= 1500, "{count} paths at fill {fill}");
                        }
                        let bleed = 0.06 * canvas.unit();
                        let bb = n.bounding_box().unwrap();
                        let frame = canvas.rect().inflate(bleed, bleed);
                        assert!(
                            frame.contains(bb.origin()) && frame.contains(Point::new(bb.x1, bb.y1)),
                            "{bb:?} outside {w}x{h}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn open_landscape_is_deterministic() {
        let p = Palette::by_name("botanica").unwrap();
        let seed = SeedTree::new(1234);
        let canvas = Canvas::new(&seed.child("canvas"), 1200.0, 1800.0);
        let a = open_landscape(&seed, &canvas, &p, 1.0, 0.05);
        let b = open_landscape(&seed, &canvas, &p, 1.0, 0.05);
        assert_eq!(a, b);
        let c = open_landscape(&SeedTree::new(1235), &canvas, &p, 1.0, 0.05);
        assert_ne!(a, c);
    }

    #[test]
    fn sky_lies_on_the_cool_side() {
        let p = Palette::by_name("botanica").unwrap();
        for s in 0..8 {
            let seed = SeedTree::new(s);
            let canvas = Canvas::new(&seed.child("canvas"), 1200.0, 1800.0);
            let n = open_landscape(&seed, &canvas, &p, 1.0, 0.05);
            let sky = find(&n, "sky").expect("sky group");
            // Grid samples inside any sky shape: nearly all on the cool side.
            let mut paths = Vec::new();
            sky.for_each_path(|path, _, _| paths.push(path.clone()));
            let (mut inside, mut cool) = (0, 0);
            for i in 0..60 {
                for j in 0..90 {
                    let q = Point::new(20.0 * i as f64 + 10.0, 20.0 * j as f64 + 10.0);
                    if paths.iter().any(|path| path.winding(q) != 0) {
                        inside += 1;
                        cool += usize::from(!canvas.is_warm(q));
                    }
                }
            }
            assert!(inside > 500, "seed {s}: {inside} samples in the sky");
            assert!(
                cool as f64 > 0.95 * inside as f64,
                "seed {s}: {cool}/{inside}"
            );
            // The sky is darkest at the top.
            assert!(canvas.side(sky.bounding_box().unwrap().center()) > 0.0);
        }
    }
}
