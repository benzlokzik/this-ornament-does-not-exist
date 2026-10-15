use kurbo::Point;
use palette::{FromColor, Mix, Oklab, Oklch, Srgb};
use serde::{Deserialize, Serialize};
use std::fmt;

/// An sRGB colour with alpha, components in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub const BLACK: Color = Color::rgb(0.0, 0.0, 0.0);
    pub const WHITE: Color = Color::rgb(1.0, 1.0, 1.0);

    pub const fn rgb(r: f64, g: f64, b: f64) -> Color {
        Color { r, g, b, a: 1.0 }
    }

    pub const fn rgba(r: f64, g: f64, b: f64, a: f64) -> Color {
        Color { r, g, b, a }
    }

    /// Parse `#rrggbb` or `#rrggbbaa`.
    pub fn hex(s: &str) -> Option<Color> {
        let s = s.strip_prefix('#')?;
        let v = |i: usize| {
            u8::from_str_radix(s.get(i..i + 2)?, 16)
                .ok()
                .map(|x| x as f64 / 255.0)
        };
        match s.len() {
            6 => Some(Color::rgb(v(0)?, v(2)?, v(4)?)),
            8 => Some(Color::rgba(v(0)?, v(2)?, v(4)?, v(6)?)),
            _ => None,
        }
    }

    /// From HSL (`h` in degrees, `s`,`l` in `[0,1]`).
    pub fn hsl(h: f64, s: f64, l: f64) -> Color {
        let h = h.rem_euclid(360.0) / 60.0;
        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - (h % 2.0 - 1.0).abs());
        let (r, g, b) = match h as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        let m = l - c / 2.0;
        Color::rgb(r + m, g + m, b + m)
    }

    /// To HSL: `(hue_degrees, saturation, lightness)`.
    pub fn to_hsl(self) -> (f64, f64, f64) {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        let l = (max + min) / 2.0;
        let d = max - min;
        if d < 1e-9 {
            return (0.0, 0.0, l);
        }
        let s = d / (1.0 - (2.0 * l - 1.0).abs());
        let h = if max == self.r {
            60.0 * ((self.g - self.b) / d).rem_euclid(6.0)
        } else if max == self.g {
            60.0 * ((self.b - self.r) / d + 2.0)
        } else {
            60.0 * ((self.r - self.g) / d + 4.0)
        };
        (h, s, l)
    }

    pub fn with_alpha(mut self, a: f64) -> Color {
        self.a = a;
        self
    }

    fn to_oklab(self) -> Oklab<f64> {
        Oklab::from_color(Srgb::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
        ))
    }

    fn from_oklab(lab: Oklab<f64>, a: f64) -> Color {
        let rgb: Srgb<f64> = Srgb::from_color(lab);
        Color::rgba(
            rgb.red.clamp(0.0, 1.0),
            rgb.green.clamp(0.0, 1.0),
            rgb.blue.clamp(0.0, 1.0),
            a,
        )
    }

    fn map_oklch(self, f: impl FnOnce(&mut Oklch<f64>)) -> Color {
        let mut lch = Oklch::from_color(self.to_oklab());
        f(&mut lch);
        lch.l = lch.l.clamp(0.0, 1.0);
        lch.chroma = lch.chroma.max(0.0);
        Color::from_oklab(Oklab::from_color(lch), self.a)
    }

    /// Mix towards `other` by `t` in `[0,1]`, perceptually (Oklab), so warm/cool blends
    /// pass through clean mid tones instead of muddy ones.
    pub fn mix(self, other: Color, t: f64) -> Color {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Exact endpoints: the Oklab round trip would otherwise perturb the last bits.
        if t <= 0.0 {
            return self;
        }
        if t >= 1.0 {
            return other;
        }
        let lab = self.to_oklab().mix(other.to_oklab(), t);
        Color::from_oklab(lab, self.a + (other.a - self.a) * t)
    }

    /// Lighten (`t > 0`) or darken (`t < 0`) by shifting Oklch lightness.
    pub fn lighten(self, t: f64) -> Color {
        self.map_oklch(|c| c.l += t)
    }

    /// Scale chroma (Oklch) by `k`.
    pub fn saturate(self, k: f64) -> Color {
        self.map_oklch(|c| c.chroma *= k.max(0.0))
    }

    /// Rotate hue by `deg` (Oklch).
    pub fn rotate_hue(self, deg: f64) -> Color {
        self.map_oklch(|c| c.hue += deg)
    }

    /// Perceptual lightness (Oklab `L`, 0 = black, 1 = white).
    pub fn lightness(self) -> f64 {
        self.to_oklab().l
    }

    /// `#rrggbb` (alpha is exported separately as opacity).
    pub fn to_hex(self) -> String {
        let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        format!("#{:02x}{:02x}{:02x}", q(self.r), q(self.g), q(self.b))
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LineCap {
    Butt,
    #[default]
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LineJoin {
    Miter,
    #[default]
    Round,
    Bevel,
}

/// The coordinate system of a gradient's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GradientUnits {
    /// The path's own (local) coordinates.
    UserSpace,
    /// Fractions of the painted path's bounding box: `(0, 0)` is its top-left corner and
    /// `(1, 1)` its bottom-right, so a gradient fits any motif without knowing its extent.
    #[default]
    BoundingBox,
}

/// A linear gradient from `start` (offset 0) to `end` (offset 1), padded beyond both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearGradient {
    pub start: Point,
    pub end: Point,
    /// `(offset, colour)` pairs, offsets in `[0, 1]` and non-decreasing.
    pub stops: Vec<(f64, Color)>,
    pub units: GradientUnits,
}

/// A radial gradient from the focal point (offset 0) to the circle around `center` with
/// `radius` (offset 1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RadialGradient {
    pub center: Point,
    pub radius: f64,
    /// Focal point; `None` = the centre.
    pub focal: Option<Point>,
    /// `(offset, colour)` pairs, offsets in `[0, 1]` and non-decreasing.
    pub stops: Vec<(f64, Color)>,
    pub units: GradientUnits,
}

/// What a fill or stroke is painted with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Paint {
    Solid(Color),
    Linear(LinearGradient),
    Radial(RadialGradient),
}

impl Paint {
    /// A linear gradient in bounding-box units from `from` to `to` (fractions of the bbox).
    pub fn linear(
        from: impl Into<Point>,
        to: impl Into<Point>,
        stops: impl IntoIterator<Item = (f64, Color)>,
    ) -> Paint {
        Paint::Linear(LinearGradient {
            start: from.into(),
            end: to.into(),
            stops: stops.into_iter().collect(),
            units: GradientUnits::BoundingBox,
        })
    }

    /// A radial gradient in bounding-box units around `center` with `radius` (fractions of
    /// the bbox), its focal point at the centre.
    pub fn radial(
        center: impl Into<Point>,
        radius: f64,
        stops: impl IntoIterator<Item = (f64, Color)>,
    ) -> Paint {
        Paint::Radial(RadialGradient {
            center: center.into(),
            radius,
            focal: None,
            stops: stops.into_iter().collect(),
            units: GradientUnits::BoundingBox,
        })
    }

    /// Tonal shading of `base` across the bounding box: a two-stop linear gradient from
    /// `base` darkened by `dark` to `base` lightened by `light` (Oklch lightness, see
    /// [`Color::lighten`]). `angle_deg` is the direction from the dark to the light end in
    /// y-down coordinates: 0 = left → right, 90 = top → bottom, -90 = bottom → top. The
    /// gradient spans the whole box along that direction, so its corners get the end tones.
    pub fn shade(base: Color, dark: f64, light: f64, angle_deg: f64) -> Paint {
        let (dark, light) = (finite_or_zero(dark), finite_or_zero(light));
        let (s, c) = finite_or_zero(angle_deg).to_radians().sin_cos();
        let h = 0.5 * (c.abs() + s.abs());
        Paint::linear(
            (0.5 - h * c, 0.5 - h * s),
            (0.5 + h * c, 0.5 + h * s),
            [(0.0, base.lighten(-dark)), (1.0, base.lighten(light))],
        )
    }

    /// Engraving-style tonal shading of a motif body: [`Paint::shade`] with ±0.12 lightness
    /// and a hue drift chosen from the base hue, so warm bodies lighten towards rose and
    /// cool bodies towards sage like the hand-tinted reference. `angle_deg` as in `shade`.
    pub fn tonal(base: Color, angle_deg: f64) -> Paint {
        let (h, _, _) = base.to_hsl();
        let warm = !(70.0..300.0).contains(&h);
        let (dark_hue, light_hue) = if warm { (-6.0, 8.0) } else { (8.0, -24.0) };
        let (s, c) = finite_or_zero(angle_deg).to_radians().sin_cos();
        let half = 0.5 * (c.abs() + s.abs());
        Paint::linear(
            (0.5 - half * c, 0.5 - half * s),
            (0.5 + half * c, 0.5 + half * s),
            [
                (0.0, base.lighten(-0.12).rotate_hue(dark_hue)),
                (1.0, base.lighten(0.12).rotate_hue(light_hue)),
            ],
        )
    }

    /// The same paint with its geometry in `units`.
    pub fn with_units(mut self, units: GradientUnits) -> Paint {
        match &mut self {
            Paint::Solid(_) => {}
            Paint::Linear(g) => g.units = units,
            Paint::Radial(g) => g.units = units,
        }
        self
    }

    /// The gradient stops (empty for a solid paint).
    pub fn stops(&self) -> &[(f64, Color)] {
        match self {
            Paint::Solid(_) => &[],
            Paint::Linear(g) => &g.stops,
            Paint::Radial(g) => &g.stops,
        }
    }

    /// One colour standing for the paint: the solid colour, or the Oklab mean of the stops
    /// (`None` for a gradient without stops).
    pub fn mean_color(&self) -> Option<Color> {
        if let Paint::Solid(c) = self {
            return Some(*c);
        }
        let (first, rest) = self.stops().split_first()?;
        let mut acc = first.1;
        for (i, (_, c)) in rest.iter().enumerate() {
            acc = acc.mix(*c, 1.0 / (i + 2) as f64);
        }
        Some(acc)
    }

    /// `true` if every coordinate, radius and stop offset is finite.
    pub fn is_finite(&self) -> bool {
        let stops = self.stops().iter().all(|(o, _)| o.is_finite());
        match self {
            Paint::Solid(_) => true,
            Paint::Linear(g) => stops && g.start.is_finite() && g.end.is_finite(),
            Paint::Radial(g) => {
                stops
                    && g.center.is_finite()
                    && g.radius.is_finite()
                    && g.focal.is_none_or(|f| f.is_finite())
            }
        }
    }
}

impl From<Color> for Paint {
    fn from(c: Color) -> Paint {
        Paint::Solid(c)
    }
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

/// Fill/stroke style of a path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Style {
    pub fill: Option<Color>,
    pub stroke: Option<Color>,
    pub stroke_width: f64,
    pub line_cap: LineCap,
    pub line_join: LineJoin,
    /// Whole-path opacity multiplier in `[0,1]`.
    pub opacity: f64,
    /// Fill paint; overrides [`fill`](Self::fill) when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill_paint: Option<Paint>,
    /// Stroke paint; overrides [`stroke`](Self::stroke) when set (the stroke still needs a
    /// positive width).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stroke_paint: Option<Paint>,
}

impl Default for Style {
    fn default() -> Self {
        Style {
            fill: None,
            stroke: Some(Color::BLACK),
            stroke_width: 1.0,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            opacity: 1.0,
            fill_paint: None,
            stroke_paint: None,
        }
    }
}

impl Style {
    pub fn fill(c: Color) -> Style {
        Style {
            fill: Some(c),
            stroke: None,
            ..Default::default()
        }
    }

    pub fn stroke(c: Color, width: f64) -> Style {
        Style {
            fill: None,
            stroke: Some(c),
            stroke_width: width,
            ..Default::default()
        }
    }

    pub fn fill_stroke(fill: Color, stroke: Color, width: f64) -> Style {
        Style {
            fill: Some(fill),
            stroke: Some(stroke),
            stroke_width: width,
            ..Default::default()
        }
    }

    /// A fill-only style painted with `paint`; [`fill`](Self::fill) is set to the paint's
    /// mean colour for code that reads plain colours.
    pub fn fill_gradient(paint: Paint) -> Style {
        Style {
            fill: None,
            stroke: None,
            ..Default::default()
        }
        .with_fill_paint(paint)
    }

    /// Paint the fill with `paint`, and set [`fill`](Self::fill) to its mean colour.
    pub fn with_fill_paint(mut self, paint: Paint) -> Style {
        self.fill = paint.mean_color().or(self.fill);
        self.fill_paint = Some(paint);
        self
    }

    /// Paint the stroke with `paint`, and set [`stroke`](Self::stroke) to its mean colour.
    pub fn with_stroke_paint(mut self, paint: Paint) -> Style {
        self.stroke = paint.mean_color().or(self.stroke);
        self.stroke_paint = Some(paint);
        self
    }

    /// The paint the fill is drawn with: [`fill_paint`](Self::fill_paint), else the plain
    /// [`fill`](Self::fill) colour.
    pub fn effective_fill(&self) -> Option<Paint> {
        self.fill_paint.clone().or(self.fill.map(Paint::Solid))
    }

    /// The paint the stroke is drawn with (see [`effective_fill`](Self::effective_fill)).
    pub fn effective_stroke(&self) -> Option<Paint> {
        self.stroke_paint.clone().or(self.stroke.map(Paint::Solid))
    }

    /// `true` if the stroke width and every paint coordinate are finite.
    pub fn is_finite(&self) -> bool {
        self.stroke_width.is_finite()
            && self.fill_paint.as_ref().is_none_or(Paint::is_finite)
            && self.stroke_paint.as_ref().is_none_or(Paint::is_finite)
    }

    pub fn with_opacity(mut self, o: f64) -> Style {
        self.opacity = o.clamp(0.0, 1.0);
        self
    }

    pub fn with_cap(mut self, cap: LineCap) -> Style {
        self.line_cap = cap;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shade_runs_dark_to_light_along_the_angle() {
        let base = Color::hex("#8c1d4a").unwrap();
        let Paint::Linear(g) = Paint::shade(base, 0.1, 0.15, 0.0) else {
            panic!("shade is linear");
        };
        assert_eq!(g.units, GradientUnits::BoundingBox);
        assert!((g.start - Point::new(0.0, 0.5)).hypot() < 1e-9);
        assert!((g.end - Point::new(1.0, 0.5)).hypot() < 1e-9);
        let l = |c: Color| Oklch::from_color(c.to_oklab()).l;
        assert!(l(g.stops[0].1) < l(base) && l(base) < l(g.stops[1].1));
        // A diagonal spans the whole box: the corners lie on the end lines.
        let Paint::Linear(g) = Paint::shade(base, 0.1, 0.1, 45.0) else {
            panic!("shade is linear");
        };
        assert!(
            (g.start - Point::new(0.0, 0.0)).hypot() < 1e-9,
            "{:?}",
            g.start
        );
        assert!((g.end - Point::new(1.0, 1.0)).hypot() < 1e-9, "{:?}", g.end);
        let Paint::Linear(g) = Paint::shade(base, 0.1, 0.1, -90.0) else {
            panic!("shade is linear");
        };
        assert!(g.start.y > g.end.y, "-90° runs bottom → top");
    }

    #[test]
    fn paints_override_colours_and_round_trip() {
        let p = Paint::linear(
            (0.0, 0.0),
            (1.0, 0.0),
            [(0.0, Color::BLACK), (1.0, Color::WHITE)],
        );
        let st = Style::fill_gradient(p.clone());
        assert_eq!(st.effective_fill(), Some(p));
        assert!(st.fill.is_some() && st.effective_stroke().is_none());
        assert_eq!(
            Style::fill(Color::WHITE).effective_fill(),
            Some(Paint::Solid(Color::WHITE))
        );
        let json = serde_json::to_string(&st).unwrap();
        assert_eq!(serde_json::from_str::<Style>(&json).unwrap(), st);
        // Plain styles serialise as before, so older scene JSON still loads.
        let plain = serde_json::to_string(&Style::fill(Color::WHITE)).unwrap();
        assert!(!plain.contains("paint"), "{plain}");
        assert_eq!(
            serde_json::from_str::<Style>(&plain).unwrap(),
            Style::fill(Color::WHITE)
        );
        let mut bad = Paint::radial((0.5, 0.5), 0.5, [(0.0, Color::BLACK)]);
        assert!(bad.is_finite());
        if let Paint::Radial(g) = &mut bad {
            g.focal = Some(Point::new(f64::NAN, 0.0));
        }
        assert!(!bad.is_finite());
        assert!(!Style::fill_gradient(bad).is_finite());
    }

    #[test]
    fn hex_roundtrip() {
        let c = Color::hex("#7a1f4c").unwrap();
        assert_eq!(c.to_hex(), "#7a1f4c");
        assert!(Color::hex("zzz").is_none());
    }

    #[test]
    fn oklab_mix_keeps_mid_tones_clean() {
        let warm = Color::hex("#b83a5e").unwrap();
        let cool = Color::hex("#4c6a8a").unwrap();
        let mid = warm.mix(cool, 0.5);
        let l = |c: Color| Oklch::from_color(c.to_oklab()).l;
        // a perceptual mix sits between the two lightnesses rather than dropping below both
        assert!(l(mid) >= l(warm).min(l(cool)) - 1e-6 && l(mid) <= l(warm).max(l(cool)) + 1e-6);
        assert_eq!(warm.mix(cool, 0.0).to_hex(), warm.to_hex());
        assert_eq!(warm.mix(cool, 1.0).to_hex(), cool.to_hex());
        assert!(warm.lighten(0.2).to_hsl().2 > warm.to_hsl().2);
        assert!(warm.saturate(0.0).to_hsl().1 < 0.02);
        assert!(warm.mix(cool, f64::NAN).r.is_finite());
    }

    #[test]
    fn hsl_roundtrip() {
        for (h, s, l) in [(0.0, 0.5, 0.5), (120.0, 0.8, 0.3), (300.0, 0.2, 0.7)] {
            let c = Color::hsl(h, s, l);
            let (h2, s2, l2) = c.to_hsl();
            assert!((h - h2).abs() < 1e-6 && (s - s2).abs() < 1e-6 && (l - l2).abs() < 1e-6);
        }
    }
}
