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

    /// Mix towards `other` by `t` in `[0,1]`.
    pub fn mix(self, other: Color, t: f64) -> Color {
        let t = t.clamp(0.0, 1.0);
        Color::rgba(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )
    }

    /// Lighten (`t > 0`) or darken (`t < 0`) in HSL space.
    pub fn lighten(self, t: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::hsl(h, s, (l + t).clamp(0.0, 1.0)).with_alpha(self.a)
    }

    /// Scale saturation by `k` in HSL space.
    pub fn saturate(self, k: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::hsl(h, (s * k).clamp(0.0, 1.0), l).with_alpha(self.a)
    }

    /// Rotate hue by `deg`.
    pub fn rotate_hue(self, deg: f64) -> Color {
        let (h, s, l) = self.to_hsl();
        Color::hsl(h + deg, s, l).with_alpha(self.a)
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
    fn hex_roundtrip() {
        let c = Color::hex("#7a1f4c").unwrap();
        assert_eq!(c.to_hex(), "#7a1f4c");
        assert!(Color::hex("zzz").is_none());
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
