//! Harmonious colour palettes.

use crate::scene::Color;
use serde::{Deserialize, Serialize};

/// A named palette: a background, ink colour and an ordered set of accent colours
/// split into warm and cool sides, in the spirit of the reference illustration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Palette {
    pub name: String,
    pub background: Color,
    /// Outline / engraving ink.
    pub ink: Color,
    /// Light highlight (paper-ish) colour used for ribbon highlights.
    pub light: Color,
    /// Warm fills, ordered dark → light.
    pub warm: Vec<Color>,
    /// Cool fills, ordered dark → light.
    pub cool: Vec<Color>,
}

impl Palette {
    /// All accent colours (warm then cool).
    pub fn accents(&self) -> Vec<Color> {
        self.warm.iter().chain(self.cool.iter()).copied().collect()
    }

    /// Pick a warm colour by normalised position `t` in `[0,1]` (dark → light).
    pub fn warm_at(&self, t: f64) -> Color {
        ramp(&self.warm, t)
    }

    /// Pick a cool colour by normalised position `t` in `[0,1]` (dark → light).
    pub fn cool_at(&self, t: f64) -> Color {
        ramp(&self.cool, t)
    }

    pub fn by_name(name: &str) -> Option<Palette> {
        all().into_iter().find(|p| p.name == name)
    }

    /// `true` when the background is dark (Oklab lightness below 0.5), so lines drawn on
    /// it need a pale ink.
    pub fn is_dark(&self) -> bool {
        self.background.lightness() < 0.5
    }

    /// Set the ink to `pale` on a dark background and to `dark` on a light one.
    fn with_ink(mut self, dark: &str, pale: &str) -> Palette {
        let ink = if self.is_dark() { pale } else { dark };
        self.ink = Color::hex(ink).expect("valid hex");
        self
    }
}

fn ramp(colors: &[Color], t: f64) -> Color {
    if colors.is_empty() {
        return Color::BLACK;
    }
    if colors.len() == 1 {
        return colors[0];
    }
    let t = t.clamp(0.0, 1.0) * (colors.len() - 1) as f64;
    let i = (t.floor() as usize).min(colors.len() - 2);
    colors[i].mix(colors[i + 1], t - i as f64)
}

fn hexes(list: &[&str]) -> Vec<Color> {
    list.iter()
        .map(|h| Color::hex(h).expect("valid hex"))
        .collect()
}

/// The built-in palettes.
pub fn all() -> Vec<Palette> {
    vec![
        Palette {
            name: "botanica".into(),
            background: Color::hex("#f1e6d8").unwrap(),
            ink: Color::hex("#2a1530").unwrap(),
            light: Color::hex("#f7ecdf").unwrap(),
            warm: hexes(&[
                "#5a1032", "#90134a", "#c12a5b", "#e46688", "#f1a5a4", "#f4ccbc",
            ]),
            cool: hexes(&[
                "#1f1b4a", "#2e3e72", "#476a8f", "#6990a1", "#99b8ac", "#c6d7c0",
            ]),
        },
        Palette {
            name: "moonlit".into(),
            background: Color::hex("#1b1a33").unwrap(),
            ink: Color::BLACK,
            light: Color::hex("#ece9f3").unwrap(),
            warm: hexes(&[
                "#4a2144", "#7e376c", "#ae5a8e", "#cf96b8", "#e7c2d8", "#f5e2ea",
            ]),
            cool: hexes(&[
                "#1d2c52", "#2a4a7e", "#456fa2", "#749ebe", "#a4c4d2", "#d3e3e5",
            ]),
        }
        .with_ink("#1b1a33", "#d9d2e6"),
        Palette {
            name: "ember".into(),
            background: Color::hex("#f4e3c9").unwrap(),
            ink: Color::hex("#3a1a0e").unwrap(),
            light: Color::hex("#fbf1e0").unwrap(),
            warm: hexes(&[
                "#5e1a12", "#9e2813", "#d34b10", "#ee8335", "#f8b26f", "#fad5a5",
            ]),
            cool: hexes(&[
                "#2f3b2a", "#4b6038", "#708b4a", "#9dac68", "#c2c992", "#dfe0bb",
            ]),
        },
        Palette {
            name: "verdigris".into(),
            background: Color::hex("#eef0e6").unwrap(),
            ink: Color::hex("#14231f").unwrap(),
            light: Color::hex("#f8f9f1").unwrap(),
            warm: hexes(&[
                "#6b3a2a", "#9f5837", "#ca874e", "#e2b070", "#eece9c", "#f5e3c1",
            ]),
            cool: hexes(&[
                "#163c3a", "#16605b", "#11887d", "#47aa9c", "#85c8b9", "#bcdfd5",
            ]),
        },
        Palette {
            name: "plum-mist".into(),
            background: Color::hex("#ebe4ee").unwrap(),
            ink: Color::hex("#241a2e").unwrap(),
            light: Color::hex("#f6f1f8").unwrap(),
            warm: hexes(&[
                "#4b1d3f", "#7d295f", "#ab3f80", "#d26fa3", "#e7a3c2", "#f2cee0",
            ]),
            cool: hexes(&[
                "#2b2d5a", "#43498a", "#646faf", "#8896cb", "#afbbdf", "#d2daec",
            ]),
        },
        // Deep and saturated, after the reference: wine, crimson and magenta to blush on the
        // warm side, midnight indigo, violet and slate to sage on the cool side, over
        // plum-indigo. Its background is dark, so the ink is the pale one (a near-black
        // violet ink lets outlines vanish into the ground).
        Palette {
            name: "nocturne".into(),
            background: Color::hex("#21163a").unwrap(),
            ink: Color::BLACK,
            light: Color::hex("#f3e6cf").unwrap(),
            warm: hexes(&[
                "#3f0a26", "#6e1233", "#a01d3c", "#b02f72", "#d97a8c", "#f1c0b4",
            ]),
            cool: hexes(&[
                "#1a1846", "#33286e", "#5a4a8e", "#5d7290", "#8eab92", "#c6dcbe",
            ]),
        }
        .with_ink("#140c1f", "#e9dcc8"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palettes_have_distinct_names_and_colors() {
        let ps = all();
        assert!(ps.len() >= 4);
        for p in &ps {
            assert!(p.warm.len() >= 3 && p.cool.len() >= 3);
            assert!(Palette::by_name(&p.name).is_some());
        }
        let names: std::collections::HashSet<_> = ps.iter().map(|p| p.name.clone()).collect();
        assert_eq!(names.len(), ps.len());
    }

    #[test]
    fn ramps_run_dark_to_light_and_dark_palettes_get_pale_ink() {
        for p in all() {
            for ramp in [&p.warm, &p.cool] {
                let l: Vec<f64> = ramp.iter().map(|c| c.lightness()).collect();
                assert!(l.windows(2).all(|w| w[0] < w[1]), "{}: {l:?}", p.name);
            }
            let dark = matches!(p.name.as_str(), "moonlit" | "nocturne");
            assert_eq!(p.is_dark(), dark, "{}", p.name);
            // Ink contrasts with the background either way.
            let gap = (p.ink.lightness() - p.background.lightness()).abs();
            assert!(gap > 0.5, "{}: ink/background gap {gap}", p.name);
        }
        let n = Palette::by_name("nocturne").unwrap();
        assert!(n.background.lightness() < 0.3 && n.light.lightness() > 0.9);
    }

    #[test]
    fn ramp_endpoints() {
        let p = Palette::by_name("botanica").unwrap();
        assert_eq!(p.warm_at(0.0), p.warm[0]);
        assert_eq!(p.warm_at(1.0), *p.warm.last().unwrap());
        assert_eq!(p.cool_at(2.0), *p.cool.last().unwrap());
    }
}
