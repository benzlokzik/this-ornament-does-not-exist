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
                "#5a1032", "#8c1d4a", "#b83a5e", "#d9708a", "#e9a9a8", "#f0cdbf",
            ]),
            cool: hexes(&[
                "#1f1b4a", "#2f3f6f", "#4c6a8a", "#6f8f9d", "#9db7ad", "#c7d6c2",
            ]),
        },
        Palette {
            name: "moonlit".into(),
            background: Color::hex("#1b1a33").unwrap(),
            ink: Color::hex("#d9d2e6").unwrap(),
            light: Color::hex("#ece9f3").unwrap(),
            warm: hexes(&[
                "#4a2144", "#7b3a6a", "#a85f8c", "#c99ab6", "#e3c4d6", "#f3e3ea",
            ]),
            cool: hexes(&[
                "#1d2c52", "#2d4a7a", "#4b6f9b", "#7b9db8", "#a9c3cf", "#d5e3e4",
            ]),
        },
        Palette {
            name: "ember".into(),
            background: Color::hex("#f4e3c9").unwrap(),
            ink: Color::hex("#3a1a0e").unwrap(),
            light: Color::hex("#fbf1e0").unwrap(),
            warm: hexes(&[
                "#5e1a12", "#9a2f1c", "#c9552b", "#e38a4f", "#f0b57f", "#f6d6ad",
            ]),
            cool: hexes(&[
                "#2f3b2a", "#4c5f3b", "#728a52", "#9eab73", "#c2c89b", "#dfe0c0",
            ]),
        },
        Palette {
            name: "verdigris".into(),
            background: Color::hex("#eef0e6").unwrap(),
            ink: Color::hex("#14231f").unwrap(),
            light: Color::hex("#f8f9f1").unwrap(),
            warm: hexes(&[
                "#6b3a2a", "#9c5a3c", "#c48a5a", "#dcb27e", "#eacfa6", "#f3e3c6",
            ]),
            cool: hexes(&[
                "#163c3a", "#1f5f5a", "#2e867c", "#5aa79b", "#8fc6b9", "#c0ded5",
            ]),
        },
        Palette {
            name: "plum-mist".into(),
            background: Color::hex("#ebe4ee").unwrap(),
            ink: Color::hex("#241a2e").unwrap(),
            light: Color::hex("#f6f1f8").unwrap(),
            warm: hexes(&[
                "#4b1d3f", "#7a2d5e", "#a4487e", "#c977a1", "#e0a7c1", "#efd0df",
            ]),
            cool: hexes(&[
                "#2b2d5a", "#444a86", "#6670a8", "#8b97c3", "#b1bbd9", "#d3daea",
            ]),
        },
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
    fn ramp_endpoints() {
        let p = Palette::by_name("botanica").unwrap();
        assert_eq!(p.warm_at(0.0), p.warm[0]);
        assert_eq!(p.warm_at(1.0), *p.warm.last().unwrap());
        assert_eq!(p.cool_at(2.0), *p.cool.last().unwrap());
    }
}
