//! SVG → PNG via resvg.

use crate::render::svg::build_document;
use crate::scene::{Color, Scene};
use anyhow::{Context, ensure};
use resvg::{tiny_skia, usvg};
use std::path::Path;

/// Largest accepted pixmap side, in pixels.
pub const MAX_PNG_SIDE: u32 = 16384;

/// Rasterise to PNG bytes at `scale` × the scene size.
///
/// The pixmap is `ceil(width·scale) × ceil(height·scale)`. The scene background, if any, is
/// painted into the pixmap first (and left out of the rasterised SVG, so a translucent
/// background is not composited twice). Fails on a non-finite or non-positive scale or canvas
/// size, and on images larger than [`MAX_PNG_SIDE`] on either side.
pub fn render_png(scene: &Scene, scale: f64) -> anyhow::Result<Vec<u8>> {
    ensure!(
        scale.is_finite() && scale > 0.0,
        "invalid PNG scale {scale}: must be finite and > 0"
    );
    ensure!(
        scene.width.is_finite() && scene.height.is_finite(),
        "invalid canvas size {}x{}",
        scene.width,
        scene.height
    );
    let side = |v: f64| -> anyhow::Result<u32> {
        let px = (v * scale).ceil();
        ensure!(
            px >= 1.0 && px <= MAX_PNG_SIDE as f64,
            "PNG size {px} px out of range 1..={MAX_PNG_SIDE} (canvas {}x{}, scale {scale})",
            scene.width,
            scene.height
        );
        Ok(px as u32)
    };
    let (w, h) = (side(scene.width)?, side(scene.height)?);

    let svg = build_document(scene, false).to_string();
    let tree =
        usvg::Tree::from_str(&svg, &usvg::Options::default()).context("parsing generated SVG")?;
    let mut pixmap = tiny_skia::Pixmap::new(w, h).context("allocating pixmap")?;
    if let Some(bg) = scene.background {
        pixmap.fill(skia_color(bg));
    }
    resvg::render(
        &tree,
        tiny_skia::Transform::from_scale(scale as f32, scale as f32),
        &mut pixmap.as_mut(),
    );
    pixmap.encode_png().context("encoding PNG")
}

/// Write `scene` as a PNG file at `path` (see [`render_png`]).
pub fn write_png(scene: &Scene, path: impl AsRef<Path>, scale: f64) -> anyhow::Result<()> {
    std::fs::write(path, render_png(scene, scale)?)?;
    Ok(())
}

/// Convert to a tiny-skia colour, clamping components and mapping NaN to 0 (alpha NaN to 1).
fn skia_color(c: Color) -> tiny_skia::Color {
    let f = |v: f64, nan: f32| {
        if v.is_nan() {
            nan
        } else {
            v.clamp(0.0, 1.0) as f32
        }
    };
    tiny_skia::Color::from_rgba(f(c.r, 0.0), f(c.g, 0.0), f(c.b, 0.0), f(c.a, 1.0))
        .unwrap_or(tiny_skia::Color::TRANSPARENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Node, Style};
    use kurbo::{Rect, Shape};

    fn pixel(png: &[u8], x: u32, y: u32) -> [u8; 4] {
        let pm = tiny_skia::Pixmap::decode_png(png).unwrap();
        let c = pm.pixel(x, y).unwrap().demultiply();
        [c.red(), c.green(), c.blue(), c.alpha()]
    }

    #[test]
    fn rejects_bad_scale_and_size() {
        let s = Scene::new(10.0, 10.0);
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(render_png(&s, scale).is_err(), "scale {scale}");
        }
        assert!(render_png(&Scene::new(0.0, 10.0), 1.0).is_err());
        assert!(render_png(&Scene::new(f64::NAN, 10.0), 1.0).is_err());
        assert!(render_png(&Scene::new(20000.0, 10.0), 1.0).is_err());
        assert!(render_png(&Scene::new(10.0, 10.0), 2000.0).is_err());
    }

    #[test]
    fn translucent_background_composited_once() {
        let s = Scene::new(4.0, 4.0).with_background(Color::rgba(0.0, 0.0, 1.0, 0.5));
        let [r, g, b, a] = pixel(&render_png(&s, 1.0).unwrap(), 1, 1);
        assert_eq!((r, g), (0, 0));
        assert!(b >= 254, "{b}");
        assert!((127..=128).contains(&a), "alpha {a}");
    }

    #[test]
    fn fractional_scale_rounds_up() {
        let mut s = Scene::new(10.0, 5.0).with_background(Color::WHITE);
        s.push(Node::path(
            Rect::new(0.0, 0.0, 5.0, 5.0).to_path(0.1),
            Style::fill(Color::BLACK),
        ));
        let pm = tiny_skia::Pixmap::decode_png(&render_png(&s, 0.33).unwrap()).unwrap();
        assert_eq!((pm.width(), pm.height()), (4, 2));
    }
}
