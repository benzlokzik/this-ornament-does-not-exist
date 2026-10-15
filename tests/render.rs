//! Integration tests for SVG export and PNG rasterisation.

use kurbo::{Affine, BezPath, Rect, Shape};
use orna::render::{render_png, to_svg_string};
use orna::scene::{Color, GradientUnits, Node, Paint, Scene, Style};
use resvg::{tiny_skia, usvg};
use std::time::Instant;

const RED: Color = Color::rgb(1.0, 0.0, 0.0);
const BG: Color = Color::rgb(1.0, 1.0, 1.0);

fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> BezPath {
    Rect::new(x0, y0, x1, y1).to_path(0.1)
}

/// 100×100 white canvas with:
/// - a red square at world (5..15, 5..15),
/// - a stroked line,
/// - a group transformed by translate(20,20)·scale(2) holding a red 20×20 square at the
///   local origin, clipped (in local space) to x ∈ 0..10. In world space the square covers
///   20..60 on both axes and the clip keeps only x ∈ 20..40.
fn test_scene() -> Scene {
    let mut s = Scene::new(100.0, 100.0).with_background(BG);
    s.push(Node::path(rect(5.0, 5.0, 15.0, 15.0), Style::fill(RED)));
    let mut line = BezPath::new();
    line.move_to((70.0, 10.0));
    line.curve_to((80.0, 0.0), (90.0, 20.0), (95.0, 10.0));
    s.push(Node::path(line, Style::stroke(Color::BLACK, 2.0)));
    let clipped = Node::group(vec![Node::path(
        rect(0.0, 0.0, 20.0, 20.0),
        Style::fill(RED),
    )])
    .clipped(rect(0.0, 0.0, 10.0, 20.0))
    .with_transform(Affine::translate((20.0, 20.0)) * Affine::scale(2.0));
    s.push(clipped.with_id("clipped"));
    s
}

fn decode(png: &[u8]) -> tiny_skia::Pixmap {
    tiny_skia::Pixmap::decode_png(png).expect("valid PNG")
}

fn rgba(pm: &tiny_skia::Pixmap, x: u32, y: u32) -> [u8; 4] {
    let c = pm.pixel(x, y).expect("pixel in bounds").demultiply();
    [c.red(), c.green(), c.blue(), c.alpha()]
}

#[test]
fn svg_has_clip_and_matrix_and_parses() {
    let svg = to_svg_string(&test_scene());
    assert!(svg.contains("<clipPath"), "{svg}");
    assert!(svg.contains("clip-path=\"url(#clip0)\""), "{svg}");
    assert!(svg.contains("matrix(2 0 0 2 20 20)"), "{svg}");
    assert!(svg.contains("viewBox=\"0 0 100 100\""), "{svg}");
    assert!(
        svg.contains("xmlns=\"http://www.w3.org/2000/svg\""),
        "{svg}"
    );
    assert!(!svg.contains("NaN") && !svg.contains("inf"), "{svg}");
    usvg::Tree::from_str(&svg, &usvg::Options::default()).expect("usvg parses the export");
}

#[test]
fn png_header_matches_scaled_size() {
    let png = render_png(&test_scene(), 1.5).unwrap();
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&png[12..16], b"IHDR");
    let w = u32::from_be_bytes(png[16..20].try_into().unwrap());
    let h = u32::from_be_bytes(png[20..24].try_into().unwrap());
    assert_eq!((w, h), (150, 150));

    let mut odd = test_scene();
    odd.width = 33.3;
    let png = render_png(&odd, 2.0).unwrap();
    let pm = decode(&png);
    assert_eq!((pm.width(), pm.height()), (67, 200));
}

#[test]
fn pixels_show_fill_transform_and_clip() {
    for scale in [1.0, 2.0] {
        let pm = decode(&render_png(&test_scene(), scale).unwrap());
        let at = |x: f64, y: f64| rgba(&pm, (x * scale) as u32, (y * scale) as u32);
        // Plain filled square.
        assert_eq!(at(10.0, 10.0), [255, 0, 0, 255]);
        // Outside everything → background.
        assert_eq!(at(50.0, 90.0), [255, 255, 255, 255]);
        // Transformed square, inside the clip. A clip applied in parent space (x < 10) or
        // without the scale (x < 30) would make this pixel background.
        assert_eq!(at(35.0, 40.0), [255, 0, 0, 255]);
        // Transformed square but clipped away (local x > 10).
        assert_eq!(at(50.0, 40.0), [255, 255, 255, 255]);
    }
}

#[test]
fn deterministic_svg_and_png() {
    let a = test_scene();
    let b = test_scene();
    assert_eq!(to_svg_string(&a), to_svg_string(&b));
    assert_eq!(render_png(&a, 1.0).unwrap(), render_png(&b, 1.0).unwrap());
}

#[test]
fn empty_scene_renders() {
    let s = Scene::new(16.0, 8.0);
    let svg = to_svg_string(&s);
    usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    let pm = decode(&render_png(&s, 1.0).unwrap());
    assert_eq!((pm.width(), pm.height()), (16, 8));
    assert_eq!(rgba(&pm, 3, 3)[3], 0, "transparent without a background");

    let pm = decode(&render_png(&s.with_background(BG), 1.0).unwrap());
    assert_eq!(rgba(&pm, 3, 3), [255, 255, 255, 255]);
}

#[test]
fn nan_path_is_skipped() {
    let mut s = Scene::new(20.0, 20.0).with_background(BG);
    let mut bad = BezPath::new();
    bad.move_to((0.0, 0.0));
    bad.line_to((f64::NAN, 5.0));
    bad.line_to((10.0, 10.0));
    s.push(Node::path(bad, Style::fill(Color::BLACK)));
    s.push(Node::path(rect(2.0, 2.0, 8.0, 8.0), Style::fill(RED)));
    let svg = to_svg_string(&s);
    assert!(!svg.contains("NaN"), "{svg}");
    assert_eq!(svg.matches("<path").count(), 1, "{svg}");
    usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    let pm = decode(&render_png(&s, 1.0).unwrap());
    assert_eq!(rgba(&pm, 5, 5), [255, 0, 0, 255]);
}

#[test]
fn zero_stroke_width_draws_no_stroke() {
    let mut s = Scene::new(20.0, 20.0).with_background(BG);
    s.push(Node::path(
        rect(5.0, 5.0, 15.0, 15.0),
        Style::stroke(Color::BLACK, 0.0),
    ));
    let svg = to_svg_string(&s);
    assert!(!svg.contains("stroke"), "{svg}");
    let pm = decode(&render_png(&s, 1.0).unwrap());
    assert_eq!(rgba(&pm, 5, 10), [255, 255, 255, 255]);
}

#[test]
fn user_id_clip0_does_not_shadow_generated_clip() {
    let mut s = Scene::new(20.0, 20.0).with_background(BG);
    let clipped = Node::group(vec![Node::path(
        rect(0.0, 0.0, 20.0, 20.0),
        Style::fill(RED),
    )])
    .clipped(rect(0.0, 0.0, 10.0, 20.0));
    s.push(Node::group(vec![clipped]).with_id("clip0"));
    let svg = to_svg_string(&s);
    assert_eq!(svg.matches(r#"id="clip0""#).count(), 1, "{svg}");
    let pm = decode(&render_png(&s, 1.0).unwrap());
    assert_eq!(rgba(&pm, 5, 5), [255, 0, 0, 255]);
    assert_eq!(rgba(&pm, 15, 5), [255, 255, 255, 255]);
}

#[test]
fn large_scene_renders_quickly() {
    let mut s = Scene::new(2000.0, 2000.0).with_background(BG);
    for i in 0..2000 {
        let (x, y) = ((i % 50) as f64 * 40.0, (i / 50) as f64 * 50.0);
        let mut p = BezPath::new();
        p.move_to((x, y));
        p.line_to((x + 30.0, y + 40.0));
        s.push(Node::path(p, Style::stroke(Color::BLACK, 1.0)));
    }
    let t = Instant::now();
    let png = render_png(&s, 1.0).unwrap();
    let elapsed = t.elapsed();
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    eprintln!("2000x2000, 2000 paths: {elapsed:?}");
}

/// A 100×20 canvas filled edge to edge by a rectangle painted with `paint`.
fn painted_rect(paint: Paint) -> Scene {
    let mut s = Scene::new(100.0, 20.0).with_background(BG);
    s.push(Node::path(
        rect(0.0, 0.0, 100.0, 20.0),
        Style::fill_gradient(paint),
    ));
    s
}

fn luma(p: [u8; 4]) -> u32 {
    p[0] as u32 + p[1] as u32 + p[2] as u32
}

fn to_color(p: [u8; 4]) -> Color {
    Color::rgb(
        p[0] as f64 / 255.0,
        p[1] as f64 / 255.0,
        p[2] as f64 / 255.0,
    )
}

#[test]
fn shade_gradient_renders_dark_to_light() {
    let base = Color::hex("#8c1d4a").unwrap();
    let s = painted_rect(Paint::shade(base, 0.15, 0.2, 0.0));
    let svg = to_svg_string(&s);
    assert!(svg.contains("<linearGradient"), "{svg}");
    usvg::Tree::from_str(&svg, &usvg::Options::default()).expect("usvg parses gradients");
    let pm = decode(&render_png(&s, 1.0).unwrap());
    let (l, m, r) = (rgba(&pm, 1, 10), rgba(&pm, 50, 10), rgba(&pm, 98, 10));
    assert!(luma(l) < luma(m) && luma(m) < luma(r), "{l:?} {m:?} {r:?}");
    // The ends are the darkened and lightened base colours.
    let close = |a: Color, b: Color| {
        (a.r - b.r).abs() < 0.04 && (a.g - b.g).abs() < 0.04 && (a.b - b.b).abs() < 0.04
    };
    assert!(close(to_color(l), base.lighten(-0.15)), "{l:?}");
    assert!(close(to_color(r), base.lighten(0.2)), "{r:?}");

    // Radial: light centre, dark rim.
    let s = painted_rect(Paint::radial(
        (0.5, 0.5),
        0.5,
        [(0.0, Color::WHITE), (1.0, Color::BLACK)],
    ));
    let pm = decode(&render_png(&s, 1.0).unwrap());
    assert!(luma(rgba(&pm, 50, 10)) > luma(rgba(&pm, 97, 10)) + 200);
}

#[test]
fn bbox_gradient_follows_transformed_group() {
    // A 10×10 local square shaded black → white left to right, placed by
    // translate(40, 0)·scale(2) at world x 40..60. A bounding-box gradient must span the
    // drawn square, not the canvas or the untransformed coordinates.
    let green = Color::rgb(0.0, 1.0, 0.0);
    let square = |paint: Paint| {
        let mut s = Scene::new(100.0, 20.0).with_background(green);
        s.push(
            Node::path(rect(0.0, 0.0, 10.0, 10.0), Style::fill_gradient(paint))
                .with_transform(Affine::translate((40.0, 0.0)) * Affine::scale(2.0)),
        );
        s
    };
    let stops = [(0.0, Color::BLACK), (1.0, Color::WHITE)];
    let pm =
        decode(&render_png(&square(Paint::linear((0.0, 0.0), (1.0, 0.0), stops)), 1.0).unwrap());
    let (l, m, r) = (rgba(&pm, 41, 10), rgba(&pm, 50, 10), rgba(&pm, 58, 10));
    assert!(luma(l) < 60, "left end dark: {l:?}");
    assert!(luma(r) > 3 * 255 - 100, "right end light: {r:?}");
    assert!((250..520).contains(&luma(m)), "middle grey: {m:?}");
    assert_eq!(
        rgba(&pm, 30, 10),
        [0, 255, 0, 255],
        "background left of the square"
    );
    assert_eq!(
        rgba(&pm, 70, 10),
        [0, 255, 0, 255],
        "background right of the square"
    );

    // User-space units are the path's local space: 0..10 local spans the square.
    let user = Paint::linear((0.0, 0.0), (10.0, 0.0), stops).with_units(GradientUnits::UserSpace);
    let s = square(user);
    let svg = to_svg_string(&s);
    assert!(svg.contains(r#"gradientUnits="userSpaceOnUse""#), "{svg}");
    let pm = decode(&render_png(&s, 1.0).unwrap());
    assert!(luma(rgba(&pm, 41, 10)) < 60, "{:?}", rgba(&pm, 41, 10));
    assert!(
        luma(rgba(&pm, 58, 10)) > 3 * 255 - 100,
        "{:?}",
        rgba(&pm, 58, 10)
    );
}

#[test]
fn equal_gradients_share_one_def() {
    let a = Color::hex("#5a1032").unwrap();
    let b = Color::hex("#4c6a8a").unwrap();
    let mut s = Scene::new(100.0, 100.0);
    // Two paths with equal (separately built) paints, one with another paint, and a stroke
    // with the first paint again.
    for i in 0..2 {
        let x = 10.0 * i as f64;
        s.push(Node::path(
            rect(x, 0.0, x + 8.0, 8.0),
            Style::fill_gradient(Paint::shade(a, 0.1, 0.1, -90.0)),
        ));
    }
    s.push(Node::path(
        rect(50.0, 50.0, 60.0, 60.0),
        Style::fill_gradient(Paint::shade(b, 0.1, 0.1, -90.0)),
    ));
    s.push(Node::path(
        rect(70.0, 70.0, 80.0, 80.0),
        Style::stroke(Color::BLACK, 2.0).with_stroke_paint(Paint::shade(a, 0.1, 0.1, -90.0)),
    ));
    let svg = to_svg_string(&s);
    assert_eq!(svg.matches("<linearGradient").count(), 2, "{svg}");
    assert_eq!(svg.matches(r##"fill="url(#g0)""##).count(), 2, "{svg}");
    assert_eq!(svg.matches(r##"fill="url(#g1)""##).count(), 1, "{svg}");
    assert_eq!(svg.matches(r##"stroke="url(#g0)""##).count(), 1, "{svg}");
    assert!(svg.find(r#"id="g0""#) < svg.find(r#"id="g1""#), "{svg}");
    assert_eq!(svg, to_svg_string(&s.clone()), "byte-deterministic");
    usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
}
