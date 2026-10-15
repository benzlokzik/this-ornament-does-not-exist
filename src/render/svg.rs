//! Scene → SVG.
//!
//! The exporter writes a compact, byte-deterministic document: numbers are formatted with
//! three fixed decimals (trailing zeros trimmed, `-0` normalised to `0`), attributes are
//! emitted in sorted order by the `svg` crate, and attributes equal to their SVG defaults
//! are omitted. Clip paths live in a single `<defs>` block at the top of the document with
//! ids `clip0`, `clip1`, … assigned in document (pre-)order.
//!
//! Non-finite geometry never reaches the output: a path with a NaN/∞ coordinate is skipped,
//! a group with a non-finite transform is skipped with its subtree, and a non-finite clip is
//! sanitised (its non-finite elements are dropped).

use crate::geom::validate;
use crate::scene::{Color, LineCap, LineJoin, Node, Scene, Style};
use kurbo::{Affine, BezPath, PathEl, Point};
use std::fmt::Write as _;
use std::path::Path;
use svg::Node as _;
use svg::node::element::{ClipPath, Definitions, Group, Path as SvgPath, Rectangle};

/// Build the SVG document for `scene`, including its background rectangle.
pub fn to_svg_document(scene: &Scene) -> svg::Document {
    build_document(scene, true)
}

/// Serialise `scene` as an SVG string (see [`to_svg_document`]).
pub fn to_svg_string(scene: &Scene) -> String {
    to_svg_document(scene).to_string()
}

/// Write `scene` as an SVG file at `path`.
pub fn write_svg(scene: &Scene, path: impl AsRef<Path>) -> anyhow::Result<()> {
    std::fs::write(path, to_svg_string(scene))?;
    Ok(())
}

/// Build the document; `with_background = false` leaves out the background rectangle (the
/// PNG renderer paints the background into the pixmap itself, so it is not composited twice).
pub(crate) fn build_document(scene: &Scene, with_background: bool) -> svg::Document {
    let (w, h) = (fmt_num(scene.width), fmt_num(scene.height));
    let mut doc = svg::Document::new()
        .set("width", w.as_str())
        .set("height", h.as_str())
        .set("viewBox", format!("0 0 {w} {h}"));

    let mut clips = Vec::new();
    let mut body = Vec::new();
    emit_node(&scene.root, &mut clips, &mut body);

    if !clips.is_empty() {
        let mut defs = Definitions::new();
        for c in clips {
            defs.append(c);
        }
        doc.append(defs);
    }
    if with_background && let Some(bg) = scene.background {
        let mut rect = Rectangle::new()
            .set("width", w.as_str())
            .set("height", h.as_str())
            .set("fill", bg.to_hex());
        let a = unit(bg.a);
        if a < 1.0 {
            rect.assign("fill-opacity", fmt_num(a));
        }
        doc.append(rect);
    }
    for n in body {
        doc.append(n);
    }
    doc
}

/// Append the SVG elements for `node` to `out`, registering clip paths in `clips`.
///
/// A group with nothing to say (no id, identity transform, no clip, opacity 1) is inlined:
/// its children go straight into `out`, which keeps the document flat and small.
fn emit_node(node: &Node, clips: &mut Vec<ClipPath>, out: &mut Vec<Box<dyn svg::Node>>) {
    match node {
        Node::Path { path, style } => {
            if let Some(p) = path_element(path, style) {
                out.push(Box::new(p));
            }
        }
        Node::Group {
            id,
            transform,
            clip,
            opacity,
            children,
        } => {
            let coeffs = transform.as_coeffs();
            if !coeffs.iter().all(|c| c.is_finite()) {
                return;
            }
            let id = id.as_deref().and_then(sanitize_id);
            let opacity = unit(*opacity);
            let has_transform = *transform != Affine::IDENTITY;

            // Register the clip before visiting children so ids follow document order.
            let clip_ref = clip.as_ref().map(|c| {
                let n = clips.len();
                let mut cp = ClipPath::new().set("id", format!("clip{n}"));
                let path = if validate::path_is_finite(&c.path) {
                    c.path.clone()
                } else {
                    validate::sanitize(&c.path, 0.0)
                };
                if let Some(d) = path_data(&path) {
                    cp.append(SvgPath::new().set("d", d));
                }
                clips.push(cp);
                format!("url(#clip{n})")
            });

            let mut kids = Vec::new();
            for c in children {
                emit_node(c, clips, &mut kids);
            }

            if id.is_none() && !has_transform && clip_ref.is_none() && opacity == 1.0 {
                out.extend(kids);
                return;
            }

            // The clip is expressed in the group's local (child) space. Put it on an inner
            // <g> nested inside the transformed one so it is unambiguously in child space.
            let mut g = Group::new();
            if has_transform {
                g.assign("transform", fmt_matrix(coeffs));
            }
            if let Some(id) = id {
                g.assign("id", id);
            }
            if opacity != 1.0 {
                g.assign("opacity", fmt_num(opacity));
            }
            match clip_ref {
                Some(url) if has_transform => {
                    let mut inner = Group::new().set("clip-path", url);
                    for k in kids {
                        inner.append(k);
                    }
                    g.append(inner);
                }
                clip_ref => {
                    if let Some(url) = clip_ref {
                        g.assign("clip-path", url);
                    }
                    for k in kids {
                        g.append(k);
                    }
                }
            }
            out.push(Box::new(g));
        }
    }
}

/// The `<path>` element for one styled path, or `None` if it would be invalid or invisible.
fn path_element(path: &BezPath, style: &Style) -> Option<SvgPath> {
    if !validate::path_is_finite(path) {
        return None;
    }
    let opacity = unit(style.opacity);
    let stroke = style
        .stroke
        .filter(|_| style.stroke_width.is_finite() && style.stroke_width > 0.0);
    if opacity == 0.0 || (style.fill.is_none() && stroke.is_none()) {
        return None;
    }
    let mut el = SvgPath::new().set("d", path_data(path)?);
    match style.fill {
        Some(c) => paint(&mut el, "fill", c),
        None => el.assign("fill", "none"),
    }
    if let Some(c) = stroke {
        paint(&mut el, "stroke", c);
        if style.stroke_width != 1.0 {
            el.assign("stroke-width", fmt_num(style.stroke_width));
        }
        // SVG defaults: butt caps, miter joins.
        match style.line_cap {
            LineCap::Butt => {}
            LineCap::Round => el.assign("stroke-linecap", "round"),
            LineCap::Square => el.assign("stroke-linecap", "square"),
        }
        match style.line_join {
            LineJoin::Miter => {}
            LineJoin::Round => el.assign("stroke-linejoin", "round"),
            LineJoin::Bevel => el.assign("stroke-linejoin", "bevel"),
        }
    }
    if opacity < 1.0 {
        el.assign("opacity", fmt_num(opacity));
    }
    Some(el)
}

/// Set `attr` (fill/stroke) to the colour, plus `attr-opacity` when alpha is below 1.
fn paint(el: &mut SvgPath, attr: &str, c: Color) {
    el.assign(attr, c.to_hex());
    let a = unit(c.a);
    if a < 1.0 {
        el.assign(format!("{attr}-opacity"), fmt_num(a));
    }
}

/// Clamp to `[0, 1]`, mapping NaN to 1 (fully opaque, i.e. "no modification").
fn unit(x: f64) -> f64 {
    if x.is_nan() { 1.0 } else { x.clamp(0.0, 1.0) }
}

/// Compact absolute path data (`M L Q C Z`). Returns `None` for an empty path, a path with a
/// non-finite coordinate, or one that draws before its first `MoveTo`.
fn path_data(path: &BezPath) -> Option<String> {
    if !validate::path_is_finite(path) {
        return None;
    }
    let els = path.elements();
    if !matches!(els.first(), Some(PathEl::MoveTo(_))) {
        return None;
    }
    let mut d = String::with_capacity(els.len() * 24);
    for el in els {
        match *el {
            PathEl::MoveTo(p) => cmd(&mut d, 'M', &[p]),
            PathEl::LineTo(p) => cmd(&mut d, 'L', &[p]),
            PathEl::QuadTo(a, p) => cmd(&mut d, 'Q', &[a, p]),
            PathEl::CurveTo(a, b, p) => cmd(&mut d, 'C', &[a, b, p]),
            PathEl::ClosePath => d.push('Z'),
        }
    }
    Some(d)
}

/// Append a command letter and its points. Numbers are separated by a space, except before
/// a minus sign, which already delimits the number (`M10-20` is valid SVG).
fn cmd(d: &mut String, c: char, pts: &[Point]) {
    d.push(c);
    let mut first = true;
    for p in pts {
        for v in [p.x, p.y] {
            debug_assert!(
                v.is_finite(),
                "non-finite coordinate reached the SVG formatter"
            );
            let s = fmt_num(v);
            if !first && !s.starts_with('-') {
                d.push(' ');
            }
            d.push_str(&s);
            first = false;
        }
    }
}

/// `matrix(a b c d e f)` for an affine with coefficients `[a b c d e f]`. The linear part
/// gets 6 decimals: at 3, a rotation about a far-away centre is off by about a pixel.
fn fmt_matrix(c: [f64; 6]) -> String {
    let mut s = String::from("matrix(");
    for (i, v) in c.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{}", fmt_dec(*v, if i < 4 { 6 } else { 3 }));
    }
    s.push(')');
    s
}

/// Format with 3 decimals, trimming trailing zeros and the dot; `-0` becomes `0`.
/// Non-finite input (only reachable through a non-finite canvas size) is written as `0`.
fn fmt_num(x: f64) -> String {
    fmt_dec(x, 3)
}

/// [`fmt_num`] with `prec` decimals.
fn fmt_dec(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return "0".to_string();
    }
    let mut s = format!("{x:.prec$}");
    if s.contains('.') {
        let t = s.trim_end_matches('0').trim_end_matches('.').len();
        s.truncate(t);
    }
    if s == "-0" {
        s = "0".to_string();
    }
    s
}

/// Make a user-supplied id a valid XML name: keep `[A-Za-z0-9_.:-]`, replace anything else
/// with `_`, and prefix `_` if it does not start with a letter or underscore or if it looks
/// like a generated clip id (`clipN`), which it would otherwise shadow.
fn sanitize_id(id: &str) -> Option<String> {
    if id.is_empty() {
        return None;
    }
    let mut s: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let generated = s
        .strip_prefix("clip")
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
    if generated || !s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        s.insert(0, '_');
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kurbo::{Rect, Shape};

    fn square(x: f64, y: f64, s: f64) -> BezPath {
        Rect::new(x, y, x + s, y + s).to_path(0.1)
    }

    #[test]
    fn number_format() {
        assert_eq!(fmt_num(1.0), "1");
        assert_eq!(fmt_num(1.5), "1.5");
        assert_eq!(fmt_num(1.23456), "1.235");
        assert_eq!(fmt_num(-0.0001), "0");
        assert_eq!(fmt_num(-0.0), "0");
        assert_eq!(fmt_num(-2.10), "-2.1");
        assert_eq!(fmt_num(100.0), "100");
        assert_eq!(fmt_num(0.0005), "0.001");
        assert_eq!(fmt_num(f64::NAN), "0");
    }

    #[test]
    fn path_data_compact() {
        let mut p = BezPath::new();
        p.move_to((10.0, -20.0));
        p.line_to((30.5, 40.0));
        p.quad_to((1.0, 2.0), (3.0, 4.0));
        p.curve_to((-1.0, -2.0), (5.0, 6.0), (7.0, 8.0));
        p.close_path();
        p.move_to((0.0, 0.0));
        p.line_to((1.0, 0.0));
        assert_eq!(
            path_data(&p).unwrap(),
            "M10-20L30.5 40Q1 2 3 4C-1-2 5 6 7 8ZM0 0L1 0"
        );
    }

    #[test]
    fn invalid_path_data() {
        assert!(path_data(&BezPath::new()).is_none());
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.line_to((f64::NAN, 1.0));
        assert!(path_data(&p).is_none());
        let mut p = BezPath::new();
        p.move_to((0.0, 0.0));
        p.line_to((f64::INFINITY, 1.0));
        assert!(path_data(&p).is_none());
    }

    #[test]
    fn style_attributes_and_defaults() {
        let p = square(0.0, 0.0, 10.0);
        let s = path_element(&p, &Style::fill(Color::rgba(1.0, 0.0, 0.0, 0.5)))
            .unwrap()
            .to_string();
        assert!(s.contains(r##"fill="#ff0000""##), "{s}");
        assert!(s.contains(r#"fill-opacity="0.5""#), "{s}");
        assert!(!s.contains("stroke"), "{s}");

        let st = Style::stroke(Color::BLACK, 2.5).with_cap(LineCap::Butt);
        let s = path_element(&p, &st).unwrap().to_string();
        assert!(s.contains(r#"fill="none""#), "{s}");
        assert!(s.contains(r#"stroke-width="2.5""#), "{s}");
        assert!(!s.contains("stroke-linecap"), "{s}");
        assert!(s.contains(r#"stroke-linejoin="round""#), "{s}");
        assert!(!s.contains("opacity"), "{s}");

        // Zero-width stroke with no fill draws nothing → skipped.
        assert!(path_element(&p, &Style::stroke(Color::BLACK, 0.0)).is_none());
        // Zero-width stroke with a fill → stroke omitted.
        let s = path_element(&p, &Style::fill_stroke(Color::WHITE, Color::BLACK, 0.0))
            .unwrap()
            .to_string();
        assert!(!s.contains("stroke"), "{s}");
        // NaN stroke width / opacity never leak.
        let mut st = Style::stroke(Color::BLACK, f64::NAN);
        st.fill = Some(Color::WHITE);
        st.opacity = f64::NAN;
        let s = path_element(&p, &st).unwrap().to_string();
        assert!(!s.contains("NaN") && !s.contains("stroke"), "{s}");
    }

    #[test]
    fn ids_are_sanitised() {
        assert_eq!(sanitize_id("leaf-1").unwrap(), "leaf-1");
        assert_eq!(sanitize_id("a\"b<c").unwrap(), "a_b_c");
        assert_eq!(sanitize_id("1st").unwrap(), "_1st");
        assert!(sanitize_id("").is_none());
        assert_eq!(sanitize_id("clip0").unwrap(), "_clip0");
        assert_eq!(sanitize_id("clip").unwrap(), "clip");
        assert_eq!(sanitize_id("clip1a").unwrap(), "clip1a");
    }

    #[test]
    fn plain_groups_are_inlined_and_clips_numbered_in_order() {
        let red = Style::fill(Color::rgb(1.0, 0.0, 0.0));
        let inner = Node::group(vec![Node::path(square(0.0, 0.0, 5.0), red.clone())])
            .clipped(square(0.0, 0.0, 2.0));
        let outer = Node::group(vec![inner, Node::path(square(1.0, 1.0, 1.0), red.clone())])
            .clipped(square(0.0, 0.0, 3.0))
            .with_transform(Affine::translate((5.0, 0.0)));
        let mut scene = Scene::new(10.0, 10.0);
        scene.push(Node::group(vec![outer]));
        let s = to_svg_string(&scene);
        let c0 = s.find(r#"id="clip0""#).unwrap();
        let c1 = s.find(r#"id="clip1""#).unwrap();
        assert!(c0 < c1);
        // clip0 (outer, 3x3) comes first in document order.
        assert!(s[c0..c1].contains("M0 0L3 0"), "{s}");
        assert!(s.contains("matrix(1 0 0 1 5 0)"), "{s}");
        // The only <g> elements are the transformed one, its clip and the inner clip.
        assert_eq!(s.matches("<g").count(), 3, "{s}");
    }

    #[test]
    fn matrix_keeps_rotation_precise() {
        let t = Affine::rotate_about(0.1234, Point::new(1000.0, 1000.0));
        let s = fmt_matrix(t.as_coeffs());
        let v: Vec<f64> = s["matrix(".len()..s.len() - 1]
            .split(' ')
            .map(|x| x.parse().unwrap())
            .collect();
        let r = Affine::new([v[0], v[1], v[2], v[3], v[4], v[5]]);
        for p in [(0.0, 0.0), (2000.0, 0.0), (0.0, 2000.0), (2000.0, 2000.0)] {
            let p = Point::from(p);
            assert!((t * p - r * p).hypot() < 0.01, "{s} at {p:?}");
        }
        assert_eq!(
            fmt_matrix([1.0, 0.0, 0.0, 1.0, 5.0, 0.0]),
            "matrix(1 0 0 1 5 0)"
        );
    }

    #[test]
    fn non_finite_transform_skips_subtree() {
        let mut scene = Scene::new(10.0, 10.0);
        scene.push(
            Node::path(square(0.0, 0.0, 5.0), Style::fill(Color::BLACK))
                .with_transform(Affine::scale(f64::NAN)),
        );
        let s = to_svg_string(&scene);
        assert!(!s.contains("<path") && !s.contains("NaN"), "{s}");
    }

    #[test]
    fn deterministic() {
        let mut scene = Scene::new(20.0, 20.0).with_background(Color::WHITE);
        for i in 0..10 {
            let f = i as f64;
            scene.push(
                Node::path(
                    square(f, f, 1.0 / 3.0),
                    Style::fill(Color::hsl(f * 30.0, 0.5, 0.5)),
                )
                .with_opacity(0.3)
                .with_id(format!("n{i}")),
            );
        }
        assert_eq!(to_svg_string(&scene), to_svg_string(&scene.clone()));
    }
}
