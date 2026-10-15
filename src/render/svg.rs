//! Scene → SVG.
//!
//! The exporter writes a compact, byte-deterministic document: numbers are formatted with
//! three fixed decimals (trailing zeros trimmed, `-0` normalised to `0`), attributes are
//! emitted in sorted order by the `svg` crate, and attributes equal to their SVG defaults
//! are omitted. Path data is relative and compact (see [`path_data`]). Clip paths live in a
//! single `<defs>` block at the top of the document with ids `clip0`, `clip1`, … assigned in
//! document (pre-)order. Path data that occurs more than once (a leaf outline is filled,
//! clipped to and stroked) is written once in `<defs>` as `<path id="p0">`, `p1`, … and
//! referenced with `<use href>`, which renders identically.
//!
//! Non-finite geometry never reaches the output: a path with a NaN/∞ coordinate is skipped,
//! a group with a non-finite transform is skipped with its subtree, and a non-finite clip is
//! sanitised (its non-finite elements are dropped).

use crate::geom::validate;
use crate::scene::{Color, LineCap, LineJoin, Node, Scene, Style};
use kurbo::{Affine, BezPath, PathEl, Point};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use svg::Node as _;
use svg::node::element::{ClipPath, Definitions, Element, Group, Path as SvgPath, Rectangle};

/// Path data at least this long (bytes) is shared through `<defs>` when it repeats; shorter
/// data costs less inline than a `<use>` reference.
const MIN_SHARED: usize = 32;

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

    let mut counts = HashMap::new();
    count_path_data(&scene.root, &mut counts);
    let mut em = Emitter {
        clips: Vec::new(),
        shared: Vec::new(),
        ids: HashMap::new(),
        counts,
    };
    let mut body = Vec::new();
    emit_node(&scene.root, &mut em, &mut body);

    if !em.clips.is_empty() || !em.shared.is_empty() {
        let mut defs = Definitions::new();
        for p in em.shared {
            defs.append(p);
        }
        for c in em.clips {
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

/// Document-wide state of the exporter.
struct Emitter {
    clips: Vec<ClipPath>,
    /// Shared path data, `<path id="pN" d="…"/>` in id order.
    shared: Vec<SvgPath>,
    /// Id number of each shared path data string.
    ids: HashMap<String, usize>,
    /// How often each path data string occurs in the document.
    counts: HashMap<String, usize>,
}

impl Emitter {
    /// A `<path>` with data `d`, or a `<use>` of the shared copy when `d` repeats.
    fn shape(&mut self, d: String) -> Element {
        if d.len() < MIN_SHARED || self.counts.get(&d).is_none_or(|&n| n < 2) {
            let mut el = Element::new("path");
            el.assign("d", d);
            return el;
        }
        let next = self.ids.len();
        let id = *self.ids.entry(d).or_insert_with_key(|d| {
            self.shared.push(
                SvgPath::new()
                    .set("id", format!("p{next}"))
                    .set("d", d.as_str()),
            );
            next
        });
        let mut el = Element::new("use");
        el.assign("href", format!("#p{id}"));
        el
    }
}

/// Count the path data strings of every path and clip that [`emit_node`] may write.
fn count_path_data(node: &Node, counts: &mut HashMap<String, usize>) {
    let mut add = |path: &BezPath| {
        if let Some(d) = path_data(path) {
            *counts.entry(d).or_insert(0) += 1;
        }
    };
    match node {
        Node::Path { path, style } => {
            if visible(style) {
                add(path);
            }
        }
        Node::Group {
            transform,
            clip,
            children,
            ..
        } => {
            if !transform.as_coeffs().iter().all(|c| c.is_finite()) {
                return;
            }
            if let Some(c) = clip {
                add(&clip_path(&c.path));
            }
            for c in children {
                count_path_data(c, counts);
            }
        }
    }
}

/// A clip path with its non-finite elements dropped.
fn clip_path(path: &BezPath) -> BezPath {
    if validate::path_is_finite(path) {
        path.clone()
    } else {
        validate::sanitize(path, 0.0)
    }
}

/// Append the SVG elements for `node` to `out`, registering clip paths and shared path data
/// in `em`.
///
/// A group with nothing to say (no id, identity transform, no clip, opacity 1) is inlined:
/// its children go straight into `out`, which keeps the document flat and small.
fn emit_node(node: &Node, em: &mut Emitter, out: &mut Vec<Box<dyn svg::Node>>) {
    match node {
        Node::Path { path, style } => {
            if visible(style)
                && let Some(d) = path_data(path)
            {
                let mut el = em.shape(d);
                apply_style(&mut el, style);
                out.push(Box::new(el));
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
                let n = em.clips.len();
                let mut cp = ClipPath::new().set("id", format!("clip{n}"));
                if let Some(d) = path_data(&clip_path(&c.path)) {
                    cp.append(em.shape(d));
                }
                em.clips.push(cp);
                format!("url(#clip{n})")
            });

            let mut kids = Vec::new();
            for c in children {
                emit_node(c, em, &mut kids);
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
#[cfg(test)]
fn path_element(path: &BezPath, style: &Style) -> Option<SvgPath> {
    if !visible(style) {
        return None;
    }
    let mut el = SvgPath::new().set("d", path_data(path)?);
    apply_style(&mut el, style);
    Some(el)
}

/// The visible stroke colour of `style`: none when the width is zero or not finite.
fn stroke_of(style: &Style) -> Option<Color> {
    style
        .stroke
        .filter(|_| style.stroke_width.is_finite() && style.stroke_width > 0.0)
}

/// `false` if a path in `style` would paint nothing.
fn visible(style: &Style) -> bool {
    unit(style.opacity) > 0.0 && (style.fill.is_some() || stroke_of(style).is_some())
}

/// Set the fill, stroke and opacity attributes of `style` on `el`, omitting SVG defaults.
fn apply_style(el: &mut impl svg::Node, style: &Style) {
    let opacity = unit(style.opacity);
    let stroke = stroke_of(style);
    match style.fill {
        Some(c) => paint(el, "fill", c),
        None => el.assign("fill", "none"),
    }
    if let Some(c) = stroke {
        paint(el, "stroke", c);
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
}

/// Set `attr` (fill/stroke) to the colour, plus `attr-opacity` when alpha is below 1.
fn paint(el: &mut impl svg::Node, attr: &str, c: Color) {
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

/// Compact path data. Returns `None` for an empty path, a path with a non-finite
/// coordinate, or one that draws before its first `MoveTo`.
///
/// Coordinates are rounded to thousandths once, as absolute values; every command after the
/// first `M` is then written relative to the rounded current point (`m l h v q c z`), so the
/// offsets are exact and rounding never accumulates along a path. A command repeating the
/// previous one omits its letter, and numbers drop leading zeros (`.5`) and separators that
/// a sign or a second decimal point already imply.
fn path_data(path: &BezPath) -> Option<String> {
    if !validate::path_is_finite(path) {
        return None;
    }
    let els = path.elements();
    if !matches!(els.first(), Some(PathEl::MoveTo(_))) {
        return None;
    }
    let mut w = PathWriter {
        d: String::with_capacity(els.len() * 16),
        last_cmd: None,
        last_dot: None,
    };
    let (mut cur, mut start) = ((0, 0), (0, 0));
    for (i, el) in els.iter().enumerate() {
        match *el {
            PathEl::MoveTo(p) => {
                let p = milli(p);
                if i == 0 {
                    w.cmd('M', &[p]);
                } else {
                    w.cmd('m', &[delta(p, cur)]);
                }
                (cur, start) = (p, p);
            }
            PathEl::LineTo(p) => {
                let p = milli(p);
                match delta(p, cur) {
                    (dx, 0) => w.cmd1('h', dx),
                    (0, dy) => w.cmd1('v', dy),
                    dp => w.cmd('l', &[dp]),
                }
                cur = p;
            }
            PathEl::QuadTo(a, p) => {
                let (a, p) = (milli(a), milli(p));
                w.cmd('q', &[delta(a, cur), delta(p, cur)]);
                cur = p;
            }
            PathEl::CurveTo(a, b, p) => {
                let (a, b, p) = (milli(a), milli(b), milli(p));
                w.cmd('c', &[delta(a, cur), delta(b, cur), delta(p, cur)]);
                cur = p;
            }
            PathEl::ClosePath => {
                w.d.push('z');
                w.last_cmd = Some('z');
                w.last_dot = None;
                cur = start;
            }
        }
    }
    Some(w.d)
}

/// A point rounded to integer thousandths of a unit.
fn milli(p: Point) -> (i64, i64) {
    debug_assert!(
        p.is_finite(),
        "non-finite coordinate reached the SVG formatter"
    );
    ((p.x * 1000.0).round() as i64, (p.y * 1000.0).round() as i64)
}

fn delta(p: (i64, i64), from: (i64, i64)) -> (i64, i64) {
    (p.0 - from.0, p.1 - from.1)
}

/// Appends commands and numbers to path data with the fewest separators.
struct PathWriter {
    d: String,
    /// The last command letter written.
    last_cmd: Option<char>,
    /// Whether the last token was a number containing a decimal point (`None` after a
    /// command letter).
    last_dot: Option<bool>,
}

impl PathWriter {
    /// Write command `c` (its letter omitted when it repeats an implicit-repeatable command)
    /// with the coordinate pairs `pts` in thousandths.
    fn cmd(&mut self, c: char, pts: &[(i64, i64)]) {
        self.letter(c);
        for &(x, y) in pts {
            self.num(x);
            self.num(y);
        }
    }

    /// Write a one-number command (`h`, `v`).
    fn cmd1(&mut self, c: char, v: i64) {
        self.letter(c);
        self.num(v);
    }

    fn letter(&mut self, c: char) {
        // `M`/`m` cannot repeat implicitly (extra pairs would mean line-to).
        if self.last_cmd != Some(c) || matches!(c, 'M' | 'm') {
            self.d.push(c);
            self.last_dot = None;
        }
        self.last_cmd = Some(c);
    }

    /// Write `v / 1000` with up to 3 decimals, no trailing zeros and no leading `0` before
    /// the point, preceded by a space only where the grammar needs one.
    fn num(&mut self, v: i64) {
        let neg = v < 0;
        let a = v.unsigned_abs();
        let (int, frac) = (a / 1000, a % 1000);
        let mut s = String::with_capacity(8);
        if neg {
            s.push('-');
        }
        if int > 0 || frac == 0 {
            let _ = write!(s, "{int}");
        }
        if frac > 0 {
            let f = format!("{frac:03}");
            s.push('.');
            s.push_str(f.trim_end_matches('0'));
        }
        let dot = frac > 0;
        // A sign always starts a new number; a leading point does too after a number that
        // already has one (`1.5.5` is 1.5 then .5).
        let joins = neg || (s.starts_with('.') && self.last_dot == Some(true));
        if self.last_dot.is_some() && !joins {
            self.d.push(' ');
        }
        self.d.push_str(&s);
        self.last_dot = Some(dot);
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
/// like a generated id (`clipN`, `pN`), which it would otherwise shadow.
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
    let generated = ["clip", "p"].iter().any(|prefix| {
        s.strip_prefix(prefix)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    });
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
            "M10-20l20.5 60q-29.5-38-27.5-36c-4-6 2 2 4 4zm-10 20h1"
        );
        // Leading zeros dropped, separators only where needed, repeated commands implicit.
        let mut p = BezPath::new();
        p.move_to((0.5, 0.25));
        p.line_to((1.0, 0.75));
        p.line_to((1.25, 0.5));
        p.line_to((1.25, 2.0));
        p.line_to((1.25, 3.0));
        assert_eq!(path_data(&p).unwrap(), "M.5.25l.5.5.25-.25v1.5 1");
    }

    /// Path data survives a round trip through usvg: every point is the input rounded to
    /// thousandths, with no drift along a long relative path.
    #[test]
    fn path_data_round_trips_through_usvg() {
        use resvg::usvg;
        let mut p = BezPath::new();
        let mut expected = Vec::new();
        let v = |i: usize| ((i * 7919) % 100_003) as f64 / 37.123 - 1300.0;
        let mut pt = |i: usize| {
            let q = Point::new(v(2 * i), v(2 * i + 1));
            expected.push(q);
            q
        };
        p.move_to(pt(0));
        for i in 0..400 {
            let k = 1 + 3 * i;
            match i % 4 {
                0 => p.line_to(pt(k)),
                1 => p.quad_to(pt(k), pt(k + 1)),
                2 => p.curve_to(pt(k), pt(k + 1), pt(k + 2)),
                _ => {
                    p.close_path();
                    p.move_to(pt(k));
                }
            }
        }
        let doc = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><path d="{}"/></svg>"#,
            path_data(&p).unwrap()
        );
        let tree = usvg::Tree::from_str(&doc, &usvg::Options::default()).unwrap();
        let Some(usvg::Node::Path(path)) = tree.root().children().first() else {
            panic!("no path parsed");
        };
        let got = path.data().points();
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(&expected) {
            let r = |x: f64| (x * 1000.0).round() / 1000.0;
            assert!(
                (g.x as f64 - r(e.x)).abs() < 2e-4 && (g.y as f64 - r(e.y)).abs() < 2e-4,
                "{g:?} vs {e:?}"
            );
        }
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
        assert_eq!(sanitize_id("p3").unwrap(), "_p3");
        assert_eq!(sanitize_id("plant").unwrap(), "plant");
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
        assert!(s[c0..c1].contains("M0 0h3"), "{s}");
        assert!(s.contains("matrix(1 0 0 1 5 0)"), "{s}");
        // The only <g> elements are the transformed one, its clip and the inner clip.
        assert_eq!(s.matches("<g").count(), 3, "{s}");
    }

    #[test]
    fn repeated_path_data_is_shared() {
        // A leaf-like motif: the outline filled, clipping the detail, then stroked.
        let mut outline = BezPath::new();
        outline.move_to((0.0, 0.0));
        outline.curve_to((10.0, -8.0), (30.0, -8.0), (40.0, 0.0));
        outline.curve_to((30.0, 8.0), (10.0, 8.0), (0.0, 0.0));
        outline.close_path();
        let detail = Node::path(square(5.0, -2.0, 4.0), Style::fill(Color::WHITE));
        let mut scene = Scene::new(50.0, 20.0);
        scene.push(
            Node::group(vec![
                Node::path(outline.clone(), Style::fill(Color::rgb(0.2, 0.5, 0.2))),
                Node::group(vec![detail]).clipped(outline.clone()),
                Node::path(outline.clone(), Style::stroke(Color::BLACK, 0.5)),
            ])
            .with_transform(Affine::translate((5.0, 10.0))),
        );
        let s = to_svg_string(&scene);
        let d = path_data(&outline).unwrap();
        assert_eq!(s.matches(&format!(r#"d="{d}""#)).count(), 1, "{s}");
        assert!(s.contains(&format!(r#"<path d="{d}" id="p0"/>"#)), "{s}");
        assert_eq!(s.matches(r##"href="#p0""##).count(), 3, "{s}");
        // Short path data stays inline.
        assert!(!s.contains("p1"), "{s}");
        assert_eq!(s, to_svg_string(&scene.clone()));
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
