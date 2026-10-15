//! Structured scene graph. Generators return [`Node`]s; renderers consume a [`Scene`].

mod style;

pub use style::{
    Color, GradientUnits, LineCap, LineJoin, LinearGradient, Paint, RadialGradient, Style,
};

use kurbo::{Affine, BezPath, Rect, Shape};
use serde::{Deserialize, Serialize};

/// A clip region applied to a [`Group`](Node::Group).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    pub path: BezPath,
}

/// A node of the scene graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// A filled and/or stroked Bézier path.
    Path { path: BezPath, style: Style },
    /// A group of children with an optional transform and clip.
    Group {
        /// Optional identifier (used for debugging and SVG `id`s).
        id: Option<String>,
        /// Transform applied to all children (child space → parent space).
        transform: Affine,
        /// Optional clip, in the group's *local* (child) coordinate space.
        clip: Option<Clip>,
        /// Group opacity in `[0, 1]`.
        opacity: f64,
        children: Vec<Node>,
    },
}

impl Node {
    pub fn path(path: BezPath, style: Style) -> Node {
        Node::Path { path, style }
    }

    pub fn group(children: Vec<Node>) -> Node {
        Node::Group {
            id: None,
            transform: Affine::IDENTITY,
            clip: None,
            opacity: 1.0,
            children,
        }
    }

    pub fn empty() -> Node {
        Node::group(Vec::new())
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Node {
        if let Node::Group { id: slot, .. } = &mut self {
            *slot = Some(id.into());
        }
        self
    }

    pub fn with_transform(mut self, t: Affine) -> Node {
        match &mut self {
            Node::Group { transform, .. } => *transform = t * *transform,
            Node::Path { .. } => return Node::group(vec![self]).with_transform(t),
        }
        self
    }

    /// Wrap this node in a group clipped by `clip` (clip is in this node's coordinate space).
    pub fn clipped(self, clip: BezPath) -> Node {
        let mut g = match self {
            Node::Group { .. } if matches!(&self, Node::Group { clip: None, transform, .. } if *transform == Affine::IDENTITY) => {
                self
            }
            other => Node::group(vec![other]),
        };
        if let Node::Group { clip: slot, .. } = &mut g {
            *slot = Some(Clip { path: clip });
        }
        g
    }

    pub fn with_opacity(mut self, o: f64) -> Node {
        match &mut self {
            Node::Group { opacity, .. } => *opacity = o.clamp(0.0, 1.0),
            Node::Path { .. } => return Node::group(vec![self]).with_opacity(o),
        }
        self
    }

    pub fn push(&mut self, child: Node) {
        match self {
            Node::Group { children, .. } => children.push(child),
            Node::Path { .. } => {
                let old = std::mem::replace(self, Node::empty());
                *self = Node::group(vec![old, child]);
            }
        }
    }

    /// Number of path leaves in the subtree.
    pub fn path_count(&self) -> usize {
        match self {
            Node::Path { .. } => 1,
            Node::Group { children, .. } => children.iter().map(Node::path_count).sum(),
        }
    }

    /// Visit every path with its accumulated world transform.
    pub fn for_each_path(&self, mut f: impl FnMut(&BezPath, &Style, Affine)) {
        fn walk(n: &Node, t: Affine, f: &mut impl FnMut(&BezPath, &Style, Affine)) {
            match n {
                Node::Path { path, style } => f(path, style, t),
                Node::Group {
                    transform,
                    children,
                    ..
                } => {
                    for c in children {
                        walk(c, t * *transform, f);
                    }
                }
            }
        }
        walk(self, Affine::IDENTITY, &mut f);
    }

    /// Bounding box of all geometry in world space (ignores stroke width and clips).
    pub fn bounding_box(&self) -> Option<Rect> {
        let mut acc: Option<Rect> = None;
        self.for_each_path(|p, _, t| {
            if p.is_empty() {
                return;
            }
            let mut q = p.clone();
            q.apply_affine(t);
            let bb = q.bounding_box();
            acc = Some(acc.map_or(bb, |a| a.union(bb)));
        });
        acc
    }

    /// `true` if every coordinate in the subtree is finite.
    pub fn is_finite(&self) -> bool {
        let mut ok = true;
        fn path_finite(p: &BezPath) -> bool {
            p.elements().iter().all(|el| match el {
                kurbo::PathEl::MoveTo(p) | kurbo::PathEl::LineTo(p) => p.is_finite(),
                kurbo::PathEl::QuadTo(a, b) => a.is_finite() && b.is_finite(),
                kurbo::PathEl::CurveTo(a, b, c) => a.is_finite() && b.is_finite() && c.is_finite(),
                kurbo::PathEl::ClosePath => true,
            })
        }
        fn walk(n: &Node, ok: &mut bool) {
            match n {
                Node::Path { path, style } => {
                    *ok &= path_finite(path) && style.is_finite();
                }
                Node::Group {
                    transform,
                    clip,
                    children,
                    opacity,
                    ..
                } => {
                    *ok &=
                        transform.as_coeffs().iter().all(|c| c.is_finite()) && opacity.is_finite();
                    if let Some(c) = clip {
                        *ok &= path_finite(&c.path);
                    }
                    for c in children {
                        walk(c, ok);
                    }
                }
            }
        }
        walk(self, &mut ok);
        ok
    }
}

/// A complete artwork: a canvas size, background and a root node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub width: f64,
    pub height: f64,
    /// Background colour (`None` = transparent).
    pub background: Option<Color>,
    pub root: Node,
}

impl Scene {
    pub fn new(width: f64, height: f64) -> Scene {
        Scene {
            width,
            height,
            background: None,
            root: Node::empty(),
        }
    }

    pub fn with_background(mut self, c: Color) -> Scene {
        self.background = Some(c);
        self
    }

    pub fn push(&mut self, node: Node) {
        self.root.push(node);
    }

    pub fn canvas(&self) -> Rect {
        Rect::new(0.0, 0.0, self.width, self.height)
    }
}
