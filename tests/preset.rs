//! Integration tests of the full-scene presets.
//!
//! Scenes are small (300 × 450 at detail 0.3) and generated on parallel threads so the
//! suite stays quick in debug builds.

use kurbo::Rect;
use orna::compose::generate;
use orna::compose::preset::PATH_BUDGET;
use orna::config::{Config, Preset};
use orna::render::to_svg_string;
use orna::scene::{Node, Scene};

fn config(preset: Preset, seed: u64) -> Config {
    Config {
        seed,
        preset,
        width: 300.0,
        height: 450.0,
        detail: 0.3,
        accuracy: 0.15,
        ..Default::default()
    }
}

/// Generate every config on its own thread, in order.
fn generate_all(configs: &[Config]) -> Vec<Scene> {
    std::thread::scope(|s| {
        let handles: Vec<_> = configs
            .iter()
            .map(|c| s.spawn(move || generate(c).expect("preset builds")))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("generation thread"))
            .collect()
    })
}

/// The named layer groups of a scene (`background`, `ribbons`, ...).
fn layers(scene: &Scene) -> Vec<(String, &Node)> {
    fn walk<'a>(n: &'a Node, out: &mut Vec<(String, &'a Node)>, depth: usize) {
        if let Node::Group { id, children, .. } = n {
            if depth > 0
                && let Some(id) = id
                && id != "artwork"
            {
                out.push((id.clone(), n));
                return;
            }
            for c in children {
                walk(c, out, depth + 1);
            }
        }
    }
    let mut out = Vec::new();
    walk(&scene.root, &mut out, 0);
    out
}

fn layer_names(scene: &Scene) -> Vec<String> {
    layers(scene).into_iter().map(|(n, _)| n).collect()
}

/// Finite geometry, a sane path count, and every layer but the tonal background on the
/// canvas up to a deliberate bleed (the divider band leaves the top edge, plants grow
/// from the bottom one).
fn check_scene(cfg: &Config, scene: &Scene) {
    let what = format!("{:?} seed {}", cfg.preset, cfg.seed);
    assert!(scene.root.is_finite(), "{what}: non-finite geometry");
    let paths = scene.root.path_count();
    let budget = (PATH_BUDGET * cfg.detail) as usize;
    assert!((30..=budget).contains(&paths), "{what}: {paths} paths");
    let found = layers(scene);
    assert!(found.len() >= 3, "{what}: layers {:?}", layer_names(scene));
    let bleed = 0.08 * cfg.height;
    let allowed = Rect::new(-bleed, -bleed, cfg.width + bleed, cfg.height + bleed);
    for (name, node) in &found {
        if name == "background" {
            continue;
        }
        let bb = node.bounding_box().expect("layer has geometry");
        assert!(
            allowed.contains(bb.origin()) && allowed.contains((bb.x1, bb.y1)),
            "{what}: layer {name} at {bb:?}"
        );
    }
}

/// Seeds 1..=3 build sane scenes; the same config twice gives identical SVG, another
/// seed a different one. `extra` configs are generated alongside and returned after them.
fn check_preset(preset: Preset, extra: &[Config]) -> Vec<Scene> {
    let mut configs: Vec<Config> = (1..=3).map(|s| config(preset, s)).collect();
    configs.push(config(preset, 1));
    configs.extend_from_slice(extra);
    let scenes = generate_all(&configs);
    for (cfg, scene) in configs.iter().zip(&scenes).take(3) {
        check_scene(cfg, scene);
    }
    let svg = |i: usize| to_svg_string(&scenes[i]);
    assert!(svg(0) == svg(3), "{preset:?}: SVG differs between runs");
    assert!(svg(0) != svg(1), "{preset:?}: the seed has no effect");
    scenes
}

#[test]
fn botanica() {
    let no_opening = Config {
        opening: false,
        ..config(Preset::Botanica, 2)
    };
    let scenes = check_preset(Preset::Botanica, std::slice::from_ref(&no_opening));
    let names = layer_names(&scenes[1]);
    for layer in [
        "background",
        "back-foliage",
        "opening",
        "ribbons",
        "plants",
        "foreground",
    ] {
        assert!(
            names.iter().any(|n| n == layer),
            "missing {layer}: {names:?}"
        );
    }
    assert!(!layer_names(&scenes[4]).iter().any(|n| n == "opening"));
    check_scene(&no_opening, &scenes[4]);
}

#[test]
fn bloom() {
    check_preset(Preset::Bloom, &[]);
}

#[test]
fn ribbons() {
    let scenes = check_preset(Preset::Ribbons, &[]);
    // Only ribbons and scrolls: no flowers, plants or foliage layers.
    for s in &scenes {
        for n in layer_names(s) {
            assert!(
                ["background", "ribbons", "foreground"].contains(&n.as_str()),
                "unexpected layer {n}"
            );
        }
    }
}
