//! Integration tests of the `orna` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A fresh, empty directory under the system temp dir, unique per test and process.
fn temp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orna-cli-{}-{test}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn orna(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_orna"))
        .args(args)
        .output()
        .expect("run orna")
}

fn ok(args: &[&str]) -> Output {
    let out = orna(args);
    assert!(
        out.status.success(),
        "orna {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn path_str(p: &Path) -> &str {
    p.to_str().expect("utf-8 temp path")
}

/// A small gallery keeps the PNG small and the test fast.
fn gallery(output: &Path, extra: &[&str]) -> Output {
    let mut args = vec![
        "gallery",
        "--seed",
        "7",
        "--width",
        "800",
        "--output",
        path_str(output),
    ];
    args.extend_from_slice(extra);
    ok(&args)
}

#[test]
fn gallery_writes_svg_and_png_deterministically() {
    let dir = temp_dir("gallery");
    let svg = dir.join("nested/g.svg");
    let png = dir.join("nested/g.png");
    let out = gallery(&svg, &[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(path_str(&svg)) && stdout.contains(path_str(&png)));
    let first = (std::fs::read(&svg).unwrap(), std::fs::read(&png).unwrap());
    assert!(first.0.starts_with(b"<svg"));
    assert!(first.1.starts_with(b"\x89PNG"));

    gallery(&svg, &[]);
    let second = (std::fs::read(&svg).unwrap(), std::fs::read(&png).unwrap());
    assert!(first.0 == second.0, "SVG differs between runs");
    assert!(first.1 == second.1, "PNG differs between runs");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn single_writes_only_the_requested_format() {
    let dir = temp_dir("single");
    gallery(&dir.join("g.svg"), &["--single"]);
    assert!(dir.join("g.svg").exists());
    assert!(!dir.join("g.png").exists());
    let names: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert_eq!(names.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gallery_honours_width() {
    let dir = temp_dir("width");
    let svg = dir.join("g.svg");
    ok(&[
        "gallery",
        "--width",
        "400",
        "--height",
        "300",
        "--single",
        "--output",
        path_str(&svg),
    ]);
    let text = std::fs::read_to_string(&svg).unwrap();
    let root = &text[..text.find('>').unwrap()];
    assert!(root.contains(r#"width="400""#), "{root}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn palettes_lists_names() {
    let out = ok(&["palettes"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<_> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(lines.len() >= 4, "{stdout}");
    assert!(stdout.contains("botanica"));
}

#[test]
fn unknown_palette_fails_with_a_clear_error() {
    let dir = temp_dir("palette");
    let svg = dir.join("g.svg");
    let out = orna(&[
        "gallery",
        "--palette",
        "no-such-palette",
        "--output",
        path_str(&svg),
    ]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("palette") && stderr.contains("no-such-palette"),
        "{stderr}"
    );
    assert!(!svg.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unsupported_output_extension_fails() {
    let dir = temp_dir("ext");
    let out = orna(&[
        "gallery",
        "--width",
        "400",
        "--output",
        path_str(&dir.join("g.jpg")),
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains(".svg or .png"));
    let _ = std::fs::remove_dir_all(&dir);
}
