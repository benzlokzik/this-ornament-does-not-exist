//! `orna` command line interface.

use anyhow::Context;
use clap::{Parser, Subcommand};
use orna::compose::gallery::{self, GalleryOptions};
use orna::config::{Config, Preset};
use orna::render;
use orna::scene::Scene;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "orna",
    version,
    about = "Procedural Art Nouveau botanical illustrations"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Render a demonstration sheet of ribbons, spirals, flowers, foliage, plants and palettes.
    Gallery {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Output path; `.svg` or `.png` decides the format. A sibling file in the other
        /// format is written too unless `--single` is given. Missing directories are created.
        #[arg(long, short, default_value = "gallery.svg")]
        output: PathBuf,
        /// Sheet width; the number of tile columns follows from it.
        #[arg(long, default_value_t = 1600.0)]
        width: f64,
        /// Minimum sheet height; the sheet grows taller when the tiles need it.
        #[arg(long, default_value_t = 2000.0)]
        height: f64,
        /// Palette of the sheet (see `orna palettes`).
        #[arg(long, default_value = "botanica")]
        palette: String,
        /// PNG scale factor relative to the SVG size.
        #[arg(long, default_value_t = 1.0)]
        scale: f64,
        /// Write only the requested format.
        #[arg(long)]
        single: bool,
    },
    /// Generate an artwork from a preset.
    Generate {
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum, default_value_t = Preset::Botanica)]
        preset: Preset,
        #[arg(long, short, default_value = "artwork.png")]
        output: PathBuf,
        #[arg(long, default_value_t = 1200.0)]
        width: f64,
        #[arg(long, default_value_t = 1800.0)]
        height: f64,
        #[arg(long, default_value = "botanica")]
        palette: String,
        /// Global detail multiplier.
        #[arg(long, default_value_t = 1.0)]
        detail: f64,
        /// Motif density multiplier for the presets.
        #[arg(long, default_value_t = 1.0)]
        density: f64,
        /// Leave out the moonlit opening (botanica preset).
        #[arg(long)]
        no_opening: bool,
        /// PNG scale factor relative to the SVG size.
        #[arg(long, default_value_t = 1.0)]
        scale: f64,
        /// Load a JSON config (CLI flags override its fields when given explicitly).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Also write the effective config as JSON next to the output.
        #[arg(long)]
        save_config: bool,
        /// Write only the requested format.
        #[arg(long)]
        single: bool,
    },
    /// List the built-in palettes.
    Palettes,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Gallery {
            seed,
            output,
            width,
            height,
            palette,
            scale,
            single,
        } => {
            let opts = GalleryOptions {
                seed,
                width,
                height,
                palette,
            };
            let scene = gallery::build(&opts)?;
            write_outputs(&scene, &output, scale, single)
        }
        Command::Generate {
            seed,
            preset,
            output,
            width,
            height,
            palette,
            detail,
            density,
            no_opening,
            scale,
            config,
            save_config,
            single,
        } => {
            let mut cfg = match config {
                Some(p) => {
                    let text = std::fs::read_to_string(&p)
                        .with_context(|| format!("reading {}", p.display()))?;
                    serde_json::from_str::<Config>(&text)
                        .with_context(|| format!("parsing {}", p.display()))?
                }
                None => Config::default(),
            };
            // explicit flags override the file
            let args: Vec<String> = std::env::args().collect();
            let given = |flag: &str| {
                args.iter()
                    .any(|a| a == flag || a.starts_with(&format!("{flag}=")))
            };
            if given("--seed") || cfg.seed == Config::default().seed {
                cfg.seed = seed;
            }
            if given("--preset") {
                cfg.preset = preset;
            }
            if given("--width") {
                cfg.width = width;
            }
            if given("--height") {
                cfg.height = height;
            }
            if given("--palette") {
                cfg.palette = palette;
            }
            if given("--detail") {
                cfg.detail = detail;
            }
            if given("--density") {
                cfg.density = density;
            }
            if no_opening {
                cfg.opening = false;
            }
            cfg.validate()?;
            let scene = orna::compose::generate(&cfg).with_context(|| {
                format!("generating the {:?} preset (seed {})", cfg.preset, cfg.seed)
            })?;
            if save_config {
                let p = output.with_extension("json");
                create_parent(&p)?;
                std::fs::write(&p, serde_json::to_string_pretty(&cfg)?)
                    .with_context(|| format!("writing {}", p.display()))?;
                println!("wrote {}", p.display());
            }
            write_outputs(&scene, &output, scale, single)
        }
        Command::Palettes => {
            for p in orna::palette::all() {
                println!(
                    "{:12} bg {} ink {} warm {} cool {}",
                    p.name,
                    p.background,
                    p.ink,
                    join(&p.warm),
                    join(&p.cool)
                );
            }
            Ok(())
        }
    }
}

fn join(cs: &[orna::scene::Color]) -> String {
    cs.iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Write SVG and PNG (unless `single`), choosing by the extension of `output`, creating
/// missing parent directories and printing each written path.
fn write_outputs(scene: &Scene, output: &Path, scale: f64, single: bool) -> anyhow::Result<()> {
    let ext = output
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    let (svg_path, png_path) = match ext.as_deref() {
        Some("svg") => (output.to_path_buf(), output.with_extension("png")),
        Some("png") => (output.with_extension("svg"), output.to_path_buf()),
        _ => anyhow::bail!("output {} must end in .svg or .png", output.display()),
    };
    let want_png = ext.as_deref() == Some("png");
    create_parent(output)?;
    if !single || !want_png {
        render::write_svg(scene, &svg_path)
            .with_context(|| format!("writing {}", svg_path.display()))?;
        println!("wrote {}", svg_path.display());
    }
    if !single || want_png {
        render::write_png(scene, &png_path, scale)
            .with_context(|| format!("writing {}", png_path.display()))?;
        println!("wrote {}", png_path.display());
    }
    Ok(())
}

/// Create the parent directory of `path` if it is missing.
fn create_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    Ok(())
}
