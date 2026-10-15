# orna

`orna` is a purely procedural, seed-based engine for Art Nouveau botanical illustrations. It draws ribbons, spirals, engraved flowers, foliage and plants from mathematical curves, decorates them with shape-aware hatching and stippling, composes them, and writes SVG and PNG. No images, models or training data are involved: a seed and a few parameters fully determine the artwork.

## Architecture

Five layers, each depending only on the ones above it:

1. `src/geom/`: geometry: the `Curve` trait, spirals, ribbons, leaves, surfaces, curve fitting and validation.
2. `src/motif/`: drawable motifs built from geometry (ribbons, spirals, flowers, foliage, plants) that return scene nodes.
3. `src/ornament/`: shape-aware ornamentation (contours, hatching, stippling) in surface parameter space.
4. `src/compose/`: composition and procedural growth: vector fields, masks, placement, presets and the demonstration gallery.
5. `src/render/`: SVG export and PNG rasterisation (via `resvg`).

Shared pieces: `src/scene/` (the scene graph: `Node`, `Scene`, `Style`, `Color`), `src/config.rs` (the serialisable `Config`), `src/palette.rs` (built-in palettes), `src/rng.rs` (`SeedTree` random streams).

## Command line

Run the binary with `pixi run cargo run --release -- <command>`, or use `target/release/orna` after `pixi run cargo build --release`.

### `orna gallery`

Renders a demonstration sheet: sections of ribbons, spirals, flowers, foliage, plants and one swatch per palette, laid out as a grid of square tiles. The column count follows from `--width` (tiles stay at least 240 units wide), sections without items are left out, and the sheet grows taller than `--height` when the tiles need it.

```sh
orna gallery --seed 42 --output out/gallery.svg
orna gallery --palette moonlit --width 800 --output out/moonlit.png --scale 2
orna gallery --output out/gallery.svg --single
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--seed <SEED>` | `42` | Master seed |
| `-o, --output <OUTPUT>` | `gallery.svg` | Output path ending in `.svg` or `.png`; missing directories are created |
| `--width <WIDTH>` | `1600` | Sheet width |
| `--height <HEIGHT>` | `2000` | Minimum sheet height |
| `--palette <PALETTE>` | `botanica` | Palette of the sheet (see `orna palettes`) |
| `--scale <SCALE>` | `1` | PNG scale factor relative to the SVG size |
| `--single` | off | Write only the format of `--output` |

Without `--single`, both `<name>.svg` and `<name>.png` are written next to each other. Each written path is printed. Tile groups carry SVG ids such as `flowers-daisy` or `palettes-ember`.

### `orna generate`

Generates a full artwork from a preset. The preset compositions are not implemented yet, so this command currently exits with the error `preset not implemented yet`.

```sh
orna generate --seed 42 --preset botanica --output out/artwork.png
orna generate --config my.json --seed 7 --save-config --output out/artwork.svg
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--seed <SEED>` | `42` | Master seed |
| `--preset <PRESET>` | `botanica` | `botanica`, `bloom` or `ribbons` |
| `-o, --output <OUTPUT>` | `artwork.png` | Output path ending in `.svg` or `.png` |
| `--width <WIDTH>` | `1200` | Canvas width |
| `--height <HEIGHT>` | `1800` | Canvas height |
| `--palette <PALETTE>` | `botanica` | Palette name |
| `--detail <DETAIL>` | `1` | Global detail multiplier, in `(0, 10]` |
| `--scale <SCALE>` | `1` | PNG scale factor relative to the SVG size |
| `--config <CONFIG>` | none | JSON config to load; flags given explicitly override its fields |
| `--save-config` | off | Also write the effective config as `<name>.json` |
| `--single` | off | Write only the format of `--output` |

### `orna palettes`

Lists the built-in palettes (`botanica`, `moonlit`, `ember`, `verdigris`, `plum-mist`) with their background, ink, warm and cool colours.

## Determinism

All randomness comes from `SeedTree` streams derived from the seed, so the same seed and options produce byte-identical SVG and PNG on every run. Different subsystems draw from independent streams, so changing one motif does not reshuffle the others.

## Development

The toolchain comes from pixi (`pixi.toml`); run every command through it:

```sh
pixi run cargo fmt
pixi run cargo clippy --all-targets -- -D warnings
pixi run cargo test
pixi run prek run --all-files
```

## Licence

See [LICENSE](LICENSE).
