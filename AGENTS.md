# AGENTS.md

Rust crate `orna`, managed via [Cargo.toml](Cargo.toml): a purely procedural, seed-based Art Nouveau botanical illustration engine with a CLI.

## Project context

- Library `orna` ([src/lib.rs](src/lib.rs)) plus binary `orna` ([src/main.rs](src/main.rs)) with `gallery`, `generate` and `palettes` subcommands
- Core stack: `kurbo` for geometry, `svg` for SVG output, `resvg` for PNG rasterisation, `rand` + `rand_chacha` for seeded RNG, `noise` for fields, `clap` for the CLI, `serde`/`serde_json` for config, `rayon` for parallelism, `anyhow`/`thiserror` for errors
- Main code layout, one layer per module; each layer depends only on the layers above it:
  - `src/geom/` contains mathematical geometry: the `Curve` trait, spirals, ribbons, leaves, surfaces, fitting, validation
  - `src/motif/` contains drawable motifs built from geometry (ribbons, spirals, flowers, foliage, plants) that return scene `Node`s
  - `src/ornament/` contains shape-aware ornamentation (contours, hatching, stippling) in surface parameter space
  - `src/compose/` contains composition and procedural growth: fields, presets, the demonstration gallery
  - `src/render/` contains SVG export and PNG rasterisation of a scene
  - `src/scene/` contains the scene graph (`Node`, `Scene`, `Style`, `Color`)
  - `src/config.rs` is the serialisable `Config`, `src/palette.rs` the built-in palettes, `src/rng.rs` the `SeedTree` and `Draw` helpers
- Everything is deterministic: the same `Config` must yield byte-identical SVG

## Tooling

- **pixi** provides the toolchain (`rust`, `prek`) from [pixi.toml](pixi.toml). Run every command through `pixi run ...` or inside `pixi shell`; never install global tools
- If many commands are needed, enter `pixi shell` once, then run normal `cargo` and `prek` commands inside it
- **prek** runs the pre-commit, pre-push and commit-msg hooks defined in [.pre-commit-config.yaml](.pre-commit-config.yaml). Run `pixi run prek run --all-files` before committing
- **Cargo** for the project: `pixi run cargo build`, `pixi run cargo test`, `pixi run cargo clippy --all-targets -- -D warnings`, `pixi run cargo fmt`
- Try a change end to end with the CLI: `pixi run cargo run -- generate --seed 42 --output out/artwork.png` or `pixi run cargo run -- gallery --output out/gallery.svg`. Keep generated images out of the repository

## Rust guidelines

- Add Rust dependencies in [Cargo.toml](Cargo.toml), tool dependencies in [pixi.toml](pixi.toml). Commit [Cargo.lock](Cargo.lock) and [pixi.lock](pixi.lock). Do not add a dependency when the existing stack already covers the need
- Respect the layer order: `geom` → `ornament` → `motif` → `compose` → `render` (ornament engraves `geom` surfaces, motifs use it). A lower layer never imports a higher one
- Draw randomness only from a `SeedTree` child or index stream (`rng::SeedTree`, `rng::Draw`); never use `rand::rng()`, thread RNG, time, or `HashMap` iteration order in generation paths, and keep `rayon` work order-independent so output stays reproducible
- Generators return scene `Node`s; only `render` touches `svg`/`resvg`. Keep geometry in `kurbo` types (`Point`, `Vec2`, `Affine`, `BezPath`)
- Use `thiserror` enums for library errors and `anyhow` with `.context(...)` only in the binary and at layer boundaries. No `unwrap`/`expect` on user input, file I/O or config
- New config fields get a default in `Config::default()`, a check in `Config::validate()` and a `serde` round trip, so existing JSON configs keep loading
- Validate numeric inputs (`is_finite`, ranges) at the boundary; geometry code may assume valid inputs
- Document public items with `///` and each module with a `//!` header; keep edits minimal and surgical

## Checks and quality gates

- Run targeted tests while working: `pixi run cargo test <module>::` for the files changed
- Behaviour changes get or update a `#[cfg(test)]` unit test in the same module; determinism-sensitive changes assert that two runs with the same seed produce equal output
- Before handing work back run, in order: `pixi run cargo fmt`, `pixi run cargo clippy --all-targets -- -D warnings`, `pixi run cargo test`, `pixi run prek run --all-files`. Fix failures unless they are unrelated and explicitly documented
- A visual change is verified by rendering with the CLI and looking at the result, not only by tests passing

## Commits

- Commit regularly, at module granularity: one Conventional Commit per finished task or feature (not necessarily compiling, may be bugged), with its review fixes folded in. Don't batch a whole milestone into one commit, and don't split one part into a trail of micro-commits
- Format `type(scope): summary`; allowed types are `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`. The scope is optional but encouraged (e.g. `feat(motif): ...`)
- The summary is a single short sentence in imperative mood, 69 characters maximum, without a trailing period; the commit-msg hook rejects anything else
- No commit body and no bullet lists. Keep rationale and design notes in the linked GitHub issue or the module docs, not in the commit message

## Security and safety

- Do not commit API keys, private keys or credentials; the `detect-private-key` hook is a backstop, not a policy
- Do not commit generated artworks, `target/` or `.pixi/` environments
- The crate is licensed AGPL-3.0-or-later; only add dependencies with compatible licenses
