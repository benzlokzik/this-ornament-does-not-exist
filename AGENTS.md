# AGENTS.md

Rust crate `orna`, managed via [Cargo.toml](Cargo.toml).

## Tooling

- **pixi** provides the toolchain (`rust`, `prek`) from `pixi.toml`. Run every command through `pixi run ...` or inside `pixi shell`; never install global tools.
- **prek** runs the pre-commit hooks defined in `.pre-commit-config.yaml`. Run `pixi run prek run --all-files` before committing.
- **Cargo** for the project: `pixi run cargo build`, `pixi run cargo test`, `pixi run cargo clippy`, `pixi run cargo fmt`.

## Rules

- Add Rust dependencies in [Cargo.toml](Cargo.toml), tool dependencies in [pixi.toml](pixi.toml). Commit [Cargo.lock](Cargo.lock) and [pixi.lock](pixi.lock).
- Keep edits minimal and run fmt, clippy and tests before handing work back.
