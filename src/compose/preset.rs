//! Presets: full-scene compositions. TODO: implemented by a parallel agent.

use crate::config::Config;
use crate::scene::Scene;

pub fn build(config: &Config) -> anyhow::Result<Scene> {
    let _ = config;
    anyhow::bail!("preset not implemented yet")
}
