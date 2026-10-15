//! Serialisable generation configuration.

use serde::{Deserialize, Serialize};

/// Which composition to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    /// Reference-inspired scene: warm and cool botanical regions, flowing ornament.
    #[default]
    Botanica,
    /// A single large flower with foliage.
    Bloom,
    /// Flowing ribbons and spirals only.
    Ribbons,
}

/// Everything needed to reproduce an artwork.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub seed: u64,
    pub width: f64,
    pub height: f64,
    pub preset: Preset,
    /// Palette name (see [`crate::palette::all`]).
    pub palette: String,
    /// Global detail multiplier (1.0 = default). Affects hatching density etc.
    pub detail: f64,
    /// Curve fitting / flattening accuracy in canvas units.
    pub accuracy: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            seed: 42,
            width: 1200.0,
            height: 1800.0,
            preset: Preset::Botanica,
            palette: "botanica".into(),
            detail: 1.0,
            accuracy: 0.05,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(self.width.is_finite() && self.height.is_finite())
            || self.width < 16.0
            || self.height < 16.0
        {
            return Err(ConfigError::Dimensions(self.width, self.height));
        }
        if !(self.detail.is_finite() && self.detail > 0.0 && self.detail <= 10.0) {
            return Err(ConfigError::Detail(self.detail));
        }
        if !(self.accuracy.is_finite() && self.accuracy > 0.0) {
            return Err(ConfigError::Accuracy(self.accuracy));
        }
        if crate::palette::Palette::by_name(&self.palette).is_none() {
            return Err(ConfigError::Palette(self.palette.clone()));
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("invalid dimensions {0}x{1} (need finite values >= 16)")]
    Dimensions(f64, f64),
    #[error("detail must be in (0, 10], got {0}")]
    Detail(f64),
    #[error("accuracy must be finite and > 0, got {0}")]
    Accuracy(f64),
    #[error("unknown palette {0:?}")]
    Palette(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let c = Config::default();
        let s = serde_json::to_string(&c).unwrap();
        let d: Config = serde_json::from_str(&s).unwrap();
        assert_eq!(c, d);
    }

    #[test]
    fn validation() {
        assert!(Config::default().validate().is_ok());
        assert!(
            Config {
                width: 0.0,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Config {
                detail: f64::NAN,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            Config {
                palette: "nope".into(),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
