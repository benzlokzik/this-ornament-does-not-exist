//! Deterministic random streams derived from a master seed.
//!
//! Every subsystem gets its own [`ChaCha8Rng`] derived from `(seed, label)`, so
//! changing how many random numbers one subsystem draws does not perturb the others.

use rand::SeedableRng;
/// Core RNG trait (`next_u64` etc.). `rand::RngExt` adds `random_range` and friends.
pub use rand::{Rng, RngExt};
pub use rand_chacha::ChaCha8Rng;

/// A seed tree: derive independent, reproducible RNG streams by label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeedTree {
    seed: u64,
}

impl SeedTree {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Derive a child seed tree for a labelled subsystem (e.g. `"flowers"`).
    pub fn child(&self, label: &str) -> SeedTree {
        SeedTree {
            seed: mix(self.seed, hash_label(label)),
        }
    }

    /// Derive a child seed tree for an indexed item (e.g. petal `i`).
    pub fn index(&self, i: u64) -> SeedTree {
        SeedTree {
            seed: mix(
                self.seed,
                i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03,
            ),
        }
    }

    /// A fresh RNG stream for this node.
    pub fn rng(&self) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(self.seed)
    }

    /// Shorthand for `self.child(label).rng()`.
    pub fn rng_for(&self, label: &str) -> ChaCha8Rng {
        self.child(label).rng()
    }

    /// A `u32` seed for the `noise` crate generators.
    pub fn noise_seed(&self, label: &str) -> u32 {
        (self.child(label).seed & 0xFFFF_FFFF) as u32
    }
}

/// FNV-1a over the label bytes.
fn hash_label(label: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// SplitMix64-style finalizer mixing two words.
fn mix(a: u64, b: u64) -> u64 {
    let mut z = a ^ b.rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Helpers for drawing floats in ranges without going through `rand`'s distributions
/// (keeps the API surface small and stable). Import as `use orna::rng::Draw;`.
pub trait Draw {
    /// Uniform in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64;
    /// Uniform in `[-amount, amount]` around `center`.
    fn jitter(&mut self, center: f64, amount: f64) -> f64;
    /// `true` with probability `p`.
    fn chance(&mut self, p: f64) -> bool;
    /// Pick a random element.
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T;
    /// Approximately normal (sum of three uniforms), mean `mean`, std `std`.
    fn normal(&mut self, mean: f64, std: f64) -> f64;
}

impl<R: Rng> Draw for R {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }

    fn jitter(&mut self, center: f64, amount: f64) -> f64 {
        self.uniform(center - amount, center + amount)
    }

    fn chance(&mut self, p: f64) -> bool {
        self.uniform(0.0, 1.0) < p
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        assert!(!items.is_empty(), "pick from empty slice");
        let i = (self.next_u64() % items.len() as u64) as usize;
        &items[i]
    }

    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let s = self.uniform(-1.0, 1.0) + self.uniform(-1.0, 1.0) + self.uniform(-1.0, 1.0);
        // Three uniforms on [-1,1] have variance 1, so `s` already has unit std.
        mean + std * s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_are_reproducible_and_independent() {
        let t = SeedTree::new(42);
        let a1: Vec<u64> = (0..4).map(|_| t.rng_for("a").next_u64()).collect();
        let a2: Vec<u64> = (0..4).map(|_| t.rng_for("a").next_u64()).collect();
        assert_eq!(a1, a2);
        assert_ne!(t.rng_for("a").next_u64(), t.rng_for("b").next_u64());
        assert_ne!(t.index(0).seed(), t.index(1).seed());
        assert_ne!(
            SeedTree::new(1).child("x").seed(),
            SeedTree::new(2).child("x").seed()
        );
    }

    #[test]
    fn uniform_is_in_range() {
        let mut r = SeedTree::new(7).rng();
        for _ in 0..1000 {
            let v = r.uniform(-2.0, 3.0);
            assert!((-2.0..3.0).contains(&v));
        }
    }
}
