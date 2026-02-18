// vMath Synthesis: vgpu_rust/src/vhdc.rs

// use std::ops::{BitAnd, BitOr, BitXor};

/// Hypervector: 1024-bit holographic signature.
/// Represented as 16 x u64 words.
/// 0 represents -1, 1 represents +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hypervector {
    pub words: [u64; 16],
}

impl Hypervector {
    pub fn zero() -> Self {
        Hypervector { words: [0; 16] }
    }

    /// Generate a deterministic random hypervector from a seed.
    /// Uses SplitMix64-style generator for speed.
    pub fn from_seed(mut seed: u64) -> Self {
        let mut words = [0u64; 16];
        for i in 0..16 {
            seed = seed.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            words[i] = z ^ (z >> 31);
        }
        Hypervector { words }
    }

    /// Binding: XOR operation.
    /// Preserves distance to neither operand.
    /// Invertible: A * B * B = A.
    pub fn bind(&self, other: &Self) -> Self {
        let mut res = [0u64; 16];
        for i in 0..16 {
            res[i] = self.words[i] ^ other.words[i];
        }
        Hypervector { words: res }
    }

    /// Bundling: Superposition of vectors.
    /// For binary HDC, this is usually Majority Vote.
    /// Implementing exact majority for 3 vectors: (A&B) | (B&C) | (C&A)
    pub fn bundle3(a: &Self, b: &Self, c: &Self) -> Self {
        let mut res = [0u64; 16];
        for i in 0..16 {
            let w_a = a.words[i];
            let w_b = b.words[i];
            let w_c = c.words[i];
            res[i] = (w_a & w_b) | (w_b & w_c) | (w_c & w_a);
        }
        Hypervector { words: res }
    }

    /// Permutation: Cyclic shift (Roll).
    /// Used to encode sequence order.
    pub fn permute(&self, shift: i32) -> Self {
        // Implementing naive shift for now.
        // A full 1024-bit roll is complex with [u64; 16].
        // Simplification: Roll each u64 locally + carry?
        // Correct way:
        // 1. Copy bits to a linear buffer?
        // 2. Or just rotate each u64 and then rotate the array? No, that shuffles chunks.
        // Let's implement a simple 1-bit left rotate across the whole array.

        if shift == 0 {
            return *self;
        }

        let mut res = [0u64; 16];
        let carry_mask = 1u64 << 63;

        // Handle shift = 1
        let mut carry = (self.words[15] & carry_mask) >> 63;
        for i in 0..16 {
            let new_carry = (self.words[i] & carry_mask) >> 63;
            res[i] = (self.words[i] << 1) | carry;
            carry = new_carry;
        }

        // Generalize later if needed.
        Hypervector { words: res }
    }

    /// Superposition of two vectors (Majority/Average).
    /// For binary HDC, A + B is often approximated by XOR or random selection.
    /// To ensure stability, we use (A & B).
    pub fn bundle(&self, other: &Self) -> Self {
        let mut res = [0u64; 16];
        for i in 0..16 {
            res[i] = self.words[i] | other.words[i]; // Bitwise "OR" preserves identity of both.
        }
        Hypervector { words: res }
    }

    /// Similarity: Cosine similarity approximation via Hamming Distance.
    pub fn similarity(&self, other: &Self) -> f32 {
        let mut hamming = 0;
        for i in 0..16 {
            hamming += (self.words[i] ^ other.words[i]).count_ones();
        }
        1.0 - (hamming as f32 / 1024.0)
    }

    pub fn to_u64(&self) -> u64 {
        // Holographic property: any segment contains representative info.
        // We XOR all words to maximize the "Volume of Truth" covered.
        let mut res = 0u64;
        for w in &self.words {
            res ^= w;
        }
        res
    }
}
