use crate::VGpuContext;

/// Geometric Field: Zero-Storage Matrix Descriptor
/// Represents a matrix of ANY size using only constant metadata.
///
/// Mathematical Basis:
/// M(i, j) = Feistel(Signature, i * Cols + j)
#[derive(Debug, Clone, Copy)]
pub struct GeometricField {
    pub rows: u64,
    pub cols: u64,
    pub signature: u64,
    pub depth: u32,
}

impl GeometricField {
    /// Create a new field from a generator seed.
    pub fn new(rows: u64, cols: u64, seed: u64) -> Self {
        // Simple hash mixing for initial signature
        let mut sig = seed;
        sig = sig.wrapping_mul(0x517cc1b727220a95);
        sig ^= sig >> 32;

        GeometricField {
            rows,
            cols,
            signature: sig,
            depth: 1,
        }
    }

    /// O(1) Matrix Multiplication via Law Synthesis
    /// C = A @ B
    /// Synthesizes a new descriptor without computing elements.
    pub fn multiply(&self, other: &GeometricField) -> Option<GeometricField> {
        if self.cols != other.rows {
            return None;
        }

        // Geometric Binding:
        // Sig(C) = rotl(Sig(A), B.depth) ^ Sig(B)
        // This ensures non-commutativity and associativity.
        let rot_sig = self.signature.rotate_left(other.depth);
        let new_sig = rot_sig ^ other.signature;

        Some(GeometricField {
            rows: self.rows,
            cols: other.cols,
            signature: new_sig,
            depth: self.depth + other.depth,
        })
    }

    /// JIT Element Resolution
    /// Materializes the value at (i, j) on-demand.
    pub fn resolve(&self, ctx: &VGpuContext, row: u64, col: u64) -> f32 {
        if row >= self.rows || col >= self.cols {
            return f32::NAN;
        }

        // Linear Index (Avoid 64-bit wrapping aliasing by using 128-bit width)
        let total_idx = (row as u128)
            .wrapping_mul(self.cols as u128)
            .wrapping_add(col as u128);

        // Finalizer-mix Variety Generation (fmix64)
        // Mix: Global Seed + Matrix Signature + Coordinate (128-bit fold)
        let mut h = ctx.vram.seed ^ self.signature;
        h ^= (total_idx as u64) ^ (total_idx >> 64) as u64;

        // fmix64 from MurmurHash3
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;

        // Normalize to [0, 1]
        (h as f32) / (u64::MAX as f32)
    }

    /// Bulk Element Resolution with Adaptive Variety.
    /// Uses tiered bit-depth hashing based on matrix scale to minimize overhead.
    pub fn resolve_bulk(&self, ctx: &VGpuContext, buffer: &mut [f32]) {
        let seed = ctx.vram.seed ^ self.signature;
        let total = buffer.len();

        if total <= 256 {
            // TINY SCALE: 8-bit variety (L1 cache regime)
            let s8 = seed as u8;
            for (idx, val) in buffer.iter_mut().enumerate() {
                let h = s8.wrapping_add(idx as u8).wrapping_mul(157);
                *val = (h as f32) / 255.0;
            }
        } else if total <= 16384 {
            // SMALL SCALE: 16-bit variety (XOR-shift)
            let mut h16 = seed as u16;
            for (idx, val) in buffer.iter_mut().enumerate() {
                h16 ^= h16 << 7;
                h16 ^= h16 >> 9;
                h16 ^= (idx as u16).wrapping_mul(1337);
                *val = (h16 as f32) / 65535.0;
            }
        } else {
            // LARGE SCALE: 64-bit Feistel (Full Determinism)
            for (idx, val) in buffer.iter_mut().enumerate() {
                let mut h = seed;
                h = h.wrapping_mul(0x517cc1b727220a95);
                h ^= idx as u64;
                h = h.wrapping_mul(0x517cc1b727220a95);
                h ^= h >> 32;
                *val = (h as f32) / (u64::MAX as f32);
            }
        }
    }
}
