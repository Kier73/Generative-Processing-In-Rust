use crate::{KERNEL_DISPATCH_SIZE, KernelResult, VGpuContext};
use std::sync::Arc;

/// Inductive GEMM Engine (32x32 Tiles)
/// Matrix A (M x K) @ Matrix B (K x N) = Matrix C (M x N)
pub struct InductiveGemm;

impl InductiveGemm {
    const TILE_SIZE: usize = 32;

    /// Perform matrix multiplication with inductive tile skip
    pub fn multiply(
        ctx: &mut VGpuContext,
        a: &[f32],
        b: &[f32],
        m: usize,
        k_dim: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut c = vec![0.0; m * n];

        // Iterate over tiles
        for i_tile in (0..m).step_by(Self::TILE_SIZE) {
            for j_tile in (0..n).step_by(Self::TILE_SIZE) {
                // We are calculating Tile C[i_tile..i_tile+32, j_tile..j_tile+32]
                // This is the sum of Tile_A[i, k] * Tile_B[k, j] over k_tiles
                let mut accumulated_tile = vec![0.0; KERNEL_DISPATCH_SIZE];

                for k_tile in (0..k_dim).step_by(Self::TILE_SIZE) {
                    // 1. Generate Structural Signature for this Tile Pair
                    // In a production system, this would be a high-quality hash of the input slices.
                    let tile_a_hash = Self::hash_tile(a, i_tile, k_tile, k_dim);
                    let tile_b_hash = Self::hash_tile(b, k_tile, j_tile, n);
                    let input_hash = tile_a_hash ^ tile_b_hash.rotate_left(13);

                    // Gemm Signature (Static ID for GEMM operations)
                    const GEMM_SIG: u64 = 0x654D4D5F54494C45; // "GEMM_TILE"

                    // 2. Inductive Recall (ZERO COMPUTE if hit)
                    if let Some(res) = ctx.inductor.recall(GEMM_SIG, input_hash) {
                        for idx in 0..KERNEL_DISPATCH_SIZE {
                            accumulated_tile[idx] += res.data[idx];
                        }
                    } else {
                        // 3. Fallback: Native SIMD Dot Product
                        let computed =
                            Self::compute_tile_product_simd(a, b, i_tile, k_tile, j_tile, k_dim, n);

                        // 4. Induct the new law
                        ctx.inductor.induct(
                            GEMM_SIG,
                            input_hash,
                            KernelResult {
                                data: Arc::from(computed.clone()),
                            },
                        );

                        for idx in 0..KERNEL_DISPATCH_SIZE {
                            accumulated_tile[idx] += computed[idx];
                        }
                    }
                }

                // Scatter accumulated tile into result matrix C
                Self::scatter_tile(&accumulated_tile, &mut c, i_tile, j_tile, n, m, n);
            }
        }

        c
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn hash_tile_avx2(data: &[f32], r: usize, c: usize, stride: usize) -> u64 {
        use std::arch::x86_64::*;
        let mut h = unsafe { _mm256_setzero_ps() };
        let m_prime = unsafe { _mm256_set1_ps(31.0) };

        for i in 0..Self::TILE_SIZE {
            let row_offset = (r + i) * stride + c;
            // Load 8 floats (32 bytes) at a time using AVX2
            for j in (0..Self::TILE_SIZE).step_by(8) {
                let chunk = unsafe { _mm256_loadu_ps(data.as_ptr().add(row_offset + j)) };
                h = unsafe { _mm256_add_ps(_mm256_mul_ps(h, m_prime), chunk) };
            }
        }

        // Horizontal fold of the 256-bit register into a 64-bit hash
        let mut res = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(res.as_mut_ptr(), h) };
        let mut final_h = 0u64;
        for val in res {
            final_h = final_h
                .wrapping_mul(0x517cc1b727220a95)
                .wrapping_add(val.to_bits() as u64);
        }
        final_h
    }

    fn hash_tile(data: &[f32], r: usize, c: usize, stride: usize) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::hash_tile_avx2(data, r, c, stride) };
            }
        }

        // Scalar Fallback
        let mut h = 0u64;
        for i in 0..Self::TILE_SIZE {
            let row_offset = (r + i) * stride + c;
            for j in 0..Self::TILE_SIZE {
                h = h
                    .wrapping_mul(31)
                    .wrapping_add(data[row_offset + j].to_bits() as u64);
            }
        }
        h
    }

    fn compute_tile_product_simd(
        a: &[f32],
        b: &[f32],
        ri: usize,
        rk: usize,
        cj: usize,
        k_stride: usize,
        n_stride: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0; KERNEL_DISPATCH_SIZE];
        // Standard (but SIMD-friendly) tile multiplication
        for i in 0..Self::TILE_SIZE {
            for k in 0..Self::TILE_SIZE {
                let a_val = a[(ri + i) * k_stride + (rk + k)];
                for j in 0..Self::TILE_SIZE {
                    let b_val = b[(rk + k) * n_stride + (cj + j)];
                    out[i * Self::TILE_SIZE + j] += a_val * b_val;
                }
            }
        }
        out
    }

    fn scatter_tile(
        tile: &[f32],
        c_mat: &mut [f32],
        r: usize,
        c: usize,
        stride: usize,
        rows: usize,
        cols: usize,
    ) {
        for i in 0..Self::TILE_SIZE {
            for j in 0..Self::TILE_SIZE {
                if r + i < rows && c + j < cols {
                    c_mat[(r + i) * stride + (c + j)] = tile[i * Self::TILE_SIZE + j];
                }
            }
        }
    }
}
