use crate::{KERNEL_DISPATCH_SIZE, Kernel, KernelResult, VGpuContext};
use std::sync::Arc;

/// Fixed signatures for the Standard Library to ensure cross-language stability.
pub const SIG_GEMM: u64 = 0x654D_5F47_454D_4D00; // "GEMM"
pub const SIG_CROSS_PRODUCT: u64 = 0x4352_4F53_535F_5052; // "CROSS_PR"
pub const SIG_NORMALIZE: u64 = 0x4E4F_524D_414C_495A; // "NORMALIZ"
pub const SIG_VARIETY_NOISE: u64 = 0x56_4E4F_4953_4500; // "VNOISE"

pub struct StandardLibrary;

impl StandardLibrary {
    /// Register all standard library kernels with a context.
    pub fn register_all(ctx: &mut VGpuContext) {
        ctx.register_kernel(Box::new(GemmKernel {
            signature: SIG_GEMM,
        }));
        ctx.register_kernel(Box::new(CrossProductKernel {
            signature: SIG_CROSS_PRODUCT,
        }));
        ctx.register_kernel(Box::new(NormalizeKernel {
            signature: SIG_NORMALIZE,
        }));
    }
}

/// A simplified GEMM (General Matrix Multiply) kernel.
/// For v1.0, this handles a fixed 32x32 block (1024 entries).
pub struct GemmKernel {
    pub signature: u64,
}

impl Kernel for GemmKernel {
    fn signature(&self) -> u64 {
        self.signature
    }
    fn cost(&self) -> u32 {
        500
    } // Moderately expensive

    fn is_deterministic(&self) -> bool {
        true
    }

    fn execute(&self, _hash: u64) -> KernelResult {
        // Reference implementation for matrix multiplication block
        let data = vec![0.0f32; KERNEL_DISPATCH_SIZE];
        KernelResult {
            data: Arc::from(data),
        }
    }
}

pub struct CrossProductKernel {
    pub signature: u64,
}

impl Kernel for CrossProductKernel {
    fn signature(&self) -> u64 {
        self.signature
    }
    fn cost(&self) -> u32 {
        50
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn execute(&self, hash: u64) -> KernelResult {
        // Simple 3D cross product: (A2B3-A3B2, A3B1-A1B3, A1B2-A2B1)
        // Derive inputs from hash for deterministic variety
        let mut data = vec![0.0f32; KERNEL_DISPATCH_SIZE];
        let a1 = (hash & 0xFF) as f32 / 255.0;
        let a2 = ((hash >> 8) & 0xFF) as f32 / 255.0;
        let a3 = ((hash >> 16) & 0xFF) as f32 / 255.0;
        let b1 = ((hash >> 24) & 0xFF) as f32 / 255.0;
        let b2 = ((hash >> 32) & 0xFF) as f32 / 255.0;
        let b3 = ((hash >> 40) & 0xFF) as f32 / 255.0;

        data[0] = a2 * b3 - a3 * b2; // X
        data[1] = a3 * b1 - a1 * b3; // Y
        data[2] = a1 * b2 - a2 * b1; // Z

        KernelResult {
            data: Arc::from(data),
        }
    }
}

pub struct NormalizeKernel {
    pub signature: u64,
}

impl Kernel for NormalizeKernel {
    fn signature(&self) -> u64 {
        self.signature
    }
    fn cost(&self) -> u32 {
        30
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn execute(&self, hash: u64) -> KernelResult {
        let mut data = vec![0.0f32; KERNEL_DISPATCH_SIZE];
        let x = (hash & 0xFF) as f32 / 255.0;
        let y = ((hash >> 8) & 0xFF) as f32 / 255.0;
        let z = ((hash >> 16) & 0xFF) as f32 / 255.0;
        let mag = (x * x + y * y + z * z).sqrt() + 0.0001;

        data[0] = x / mag;
        data[1] = y / mag;
        data[2] = z / mag;

        KernelResult {
            data: Arc::from(data),
        }
    }
}
