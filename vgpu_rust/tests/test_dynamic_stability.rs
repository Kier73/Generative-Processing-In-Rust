use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use vgpu_rust::{KERNEL_DISPATCH_SIZE, Kernel, KernelResult, VGpuContext};

pub struct MockNoisyKernel {
    pub sig: u64,
    pub counter: AtomicU64,
}

impl Kernel for MockNoisyKernel {
    fn signature(&self) -> u64 {
        self.sig
    }
    fn is_deterministic(&self) -> bool {
        false
    }
    fn cost(&self) -> u32 {
        100
    }
    fn execute(&self, _input_hash: u64) -> KernelResult {
        let val = self.counter.fetch_add(1, Ordering::Relaxed) as f32;
        // This will create a sequence 0, 1, 2, 3... which is very high variance
        let data = vec![val; KERNEL_DISPATCH_SIZE];
        KernelResult {
            data: Arc::from(data),
        }
    }
}

#[test]
fn test_dynamic_induction_noise_resistance() {
    let mut ctx = VGpuContext::new(4, 42);
    let sig = 0xBADBEEF;
    let input_hash = 12345;

    // Register a noisy kernel
    ctx.register_kernel(Box::new(MockNoisyKernel {
        sig,
        counter: AtomicU64::new(0),
    }));

    // Dispatch many times. Because it's noisy (0, 1, 2, 3...), it should NEVER induct.
    for _ in 0..50 {
        ctx.dispatch(sig, input_hash);
    }

    assert_eq!(
        ctx.induction_events(),
        0,
        "Noisy process should not have been inducted"
    );
}

#[test]
fn test_statistical_stability_convergence() {
    let mut ctx = VGpuContext::new(4, 42);
    let sig = 0x600D;
    let input_hash = 0xCAFE;

    // We need a stable but non-deterministic kernel (to trigger induction path)
    pub struct StableKernel {
        pub sig: u64,
    }
    impl Kernel for StableKernel {
        fn signature(&self) -> u64 {
            self.sig
        }
        fn is_deterministic(&self) -> bool {
            false
        }
        fn execute(&self, _h: u64) -> KernelResult {
            KernelResult {
                data: Arc::from([10.0; KERNEL_DISPATCH_SIZE]),
            }
        }
    }

    ctx.register_kernel(Box::new(StableKernel { sig }));

    // 1. Perfectly stable data
    // Should induct after 3 samples (our minimum)
    for _ in 0..2 {
        ctx.dispatch(sig, input_hash);
    }
    assert_eq!(
        ctx.induction_events(),
        0,
        "Should not induct under 3 samples"
    );

    ctx.dispatch(sig, input_hash);
    assert_eq!(
        ctx.induction_events(),
        1,
        "Should induct exactly at 3 samples for perfectly stable data"
    );
}
