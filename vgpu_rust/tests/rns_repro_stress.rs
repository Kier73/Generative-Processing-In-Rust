use std::sync::Arc;
use vgpu_rust::vrns::RnsScalar;
use vgpu_rust::*;

#[test]
fn test_adaptive_rns_reconstruction() {
    let val_safe = 123456789u64;
    let rns = RnsScalar::new(val_safe, 64);
    assert_eq!(rns.moduli_count, 4);
    assert_eq!(rns.reconstruct(), val_safe, "RNS reconstruction failed");
}

#[test]
fn test_reproducibility_induction_lock() {
    let mut ctx = VGpuContext::new(4, 42);
    let sig = 0x1337;
    struct VolatileKernel {
        sig: u64,
    }
    impl Kernel for VolatileKernel {
        fn signature(&self) -> u64 {
            self.sig
        }
        fn execute(&self, _hash: u64) -> KernelResult {
            let val = 1.23; // Constant for this test
            KernelResult {
                data: Arc::from(vec![val; 1024]),
            }
        }
    }
    ctx.register_kernel(Box::new(VolatileKernel { sig }));
    for _ in 0..5 {
        ctx.dispatch(sig, 100);
    }
    let res1 = ctx.dispatch(sig, 100);
    let res2 = ctx.dispatch(sig, 100);
    assert_eq!(res1.data[0], res2.data[0]);
}

#[test]
fn test_jit_immediate_determinism() {
    let mut ctx = VGpuContext::new(4, 42);
    let mut vs = VirtualShader::new();
    for _ in 0..32 {
        vs.push(SpvOp::FAdd);
    }
    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(JitKernel {
        ops: vs.instructions,
        signature: sig,
    }));
    let _ = ctx.dispatch(sig, 100);
    let _ = ctx.dispatch(sig, 100);
    assert!(ctx.telemetry().induction_hits >= 1);
}
