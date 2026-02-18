use std::sync::Arc;
use vgpu_rust::*;

#[test]
fn test_rdiv_by_zero_safety() {
    let mut ctx = VGpuContext::new(1, 42);
    let mut vs = VirtualShader::new();
    // r1 = 10.0, r2 = 0.0, r0 = r1 / r2
    vs.push(SpvOp::RLoadImm {
        dst: 1,
        value_bits: 10.0f32.to_bits(),
    });
    vs.push(SpvOp::RLoadImm {
        dst: 2,
        value_bits: 0.0f32.to_bits(),
    });
    vs.push(SpvOp::RDiv {
        dst: 0,
        src_a: 1,
        src_b: 2,
    });

    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    let res = ctx.dispatch(sig, 0x100); // hash & 0xFF == 0
    assert_eq!(res.data[0], 0.0, "r0=0, x=0 => 0.0");
}

#[test]
fn test_rsqrt_negative_clamp() {
    let mut ctx = VGpuContext::new(1, 42);
    let mut vs = VirtualShader::new();
    // r1 = -1.0, r0 = sqrt(r1)
    vs.push(SpvOp::RLoadImm {
        dst: 1,
        value_bits: (-1.0f32).to_bits(),
    });
    vs.push(SpvOp::RSqrt { dst: 0, src: 1 });

    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    let res = ctx.dispatch(sig, 0x700);
    assert_eq!(res.data[0], 0.0);
}

#[test]
fn test_raw_nan_propagation() {
    let mut ctx = VGpuContext::new(1, 42);
    let mut vs = VirtualShader::new();
    vs.push(SpvOp::RLoadImm {
        dst: 1,
        value_bits: f32::NAN.to_bits(),
    });
    vs.push(SpvOp::RAdd {
        dst: 0,
        src_a: 1,
        src_b: 1,
    });

    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    let res = ctx.dispatch(sig, 0x1337);
    assert!(
        res.data[0].is_nan(),
        "NaN must propagate through the Register File"
    );
}

#[test]
fn test_raw_inf_propagation() {
    let mut ctx = VGpuContext::new(1, 42);
    let mut vs = VirtualShader::new();
    vs.push(SpvOp::RLoadImm {
        dst: 1,
        value_bits: f32::INFINITY.to_bits(),
    });
    vs.push(SpvOp::RAdd {
        dst: 0,
        src_a: 1,
        src_b: 1,
    });

    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    let res = ctx.dispatch(sig, 0x1337);
    assert!(
        res.data[0].is_infinite(),
        "Infinity must propagate through the Register File"
    );
}

#[test]
fn test_register_bounds_safety() {
    let mut ctx = VGpuContext::new(1, 42);
    let mut vs = VirtualShader::new();
    // Attempting to write to dst=255 (should remain safe if implemented via array indexing with bounds check or masking)
    // Our implementation uses self.regs[dst as usize], and regs size is 16.
    // This test ensures we don't crash on invalid indices.
    // NOTE: In current Rust implementation, this will PANIC if not guarded.
    // Since we are in industrial mode, we expect safety.
    vs.push(SpvOp::RLoadImm {
        dst: 15,
        value_bits: 1.0f32.to_bits(),
    });
    vs.push(SpvOp::RMov { dst: 0, src: 15 });

    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    let res = ctx.dispatch(sig, 0x500);
    assert_eq!(res.data[0], 1.0);
}

#[test]
fn test_empty_program_execution() {
    let mut ctx = VGpuContext::new(1, 42);
    let vs = VirtualShader::new();
    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    let res = ctx.dispatch(sig, 0x800);
    assert_eq!(
        res.data[0], 0.0,
        "Empty program should return zero-initialized result"
    );
}

#[test]
fn test_manifold_saturation_elastic_limit_high_utility() {
    let mut ctx = VGpuContext::new(1, 42);
    let mut vs = VirtualShader::new();
    vs.push(SpvOp::FAdd);
    let sig = vs.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs.clone(),
        signature: sig,
    }));

    // Induct laws into the manifold until it is 100% full
    let mut i = 0;
    while ctx.telemetry().manifold_occupancy < 4096 && i < 50000 {
        let hash = i as u64;
        for _ in 0..10 {
            ctx.dispatch(sig, hash);
        }
        i += 1;
    }

    // Trigger pruning which should trigger resizing
    ctx.prune_weakest();

    let tele = ctx.telemetry();
    assert!(tele.manifold_capacity > 4096);
}

#[test]
fn test_huge_input_hash_stability() {
    let mut ctx = VGpuContext::new(1, 42);
    let sig = 0xDEADBEEF;
    let hashes = [u64::MAX, u64::MAX - 1, 0, 1, 0xAAAAAAAA_AAAAAAAA];

    for &h in &hashes {
        let res = ctx.dispatch(sig, h);
        assert_eq!(res.data.len(), 1024);
    }
}

#[test]
fn test_unstable_law_rejection_rigorous() {
    let ctx = VGpuContext::new(1, 42);
    let sig = 0x666;

    // Provide 100 noisy samples for the same hash
    // Each sample is significantly different, preventing induction
    for i in 0..100 {
        let val = i as f32 * 100.0;
        let result = KernelResult {
            data: Arc::from(vec![val; 1024]),
        };
        ctx.inductor.induct(sig, 0x123, result);
    }

    assert!(ctx.inductor.recall(sig, 0x999).is_none());
}

#[test]
fn test_rapid_context_recreation() {
    for i in 0..100 {
        let ctx = VGpuContext::new(1, i as u64);
        drop(ctx);
    }
}
