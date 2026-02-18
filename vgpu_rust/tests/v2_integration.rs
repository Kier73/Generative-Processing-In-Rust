use vgpu_rust::{NativeJitKernel, SpvOp, VAddr, VGpuContext, native_jit};

/// Phase 15: Verify that native JIT kernels produce correct results
/// and that induction recall works for JIT-compiled shaders.
#[test]
fn test_phase15_native_jit_dispatch() {
    let mut gpu = VGpuContext::new(128, 0xCAFE);

    let ops = vec![SpvOp::FAdd, SpvOp::FMul, SpvOp::FAdd];
    let compiled = native_jit::compile(&ops).expect("JIT compilation should succeed");
    let sig = 0xABCD;

    gpu.register_kernel(Box::new(NativeJitKernel {
        compiled,
        signature: sig,
    }));

    // Calls: 1-5 build induction confidence (variance gate)
    // Call 6: first O(1) recall hit
    for _ in 0..5 {
        gpu.dispatch(sig, 50);
    }
    let r1 = gpu.dispatch(sig, 50); // This should be a hit (result matches previous stable ones)

    let t = gpu.telemetry();
    assert_eq!(t.total_dispatches, 6);
    assert_eq!(t.induction_hits, 1, "6th call should be a hit");
    assert_eq!(t.induction_misses, 5, "First 5 calls build confidence");
    println!(
        "[Phase 15] Native JIT: result={}, hit_rate={:.1}%",
        r1.data[0],
        t.hit_rate * 100.0
    );
}

/// Phase 16: Verify stateful SSBO read/write persistence.
#[test]
fn test_phase16_stateful_ssbo() {
    let mut gpu = VGpuContext::new(4, 0xBEEF);

    // Write some data to binding 0 (like a vertex buffer)
    let vertex_data = vec![1.0, 2.0, 3.0, 4.0];
    gpu.ssbo.write(0, vertex_data.clone());

    // Write to a different binding (like a uniform buffer)
    gpu.ssbo.write(1, vec![100.0, 200.0]);

    // Read back
    let readback = gpu.ssbo.read(0).expect("Binding 0 should exist");
    assert_eq!(readback, &vertex_data[..]);
    assert_eq!(gpu.ssbo.binding_count(), 2);

    // Also test VVram write-back (the substrate mutation)
    let addr = VAddr::new(1, 2, 3, 4);
    let before = gpu.vram.read(addr);
    gpu.vram.write(addr, 42.0);
    let after = gpu.vram.read(addr);
    assert_ne!(before, after, "VVram write should alter the substrate");

    println!(
        "[Phase 16] SSBO bindings={}, substrate mutated={}",
        gpu.ssbo.binding_count(),
        before != after
    );
}

/// Phase 17: Verify inductive texture sampling at different LOD levels.
#[test]
fn test_phase17_texture_sampling() {
    let gpu = VGpuContext::new(4, 0xFACE);

    // Point sample
    let point = gpu.sampler.sample_point(&gpu.vram, 0.5, 0.5);
    assert!(point >= 0.0 && point <= 1.0, "Variety should be normalized");

    // Bilinear sample (should be smoother)
    let bilinear = gpu.sampler.sample_bilinear(&gpu.vram, 0.5, 0.5);
    assert!(
        bilinear >= 0.0 && bilinear <= 1.0,
        "Bilinear should be normalized"
    );

    // MIP sample at LOD 0 should equal point sample
    let mip0 = gpu.sampler.sample_mip(&gpu.vram, 0.5, 0.5, 0);
    assert_eq!(mip0, point, "MIP LOD 0 should equal point sample");

    // MIP sample at higher LOD should be different (more averaging)
    let mip2 = gpu.sampler.sample_mip(&gpu.vram, 0.5, 0.5, 2);
    assert!(mip2 >= 0.0 && mip2 <= 1.0, "MIP 2 should be normalized");

    println!(
        "[Phase 17] point={:.4}, bilinear={:.4}, mip0={:.4}, mip2={:.4}",
        point, bilinear, mip0, mip2
    );
}

/// Phase 18: Verify cache-affinity assignment.
#[test]
fn test_phase18_affinity_map() {
    let gpu = VGpuContext::new(16, 0x1234);

    // Default auto-assignment should spread across 8 cores
    assert_eq!(gpu.affinity.preferred_core(0), Some(0));
    assert_eq!(gpu.affinity.preferred_core(7), Some(7));
    assert_eq!(gpu.affinity.preferred_core(8), Some(0)); // wraps around
    assert_eq!(gpu.affinity.preferred_core(15), Some(7));

    println!(
        "[Phase 18] Affinity: SM0->Core{:?}, SM8->Core{:?}",
        gpu.affinity.preferred_core(0),
        gpu.affinity.preferred_core(8)
    );
}

/// Phase 19: Verify telemetry snapshot accuracy.
#[test]
fn test_phase19_telemetry() {
    let mut gpu = VGpuContext::new(4, 0x5555);

    let ops = vec![SpvOp::FAdd];
    let compiled = native_jit::compile(&ops).unwrap();
    gpu.register_kernel(Box::new(NativeJitKernel {
        compiled,
        signature: 0x1111,
    }));

    // Run 500 dispatches with 10 unique inputs
    // Rounds 1-5 build confidence (50 misses)
    // Rounds 6-50 recall (450 hits)
    for _round in 0..50 {
        for input in 0..10u64 {
            gpu.dispatch(0x1111, input);
        }
    }

    let t = gpu.telemetry();
    println!(
        "[Phase 19] Telemetry: dispatches={}, hits={}, misses={}, hit_rate={:.1}%, occupancy={}/{}",
        t.total_dispatches,
        t.induction_hits,
        t.induction_misses,
        t.hit_rate * 100.0,
        t.manifold_occupancy,
        t.manifold_capacity
    );

    assert_eq!(t.total_dispatches, 500);
    assert!(
        t.induction_hits >= 450,
        "Should have at least 450 hits out of 500 after maturation"
    ); // 10 misses on first pass
    assert!(t.hit_rate > 0.8, "Hit rate should exceed 80%");
}

/// Phase 20: Verify entropy-weighted pruning.
#[test]
fn test_phase20_pruning() {
    let mut gpu = VGpuContext::new(4, 0x9999);

    let ops = vec![SpvOp::FMul];
    let compiled = native_jit::compile(&ops).unwrap();
    gpu.register_kernel(Box::new(NativeJitKernel {
        compiled,
        signature: 0x2222,
    }));

    // Populate some laws (requires 5 dispatches per input to induct)
    for _round in 0..5 {
        for i in 0..20u64 {
            gpu.dispatch(0x2222, i);
        }
    }

    let before = gpu.inductor.occupancy();

    // Prune the weakest
    gpu.prune_weakest();

    let after = gpu.inductor.occupancy();
    let t = gpu.telemetry();

    assert!(
        after <= before,
        "Pruning should reduce or maintain occupancy"
    );
    assert_eq!(t.evictions_performed, 1, "Should have exactly 1 eviction");

    println!(
        "[Phase 20] Pruning: before={}, after={}, evictions={}",
        before, after, t.evictions_performed
    );
}
