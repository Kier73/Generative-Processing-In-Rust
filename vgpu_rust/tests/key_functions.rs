use std::sync::Arc;
use vgpu_rust::*;

#[test]
fn test_manifold_key_hashing() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let k1 = ManifoldKey { sig: 1, hash: 100 };
    let k2 = ManifoldKey { sig: 1, hash: 101 };

    let mut h1 = DefaultHasher::new();
    k1.hash(&mut h1);
    let r1 = h1.finish();

    let mut h2 = DefaultHasher::new();
    k2.hash(&mut h2);
    let r2 = h2.finish();

    assert_ne!(r1, r2, "Different keys must produce different hashes");
}

#[test]
fn test_spectral_utility_math_verification() {
    let mut tracker = EfficiencyTracker::new(4096);
    let key = ManifoldKey {
        sig: 0xDEAD,
        hash: 0xBEEF,
    };

    // 10 hits, 100 energy each
    for _ in 0..10 {
        tracker.record_hit(key, 100);
    }

    // Consistent output history (1.0, 1.0, 1.0) -> Entropy = 0
    tracker.record_output(key, 1.0);
    tracker.record_output(key, 1.0);
    tracker.record_output(key, 1.0);

    let u = tracker.spectral_utility(&key);
    // U = (Hits * TotalEnergy) / (Entropy + 1)
    // U = (10 * 1000) / (0 + 1) = 10000
    assert_eq!(u, 10000.0);

    // Add noise to history -> Entropy increases
    tracker.record_output(key, 2.0);
    tracker.record_output(key, 3.0);
    let u_noisy = tracker.spectral_utility(&key);
    assert!(u_noisy < 10000.0, "Noise must reduce spectral utility");
}

#[test]
fn test_pruning_determinism_priority() {
    let mut ctx = VGpuContext::new(1, 42);
    // Fill two slots
    let sig = 0xABC;
    let hash1 = 0x1;
    let hash2 = 0x2;

    let key1 = ManifoldKey { sig, hash: hash1 };
    let key2 = ManifoldKey { sig, hash: hash2 };

    // Mature both laws (requires 5 observations)
    for _ in 0..5 {
        ctx.inductor.induct(
            sig,
            hash1,
            KernelResult {
                data: Arc::from(vec![0.0; KERNEL_DISPATCH_SIZE]),
            },
        );
        ctx.inductor.induct(
            sig,
            hash2,
            KernelResult {
                data: Arc::from(vec![0.0; KERNEL_DISPATCH_SIZE]),
            },
        );
    }

    // Ensure we don't accidentally prune these by giving them very high utility
    ctx.inductor
        .efficiency
        .write()
        .unwrap()
        .record_hit(key1, 10_000);
    ctx.inductor
        .efficiency
        .write()
        .unwrap()
        .record_hit(key2, 5_000);

    let t_before = ctx.telemetry();
    // ctx.prune_weakest(); // Only called later
    // let t_after = ctx.telemetry(); // Removed unused

    // With sparse map, pruning only happens when explicitly requested.
    // However, if we didn't add any *other* low utility items, correct behavior depends on implementation.
    // If the map only has these two high utility items, pruning might skip them if they are above threshold,
    // or prune the lower of the two if forced.
    // In current implementation prune_weakest forces eviction if above capacity, but here we are not above capacity.
    // But specific test logic: "In this test, we have many empty slots with utility 0" (FROM OLD TEST).
    // NEW REALITY: Sparse map => NO empty slots with utility 0 exist unless we insert them!

    // So we must insert a "victim" low utility law to verifying pruning works.
    let victim_sig = 0xBAD;
    let victim_hash = 0xDEAD;
    // victim_key removed as it was unused

    ctx.inductor.induct(
        victim_sig,
        victim_hash,
        KernelResult {
            data: Arc::from(vec![0.0; KERNEL_DISPATCH_SIZE]),
        },
    );
    // Register in efficiency tracker so it can be found (utility 0)
    ctx.inductor.efficiency.write().unwrap().record_output(
        ManifoldKey {
            sig: victim_sig,
            hash: victim_hash,
        },
        0.0,
    );
    // Don't give it any hits/utility

    ctx.prune_weakest();
    let t_after_prune = ctx.telemetry();

    // Verify eviction count increased
    assert_eq!(
        t_after_prune.evictions_performed,
        t_before.evictions_performed + 1
    );

    // Check if victim was evicted
    assert!(
        ctx.inductor.recall(victim_sig, victim_hash).is_none(),
        "Low utility victim should be evicted"
    );

    // High utility must remain
    assert!(
        ctx.inductor.recall(sig, hash1).is_some(),
        "High-utility entry (hash1) must remain"
    );
    assert!(
        ctx.inductor.recall(sig, hash2).is_some(),
        "High-utility entry (hash2) must remain"
    );
}

#[test]
fn test_variance_gate_rejection() {
    let ctx = VInductor::new(0);
    let sig = 0xAA;
    let hash = 0xBB;

    // Provide 5 wildly different results
    for i in 0..5 {
        ctx.induct(
            sig,
            hash,
            KernelResult {
                data: Arc::from(vec![i as f32; KERNEL_DISPATCH_SIZE]),
            },
        );
    }

    assert!(
        ctx.recall(sig, hash).is_none(),
        "Variance gate should block unstable induction"
    );

    // Provide 40 stable results to flush the noise
    for _ in 0..40 {
        ctx.induct(
            sig,
            hash,
            KernelResult {
                data: Arc::from(vec![10.0; KERNEL_DISPATCH_SIZE]),
            },
        );
    }
    assert!(
        ctx.recall(sig, hash).is_some(),
        "Stable induction should pass after flushing noise"
    );
}

#[test]
fn test_signature_uniqueness() {
    let mut vs1 = VirtualShader::new();
    vs1.push(SpvOp::FAdd);

    let mut vs2 = VirtualShader::new();
    vs2.push(SpvOp::FSub);

    assert_ne!(vs1.generate_signature(), vs2.generate_signature());
}

#[test]
fn test_efficiency_tracker_hit_rate_logic() {
    let mut ctx = VGpuContext::new(1, 42);
    let sig = 0x1;
    let mut vs = VirtualShader::new();
    for _ in 0..25 {
        vs.push(SpvOp::FAdd);
    }
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    // 10 dispatches
    for _ in 0..5 {
        ctx.dispatch(sig, 0x123);
    } // 5 misses
    for _ in 0..5 {
        ctx.dispatch(sig, 0x123);
    } // 5 hits

    let t = ctx.telemetry();
    assert_eq!(t.induction_hits, 5);
    // Total dispatches = 10. Misses = 10 - 5 = 5.
    assert_eq!(t.induction_misses, 5);
    assert_eq!(t.hit_rate, 0.5);
}

#[test]
fn test_dissonance_control_bayesian_block() {
    let mut dc = DissonanceControl::new(0.1, 1.0); // Strict threshold
    assert!(dc.check(1.0, 1.05).0); // 0.05 < 0.1 PASS
    assert!(!dc.check(1.0, 1.2).0); // 0.2 > 0.1 FAIL
}

#[test]
fn test_virtual_amplification_coherence_check() {
    let mut va = VAmplifier::new(3.9);
    let nodes = va.bifurcate(4);
    assert_eq!(nodes.len(), 16);
}

#[test]
fn test_vram_variety_persistence() {
    let vram = VVram::new(12345);
    let addr = VAddr::new(10, 20, 30, 40);
    let val1 = vram.read(addr);
    let val2 = vram.read(addr);
    assert_eq!(
        val1, val2,
        "Vram reads must be deterministic for same seed/addr"
    );
}
