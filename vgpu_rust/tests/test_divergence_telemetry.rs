use vgpu_rust::{SpvOp, VGpuContext, VirtualShader};

#[test]
fn test_divergence_telemetry_reporting() {
    // 1. Setup context with high verification rate for testing
    let mut ctx = VGpuContext::new(1, 42);
    ctx.dissonance_control.validation_rate = 1.0; // Force verification on every hit

    // 2. Register a Law of Truth
    let mut vs_truth = VirtualShader::new();
    for _ in 0..25 {
        vs_truth.push(SpvOp::FAdd);
    }
    let sig = vs_truth.generate_signature();
    ctx.register_kernel(Box::new(vgpu_rust::ShaderKernel {
        shader: vs_truth,
        signature: sig,
    }));

    // 3. Train the induction engine (N=5 to ground the Law)
    for _ in 0..5 {
        let _ = ctx.dispatch(sig, 1);
    }
    assert!(
        ctx.induction_events() > 0,
        "Induction failed to ground Truth"
    );

    // 4. Manually corrupt the induction manifold (Simulate a silent breach)
    // We'll replace the grounded result with something slightly wrong
    {
        let key = vgpu_rust::ManifoldKey { sig, hash: 1 };
        let mut manifold = ctx.inductor.manifold.write().unwrap();
        if let Some(res) = manifold.get_mut(&key) {
            let mut data = res.data.to_vec();
            data[0] += 0.5; // Introduce a 0.5 delta
            res.data = std::sync::Arc::from(data);
        }
    }

    // 5. Dispatch and check Telemetry
    // Since validation_rate is 1.0, it will verify against ground truth immediately.
    let _ = ctx.dispatch(sig, 1);

    let delta = ctx.last_divergence();
    println!("Measured Divergence Delta: {}", delta);

    // The delta should be exactly 0.5 (or very close)
    assert!(
        delta > 0.49 && delta < 0.51,
        "Divergence reporting is inaccurate: {}",
        delta
    );

    println!("Success: Divergence delta accurately reported via telemetry.");
}
