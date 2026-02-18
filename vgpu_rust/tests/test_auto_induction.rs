use std::time::Instant;
use vgpu_rust::{SpvOp, VGpuContext, VirtualShader};

#[test]
fn test_autonomous_v_induction() {
    let mut ctx = VGpuContext::new(4, 42);

    // Define a "Raw" algorithm (FAdd repeated)
    let mut vs = VirtualShader::new();
    for _ in 0..25 {
        vs.push(SpvOp::FAdd);
    }
    let sig = vs.generate_signature();

    // Register it as a ShaderKernel (Interpreted, not Law yet)
    ctx.register_kernel(Box::new(vgpu_rust::ShaderKernel {
        shader: vs,
        signature: sig,
    }));

    println!("\n{}", "=".repeat(60));
    println!(" VGPU AUTONOMOUS INDUCTION: PROCESS PROMOTION ");
    println!("{}", "=".repeat(60));

    println!("\n--- Phase 1: The Training Interval (N=5) ---");
    // Dispatch it 5 times. The vInductor will observe the identity.
    for i in 0..5 {
        let start = Instant::now();
        let _ = ctx.dispatch(sig, 1);
        println!("Call {} | Time: {:?}", i + 1, start.elapsed());
    }

    // At the 5th call, the system should have "Noticed" the results are static/deterministic
    // and promoted it to a Law.
    let events = ctx.induction_events();
    println!("\nInduction Events Detected: {}", events);
    assert!(
        events > 0,
        "System failed to autonomously promote the process."
    );

    println!("\n--- Phase 2: The Inducted O(1) Recall ---");
    let start = Instant::now();
    let _ = ctx.dispatch(sig, 1);
    let duration = start.elapsed();
    println!("Inducted Recall | Time: {:?}", duration);

    // Verification: Recall should be significantly faster than interpretation
    assert!(
        duration.as_micros() < 500,
        "Inducted recall failed O(1) speed check."
    );
    println!("Success: Process promoted to Imperative Law autonomously.");
    println!("{}\n", "=".repeat(60));
}
