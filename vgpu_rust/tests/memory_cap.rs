use std::sync::Arc;
use vgpu_rust::*;

#[test]
fn test_memory_cap_enforcement() {
    // 800MB cap / ~4KB per result = ~200k entries.
    // We flood with 250k entries to force evictions.
    let mut ctx = VGpuContext::new(1, 42);

    // Register a dummy kernel
    let sig = 0xDEADBEEF;
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: VirtualShader::new(), // Simple empty shader
        signature: sig,
    }));

    println!("Starting memory flood (250k promotions)...");
    for i in 0..250_000 {
        let result = KernelResult {
            data: Arc::from(vec![1.0; KERNEL_DISPATCH_SIZE]),
        };
        // Needs 5 calls for the same hash to pass the "confidence" gate
        for _ in 0..5 {
            ctx.inductor.induct(sig, i as u64, result.clone());
        }

        if i % 10_000 == 0 {
            let tele = ctx.telemetry();
            println!(
                "  Batch {}: Usage {}/{} MB, Evictions: {}",
                i,
                tele.memory_usage_bytes / (1024 * 1024),
                tele.memory_capacity_bytes / (1024 * 1024),
                tele.evictions_performed
            );
        }
    }

    let tele = ctx.telemetry();
    println!(
        "Final State: Usage {}/{} MB, Evictions: {}",
        tele.memory_usage_bytes / (1024 * 1024),
        tele.memory_capacity_bytes / (1024 * 1024),
        tele.evictions_performed
    );

    assert!(
        tele.memory_usage_bytes <= tele.memory_capacity_bytes,
        "Memory usage exceeded cap!"
    );
    assert!(
        tele.evictions_performed > 0,
        "Should have performed at least some evictions"
    );
}
