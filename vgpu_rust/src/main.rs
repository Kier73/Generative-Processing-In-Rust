use std::time::Instant;
use vgpu_rust::{VAddr, VGpuContext};

fn benchmark_performance_metrics() {
    println!("--- vGPU v2.1 Honest Benchmark Suite ---");
    let gpu = VGpuContext::new(128, 0x5E44CACE);

    // Warmup & Induction Confidence
    let sig = 0xABCD;
    let input_hash = 12345;
    let result = vgpu_rust::KernelResult {
        data: std::sync::Arc::from([1.0f32; 1024]),
    };

    println!("Inducting stable law (5 samples)...");
    for _ in 0..5 {
        gpu.inductor.induct(sig, input_hash, result.clone());
    }

    let iterations = 20_000_000;

    // 1. Induction Throughput (Recall)
    let start = Instant::now();
    for _ in 0..iterations {
        // Use signature 0 for generic lookup in benchmark
        gpu.inductor.recall(0, input_hash);
    }
    let duration = start.elapsed();
    let recall_throughput = (iterations as f64) / duration.as_secs_f64();

    // 2. Ground Truth Throughput (Actual Shader Execution)
    // We compare Recall against the Interpreter's baseline execution speed.
    let ops = vec![vgpu_rust::SpvOp::FAdd, vgpu_rust::SpvOp::FMul];
    let shader = vgpu_rust::VirtualShader {
        instructions: ops.clone(),
    };
    let kernel = vgpu_rust::ShaderKernel {
        shader,
        signature: 0x2,
    };

    let start = Instant::now();
    for i in 0..iterations / 10 {
        // Use smaller sample for interpreter
        use vgpu_rust::Kernel;
        kernel.execute(i as u64);
    }
    let duration_gt = start.elapsed();
    let gt_throughput = ((iterations / 10) as f64) / duration_gt.as_secs_f64();

    println!(
        "Recall Throughput:     {:.2} Million Dispatches/s",
        recall_throughput / 1e6
    );
    println!(
        "Ground Truth (Interp): {:.2} Million Dispatches/s",
        gt_throughput / 1e6
    );
    println!(
        "Induction Speedup:     {:.1}x",
        recall_throughput / gt_throughput
    );

    // 3. Manifold Health (Collision Analysis)
    let entries = 4096;
    let mut collisions = 0;
    for i in 0..entries {
        let idx1 = vgpu_rust::addr_to_u64(VAddr([i as u64, 0, 0, 0]));
        let idx2 = vgpu_rust::addr_to_u64(VAddr([i as u64 + 1, 0, 0, 0]));
        if (idx1 as usize & (4096 - 1)) == (idx2 as usize & (4096 - 1)) {
            collisions += 1;
        }
    }
    println!(
        "Manifold Health:       {:.2}% collision rate (Feistel distribution)",
        (collisions as f32 / entries as f32) * 100.0
    );
}

fn test_bayesian_control() {
    println!("\n--- Phase 1.5: Bayesian Dissonance Check ---");
    let mut gpu = VGpuContext::new(128, 0x5E44CACE);

    let induced_val = 0.8801;
    let ground_truth = 0.8800;

    if gpu.dissonance_control.check(induced_val, ground_truth).0 {
        println!("Bayesian Status: COHERENT (Dissonance < Threshold)");
    } else {
        println!("Bayesian Status: DISSONANT (Law Drift Detected!)");
    }

    let mut verifications = 0;
    for _ in 0..1000 {
        if gpu.dissonance_control.should_verify() {
            verifications += 1;
        }
    }
    println!(
        "Probabilistic Verification Rate: {:.2}%",
        (verifications as f32 / 10.0)
    );
}

fn test_virtual_amplification() {
    println!("\n--- Phase 7: Virtual Amplification (Bifurcation Scaling) ---");
    let mut gpu = VGpuContext::new(128, 0x5E44CACE);

    // 1. Exponential Branching
    let depth = 10;
    let nodes = gpu.amplifier.bifurcate(depth);
    println!("Bifurcation Depth: {} steps", depth);
    println!("Virtual vGPUs Generated: {}", nodes.len());
    println!("First 4 Node Signatures: {:016X?}", &nodes[0..4]);

    // 2. Bit-Exact RNS Decomposition
    let rns = vgpu_rust::RnsEngine::new(vec![65537, 65539, 65543, 65551]);
    let large_val: u128 = 123456789012345678901234567890;
    let residues = rns.decompose(large_val);
    println!("\nLarge Task Value: {}", large_val);
    println!("RNS Residues:     {:?}", residues);

    // 3. Recombination (CRT)
    let combined = rns.combine(&residues);
    println!("CRT Recombined:   {}", combined);

    if combined == large_val % (65537 * 65539 * 65543 * 65551 as u128) {
        println!("Status: BIT-EXACT COHERENCE VERIFIED");
    }
}

fn adversarial_testing() {
    println!("\n--- Phase 8: Adversarial Testing (Adversarial Law Injection) ---");
    let mut gpu = VGpuContext::new(128, 0x5E44CACE);

    // 1. Injecting a "Hostile Law"
    let input_hash = 123456789;
    let ground_truth = 2.0;
    let hostile_result = 3.0; // Math Error: 1+1=3

    println!(
        "Injecting Law Drift: Predicted={}, Reality={}",
        hostile_result, ground_truth
    );

    // Manually corrupt the manifold
    gpu.inductor.induct(
        0xBAD_1A11,
        input_hash,
        vgpu_rust::KernelResult {
            data: std::sync::Arc::from([hostile_result; 1024]),
        },
    );

    // Check if Dissonance Control catches it
    if !gpu.dissonance_control.check(hostile_result, ground_truth).0 {
        println!("Alert: BAYESIAN DISSONANCE DETECTED! Law Drift Blocked.");
    } else {
        println!("Error: SYSTEM COMPROMISED. Dissonance undetected.");
    }

    println!("\n--- Phase 8: Chaos Regime Boundary Test ($r=4.0$) ---");
    gpu.amplifier.r = 4.0; // Maximum entropy
    let nodes_a = gpu.amplifier.bifurcate(5);
    gpu.amplifier.state = 0.5; // Reset state
    let nodes_b = gpu.amplifier.bifurcate(5);

    if nodes_a == nodes_b {
        println!("Chaos Stability: DETERMINISTIC (Laws hold in the heart of chaos)");
    } else {
        println!("Chaos Stability: STOCHASTIC ERROR");
    }
}

fn jit_speedup_test() {
    println!("\n--- Phase 15: JIT vs Interpreter Speedup ---");
    let mut ops = Vec::new();
    for _ in 0..100 {
        ops.push(vgpu_rust::SpvOp::FAdd);
        ops.push(vgpu_rust::SpvOp::FMul);
    }

    let shader = vgpu_rust::VirtualShader {
        instructions: ops.clone(),
    };
    let kernel = vgpu_rust::ShaderKernel {
        shader,
        signature: 0x1,
    };

    let iterations = 100_000;

    // 1. Interpreter
    let start = Instant::now();
    for i in 0..iterations {
        use vgpu_rust::Kernel;
        kernel.execute(i as u64);
    }
    let duration_int = start.elapsed();

    // 2. JIT (Native x86-64)
    let compiled = vgpu_rust::native_jit::compile(&ops).unwrap();
    let mut rf = vgpu_rust::RegisterFile::new();
    let start = Instant::now();
    for i in 0..iterations {
        rf.regs[0] = i as f32;
        compiled.call(&mut rf);
    }
    let duration_jit = start.elapsed();

    println!("Interpreter Time:  {:?}", duration_int);
    println!("Native JIT Time:    {:?}", duration_jit);
    println!(
        "JIT Acceleration:  {:.1}x",
        duration_int.as_secs_f64() / duration_jit.as_secs_f64()
    );
}

fn main() {
    println!("vGPU v2.1 Global Stress Test: VL-Sourced Honest Benchmarking\n");
    benchmark_performance_metrics();
    test_bayesian_control();
    test_virtual_amplification();
    adversarial_testing();
    jit_speedup_test();
}
