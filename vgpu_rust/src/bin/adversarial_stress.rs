use std::sync::Arc;
use std::time::Instant;
use vgpu_rust::{KERNEL_DISPATCH_SIZE, Kernel, KernelResult, VGpuContext, vmatrix::GeometricField};

struct DummyKernel {
    sig: u64,
}
impl Kernel for DummyKernel {
    fn signature(&self) -> u64 {
        self.sig
    }
    fn execute(&self, _hash: u64) -> KernelResult {
        KernelResult {
            data: Arc::from([1.0; KERNEL_DISPATCH_SIZE]),
        }
    }
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("vGPU ADVERSARIAL STRESS SUITE (V2)");
    println!("{}", "=".repeat(70));

    let seed = 0x5E44C_ACE;
    let mut ctx = VGpuContext::new(8, seed);

    test_numerical_stability(&ctx);
    test_collision_probability(&ctx);
    test_manifold_thrashing(&mut ctx);
    test_extreme_scales(&ctx);

    println!("{}", "=".repeat(70));
    println!("STRESS SUITE COMPLETE");
    println!("{}", "=".repeat(70));
}

fn test_numerical_stability(ctx: &VGpuContext) {
    println!("\n[TEST] NUMERICAL STABILITY (Deep Law Chains)");
    let mut field = GeometricField::new(1024, 1024, 0x111);
    let identity = GeometricField::new(1024, 1024, 0x222);

    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        field = field.multiply(&identity).unwrap();
    }
    let duration = start.elapsed();

    let val = field.resolve(ctx, 512, 512);
    println!("Descent Depth: {} multiplications", iterations);
    println!("Final Depth:   {}", field.depth);
    println!("Resolve Latency: {:?}", duration / iterations as u32);
    println!("Sample Value:  {:.12}", val);

    if val.is_nan() || val.is_infinite() {
        println!("[FAIL] Numerical Divergence Detected!");
    } else {
        println!("[PASS] Geometric Stability Maintained.");
    }
}

fn test_collision_probability(ctx: &VGpuContext) {
    println!("\n[TEST] COLLISION SEARCH (Entropy Sweep)");
    let field = GeometricField::new(1_000_000, 1_000_000, 0x333);
    let mut samples = std::collections::HashSet::new();
    let num_samples = 1_000_000;
    let mut collisions = 0;

    let start = Instant::now();
    for i in 0..num_samples {
        let val = field.resolve(ctx, i, i);
        let bits = val.to_bits();
        if !samples.insert(bits) {
            collisions += 1;
        }
    }
    let duration = start.elapsed();

    println!("Samples:      {}", num_samples);
    println!("Collisions:   {}", collisions);
    println!("Search Time:  {:?}", duration);

    let rate = (collisions as f64 / num_samples as f64) * 100.0;
    println!("Entropy Leak: {:.6}%", rate);

    if collisions > (num_samples / 10000) {
        println!("[WARNING] Collision rate higher than target (0.01%)!");
    } else {
        println!("[PASS] Variety Uniqueness Verified.");
    }
}

fn test_manifold_thrashing(ctx: &mut VGpuContext) {
    println!("\n[TEST] MANIFOLD THRASHING (Pruning Logic)");
    let num_kernels = 10_000;

    // Register a persistent kernel so dispatches have something to execute
    // Use a range of signatures
    for i in 0..2000 {
        ctx.register_kernel(Box::new(DummyKernel { sig: i as u64 }));
    }

    println!(
        "Flooding manifold with {} distinct dispatches...",
        num_kernels
    );
    for i in 0..num_kernels {
        let sig = (i % 2000) as u64;
        let input_hash = i as u64; // Every dispatch is a new law instance

        // Dispatch 5 times per hash to pass the "Induction Confidence" (variance gate)
        for _ in 0..5 {
            ctx.dispatch(sig, input_hash);
        }

        // Trigger pruning periodically to simulate production reclamation
        if i % 100 == 0 {
            ctx.prune_weakest();
        }

        if i % 2000 == 0 {
            let tele = ctx.telemetry();
            println!(
                "  Batch {}: Occupancy {}/{}",
                i, tele.manifold_occupancy, tele.manifold_capacity
            );
        }
    }

    let tele = ctx.telemetry();
    println!("Total Dispatches: {}", tele.total_dispatches);
    println!("Induction Hits:   {}", tele.induction_hits);
    println!("Evictions:        {}", tele.evictions_performed);

    if tele.evictions_performed > 5000 {
        println!("[PASS] Pruning Engine Correctly Managed Overflow.");
    } else if tele.evictions_performed > 0 {
        println!("[OK] Pruning active, but occupancy remains within bounds.");
    } else {
        println!("[FAIL] Manifold failed to prune despite overflow!");
    }
}

fn test_extreme_scales(ctx: &VGpuContext) {
    println!("\n[TEST] EXTREME SCALES (Aliasing Check)");
    let field = GeometricField::new(u64::MAX, u64::MAX, 0x999);

    println!("Matrix Size: {} x {}", u64::MAX, u64::MAX);
    let v1 = field.resolve(ctx, 0, 0);
    let v2 = field.resolve(ctx, u64::MAX - 1, u64::MAX - 1);
    let v3 = field.resolve(ctx, 1234, 5678);

    println!("Resolve (0,0):       {:.12}", v1);
    println!("Resolve (MAX, MAX):  {:.12}", v2);
    println!("Resolve (1234,5678): {:.12}", v3);

    if v1 == v2 {
        println!("[FAIL] Coordinate Aliasing Detected!");
    } else {
        println!("[PASS] Coordinate Uniqueness Maintained at Scale.");
    }
}
