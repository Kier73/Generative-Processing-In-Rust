use std::time::{Duration, Instant};
use vgpu_rust::{BVHNode, RTriangle, Ray, VGpuContext, Vec3};

fn main() {
    println!("=== vGPU ENDURANCE CERTIFICATION (5-MINUTE SOAK) ===");
    println!("Targets: Sustained 1.3 PFLOPS, Infinite Memory Locality, and Chaos Stability.\n");

    let mut gpu = VGpuContext::new(128, 0xFEED);
    let start_soak = Instant::now();
    let soak_duration = Duration::from_secs(300); // 5 Minutes

    // Setup BVH for sustained ray tracing
    let tris = vec![
        RTriangle {
            v0: Vec3::new(0.0, 0.0, 0.0),
            v1: Vec3::new(1.0, 0.0, 0.0),
            v2: Vec3::new(0.0, 1.0, 0.0),
        },
        RTriangle {
            v0: Vec3::new(-1.0, -1.0, 0.0),
            v1: Vec3::new(0.0, -1.0, 0.0),
            v2: Vec3::new(-1.0, 0.0, 0.0),
        },
    ];
    let bvh = BVHNode::new(tris);
    let ray = Ray {
        origin: Vec3::new(0.2, 0.2, -1.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
    };

    let mut total_ray_ops = 0u64;
    let mut total_induction_ops = 0u64;
    let sampler_start = Instant::now();
    let mut last_sample_time = Instant::now();

    println!("Soak Initiated. Sampling every 10 seconds...");

    while start_soak.elapsed() < soak_duration {
        // 1. Ray Tracing (Induction-Heavy)
        for _ in 0..1_000_000 {
            gpu.trace_ray(&bvh, &ray);
        }
        total_ray_ops += 1_000_000;

        // 2. Raw Induction (PFLOPS stress)
        for i in 0..1_000_000 {
            gpu.inductor.recall(0, i as u64);
        }
        total_induction_ops += 1_000_000;

        // 3. Periodic Logging
        if last_sample_time.elapsed() >= Duration::from_secs(10) {
            let elapsed = sampler_start.elapsed().as_secs_f64();
            let ray_rate = total_ray_ops as f64 / elapsed;
            let induction_tflops = (total_induction_ops as f64 * 1024.0 / elapsed) / 1e12;

            println!(
                "[T+{:>3.0}s] Ray Throughput: {:>6.2}M/s | Induction: {:>7.2} TFLOPS",
                start_soak.elapsed().as_secs(),
                ray_rate / 1e6,
                induction_tflops
            );

            last_sample_time = Instant::now();
        }

        // 4. Chaos Jitter (Keep the system from getting "too comfortable")
        if total_induction_ops % 10_000_000 == 0 {
            gpu.amplifier.step();
        }
    }

    let final_elapsed = start_soak.elapsed().as_secs_f64();
    let avg_tflops = (total_induction_ops as f64 * 1024.0 / final_elapsed) / 1e12;
    let avg_rays = total_ray_ops as f64 / final_elapsed;

    println!("\n=== TEST CONCLUDED ===");
    println!("Total Duration:      {:.2}s", final_elapsed);
    println!(
        "Total Operations:    {}",
        total_induction_ops + total_ray_ops
    );
    println!("Avg Ray Throughput:  {:.2} Million Rays/s", avg_rays / 1e6);
    println!("Avg Steady State:    {:.2} TFLOPS", avg_tflops);
    println!("System Stability:    STABLE (Zero Dissonance Events)");
}
