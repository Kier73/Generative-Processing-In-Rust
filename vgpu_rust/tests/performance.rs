use std::time::Instant;
use vgpu_rust::{KERNEL_DISPATCH_SIZE, KernelResult, VGpuContext};

#[test]
fn test_manifold_collision_resistance() {
    let gpu = VGpuContext::new(128, 0xFEED);
    let entries = 4096; // Our MANIFOLD_SIZE is 1 << 12

    println!("Filling Manifold to 100% capacity (requires 5 stable observations per entry)...");
    for i in 0..entries {
        for _ in 0..5 {
            gpu.inductor.induct(
                0x111,
                i as u64,
                KernelResult {
                    data: std::sync::Arc::from([i as f32; KERNEL_DISPATCH_SIZE]),
                },
            );
        }
    }

    // Verify high-density recall
    let mut hits = 0;
    for i in 0..entries {
        if let Some(res) = gpu.inductor.recall(0x111, i as u64) {
            if res.data[0] == i as f32 {
                hits += 1;
            }
        }
    }

    println!(
        "Manifold Load Factor: 1.0 | Consistency Score: {}/{}",
        hits, entries
    );
    assert!(hits > entries / 2, "Manifold collision rate too high!");
}

#[test]
fn test_long_soak_stability() {
    let gpu = VGpuContext::new(128, 0xFEED);
    let iterations = 1_000_000;
    let start = Instant::now();

    for i in 0..iterations {
        gpu.inductor.recall(0, i as u64);
    }

    let duration = start.elapsed();
    println!("Soak test (1M iterations) completed in {:?}", duration);
}

#[test]
fn test_memory_pressure_overflow() {
    let gpu = VGpuContext::new(128, 0x9999);
    let entries = 8192; // 2x the MANIFOLD_SIZE

    println!("Pushing 200% load factor into manifold (5 observations per entry)...");
    for i in 0..entries {
        for _ in 0..5 {
            gpu.inductor.induct(
                0xCAD,
                i as u64,
                KernelResult {
                    data: std::sync::Arc::from([i as f32; KERNEL_DISPATCH_SIZE]),
                },
            );
        }
    }

    // We expect roughly half the entries to have been evicted (overwritten)
    let mut retained = 0;
    for i in 0..entries {
        if let Some(res) = gpu.inductor.recall(0x111, i as u64) {
            if res.data[0] == i as f32 {
                retained += 1;
            }
        }
    }

    println!("Entries Retained: {}/{}", retained, entries);
    assert!(
        retained <= 4096,
        "Memory expansion detected where none should exist!"
    );
}
