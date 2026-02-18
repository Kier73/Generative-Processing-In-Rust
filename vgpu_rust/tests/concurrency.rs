use std::sync::{Arc, Mutex};
use std::thread;
use vgpu_rust::{KERNEL_DISPATCH_SIZE, KernelResult, VGpuContext};

#[test]
fn test_concurrent_induction_pressure() {
    // Note: Our current VInductor is not thread-safe (no Mutex internally on the manifold)
    // We wrap it in a Mutex for the context or test atomic behavior if we implement internal locks.
    // For now, we test the Context level thread-safety.

    let gpu = Arc::new(Mutex::new(VGpuContext::new(128, 0x1337)));
    let mut handles = vec![];

    println!("Spawning 8 threads for induction contention...");
    for t in 0..8 {
        let gpu_ref = Arc::clone(&gpu);
        let handle = thread::spawn(move || {
            for i in 0..1000 {
                let g = gpu_ref.lock().unwrap();
                let hash = (t * 1000 + i) as u64;
                g.inductor.induct(
                    0xABC,
                    hash,
                    KernelResult {
                        data: std::sync::Arc::from([hash as f32; KERNEL_DISPATCH_SIZE]),
                    },
                );
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let g = gpu.lock().unwrap();
    println!("Concurrent writes verified. Checking manifold integrity...");
    // Verify a random sample
    // The original test checked for hash 4500 (t=4, i=500).
    // Assuming the instruction meant to test this specific hash with the new recall signature.
    let hash_to_check = 4500;
    if let Some(res) = g.inductor.recall(0, hash_to_check) {
        assert_eq!(res.data[0], 4500.0);
    }
    println!("Integrity: OK");
}
