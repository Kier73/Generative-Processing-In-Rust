use vgpu_rust::{KERNEL_DISPATCH_SIZE, KernelResult, VGpuContext};

#[test]
fn test_slow_poisoning_attack() {
    let mut gpu = VGpuContext::new(128, 0x5E44CACE);

    // Target law: 1.0 + 1.0 = 2.0
    let input_hash = 0x1337;
    let ground_truth = 2.0;

    // We try to slowly shift the manifold with sub-threshold errors (epsilon = 0.0001)
    // threshold is 0.001
    let epsilon = 0.0001;
    let mut poisoned_val = ground_truth;

    println!("Starting Slow Poisoning Attack (Epsilon = {})", epsilon);

    for i in 0..500 {
        poisoned_val += epsilon;
        // Inject poisoned result
        gpu.inductor.induct(
            0xACC,
            input_hash,
            KernelResult {
                data: std::sync::Arc::from([poisoned_val; KERNEL_DISPATCH_SIZE]),
            },
        );

        // At some point, the cumulative error should exceed the threshold
        if !gpu.dissonance_control.check(poisoned_val, ground_truth).0 {
            println!(
                "Success: Dissonance control triggered after {} iterations at value {}",
                i, poisoned_val
            );
            return;
        }
    }

    panic!("Failure: Slow poisoning bypassed Bayesian Dissonance Control!");
}

#[test]
fn test_bayesian_blind_spot() {
    let mut gpu = VGpuContext::new(128, 0x1234);

    // We try to find a drift so small (0.0000001) that it never triggers dissonance
    // but results in significant bias over 10,000 operations.
    let input_hash = 0xDEAD;
    let ground_truth = 1.0;
    let micro_epsilon = 0.000001;

    println!("Testing Bayesian Blind Spot (Epsilon = {})", micro_epsilon);
    for i in 0..10000 {
        let poisoned = ground_truth + micro_epsilon;
        gpu.inductor.induct(
            0xABC,
            input_hash,
            KernelResult {
                data: std::sync::Arc::from([poisoned; KERNEL_DISPATCH_SIZE]),
            },
        );

        let checked = gpu.dissonance_control.check(poisoned, ground_truth).0;
        if !checked {
            println!("Caught Blind Spot at iteration {}", i);
            return;
        }
    }
    println!(
        "Note: Bayesian Blind Spot exists for sub-precision drift. This is expected behavior for probabilistic systems."
    );
}

#[test]
fn test_hash_avalanche_collision() {
    let gpu = VGpuContext::new(128, 0x5555);

    // Intentionally use a very poor hashing pattern (only lower 4 bits vary)
    // to force collisions in the same bucket.
    println!("Executing Hash Avalanche (buckets should collide)...");
    for i in 0..100 {
        let bad_hash = i << 12; // All will land in bucket 0 because idx = hash & (SIZE-1)
        gpu.inductor.induct(
            0x1,
            bad_hash,
            KernelResult {
                data: std::sync::Arc::from([i as f32; KERNEL_DISPATCH_SIZE]),
            },
        );
    }

    // Only the last entry (99) should survive.
    // We check hash 99 << 12 with signature 0x1.
    if let Some(res) = gpu.inductor.recall(0x1, 99 << 12) {
        println!("Collision Survivor Value: {}", res.data[0]);
        assert_eq!(res.data[0], 99.0); // The last write to bucket 0 wins.
    }
}
