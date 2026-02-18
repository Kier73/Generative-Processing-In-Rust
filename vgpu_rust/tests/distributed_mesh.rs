use vgpu_rust::{KernelResult, VGpuContext};

#[test]
fn test_manifold_merging_mesh() {
    let gpu1 = VGpuContext::new(128, 0x1111);
    let gpu2 = VGpuContext::new(128, 0x2222);

    let input_hash = 999;
    let sig = 0xDEADC0DE;
    let result = KernelResult {
        data: std::sync::Arc::from([123.456f32; vgpu_rust::KERNEL_DISPATCH_SIZE]),
    };

    // 1. GPU1 learns a law (requires 5 stable observations for induction confidence)
    for _ in 0..5 {
        gpu1.inductor.induct(sig, input_hash, result.clone());
    }
    assert!(gpu1.inductor.recall(sig, input_hash).is_some());
    assert!(gpu2.inductor.recall(sig, input_hash).is_none());

    // 2. Perform a "Mesh Sync" (Merge GPU1 into GPU2)
    println!("Merging GPU1 manifold into GPU2...");
    gpu2.inductor.merge(&gpu1.inductor);

    // 3. Verify GPU2 now "remembers" what GPU1 learned
    if let Some(recalled) = gpu2.inductor.recall(sig, input_hash) {
        println!("GPU2 Recall Successful: {}", recalled.data[0]);
        assert_eq!(recalled.data[0], 123.456);
    } else {
        panic!("GPU2 failed to receive the synced law from the mesh!");
    }
}
