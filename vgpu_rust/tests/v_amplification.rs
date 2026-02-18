use vgpu_rust::{RnsEngine, VGpuContext};

#[test]
fn test_recursive_bifurcation_depth() {
    let mut gpu = VGpuContext::new(128, 0x5E44CACE);

    // Push the limits: Depth 14 = 16,384 virtual nodes
    let depth = 14;
    let nodes = gpu.amplifier.bifurcate(depth);

    println!("Amplification at Depth {}: {} nodes", depth, nodes.len());
    assert_eq!(nodes.len(), 1 << depth);

    // Chaos check
    gpu.amplifier.r = 4.0;
    let chaos_val = gpu.amplifier.step();
    assert!(chaos_val >= 0.0 && chaos_val <= 1.0);
}

#[test]
fn test_rns_precision_coherence() {
    let rns = RnsEngine::new(vec![65537, 65539, 65543, 65551, 65563]);
    let test_val: u128 = 98765432109876543210987654321;

    let residues = rns.decompose(test_val);
    let combined = rns.combine(&residues);

    let m_total: u128 = rns.moduli.iter().map(|&x| x as u128).product();
    assert_eq!(
        combined,
        test_val % m_total,
        "RNS Recombination failed bit-exactness check"
    );
}
