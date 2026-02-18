use vgpu_rust::{SpvOp, VirtualShader};

#[test]
fn test_alpha_renaming_immunity() {
    // Program A: r0 = dot(r1, r2); r0 = normalize(r0)
    let mut vs_a = VirtualShader::new();
    vs_a.push(SpvOp::VDot {
        dst: 0,
        src_a: 1,
        src_b: 2,
    });
    vs_a.push(SpvOp::VNormalize { dst: 0, src: 0 });

    // Program B: r3 = dot(r4, r5); r3 = normalize(r3) (Alpha-renamed)
    let mut vs_b = VirtualShader::new();
    vs_b.push(SpvOp::VDot {
        dst: 3,
        src_a: 4,
        src_b: 5,
    });
    vs_b.push(SpvOp::VNormalize { dst: 3, src: 3 });

    let sig_a = vs_a.generate_semantic_signature();
    let sig_b = vs_b.generate_semantic_signature();

    println!("Sig A: {}", sig_a);
    println!("Sig B: {}", sig_b);

    // Standard signature should FAIL (they are syntactically different)
    assert_ne!(
        vs_a.generate_signature(),
        vs_b.generate_signature(),
        "Standard signatures collided unexpectedly"
    );

    // Semantic signature should PASS (they are semantically identical)
    assert_eq!(sig_a, sig_b, "Alpha-renaming immunity failed");
}

#[test]
fn test_reordering_immunity() {
    // Program A: FAdd, FSub
    let mut vs_a = VirtualShader::new();
    vs_a.push(SpvOp::FAdd);
    vs_a.push(SpvOp::FSub);

    // Program B: FSub, FAdd (Reordered)
    let mut vs_b = VirtualShader::new();
    vs_b.push(SpvOp::FSub);
    vs_b.push(SpvOp::FAdd);

    let sig_a = vs_a.generate_semantic_signature();
    let sig_b = vs_b.generate_semantic_signature();

    assert_eq!(sig_a, sig_b, "Commutative reordering immunity failed");
}

#[test]
fn test_collision_resistance() {
    // Program A: normalize
    let mut vs_a = VirtualShader::new();
    vs_a.push(SpvOp::VNormalize { dst: 0, src: 1 });

    // Program B: rsqrt
    let mut vs_b = VirtualShader::new();
    vs_b.push(SpvOp::VRSqrt { dst: 0, src: 1 });

    let sig_a = vs_a.generate_semantic_signature();
    let sig_b = vs_b.generate_semantic_signature();

    assert_ne!(sig_a, sig_b, "Semantically different programs collided");
}
