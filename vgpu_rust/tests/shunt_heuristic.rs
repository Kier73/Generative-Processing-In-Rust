use vgpu_rust::*;

#[test]
fn test_shunting_heuristic_skip_simple() {
    let mut ctx = VGpuContext::new(1, 42);

    // 1. Simple Kernel (Cost < 20)
    let mut vs_simple = VirtualShader::new();
    vs_simple.push(SpvOp::FAdd);
    let sig_simple = vs_simple.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs_simple,
        signature: sig_simple,
    }));

    // Dispatch 10 times. Should NOT be in manifold because it's too simple.
    for _ in 0..10 {
        ctx.dispatch(sig_simple, 0x123);
    }
    assert_eq!(
        ctx.inductor.manifold.read().unwrap().len(),
        0,
        "Simple kernel should NOT be memoized"
    );

    // 2. Complex Kernel (Cost > 20)
    let mut vs_complex = VirtualShader::new();
    for _ in 0..10 {
        vs_complex.push(SpvOp::RSin { dst: 0, src: 0 });
    }
    // Cost = 10 * 5 = 50
    let sig_complex = vs_complex.generate_signature();
    ctx.register_kernel(Box::new(ShaderKernel {
        shader: vs_complex,
        signature: sig_complex,
    }));

    // Dispatch 10 times. Should be in manifold.
    for _ in 0..10 {
        ctx.dispatch(sig_complex, 0x123);
    }
    assert!(
        ctx.inductor.manifold.read().unwrap().len() > 0,
        "Complex kernel SHOULD be memoized"
    );
}
