use vgpu_rust::{NativeJitKernel, SpvOp, VGpuContext, native_jit};

#[test]
fn test_native_jit_matches_interpreter() {
    let ops = vec![SpvOp::FAdd, SpvOp::FMul, SpvOp::FAdd, SpvOp::FDiv];
    let input_hash: u64 = 42;
    let x_input = (input_hash & 0xFF) as f32;

    // 1. Run the interpreter path (JitKernel style)
    let mut x = x_input;
    for op in &ops {
        match op {
            SpvOp::FAdd => x += 1.0,
            SpvOp::FMul => x *= 1.1,
            SpvOp::FDiv => x /= 1.1,
            _ => {}
        }
    }
    let interpreter_result = x;

    // 2. Run the native JIT path
    let mut rf = vgpu_rust::RegisterFile::new();
    rf.regs[0] = x_input;
    let compiled = native_jit::compile(&ops).expect("JIT compilation failed");
    compiled.call(&mut rf);
    let jit_result = rf.regs[0];

    println!("Interpreter: {}", interpreter_result);
    println!("Native JIT:  {}", jit_result);

    // They should produce the same float result
    let diff = (interpreter_result - jit_result).abs();
    assert!(diff < 0.001, "JIT diverged from interpreter by {}", diff);
}

#[test]
fn test_native_jit_kernel_in_context() {
    let mut gpu = VGpuContext::new(128, 0xBEEF);

    let ops = vec![SpvOp::FAdd, SpvOp::FMul];
    let compiled = native_jit::compile(&ops).expect("JIT compilation failed");
    let sig = 0x11111;

    gpu.register_kernel(Box::new(NativeJitKernel {
        compiled,
        signature: sig,
    }));

    // Dispatch should execute the native code and induct the result
    let result = gpu.dispatch(sig, 100);
    println!("NativeJIT dispatch result: {}", result.data[0]);

    // Second dispatch should be an induction recall (O(1) hit)
    let recalled = gpu.dispatch(sig, 100);
    assert_eq!(result.data[0], recalled.data[0]);
}
