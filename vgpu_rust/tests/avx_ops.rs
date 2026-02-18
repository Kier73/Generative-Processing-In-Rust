use vgpu_rust::{RegisterFile, SpvOp, native_jit};

#[test]
fn test_avx_ops() {
    // 1. Setup Kernel
    // We will test VSub, VDiv, VRSqrt.
    // Registers:
    // 0-3: Source A (populated by test)
    // 4-7: Source B (populated by test)
    // 8-11: Destination Sub
    // 12-15: Destination Div

    // We reuse 0-3 for VRSqrt dst at the end.

    let ops = vec![
        SpvOp::VSub {
            dst: 8,
            src_a: 0,
            src_b: 4,
        },
        SpvOp::VDiv {
            dst: 12,
            src_a: 0,
            src_b: 4,
        },
        SpvOp::VRSqrt { dst: 0, src: 4 }, // Overwrite A
    ];

    // 2. Compile
    let code = native_jit::compile_avx_aware(&ops).expect("Failed to compile JIT kernel");

    // 3. Prepare Register File
    let mut rf = RegisterFile::new();

    // Source A (0-3)
    rf.regs[0] = 10.0;
    rf.regs[1] = 20.0;
    rf.regs[2] = 30.0;
    rf.regs[3] = 40.0;

    // Source B (4-7)
    rf.regs[4] = 2.0;
    rf.regs[5] = 4.0;
    rf.regs[6] = 5.0;
    rf.regs[7] = 8.0;

    // 4. Calculate Expected Values
    let expected_sub = [8.0, 16.0, 25.0, 32.0];
    let expected_div = [5.0, 5.0, 6.0, 5.0];
    let expected_rsqrt = [
        1.0 / (2.0f32).sqrt(),
        1.0 / (4.0f32).sqrt(), // 0.5
        1.0 / (5.0f32).sqrt(),
        1.0 / (8.0f32).sqrt(),
    ];

    // 5. Execute
    code.call(&mut rf);

    // 6. Verify Sub
    println!("VSub Results: {:?}", &rf.regs[8..12]);
    for i in 0..4 {
        assert!(
            (rf.regs[8 + i] - expected_sub[i]).abs() < 1e-5,
            "VSub failed at index {}",
            i
        );
    }

    // 7. Verify Div
    println!("VDiv Results: {:?}", &rf.regs[12..16]);
    for i in 0..4 {
        assert!(
            (rf.regs[12 + i] - expected_div[i]).abs() < 1e-5,
            "VDiv failed at index {}",
            i
        );
    }

    // 8. Verify VRSqrt
    println!("VRSqrt Results: {:?}", &rf.regs[0..4]);
    for i in 0..4 {
        assert!(
            (rf.regs[0 + i] - expected_rsqrt[i]).abs() < 1e-3,
            "VRSqrt failed at index {} (approx)",
            i
        );
        // Note: rsqrt is an approximation, so 1e-3 tolerance is appropriate
    }
}
