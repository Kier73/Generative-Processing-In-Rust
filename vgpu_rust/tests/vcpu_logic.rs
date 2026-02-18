use vgpu_rust::*;

#[test]
fn test_vcpu_integer_math_basic() {
    let mut rf = RegisterFile::new();
    rf.execute_op(&SpvOp::ILoadImm { dst: 1, value: 100 });
    rf.execute_op(&SpvOp::ILoadImm { dst: 2, value: 50 });
    rf.execute_op(&SpvOp::IAdd {
        dst: 3,
        src_a: 1,
        src_b: 2,
    });
    assert_eq!(rf.iregs[3], 150);
}

#[test]
fn test_vcpu_conditional_branch_loop() {
    let mut rf = RegisterFile::new();
    // Simple Loop: for (i=0; i<10; i++) { acc += 1.0; }
    let ops = vec![
        SpvOp::ILoadImm { dst: 1, value: 0 },  // i = 0
        SpvOp::ILoadImm { dst: 2, value: 10 }, // Limit = 10
        SpvOp::ILoadImm { dst: 3, value: 1 },  // Inc = 1
        // Loop Start (Offset 3)
        SpvOp::ICmp { src_a: 1, src_b: 2 },  // i vs Limit
        SpvOp::BTrap { cond: 0, offset: 8 }, // If i == 10, jump to End (Offset 8)
        SpvOp::FAdd,                         // regs[0] += 1.0
        SpvOp::IAdd {
            dst: 1,
            src_a: 1,
            src_b: 3,
        }, // i += 1
        SpvOp::BJump { offset: 3 },          // Jump to Loop Start
                                             // End (Offset 8)
    ];

    rf.execute_program(&ops, None);
    assert_eq!(rf.regs[0], 10.0);
    assert_eq!(rf.iregs[1], 10);
}

#[test]
fn test_vcpu_nested_branching() {
    let mut rf = RegisterFile::new();
    // if (10 > 5) { x = 1.0 } else { x = 2.0 }
    let ops = vec![
        SpvOp::ILoadImm { dst: 1, value: 10 },
        SpvOp::ILoadImm { dst: 2, value: 5 },
        SpvOp::ICmp { src_a: 1, src_b: 2 }, // 10 > 5, so flags = 1 (Gt)
        SpvOp::BTrap { cond: 1, offset: 6 }, // If Gt, jump to TruePath (Offset 6)
        // FalsePath
        SpvOp::RLoadImm {
            dst: 0,
            value_bits: 2.0f32.to_bits(),
        },
        SpvOp::BJump { offset: 7 }, // Jump to End
        // TruePath (Offset 6)
        SpvOp::RLoadImm {
            dst: 0,
            value_bits: 1.0f32.to_bits(),
        },
        // End (Offset 7)
    ];

    rf.execute_program(&ops, None);
    assert_eq!(rf.regs[0], 1.0);

    // native
    let mut rf2 = RegisterFile::new();
    let code = native_jit::compile(&ops).unwrap();
    code.call(&mut rf2);
    assert_eq!(rf2.regs[0], 1.0);
}

#[test]
fn test_vcpu_jit_loop_acceleration() {
    let mut rf = RegisterFile::new();
    // Loop: for (i=0; i<100; i++) { acc += 1.0; }
    let ops = vec![
        SpvOp::ILoadImm { dst: 1, value: 0 },
        SpvOp::ILoadImm { dst: 2, value: 100 },
        SpvOp::ILoadImm { dst: 3, value: 1 },
        // Start (Offset 3)
        SpvOp::ICmp { src_a: 1, src_b: 2 },
        SpvOp::BTrap { cond: 0, offset: 8 }, // End at 8
        SpvOp::FAdd,
        SpvOp::IAdd {
            dst: 1,
            src_a: 1,
            src_b: 3,
        },
        SpvOp::BJump { offset: 3 },
        // End (Offset 8)
    ];

    let code = native_jit::compile(&ops).expect("JIT Failed");
    code.call(&mut rf);
    assert_eq!(rf.regs[0], 100.0);
    assert_eq!(rf.iregs[1], 100);
}
