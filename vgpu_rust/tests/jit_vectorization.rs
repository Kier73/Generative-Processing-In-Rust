#[cfg(test)]
mod tests {
    use vgpu_rust::{RegisterFile, SpvOp};

    #[test]
    fn test_jit_simd_add() {
        // VAdd regs[0..3] = regs[4..7] + regs[8..11]
        let mut rf = RegisterFile::new();
        // Setup inputs
        rf.regs[4] = 1.0;
        rf.regs[5] = 2.0;
        rf.regs[6] = 3.0;
        rf.regs[7] = 4.0;
        rf.regs[8] = 10.0;
        rf.regs[9] = 20.0;
        rf.regs[10] = 30.0;
        rf.regs[11] = 40.0;

        let ops = vec![SpvOp::VAdd {
            dst: 0,
            src_a: 4,
            src_b: 8,
        }];

        let code = vgpu_rust::native_jit::compile(&ops).expect("Failed to compile");
        code.call(&mut rf);

        assert_eq!(rf.regs[0], 11.0);
        assert_eq!(rf.regs[1], 22.0);
        assert_eq!(rf.regs[2], 33.0);
        assert_eq!(rf.regs[3], 44.0);
    }

    #[test]
    fn test_jit_simd_mul() {
        // VMul regs[0..3] = regs[4..7] * regs[8..11]
        let mut rf = RegisterFile::new();
        rf.regs[4] = 2.0;
        rf.regs[5] = 3.0;
        rf.regs[6] = 4.0;
        rf.regs[7] = 5.0;
        rf.regs[8] = 2.0;
        rf.regs[9] = 2.0;
        rf.regs[10] = 2.0;
        rf.regs[11] = 2.0;

        let ops = vec![SpvOp::VMul {
            dst: 0,
            src_a: 4,
            src_b: 8,
        }];

        let code = vgpu_rust::native_jit::compile(&ops).expect("Failed to compile");
        code.call(&mut rf);

        assert_eq!(rf.regs[0], 4.0);
        assert_eq!(rf.regs[1], 6.0);
        assert_eq!(rf.regs[2], 8.0);
        assert_eq!(rf.regs[3], 10.0);
    }
}
