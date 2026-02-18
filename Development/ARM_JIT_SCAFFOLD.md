# ARM / AArch64 JIT Support

## Mathematic Formalization
Modern ARM (Apple Silicon, Graviton) uses 32-bit fixed-length instructions. Mapping `SpvOp` requires a translation layer to the A64 instruction set.

| vGPU Op | x86-64 (Current) | AArch64 (Proposed) |
| :--- | :--- | :--- |
| **FAdd** | `ADDSS xmm0, xmm1` | `FADD s0, s0, s1` |
| **VAdd** | `ADDPS xmm0, xmm1` | `FADD v0.4s, v0.4s, v1.4s` |
| **Dot** | `DPPS xmm0, xmm1, imm`| `FMUL v2.4s, v0.4s, v1.4s` + `FADDP` |

## Code Scaffold

```rust
pub struct Aarch64Emitter {
    buf: Vec<u8>,
}

impl Aarch64Emitter {
    /// Emit FADD s0, s0, s1 (32-bit scalar float add)
    /// Encoding: 0x1e212800
    pub fn emit_fadd(&mut self, rd: u8, rn: u8, rm: u8) {
        let instr = 0x1e202800 | 
                   ((rm as u32) << 16) | 
                   ((rn as u32) << 5) | 
                   (rd as u32);
        self.buf.extend_from_slice(&instr.to_le_bytes());
    }
    
    // Future: Vector (NEON) extensions for VAdd
}
```
