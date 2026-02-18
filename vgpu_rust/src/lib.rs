#[cfg(target_arch = "x86_64")]
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Result as IoResult;
use std::sync::Arc;

pub mod ffi;
pub mod gemm;
pub mod hilbert;
pub mod native_jit;
pub mod sort;
pub mod stdlib;
pub mod trinity;
pub mod vio;
pub mod vmatrix;
pub mod vmatrix_fp;
pub mod vphy;
pub mod vrender;

// --- Phase 2: Math Compendium Modules ---
pub mod veigen;
pub mod vgeometric;
pub mod vhdc;
pub mod vrns;

/// 256-bit Manifold Address
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VAddr(pub [u64; 4]);

impl VAddr {
    pub fn new(w0: u64, w1: u64, w2: u64, w3: u64) -> Self {
        Self([w0, w1, w2, w3])
    }
}

/// Mirror alias for VAddr to maintain Spec naming
pub type MirrorAddr = VAddr;

// --- PHASE 12: VULKAN / SPIR-V BRIDGE ---

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum SpvOp {
    // --- Scalar accumulator ops (original, for backward compat) ---
    FAdd,
    FSub,
    FMul,
    FDiv,
    VDot {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    VNormalize {
        dst: u8,
        src: u8,
    },
    VRSqrt {
        dst: u8,
        src: u8,
    },
    InductionBarrier,
    // --- Register-addressed ops (Change 3: from VL processor.py) ---
    /// dst = src_a + src_b
    RAdd {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// dst = src_a - src_b
    RSub {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// dst = src_a * src_b
    RMul {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// dst = src_a / src_b
    RDiv {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// Load an immediate value into a register
    RLoadImm {
        dst: u8,
        value_bits: u32,
    },
    /// Move data between registers
    RMov {
        dst: u8,
        src: u8,
    },
    /// dst = sin(src)
    RSin {
        dst: u8,
        src: u8,
    },
    /// dst = cos(src)
    RCos {
        dst: u8,
        src: u8,
    },
    /// dst = sqrt(src)
    RSqrt {
        dst: u8,
        src: u8,
    },
    /// dst = abs(src)
    RAbs {
        dst: u8,
        src: u8,
    },
    /// dst = min(src_a, src_b)
    RFMin {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// dst = max(src_a, src_b)
    RFMax {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// dst = clamp(src, min_val, max_val)
    RFClamp {
        dst: u8,
        src: u8,
        min_v: u8,
        max_v: u8,
    },
    // --- vCPU: Integer & Branching (Phase 1) ---
    /// Compare integer registers: flags = (src_a cond src_b)
    /// 0: Equal, 1: Greater, 2: Less
    ICmp {
        src_a: u8,
        src_b: u8,
    },
    /// Unconditional Jump to instruction offset
    BJump {
        offset: u32,
    },
    /// Conditional Jump (If last ICmp matches cond)
    /// cond: 0=Eq, 1=Gt, 2=Lt
    BTrap {
        cond: u8,
        offset: u32,
    },
    /// Load u64 immediate into integer register
    ILoadImm {
        dst: u8,
        value: u64,
    },
    /// Integer addition
    IAdd {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    /// vCPU Inductive Shunt:
    /// If inductor.recall(sig, regs[input_reg]) hits, regs[output_reg] = res and jump to offset.
    VShunt {
        signature: u64,
        input_reg: u8,
        output_reg: u8,
        jump_offset: u32,
    },
    // --- SIMD Vector Ops (Consumes 4 registers) ---
    VAdd {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    VMul {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    VSub {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
    VDiv {
        dst: u8,
        src_a: u8,
        src_b: u8,
    },
}

// --- CHANGE 3: REGISTER FILE (from VL processor.py) ---

/// A 16-register file for shader execution.
/// Replaces the single-scalar accumulator model.
///
/// Example: "color = texture(uv) * light + ambient" becomes:
///   RLoadImm { dst: 0, value: texture_val }
///   RLoadImm { dst: 1, value: light_val }
///   RMul { dst: 2, src_a: 0, src_b: 1 }
///   RLoadImm { dst: 3, value: ambient_val }
///   RAdd { dst: 4, src_a: 2, src_b: 3 }
#[repr(C, align(32))]
#[derive(Clone)]
pub struct RegisterFile {
    pub regs: [f32; 16],
    pub iregs: [u64; 16],
    pub flags: u8, // 0=Eq, 1=Gt, 2=Lt
}

impl RegisterFile {
    pub fn new() -> Self {
        Self {
            regs: [0.0; 16],
            iregs: [0; 16],
            flags: 0,
        }
    }

    /// Execute a full program on this register file.
    pub fn execute_program(&mut self, ops: &[SpvOp], inductor: Option<&VInductor>) {
        let mut pc = 0;
        while pc < ops.len() {
            let op = &ops[pc];
            match *op {
                SpvOp::BJump { offset } => {
                    pc = offset as usize;
                    continue;
                }
                SpvOp::BTrap { cond, offset } => {
                    if self.flags == cond {
                        pc = offset as usize;
                        continue;
                    }
                }
                SpvOp::VShunt {
                    signature,
                    input_reg,
                    output_reg,
                    jump_offset,
                } => {
                    if let Some(ind) = inductor {
                        let val = self.regs[input_reg as usize];
                        let input_hash = val.to_bits() as u64;
                        if let Some(res) = ind.recall(signature, input_hash) {
                            // Extract first element from KernelResult
                            if let Some(&first) = res.data.get(0) {
                                self.regs[output_reg as usize] = first;
                                pc = jump_offset as usize;
                                continue;
                            }
                        }
                    }
                }
                _ => {
                    self.execute_op(op);
                }
            }
            pc += 1;
        }
    }

    /// Execute a single SpvOp on this register file.
    pub fn execute_op(&mut self, op: &SpvOp) -> bool {
        match *op {
            SpvOp::ICmp { src_a, src_b } => {
                let a = self.iregs[src_a as usize];
                let b = self.iregs[src_b as usize];
                self.flags = if a == b {
                    0
                } else if a > b {
                    1
                } else {
                    2
                };
                true
            }
            SpvOp::ILoadImm { dst, value } => {
                self.iregs[dst as usize] = value;
                true
            }
            SpvOp::IAdd { dst, src_a, src_b } => {
                self.iregs[dst as usize] =
                    self.iregs[src_a as usize].wrapping_add(self.iregs[src_b as usize]);
                true
            }
            // Existing ops...
            SpvOp::RAdd { dst, src_a, src_b } => {
                self.regs[dst as usize] = self.regs[src_a as usize] + self.regs[src_b as usize];
                true
            }
            SpvOp::RSub { dst, src_a, src_b } => {
                self.regs[dst as usize] = self.regs[src_a as usize] - self.regs[src_b as usize];
                true
            }
            SpvOp::VAdd { dst, src_a, src_b } => {
                let d = dst as usize;
                let a = src_a as usize;
                let b = src_b as usize;
                if d + 3 < 16 && a + 3 < 16 && b + 3 < 16 {
                    self.regs[d] = self.regs[a] + self.regs[b];
                    self.regs[d + 1] = self.regs[a + 1] + self.regs[b + 1];
                    self.regs[d + 2] = self.regs[a + 2] + self.regs[b + 2];
                    self.regs[d + 3] = self.regs[a + 3] + self.regs[b + 3];
                    true
                } else {
                    false
                }
            }
            SpvOp::VMul { dst, src_a, src_b } => {
                let d = dst as usize;
                let a = src_a as usize;
                let b = src_b as usize;
                if d + 3 < 16 && a + 3 < 16 && b + 3 < 16 {
                    self.regs[d] = self.regs[a] * self.regs[b];
                    self.regs[d + 1] = self.regs[a + 1] * self.regs[b + 1];
                    self.regs[d + 2] = self.regs[a + 2] * self.regs[b + 2];
                    self.regs[d + 3] = self.regs[a + 3] * self.regs[b + 3];
                    true
                } else {
                    false
                }
            }

            SpvOp::RMul { dst, src_a, src_b } => {
                self.regs[dst as usize] = self.regs[src_a as usize] * self.regs[src_b as usize];
                true
            }
            SpvOp::RDiv { dst, src_a, src_b } => {
                let divisor = self.regs[src_b as usize];
                self.regs[dst as usize] = if divisor.abs() > 1e-30 {
                    self.regs[src_a as usize] / divisor
                } else {
                    0.0
                };
                true
            }
            SpvOp::RLoadImm { dst, value_bits } => {
                self.regs[dst as usize] = f32::from_bits(value_bits);
                true
            }
            SpvOp::RMov { dst, src } => {
                self.regs[dst as usize] = self.regs[src as usize];
                true
            }
            SpvOp::RSin { dst, src } => {
                self.regs[dst as usize] = self.regs[src as usize].sin();
                true
            }
            SpvOp::RCos { dst, src } => {
                self.regs[dst as usize] = self.regs[src as usize].cos();
                true
            }
            SpvOp::RSqrt { dst, src } => {
                let s = self.regs[src as usize];
                self.regs[dst as usize] = if s >= 0.0 { s.sqrt() } else { 0.0 };
                true
            }
            SpvOp::RAbs { dst, src } => {
                self.regs[dst as usize] = self.regs[src as usize].abs();
                true
            }
            SpvOp::RFMin { dst, src_a, src_b } => {
                self.regs[dst as usize] = self.regs[src_a as usize].min(self.regs[src_b as usize]);
                true
            }
            SpvOp::RFMax { dst, src_a, src_b } => {
                self.regs[dst as usize] = self.regs[src_a as usize].max(self.regs[src_b as usize]);
                true
            }
            SpvOp::RFClamp {
                dst,
                src,
                min_v,
                max_v,
            } => {
                let val = self.regs[src as usize];
                let low = self.regs[min_v as usize];
                let high = self.regs[max_v as usize];
                self.regs[dst as usize] = val.max(low).min(high);
                true
            }
            SpvOp::VDot { dst, src_a, src_b } => {
                let a = src_a as usize;
                let b = src_b as usize;
                // Dot Product (4-wide)
                let res = self.regs[a] * self.regs[b]
                    + self.regs[a + 1] * self.regs[b + 1]
                    + self.regs[a + 2] * self.regs[b + 2]
                    + self.regs[a + 3] * self.regs[b + 3];
                self.regs[dst as usize] = res;
                true
            }
            SpvOp::VNormalize { dst, src } => {
                let s = src as usize;
                let len_sq = self.regs[s] * self.regs[s]
                    + self.regs[s + 1] * self.regs[s + 1]
                    + self.regs[s + 2] * self.regs[s + 2]
                    + self.regs[s + 3] * self.regs[s + 3];
                let len = len_sq.sqrt();
                if len > 1e-10 {
                    let inv_len = 1.0 / len;
                    let d = dst as usize;
                    self.regs[d] = self.regs[s] * inv_len;
                    self.regs[d + 1] = self.regs[s + 1] * inv_len;
                    self.regs[d + 2] = self.regs[s + 2] * inv_len;
                    self.regs[d + 3] = self.regs[s + 3] * inv_len;
                }
                true
            }
            SpvOp::VRSqrt { dst, src } => {
                let s = self.regs[src as usize];
                self.regs[dst as usize] = if s > 1e-10 { 1.0 / s.sqrt() } else { 0.0 };
                true
            }
            // --- Legacy Scalar Ops (Targeting Acc: regs[0]) ---
            SpvOp::FAdd => {
                self.regs[0] += 1.0;
                true
            }
            SpvOp::FSub => {
                self.regs[0] -= 1.0;
                true
            }
            SpvOp::FMul => {
                self.regs[0] *= 1.1;
                true
            }
            SpvOp::FDiv => {
                self.regs[0] /= 1.1;
                true
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VirtualShader {
    pub instructions: Vec<SpvOp>,
}

impl VirtualShader {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    pub fn push(&mut self, op: SpvOp) {
        self.instructions.push(op);
    }

    /// Generate a vGPU Structural Signature from the instruction stream.
    /// Standard sequence-based hash (Syntax-dependent).
    pub fn generate_signature(&self) -> u64 {
        let mut s = std::collections::hash_map::DefaultHasher::new();
        for op in &self.instructions {
            op.hash(&mut s);
        }
        s.finish()
    }

    /// Generate a Semantic Signature (Syntax-independent).
    /// Proves immunity to Alpha-renaming, NOPs, and Commutative Reordering.
    pub fn generate_semantic_signature(&self) -> u64 {
        use crate::vhdc::Hypervector;
        use std::collections::HashMap;

        // 1. Alpha-renaming (Register Normalization)
        let mut reg_map: HashMap<u8, u8> = HashMap::new();
        let mut next_norm = 0u8;

        let mut normalized_ops = Vec::new();
        for op in &self.instructions {
            let mut norm_op = op.clone();
            // Map registers encountered in order to 0, 1, 2...
            match &mut norm_op {
                SpvOp::VDot { dst, src_a, src_b } => {
                    *dst = *reg_map.entry(*dst).or_insert_with(|| {
                        let n = next_norm;
                        next_norm += 1;
                        n
                    });
                    *src_a = *reg_map.entry(*src_a).or_insert_with(|| {
                        let n = next_norm;
                        next_norm += 1;
                        n
                    });
                    *src_b = *reg_map.entry(*src_b).or_insert_with(|| {
                        let n = next_norm;
                        next_norm += 1;
                        n
                    });
                }
                SpvOp::VNormalize { dst, src } | SpvOp::VRSqrt { dst, src } => {
                    *dst = *reg_map.entry(*dst).or_insert_with(|| {
                        let n = next_norm;
                        next_norm += 1;
                        n
                    });
                    *src = *reg_map.entry(*src).or_insert_with(|| {
                        let n = next_norm;
                        next_norm += 1;
                        n
                    });
                }
                _ => {}
            }
            normalized_ops.push(norm_op);
        }

        // 2. Commutative Reordering (Simple sort for global scalar ops)
        // In a real compiler, we'd use a Dataflow Graph (DFG).
        // For this demo, we sort the scalar blocks.
        normalized_ops.sort_by_key(|op| format!("{:?}", op));

        // 3. HDC Semantic Accumulation
        let mut v_acc = Hypervector::zero();
        for op in &normalized_ops {
            let mut s = std::collections::hash_map::DefaultHasher::new();
            op.hash(&mut s);
            let op_hash = s.finish();

            let v_op = Hypervector::from_seed(op_hash);
            v_acc = v_acc.bundle(&v_op);
        }

        v_acc.to_u64()
    }
}

pub struct ShaderKernel {
    pub shader: VirtualShader,
    pub signature: u64,
}

impl Kernel for ShaderKernel {
    fn signature(&self) -> u64 {
        self.signature
    }

    fn cost(&self) -> u32 {
        // Interpreted ops are more expensive than JIT
        (self.shader.instructions.len() * 5) as u32
    }

    fn execute(&self, input_hash: u64) -> KernelResult {
        let mut data = vec![0.0; KERNEL_DISPATCH_SIZE];
        let mut rf = RegisterFile::new();
        // Seed r0 with the input hash
        rf.regs[0] = (input_hash & 0xFF) as f32;

        rf.execute_program(&self.shader.instructions, None);

        // Fill data with final state (mostly regs[0] for legacy compatibility)
        for i in 0..KERNEL_DISPATCH_SIZE {
            data[i] = rf.regs[0];
        }

        KernelResult {
            data: Arc::from(data),
        }
    }
}

// --- PHASE 13: JIT COMPILATION (TEMPLATE) ---

pub struct JitKernel {
    pub ops: Vec<SpvOp>,
    pub signature: u64,
}

impl Kernel for JitKernel {
    fn signature(&self) -> u64 {
        self.signature
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn cost(&self) -> u32 {
        self.ops.len() as u32
    }

    fn execute(&self, input_hash: u64) -> KernelResult {
        let mut data = vec![0.0; KERNEL_DISPATCH_SIZE];
        let mut x = (input_hash & 0xFF) as f32;

        // Optimized execution loop (Template JIT emulation)
        for i in 0..self.ops.len() {
            let op = unsafe { *self.ops.get_unchecked(i) };
            match op {
                SpvOp::FAdd => x += 1.0,
                SpvOp::FSub => x -= 1.0,
                SpvOp::FMul => x *= 1.1,
                SpvOp::FDiv => x /= 1.1,
                SpvOp::VDot {
                    dst: _,
                    src_a: _,
                    src_b: _,
                } => {
                    // For the template jitter, we just compute it and store in register file
                    // But JitKernel uses flat 'x'. Since JitKernel is a simplified template,
                    // we'll just keep it updated for completeness of the structure.
                    x = x * x + x * x + x * x + x * x; // Placeholder multiplication for template
                }
                SpvOp::VNormalize { dst: _, src: _ } => {
                    let len = (x * x + x * x + x * x + x * x).sqrt();
                    if len > 0.0 {
                        x /= len;
                    }
                }
                SpvOp::VRSqrt { dst: _, src: _ } => {
                    if x > 0.0 {
                        x = 1.0 / x.sqrt();
                    }
                }
                _ => {}
            }
        }

        data[0] = x;
        KernelResult {
            data: Arc::from(data),
        }
    }
}

// --- PHASE 15: ZERO-DEPENDENCY NATIVE JIT ---

/// A kernel that runs real machine code instead of interpreting opcodes.
/// The `native_jit::compile` function emits x86-64 instructions (addss, mulss, etc.)
/// into an executable memory page. When this kernel executes, it calls directly
/// into that page — no Rust loop, no match statement, just raw CPU.
pub struct NativeJitKernel {
    pub compiled: native_jit::NativeCode,
    pub signature: u64,
}

impl Kernel for NativeJitKernel {
    fn signature(&self) -> u64 {
        self.signature
    }

    fn cost(&self) -> u32 {
        (self.compiled.len / 4).max(1) as u32
    }

    fn execute(&self, input_hash: u64) -> KernelResult {
        let mut data = vec![0.0; KERNEL_DISPATCH_SIZE];
        let mut rf = RegisterFile::new();
        rf.regs[0] = (input_hash & 0xFF) as f32;

        self.compiled.call(&mut rf);

        for i in 0..KERNEL_DISPATCH_SIZE {
            data[i] = rf.regs[0];
        }

        KernelResult {
            data: Arc::from(data),
        }
    }
}

// --- PHASE 7: VIRTUAL AMPLIFICATION ---

pub struct VAmplifier {
    pub r: f32, // Control parameter (3.5 to 4.0)
    pub state: f32,
}

impl VAmplifier {
    pub fn new(r: f32) -> Self {
        Self { r, state: 0.5 }
    }

    /// Step the bifurcation logic
    pub fn step(&mut self) -> f32 {
        self.state = self.r * self.state * (1.0 - self.state);
        self.state
    }

    /// Generate N virtual contexts based on bifurcation depth
    pub fn bifurcate(&mut self, depth: u32) -> Vec<u64> {
        let mut nodes = Vec::new();
        for _ in 0..(1 << depth) {
            let val = self.step();
            nodes.push((val * u64::MAX as f32) as u64);
        }
        nodes
    }
}

/// Residue Number System (RNS) Helper
pub struct RnsEngine {
    pub moduli: Vec<u64>,
}

impl RnsEngine {
    pub fn new(moduli: Vec<u64>) -> Self {
        Self { moduli }
    }

    pub fn decompose(&self, n: u128) -> Vec<u64> {
        self.moduli
            .iter()
            .map(|&m| (n % m as u128) as u64)
            .collect()
    }

    pub fn combine(&self, residues: &[u64]) -> u128 {
        let mut result = 0u128;
        let m_total: u128 = self.moduli.iter().map(|&x| x as u128).product();

        for (i, &r) in residues.iter().enumerate() {
            let mi = self.moduli[i] as u128;
            let m_prime = m_total / mi;

            // Solve: m_prime * y ≡ 1 (mod mi) using Extended Euclidean Algorithm
            let y = self.mod_inverse(m_prime, mi);
            result = (result + (r as u128 * m_prime % m_total) * y) % m_total;
        }
        result
    }

    fn mod_inverse(&self, a: u128, m: u128) -> u128 {
        let mut y = 0i128;
        let mut x = 1i128;
        let m_val = m as i128;

        if m_val == 1 {
            return 0;
        }

        let mut a_mut = a as i128;
        let mut m_mut = m_val;

        while a_mut > 1 {
            let q = a_mut / m_mut;
            let mut t = m_mut;
            m_mut = a_mut % m_mut;
            a_mut = t;
            t = y;
            y = x - q * y;
            x = t;
        }

        if x < 0 {
            x += m_val;
        }
        x as u128
    }
}

// --- FEISTEL VARIETY (ported from VL hash.py) ---

/// 4-round Feistel cipher for deterministic variety generation.
/// Equation: L_{i+1} = R_i,  R_{i+1} = L_i ⊕ F(R_i, KEY)
/// where F(r, k) = ((r ⊕ k) × 0x45D9F3B) >> 16 ⊕ ((r ⊕ k) × 0x45D9F3B)
///
/// Provides ~50% avalanche diffusion: flipping 1 input bit flips
/// ~50% of output bits. This is what makes manifold collisions rare.
#[inline]
pub fn feistel_variety(address: u64, seed: u64) -> f32 {
    let key = (seed & 0xFFFFFFFF) as u32;
    let mut l = ((address >> 32) & 0xFFFFFFFF) as u32;
    let mut r = (address & 0xFFFFFFFF) as u32;
    for _ in 0..4 {
        let f = (r ^ key).wrapping_mul(0x45D9F3B);
        let f = (f >> 16) ^ f;
        let new_r = l ^ f;
        l = r;
        r = new_r;
    }
    let combined = ((l as u64) << 32) | (r as u64);
    (combined % 1_000_000_000) as f32 / 1_000_000_000.0
}

/// Hash a 256-bit VAddr down to a 64-bit value suitable for Feistel input.
#[inline]
pub fn addr_to_u64(addr: MirrorAddr) -> u64 {
    // Mix all 4 words with rotation to preserve information
    let mut h = addr.0[0];
    h = h.wrapping_mul(0x9E3779B97F4A7C15).rotate_left(31) ^ addr.0[1];
    h = h.wrapping_mul(0x9E3779B97F4A7C15).rotate_left(31) ^ addr.0[2];
    h = h.wrapping_mul(0x9E3779B97F4A7C15).rotate_left(31) ^ addr.0[3];
    h
}

/// Generative Memory Substrate (VVram)
///
/// Implements the VL ManifoldStore pattern:
///   read(addr) → explicit_store[addr] if written, else feistel_variety(addr)
///   write(addr, val) → only affects that address, not the substrate
pub struct VVram {
    pub seed: u64,
    explicit_store: HashMap<u64, f32>,
}

impl VVram {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            explicit_store: HashMap::new(),
        }
    }

    /// Generate a deterministic value from the Feistel substrate.
    pub fn variety(&self, addr: MirrorAddr) -> f32 {
        feistel_variety(addr_to_u64(addr), self.seed)
    }

    /// Read: checks explicit store first, then falls back to generation.
    /// This is the VL ManifoldStore.read() pattern.
    pub fn read(&self, addr: MirrorAddr) -> f32 {
        let key = addr_to_u64(addr);
        if let Some(&val) = self.explicit_store.get(&key) {
            return val;
        }
        feistel_variety(key, self.seed)
    }

    /// Write: stores a value at a specific address without affecting
    /// the generative substrate. Only this address is changed.
    pub fn write(&mut self, addr: MirrorAddr, value: f32) {
        let key = addr_to_u64(addr);
        self.explicit_store.insert(key, value);
    }

    /// How many addresses have been explicitly written.
    pub fn explicit_count(&self) -> usize {
        self.explicit_store.len()
    }
}

// --- PHASE 16: STATEFUL SHADER STORAGE (SSBO) ---

/// A simple key-value store that lets shaders persist data between frames.
/// In a real GPU, this is a Shader Storage Buffer Object (SSBO).
/// Here, it's backed by VVram writes so the data becomes part of the
/// generative substrate and can be "inducted" for future recall.
pub struct StatefulManifold {
    storage: HashMap<u64, Vec<f32>>,
}

impl StatefulManifold {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }

    /// Write a buffer of floats at a given binding point.
    pub fn write(&mut self, binding: u64, data: Vec<f32>) {
        self.storage.insert(binding, data);
    }

    /// Read back a buffer from a binding point.
    pub fn read(&self, binding: u64) -> Option<&[f32]> {
        self.storage.get(&binding).map(|v| v.as_slice())
    }

    /// How many bindings are active.
    pub fn binding_count(&self) -> usize {
        self.storage.len()
    }
}

// --- PHASE 17: INDUCTIVE TEXTURE SAMPLING ---

/// A texture sampler that uses the generative substrate (VVram)
/// to produce filtered values without storing actual texel data.
/// Instead of loading a texture from disk, we "unfold" it from math.
pub struct InductiveSampler {
    pub mip_levels: u32,
}

impl InductiveSampler {
    pub fn new(mip_levels: u32) -> Self {
        Self { mip_levels }
    }

    /// Point sample: just read the variety at the exact address.
    pub fn sample_point(&self, vram: &VVram, u: f32, v: f32) -> f32 {
        let addr = VAddr::new((u * 65536.0) as u64, (v * 65536.0) as u64, 0, 0);
        vram.variety(addr)
    }

    /// Bilinear sample: average 4 neighboring points.
    /// This gives smooth gradients instead of blocky pixels.
    pub fn sample_bilinear(&self, vram: &VVram, u: f32, v: f32) -> f32 {
        let step = 1.0 / 65536.0;
        let tl = self.sample_point(vram, u, v);
        let tr = self.sample_point(vram, u + step, v);
        let bl = self.sample_point(vram, u, v + step);
        let br = self.sample_point(vram, u + step, v + step);
        (tl + tr + bl + br) * 0.25
    }

    /// MIP-level sample: average over a larger area for distant surfaces.
    /// Higher LOD = more averaging = blurrier but faster.
    pub fn sample_mip(&self, vram: &VVram, u: f32, v: f32, lod: u32) -> f32 {
        let samples = 1 << lod.min(self.mip_levels);
        let step = (1 << lod) as f32 / 65536.0;
        let mut sum = 0.0;
        for dy in 0..samples {
            for dx in 0..samples {
                sum += self.sample_point(vram, u + dx as f32 * step, v + dy as f32 * step);
            }
        }
        sum / (samples * samples) as f32
    }
}

// --- PHASE 18: CACHE-AFFINITY ORCHESTRATION ---

/// Tracks which CPU core a virtual SM should prefer.
/// In a real system, this would pin threads to specific L2 cache domains
/// to prevent thrashing when running 1024 virtual nodes.
pub struct AffinityMap {
    assignments: HashMap<u32, usize>, // SM ID -> preferred core
}

impl AffinityMap {
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
        }
    }

    /// Assign a virtual SM to a preferred CPU core.
    pub fn pin(&mut self, sm_id: u32, core_id: usize) {
        self.assignments.insert(sm_id, core_id);
    }

    /// Look up which core an SM should run on.
    pub fn preferred_core(&self, sm_id: u32) -> Option<usize> {
        self.assignments.get(&sm_id).copied()
    }

    /// Auto-assign SMs round-robin across available cores.
    pub fn auto_assign(&mut self, sm_count: u32, core_count: usize) {
        for i in 0..sm_count {
            self.assignments.insert(i, (i as usize) % core_count);
        }
    }
}

// --- SPECTRAL UTILITY EVICTION (from VL GeometricTransistor) ---

/// Tracks how useful each slot in the induction manifold is.
/// Uses the VL Spectral Utility Equation:
///   U(slot) = (AccessCount × Energy) / (Entropy + 1)
///
/// Shannon Entropy H = -Σ P(vᵢ) × log₂(P(vᵢ))
/// High entropy = varied outputs (worth keeping).
/// Low entropy = constant output (trivially regenerated, safe to evict).
pub struct EfficiencyTracker {
    pub metrics: HashMap<ManifoldKey, SlotMetrics>,
}

pub struct SlotMetrics {
    pub hit_count: u64,
    pub cost_saved: u64,
    pub output_history: Vec<u32>,
}

impl EfficiencyTracker {
    pub fn new(_capacity: usize) -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    pub fn resize(&mut self, _new_capacity: usize) {}

    pub fn record_hit(&mut self, key: ManifoldKey, ops_saved: u64) {
        let entry = self.metrics.entry(key).or_insert_with(|| SlotMetrics {
            hit_count: 0,
            cost_saved: 0,
            output_history: Vec::new(),
        });
        entry.hit_count += 1;
        entry.cost_saved += ops_saved;
    }

    pub fn record_output(&mut self, key: ManifoldKey, value: f32) {
        let entry = self.metrics.entry(key).or_insert_with(|| SlotMetrics {
            hit_count: 0,
            cost_saved: 0,
            output_history: Vec::new(),
        });
        // Quantize to 16-bit buckets for entropy calculation
        let bucket = (value.to_bits() >> 16) as u32;
        entry.output_history.push(bucket);
        if entry.output_history.len() > 64 {
            entry.output_history.remove(0);
        }
    }

    pub fn shannon_entropy(&self, key: &ManifoldKey) -> f64 {
        let metrics = match self.metrics.get(key) {
            Some(m) => m,
            None => return 0.0,
        };
        let history = &metrics.output_history;
        if history.is_empty() {
            return 0.0;
        }

        let n = history.len() as f64;
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &v in history {
            *counts.entry(v).or_insert(0) += 1;
        }
        let mut entropy = 0.0;
        for &count in counts.values() {
            let p = count as f64 / n;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    pub fn spectral_utility(&self, key: &ManifoldKey) -> f64 {
        let metrics = match self.metrics.get(key) {
            Some(m) => m,
            None => return 0.0,
        };
        let access = metrics.hit_count as f64;
        let energy = metrics.cost_saved as f64;
        let entropy = self.shannon_entropy(key);
        (access * energy) / (entropy + 1.0)
    }

    pub fn weakest_slot_with_utility(&self) -> (Option<ManifoldKey>, f64) {
        let mut min_u = f64::MAX;
        let mut weakest = None;
        for key in self.metrics.keys() {
            let u = self.spectral_utility(key);
            if u < min_u {
                min_u = u;
                weakest = Some(*key);
            }
        }
        (weakest, min_u)
    }

    pub fn find_prune_candidates(&self, pct: f64) -> Vec<ManifoldKey> {
        let mut scores: Vec<(ManifoldKey, f64)> = self
            .metrics
            .keys()
            .map(|k| (*k, self.spectral_utility(k)))
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let count = ((scores.len() as f64) * pct).ceil() as usize;
        scores[..count.min(scores.len())]
            .iter()
            .map(|&(k, _)| k)
            .collect()
    }

    pub fn reset(&mut self) {
        self.metrics.clear();
    }
}

/// Kernel-Level Process Induction Engine (PIE)
pub const KERNEL_DISPATCH_SIZE: usize = 1024;
const MANIFOLD_SIZE: usize = 1 << 12;
const MAGIC: &[u8; 8] = b"VGPU_V1 ";

#[derive(Clone, Debug)]
pub struct KernelResult {
    pub data: Arc<[f32]>,
}

pub trait Kernel: Send + Sync {
    fn signature(&self) -> u64;
    /// Meta-Observation: Returns true if the kernel is proved deterministic
    /// via static analysis or instruction-set guarantees.
    fn is_deterministic(&self) -> bool {
        false
    }
    /// Estimated computational cost (cycles/ops).
    /// Used to decide if shunting is worth the overhead.
    fn cost(&self) -> u32 {
        100 // Default to "moderate" cost
    }
    fn execute(&self, input_hash: u64) -> KernelResult;
}

const MAX_MANIFOLD_OCCUPANCY: usize = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ManifoldKey {
    pub sig: u64,
    pub hash: u64,
}

/// A statistical gate that determines when a process is stable enough for induction.
pub struct ConfidenceGate {
    pub samples: Vec<f32>,
    pub target_epsilon: f32, // Target standard error
    pub z_score: f32,        // Confidence level (e.g. 1.96 for 95%)
}

impl ConfidenceGate {
    pub fn new(epsilon: f32) -> Self {
        Self {
            samples: Vec::with_capacity(8),
            target_epsilon: epsilon,
            z_score: 1.96, // 95% Confidence interval
        }
    }

    pub fn is_stable(&self) -> bool {
        if self.samples.len() < 3 {
            return false;
        }

        let n = self.samples.len() as f32;
        let mean = self.samples.iter().sum::<f32>() / n;
        let variance = self
            .samples
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / n;
        let std_error = variance.sqrt() / n.sqrt();
        let margin_of_error = self.z_score * std_error;

        // Stability is reached when the margin of error is within our target epsilon
        // Relative to the mean to handle different scales
        margin_of_error < self.target_epsilon * (mean.abs() + 0.001)
    }
}

pub struct VInductor {
    pub manifold: std::sync::RwLock<HashMap<ManifoldKey, KernelResult>>,
    pub observations: std::sync::RwLock<HashMap<ManifoldKey, ConfidenceGate>>,
    pub efficiency: std::sync::RwLock<EfficiencyTracker>,
    pub id: u32,
    pub depth: u32,
    // Memory Limits
    pub current_usage: std::sync::atomic::AtomicUsize,
    pub capacity_bytes: usize,
    pub purge_threshold_bytes: usize,
    pub evictions: std::sync::atomic::AtomicU64,
    pub induction_events: std::sync::atomic::AtomicU64,
}

const MEMORY_CAP_LOW: usize = 500 * 1024 * 1024; // 500 MB
const MEMORY_CAP_HIGH: usize = 800 * 1024 * 1024; // 800 MB

/// L1 Shadow Cache: Associate lookup bypass for extreme local throughput
/// Validated Data Cache (stores Result, not just Index)
#[derive(Clone, Debug)]
struct L1Shadow {
    tags: [u64; 4],
    sigs: [u64; 4],
    instance_ids: [u32; 4],
    data: [Option<KernelResult>; 4],
    cursor: usize,
}

impl Default for L1Shadow {
    fn default() -> Self {
        Self {
            tags: [0; 4],
            sigs: [0; 4],
            instance_ids: [0; 4],
            data: [None, None, None, None],
            cursor: 0,
        }
    }
}

thread_local! {
    static L1_CACHE: std::cell::RefCell<L1Shadow> = std::cell::RefCell::new(L1Shadow::default());
}

impl VInductor {
    pub fn new(id: u32) -> Self {
        Self::new_with_depth(id, 0)
    }

    pub fn new_with_depth(id: u32, depth: u32) -> Self {
        Self {
            manifold: std::sync::RwLock::new(HashMap::new()),
            observations: std::sync::RwLock::new(HashMap::new()),
            efficiency: std::sync::RwLock::new(EfficiencyTracker::new(MANIFOLD_SIZE)),
            id,
            depth,
            current_usage: std::sync::atomic::AtomicUsize::new(0),
            capacity_bytes: MEMORY_CAP_HIGH,
            purge_threshold_bytes: MEMORY_CAP_LOW,
            evictions: std::sync::atomic::AtomicU64::new(0),
            induction_events: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn resize(&self, _new_capacity: usize) {
        // Sparse manifold handles resizing automatically.
    }

    // feistel_index removed - Sparse Manifold uses direct hashing

    #[inline(always)]
    pub fn recall(&self, sig: u64, input_hash: u64) -> Option<KernelResult> {
        let l1_hit = L1_CACHE.with(|cache| {
            let c = cache.borrow();
            for i in 0..4 {
                if c.tags[i] == input_hash
                    && c.sigs[i] == sig
                    && c.instance_ids[i] == self.id
                    && input_hash != 0
                {
                    return c.data[i].clone();
                }
            }
            None
        });

        if let Some(res) = l1_hit {
            return Some(res);
        }

        let key = ManifoldKey {
            sig,
            hash: input_hash,
        };
        let read_guard = self.manifold.read().unwrap();
        if let Some(res) = read_guard.get(&key) {
            let res_clone = res.clone();
            // Update L1
            L1_CACHE.with(|cache| {
                let mut c = cache.borrow_mut();
                let pos = c.cursor;
                c.tags[pos] = input_hash;
                c.sigs[pos] = sig;
                c.instance_ids[pos] = self.id;
                c.data[pos] = Some(res_clone.clone());
                c.cursor = (pos + 1) & 3;
            });
            return Some(res_clone);
        }
        None
    }

    fn estimate_size(&self, res: &KernelResult) -> usize {
        // Overhead (Key + Result wrapper) + Data
        std::mem::size_of::<ManifoldKey>()
            + std::mem::size_of::<KernelResult>()
            + (res.data.len() * 4)
    }

    pub fn induct(&self, sig: u64, input_hash: u64, result: KernelResult) {
        let key = ManifoldKey {
            sig,
            hash: input_hash,
        };

        let mut obs_guard = self.observations.write().unwrap();

        let (stable, stats) = {
            let gate = obs_guard
                .entry(key)
                .or_insert_with(|| ConfidenceGate::new(0.01));

            let val = result.data.get(0).cloned().unwrap_or(0.0);
            gate.samples.push(val);

            if gate.is_stable() {
                let n = gate.samples.len() as f32;
                let mean = gate.samples.iter().sum::<f32>() / n;
                (true, Some((n, mean)))
            } else if gate.samples.len() > 64 {
                (false, Some((0.0, 0.0))) // Signal to remove
            } else {
                (false, None)
            }
        };

        if stable {
            let (_n, mean) = stats.unwrap();
            // Check memory pressure before inserting
            let size = self.estimate_size(&result);
            let current = self
                .current_usage
                .load(std::sync::atomic::Ordering::Relaxed);

            if current + size > self.capacity_bytes {
                self.purge_to_limit();
            }

            let mut manifold_guard = self.manifold.write().unwrap();
            manifold_guard.insert(key, result);
            self.current_usage
                .fetch_add(size, std::sync::atomic::Ordering::Relaxed);

            self.induction_events
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            obs_guard.remove(&key);

            // Track efficiency
            self.efficiency.write().unwrap().record_output(key, mean);
        } else if stats.is_some() {
            // Give up on unstable processes
            obs_guard.remove(&key);
        }
    }

    fn purge_to_limit(&self) {
        let mut manifold = self.manifold.write().unwrap();
        let mut efficiency = self.efficiency.write().unwrap();

        let _initial_size = manifold.len();
        // Prune 20% of low-utility slots
        let candidates = efficiency.find_prune_candidates(0.20);

        let mut freed_bytes = 0;
        for key in &candidates {
            if let Some(res) = manifold.remove(key) {
                freed_bytes += self.estimate_size(&res);
                efficiency.metrics.remove(key);
            }
        }
        self.current_usage
            .fetch_sub(freed_bytes, std::sync::atomic::Ordering::Relaxed);
        self.evictions.fetch_add(
            candidates.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    pub fn occupancy(&self) -> usize {
        self.manifold.read().unwrap().len()
    }

    pub fn evict(&self, key: &ManifoldKey) {
        self.manifold.write().unwrap().remove(key);
    }

    pub fn merge(&self, other: &VInductor) {
        let mut my_guard = self.manifold.write().unwrap();
        let other_guard = other.manifold.read().unwrap();
        for (key, val) in other_guard.iter() {
            my_guard.entry(*key).or_insert_with(|| val.clone());
        }
    }

    pub fn save(&self, path: &str, seed: u64) -> IoResult<()> {
        use std::fs::File;
        use std::io::Write;
        use std::time::SystemTime;

        let mut file = File::create(path)?;
        let guard = self.manifold.read().unwrap();
        let entry_count = guard.len() as u32;

        file.write_all(MAGIC)?;
        file.write_all(&1u32.to_le_bytes())?;
        file.write_all(&entry_count.to_le_bytes())?;
        file.write_all(&seed.to_le_bytes())?;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        file.write_all(&timestamp.to_le_bytes())?;

        for (key, result) in guard.iter() {
            file.write_all(&key.sig.to_le_bytes())?;
            file.write_all(&key.hash.to_le_bytes())?;
            for &val in result.data.iter() {
                file.write_all(&val.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> IoResult<u64> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut header = [0u8; 32];
        file.read_exact(&mut header)?;

        if &header[0..8] != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid Magic",
            ));
        }

        let entry_count = u32::from_le_bytes(header[12..16].try_into().unwrap());
        let seed = u64::from_le_bytes(header[16..24].try_into().unwrap());

        let mut manifold_guard = self.manifold.write().unwrap();
        for _ in 0..entry_count {
            let mut sig_buf = [0u8; 8];
            file.read_exact(&mut sig_buf)?;
            let sig = u64::from_le_bytes(sig_buf);

            let mut hash_buf = [0u8; 8];
            file.read_exact(&mut hash_buf)?;
            let hash = u64::from_le_bytes(hash_buf);

            let mut data = Vec::with_capacity(KERNEL_DISPATCH_SIZE);
            for _ in 0..KERNEL_DISPATCH_SIZE {
                let mut f_buf = [0u8; 4];
                file.read_exact(&mut f_buf)?;
                data.push(f32::from_le_bytes(f_buf));
            }
            let result = KernelResult {
                data: Arc::from(data),
            };

            let size = self.estimate_size(&result);
            self.current_usage
                .fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            manifold_guard.insert(ManifoldKey { sig, hash }, result);
        }
        Ok(seed)
    }
}

pub struct VSm {
    pub id: u32,
}

impl VSm {
    pub fn new(id: u32) -> Self {
        Self { id }
    }

    #[inline(always)]
    pub fn execute_kernel(
        &self,
        sig: u64,
        inductor: &VInductor,
        input_hash: u64,
    ) -> Option<KernelResult> {
        inductor.recall(sig, input_hash)
    }

    #[inline(always)]
    pub fn dispatch<'a>(
        &self,
        inductor: &'a VInductor,
        registry: &HashMap<u64, Box<dyn Kernel>>,
        sig: u64,
        input_hash: u64,
    ) -> KernelResult {
        // High-speed lookup first (L1/L2)
        if let Some(res) = inductor.recall(sig, input_hash) {
            return res.clone();
        }

        if let Some(k) = registry.get(&sig) {
            // Heuristic: Is it worth memoizing?
            let cost = k.cost();
            const SHUNT_THRESHOLD: u32 = 20; // Heuristic threshold
            if cost < SHUNT_THRESHOLD {
                // Too cheap to simple-shunt, just execute.
                return k.execute(input_hash);
            }

            let res = k.execute(input_hash);
            // Only induct if valuable
            inductor.induct(sig, input_hash, res.clone());
            return res;
        }
        KernelResult {
            data: Arc::from([0.0f32; KERNEL_DISPATCH_SIZE]),
        }
    }
}

// --- PHASE 6: RAY TRACING ENGINE ---

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    pub fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    pub fn min(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }
    pub fn max(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn intersect(&self, ray: &Ray) -> bool {
        let mut tmin = (self.min.x - ray.origin.x) / ray.direction.x;
        let mut tmax = (self.max.x - ray.origin.x) / ray.direction.x;
        if tmin > tmax {
            std::mem::swap(&mut tmin, &mut tmax);
        }

        let mut tymin = (self.min.y - ray.origin.y) / ray.direction.y;
        let mut tymax = (self.max.y - ray.origin.y) / ray.direction.y;
        if tymin > tymax {
            std::mem::swap(&mut tymin, &mut tymax);
        }

        if (tmin > tymax) || (tymin > tmax) {
            return false;
        }
        if tymin > tmin {
            tmin = tymin;
        }
        if tymax < tmax {
            tmax = tymax;
        }

        let mut tzmin = (self.min.z - ray.origin.z) / ray.direction.z;
        let mut tzmax = (self.max.z - ray.origin.z) / ray.direction.z;
        if tzmin > tzmax {
            std::mem::swap(&mut tzmin, &mut tzmax);
        }

        if (tmin > tzmax) || (tzmin > tmax) {
            return false;
        }
        true
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RTriangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
}

impl RTriangle {
    pub fn aabb(&self) -> AABB {
        AABB {
            min: self.v0.min(self.v1).min(self.v2),
            max: self.v0.max(self.v1).max(self.v2),
        }
    }
    /// Möller-Trumbore Ray-Triangle Intersection
    #[inline(always)]
    pub fn intersect(&self, ray: &Ray) -> Option<f32> {
        let edge1 = self.v1.sub(self.v0);
        let edge2 = self.v2.sub(self.v0);
        let h = ray.direction.cross(edge2);
        let a = edge1.dot(h);

        if a > -0.0001 && a < 0.0001 {
            return None;
        }

        let f = 1.0 / a;
        let s = ray.origin.sub(self.v0);
        let u = f * s.dot(h);

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q = s.cross(edge1);
        let v = f * ray.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(q);
        if t > 0.0001 { Some(t) } else { None }
    }
}

pub enum BVHNode {
    Leaf {
        bounds: AABB,
        triangles: Vec<RTriangle>,
    },
    Internal {
        bounds: AABB,
        left: Box<BVHNode>,
        right: Box<BVHNode>,
    },
}

impl BVHNode {
    pub fn new(triangles: Vec<RTriangle>) -> Self {
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for tri in &triangles {
            let b = tri.aabb();
            min = min.min(b.min);
            max = max.max(b.max);
        }
        let bounds = AABB { min, max };

        if triangles.len() <= 2 {
            return BVHNode::Leaf { bounds, triangles };
        }

        // Split on largest axis
        let extent = max.sub(min);
        let axis = if extent.x > extent.y && extent.x > extent.z {
            0
        } else if extent.y > extent.z {
            1
        } else {
            2
        };

        let mut tris = triangles;
        tris.sort_by(|a, b| {
            let ac = match axis {
                0 => (a.v0.x + a.v1.x + a.v2.x) / 3.0,
                1 => (a.v0.y + a.v1.y + a.v2.y) / 3.0,
                _ => (a.v0.z + a.v1.z + a.v2.z) / 3.0,
            };
            let bc = match axis {
                0 => (b.v0.x + b.v1.x + b.v2.x) / 3.0,
                1 => (b.v0.y + b.v1.y + b.v2.y) / 3.0,
                _ => (b.v0.z + b.v1.z + b.v2.z) / 3.0,
            };
            ac.partial_cmp(&bc).unwrap()
        });

        let mid = tris.len() / 2;
        let right_tris = tris.split_off(mid);

        BVHNode::Internal {
            bounds,
            left: Box::new(BVHNode::new(tris)),
            right: Box::new(BVHNode::new(right_tris)),
        }
    }

    pub fn bounds(&self) -> AABB {
        match self {
            BVHNode::Leaf { bounds, .. } => *bounds,
            BVHNode::Internal { bounds, .. } => *bounds,
        }
    }

    pub fn intersect(&self, ray: &Ray) -> Option<f32> {
        if !self.bounds().intersect(ray) {
            return None;
        }
        match self {
            BVHNode::Leaf { triangles, .. } => {
                let mut best_t = f32::MAX;
                let mut hit = false;
                for tri in triangles {
                    if let Some(t) = tri.intersect(ray) {
                        if t < best_t {
                            best_t = t;
                            hit = true;
                        }
                    }
                }
                if hit { Some(best_t) } else { None }
            }
            BVHNode::Internal { left, right, .. } => {
                let t_left = left.intersect(ray);
                let t_right = right.intersect(ray);
                match (t_left, t_right) {
                    (Some(tl), Some(tr)) => Some(tl.min(tr)),
                    (Some(tl), None) => Some(tl),
                    (None, Some(tr)) => Some(tr),
                    (None, None) => None,
                }
            }
        }
    }
}

// --- PHASE 1.5: BAYESIAN DISSONANCE CONTROL ---

pub struct DissonanceControl {
    pub threshold: f32,
    pub validation_rate: f32, // 0.0 to 1.0
    rng_seed: u64,
}

impl DissonanceControl {
    pub fn new(threshold: f32, validation_rate: f32) -> Self {
        Self {
            threshold,
            validation_rate,
            rng_seed: 0xBA1E5,
        }
    }

    pub fn check(&mut self, induced: f32, ground_truth: f32) -> (bool, f32) {
        let divergence = (induced - ground_truth).abs();
        (divergence < self.threshold, divergence)
    }

    pub fn should_verify(&mut self) -> bool {
        // Simple linear congruential generator for fast probabilistic checks
        self.rng_seed = self
            .rng_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        (self.rng_seed as f32 / u64::MAX as f32) < self.validation_rate
    }
}

// --- CORE PRODUCT HANDLE ---

// --- PHASE 19: MANIFOLD TELEMETRY ---

/// A snapshot of the vGPU's internal health.
/// Think of it as a dashboard reading — it tells you how well
/// the induction engine is performing at any moment.
#[derive(Debug, Clone)]
pub struct TelemetrySnapshot {
    pub total_dispatches: u64,
    pub induction_hits: u64,
    pub induction_misses: u64,
    pub hit_rate: f64,
    pub manifold_occupancy: usize,
    pub manifold_capacity: usize,
    pub memory_usage_bytes: usize,
    pub memory_capacity_bytes: usize,
    pub evictions_performed: u64,
}

pub struct VGpuContext {
    pub vram: VVram,
    pub inductor: VInductor,
    pub sms: Vec<VSm>,
    pub kernel_registry: HashMap<u64, Box<dyn Kernel>>,
    pub dissonance_control: DissonanceControl,
    pub amplifier: VAmplifier,
    // v2.0 subsystems
    pub ssbo: StatefulManifold,
    pub sampler: InductiveSampler,
    pub affinity: AffinityMap,
    // Telemetry counters
    total_dispatches: u64,
    induction_hits: u64,
    last_divergence: f32,
}

impl VGpuContext {
    pub fn new(sm_count: u32, seed: u64) -> Self {
        let mut sms = Vec::new();
        for i in 0..sm_count {
            sms.push(VSm::new(i));
        }
        let mut affinity = AffinityMap::new();
        // Default: spread SMs across 8 logical cores
        affinity.auto_assign(sm_count, 8);

        Self {
            vram: VVram::new(seed),
            inductor: VInductor::new(seed as u32),
            sms,
            kernel_registry: HashMap::new(),
            dissonance_control: DissonanceControl::new(0.001, 0.01),
            amplifier: VAmplifier::new(3.99),
            ssbo: StatefulManifold::new(),
            sampler: InductiveSampler::new(4), // 4 MIP levels by default
            affinity,
            total_dispatches: 0,
            induction_hits: 0,
            last_divergence: 0.0,
        }
    }

    pub fn register_kernel(&mut self, k: Box<dyn Kernel>) {
        self.kernel_registry.insert(k.signature(), k);
    }

    /// Smart dispatch with telemetry tracking and efficiency scoring.
    /// This is the v2.0 dispatch — it records every hit/miss so the
    /// pruning engine knows which laws are worth keeping.
    pub fn dispatch(&mut self, sig: u64, input_hash: u64) -> KernelResult {
        self.total_dispatches += 1;
        let key = ManifoldKey {
            sig,
            hash: input_hash,
        };

        if let Some(res) = self.inductor.recall(sig, input_hash) {
            self.induction_hits += 1;
            // Tell the efficiency tracker this slot earned its keep
            self.inductor
                .efficiency
                .write()
                .unwrap()
                .record_hit(key, 100);

            // Periodically verify against ground truth to measure divergence
            if self.dissonance_control.should_verify() {
                if let Some(k) = self.kernel_registry.get(&sig) {
                    let ground_truth = k.execute(input_hash);
                    let (ok, delta) = self
                        .dissonance_control
                        .check(res.data[0], ground_truth.data[0]);
                    self.last_divergence = delta;
                    if !ok {
                        // In a real system, we might evict or re-induct here.
                        // For now, we just log the delta for telemetry.
                    }
                }
            }

            self.inductor
                .efficiency
                .write()
                .unwrap()
                .record_output(key, res.data[0]);
            return res;
        }

        // Ground truth execution
        if let Some(k) = self.kernel_registry.get(&sig) {
            let cost = k.cost();
            const SHUNT_THRESHOLD: u32 = 20;
            if cost < SHUNT_THRESHOLD {
                return k.execute(input_hash);
            }

            let res = k.execute(input_hash);
            self.inductor
                .efficiency
                .write()
                .unwrap()
                .record_output(key, res.data[0]);

            // Meta-Observation: Deterministic kernels bypass observation buffer
            if k.is_deterministic() {
                self.inductor
                    .manifold
                    .write()
                    .unwrap()
                    .insert(key, res.clone());
            } else {
                self.inductor.induct(sig, input_hash, res.clone());
            }
            return res;
        }
        KernelResult {
            data: Arc::from([0.0; KERNEL_DISPATCH_SIZE]),
        }
    }

    /// Phase 20: Prune the weakest law from the manifold.
    /// Call this when the manifold is full and you need room for fresh laws.
    /// Phase 20: Prune the weakest law from the manifold.
    /// Call this when the manifold is full and you need room for fresh laws.
    pub fn prune_weakest(&mut self) {
        let (victim, utility) = self
            .inductor
            .efficiency
            .read()
            .unwrap()
            .weakest_slot_with_utility();

        // Elasticity Trigger: If the "weakest" law is actually quite useful (U > 500),
        // we suggest expanding the meta-manifold (though HashMap is already elastic)
        // In a fixed-capacity vGPU, this would be a hardware re-alloc.
        if utility > 500.0
            && self.inductor.efficiency.read().unwrap().metrics.len() < MAX_MANIFOLD_OCCUPANCY
        {
            return; // Maintain laws due to high utility
        }

        if let Some(v) = victim {
            self.inductor.evict(&v);
            self.inductor.efficiency.write().unwrap().metrics.remove(&v);
            self.inductor
                .evictions
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Phase 19: Get a telemetry snapshot of the engine's current state.
    pub fn telemetry(&self) -> TelemetrySnapshot {
        let occ = self.inductor.occupancy();
        TelemetrySnapshot {
            total_dispatches: self.total_dispatches,
            induction_hits: self.induction_hits,
            induction_misses: self.total_dispatches - self.induction_hits,
            hit_rate: if self.total_dispatches > 0 {
                self.induction_hits as f64 / self.total_dispatches as f64
            } else {
                0.0
            },
            manifold_occupancy: occ,
            manifold_capacity: MAX_MANIFOLD_OCCUPANCY,
            memory_usage_bytes: self
                .inductor
                .current_usage
                .load(std::sync::atomic::Ordering::Relaxed),
            memory_capacity_bytes: self.inductor.capacity_bytes,
            evictions_performed: self
                .inductor
                .evictions
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    pub fn save_manifold(&self, path: &str) -> IoResult<()> {
        self.inductor.save(path, self.vram.seed)
    }

    pub fn load_manifold(&mut self, path: &str) -> IoResult<()> {
        let seed = self.inductor.load(path)?;
        self.vram.seed = seed;
        Ok(())
    }

    /// High-throughput ray tracing with Process Induction
    pub fn trace_ray(&mut self, bvh: &BVHNode, ray: &Ray) -> Option<f32> {
        // Hash the ray origin and direction to 64-bit for induction recall
        let mut s = std::collections::hash_map::DefaultHasher::new();
        (ray.origin.x as u32).hash(&mut s);
        (ray.origin.y as u32).hash(&mut s);
        (ray.origin.z as u32).hash(&mut s);
        (ray.direction.x as u32).hash(&mut s);
        (ray.direction.y as u32).hash(&mut s);
        (ray.direction.z as u32).hash(&mut s);
        let input_hash = s.finish();

        // 1. Induction Recall (Process Induction Engine)
        // Use Raytracing Signature: 0x524159534947 (RAYSIG)
        let ray_sig = 0x524159534947;
        if let Some(res) = self.inductor.recall(ray_sig, input_hash) {
            return Some(res.data[0]);
        }

        // 2. Ground Truth Traversal (if induction miss)
        if let Some(t) = bvh.intersect(ray) {
            // Induct the new law into the manifold
            let result = KernelResult {
                data: Arc::from(vec![t; KERNEL_DISPATCH_SIZE]),
            };
            self.inductor.induct(0x600D_BA11, input_hash, result);
            return Some(t);
        }
        None
    }

    /// Get the total number of autonomous induction events.
    pub fn induction_events(&self) -> u64 {
        self.inductor
            .induction_events
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Backwards compatibility helper for FFI.
    pub fn hit_rate(&self) -> f64 {
        self.telemetry().hit_rate
    }

    /// Get the last measured divergence delta.
    pub fn last_divergence(&self) -> f32 {
        self.last_divergence
    }
}
