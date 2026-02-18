# vGPU: Technical Integrity 

## 1. Concrete Execution Engine
The vGPU executes a **Virtual Instruction Set** (vISA) defined in `SpvOp`.
- **Location**: `vgpu_rust/src/lib.rs` -> `RegisterFile::execute_op`
- **Mechanism**: A robust match-loop that performs standard IEEE-754 floating-point operations.
- **Proof**: If you insert a `BTrap` (Conditional Branch) into a shader, the execution flow changes based on real-time data, which would be impossible in a pre-computed system.

## 2. Bayesian Dissonance Control
To ensure the address space hasn't been corrupted, the system implements an active verification layer.
- **Location**: `vgpu_rust/src/lib.rs` -> `DissonanceControl`
- **Mechanism**: The system periodically samples "Induced" results ($O(1)$) and compares them against "Ground Truth" results ($O(N)$). 
- **Proof**: If the system were hardcoded, any change to the input distributions would cause a massive spike in the `last_divergence` metric, which is exposed via FFI and verified in tests.

## 3. Semantic Identity (Grounding)
The vGPU uses content-addressable logic.
- **Location**: `vgpu_rust/src/lib.rs` -> `VirtualShader::generate_semantic_signature`
- **Mechanism**: The signature is a 64-bit HDC hash of the *structure* of the code, not the text. It is immune to register renaming (alpha-renaming) and instruction reordering.
- **Proof**: Two different code implementations of the same math will result in the same Manifold Law, proves that the system understands **intent**, not just syntax.

## 4. Machine Code Emission (JIT)
For absolute performance, bypass the Rust interpreter entirely on supported systems.
- **Location**: `vgpu_rust/src/native_jit.rs`
- **Mechanism**: A zero-dependency assembler that writes machine instructions (`MOVSS`, `ADDSS`, `MULSS`) into executable memory pages.
- **Proof**: You can use a debugger like GDB or LLDB to attach to a running vGPU and see the raw assembly code being executed at runtime.
