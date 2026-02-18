use crate::{KERNEL_DISPATCH_SIZE, KernelResult, SpvOp, VGpuContext, VirtualShader, trinity, vphy};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

/// Create a new vGPU Context and return as an opaque pointer.
/// Must be freed with vgpu_free or vgpu_destroy_context.
#[unsafe(no_mangle)]
pub extern "C" fn vgpu_new(seed: u64) -> *mut VGpuContext {
    let mut ctx = VGpuContext::new(8, seed); // Default to 8 SMs
    crate::stdlib::StandardLibrary::register_all(&mut ctx);
    Box::into_raw(Box::new(ctx))
}

/// Alias for vgpu_new to match legacy scripts.
#[unsafe(no_mangle)]
pub extern "C" fn vgpu_create_context(sm_count: u32, seed: u64) -> *mut VGpuContext {
    let mut ctx = VGpuContext::new(sm_count, seed);
    crate::stdlib::StandardLibrary::register_all(&mut ctx);
    Box::into_raw(Box::new(ctx))
}

/// Free a vGPU Context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_free(ctx: *mut VGpuContext) {
    if !ctx.is_null() {
        let _ = unsafe { Box::from_raw(ctx) };
    }
}

/// Alias for vgpu_free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_destroy_context(ctx: *mut VGpuContext) {
    unsafe { vgpu_free(ctx) };
}

/// Execute a kernel by its signature.
/// out_ptr must point to a float buffer of size KERNEL_DISPATCH_SIZE.
/// Returns 0 on success (Standard C practice).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_dispatch(
    ctx: *mut VGpuContext,
    sig: u64,
    hash: u64,
    out_ptr: *mut f32,
) -> i32 {
    if ctx.is_null() || out_ptr.is_null() {
        return -1;
    }

    let ctx_ref = unsafe { &mut *ctx };
    let res = ctx_ref.dispatch(sig, hash);

    // Copy result data to output pointer
    unsafe {
        ptr::copy_nonoverlapping(res.data.as_ptr(), out_ptr, KERNEL_DISPATCH_SIZE);
    }

    0
}

/// Perform a direct manifold recall (O(1) skip).
/// out_ptr must point to a float buffer of size KERNEL_DISPATCH_SIZE.
/// Returns 1 on hit, 0 on miss.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_recall(ctx: *const VGpuContext, hash: u64, out_ptr: *mut f32) -> i32 {
    if ctx.is_null() || out_ptr.is_null() {
        return 0;
    }
    let ctx_ref = unsafe { &*ctx };
    if let Some(res) = ctx_ref.inductor.recall(0, hash) {
        unsafe {
            ptr::copy_nonoverlapping(res.data.as_ptr(), out_ptr, KERNEL_DISPATCH_SIZE);
        }
        return 1;
    }
    0
}

/// Manually induct a known result into the manifold.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_induct(
    ctx: *mut VGpuContext,
    sig: u64,
    hash: u64,
    data: *const f32,
    _len: usize,
) {
    if ctx.is_null() || data.is_null() {
        return;
    }
    let ctx_ref = unsafe { &mut *ctx };
    let slice = unsafe { std::slice::from_raw_parts(data, KERNEL_DISPATCH_SIZE) };
    ctx_ref.inductor.induct(
        sig,
        hash,
        KernelResult {
            data: Arc::from(slice.to_vec()),
        },
    );
}

/// Save the process manifold to a file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_save_manifold(ctx: *const VGpuContext, path: *const c_char) -> i32 {
    if ctx.is_null() || path.is_null() {
        return -1;
    }
    let ctx_ref = unsafe { &*ctx };
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match ctx_ref.save_manifold(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Load a process manifold from a file.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_load_manifold(ctx: *mut VGpuContext, path: *const c_char) -> i32 {
    if ctx.is_null() || path.is_null() {
        return -1;
    }
    let ctx_ref = unsafe { &mut *ctx };
    let c_str = unsafe { CStr::from_ptr(path) };
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    match ctx_ref.load_manifold(path_str) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

/// Inductive Matrix Multiplication (GEMM)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_gemm(
    ctx: *mut VGpuContext,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    if ctx.is_null() || a.is_null() || b.is_null() || c.is_null() {
        return;
    }

    let ctx_ref = unsafe { &mut *ctx };
    let a_slice = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, k * n) };

    let res = crate::gemm::InductiveGemm::multiply(ctx_ref, a_slice, b_slice, m, k, n);

    unsafe {
        ptr::copy_nonoverlapping(res.as_ptr(), c, m * n);
    }
}

/// Inductive Sort
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_sort(ctx: *mut VGpuContext, data: *mut f32, len: usize) {
    if ctx.is_null() || data.is_null() {
        return;
    }

    let ctx_ref = unsafe { &mut *ctx };
    let data_slice = unsafe { std::slice::from_raw_parts_mut(data, len) };
    crate::sort::InductiveSort::sort(ctx_ref, data_slice);
}

/// Register a new kernel with the context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_register_kernel(
    ctx: *mut VGpuContext,
    sig: u64,
    ops_ptr: *const SpvOp,
    len: usize,
) -> i32 {
    if ctx.is_null() || ops_ptr.is_null() {
        return -1;
    }

    let ctx_ref = unsafe { &mut *ctx };
    let ops = unsafe { std::slice::from_raw_parts(ops_ptr, len).to_vec() };

    ctx_ref.register_kernel(Box::new(crate::ShaderKernel {
        shader: VirtualShader { instructions: ops },
        signature: sig,
    }));

    0
}

/// Legacy shader compilation bridge.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_compile_shader(
    ctx: *mut VGpuContext,
    ops: *const u32,
    count: usize,
) -> u64 {
    let ctx_ref = unsafe { &mut *ctx };
    let op_slice = unsafe { std::slice::from_raw_parts(ops, count) };

    let mut vs = VirtualShader::new();
    for &op_code in op_slice {
        let op = match op_code {
            0 => SpvOp::FAdd,
            1 => SpvOp::FSub,
            2 => SpvOp::FMul,
            3 => SpvOp::FDiv,
            4 => SpvOp::VDot {
                dst: 0,
                src_a: 0,
                src_b: 4,
            },
            5 => SpvOp::VNormalize { dst: 0, src: 0 },
            6 => SpvOp::FMul, // Anomaly test uses code 6 for mul as well
            _ => SpvOp::InductionBarrier,
        };
        vs.push(op);
    }

    let sig = vs.generate_signature();
    ctx_ref.register_kernel(Box::new(crate::JitKernel {
        ops: vs.instructions,
        signature: sig,
    }));
    sig
}

/// Create a new VirtualShader object.
#[unsafe(no_mangle)]
pub extern "C" fn vgpu_shader_new() -> *mut VirtualShader {
    Box::into_raw(Box::new(VirtualShader::new()))
}

/// Free a VirtualShader object.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_shader_free(vs: *mut VirtualShader) {
    if !vs.is_null() {
        let _ = unsafe { Box::from_raw(vs) };
    }
}

/// Push a VDot instruction to a shader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_shader_push_vdot(
    vs: *mut VirtualShader,
    dst: u8,
    src_a: u8,
    src_b: u8,
) {
    if !vs.is_null() {
        let vs_ref = unsafe { &mut *vs };
        vs_ref.push(SpvOp::VDot { dst, src_a, src_b });
    }
}

/// Push a VNormalize instruction to a shader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_shader_push_vnormalize(vs: *mut VirtualShader, dst: u8, src: u8) {
    if !vs.is_null() {
        let vs_ref = unsafe { &mut *vs };
        vs_ref.push(SpvOp::VNormalize { dst, src });
    }
}

/// Push a FAdd instruction to a shader.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_shader_push_fadd(vs: *mut VirtualShader) {
    if !vs.is_null() {
        let vs_ref = unsafe { &mut *vs };
        vs_ref.push(SpvOp::FAdd);
    }
}

/// Get the Semantic Signature of a shader.
/// This signature is immune to register renaming (Alpha-renaming) and reordering.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_shader_get_semantic_signature(vs: *const VirtualShader) -> u64 {
    if vs.is_null() {
        return 0;
    }
    let vs_ref = unsafe { &*vs };
    vs_ref.generate_semantic_signature()
}

/// Physics Step Bridge.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_phy_step(
    ctx: *mut VGpuContext,
    pos_a: *mut f32,
    vel_a: *mut f32,
    pos_b: *mut f32,
    vel_b: *mut f32,
    dt: f32,
) {
    let ctx_ref = unsafe { &mut *ctx };
    let mut body_a = unsafe {
        vphy::RigidBody {
            pos: vphy::Vec3::new(*pos_a, *pos_a.add(1), *pos_a.add(2)),
            vel: vphy::Vec3::new(*vel_a, *vel_a.add(1), *vel_a.add(2)),
            inv_mass: 1.0,
            radius: 0.5,
        }
    };
    let mut body_b = unsafe {
        vphy::RigidBody {
            pos: vphy::Vec3::new(*pos_b, *pos_b.add(1), *pos_b.add(2)),
            vel: vphy::Vec3::new(*vel_b, *vel_b.add(1), *vel_b.add(2)),
            inv_mass: 1.0,
            radius: 0.5,
        }
    };

    vphy::vphy_step(ctx_ref, &mut body_a, &mut body_b, dt);

    unsafe {
        *pos_a = body_a.pos.x;
        *pos_a.add(1) = body_a.pos.y;
        *pos_a.add(2) = body_a.pos.z;
        *vel_a = body_a.vel.x;
        *vel_a.add(1) = body_a.vel.y;
        *vel_a.add(2) = body_a.vel.z;
        *pos_b = body_b.pos.x;
        *pos_b.add(1) = body_b.pos.y;
        *pos_b.add(2) = body_b.pos.z;
        *vel_b = body_b.vel.x;
        *vel_b.add(1) = body_b.vel.y;
        *vel_b.add(2) = body_b.vel.z;
    }
}

/// Trinity Bulk Solver Bridge.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_trinity_solve_bulk(
    ctx: *const VGpuContext,
    rows: u64,
    cols: u64,
    buffer: *mut f32,
    count: usize,
    event_sig_ptr: *const u64,
) {
    let ctx_ref = unsafe { &*ctx };
    let trinity = trinity::TrinityConsensus::new(ctx_ref.vram.seed);
    let buf = unsafe { std::slice::from_raw_parts_mut(buffer, count) };

    let event_sig = if event_sig_ptr.is_null() {
        None
    } else {
        Some(unsafe { *event_sig_ptr })
    };

    trinity.solve_matrix_bulk(ctx_ref, rows, cols, buf, event_sig);
}

/// Get the current induction hit rate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_hit_rate(ctx: *mut VGpuContext) -> f64 {
    if ctx.is_null() {
        return 0.0;
    }
    let ctx_ref = unsafe { &*ctx };
    ctx_ref.hit_rate()
}

/// Get the total number of induction events (autonomous promotions).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_induction_events(ctx: *mut VGpuContext) -> u64 {
    if ctx.is_null() {
        return 0;
    }
    let ctx_ref = unsafe { &*ctx };
    ctx_ref.induction_events()
}
/// Get the last measured divergence delta (quantified dissonance).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vgpu_last_divergence(ctx: *mut VGpuContext) -> f32 {
    if ctx.is_null() {
        return 0.0;
    }
    let ctx_ref = unsafe { &*ctx };
    ctx_ref.last_divergence()
}
