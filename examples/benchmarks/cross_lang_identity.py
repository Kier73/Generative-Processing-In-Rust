import ctypes
import os

# Load the vGPU shared library
lib_path = os.path.abspath("vgpu_rust/target/debug/vgpu_rust.dll")
vgpu = ctypes.CDLL(lib_path)

# Setup argument and return types
vgpu.vgpu_shader_new.restype = ctypes.c_void_p
vgpu.vgpu_shader_free.argtypes = [ctypes.c_void_p]
vgpu.vgpu_shader_push_vdot.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8]
vgpu.vgpu_shader_push_vnormalize.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8]
vgpu.vgpu_shader_get_semantic_signature.argtypes = [ctypes.c_void_p]
vgpu.vgpu_shader_get_semantic_signature.restype = ctypes.c_uint64

def get_demo_signature(rename_regs=False):
    vs = vgpu.vgpu_shader_new()
    
    if rename_regs:
        # Program B: Renamed registers (r3, r4, r5)
        # Sequence: VNormalize THEN VDot (Reordered)
        vgpu.vgpu_shader_push_vnormalize(vs, 3, 3)
        vgpu.vgpu_shader_push_vdot(vs, 3, 4, 5)
    else:
        # Program A: Standard registers (r0, r1, r2)
        # Sequence: VDot THEN VNormalize
        vgpu.vgpu_shader_push_vdot(vs, 0, 1, 2)
        vgpu.vgpu_shader_push_vnormalize(vs, 0, 0)
    
    sig = vgpu.vgpu_shader_get_semantic_signature(vs)
    vgpu.vgpu_shader_free(vs)
    return sig

if __name__ == "__main__":
    print("\n--- vGPU Cross-Language Structural Identity (Python Side) ---")
    sig_a = get_demo_signature(rename_regs=False)
    sig_b = get_demo_signature(rename_regs=True)
    
    print(f"Python Signature (Original): {sig_a}")
    print(f"Python Signature (Renamed/Reordered): {sig_b}")
    
    if sig_a == sig_b:
        print("Success: Structural Identity preserved in Python!")
    else:
        print("Error: Signatures diverged.")
