import ctypes
import os
import sys

# Load the shared library
lib_path = "vgpu_rust/target/release/vgpu_rust.dll" # Windows extension
if not os.path.exists(lib_path):
    print(f"Error: {lib_path} not found. Build the project first.")
    sys.exit(1)

vgpu = ctypes.CDLL(lib_path)

# Define function signatures
vgpu.vgpu_new.restype = ctypes.c_void_p
vgpu.vgpu_new.argtypes = [ctypes.c_uint64]

vgpu.vgpu_free.argtypes = [ctypes.c_void_p]

vgpu.vgpu_dispatch.restype = ctypes.c_int32
vgpu.vgpu_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)]

vgpu.vgpu_hit_rate.restype = ctypes.c_double
vgpu.vgpu_hit_rate.argtypes = [ctypes.c_void_p]

# KERNEL_DISPATCH_SIZE = 1024
DISPATCH_SIZE = 1024

def main():
    print("--- vGPU Python FFI Demo ---")
    
    # 1. Initialize Context
    ctx = vgpu.vgpu_new(42)
    if not ctx:
        print("Failed to initialize vGPU context")
        return

    print(f"Initialized vGPU Context at: {hex(ctx)}")

    # 2. Dispatch a Standard Kernel (Normalize = 0x4E4F_524D_414C_495A)
    SIG_NORMALIZE = 0x4E4F_524D_414C_495A
    out_buffer = (ctypes.c_float * DISPATCH_SIZE)()
    
    # We call it 5 times to see if induction kicks in (though it needs registration first)
    # The FFI doesn't auto-register StdLib yet in the ffi_new, let's fix that or handle here.
    
    print("Executing Normalize Dispatch...")
    res = vgpu.vgpu_dispatch(ctx, SIG_NORMALIZE, 123, out_buffer)
    
    if res == 0:
        print(f"Dispatch Success! Sample [0]: {out_buffer[0]}")
    else:
        print(f"Dispatch Failed (Code: {res}). Kernel probably not registered.")

    # 3. Check Hit Rate
    hr = vgpu.vgpu_hit_rate(ctx)
    print(f"Current Induction Hit Rate: {hr:.2f}")

    # 4. Cleanup
    vgpu.vgpu_free(ctx)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
