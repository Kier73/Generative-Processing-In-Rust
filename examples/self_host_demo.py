import ctypes
import os
import time

# Load vGPU from professional path
lib_path = "./vgpu_rust/target/release/vgpu_rust.dll"
if not os.path.exists(lib_path):
    lib_path = "./vgpu_rust/target/debug/vgpu_rust.dll"

lib = ctypes.CDLL(lib_path)
lib.vgpu_create_context.restype = ctypes.c_void_p
lib.vgpu_compile_shader.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
lib.vgpu_compile_shader.restype = ctypes.c_uint64
lib.vgpu_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)]

def main():
    print("--- vGPU: Self-Hosting & Recursive Induction Demo ---")
    ctx = lib.vgpu_create_context(128, 0x5E44CACE)
    
    # Define a 'Self-Hosting' Shader: 
    # This shader simulates the Feistel variety math used internally.
    # OpCodes (from lib.rs): 
    # 0 = FAdd, 1 = FSub, 2 = FMul, 3 = FDiv
    # We'll use a complex chain of math to simulate 'heavy' variety logic.
    ops = (ctypes.c_uint32 * 20)(*[
        2, 2, 2, 2, 2, # x * 1.1 (5 times)
        0, 0, 0, 0, 0, # x + 1.0 (5 times)
        2, 2, 2, 2, 2, # x * 1.1 (5 times)
        3, 3, 3, 3, 3  # x / 1.1 (5 times)
    ])
    
    print("Compiling 'Self-Variety' Kernel (JIT)...")
    sig = lib.vgpu_compile_shader(ctx, ops, len(ops))
    
    input_hash = 99999
    out_data = (ctypes.c_float * 1024)()

    # 1. Cold Pass (JIT Execution)
    print("\nPass 1 (Cold - JIT Execution):")
    start = time.perf_counter()
    lib.vgpu_dispatch(ctx, sig, input_hash, out_data)
    print(f"Time: {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"Result[0]: {out_data[0]:.4f}")

    # 2. Warm Pass (Recall Induction)
    # The first pass 'Inducts' the law. The second pass should 'Recall'.
    # Note: Induction requires 5 samples by default for stability.
    print("\nStabilizing Law (Inducting 5 samples)...")
    for _ in range(5):
        lib.vgpu_dispatch(ctx, sig, input_hash, out_data)

    print("\nPass 7 (Warm - Manifold Recall):")
    start = time.perf_counter()
    lib.vgpu_dispatch(ctx, sig, input_hash, out_data)
    print(f"Time: {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"Result[0]: {out_data[0]:.4f}")

    print("\nObservation: The vGPU has 'Self-Hosted' its own logic. It now remembers the output of its math,")
    print("effectively bypassing its own JIT compiler for recurring patterns.")

if __name__ == "__main__":
    main()
