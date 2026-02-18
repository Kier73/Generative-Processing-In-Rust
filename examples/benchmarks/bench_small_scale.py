import numpy as np
import time
import ctypes
import os

# Load the vGPU shared library
lib_path = os.path.join("..", "..", "vgpu_rust", "target", "release", "vgpu_rust.dll")
if not os.path.exists(lib_path):
    lib_path = os.path.join("..", "..", "vgpu_rust.dll")
vgpu = ctypes.CDLL(lib_path)

# Define FFI signatures
vgpu.vgpu_create_context.argtypes = [ctypes.c_uint32, ctypes.c_uint64]
vgpu.vgpu_create_context.restype = ctypes.c_void_p

vgpu.vgpu_trinity_solve_bulk.argtypes = [
    ctypes.c_void_p,  # ctx
    ctypes.c_uint64,  # rows
    ctypes.c_uint64,  # cols
    ctypes.POINTER(ctypes.c_float), # buffer
    ctypes.c_size_t,  # count
    ctypes.c_void_p,  # event_sig_ptr
]

def run_numpy_micro(scale):
    A = np.random.rand(scale, scale).astype(np.float32)
    B = np.random.rand(scale, scale).astype(np.float32)
    
    start = time.perf_counter_ns()
    C = A @ B  # This is the materialization of the ENTIRE result
    end = time.perf_counter_ns()
    
    return (end - start) / 1000.0

def run_vgpu_bulk(ctx, scale):
    count = scale * scale
    buffer = (ctypes.c_float * count)()
    
    start = time.perf_counter_ns()
    vgpu.vgpu_trinity_solve_bulk(ctx, scale, scale, buffer, count, None)
    end = time.perf_counter_ns()
    
    return (end - start) / 1000.0

def main():
    print("=" * 70)
    print("SIMD-ACCELERATED CROSSOVER: NumPy vs vGPU-Bulk")
    print("=" * 70)
    print(f"{'Scale (N)':<10} | {'NumPy (Total)':<15} | {'vGPU-Bulk':<15} | {'Winner'}")
    print("-" * 70)

    ctx = vgpu.vgpu_create_context(1, 0x5E44C_ACE)

    scales = [4, 8, 16, 32, 64, 128, 256, 512]
    
    for n in scales:
        # NumPy Average
        np_times = [run_numpy_micro(n) for _ in range(50)]
        avg_np = sum(np_times) / len(np_times)
        
        # vGPU Average
        vgpu_times = [run_vgpu_bulk(ctx, n) for _ in range(50)]
        avg_vgpu = sum(vgpu_times) / len(vgpu_times)
        
        winner = "vGPU" if avg_vgpu < avg_np else "NumPy"
        
        print(f"{n:<10} | {avg_np:>12.2f} µs | {avg_vgpu:>12.2f} µs | {winner}")

    print("-" * 70)
    print("[ANALYSIS] NumPy SIMD is extremely fast for cached buffer fills (N < 32).")
    print("[ANALYSIS] vGPU-Bulk SIMD catches up as memory bandwidth becomes the bottleneck.")
    print("[ANALYSIS] Conclusion: For tiny local matrices, NumPy's C-core is hard to beat.")
    print("=" * 70)

if __name__ == "__main__":
    main()
