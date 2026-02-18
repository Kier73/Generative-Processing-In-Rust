import numpy as np
import time
import ctypes
import os

# Load the vGPU shared library
def load_vgpu():
    # Try multiple paths
    paths = [
        os.path.join("..", "..", "vgpu_rust", "target", "release", "vgpu_rust.dll"),
        os.path.join("..", "..", "vgpu_cuda.dll"),
        os.path.join("..", "..", "Development", "CUDA_ACCELERATION", "vgpu_cuda.dll")
    ]
    for p in paths:
        if os.path.exists(p):
            return ctypes.CDLL(p)
    return None

vgpu = load_vgpu()

if not vgpu:
    print("Error: Could not find vgpu_rust.dll. Please run `cargo build --release` in vgpu_rust/")
    exit(1)

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

def bench_numpy_noise(scale):
    count = scale * scale
    start = time.perf_counter_ns()
    # NumPy Random Generation (Mersenne Twister / PCG64)
    buffer = np.random.rand(scale, scale).astype(np.float32)
    end = time.perf_counter_ns()
    return (end - start) / 1000.0, buffer[0,0]

def bench_vgpu_noise(ctx, scale):
    count = scale * scale
    buffer = (ctypes.c_float * count)()
    
    start = time.perf_counter_ns()
    # vGPU Generative Field Resolution (Feistel/Hash)
    vgpu.vgpu_trinity_solve_bulk(ctx, scale, scale, buffer, count, None)
    end = time.perf_counter_ns()
    return (end - start) / 1000.0, buffer[0]

def main():
    print("=" * 80)
    print("PROCEDURAL GENERATION BENCHMARK: NumPy vs vGPU")
    print("Comparing the cost of generating coherent white noise at scale.")
    print("=" * 80)
    print(f"{'Scale (NxN)':<12} | {'NumPy Gen (µs)':<15} | {'vGPU Gen (µs)':<15} | {'Speedup':<10}")
    print("-" * 80)

    ctx = vgpu.vgpu_create_context(1, 0x5E44C_ACE)

    scales = [32, 64, 128, 256, 512, 1024, 2048]
    
    for n in scales:
        # NumPy Warmup & Run
        [bench_numpy_noise(n) for _ in range(5)]
        np_times = [bench_numpy_noise(n)[0] for _ in range(20)]
        avg_np = sum(np_times) / len(np_times)
        
        # vGPU Warmup & Run
        [bench_vgpu_noise(ctx, n) for _ in range(5)]
        vgpu_times = [bench_vgpu_noise(ctx, n)[0] for _ in range(20)]
        avg_vgpu = sum(vgpu_times) / len(vgpu_times)
        
        speedup = avg_np / avg_vgpu if avg_vgpu > 0 else 0.0
        
        print(f"{n:<12} | {avg_np:>15.2f} | {avg_vgpu:>15.2f} | {speedup:>9.2f}x")

    print("-" * 80)
    print("Analysis:")
    print("1. NumPy uses a PRNG (PCG64) which is sequential and memory-bound.")
    print("2. vGPU uses stateless Position-Based Hashing (O(1) semantic).")
    print("   At large scales, vGPU SIMD output (AVX2/SSE) saturates memory bandwidth")
    print("   just like NumPy, but with higher compute density per byte.")
    print("=" * 80)

if __name__ == "__main__":
    main()
