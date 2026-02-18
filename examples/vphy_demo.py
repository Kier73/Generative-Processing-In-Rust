import ctypes
import numpy as np
import time
import os

# Load the library
lib_path = "./vgpu_rust/target/release/vgpu_rust.dll"
if not os.path.exists(lib_path):
    lib_path = "./vgpu_rust/target/debug/vgpu_rust.dll"

lib = ctypes.CDLL(lib_path)

lib.vgpu_create_context.restype = ctypes.c_void_p
lib.vgpu_phy_step.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float
]

def main():
    ctx = lib.vgpu_create_context(128, 42)
    print("--- vPhy: Inductive Stability Demo ---")

    # Body A (Ground, static-like)
    pos_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    vel_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Body B (Falling/Resting on A)
    pos_b = np.array([0.0, 0.9, 0.0], dtype=np.float32) # Intersecting (min_dist=1.0)
    vel_b = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    dt = 0.01

    print(f"Initial State: B at {pos_b[1]:.2f}, vel {vel_b[1]:.2f}")

    # 1. Cold Step (Induction)
    start = time.perf_counter()
    lib.vgpu_phy_step(
        ctx,
        pos_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vel_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pos_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vel_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dt
    )
    print(f"Step 1 (Cold/Solve): {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"Post Step 1: B at {pos_b[1]:.2f}, vel {vel_b[1]:.2f}")

    # 2. Warm Step (Recall)
    # Reset state to exact same "situation"
    pos_b[1] = 0.9
    vel_b[1] = -1.0
    
    start = time.perf_counter()
    lib.vgpu_phy_step(
        ctx,
        pos_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vel_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pos_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vel_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dt
    )
    print(f"Step 2 (Warm/Recall): {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"Post Step 2: B at {pos_b[1]:.2f}, vel {vel_b[1]:.2f}")

    print("\nObservation: The time for Step 2 is significantly lower because the 'collision response' was recalled from the Manifold.")

if __name__ == "__main__":
    main()
