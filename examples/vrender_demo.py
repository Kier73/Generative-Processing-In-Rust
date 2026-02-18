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
lib.vgpu_render_pixel.argtypes = [
    ctypes.c_void_p,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, # Origin
    ctypes.c_float, ctypes.c_float, ctypes.c_float, # Direction
    ctypes.POINTER(ctypes.c_float)                  # Out RGB
]

def main():
    ctx = lib.vgpu_create_context(128, 42)
    print("--- vRender: Inductive Radiance Demo ---")

    # Camera settings
    origin = (0.0, 0.0, 0.0)
    # Direction toward the sphere at (0, 0, -5)
    dir_vec = (0.0, 0.0, -1.0)
    
    out_rgb = (ctypes.c_float * 3)()

    # 1. Cold Render (Trace + Induction)
    print("Tracing Pixel (Cold)...")
    start = time.perf_counter()
    lib.vgpu_render_pixel(ctx, *origin, *dir_vec, out_rgb)
    print(f"Cold Pass: {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"RGB: [{out_rgb[0]:.2f}, {out_rgb[1]:.2f}, {out_rgb[2]:.2f}]")

    # 2. Warm Render (Recall)
    print("\nRecalling Pixel (Warm)...")
    start = time.perf_counter()
    lib.vgpu_render_pixel(ctx, *origin, *dir_vec, out_rgb)
    print(f"Warm Pass (Recall): {(time.perf_counter()-start)*1000:.3f}ms")
    print(f"RGB: [{out_rgb[0]:.2f}, {out_rgb[1]:.2f}, {out_rgb[2]:.2f}]")

    print("\nObservation: The warm pass is nearly instantaneous because the radiance at that voxel/direction was recalled.")

if __name__ == "__main__":
    main()
