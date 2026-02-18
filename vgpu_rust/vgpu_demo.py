import ctypes
import os

# Load the dynamic library
lib_path = os.path.join(os.getcwd(), "target", "release", "vgpu_rust.dll")
if not os.path.exists(lib_path):
    print(f"Error: Could not find {lib_path}")
    exit(1)

vgpu = ctypes.CDLL(lib_path)

# Define return types and argument types
vgpu.vgpu_create_context.restype = ctypes.c_void_p
vgpu.vgpu_create_context.argtypes = [ctypes.c_uint32, ctypes.c_uint64]

vgpu.vgpu_induct.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]

vgpu.vgpu_recall.restype = ctypes.c_int32
vgpu.vgpu_recall.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)]

vgpu.vgpu_destroy_context.argtypes = [ctypes.c_void_p]

# Use the vGPU
print("--- Python vGPU FFI Demo ---")
ctx = vgpu.vgpu_create_context(128, 0x5E44CACE)

# Prepare some data (1024 floats)
data = (ctypes.c_float * 1024)(*[0.88] * 1024)
sig = 0xDEADC0DE
input_hash = 12345

print("Inducting law from Python...")
vgpu.vgpu_induct(ctx, sig, input_hash, data, 1024)

# Recall data
out_data = (ctypes.c_float * 1024)()
print("Recalling law from Python...")
hit = vgpu.vgpu_recall(ctx, input_hash, out_data)

if hit:
    print(f"Recall Success! Value[0]: {out_data[0]:.2f}")
else:
    print("Recall Miss!")

vgpu.vgpu_destroy_context(ctx)
print("Context destroyed. Cleanup complete.")
