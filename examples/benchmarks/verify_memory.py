import sys
import os
import time

# Attempt to find the vgpu_rust library
paths = [
    os.path.join("..", "..", "vgpu_rust", "target", "release", "vgpu_rust.dll"),
    os.path.join("..", "..", "vgpu_rust.dll")
]

print("vGPU Memory Verification Tool")
print("-" * 30)

found = False
for p in paths:
    if os.path.exists(p):
        print(f"Found substrate at: {p}")
        found = True
        break

if not found:
    print("Substrate not found. Please build the project.")
else:
    print("Ready for memory audit.")
