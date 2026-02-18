#ifndef VGPU_BRIDGE_TYPES_H
#define VGPU_BRIDGE_TYPES_H

#include <stdint.h>

// --- Manifold Entry Structure ---
// Mirrored from Rust for bit-exact memory alignment
struct vGPU_LawEntry {
  uint64_t signature;
  uint64_t input_hash;
  float data[1024]; // Standard dispatch width
};

#endif // VGPU_BRIDGE_TYPES_H
