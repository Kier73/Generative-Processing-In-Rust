// -*- mode: cuda -*-
#ifndef VGPU_BRIDGE_CUH
#define VGPU_BRIDGE_CUH

#ifndef __CUDACC__
#define __device__
#define __constant__
#define __global__
#endif

#include "vbridge_types.h"

#ifndef VGPU_MANIFOLD_DEFINED
// --- Global Manifold Access ---
// mapped via cudaHostRegister or cudaMemcpy of the Rust manifold
extern __constant__ vGPU_LawEntry *g_vGPU_Manifold;
extern __constant__ uint32_t g_vGPU_Capacity;
#endif

/**
 * vGPU_Recall
 *
 * Performs an O(1) recall from the manifold inside a CUDA kernel.
 */
__device__ inline bool vGPU_Recall(uint64_t sig, uint64_t hash,
                                   float *out_data) {
  // 1. Feistel Indexing (Hardware-friendly version)
  uint32_t key = 0x45D9F3B;
  uint32_t l = (uint32_t)(hash >> 32);
  uint32_t r = (uint32_t)hash;

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t f = ((r ^ key) * 0x45D9F3B);
    f = (f >> 16) ^ f;
    uint32_t next_r = l ^ f;
    l = r;
    r = next_r;
  }

  uint32_t idx = r & (g_vGPU_Capacity - 1);
  vGPU_LawEntry entry = g_vGPU_Manifold[idx];

  // 2. Signature Validation
  if (entry.signature == sig && entry.input_hash == hash) {
// Inductive Hit: Shunt the compute
#pragma unroll
    for (int i = 0; i < 1024; ++i) {
      out_data[i] = entry.data[i];
    }
    return true;
  }

  return false;
}

#endif // VGPU_BRIDGE_CUH
