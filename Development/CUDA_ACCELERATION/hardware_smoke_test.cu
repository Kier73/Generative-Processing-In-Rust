#include <cuda_runtime.h>
#include <iostream>

// 1. Include the types first
#include "vbridge_types.h"

// 2. Define the symbols (now that vGPU_LawEntry is known)
__constant__ vGPU_LawEntry *g_vGPU_Manifold;
__constant__ uint32_t g_vGPU_Capacity;
#define VGPU_MANIFOLD_DEFINED

// 3. Include the logic functions
#include "vbridge.cuh"

__device__ vGPU_LawEntry d_manifold_storage[1024];

__global__ void test_shunting_kernel(uint64_t sig, uint64_t hash,
                                     float *out_val) {
  float local_data[1024];
  if (vGPU_Recall(sig, hash, local_data)) {
    *out_val = local_data[0]; // Hit!
  } else {
    *out_val = -1.0f; // Miss!
  }
}

int main() {
  std::cout << "--- vGPU: CUDA Hardware Smoke Test (RTX 4060) ---" << std::endl;

  // 1. Setup Mock Law in Device Memory
  vGPU_LawEntry h_entry;
  h_entry.signature = 0x1234;
  h_entry.input_hash = 0x5678;
  for (int i = 0; i < 1024; ++i)
    h_entry.data[i] = 42.0f;

  vGPU_LawEntry *h_manifold = new vGPU_LawEntry[1024];
  for (int i = 0; i < 1024; ++i)
    h_manifold[i] = h_entry;

  // Get device pointer to the storage
  vGPU_LawEntry *d_ptr;
  cudaGetSymbolAddress((void **)&d_ptr, d_manifold_storage);

  // Initialize the constant pointers
  uint32_t h_cap = 1024;
  cudaMemcpyToSymbol(g_vGPU_Manifold, &d_ptr, sizeof(vGPU_LawEntry *));
  cudaMemcpyToSymbol(g_vGPU_Capacity, &h_cap, sizeof(uint32_t));

  // Copy data to storage
  cudaMemcpy(d_ptr, h_manifold, sizeof(vGPU_LawEntry) * 1024,
             cudaMemcpyHostToDevice);

  float *d_out;
  cudaMalloc(&d_out, sizeof(float));

  // 2. Launch Kernel
  std::cout << "Launching Inductive Shunting Kernel..." << std::endl;
  test_shunting_kernel<<<1, 1>>>(0x1234, 0x5678, d_out);
  cudaDeviceSynchronize();

  float h_out;
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  if (h_out == 42.0f) {
    std::cout << "Status: SUCCESS. Manifold Recall Verified on NVIDIA Silicon."
              << std::endl;
  } else {
    std::cout << "Status: FAILED. Expected 42.0f, got " << h_out << std::endl;
  }

  cudaFree(d_out);
  delete[] h_manifold;
  return 0;
}
