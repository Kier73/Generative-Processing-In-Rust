// -*- mode: cuda -*-
#ifndef __CUDACC__
#define __device__
#define __constant__
#define __global__
#endif

#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "vbridge_types.h"

// 1. Manifold Definition
__constant__ vGPU_LawEntry *g_vGPU_Manifold;
__constant__ uint32_t g_vGPU_Capacity;
#define VGPU_MANIFOLD_DEFINED

#include "vbridge.cuh"

__device__ vGPU_LawEntry d_manifold_storage[1024];

/**
 * vFLOPS Stress Kernel
 * PHI (Virtual Depth) = 10,000 instructions
 * W (Data Width) = 1,024 floats
 */
__global__ void vflops_stress_kernel(uint64_t sig, uint64_t hash,
                                     float *out_val, int iterations) {
  float local_data[1024];
  float sum = 0;

  for (int i = 0; i < iterations; ++i) {
    if (vGPU_Recall(sig, hash + i, local_data)) {
      sum += local_data[0];
    }
  }
  *out_val = sum;
}

int main() {
  // Hardware Setup
  uint32_t capacity = 1024;
  vGPU_LawEntry *h_manifold = new vGPU_LawEntry[capacity];
  uint64_t sig = 0x512353;

  for (int i = 0; i < capacity; ++i) {
    h_manifold[i].signature = sig;
    h_manifold[i].input_hash = 0x5678 + i;
    for (int j = 0; j < 1024; ++j)
      h_manifold[i].data[j] = 1.0f;
  }

  vGPU_LawEntry *d_ptr;
  cudaGetSymbolAddress((void **)&d_ptr, d_manifold_storage);
  cudaMemcpyToSymbol(g_vGPU_Manifold, &d_ptr, sizeof(vGPU_LawEntry *));
  cudaMemcpyToSymbol(g_vGPU_Capacity, &capacity, sizeof(uint32_t));
  cudaMemcpy(d_ptr, h_manifold, sizeof(vGPU_LawEntry) * capacity,
             cudaMemcpyHostToDevice);

  float *d_out;
  cudaMalloc(&d_out, sizeof(float));

  // Warmup
  vflops_stress_kernel<<<256, 256>>>(sig, 0x5678, d_out, 10);
  cudaDeviceSynchronize();

  // Benchmark
  int iterations = 10000;
  int blocks = 256;
  int threads = 256;

  auto start = std::chrono::high_resolution_clock::now();
  vflops_stress_kernel<<<blocks, threads>>>(sig, 0x5678, d_out, iterations);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  // Metrics
  std::chrono::duration<double> elapsed = end - start;
  double total_dispatches = (double)blocks * threads * iterations;
  double dispatch_rate = total_dispatches / elapsed.count();

  double phi = 10000.0;
  double width = 1024.0;
  double v_flops = (dispatch_rate * phi * width);

  // Data-Oriented Output
  std::cout << "--- vGPU Performance Metrics ---" << std::endl;
  std::cout << "Device:            RTX 4060" << std::endl;
  std::cout << "Time:              " << elapsed.count() << "s" << std::endl;
  std::cout << "Recalls/sec:       " << dispatch_rate << std::endl;
  std::cout << "vFLOPS (Projected):" << v_flops << std::endl;
  std::cout << "vFLOPS (Unit):      " << (v_flops / 1e15) << " Peta"
            << std::endl;

  cudaFree(d_out);
  delete[] h_manifold;
  return 0;
}
