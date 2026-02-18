#ifndef VGPU_FFI_H
#define VGPU_FFI_H

#include <stddef.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the Rust vGpuContext
typedef struct vGpuContext vGpuContext;

/**
 * Creates a new vGPU context.
 * @param sm_count Number of Streaming Multiprocessors.
 * @param seed Initial seed for the generative substrate.
 */
vGpuContext *vgpu_create_context(uint32_t sm_count, uint64_t seed);

/**
 * Destroys a vGPU context and frees associated memory.
 */
void vgpu_destroy_context(vGpuContext *ctx);

/**
 * Manually inducts a law into the manifold.
 * @param data Array of 1024 floats.
 */
void vgpu_induct(vGpuContext *ctx, uint64_t sig, uint64_t input_hash,
                 const float *data, size_t len);

/**
 * Recalls a law from the manifold.
 * @param out_data Pointer to a buffer of 1024 floats.
 * @return 1 on hit, 0 on miss.
 */
int32_t vgpu_recall(const vGpuContext *ctx, uint64_t input_hash,
                    float *out_data);

/**
 * Saves the current manifold state to a .vman file.
 */
int32_t vgpu_save_manifold(const vGpuContext *ctx, const char *path);

#ifdef __cplusplus
}
#endif

#endif // VGPU_FFI_H
