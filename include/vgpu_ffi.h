#ifndef VGPU_FFI_H
#define VGPU_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VGpuContext VGpuContext;
typedef struct VVramPtr VVramPtr;

#define KERNEL_DISPATCH_SIZE 1024

typedef struct {
  float data[KERNEL_DISPATCH_SIZE];
} KernelResult;

typedef struct {
  uint64_t w[4];
} VAddr;

// Context Management
VGpuContext *vgpu_create_context(uint32_t sm_count, uint64_t seed);
void vgpu_destroy_context(VGpuContext *ctx);

// Persistence
int32_t vgpu_save_manifold(const VGpuContext *ctx, const char *path);
int32_t vgpu_load_manifold(VGpuContext *ctx, const char *path);

// Shader Bridge (Phase 12-14)
uint64_t vgpu_compile_shader(VGpuContext *ctx, const uint32_t *ops,
                             size_t count);
int32_t vgpu_dispatch(VGpuContext *ctx, uint64_t sig, uint64_t input_hash,
                      float *out_data);
int32_t vgpu_recall(const VGpuContext *ctx, uint64_t input_hash,
                    float *out_data);
void vgpu_induct(VGpuContext *ctx, uint64_t sig, uint64_t input_hash,
                 const float *data, size_t len);

// Inductive GEMM (Phase 17)
void vgpu_gemm(VGpuContext *ctx, const float *a, const float *b, float *c,
               size_t m, size_t k, size_t n);

// Inductive Sort (Phase 17)
void vgpu_sort(VGpuContext *ctx, float *data, size_t len);

// VVram FFI (Phase 16)
VVramPtr *vgpu_vram_new(uint64_t seed);
void vgpu_vram_destroy(VVramPtr *vram);
float vgpu_vram_read(const VVramPtr *vram, VAddr addr);
void vgpu_vram_write(VVramPtr *vram, VAddr addr, float value);

// VPhy FFI
void vgpu_phy_step(void *ctx, float *pos_a, float *vel_a, float *pos_b,
                   float *vel_b, float dt);
void vgpu_render_pixel(void *ctx, float ox, float oy, float oz, float dx,
                       float dy, float dz, float *out_rgb);

#ifdef __cplusplus
}
#endif

#endif // VGPU_FFI_H
