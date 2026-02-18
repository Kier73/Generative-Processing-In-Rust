#ifndef VGPU_H
#define VGPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque context handle
typedef struct VGpuContext VGpuContext;

// Kernel configuration constants
#define KERNEL_DISPATCH_SIZE 1024

// Standard Library Signatures (Locked)
#define SIG_GEMM 0x654D5F47454D4D00ULL
#define SIG_CROSS_PRODUCT 0x43524F53535F5052ULL
#define SIG_NORMALIZE 0x4E4F524D414C495AULL
#define SIG_VARIETY_NOISE 0x564E4F49534500ULL

// --- Lifecycle ---
VGpuContext *vgpu_new(uint64_t seed);
VGpuContext *vgpu_create_context(uint32_t sm_count, uint64_t seed);
void vgpu_free(VGpuContext *ctx);
void vgpu_destroy_context(VGpuContext *ctx);

// --- Perseverance & IO ---
// Returns 0 on success, -1 on failure.
int32_t vgpu_save_manifold(const VGpuContext *ctx, const char *path);
int32_t vgpu_load_manifold(VGpuContext *ctx, const char *path);

// --- Execution ---
// Returns 0 on success, -1 on failure.
int32_t vgpu_dispatch(VGpuContext *ctx, uint64_t sig, uint64_t hash,
                      float *out_data);

// Direct recall (O(1) skip). Returns 1 on hit, 0 on miss.
int32_t vgpu_recall(const VGpuContext *ctx, uint64_t hash, float *out_data);

// Manual induction of a known result.
void vgpu_induct(VGpuContext *ctx, uint64_t sig, uint64_t hash,
                 const float *data, size_t len);

// --- Registration ---
// ops: pointer to an array of SpvOp codes (internal enum)
int32_t vgpu_register_kernel(VGpuContext *ctx, uint64_t sig, const void *ops,
                             size_t len);

// Compile a sequence of OP codes into a kernel signature.
uint64_t vgpu_compile_shader(VGpuContext *ctx, const uint32_t *ops,
                             size_t count);

// --- Specialized Bridges ---
void vgpu_trinity_solve_bulk(const VGpuContext *ctx, uint64_t rows,
                             uint64_t cols, float *buffer, size_t count,
                             const uint64_t *event_sig_ptr);
void vgpu_phy_step(VGpuContext *ctx, float *pos_a, float *vel_a, float *pos_b,
                   float *vel_b, float dt);

// Advanced Compute (Inductive)
void vgpu_gemm(VGpuContext *ctx, const float *a, const float *b, float *c,
               size_t m, size_t k, size_t n);
void vgpu_sort(VGpuContext *ctx, float *data, size_t len);

// --- Shaders & Structural Identity ---
typedef struct VirtualShader VirtualShader;

VirtualShader *vgpu_shader_new(void);
void vgpu_shader_free(VirtualShader *vs);
void vgpu_shader_push_vdot(VirtualShader *vs, uint8_t dst, uint8_t src_a,
                           uint8_t src_b);
void vgpu_shader_push_vnormalize(VirtualShader *vs, uint8_t dst, uint8_t src);
void vgpu_shader_push_fadd(VirtualShader *vs);
uint64_t vgpu_shader_get_semantic_signature(const VirtualShader *vs);

// --- Telemetry ---
double vgpu_hit_rate(VGpuContext *ctx);
uint64_t vgpu_induction_events(VGpuContext *ctx);
float vgpu_last_divergence(VGpuContext *ctx);

#ifdef __cplusplus
}
#endif

#endif // VGPU_H
