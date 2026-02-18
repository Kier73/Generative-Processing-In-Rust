#ifndef VGPU_HPP
#define VGPU_HPP

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Forward declaration of the Rust context
struct VGpuContext;

// Industry Standard Constants
const size_t KERNEL_DISPATCH_SIZE = 1024;

// --- FFI TYPES ---

/**
 * VAddr: 256-bit Manifold Address
 * Matches Rust: pub struct VAddr(pub [u64; 4]);
 */
struct VAddr {
  uint64_t w[4];
};

extern "C" {
// FFI Declarations (from lib.rs)
struct VGpuContext;
struct VVramPtr; // Opaque pointer to the Rust VVram substrate

VGpuContext *vgpu_create_context(uint32_t sm_count, uint64_t seed);
void vgpu_destroy_context(VGpuContext *ctx);

uint64_t vgpu_compile_shader(VGpuContext *ctx, const uint32_t *ops,
                             size_t count);

int32_t vgpu_dispatch(VGpuContext *ctx, uint64_t sig, uint64_t input_hash,
                      float *out_data);

int32_t vgpu_recall(const VGpuContext *ctx, uint64_t input_hash,
                    float *out_data);

void vgpu_induct(VGpuContext *ctx, uint64_t sig, uint64_t input_hash,
                 const float *data, size_t len);

int32_t vgpu_save_manifold(const VGpuContext *ctx, const char *path);
int32_t vgpu_load_manifold(VGpuContext *ctx, const char *path);

// Inductive GEMM (Phase 17)
void vgpu_gemm(void *ctx, float *a, float *b, float *c, size_t m, size_t n,
               size_t k);
void vgpu_sort(void *ctx, float *data, size_t n);
void vgpu_phy_step(void *ctx, float *pos_a, float *vel_a, float *pos_b,
                   float *vel_b, float dt);
void vgpu_render_pixel(void *ctx, float ox, float oy, float oz, float dx,
                       float dy, float dz, float *out_rgb);

// VVram FFI (Phase 16: Generative Substrate)
VVramPtr *vgpu_vram_new(uint64_t seed);
void vgpu_vram_destroy(VVramPtr *vram);
float vgpu_vram_read(const VVramPtr *vram, VAddr addr);
void vgpu_vram_write(VVramPtr *vram, VAddr addr, float value);

double vgpu_hit_rate(VGpuContext *ctx);
uint64_t vgpu_induction_events(VGpuContext *ctx);
float vgpu_last_divergence(VGpuContext *ctx);
}

/**
 * vResult: Industry-standard 1024-float kernel result buffer.
 */
struct vResult {
  float data[KERNEL_DISPATCH_SIZE];
};

/**
 * VVram: High-level wrapper for the Generative Memory Substrate.
 */
class VVram {
public:
  VVram(uint64_t seed = 0x5E44CACE) { ptr = vgpu_vram_new(seed); }
  ~VVram() {
    if (ptr)
      vgpu_vram_destroy(ptr);
  }

  float read(VAddr addr) const { return vgpu_vram_read(ptr, addr); }
  void write(VAddr addr, float val) { vgpu_vram_write(ptr, addr, val); }

private:
  VVramPtr *ptr = nullptr;
};

/**
 * vGPU: High-level C++ Bridge for the vGPU v2.1 Virtual Layer.
 *
 * Provides an authenticated, zero-latency interface to the
 * Rust Process Induction Engine (PIE).
 */
class vGPU {
public:
  vGPU(uint32_t sm_count = 128, uint64_t seed = 0x5E44CACE) {
    ctx = vgpu_create_context(sm_count, seed);
  }

  ~vGPU() {
    if (ctx)
      vgpu_destroy_context(ctx);
  }

  // DISPATCH: The main compute entry point (O(1) after induction)
  void dispatch(uint64_t signature, uint64_t input_hash, vResult &out) {
    vgpu_dispatch(ctx, signature, input_hash, out.data);
  }

  // COMPILE: Transform high-level ops into an authenticated signature
  uint64_t compile(const std::vector<uint32_t> &ops) {
    return vgpu_compile_shader(ctx, ops.data(), ops.size());
  }

  // IO: Manifold persistence
  bool save(const std::string &path) {
    return vgpu_save_manifold(ctx, path.c_str()) == 1;
  }

  bool load(const std::string &path) {
    return vgpu_load_manifold(ctx, path.c_str()) == 1;
  }

  // LOW-LEVEL: Direct manifold manipulation
  bool recall(uint64_t input_hash, vResult &out) {
    return vgpu_recall(ctx, input_hash, out.data) == 1;
  }

  void induct(uint64_t sig, uint64_t input_hash, const vResult &res) {
    vgpu_induct(ctx, sig, input_hash, res.data, KERNEL_DISPATCH_SIZE);
  }

  // GEMM: Inductive Matrix Multiplication
  void gemm(const float *a, const float *b, float *c, size_t m, size_t n,
            size_t k) {
    vgpu_gemm(ctx, const_cast<float *>(a), const_cast<float *>(b), c, m, n, k);
  }

  // Sort: Inductive Permutation Recall
  void sort(float *data, size_t n) { vgpu_sort(ctx, data, n); }

  void physics_step(float *pos_a, float *vel_a, float *pos_b, float *vel_b,
                    float dt) {
    vgpu_phy_step(ctx, pos_a, vel_a, pos_b, vel_b, dt);
  }

  void render_pixel(float ox, float oy, float oz, float dx, float dy, float dz,
                    float *out_rgb) {
    vgpu_render_pixel(ctx, ox, oy, oz, dx, dy, dz, out_rgb);
  }

  // TELEMETRY: Quantified system health
  double hit_rate() const { return vgpu_hit_rate(ctx); }
  uint64_t induction_events() const { return vgpu_induction_events(ctx); }
  float last_divergence() const { return vgpu_last_divergence(ctx); }

private:
  VGpuContext *ctx = nullptr;

  // Disable copying to prevent double-free of context
  vGPU(const vGPU &) = delete;
  vGPU &operator=(const vGPU &) = delete;
};

#endif // VGPU_HPP
