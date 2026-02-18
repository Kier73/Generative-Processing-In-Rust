use vgpu_rust::KERNEL_DISPATCH_SIZE;
use vgpu_rust::ffi::*;
use vgpu_rust::stdlib::SIG_NORMALIZE;

#[test]
fn test_ffi_lifecycle() {
    unsafe {
        // 1. Initialize
        let ctx = vgpu_new(1337);
        assert!(!ctx.is_null());

        // 2. Prepare output buffer
        let mut out_buffer = [0.0f32; KERNEL_DISPATCH_SIZE];

        // 3. Dispatch (Normalize kernel is registered by register_all in vgpu_new)
        let status = vgpu_dispatch(ctx, SIG_NORMALIZE, 42, out_buffer.as_mut_ptr());
        assert_eq!(status, 0); // 0 for success

        // 4. Check Telemetry via FFI
        let hit_rate = vgpu_hit_rate(ctx);
        assert!(hit_rate >= 0.0 && hit_rate <= 1.0);

        // 5. Cleanup
        vgpu_free(ctx);
    }
}
