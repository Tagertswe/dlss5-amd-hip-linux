#pragma once
#include <cstddef>
#include <cstdint>
namespace dlss5 {
namespace gpu {

enum Fmt { FmtUnsupported = 0, FmtRgba8 = 1, FmtBgra8 = 2, FmtR16Unorm = 3, FmtRgba16F = 4 };

Fmt format_from_dxgi(unsigned format);
// Win32 D3D12 shared handle -> dma-buf fd -> hipImportExternalMemory; cached for the
// process lifetime. Returns false when no fd could be obtained or the import failed.
bool map_device(void* handle, size_t size, void** dev);
// Device kernel launches on the default stream (bit-exact CPU pipeline replicas).
void launch_px_to_rgba(const uint8_t* dev_in, float* out, int w, int h, unsigned dxgi_format);
void launch_rgb_to_pix(const float* rgb, const uint8_t* alpha_src, uint8_t* out, int w, int h,
                       unsigned dxgi_format);
// Whole-frame guards mirroring the CPU pre-passes; true when the frame must be
// bypassed (non-finite input/output or non-representable half rounding).
bool guard_result_f32(const float* dev, size_t n, bool need_half_repr);
bool input16_bad(const uint8_t* dev, size_t n16);
}  // namespace gpu
}  // namespace dlss5
