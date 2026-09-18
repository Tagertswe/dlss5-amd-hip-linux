// Isolated k_linear_f32 microbenchmark at real window-block shapes.
// Variant arg: 0 = current 16-row kernel, 1 = fat 32-row, 2 = fat 64-row.
// Path arg: 0 = p0 (F8IN+F8OUT+F8RES), 1 = p1 (F8IN+F8OUT_RASTER), 2 = C512 p1 (F8IN only).
#include "dlss5_runtime.hpp"
#include "../kernels/qkv_norm.hip"
#include "../kernels/ops.hip"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;

static void run_cur(const u8* in, const u8* w, float* out, u8* out8, uint m, uint n, uint k,
                    const u8* resid8, const float* residf, const float* sc, hipStream_t s, int path) {
    // p0: F8IN F8OUT F8RES mode5 matrix_residual=1;
    // p1: F8IN F8OUT_RASTER mode5 f32 residual matrix_residual=1;
    // c512p1: F8IN only mode5 f32 residual matrix_residual=0
    if (path == 0)
        k_linear_f32<true, true, false, false, false, true><<<dim3((m + 15) / 16, (n + 63) / 64), 128, 0, s>>>(
            nullptr, in, w, out, out8, m, n, k, nullptr, sc, 5, 0, false, nullptr, nullptr,
            nullptr, 0, 0, 0, 0, 0, resid8);
    else if (path == 1)
        k_linear_f32<true, false, false, false, true><<<dim3((m + 15) / 16, (n + 63) / 64), 128, 0, s>>>(
            nullptr, in, w, out, nullptr, m, n, k, residf, sc, 5, 0, true, nullptr, nullptr,
            out8, 0, 0, 0, 0, 0, nullptr);
    else
        k_linear_f32<true, false><<<dim3((m + 15) / 16, (n + 63) / 64), 128, 0, s>>>(
            nullptr, in, w, out, nullptr, m, n, k, residf, sc, 5, 0, false);
}

int main(int argc, char** argv) {
    const uint c = argc > 1 ? std::stoul(argv[1]) : 256;
    const uint m = argc > 2 ? std::stoul(argv[2]) : (c == 256 ? 8640u : c == 128 ? 34560u : c == 64 ? 138240u : 2160u);
    const int variant = argc > 3 ? std::atoi(argv[3]) : 0;
    const int path = argc > 4 ? std::atoi(argv[4]) : 1;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 20;
    const uint n = c, k = c;
    std::vector<u8> x(size_t(m) * k);
    std::vector<u8> w(size_t(n) * k);
    for (size_t i = 0; i < x.size(); ++i) x[i] = e4m3_byte(H(float(int((i * 2654435761u) % 1000) - 500) * 0.002f));
    for (size_t i = 0; i < w.size(); ++i) w[i] = e4m3_byte(float(int((i * 40503u + i / 37u) % 97) - 48) * 0.01f);
    std::vector<u8> resid(size_t(m) * n);
    std::vector<float> residf(size_t(m) * n), sc(n);
    for (size_t i = 0; i < resid.size(); ++i) {
        float v = H(float(int((i * 668265263u) % 777) - 388) * 0.004f);
        resid[i] = e4m3_byte(v);
        residf[i] = from_e4m3(resid[i]);
    }
    for (size_t i = 0; i < n; ++i) sc[i] = H(float(int(i % 13) - 6) * 0.077f);
    DeviceBuf dx(x.size()), dw(w.size()), dr(resid.size()), drf(residf.size() * 4), dsc(n * 4),
        dy(size_t(m) * n * 4), dy8(size_t(m) * n);
    dx.upload(x.data(), x.size());
    dw.upload(w.data(), w.size());
    dr.upload(resid.data(), resid.size());
    drf.upload(residf.data(), residf.size() * 4);
    dsc.upload(sc.data(), n * 4);
    hipStream_t stream;
    hip_check(hipStreamCreate(&stream), "stream");
    auto go = [&] {
        (void)variant;
        run_cur(dx.as<u8>(), dw.as<u8>(), dy.as<float>(), dy8.as<u8>(), m, n, k, dr.as<u8>(),
                drf.as<float>(), dsc.as<float>(), stream, path);
    };
    for (int i = 0; i < 5; ++i) go();
    hip_check(hipStreamSynchronize(stream), "warm");
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0, stream);
    for (int i = 0; i < iters; ++i) go();
    hipEventRecord(e1, stream);
    hip_check(hipEventSynchronize(e1), "sync");
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    ms /= iters;
    std::vector<float> out(size_t(m) * n);
    dy.download(out.data(), out.size() * 4);
    uint64_t hsh = 1469598103934665603ull;
    for (float v : out) {
        hsh ^= uint64_t(as_u32(v));
        hsh *= 1099511628211ull;
    }
    std::printf("LIN C%u m=%u path=%d variant=%d iters=%d: %.3f ms/call out_hash=%016llx\n", c, m, path,
                variant, iters, ms, (unsigned long long)hsh);
    return 0;
}
