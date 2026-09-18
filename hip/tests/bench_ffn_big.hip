// Isolated k_ffn_f32 microbenchmark for C64/C128/C256 at real stage sizes,
// covering the F32/F8IN/HIN input paths used by the split network. A/B-able by
// GROUPS template argument; reports ms/call and an output hash so variants can
// be checked for bit-identity without the full graph suite.
#include "dlss5_runtime.hpp"
#include "../kernels/qkv_norm.hip"
#include "../kernels/ops.hip"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace dlss5;

// F8IN: E4M3 byte input; HIN: f16 input (HIN implies F8IN=false in the kernel).
template <int C, bool F8IN, bool HIN, int GROUPS>
static void go(const float* in, const u8* in8, const __half* inh, const u8* w, const float* scales,
               float* out, u8* out8, uint tokens, hipStream_t s) {
    uint block = 32u * (C / 16) * GROUPS;
    uint grid = (tokens + 16u * GROUPS - 1u) / (16u * GROUPS);
    if constexpr (HIN)
        k_ffn_f32<C, true, GROUPS, false, false, true><<<grid, block, 0, s>>>(
            nullptr, w, scales, out, out8, tokens, false, false, nullptr, nullptr, inh);
    else if constexpr (F8IN)
        k_ffn_f32<C, true, GROUPS, false, true, false><<<grid, block, 0, s>>>(
            nullptr, w, scales, out, out8, tokens, false, false, nullptr, in8, nullptr);
    else
        k_ffn_f32<C, true, GROUPS, false, false, false><<<grid, block, 0, s>>>(
            in, w, scales, out, out8, tokens);
}

template <int C, bool F8IN, bool HIN>
static void dispatch(int g, const float* in, const u8* in8, const __half* inh, const u8* w,
                     const float* scales, float* out, u8* out8, uint tokens, hipStream_t s) {
    switch (g) {
    case 1: go<C, F8IN, HIN, 1>(in, in8, inh, w, scales, out, out8, tokens, s); break;
    case 2: go<C, F8IN, HIN, 2>(in, in8, inh, w, scales, out, out8, tokens, s); break;
    case 4: go<C, F8IN, HIN, 4>(in, in8, inh, w, scales, out, out8, tokens, s); break;
    default:
        if constexpr (C == 64)
            go<C, F8IN, HIN, 8>(in, in8, inh, w, scales, out, out8, tokens, s);
        else {
            std::printf("bad groups\n");
            std::exit(2);
        }
    }
}

int main(int argc, char** argv) {
    const uint c = argc > 1 ? std::stoul(argv[1]) : 256;
    const uint tokens = argc > 2 ? std::stoul(argv[2]) : (c == 256 ? 8640u : c == 128 ? 34560u : 138240u);
    const int iters = argc > 3 ? std::atoi(argv[3]) : 20;
    const int groups_variant = argc > 4 ? std::atoi(argv[4]) : 1;
    const int inpath = argc > 5 ? std::atoi(argv[5]) : 1; // 0=f32 1=f8in 2=hin
    std::vector<float> x(size_t(tokens) * c);
    for (size_t i = 0; i < x.size(); ++i) x[i] = H(float(int((i * 2654435761u) % 1000) - 500) * 0.002f);
    std::vector<u8> x8(x.size());
    std::vector<__half> xh(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x8[i] = e4m3_byte(x[i]);
        xh[i] = __float2half_rn(x[i]);
    }
    std::vector<float> w(9ull * c * c);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int((i * 40503u + i / 37u) % 97) - 48) * 0.01f;
    std::vector<float> sc(c);
    for (uint i = 0; i < c; ++i) sc[i] = H(float(int(i % 13) - 6) * 0.077f);
    std::vector<u8> tiled(w.size());
    pack_tiled_e4m3(tiled.data(), w.data(), 4 * c, c);
    pack_tiled_e4m3(tiled.data() + 4 * c * c, w.data() + 4 * c * c, c, 4 * c);
    pack_tiled_e4m3(tiled.data() + 8 * c * c, w.data() + 8 * c * c, c, c);
    DeviceBuf dx(x.size() * 4), dx8(x8.size()), dxh(xh.size() * 2), dw(tiled.size()), ds(sc.size() * 4),
        dy(size_t(tokens) * c * 4), dy8(size_t(tokens) * c);
    dx.upload(x.data(), x.size() * 4);
    dx8.upload(x8.data(), x8.size());
    dxh.upload(xh.data(), xh.size() * 2);
    dw.upload(tiled.data(), tiled.size());
    ds.upload(sc.data(), sc.size() * 4);
    hipStream_t stream;
    hip_check(hipStreamCreate(&stream), "stream");
    auto go = [&] {
        if (c == 64) {
            if (inpath == 0) dispatch<64, false, false>(groups_variant, dx.as<float>(), nullptr, nullptr, dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
            else if (inpath == 1) dispatch<64, true, false>(groups_variant, nullptr, dx8.as<u8>(), nullptr, dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
            else dispatch<64, false, true>(groups_variant, nullptr, nullptr, dxh.as<__half>(), dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
        } else if (c == 128) {
            if (inpath == 0) dispatch<128, false, false>(groups_variant, dx.as<float>(), nullptr, nullptr, dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
            else if (inpath == 1) dispatch<128, true, false>(groups_variant, nullptr, dx8.as<u8>(), nullptr, dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
            else dispatch<128, false, true>(groups_variant, nullptr, nullptr, dxh.as<__half>(), dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
        } else {
            if (inpath == 0) dispatch<256, false, false>(groups_variant, dx.as<float>(), nullptr, nullptr, dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
            else if (inpath == 1) dispatch<256, true, false>(groups_variant, nullptr, dx8.as<u8>(), nullptr, dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
            else dispatch<256, false, true>(groups_variant, nullptr, nullptr, dxh.as<__half>(), dw.as<u8>(), ds.as<float>(), dy.as<float>(), dy8.as<u8>(), tokens, stream);
        }
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
    std::vector<float> out(size_t(tokens) * c);
    dy.download(out.data(), out.size() * 4);
    uint64_t hsh = 1469598103934665603ull;
    for (float v : out) {
        hsh ^= uint64_t(as_u32(v));
        hsh *= 1099511628211ull;
    }
    std::printf("FFN C%u tokens=%u groups=%d inpath=%d iters=%d: %.3f ms/call out_hash=%016llx\n", c,
                tokens, groups_variant, inpath, iters, ms, (unsigned long long)hsh);
    return 0;
}
