#include "dlss5_runtime.hpp"
#include "dlss5_launch.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace dlss5;

static size_t compare_half(const __half* a, const __half* b, size_t n, const char* what) {
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        float x = __half2float(a[i]), y = __half2float(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y) || x != y) {
            if (bad < 20)
                std::fprintf(stderr, "%s[%zu] tok=%zu ch=%zu: ref=%f fused=%f\n", what, i, i / 32,
                             i % 32, x, y);
            ++bad;
        }
    }
    return bad;
}

int main() {
    std::mt19937 rng(0xC32F15ED);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    auto rnd = [&] { return dist(rng); };

    const uint rw = 10, rh = 10, rpx = 4, rpy = 4, rsw = 16, rsh = 16;
    const size_t tokens = rsw * rsh;
    std::vector<float> ffn_src(8736);
    for (auto& v : ffn_src) v = rnd();
    std::vector<u8> ffn_tiled(8192);
    pack_tiled_e4m3(ffn_tiled.data(), ffn_src.data() + 512, 128, 32);
    pack_tiled_e4m3(ffn_tiled.data() + 4096, ffn_src.data() + 4608, 32, 128);
    std::vector<float> ffn_s(32);
    for (auto& v : ffn_s) v = 0.25f + 0.5f * std::fabs(rnd());

    std::vector<float> att_src(4 * 32 * 32 + 4096 + 32 + 32);
    for (auto& v : att_src) v = rnd();
    std::vector<u8> qkv_tiled(3072), proj_tiled(1024);
    pack_tiled_e4m3(qkv_tiled.data(), att_src.data(), 96, 32);
    pack_tiled_e4m3(proj_tiled.data(), att_src.data() + 3072, 32, 32);
    std::vector<float> bias(4096), qs(32), proj_s(32);
    for (auto& v : bias) v = rnd() * 8.f;
    for (auto& v : qs) v = 0.25f + 0.5f * std::fabs(rnd());
    for (auto& v : proj_s) v = 0.25f + 0.5f * std::fabs(rnd());

    std::vector<u8> ffn_rw(8192), qkv_rw(3072), proj_rw(1024);
    for (uint n = 0; n < 128; ++n)
        for (uint k = 0; k < 32; ++k)
            ffn_rw[n * 32 + k] = e4m3_byte(ffn_src[512 + n * 32 + k]);
    for (uint n = 0; n < 32; ++n)
        for (uint k = 0; k < 128; ++k)
            ffn_rw[4096 + n * 128 + k] = e4m3_byte(ffn_src[4608 + n * 128 + k]);
    for (uint n = 0; n < 96; ++n)
        for (uint k = 0; k < 32; ++k)
            qkv_rw[n * 32 + k] = e4m3_byte(att_src[n * 32 + k]);
    for (uint n = 0; n < 32; ++n)
        for (uint k = 0; k < 32; ++k)
            proj_rw[n * 32 + k] = e4m3_byte(att_src[3072 + n * 32 + k]);

    std::vector<__half> in(tokens * 32);
    for (auto& h : in) h = __float2half_rn(rnd());

    DeviceBuf in_d(in.size() * 2), ffn_w(ffn_tiled.size()), ffn_s_d(ffn_s.size() * 4),
        qkv_w(qkv_tiled.size()), proj_w(proj_tiled.size()), bias_d(bias.size() * 4),
        qs_d(qs.size() * 4), proj_s_d(proj_s.size() * 4), ffn_rw_d(ffn_rw.size()),
        qkv_rw_d(qkv_rw.size()), proj_rw_d(proj_rw.size());
    in_d.upload(in.data(), in.size() * 2);
    ffn_w.upload(ffn_tiled.data(), ffn_tiled.size());
    ffn_s_d.upload(ffn_s.data(), ffn_s.size() * 4);
    qkv_w.upload(qkv_tiled.data(), qkv_tiled.size());
    proj_w.upload(proj_tiled.data(), proj_tiled.size());
    bias_d.upload(bias.data(), bias.size() * 4);
    qs_d.upload(qs.data(), qs.size() * 4);
    proj_s_d.upload(proj_s.data(), proj_s.size() * 4);
    ffn_rw_d.upload(ffn_rw.data(), ffn_rw.size());
    qkv_rw_d.upload(qkv_rw.data(), qkv_rw.size());
    proj_rw_d.upload(proj_rw.data(), proj_rw.size());

    DeviceBuf ffn_ref(tokens * 32 * 2), q8_ref(tokens * 96), att8_ref(tokens * 32),
        out_ref(tokens * 32 * 2), out_raster_ref(tokens * 32 * 2);
    DeviceBuf ffn_fus(tokens * 32 * 2), out_fus(tokens * 32 * 2), out_raster_fus(tokens * 32 * 2);

    size_t bad = 0;
    for (bool chain : {false, true}) {
        launch_ffn_f32_hin_hout(32, in_d.as<__half>(), ffn_w.as<u8>(), ffn_s_d.as<float>(),
                                ffn_ref.as<__half>(), tokens, chain, true);
        launch_qkv_norm_f32_hin(32, ffn_ref.as<__half>(), qkv_w.as<u8>(), q8_ref.as<u8>(), tokens,
                                qs_d.as<float>(), 1.f, true);
        launch_attention(32, q8_ref.as<u8>(), bias_d.as<float>(), att8_ref.as<u8>(), rsw, rsh, 1, true);
        launch_linear_f32_f8in_hrid_hout_raster(att8_ref.as<u8>(), proj_w.as<u8>(), out_ref.as<__half>(),
                                                out_raster_ref.as<__half>(), tokens, 32, 32,
                                                ffn_ref.as<__half>(), proj_s_d.as<float>(), 4, 0, false,
                                                rw, rh, rsw, rpx, rpy);
        launch_c32_fused(in_d.as<__half>(), ffn_rw_d.as<u8>(), ffn_s_d.as<float>(), qkv_rw_d.as<u8>(),
                          qs_d.as<float>(), bias_d.as<float>(), proj_rw_d.as<u8>(), proj_s_d.as<float>(),
                         ffn_fus.as<__half>(), out_fus.as<__half>(), out_raster_fus.as<__half>(),
                         rw, rh, rsw, rsh, rpx, rpy, chain, true);

        std::vector<__half> a(tokens * 32), b(tokens * 32);
        ffn_ref.download(a.data(), a.size() * 2);
        ffn_fus.download(b.data(), b.size() * 2);
        bad += compare_half(a.data(), b.data(), a.size(), chain ? "ffn chain" : "ffn");
        out_ref.download(a.data(), a.size() * 2);
        out_fus.download(b.data(), b.size() * 2);
        bad += compare_half(a.data(), b.data(), a.size(), chain ? "out chain" : "out");
        out_raster_ref.download(a.data(), a.size() * 2);
        out_raster_fus.download(b.data(), b.size() * 2);
        bad += compare_half(a.data(), b.data(), a.size(), chain ? "raster chain" : "raster");
    }
    if (bad)
        std::exit(1);
    std::printf("test_c32_fused PASS\n");
    return 0;
}
