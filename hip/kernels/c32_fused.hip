#pragma once
#include "dlss5_linalg.hpp"

namespace dlss5 {

template <bool CHAIN, bool WINDOWED, bool WRITE_FFN, bool WRITE_OUT>
__global__ void __launch_bounds__(128) k_c32_fused(const __half* in_h, const u8* ffn_w,
                                                   const float* ffn_s, const u8* qkv_w,
                                                   const float* qscale, const float* bias,
                                                   const u8* proj_w, const float* proj_s,
                                                   __half* ffn_h, __half* out_h,
                                                   __half* out_h_raster, uint w, uint h, uint sw,
                                                   uint sh, uint px, uint py) {
    using namespace linalg;
    constexpr uint PACKED_BYTES = 3u * 64u * 36u;
    constexpr uint RAW_STRIDE = 36u;
    constexpr uint SCRATCH_BYTES = 64u * RAW_STRIDE * 4u;
    constexpr uint PROB_BYTES = 64u * 68u;
    constexpr uint LDS_BYTES = PACKED_BYTES + SCRATCH_BYTES + PROB_BYTES;
    constexpr uint HIDDEN_STRIDE = 132u;

    __shared__ u8 lds[LDS_BYTES];
    u8* packed = lds;
    u8* ffn8 = packed + 128u * 36u;
    u8* hidden = lds + PACKED_BYTES;
    float* raw = reinterpret_cast<float*>(hidden);
    __half* scores = reinterpret_cast<__half*>(hidden);
    u8* prob = lds + PACKED_BYTES + SCRATCH_BYTES;
    __half* in16 = reinterpret_cast<__half*>(prob);

    const uint window = blockIdx.x;
    const uint wave = threadIdx.x / 32u;
    const uint lane = threadIdx.x & 31u;
    const uint first = wave * 16u;

    for (uint i4 = threadIdx.x; i4 < 512u; i4 += 128u) {
        const uint t = i4 / 8u;
        const uint c4 = (i4 % 8u) * 4u;
        const uint2 hv =
            *reinterpret_cast<const uint2*>(in_h + (window * 64u + t) * 32u + c4);
        const __half2* hp = reinterpret_cast<const __half2*>(&hv);
        *reinterpret_cast<__half2*>(in16 + t * 34u + c4) = hp[0];
        *reinterpret_cast<__half2*>(in16 + t * 34u + c4 + 2u) = hp[1];
        *reinterpret_cast<u32*>(packed + t * 36u + c4) =
            u32(e4m3_byte(__half2float(__low2half(hp[0])))) |
            u32(e4m3_byte(__half2float(__high2half(hp[0])))) << 8 |
            u32(e4m3_byte(__half2float(__low2half(hp[1])))) << 16 |
            u32(e4m3_byte(__half2float(__high2half(hp[1])))) << 24;
    }
    __syncthreads();

    const uint shuffle_base = lane & ~3u;
    const uint acc_row0 = (lane >= 16u ? 8u : 0u);
    const uint acc_col0 = (lane & 15u) & ~3u;

    for (uint cb = 0; cb < 8u; ++cb) {
        const uint col = cb * 16u;
        C32 ex = C32::Splat(0.f);
        A8 a = A8::Load(packed + first * 36u, 0, 36);
        B8 b = B8::LoadRow(ffn_w, (col / 16u) * 512u, 16);
        ex.MultiplyAccumulate(a, b);
        const uint hcol = col + acc_col0;
        for (uint i = 0; i < 8u; ++i) {
            const uint b0 = e4m3_byte(ActivatePolyC32(ex.Get(i)));
            const uint b1 = __shfl(b0, shuffle_base + 1u);
            const uint b2 = __shfl(b0, shuffle_base + 2u);
            const uint b3 = __shfl(b0, shuffle_base + 3u);
            if ((lane & 3u) == 0u)
                *reinterpret_cast<u32*>(hidden + (first + acc_row0 + i) * HIDDEN_STRIDE + hcol) =
                    b0 | b1 << 8 | b2 << 16 | b3 << 24;
        }
    }
    __syncthreads();

    C32 acc0 = C32::Splat(0.f);
    C32 acc1 = C32::Splat(0.f);
    for (uint i = 0; i < 8u; ++i) {
        const uint2 rc = acc0.GetCoordinate(i);
        float iv0 = __half2float(in16[(first + rc.x) * 34u + rc.y]);
        float iv1 = __half2float(in16[(first + rc.x) * 34u + 16u + rc.y]);
        if constexpr (CHAIN) {
            iv0 = from_e4m3(e4m3_byte(iv0));
            iv1 = from_e4m3(e4m3_byte(iv1));
        }
        acc0.Set(i, iv0 * ffn_s[rc.y]);
        acc1.Set(i, iv1 * ffn_s[16u + rc.y]);
    }
    for (uint g = 0; g < 4u; ++g) {
        A8 a = A8::Load(hidden + first * HIDDEN_STRIDE, g * 32u, HIDDEN_STRIDE);
        B8 b0 = B8::LoadRow(ffn_w, 4096u + g * 512u, 16);
        B8 b1 = B8::LoadRow(ffn_w, 4096u + (4u + g) * 512u, 16);
        acc0.MultiplyAccumulate(a, b0);
        acc1.MultiplyAccumulate(a, b1);
    }
    __half saved0[8];
    __half saved1[8];
    for (uint i = 0; i < 8u; ++i) {
        const uint2 rc = acc0.GetCoordinate(i);
        saved0[i] = __float2half_rn(acc0.Get(i));
        ffn8[(first + rc.x) * 36u + rc.y] = e4m3_byte(__half2float(saved0[i]));
        if constexpr (WRITE_FFN)
            ffn_h[(window * 64u + first + rc.x) * 32u + rc.y] = saved0[i];
        const uint2 rc1 = acc1.GetCoordinate(i);
        saved1[i] = __float2half_rn(acc1.Get(i));
        ffn8[(first + rc1.x) * 36u + 16u + rc1.y] = e4m3_byte(__half2float(saved1[i]));
        if constexpr (WRITE_FFN)
            ffn_h[(window * 64u + first + rc1.x) * 32u + 16u + rc1.y] = saved1[i];
    }
    __syncthreads();

    for (uint part = 0; part < 3u; ++part) {
        C32 q0 = C32::Splat(0.f);
        C32 q1 = C32::Splat(0.f);
        A8 a = A8::Load(ffn8 + first * 36u, 0, 36);
        B8 b0 = B8::LoadRow(qkv_w, (part * 2u) * 512u, 16);
        B8 b1 = B8::LoadRow(qkv_w, (part * 2u + 1u) * 512u, 16);
        q0.MultiplyAccumulate(a, b0);
        q1.MultiplyAccumulate(a, b1);
        if (part < 2u) {
            for (uint i = 0; i < 8u; ++i) {
                const uint2 rc = q0.GetCoordinate(i);
                raw[(first + rc.x) * RAW_STRIDE + part * 32u + rc.y] = q0.Get(i);
                const uint2 rc1 = q1.GetCoordinate(i);
                raw[(first + rc1.x) * RAW_STRIDE + part * 32u + 16u + rc1.y] = q1.Get(i);
            }
            __syncthreads();
            if (lane < 16u) {
                const uint row = lane;
                const float* s0 = raw + (first + row) * RAW_STRIDE + part * 32u;
                const float* s1 = s0 + 16u;
                float sum = 0.f;
                const float4* q0f = reinterpret_cast<const float4*>(s0);
                const float4* q1f = reinterpret_cast<const float4*>(s1);
#pragma unroll
                for (uint q = 0; q < 4u; ++q) {
                    float4 v = q0f[q];
                    sum += sq_half_f32(v.x);
                    sum += sq_half_f32(v.y);
                    sum += sq_half_f32(v.z);
                    sum += sq_half_f32(v.w);
                }
#pragma unroll
                for (uint q = 0; q < 4u; ++q) {
                    float4 v = q1f[q];
                    sum += sq_half_f32(v.x);
                    sum += sq_half_f32(v.y);
                    sum += sq_half_f32(v.z);
                    sum += sq_half_f32(v.w);
                }
                float inv = rsqrtf(fmaxf(sum, 6.198883056640625e-5f));
                if (part == 0u)
                    inv *= qscale[0] * 1.f;
                u32 pk[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
                for (uint ch = 0; ch < 16u; ++ch)
                    pk[ch >> 2] |= u32(e4m3_byte(s0[ch] * inv)) << ((ch & 3u) * 8u);
#pragma unroll
                for (uint ch = 0; ch < 16u; ++ch)
                    pk[4u + (ch >> 2)] |= u32(e4m3_byte(s1[ch] * inv)) << ((ch & 3u) * 8u);
                uint* out = reinterpret_cast<uint*>(packed + (part * 64u + first + row) * 36u);
#pragma unroll
                for (uint k = 0; k < 8u; ++k)
                    out[k] = pk[k];
            }
        } else {
            __syncthreads();
            for (uint i = 0; i < 8u; ++i) {
                const uint2 rc = q0.GetCoordinate(i);
                ffn8[(first + rc.x) * 36u + rc.y] = e4m3_byte(q0.Get(i));
                const uint2 rc1 = q1.GetCoordinate(i);
                ffn8[(first + rc1.x) * 36u + 16u + rc1.y] = e4m3_byte(q1.Get(i));
            }
        }
        __syncthreads();
    }

    A8 q = A8::Load(packed + first * 36u, 0, 36);
    for (uint k0 = 0; k0 < 64u; k0 += 16u) {
        B8 k = B8::LoadCol(packed, (64u + k0) * 36u, 36u);
        C32 s = Multiply(q, k);
        for (uint i = 0; i < 8u; ++i) {
            const uint2 rc = s.GetCoordinate(i);
            const uint qi = first + rc.x;
            const uint kj = k0 + rc.y;
            scores[qi * 68u + kj] =
                __float2half_rn(fast_exp_raw(s.Get(i) + bias[qi * 64u + kj]));
        }
    }
    __syncthreads();

    if (lane < 16u) {
        const uint qi = first + lane;
        float sum = 0.f;
#pragma unroll
        for (uint j = 0; j < 16u; ++j) {
            uint2 v = *reinterpret_cast<const uint2*>(&scores[qi * 68u + j * 4u]);
            const __half2* hp = reinterpret_cast<const __half2*>(&v);
            sum += __half2float(__low2half(hp[0]));
            sum += __half2float(__high2half(hp[0]));
            sum += __half2float(__low2half(hp[1]));
            sum += __half2float(__high2half(hp[1]));
        }
        const float inv = 1.f / fmaxf(sum, 1e-20f);
        uint* prow = reinterpret_cast<uint*>(prob + qi * 68u);
#pragma unroll
        for (uint q4 = 0; q4 < 16u; ++q4) {
            u32 packed_bytes = 0;
#pragma unroll
            for (uint t = 0; t < 4u; ++t)
                packed_bytes |= u32(e4m3_byte(__half2float(scores[qi * 68u + q4 * 4u + t]) * inv))
                                 << (t * 8u);
            prow[q4] = packed_bytes;
        }
    }
    __syncthreads();

    C32 av0 = C32::Splat(0.f);
    C32 av1 = C32::Splat(0.f);
    for (uint k0 = 0; k0 < 64u; k0 += 32u) {
        A8 p = A8::Load(prob + first * 68u, k0, 68);
        B8 v0 = B8::LoadRow(packed + (128u + k0) * 36u, 0, 36);
        B8 v1 = B8::LoadRow(packed + (128u + k0) * 36u, 16u, 36);
        av0.MultiplyAccumulate(p, v0);
        av1.MultiplyAccumulate(p, v1);
    }
    for (uint i = 0; i < 8u; ++i) {
        const uint2 rc = av0.GetCoordinate(i);
        packed[(first + rc.x) * 36u + rc.y] = e4m3_byte(av0.Get(i));
        const uint2 rc1 = av1.GetCoordinate(i);
        packed[(first + rc1.x) * 36u + 16u + rc1.y] = e4m3_byte(av1.Get(i));
    }
    __syncthreads();

    C32 p0 = C32::Splat(0.f);
    C32 p1 = C32::Splat(0.f);
    A8 a = A8::Load(packed + first * 36u, 0, 36);
    B8 b0 = B8::LoadRow(proj_w, 0, 16);
    B8 b1 = B8::LoadRow(proj_w, 512u, 16);
    p0.MultiplyAccumulate(a, b0);
    p1.MultiplyAccumulate(a, b1);

    for (uint i = 0; i < 8u; ++i) {
        const uint2 rc = p0.GetCoordinate(i);
        const uint t = first + rc.x;
        const float v = p0.Get(i) + __half2float(saved0[i]) * proj_s[rc.y];
        const __half hv = __float2half_rn(v);
        if constexpr (WRITE_OUT)
            out_h[(window * 64u + t) * 32u + rc.y] = hv;
        if constexpr (WINDOWED) {
            const uint x = (window % (sw / 8u)) * 8u + (t % 8u);
            const uint y = (window / (sw / 8u)) * 8u + (t / 8u);
            if (x >= px && y >= py && x < w + px && y < h + py)
                out_h_raster[((y - py) * w + (x - px)) * 32u + rc.y] = hv;
        } else {
            out_h_raster[(window * 64u + t) * 32u + rc.y] = hv;
        }
    }
    for (uint i = 0; i < 8u; ++i) {
        const uint2 rc = p1.GetCoordinate(i);
        const uint t = first + rc.x;
        const float v = p1.Get(i) + __half2float(saved1[i]) * proj_s[16u + rc.y];
        const __half hv = __float2half_rn(v);
        if constexpr (WRITE_OUT)
            out_h[(window * 64u + t) * 32u + 16u + rc.y] = hv;
        if constexpr (WINDOWED) {
            const uint x = (window % (sw / 8u)) * 8u + (t % 8u);
            const uint y = (window / (sw / 8u)) * 8u + (t / 8u);
            if (x >= px && y >= py && x < w + px && y < h + py)
                out_h_raster[((y - py) * w + (x - px)) * 32u + 16u + rc.y] = hv;
        } else {
            out_h_raster[(window * 64u + t) * 32u + 16u + rc.y] = hv;
        }
    }
}

} // namespace dlss5
