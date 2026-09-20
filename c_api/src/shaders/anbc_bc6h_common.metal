// BC6H (UF16, mode 11) encoder -- refinement and packing (prepended to the
// kernel, see CMakeLists.txt). Needs anbc_common.metal first.
//
// Mirrors src/bc6h_codec.py (encode_bc6h_blocks). There is no network for
// BC6H: like BC5, the block's own bounding box (per-channel min/max) plus
// the least-squares refinement reaches the same quality as the mode-11 MLP
// that was trained for it (45.7 dB vs 45.7 dB over 20 HDR validation images
// after 2 refinement rounds; the net only helped *before* refinement,
// 43.7 vs 42.6 dB). The training mode is kept in Python for experiments.
//
// BC6H interpolates in the half-float bit pattern as an integer (unsigned:
// 0..0x7BFF), so every pixel here is x = halfbits(v) / 0x7BFF in [0,1]
// (negatives clamp to 0, inf/NaN to 0x7BFF) and the BC7 refinement
// machinery applies unchanged:
//   1. endpoints start at the block's per-channel min / max
//   2. refine: exact nearest-palette indices, least-squares endpoint refit,
//      snap endpoints to 10 bits; repeat P.refineIters times
//   3. bit-pack mode 11
// Mode 11 layout (LSB-first): mode 00011 (5 bits), e0.r e0.g e0.b e1.r e1.g
// e1.b at 10 bits each, then 16 indices (anchor 3 bits, the rest 4).
// Decoder (D3D11.3 spec):
//   quantize   q   = (x << 10) / 0x7C00
//   unquantize unq = 0 if q == 0, 0xFFFF if q == 1023, else (q << 16 | 0x8000) >> 10
//   palette    v   = (unq0 * (64 - w) + unq1 * w + 32) >> 6, then halfbits = (v * 31) >> 6

#ifndef ANBC_BC6H_COMMON_METAL
#define ANBC_BC6H_COMMON_METAL

#include "anbc_common.metal"

#define ANBC_HALF_MAX 31743.0f // 0x7BFF

constant float kWeights4Bc6h[16] = { 0.0f, 4.0f / 64, 9.0f / 64, 13.0f / 64, 17.0f / 64, 21.0f / 64, 26.0f / 64, 30.0f / 64,
                                     34.0f / 64, 38.0f / 64, 43.0f / 64, 47.0f / 64, 51.0f / 64, 55.0f / 64, 60.0f / 64, 1.0f };

// ---------------------------------------------------------------------------
// Domain conversion
// ---------------------------------------------------------------------------

static inline float halfIntNorm(float v)
{
    ushort bits = as_type<ushort>(half(v)); // > 65504 -> inf (0x7C00), clamped below
    if (bits & 0x8000u)
        bits = 0; // unsigned format: negatives clamp to 0
    return float(min(bits, ushort(0x7BFF))) / ANBC_HALF_MAX;
}

// ---------------------------------------------------------------------------
// Endpoint quantization (what the decoder will actually see)
// ---------------------------------------------------------------------------

struct Quant11 { uint3 q; float3 dec; };

static inline uint quant10(float v)
{
    const uint x = uint(clamp(round(v * ANBC_HALF_MAX), 0.0f, ANBC_HALF_MAX));
    return (x << 10) / 0x7C00u;
}

static inline float dec10(uint q)
{
    uint unq = ((q << 16) + 0x8000u) >> 10;
    if (q == 0) unq = 0;
    else if (q == 1023) unq = 0xFFFFu;
    return float((unq * 31u) >> 6) / ANBC_HALF_MAX;
}

static Quant11 quantBc6h(float3 e)
{
    Quant11 r;
    r.q = uint3(quant10(e.r), quant10(e.g), quant10(e.b));
    r.dec = float3(dec10(r.q.x), dec10(r.q.y), dec10(r.q.z));
    return r;
}

static inline uint nearestIndex4Bc6h(float t)
{
    uint best = 0;
    float bestD = 2.0f;
    for (uint k = 0; k < 16; k++) {
        const float d = abs(t - kWeights4Bc6h[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Refinement + packing
// ---------------------------------------------------------------------------

struct Mode11Result { Quant11 q0, q1; uint idx[16]; };

static Mode11Result refineMode11(thread const float3* px, float3 e0, float3 e1, uint iters)
{
    Mode11Result r;
    r.q0 = quantBc6h(e0);
    r.q1 = quantBc6h(e1);
    float w[16];
    for (uint it = 0; it < iters; it++) {
        for (uint i = 0; i < 16; i++)
            w[i] = kWeights4Bc6h[nearestIndex4Bc6h(projectT(px[i], r.q0.dec, r.q1.dec))];
        lsEndpoints(px, w, e0, e1);
        r.q0 = quantBc6h(e0);
        r.q1 = quantBc6h(e1);
    }
    for (uint i = 0; i < 16; i++)
        r.idx[i] = nearestIndex4Bc6h(projectT(px[i], r.q0.dec, r.q1.dec));
    return r;
}

static uint4 packMode11(Mode11Result r)
{
    // Anchor index only stores 3 bits: swap endpoints and invert if needed.
    if (r.idx[0] > 7) {
        const Quant11 t = r.q0; r.q0 = r.q1; r.q1 = t;
        for (uint i = 0; i < 16; i++)
            r.idx[i] = 15 - r.idx[i];
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, 3u, 5);                 // mode 11 = 00011
    for (uint c = 0; c < 3; c++)     // e0.r e0.g e0.b
        bwPut(b, r.q0.q[c], 10);
    for (uint c = 0; c < 3; c++)     // e1.r e1.g e1.b
        bwPut(b, r.q1.q[c], 10);
    bwPut(b, r.idx[0], 3);
    for (uint i = 1; i < 16; i++)
        bwPut(b, r.idx[i], 4);
    return uint4(b.words[0], b.words[1], b.words[2], b.words[3]);
}

// A 4x4 RGBA block (linear float) -> packed BC6H mode-11 block: half-int
// domain, per-channel min/max start, refine, pack.
static uint4 encodeBlockBc6h(thread const float4* px, constant EncodeParams& P)
{
    float3 v[16];
    float3 lo = float3(1e30f), hi = float3(-1e30f);
    for (uint i = 0; i < 16; i++) {
        v[i] = float3(halfIntNorm(px[i].r), halfIntNorm(px[i].g), halfIntNorm(px[i].b));
        lo = min(lo, v[i]);
        hi = max(hi, v[i]);
    }
    return packMode11(refineMode11(v, lo, hi, P.refineIters));
}

#endif // ANBC_BC6H_COMMON_METAL
