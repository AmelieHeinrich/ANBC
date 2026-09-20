// BC6H (UF16, mode 11) encoder -- refinement and packing (GLSL port of
// anbc_bc6h_common.metal). Needs anbc_common.glsl first.
//
// Mirrors src/bc6h_codec.py (encode_bc6h_blocks). No network: the block's
// bounding box (per-channel min/max) plus the least-squares refinement
// reaches the same quality as the mode-11 MLP that was tried.
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

#ifndef ANBC_BC6H_COMMON_GLSL
#define ANBC_BC6H_COMMON_GLSL

#include "anbc_common.glsl"

#define ANBC_HALF_MAX 31743.0 // 0x7BFF

const float kWeights4Bc6h[16] = float[16](0.0, 4.0 / 64.0, 9.0 / 64.0, 13.0 / 64.0, 17.0 / 64.0, 21.0 / 64.0, 26.0 / 64.0, 30.0 / 64.0,
                                          34.0 / 64.0, 38.0 / 64.0, 43.0 / 64.0, 47.0 / 64.0, 51.0 / 64.0, 55.0 / 64.0, 60.0 / 64.0, 1.0);

// ---------------------------------------------------------------------------
// Domain conversion
// ---------------------------------------------------------------------------

float halfIntNorm(float v)
{
    uint bits = packHalf2x16(vec2(v, 0.0)) & 0xFFFFu; // > 65504 -> inf (0x7C00), clamped below
    if ((bits & 0x8000u) != 0u)
        bits = 0u; // unsigned format: negatives clamp to 0
    return float(min(bits, 0x7BFFu)) / ANBC_HALF_MAX;
}

// ---------------------------------------------------------------------------
// Endpoint quantization (what the decoder will actually see)
// ---------------------------------------------------------------------------

struct Quant11 { uvec3 q; vec3 dec; };

uint quant10(float v)
{
    const uint x = uint(clamp(round(v * ANBC_HALF_MAX), 0.0, ANBC_HALF_MAX));
    return (x << 10) / 0x7C00u;
}

float dec10(uint q)
{
    uint unq = ((q << 16) + 0x8000u) >> 10;
    if (q == 0u) unq = 0u;
    else if (q == 1023u) unq = 0xFFFFu;
    return float((unq * 31u) >> 6) / ANBC_HALF_MAX;
}

Quant11 quantBc6h(vec3 e)
{
    Quant11 r;
    r.q = uvec3(quant10(e.r), quant10(e.g), quant10(e.b));
    r.dec = vec3(dec10(r.q.x), dec10(r.q.y), dec10(r.q.z));
    return r;
}

uint nearestIndex4Bc6h(float t)
{
    uint best = 0u;
    float bestD = 2.0;
    for (uint k = 0u; k < 16u; k++) {
        const float d = abs(t - kWeights4Bc6h[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Refinement + packing
// ---------------------------------------------------------------------------

struct Mode11Result { Quant11 q0, q1; uint idx[16]; };

Mode11Result refineMode11(in vec3 px[16], vec3 e0, vec3 e1, uint iters)
{
    Mode11Result r;
    r.q0 = quantBc6h(e0);
    r.q1 = quantBc6h(e1);
    float w[16];
    for (uint it = 0u; it < iters; it++) {
        for (uint i = 0u; i < 16u; i++)
            w[i] = kWeights4Bc6h[nearestIndex4Bc6h(projectT(px[i], r.q0.dec, r.q1.dec))];
        lsEndpoints(px, w, e0, e1);
        r.q0 = quantBc6h(e0);
        r.q1 = quantBc6h(e1);
    }
    for (uint i = 0u; i < 16u; i++)
        r.idx[i] = nearestIndex4Bc6h(projectT(px[i], r.q0.dec, r.q1.dec));
    return r;
}

uvec4 packMode11(Mode11Result r)
{
    // Anchor index only stores 3 bits: swap endpoints and invert if needed.
    if (r.idx[0] > 7u) {
        const Quant11 t = r.q0; r.q0 = r.q1; r.q1 = t;
        for (uint i = 0u; i < 16u; i++)
            r.idx[i] = 15u - r.idx[i];
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, 3u, 5u);                 // mode 11 = 00011
    for (uint c = 0u; c < 3u; c++)    // e0.r e0.g e0.b
        bwPut(b, r.q0.q[c], 10u);
    for (uint c = 0u; c < 3u; c++)    // e1.r e1.g e1.b
        bwPut(b, r.q1.q[c], 10u);
    bwPut(b, r.idx[0], 3u);
    for (uint i = 1u; i < 16u; i++)
        bwPut(b, r.idx[i], 4u);
    return uvec4(b.words[0], b.words[1], b.words[2], b.words[3]);
}

// A 4x4 RGBA block (linear float) -> packed BC6H mode-11 block: half-int
// domain, per-channel min/max start, refine, pack.
uvec4 encodeBlockBc6h(in vec4 px[16])
{
    vec3 v[16];
    vec3 lo = vec3(1e30), hi = vec3(-1e30);
    for (uint i = 0u; i < 16u; i++) {
        v[i] = vec3(halfIntNorm(px[i].r), halfIntNorm(px[i].g), halfIntNorm(px[i].b));
        lo = min(lo, v[i]);
        hi = max(hi, v[i]);
    }
    return packMode11(refineMode11(v, lo, hi, P.refineIters));
}

#endif // ANBC_BC6H_COMMON_GLSL
