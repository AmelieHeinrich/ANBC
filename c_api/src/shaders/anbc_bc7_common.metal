// Neural BC7 encoder -- shared part (prepended to the scalar kernel and
// #included by the tensor-ops kernel, see CMakeLists.txt). Needs
// anbc_common.metal first.
//
// Mirrors the Python inference path (src/bc7_codec.py refine_mode6/refine_mode5
// + pack_mode6/5_blocks_batch). The kernels only differ in how the two MLPs
// are evaluated; everything after the endpoints come out is here:
//   1. refine: exact nearest-palette indices, least-squares endpoint refit,
//      snap endpoints to what the bitstream stores; repeat P.refineIters times
//   2. pick the mode with the lower squared error of its real reconstruction
//   3. bit-pack (mode 5 always rotation 0)

#ifndef ANBC_BC7_COMMON_METAL
#define ANBC_BC7_COMMON_METAL

#include "anbc_common.metal"

#define ANBC_MLP_IN 64

constant float kWeights4[16] = { 0.0f, 4.0f / 64, 9.0f / 64, 13.0f / 64, 17.0f / 64, 21.0f / 64, 26.0f / 64, 30.0f / 64,
                                 34.0f / 64, 38.0f / 64, 43.0f / 64, 47.0f / 64, 51.0f / 64, 55.0f / 64, 60.0f / 64, 1.0f };
constant float kWeights2[4] = { 0.0f, 21.0f / 64, 43.0f / 64, 1.0f };

// ---------------------------------------------------------------------------
// Endpoint quantization (what the decoder will actually see)
// ---------------------------------------------------------------------------

struct Quant6 { uint4 v7; uint p; float4 dec; };

// Mode 6: 7 bits + one p-bit shared by the 4 channels, p chosen to minimise
// squared error (quantize_endpoint_and_pbit in bc7_codec.py).
static Quant6 quantMode6(float4 e)
{
    const float4 y = clamp(round(e * 255.0f), 0.0f, 255.0f);
    Quant6 best;
    float bestErr = 1e30f;
    for (uint p = 0; p < 2; p++) {
        const float4 v7 = clamp(round((y - float(p)) * 0.5f), 0.0f, 127.0f);
        const float4 rec = v7 * 2.0f + float(p);
        const float4 d = rec - y;
        const float err = dot(d, d);
        if (err < bestErr) {
            bestErr = err;
            best.v7 = uint4(v7);
            best.p = p;
            best.dec = rec / 255.0f;
        }
    }
    return best;
}

// Mode 5 colour: 7 bits, no p-bit, MSB-replicated on decode.
static inline uint3 quantRgb7(float3 e)
{
    const float3 y = clamp(round(e * 255.0f), 0.0f, 255.0f);
    return uint3(clamp(round(y / 255.0f * 127.0f), 0.0f, 127.0f));
}
static inline float3 decodeRgb7(uint3 v) { return float3((v << 1) | (v >> 6)) / 255.0f; }

// Mode 5 alpha: plain 8 bits.
static inline uint quantA8(float a) { return uint(clamp(round(a * 255.0f), 0.0f, 255.0f)); }
static inline float decodeA8(uint v) { return float(v) / 255.0f; }

// ---------------------------------------------------------------------------
// Index snapping
// ---------------------------------------------------------------------------

static inline uint nearestIndex4(float t)
{
    uint best = 0;
    float bestD = 2.0f;
    for (uint k = 0; k < 16; k++) {
        const float d = abs(t - kWeights4[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

static inline uint nearestIndex2(float t)
{
    uint best = 0;
    float bestD = 2.0f;
    for (uint k = 0; k < 4; k++) {
        const float d = abs(t - kWeights2[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Mode 6
// ---------------------------------------------------------------------------

struct Mode6Result { Quant6 q0, q1; uint idx[16]; float sse; };

static Mode6Result refineMode6(thread const float4* px, float4 e0, float4 e1, uint iters)
{
    Mode6Result r;
    r.q0 = quantMode6(e0);
    r.q1 = quantMode6(e1);
    float w[16];
    for (uint it = 0; it < iters; it++) {
        for (uint i = 0; i < 16; i++)
            w[i] = kWeights4[nearestIndex4(projectT(px[i], r.q0.dec, r.q1.dec))];
        lsEndpoints(px, w, e0, e1);
        r.q0 = quantMode6(e0);
        r.q1 = quantMode6(e1);
    }
    r.sse = 0.0f;
    for (uint i = 0; i < 16; i++) {
        r.idx[i] = nearestIndex4(projectT(px[i], r.q0.dec, r.q1.dec));
        const float4 rec = mix(r.q0.dec, r.q1.dec, kWeights4[r.idx[i]]);
        const float4 d = rec - px[i];
        r.sse += dot(d, d);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Mode 5 (rotation 0: separate RGB and alpha index sets)
// ---------------------------------------------------------------------------

struct Mode5Result { uint3 c0, c1; uint a0, a1; uint cidx[16]; uint aidx[16]; float sse; };

static Mode5Result refineMode5(thread const float4* px, float3 c0, float3 c1, float a0, float a1, uint iters)
{
    float3 rgb[16];
    float alpha[16];
    for (uint i = 0; i < 16; i++) { rgb[i] = px[i].rgb; alpha[i] = px[i].a; }

    Mode5Result r;
    r.c0 = quantRgb7(c0); r.c1 = quantRgb7(c1);
    r.a0 = quantA8(a0);   r.a1 = quantA8(a1);
    float3 d0 = decodeRgb7(r.c0), d1 = decodeRgb7(r.c1);
    float da0 = decodeA8(r.a0), da1 = decodeA8(r.a1);

    float w[16];
    for (uint it = 0; it < iters; it++) {
        for (uint i = 0; i < 16; i++)
            w[i] = kWeights2[nearestIndex2(projectT(rgb[i], d0, d1))];
        lsEndpoints(rgb, w, c0, c1);
        r.c0 = quantRgb7(c0); r.c1 = quantRgb7(c1);
        d0 = decodeRgb7(r.c0); d1 = decodeRgb7(r.c1);

        for (uint i = 0; i < 16; i++)
            w[i] = kWeights2[nearestIndex2(projectT(alpha[i], da0, da1))];
        lsEndpoints(alpha, w, a0, a1);
        r.a0 = quantA8(a0); r.a1 = quantA8(a1);
        da0 = decodeA8(r.a0); da1 = decodeA8(r.a1);
    }

    r.sse = 0.0f;
    for (uint i = 0; i < 16; i++) {
        r.cidx[i] = nearestIndex2(projectT(rgb[i], d0, d1));
        r.aidx[i] = nearestIndex2(projectT(alpha[i], da0, da1));
        const float3 dc = mix(d0, d1, kWeights2[r.cidx[i]]) - rgb[i];
        const float dA = mix(da0, da1, kWeights2[r.aidx[i]]) - alpha[i];
        r.sse += dot(dc, dc) + dA * dA;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

static uint4 packMode6(Mode6Result r)
{
    // Anchor index only stores 3 bits: swap endpoints and invert if needed.
    if (r.idx[0] > 7) {
        const Quant6 t = r.q0; r.q0 = r.q1; r.q1 = t;
        for (uint i = 0; i < 16; i++)
            r.idx[i] = 15 - r.idx[i];
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, 1u << 6, 7);            // mode 6 prefix
    for (uint c = 0; c < 4; c++) {   // R0 R1 G0 G1 B0 B1 A0 A1
        bwPut(b, r.q0.v7[c], 7);
        bwPut(b, r.q1.v7[c], 7);
    }
    bwPut(b, r.q0.p, 1);
    bwPut(b, r.q1.p, 1);
    bwPut(b, r.idx[0], 3);
    for (uint i = 1; i < 16; i++)
        bwPut(b, r.idx[i], 4);
    return uint4(b.words[0], b.words[1], b.words[2], b.words[3]);
}

static uint4 packMode5(Mode5Result r)
{
    if (r.cidx[0] > 1) {
        const uint3 t = r.c0; r.c0 = r.c1; r.c1 = t;
        for (uint i = 0; i < 16; i++)
            r.cidx[i] = 3 - r.cidx[i];
    }
    if (r.aidx[0] > 1) {
        const uint t = r.a0; r.a0 = r.a1; r.a1 = t;
        for (uint i = 0; i < 16; i++)
            r.aidx[i] = 3 - r.aidx[i];
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, 1u << 5, 6);            // mode 5 prefix
    bwPut(b, 0, 2);                  // rotation 0
    for (uint c = 0; c < 3; c++) {
        bwPut(b, r.c0[c], 7);
        bwPut(b, r.c1[c], 7);
    }
    bwPut(b, r.a0, 8);
    bwPut(b, r.a1, 8);
    bwPut(b, r.cidx[0], 1);
    for (uint i = 1; i < 16; i++)
        bwPut(b, r.cidx[i], 2);
    bwPut(b, r.aidx[0], 1);
    for (uint i = 1; i < 16; i++)
        bwPut(b, r.aidx[i], 2);
    return uint4(b.words[0], b.words[1], b.words[2], b.words[3]);
}

// ---------------------------------------------------------------------------
// Shared kernel pieces
// ---------------------------------------------------------------------------

// Network outputs -> refined, packed block.
//   y6: mode 6 = endpoint0 (4), endpoint1 (4), [16 blend factors, unused]
//   y5: mode 5 = rgb0 (3), rgb1 (3), [16 rgb blends], a0, a1, [16 alpha blends]
static uint4 finishBlock(thread const float4* px, thread const float* y6, thread const float* y5, constant EncodeParams& P)
{
    const float4 e0 = float4(y6[0], y6[1], y6[2], y6[3]);
    const float4 e1 = float4(y6[4], y6[5], y6[6], y6[7]);
    const Mode6Result m6 = refineMode6(px, e0, e1, P.refineIters);

    const float3 c0 = float3(y5[0], y5[1], y5[2]);
    const float3 c1 = float3(y5[3], y5[4], y5[5]);
    const Mode5Result m5 = refineMode5(px, c0, c1, y5[22], y5[23], P.refineIters);

    return (m5.sse < m6.sse) ? packMode5(m5) : packMode6(m6);
}

#endif // ANBC_BC7_COMMON_METAL
