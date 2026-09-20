// Neural BC7 encoder -- shared part (GLSL port of anbc_bc7_common.metal).
// Needs anbc_common.glsl first.
//
// Mirrors the Python inference path (src/bc7_codec.py refine_mode6/refine_mode5
// + pack_mode6/5_blocks_batch). Everything after the endpoints come out of
// the MLPs is here:
//   1. refine: exact nearest-palette indices, least-squares endpoint refit,
//      snap endpoints to what the bitstream stores; repeat P.refineIters times
//   2. pick the mode with the lower squared error of its real reconstruction
//   3. bit-pack (mode 5 always rotation 0)

#ifndef ANBC_BC7_COMMON_GLSL
#define ANBC_BC7_COMMON_GLSL

#include "anbc_common.glsl"

const float kWeights4[16] = float[16](0.0, 4.0 / 64.0, 9.0 / 64.0, 13.0 / 64.0, 17.0 / 64.0, 21.0 / 64.0, 26.0 / 64.0, 30.0 / 64.0,
                                      34.0 / 64.0, 38.0 / 64.0, 43.0 / 64.0, 47.0 / 64.0, 51.0 / 64.0, 55.0 / 64.0, 60.0 / 64.0, 1.0);
const float kWeights2[4] = float[4](0.0, 21.0 / 64.0, 43.0 / 64.0, 1.0);

// ---------------------------------------------------------------------------
// Endpoint quantization (what the decoder will actually see)
// ---------------------------------------------------------------------------

struct Quant6 { uvec4 v7; uint p; vec4 dec; };

// Mode 6: 7 bits + one p-bit shared by the 4 channels, p chosen to minimise
// squared error (quantize_endpoint_and_pbit in bc7_codec.py).
Quant6 quantMode6(vec4 e)
{
    const vec4 y = clamp(round(e * 255.0), 0.0, 255.0);
    Quant6 best;
    float bestErr = 1e30;
    for (uint p = 0u; p < 2u; p++) {
        const vec4 v7 = clamp(round((y - float(p)) * 0.5), 0.0, 127.0);
        const vec4 rec = v7 * 2.0 + float(p);
        const vec4 d = rec - y;
        const float err = dot(d, d);
        if (err < bestErr) {
            bestErr = err;
            best.v7 = uvec4(v7);
            best.p = p;
            best.dec = rec / 255.0;
        }
    }
    return best;
}

// Mode 5 colour: 7 bits, no p-bit, MSB-replicated on decode.
uvec3 quantRgb7(vec3 e)
{
    const vec3 y = clamp(round(e * 255.0), 0.0, 255.0);
    return uvec3(clamp(round(y / 255.0 * 127.0), 0.0, 127.0));
}
vec3 decodeRgb7(uvec3 v) { return vec3((v << 1) | (v >> 6)) / 255.0; }

// Mode 5 alpha: plain 8 bits.
uint quantA8(float a) { return uint(clamp(round(a * 255.0), 0.0, 255.0)); }
float decodeA8(uint v) { return float(v) / 255.0; }

// ---------------------------------------------------------------------------
// Index snapping
// ---------------------------------------------------------------------------

uint nearestIndex4(float t)
{
    uint best = 0u;
    float bestD = 2.0;
    for (uint k = 0u; k < 16u; k++) {
        const float d = abs(t - kWeights4[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

uint nearestIndex2(float t)
{
    uint best = 0u;
    float bestD = 2.0;
    for (uint k = 0u; k < 4u; k++) {
        const float d = abs(t - kWeights2[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Mode 6
// ---------------------------------------------------------------------------

struct Mode6Result { Quant6 q0, q1; uint idx[16]; float sse; };

Mode6Result refineMode6(in vec4 px[16], vec4 e0, vec4 e1, uint iters)
{
    Mode6Result r;
    r.q0 = quantMode6(e0);
    r.q1 = quantMode6(e1);
    float w[16];
    for (uint it = 0u; it < iters; it++) {
        for (uint i = 0u; i < 16u; i++)
            w[i] = kWeights4[nearestIndex4(projectT(px[i], r.q0.dec, r.q1.dec))];
        lsEndpoints(px, w, e0, e1);
        r.q0 = quantMode6(e0);
        r.q1 = quantMode6(e1);
    }
    r.sse = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        r.idx[i] = nearestIndex4(projectT(px[i], r.q0.dec, r.q1.dec));
        const vec4 rec = mix(r.q0.dec, r.q1.dec, kWeights4[r.idx[i]]);
        const vec4 d = rec - px[i];
        r.sse += dot(d, d);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Mode 5 (rotation 0: separate RGB and alpha index sets)
// ---------------------------------------------------------------------------

struct Mode5Result { uvec3 c0, c1; uint a0, a1; uint cidx[16]; uint aidx[16]; float sse; };

Mode5Result refineMode5(in vec4 px[16], vec3 c0, vec3 c1, float a0, float a1, uint iters)
{
    vec3 rgb[16];
    float alpha[16];
    for (uint i = 0u; i < 16u; i++) { rgb[i] = px[i].rgb; alpha[i] = px[i].a; }

    Mode5Result r;
    r.c0 = quantRgb7(c0); r.c1 = quantRgb7(c1);
    r.a0 = quantA8(a0);   r.a1 = quantA8(a1);
    vec3 d0 = decodeRgb7(r.c0), d1 = decodeRgb7(r.c1);
    float da0 = decodeA8(r.a0), da1 = decodeA8(r.a1);

    float w[16];
    for (uint it = 0u; it < iters; it++) {
        for (uint i = 0u; i < 16u; i++)
            w[i] = kWeights2[nearestIndex2(projectT(rgb[i], d0, d1))];
        lsEndpoints(rgb, w, c0, c1);
        r.c0 = quantRgb7(c0); r.c1 = quantRgb7(c1);
        d0 = decodeRgb7(r.c0); d1 = decodeRgb7(r.c1);

        for (uint i = 0u; i < 16u; i++)
            w[i] = kWeights2[nearestIndex2(projectT(alpha[i], da0, da1))];
        lsEndpoints(alpha, w, a0, a1);
        r.a0 = quantA8(a0); r.a1 = quantA8(a1);
        da0 = decodeA8(r.a0); da1 = decodeA8(r.a1);
    }

    r.sse = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        r.cidx[i] = nearestIndex2(projectT(rgb[i], d0, d1));
        r.aidx[i] = nearestIndex2(projectT(alpha[i], da0, da1));
        const vec3 dc = mix(d0, d1, kWeights2[r.cidx[i]]) - rgb[i];
        const float dA = mix(da0, da1, kWeights2[r.aidx[i]]) - alpha[i];
        r.sse += dot(dc, dc) + dA * dA;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

uvec4 packMode6(Mode6Result r)
{
    // Anchor index only stores 3 bits: swap endpoints and invert if needed.
    if (r.idx[0] > 7u) {
        const Quant6 t = r.q0; r.q0 = r.q1; r.q1 = t;
        for (uint i = 0u; i < 16u; i++)
            r.idx[i] = 15u - r.idx[i];
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, 1u << 6, 7u);            // mode 6 prefix
    for (uint c = 0u; c < 4u; c++) {  // R0 R1 G0 G1 B0 B1 A0 A1
        bwPut(b, r.q0.v7[c], 7u);
        bwPut(b, r.q1.v7[c], 7u);
    }
    bwPut(b, r.q0.p, 1u);
    bwPut(b, r.q1.p, 1u);
    bwPut(b, r.idx[0], 3u);
    for (uint i = 1u; i < 16u; i++)
        bwPut(b, r.idx[i], 4u);
    return uvec4(b.words[0], b.words[1], b.words[2], b.words[3]);
}

uvec4 packMode5(Mode5Result r)
{
    if (r.cidx[0] > 1u) {
        const uvec3 t = r.c0; r.c0 = r.c1; r.c1 = t;
        for (uint i = 0u; i < 16u; i++)
            r.cidx[i] = 3u - r.cidx[i];
    }
    if (r.aidx[0] > 1u) {
        const uint t = r.a0; r.a0 = r.a1; r.a1 = t;
        for (uint i = 0u; i < 16u; i++)
            r.aidx[i] = 3u - r.aidx[i];
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, 1u << 5, 6u);            // mode 5 prefix
    bwPut(b, 0u, 2u);                 // rotation 0
    for (uint c = 0u; c < 3u; c++) {
        bwPut(b, r.c0[c], 7u);
        bwPut(b, r.c1[c], 7u);
    }
    bwPut(b, r.a0, 8u);
    bwPut(b, r.a1, 8u);
    bwPut(b, r.cidx[0], 1u);
    for (uint i = 1u; i < 16u; i++)
        bwPut(b, r.cidx[i], 2u);
    bwPut(b, r.aidx[0], 1u);
    for (uint i = 1u; i < 16u; i++)
        bwPut(b, r.aidx[i], 2u);
    return uvec4(b.words[0], b.words[1], b.words[2], b.words[3]);
}

// Network outputs -> refined, packed block.
//   y6: mode 6 = endpoint0 (4), endpoint1 (4), [16 blend factors, unused]
//   y5: mode 5 = rgb0 (3), rgb1 (3), [16 rgb blends], a0, a1, [16 alpha blends]
uvec4 finishBlock(in vec4 px[16], in float y6[ANBC_MLP_MAX_OUT], in float y5[ANBC_MLP_MAX_OUT])
{
    const vec4 e0 = vec4(y6[0], y6[1], y6[2], y6[3]);
    const vec4 e1 = vec4(y6[4], y6[5], y6[6], y6[7]);
    const Mode6Result m6 = refineMode6(px, e0, e1, P.refineIters);

    const vec3 c0 = vec3(y5[0], y5[1], y5[2]);
    const vec3 c1 = vec3(y5[3], y5[4], y5[5]);
    const Mode5Result m5 = refineMode5(px, c0, c1, y5[22], y5[23], P.refineIters);

    return (m5.sse < m6.sse) ? packMode5(m5) : packMode6(m6);
}

#endif // ANBC_BC7_COMMON_GLSL
