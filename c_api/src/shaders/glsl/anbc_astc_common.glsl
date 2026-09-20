// ASTC 4x4 encoder -- shared part (GLSL port of anbc_astc_common.metal).
// Needs anbc_common.glsl and anbc_astc_tables.glsl first.
//
// Mirrors src/astc_codec.py: the same refine (exact nearest-weight indices,
// least-squares endpoint refit, snap endpoints to what the block can store)
// and pack, then per-block config selection by squared error. Three texture
// kinds, each with its shortlist of single-partition 4x4 block configs:
//   GAME   (RGBA8)         A: CEM 8 RGB, 4-bit weights, QUANT_192 endpoints
//                          A3: CEM 8 RGB, 3-bit weights, QUANT_256
//                          B: CEM 12 RGBA, 2-bit weights, alpha in its own
//                             weight plane (dual plane), QUANT_48
//   NORMAL (X,Y -> L,A)    A: CEM 4 L+A, 4-bit weights, QUANT_256
//                          B: CEM 4, 2-bit weights x 2 planes, QUANT_256
//   FLOAT  (HDR RGB)       A: CEM 11 HDR RGB, 3-bit weights, QUANT_256
// No network: endpoints start from the block's per-channel min/max.
//
// Block layout (LSB-first): bits 0..10 block mode, 11..12 partitions-1 = 0,
// 13..16 CEM, 17.. endpoint ISE; dual plane: 2-bit CCS just below the
// weights; weight ISE stored bit-reversed from bit 127 down. The endpoint
// quant level is implied by the bits left (see astc_ise.py).
//
// HDR blocks work in ASTC's 16-bit LNS domain (x = lns / 65535, 2048 per
// octave); pixels arrive as half floats and are converted with the exact
// inverse of the decoder's lns_to_sf16.
//
// The Metal templates on the quant level Q / channel count V are stamped out
// here with macros for exactly the instantiations the kernels use.

#ifndef ANBC_ASTC_COMMON_GLSL
#define ANBC_ASTC_COMMON_GLSL

#include "anbc_common.glsl"
#include "anbc_astc_tables.glsl"

#define ANBC_LNS_MAX 65535.0
#define ANBC_LNS_MAX_FINITE 63487 // LNS of the largest finite half (0x7BFF)

// Block mode for a 4x4 weight grid: bits 6:5 = 10 (A = 2), 8:7 = 00 (B = 0),
// 3:2 = 00; the weight range is R = (bit 4 as R0, bits 1:0 as R2 R1) + H (bit
// 9); D (bit 10) = dual plane. QUANT_4: R=4 H=0, QUANT_8: R=7 H=0, QUANT_16:
// R=4 H=1.
#define ANBC_ASTC_MODE_W2 0x042u
#define ANBC_ASTC_MODE_W3 0x053u
#define ANBC_ASTC_MODE_W4 0x242u
#define ANBC_ASTC_MODE_DUAL 0x400u

#define ANBC_CEM_LA 4u
#define ANBC_CEM_RGB 8u
#define ANBC_CEM_HDR_RGB 11u
#define ANBC_CEM_RGBA 12u

// ---------------------------------------------------------------------------
// Endpoint quantization (what the decoder will actually see)
// ---------------------------------------------------------------------------

// QUANT levels: 256 (8 bits, identity), 192 and 48 (trit ranges, table
// driven). quantCodeQ -> the ISE code; decodeCodeQ -> the 8-bit value the
// decoder reconstructs from it; quantDecQ -> decoded value of a [0,1]
// endpoint after quantization, per channel.
uint quantCode256(float v) { return uint(clamp(round(v * 255.0), 0.0, 255.0)); }
float decodeCode256(uint code) { return float(code) / 255.0; }
uint quantCode192(float v) { return kAstcCodeOfValue192[uint(clamp(round(v * 255.0), 0.0, 255.0))]; }
float decodeCode192(uint code) { return float(kAstcUnquant192[code]) / 255.0; }
uint quantCode48(float v) { return kAstcCodeOfValue48[uint(clamp(round(v * 255.0), 0.0, 255.0))]; }
float decodeCode48(uint code) { return float(kAstcUnquant48[code]) / 255.0; }

#define ANBC_QUANT_DEC(Q)                                                                                       \
    float quantDec##Q(float v) { return decodeCode##Q(quantCode##Q(v)); }                                       \
    vec2 quantDec##Q(vec2 v) { return vec2(quantDec##Q(v.x), quantDec##Q(v.y)); }                               \
    vec3 quantDec##Q(vec3 v) { return vec3(quantDec##Q(v.x), quantDec##Q(v.y), quantDec##Q(v.z)); }             \
    vec4 quantDec##Q(vec4 v) { return vec4(quantDec##Q(v.x), quantDec##Q(v.y), quantDec##Q(v.z), quantDec##Q(v.w)); }
ANBC_QUANT_DEC(256)
ANBC_QUANT_DEC(192)
ANBC_QUANT_DEC(48)

// ---------------------------------------------------------------------------
// Index snapping + one refined line (LDR, per-channel quantization)
// ---------------------------------------------------------------------------

#define ANBC_NEAREST_WEIGHT(COUNT)                                                           \
    uint nearestWeight##COUNT(float t)                                                       \
    {                                                                                        \
        uint best = 0u;                                                                      \
        float bestD = 2.0;                                                                   \
        for (uint k = 0u; k < uint(COUNT); k++) {                                             \
            const float d = abs(t - kAstcWeights##COUNT[k]);                                 \
            if (d < bestD) { bestD = d; best = k; }                                          \
        }                                                                                    \
        return best;                                                                         \
    }
ANBC_NEAREST_WEIGHT(4)
ANBC_NEAREST_WEIGHT(8)
ANBC_NEAREST_WEIGHT(16)

struct LineResult1 { float d0, d1; uint idx[16]; float sse; };
struct LineResult2 { vec2 d0, d1; uint idx[16]; float sse; };
struct LineResult3 { vec3 d0, d1; uint idx[16]; float sse; };

// bc7's refineMode6 for any channel count / weight table / quant level:
// returns decoded endpoints, exact indices and the squared error.
// ANBC_REFINE_LINE(name, result struct, V, quant level, weight count)
#define ANBC_REFINE_LINE(NAME, R, V, Q, COUNT)                                               \
    R NAME(in V px[16], V e0, V e1, uint iters)                                              \
    {                                                                                        \
        R r;                                                                                 \
        r.d0 = quantDec##Q(e0);                                                              \
        r.d1 = quantDec##Q(e1);                                                              \
        float w[16];                                                                         \
        for (uint it = 0u; it < iters; it++) {                                               \
            for (uint i = 0u; i < 16u; i++)                                                  \
                w[i] = kAstcWeights##COUNT[nearestWeight##COUNT(projectT(px[i], r.d0, r.d1))]; \
            lsEndpoints(px, w, e0, e1);                                                      \
            r.d0 = quantDec##Q(e0);                                                          \
            r.d1 = quantDec##Q(e1);                                                          \
        }                                                                                    \
        r.sse = 0.0;                                                                         \
        for (uint i = 0u; i < 16u; i++) {                                                    \
            r.idx[i] = nearestWeight##COUNT(projectT(px[i], r.d0, r.d1));                    \
            const V d = mix(r.d0, r.d1, kAstcWeights##COUNT[r.idx[i]]) - px[i];              \
            r.sse += vdot(d, d);                                                             \
        }                                                                                    \
        return r;                                                                            \
    }
ANBC_REFINE_LINE(refineLine192W16v3, LineResult3, vec3, 192, 16)
ANBC_REFINE_LINE(refineLine256W8v3, LineResult3, vec3, 256, 8)
ANBC_REFINE_LINE(refineLine48W4v3, LineResult3, vec3, 48, 4)
ANBC_REFINE_LINE(refineLine48W4v1, LineResult1, float, 48, 4)
ANBC_REFINE_LINE(refineLine256W16v2, LineResult2, vec2, 256, 16)
ANBC_REFINE_LINE(refineLine256W4v1, LineResult1, float, 256, 4)

// ---------------------------------------------------------------------------
// HDR: LNS domain + CEM 11 endpoint packing (port of astc_codec.py
// pack_hdr_endpoints / unpack_hdr_endpoints, themselves astcenc's
// quantize_hdr_rgb with QUANT_256 and hdr_rgb_unpack)
// ---------------------------------------------------------------------------

// half float -> 16-bit LNS (the smallest LNS value the decoder maps back to
// this half; exact inverse of lns_to_sf16). Negatives clamp to 0.
uint halfToLns(float v)
{
    uint bits = packHalf2x16(vec2(v, 0.0)) & 0xFFFFu;
    if ((bits & 0x8000u) != 0u)
        bits = 0u;
    bits = min(bits, 0x7BFFu);
    const uint ec = bits >> 10;
    const uint t = (bits & 0x3FFu) * 8u;
    uint mc = (t + 2u) / 3u;                 // ceil(t / 3)
    if (mc >= 512u) mc = (t + 512u + 3u) / 4u;
    if (mc >= 1536u) mc = (t + 2048u + 4u) / 5u;
    return (ec << 11) | mc;
}

float lnsNorm(float v) { return float(halfToLns(v)) / ANBC_LNS_MAX; }

// astcenc flt2int_rtn: (int)(x + 0.5), truncation toward zero.
int rtn(float x) { return int(trunc(x + 0.5)); }

struct HdrPacked { uint v[6]; bool swapped; };

const ivec4 kHdrModeBits[8] = ivec4[8](ivec4(9, 7, 6, 7), ivec4(9, 8, 6, 6), ivec4(10, 6, 7, 7), ivec4(10, 7, 7, 6),
                                       ivec4(11, 8, 6, 5), ivec4(11, 6, 8, 6), ivec4(12, 7, 7, 5), ivec4(12, 6, 7, 6));
const vec3 kHdrModeCutoffs[8] = vec3[8](vec3(16384, 8192, 8192), vec3(32768, 8192, 4096), vec3(4096, 8192, 4096), vec3(8192, 8192, 2048),
                                        vec3(8192, 2048, 512), vec3(2048, 8192, 1024), vec3(2048, 2048, 256), vec3(1024, 2048, 512));
const int kHdrModeShift[8] = int[8](7, 7, 6, 6, 5, 5, 4, 4);
const int kHdrDbits[8] = int[8](7, 6, 7, 6, 5, 6, 5, 6);

// e0/e1: LNS colours in [0, 65535]. Stores e1's major component and
// non-negative deltas down to e0, so a block whose e0 is brighter on the
// major axis is stored swapped (the caller inverts the weights).
HdrPacked packHdrEndpoints(vec3 e0, vec3 e1)
{
    HdrPacked o;
    vec3 c0 = clamp(e0, 0.0, float(ANBC_LNS_MAX_FINITE));
    vec3 c1 = clamp(e1, 0.0, float(ANBC_LNS_MAX_FINITE));
    const vec3 hi = max(c0, c1);
    const uint majcomp = (hi.x >= hi.y && hi.x >= hi.z) ? 0u : (hi.y >= hi.z ? 1u : 2u);
    o.swapped = c0[majcomp] > c1[majcomp];
    if (o.swapped) { const vec3 t = c0; c0 = c1; c1 = t; }
    if (majcomp == 1u) { c0 = c0.yxz; c1 = c1.yxz; }
    else if (majcomp == 2u) { c0 = c0.zyx; c1 = c1.zyx; }

    const float aBase = c1.x;
    const float b0Base = aBase - c1.y;
    const float b1Base = aBase - c1.z;
    const float cBase = aBase - c0.x;
    const float d0Base = aBase - b0Base - cBase - c0.y;
    const float d1Base = aBase - b1Base - cBase - c0.z;

    for (int mode = 7; mode >= 0; mode--) {
        if (b0Base > kHdrModeCutoffs[mode][0] || b1Base > kHdrModeCutoffs[mode][0] || cBase > kHdrModeCutoffs[mode][1] ||
            abs(d0Base) > kHdrModeCutoffs[mode][2] || abs(d1Base) > kHdrModeCutoffs[mode][2])
            continue;
        const int sh = kHdrModeShift[mode];
        const float scale = 1.0 / float(1 << sh);
        const int bCut = 1 << kHdrModeBits[mode][1];
        const int cCut = 1 << kHdrModeBits[mode][2];
        const int dCut = 1 << (kHdrModeBits[mode][3] - 1);

        const int aInt = rtn(aBase * scale);
        const float aF = float(aInt << sh);
        const int cInt = rtn(clamp(aF - c0.x, 0.0, 65535.0) * scale);
        if (cInt >= cCut)
            continue;
        const float cF = float(cInt << sh);
        const int b0Int = rtn(clamp(aF - c1.y, 0.0, 65535.0) * scale);
        const int b1Int = rtn(clamp(aF - c1.z, 0.0, 65535.0) * scale);
        if (b0Int >= bCut || b1Int >= bCut)
            continue;
        const float b0F = float(b0Int << sh), b1F = float(b1Int << sh);
        const int d0Int = rtn(clamp(aF - b0F - cF - c0.y, -65535.0, 65535.0) * scale);
        const int d1Int = rtn(clamp(aF - b1F - cF - c0.z, -65535.0, 65535.0) * scale);
        if (abs(d0Int) >= dCut || abs(d1Int) >= dCut)
            continue;

        int bit0, bit1, bit2, bit3, bit4, bit5;
        if (mode == 0 || mode == 1 || mode == 3 || mode == 4 || mode == 6) { bit0 = (b0Int >> 6) & 1; bit1 = (b1Int >> 6) & 1; }
        else if (mode == 2) { bit0 = (aInt >> 9) & 1; bit1 = (cInt >> 6) & 1; }
        else { bit0 = (aInt >> 9) & 1; bit1 = (aInt >> 10) & 1; }
        if (mode == 0 || mode == 2) { bit2 = (d0Int >> 6) & 1; bit3 = (d1Int >> 6) & 1; }
        else if (mode == 1 || mode == 4) { bit2 = (b0Int >> 7) & 1; bit3 = (b1Int >> 7) & 1; }
        else if (mode == 3) { bit2 = (aInt >> 9) & 1; bit3 = (cInt >> 6) & 1; }
        else if (mode == 5) { bit2 = (cInt >> 7) & 1; bit3 = (cInt >> 6) & 1; }
        else { bit2 = (aInt >> 11) & 1; bit3 = (cInt >> 6) & 1; }
        if (mode == 4 || mode == 6) { bit4 = (aInt >> 9) & 1; bit5 = (aInt >> 10) & 1; }
        else { bit4 = (d0Int >> 5) & 1; bit5 = (d1Int >> 5) & 1; }

        o.v[0] = uint(aInt & 0xFF);
        o.v[1] = uint((cInt & 0x3F) | ((mode & 1) << 7) | ((aInt & 0x100) >> 2));
        o.v[2] = uint((b0Int & 0x3F) | (bit0 << 6) | (((mode >> 1) & 1) << 7));
        o.v[3] = uint((b1Int & 0x3F) | (bit1 << 6) | (((mode >> 2) & 1) << 7));
        o.v[4] = uint((d0Int & 0x1F) | (bit2 << 6) | (bit4 << 5) | (int(majcomp & 1u) << 7));
        o.v[5] = uint((d1Int & 0x1F) | (bit3 << 6) | (bit5 << 5) | (int((majcomp >> 1) & 1u) << 7));
        return o;
    }
    // Flat fallback (majcomp 3): 8 bits for R and G, 7 for B.
    const vec3 f0 = clamp(e0, 0.0, 65020.0), f1 = clamp(e1, 0.0, 65020.0);
    o.swapped = false;
    o.v[0] = uint(rtn(f0.x / 256.0)); o.v[1] = uint(rtn(f1.x / 256.0));
    o.v[2] = uint(rtn(f0.y / 256.0)); o.v[3] = uint(rtn(f1.y / 256.0));
    o.v[4] = uint(rtn(f0.z / 512.0) + 128); o.v[5] = uint(rtn(f1.z / 512.0) + 128);
    return o;
}

// The 16-bit LNS endpoints a decoder reconstructs from six CEM 11 values.
void unpackHdrEndpoints(in uint v[6], out vec3 e0, out vec3 e1)
{
    const int v0 = int(v[0]), v1 = int(v[1]), v2 = int(v[2]), v3 = int(v[3]), v4 = int(v[4]), v5 = int(v[5]);
    const int modeval = ((v1 >> 7) & 1) | (((v2 >> 7) & 1) << 1) | (((v3 >> 7) & 1) << 2);
    const int majcomp = ((v4 >> 7) & 1) | (((v5 >> 7) & 1) << 1);
    if (majcomp == 3) {
        e0 = vec3(v0 << 8, v2 << 8, (v4 & 0x7F) << 9);
        e1 = vec3(v1 << 8, v3 << 8, (v5 & 0x7F) << 9);
        return;
    }
    int a = v0 | ((v1 & 0x40) << 2);
    int b0 = v2 & 0x3F, b1 = v3 & 0x3F, c = v1 & 0x3F, d0 = v4 & 0x7F, d1 = v5 & 0x7F;
    const int dbits = kHdrDbits[modeval];
    const int bit0 = (v2 >> 6) & 1, bit1 = (v3 >> 6) & 1, bit2 = (v4 >> 6) & 1, bit3 = (v5 >> 6) & 1;
    const int bit4 = (v4 >> 5) & 1, bit5 = (v5 >> 5) & 1;
    const int ohmod = 1 << modeval;
    if ((ohmod & 0xA4) != 0) a |= bit0 << 9;
    if ((ohmod & 0x8) != 0) a |= bit2 << 9;
    if ((ohmod & 0x50) != 0) a |= bit4 << 9;
    if ((ohmod & 0x50) != 0) a |= bit5 << 10;
    if ((ohmod & 0xA0) != 0) a |= bit1 << 10;
    if ((ohmod & 0xC0) != 0) a |= bit2 << 11;
    if ((ohmod & 0x4) != 0) c |= bit1 << 6;
    if ((ohmod & 0xE8) != 0) c |= bit3 << 6;
    if ((ohmod & 0x20) != 0) c |= bit2 << 7;
    if ((ohmod & 0x5B) != 0) { b0 |= bit0 << 6; b1 |= bit1 << 6; }
    if ((ohmod & 0x12) != 0) { b0 |= bit2 << 7; b1 |= bit3 << 7; }
    if ((ohmod & 0xAF) != 0) { d0 |= bit4 << 5; d1 |= bit5 << 5; }
    if ((ohmod & 0x5) != 0) { d0 |= bit2 << 6; d1 |= bit3 << 6; }
    // sign-extend d0/d1 from dbits (higher bits are other fields)
    const int sign = 1 << (dbits - 1);
    d0 &= (sign << 1) - 1; d1 &= (sign << 1) - 1;
    if ((d0 & sign) != 0) d0 -= sign << 1;
    if ((d1 & sign) != 0) d1 -= sign << 1;
    const int shamt = (modeval >> 1) ^ 3;
    a <<= shamt; b0 <<= shamt; b1 <<= shamt; c <<= shamt; d0 <<= shamt; d1 <<= shamt;
    ivec3 r0 = clamp(ivec3(a - c, a - b0 - c - d0, a - b1 - c - d1), 0, 4095);
    ivec3 r1 = clamp(ivec3(a, a - b0, a - b1), 0, 4095);
    if (majcomp == 1) { r0 = r0.yxz; r1 = r1.yxz; }
    else if (majcomp == 2) { r0 = r0.zyx; r1 = r1.zyx; }
    e0 = vec3(r0 << 4);
    e1 = vec3(r1 << 4);
}

struct HdrLineResult { HdrPacked packed; vec3 d0, d1; uint idx[16]; float sse; };

// The CEM 11 quantizer is joint over both endpoints and not a fixed point
// of itself (decoding can tie two channels for the major component), so
// the packed values of the last quantization are kept for the packer.
void quantHdr(vec3 e0, vec3 e1, inout HdrLineResult r)
{
    r.packed = packHdrEndpoints(e0 * ANBC_LNS_MAX, e1 * ANBC_LNS_MAX);
    vec3 d0, d1;
    unpackHdrEndpoints(r.packed.v, d0, d1);
    if (r.packed.swapped) { const vec3 t = d0; d0 = d1; d1 = t; }
    r.d0 = d0 / ANBC_LNS_MAX;
    r.d1 = d1 / ANBC_LNS_MAX;
}

// FLOAT_A uses the 8-entry weight table.
HdrLineResult refineHdr(in vec3 px[16], vec3 e0, vec3 e1, uint iters)
{
    HdrLineResult r;
    quantHdr(e0, e1, r);
    float w[16];
    for (uint it = 0u; it < iters; it++) {
        for (uint i = 0u; i < 16u; i++)
            w[i] = kAstcWeights8[nearestWeight8(projectT(px[i], r.d0, r.d1))];
        lsEndpoints(px, w, e0, e1);
        quantHdr(e0, e1, r);
    }
    r.sse = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        r.idx[i] = nearestWeight8(projectT(px[i], r.d0, r.d1));
        const vec3 d = mix(r.d0, r.d1, kAstcWeights8[r.idx[i]]) - px[i];
        r.sse += dot(d, d);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

// Trit ISE block layout: T bits following each of the 5 elements.
const uint kTBits[5] = uint[5](2u, 2u, 1u, 2u, 1u);
const uint kTShift[5] = uint[5](0u, 2u, 4u, 5u, 7u);

// Writes `count` ISE codes (bits-only or trit ranges) to b, LSB-first.
// `bits` is the number of bits below the trit: 8 (QUANT_256), 6 (192), 4 (48).
void iseWrite(inout BitWriter b, in uint codes[8], uint count, uint bits)
{
    if (bits == 8u) {
        for (uint i = 0u; i < count; i++)
            bwPut(b, codes[i], bits);
        return;
    }
    for (uint start = 0u; start < count; start += 5u) {
        const uint n = min(5u, count - start);
        uint index = 0u;
        for (uint j = 0u; j < 5u; j++) { // t4*81 + t3*27 + t2*9 + t1*3 + t0
            const uint trit = (j < n) ? (codes[start + j] >> bits) : 0u;
            index += trit * (j == 0u ? 1u : j == 1u ? 3u : j == 2u ? 9u : j == 3u ? 27u : 81u);
        }
        const uint T = kAstcIntegerOfTrits[index];
        for (uint j = 0u; j < n; j++) {
            bwPut(b, codes[start + j] & ((1u << bits) - 1u), bits);
            bwPut(b, (T >> kTShift[j]) & ((1u << kTBits[j]) - 1u), kTBits[j]);
        }
    }
}

// Header + endpoints go in from bit 0 (b), the weight stream (wb) is
// bit-reversed into the top of the block.
uvec4 assembleBlock(in BitWriter b, in BitWriter wb)
{
    uvec4 o = uvec4(b.words[0], b.words[1], b.words[2], b.words[3]);
    o.w |= bitfieldReverse(wb.words[0]);
    o.z |= bitfieldReverse(wb.words[1]);
    o.y |= bitfieldReverse(wb.words[2]);
    o.x |= bitfieldReverse(wb.words[3]);
    return o;
}

void writeHeader(inout BitWriter b, uint blockMode, uint cem)
{
    bwInit(b);
    bwPut(b, blockMode, 11u);
    bwPut(b, 0u, 2u); // single partition
    bwPut(b, cem, 4u);
}

// Single-plane LDR block: values are v0[c], v1[c] interleaved per channel.
// `iseBits` selects the quant level (8 / 6 / 4), `wbits` the weight width.
uvec4 packSinglePlane(uint blockMode, uint cem, in uint codes[8], uint nValues, uint iseBits, in uint idx[16], uint wbits)
{
    BitWriter b, wb;
    writeHeader(b, blockMode, cem);
    iseWrite(b, codes, nValues, iseBits);
    bwInit(wb);
    for (uint i = 0u; i < 16u; i++)
        bwPut(wb, idx[i], wbits);
    return assembleBlock(b, wb);
}

// Dual-plane LDR block: plane 1 / plane 2 weights interleaved per texel,
// CCS just below the weight stream (bit 128 - 32 * wbits - 2).
uvec4 packDualPlane(uint blockMode, uint cem, uint ccs, in uint codes[8], uint nValues, uint iseBits,
                    in uint idx1[16], in uint idx2[16], uint wbits)
{
    BitWriter b, wb;
    writeHeader(b, blockMode | ANBC_ASTC_MODE_DUAL, cem);
    iseWrite(b, codes, nValues, iseBits);
    b.pos = 128u - 32u * wbits - 2u;
    bwPut(b, ccs, 2u);
    bwInit(wb);
    for (uint i = 0u; i < 16u; i++) {
        bwPut(wb, idx1[i], wbits);
        bwPut(wb, idx2[i], wbits);
    }
    return assembleBlock(b, wb);
}

// CEM 8/12 decoders swap the endpoints (and blue-contract) when
// sum(rgb0) > sum(rgb1): keep sum(rgb0) <= sum(rgb1) by swapping ourselves
// and inverting the weights (every weight table is symmetric).
bool rgbSwapNeeded(vec3 d0, vec3 d1)
{
    // d0/d1 are decoded values, so round(d * 255) is exactly the stored 8-bit value
    const vec3 y0 = round(d0 * 255.0), y1 = round(d1 * 255.0);
    return (y0.x + y0.y + y0.z) > (y1.x + y1.y + y1.z);
}

// ---------------------------------------------------------------------------
// GAME: RGBA8, configs A (CEM 8, W4, Q192), A3 (CEM 8, W3, Q256), B (CEM 12
// dual plane on alpha, W2, Q48). e0/e1: initial RGBA endpoints.
// ---------------------------------------------------------------------------

uvec4 encodeAstcGame(in vec4 px[16], vec4 e0, vec4 e1, uint iters)
{
    vec3 rgb[16];
    float alpha[16];
    float alphaSse = 0.0; // error of the not-stored alpha (decodes as 1) for the CEM 8 configs
    for (uint i = 0u; i < 16u; i++) {
        rgb[i] = px[i].rgb;
        alpha[i] = px[i].a;
        alphaSse += (1.0 - px[i].a) * (1.0 - px[i].a);
    }

    const LineResult3 a = refineLine192W16v3(rgb, e0.rgb, e1.rgb, iters);
    const LineResult3 a3 = refineLine256W8v3(rgb, e0.rgb, e1.rgb, iters);
    const LineResult3 bRgb = refineLine48W4v3(rgb, e0.rgb, e1.rgb, iters);
    const LineResult1 bA = refineLine48W4v1(alpha, e0.a, e1.a, iters);

    const float sseA = a.sse + alphaSse, sseA3 = a3.sse + alphaSse, sseB = bRgb.sse + bA.sse;
    uint codes[8];
    uint idx[16], idx2[16];
    if (sseB < sseA && sseB < sseA3) {
        vec3 d0 = bRgb.d0, d1 = bRgb.d1;
        float a0 = bA.d0, a1 = bA.d1;
        const bool swap = rgbSwapNeeded(d0, d1);
        if (swap) { const vec3 t = d0; d0 = d1; d1 = t; const float ta = a0; a0 = a1; a1 = ta; }
        for (uint i = 0u; i < 16u; i++) {
            idx[i] = swap ? 3u - bRgb.idx[i] : bRgb.idx[i];
            idx2[i] = swap ? 3u - bA.idx[i] : bA.idx[i];
        }
        codes[0] = quantCode48(d0.x); codes[1] = quantCode48(d1.x);
        codes[2] = quantCode48(d0.y); codes[3] = quantCode48(d1.y);
        codes[4] = quantCode48(d0.z); codes[5] = quantCode48(d1.z);
        codes[6] = quantCode48(a0);   codes[7] = quantCode48(a1);
        return packDualPlane(ANBC_ASTC_MODE_W2, ANBC_CEM_RGBA, 3u, codes, 8u, 4u, idx, idx2, 2u);
    }
    if (sseA3 < sseA) {
        vec3 d0 = a3.d0, d1 = a3.d1;
        const bool swap = rgbSwapNeeded(d0, d1);
        if (swap) { const vec3 t = d0; d0 = d1; d1 = t; }
        for (uint i = 0u; i < 16u; i++)
            idx[i] = swap ? 7u - a3.idx[i] : a3.idx[i];
        codes[0] = quantCode256(d0.x); codes[1] = quantCode256(d1.x);
        codes[2] = quantCode256(d0.y); codes[3] = quantCode256(d1.y);
        codes[4] = quantCode256(d0.z); codes[5] = quantCode256(d1.z);
        codes[6] = 0u; codes[7] = 0u;
        return packSinglePlane(ANBC_ASTC_MODE_W3, ANBC_CEM_RGB, codes, 6u, 8u, idx, 3u);
    }
    vec3 d0 = a.d0, d1 = a.d1;
    const bool swap = rgbSwapNeeded(d0, d1);
    if (swap) { const vec3 t = d0; d0 = d1; d1 = t; }
    for (uint i = 0u; i < 16u; i++)
        idx[i] = swap ? 15u - a.idx[i] : a.idx[i];
    codes[0] = quantCode192(d0.x); codes[1] = quantCode192(d1.x);
    codes[2] = quantCode192(d0.y); codes[3] = quantCode192(d1.y);
    codes[4] = quantCode192(d0.z); codes[5] = quantCode192(d1.z);
    codes[6] = 0u; codes[7] = 0u;
    return packSinglePlane(ANBC_ASTC_MODE_W4, ANBC_CEM_RGB, codes, 6u, 6u, idx, 4u);
}

// ---------------------------------------------------------------------------
// NORMAL: (X, Y) from the source's R, G -> CEM 4 luminance + alpha (sample
// .ra). Configs A (W4, one plane), B (W2, two planes). e0/e1: initial (X,Y).
// ---------------------------------------------------------------------------

uvec4 encodeAstcNormal(in vec4 px[16], vec2 e0, vec2 e1, uint iters)
{
    vec2 xy[16];
    float x[16], y[16];
    for (uint i = 0u; i < 16u; i++) { xy[i] = px[i].rg; x[i] = px[i].r; y[i] = px[i].g; }

    const LineResult2 a = refineLine256W16v2(xy, e0, e1, iters);
    const LineResult1 bx = refineLine256W4v1(x, e0.x, e1.x, iters);
    const LineResult1 by = refineLine256W4v1(y, e0.y, e1.y, iters);

    uint codes[8]; // L0 L1 A0 A1
    codes[4] = codes[5] = codes[6] = codes[7] = 0u;
    if (bx.sse + by.sse < a.sse) {
        codes[0] = quantCode256(bx.d0); codes[1] = quantCode256(bx.d1);
        codes[2] = quantCode256(by.d0); codes[3] = quantCode256(by.d1);
        return packDualPlane(ANBC_ASTC_MODE_W2, ANBC_CEM_LA, 3u, codes, 4u, 8u, bx.idx, by.idx, 2u);
    }
    codes[0] = quantCode256(a.d0.x); codes[1] = quantCode256(a.d1.x);
    codes[2] = quantCode256(a.d0.y); codes[3] = quantCode256(a.d1.y);
    return packSinglePlane(ANBC_ASTC_MODE_W4, ANBC_CEM_LA, codes, 4u, 8u, a.idx, 4u);
}

// ---------------------------------------------------------------------------
// FLOAT: HDR RGB, CEM 11, W3, QUANT_256. px in the [0,1] LNS domain
// (lnsNorm), e0/e1 likewise.
// ---------------------------------------------------------------------------

uvec4 encodeAstcFloat(in vec3 px[16], vec3 e0, vec3 e1, uint iters)
{
    const HdrLineResult r = refineHdr(px, e0, e1, iters);
    uint idx[16];
    for (uint i = 0u; i < 16u; i++)
        idx[i] = r.packed.swapped ? 7u - r.idx[i] : r.idx[i];
    uint codes[8];
    for (uint i = 0u; i < 6u; i++)
        codes[i] = r.packed.v[i];
    codes[6] = codes[7] = 0u;
    return packSinglePlane(ANBC_ASTC_MODE_W3, ANBC_CEM_HDR_RGB, codes, 6u, 8u, idx, 3u);
}

#endif // ANBC_ASTC_COMMON_GLSL
