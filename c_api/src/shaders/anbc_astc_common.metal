// ASTC 4x4 encoder -- shared part (prepended to the kernels, see
// CMakeLists.txt). Needs anbc_common.metal and anbc_astc_tables.metal first.
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

#ifndef ANBC_ASTC_COMMON_METAL
#define ANBC_ASTC_COMMON_METAL

#include "anbc_common.metal"
#include "anbc_astc_tables.metal"

#define ANBC_LNS_MAX 65535.0f
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

// QUANT levels as template parameters: 256 (8 bits, identity), 192 and 48
// (trit ranges, table driven). quantValue -> the ISE code; decodeCode -> the
// 8-bit value the decoder reconstructs from it.
template <uint Q> static inline uint quantCode(float v);
template <uint Q> static inline float decodeCode(uint code);

template <> inline uint quantCode<256>(float v) { return uint(clamp(round(v * 255.0f), 0.0f, 255.0f)); }
template <> inline float decodeCode<256>(uint code) { return float(code) / 255.0f; }
template <> inline uint quantCode<192>(float v) { return kAstcCodeOfValue192[uint(clamp(round(v * 255.0f), 0.0f, 255.0f))]; }
template <> inline float decodeCode<192>(uint code) { return float(kAstcUnquant192[code]) / 255.0f; }
template <> inline uint quantCode<48>(float v) { return kAstcCodeOfValue48[uint(clamp(round(v * 255.0f), 0.0f, 255.0f))]; }
template <> inline float decodeCode<48>(uint code) { return float(kAstcUnquant48[code]) / 255.0f; }

// Decoded value of a [0,1] endpoint after quantization, per channel.
template <uint Q> static inline float quantDec(float v) { return decodeCode<Q>(quantCode<Q>(v)); }
template <uint Q> static inline float2 quantDec(float2 v) { return float2(quantDec<Q>(v.x), quantDec<Q>(v.y)); }
template <uint Q> static inline float3 quantDec(float3 v) { return float3(quantDec<Q>(v.x), quantDec<Q>(v.y), quantDec<Q>(v.z)); }
template <uint Q> static inline float4 quantDec(float4 v) { return float4(quantDec<Q>(v.x), quantDec<Q>(v.y), quantDec<Q>(v.z), quantDec<Q>(v.w)); }

// ---------------------------------------------------------------------------
// Index snapping + one refined line (LDR, per-channel quantization)
// ---------------------------------------------------------------------------

static inline uint nearestWeight(float t, constant float* weights, uint count)
{
    uint best = 0;
    float bestD = 2.0f;
    for (uint k = 0; k < count; k++) {
        const float d = abs(t - weights[k]);
        if (d < bestD) { bestD = d; best = k; }
    }
    return best;
}

template <typename V>
struct LineResult { V d0, d1; uint idx[16]; float sse; };

// bc7's refineMode6 for any channel count / weight table / quant level:
// returns decoded endpoints, exact indices and the squared error.
template <uint Q, typename V>
static LineResult<V> refineLine(thread const V* px, V e0, V e1, constant float* weights, uint count, uint iters)
{
    LineResult<V> r;
    r.d0 = quantDec<Q>(e0);
    r.d1 = quantDec<Q>(e1);
    float w[16];
    for (uint it = 0; it < iters; it++) {
        for (uint i = 0; i < 16; i++)
            w[i] = weights[nearestWeight(projectT(px[i], r.d0, r.d1), weights, count)];
        lsEndpoints(px, w, e0, e1);
        r.d0 = quantDec<Q>(e0);
        r.d1 = quantDec<Q>(e1);
    }
    r.sse = 0.0f;
    for (uint i = 0; i < 16; i++) {
        r.idx[i] = nearestWeight(projectT(px[i], r.d0, r.d1), weights, count);
        const V d = mix(r.d0, r.d1, weights[r.idx[i]]) - px[i];
        r.sse += vdot(d, d);
    }
    return r;
}

// ---------------------------------------------------------------------------
// HDR: LNS domain + CEM 11 endpoint packing (port of astc_codec.py
// pack_hdr_endpoints / unpack_hdr_endpoints, themselves astcenc's
// quantize_hdr_rgb with QUANT_256 and hdr_rgb_unpack)
// ---------------------------------------------------------------------------

// half float -> 16-bit LNS (the smallest LNS value the decoder maps back to
// this half; exact inverse of lns_to_sf16). Negatives clamp to 0.
static inline uint halfToLns(float v)
{
    ushort bits = as_type<ushort>(half(v));
    if (bits & 0x8000u)
        bits = 0;
    bits = min(bits, ushort(0x7BFF));
    const uint ec = bits >> 10;
    const uint t = (bits & 0x3FFu) * 8u;
    uint mc = (t + 2u) / 3u;                 // ceil(t / 3)
    if (mc >= 512u) mc = (t + 512u + 3u) / 4u;
    if (mc >= 1536u) mc = (t + 2048u + 4u) / 5u;
    return (ec << 11) | mc;
}

static inline float lnsNorm(float v) { return float(halfToLns(v)) / ANBC_LNS_MAX; }

// astcenc flt2int_rtn: (int)(x + 0.5), truncation toward zero.
static inline int rtn(float x) { return int(trunc(x + 0.5f)); }

struct HdrPacked { uint v[6]; bool swapped; };

constant int kHdrModeBits[8][4] = { {9, 7, 6, 7}, {9, 8, 6, 6}, {10, 6, 7, 7}, {10, 7, 7, 6},
                                    {11, 8, 6, 5}, {11, 6, 8, 6}, {12, 7, 7, 5}, {12, 6, 7, 6} };
constant float kHdrModeCutoffs[8][3] = { {16384, 8192, 8192}, {32768, 8192, 4096}, {4096, 8192, 4096}, {8192, 8192, 2048},
                                         {8192, 2048, 512}, {2048, 8192, 1024}, {2048, 2048, 256}, {1024, 2048, 512} };
constant int kHdrModeShift[8] = { 7, 7, 6, 6, 5, 5, 4, 4 };
constant int kHdrDbits[8] = { 7, 6, 7, 6, 5, 6, 5, 6 };

// e0/e1: LNS colours in [0, 65535]. Stores e1's major component and
// non-negative deltas down to e0, so a block whose e0 is brighter on the
// major axis is stored swapped (the caller inverts the weights).
static HdrPacked packHdrEndpoints(float3 e0, float3 e1)
{
    HdrPacked out;
    float3 c0 = clamp(e0, 0.0f, float(ANBC_LNS_MAX_FINITE));
    float3 c1 = clamp(e1, 0.0f, float(ANBC_LNS_MAX_FINITE));
    const float3 hi = max(c0, c1);
    const uint majcomp = (hi.x >= hi.y && hi.x >= hi.z) ? 0u : (hi.y >= hi.z ? 1u : 2u);
    out.swapped = c0[majcomp] > c1[majcomp];
    if (out.swapped) { const float3 t = c0; c0 = c1; c1 = t; }
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
        const float scale = 1.0f / float(1 << sh);
        const int bCut = 1 << kHdrModeBits[mode][1];
        const int cCut = 1 << kHdrModeBits[mode][2];
        const int dCut = 1 << (kHdrModeBits[mode][3] - 1);

        const int aInt = rtn(aBase * scale);
        const float aF = float(aInt << sh);
        const int cInt = rtn(clamp(aF - c0.x, 0.0f, 65535.0f) * scale);
        if (cInt >= cCut)
            continue;
        const float cF = float(cInt << sh);
        const int b0Int = rtn(clamp(aF - c1.y, 0.0f, 65535.0f) * scale);
        const int b1Int = rtn(clamp(aF - c1.z, 0.0f, 65535.0f) * scale);
        if (b0Int >= bCut || b1Int >= bCut)
            continue;
        const float b0F = float(b0Int << sh), b1F = float(b1Int << sh);
        const int d0Int = rtn(clamp(aF - b0F - cF - c0.y, -65535.0f, 65535.0f) * scale);
        const int d1Int = rtn(clamp(aF - b1F - cF - c0.z, -65535.0f, 65535.0f) * scale);
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

        out.v[0] = uint(aInt & 0xFF);
        out.v[1] = uint((cInt & 0x3F) | ((mode & 1) << 7) | ((aInt & 0x100) >> 2));
        out.v[2] = uint((b0Int & 0x3F) | (bit0 << 6) | (((mode >> 1) & 1) << 7));
        out.v[3] = uint((b1Int & 0x3F) | (bit1 << 6) | (((mode >> 2) & 1) << 7));
        out.v[4] = uint((d0Int & 0x1F) | (bit2 << 6) | (bit4 << 5) | (int(majcomp & 1u) << 7));
        out.v[5] = uint((d1Int & 0x1F) | (bit3 << 6) | (bit5 << 5) | (int((majcomp >> 1) & 1u) << 7));
        return out;
    }
    // Flat fallback (majcomp 3): 8 bits for R and G, 7 for B.
    const float3 f0 = clamp(e0, 0.0f, 65020.0f), f1 = clamp(e1, 0.0f, 65020.0f);
    out.swapped = false;
    out.v[0] = uint(rtn(f0.x / 256.0f)); out.v[1] = uint(rtn(f1.x / 256.0f));
    out.v[2] = uint(rtn(f0.y / 256.0f)); out.v[3] = uint(rtn(f1.y / 256.0f));
    out.v[4] = uint(rtn(f0.z / 512.0f) + 128); out.v[5] = uint(rtn(f1.z / 512.0f) + 128);
    return out;
}

// The 16-bit LNS endpoints a decoder reconstructs from six CEM 11 values.
static void unpackHdrEndpoints(thread const uint* v, thread float3& e0, thread float3& e1)
{
    const int v0 = v[0], v1 = v[1], v2 = v[2], v3 = v[3], v4 = v[4], v5 = v[5];
    const int modeval = ((v1 >> 7) & 1) | (((v2 >> 7) & 1) << 1) | (((v3 >> 7) & 1) << 2);
    const int majcomp = ((v4 >> 7) & 1) | (((v5 >> 7) & 1) << 1);
    if (majcomp == 3) {
        e0 = float3(v0 << 8, v2 << 8, (v4 & 0x7F) << 9);
        e1 = float3(v1 << 8, v3 << 8, (v5 & 0x7F) << 9);
        return;
    }
    int a = v0 | ((v1 & 0x40) << 2);
    int b0 = v2 & 0x3F, b1 = v3 & 0x3F, c = v1 & 0x3F, d0 = v4 & 0x7F, d1 = v5 & 0x7F;
    const int dbits = kHdrDbits[modeval];
    const int bit0 = (v2 >> 6) & 1, bit1 = (v3 >> 6) & 1, bit2 = (v4 >> 6) & 1, bit3 = (v5 >> 6) & 1;
    const int bit4 = (v4 >> 5) & 1, bit5 = (v5 >> 5) & 1;
    const int ohmod = 1 << modeval;
    if (ohmod & 0xA4) a |= bit0 << 9;
    if (ohmod & 0x8) a |= bit2 << 9;
    if (ohmod & 0x50) a |= bit4 << 9;
    if (ohmod & 0x50) a |= bit5 << 10;
    if (ohmod & 0xA0) a |= bit1 << 10;
    if (ohmod & 0xC0) a |= bit2 << 11;
    if (ohmod & 0x4) c |= bit1 << 6;
    if (ohmod & 0xE8) c |= bit3 << 6;
    if (ohmod & 0x20) c |= bit2 << 7;
    if (ohmod & 0x5B) { b0 |= bit0 << 6; b1 |= bit1 << 6; }
    if (ohmod & 0x12) { b0 |= bit2 << 7; b1 |= bit3 << 7; }
    if (ohmod & 0xAF) { d0 |= bit4 << 5; d1 |= bit5 << 5; }
    if (ohmod & 0x5) { d0 |= bit2 << 6; d1 |= bit3 << 6; }
    // sign-extend d0/d1 from dbits (higher bits are other fields)
    const int sign = 1 << (dbits - 1);
    d0 &= (sign << 1) - 1; d1 &= (sign << 1) - 1;
    if (d0 & sign) d0 -= sign << 1;
    if (d1 & sign) d1 -= sign << 1;
    const int shamt = (modeval >> 1) ^ 3;
    a <<= shamt; b0 <<= shamt; b1 <<= shamt; c <<= shamt; d0 <<= shamt; d1 <<= shamt;
    int3 r0 = clamp(int3(a - c, a - b0 - c - d0, a - b1 - c - d1), 0, 4095);
    int3 r1 = clamp(int3(a, a - b0, a - b1), 0, 4095);
    if (majcomp == 1) { r0 = r0.yxz; r1 = r1.yxz; }
    else if (majcomp == 2) { r0 = r0.zyx; r1 = r1.zyx; }
    e0 = float3(r0 << 4);
    e1 = float3(r1 << 4);
}

struct HdrLineResult { HdrPacked packed; float3 d0, d1; uint idx[16]; float sse; };

// The CEM 11 quantizer is joint over both endpoints and not a fixed point
// of itself (decoding can tie two channels for the major component), so
// the packed values of the last quantization are kept for the packer.
static void quantHdr(float3 e0, float3 e1, thread HdrLineResult& r)
{
    r.packed = packHdrEndpoints(e0 * ANBC_LNS_MAX, e1 * ANBC_LNS_MAX);
    float3 d0, d1;
    unpackHdrEndpoints(r.packed.v, d0, d1);
    if (r.packed.swapped) { const float3 t = d0; d0 = d1; d1 = t; }
    r.d0 = d0 / ANBC_LNS_MAX;
    r.d1 = d1 / ANBC_LNS_MAX;
}

static HdrLineResult refineHdr(thread const float3* px, float3 e0, float3 e1, constant float* weights, uint count, uint iters)
{
    HdrLineResult r;
    quantHdr(e0, e1, r);
    float w[16];
    for (uint it = 0; it < iters; it++) {
        for (uint i = 0; i < 16; i++)
            w[i] = weights[nearestWeight(projectT(px[i], r.d0, r.d1), weights, count)];
        lsEndpoints(px, w, e0, e1);
        quantHdr(e0, e1, r);
    }
    r.sse = 0.0f;
    for (uint i = 0; i < 16; i++) {
        r.idx[i] = nearestWeight(projectT(px[i], r.d0, r.d1), weights, count);
        const float3 d = mix(r.d0, r.d1, weights[r.idx[i]]) - px[i];
        r.sse += dot(d, d);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Bit packing
// ---------------------------------------------------------------------------

// Trit ISE block layout: T bits following each of the 5 elements.
constant uint kTBits[5] = { 2, 2, 1, 2, 1 };
constant uint kTShift[5] = { 0, 2, 4, 5, 7 };

// Writes `count` ISE codes (bits-only or trit ranges) to b, LSB-first.
template <uint Q>
static void iseWrite(thread BitWriter& b, thread const uint* codes, uint count)
{
    constexpr uint bits = (Q == 256) ? 8 : (Q == 192) ? 6 : 4;
    if (Q == 256) {
        for (uint i = 0; i < count; i++)
            bwPut(b, codes[i], bits);
        return;
    }
    for (uint start = 0; start < count; start += 5) {
        const uint n = min(5u, count - start);
        uint index = 0;
        for (uint j = 0; j < 5; j++) { // t4*81 + t3*27 + t2*9 + t1*3 + t0
            const uint trit = (j < n) ? (codes[start + j] >> bits) : 0u;
            index += trit * (j == 0 ? 1u : j == 1 ? 3u : j == 2 ? 9u : j == 3 ? 27u : 81u);
        }
        const uint T = kAstcIntegerOfTrits[index];
        for (uint j = 0; j < n; j++) {
            bwPut(b, codes[start + j] & ((1u << bits) - 1u), bits);
            bwPut(b, (T >> kTShift[j]) & ((1u << kTBits[j]) - 1u), kTBits[j]);
        }
    }
}

// Header + endpoints go in from bit 0 (b), the weight stream (wb, `wbits`
// long) is bit-reversed into the top of the block.
static uint4 assembleBlock(thread BitWriter& b, thread const BitWriter& wb)
{
    uint4 out = uint4(b.words[0], b.words[1], b.words[2], b.words[3]);
    out.w |= reverse_bits(wb.words[0]);
    out.z |= reverse_bits(wb.words[1]);
    out.y |= reverse_bits(wb.words[2]);
    out.x |= reverse_bits(wb.words[3]);
    return out;
}

static inline void writeHeader(thread BitWriter& b, uint blockMode, uint cem)
{
    bwInit(b);
    bwPut(b, blockMode, 11);
    bwPut(b, 0, 2); // single partition
    bwPut(b, cem, 4);
}

// Single-plane LDR block: values are v0[c], v1[c] interleaved per channel.
template <uint Q, uint WBITS>
static uint4 packSinglePlane(uint blockMode, uint cem, thread const uint* codes, uint nValues, thread const uint* idx)
{
    BitWriter b, wb;
    writeHeader(b, blockMode, cem);
    iseWrite<Q>(b, codes, nValues);
    bwInit(wb);
    for (uint i = 0; i < 16; i++)
        bwPut(wb, idx[i], WBITS);
    return assembleBlock(b, wb);
}

// Dual-plane LDR block: plane 1 / plane 2 weights interleaved per texel,
// CCS just below the weight stream (bit 128 - 32 * WBITS - 2).
template <uint Q, uint WBITS>
static uint4 packDualPlane(uint blockMode, uint cem, uint ccs, thread const uint* codes, uint nValues,
                           thread const uint* idx1, thread const uint* idx2)
{
    BitWriter b, wb;
    writeHeader(b, blockMode | ANBC_ASTC_MODE_DUAL, cem);
    iseWrite<Q>(b, codes, nValues);
    b.pos = 128 - 32 * WBITS - 2;
    bwPut(b, ccs, 2);
    bwInit(wb);
    for (uint i = 0; i < 16; i++) {
        bwPut(wb, idx1[i], WBITS);
        bwPut(wb, idx2[i], WBITS);
    }
    return assembleBlock(b, wb);
}

// CEM 8/12 decoders swap the endpoints (and blue-contract) when
// sum(rgb0) > sum(rgb1): keep sum(rgb0) <= sum(rgb1) by swapping ourselves
// and inverting the weights (every weight table is symmetric).
static inline bool rgbSwapNeeded(float3 d0, float3 d1)
{
    // d0/d1 are decoded values, so round(d * 255) is exactly the stored 8-bit value
    const float3 y0 = round(d0 * 255.0f), y1 = round(d1 * 255.0f);
    return (y0.x + y0.y + y0.z) > (y1.x + y1.y + y1.z);
}

// ---------------------------------------------------------------------------
// GAME: RGBA8, configs A (CEM 8, W4, Q192), A3 (CEM 8, W3, Q256), B (CEM 12
// dual plane on alpha, W2, Q48). e0/e1: initial RGBA endpoints.
// ---------------------------------------------------------------------------

static uint4 encodeAstcGame(thread const float4* px, float4 e0, float4 e1, uint iters)
{
    float3 rgb[16];
    float alpha[16];
    float alphaSse = 0.0f; // error of the not-stored alpha (decodes as 1) for the CEM 8 configs
    for (uint i = 0; i < 16; i++) {
        rgb[i] = px[i].rgb;
        alpha[i] = px[i].a;
        alphaSse += (1.0f - px[i].a) * (1.0f - px[i].a);
    }

    const LineResult<float3> a = refineLine<192>(rgb, e0.rgb, e1.rgb, kAstcWeights16, 16, iters);
    const LineResult<float3> a3 = refineLine<256>(rgb, e0.rgb, e1.rgb, kAstcWeights8, 8, iters);
    const LineResult<float3> bRgb = refineLine<48>(rgb, e0.rgb, e1.rgb, kAstcWeights4, 4, iters);
    const LineResult<float> bA = refineLine<48>(alpha, e0.a, e1.a, kAstcWeights4, 4, iters);

    const float sseA = a.sse + alphaSse, sseA3 = a3.sse + alphaSse, sseB = bRgb.sse + bA.sse;
    uint codes[8];
    uint idx[16], idx2[16];
    if (sseB < sseA && sseB < sseA3) {
        float3 d0 = bRgb.d0, d1 = bRgb.d1;
        float a0 = bA.d0, a1 = bA.d1;
        const bool swap = rgbSwapNeeded(d0, d1);
        if (swap) { const float3 t = d0; d0 = d1; d1 = t; const float ta = a0; a0 = a1; a1 = ta; }
        for (uint i = 0; i < 16; i++) {
            idx[i] = swap ? 3 - bRgb.idx[i] : bRgb.idx[i];
            idx2[i] = swap ? 3 - bA.idx[i] : bA.idx[i];
        }
        codes[0] = quantCode<48>(d0.x); codes[1] = quantCode<48>(d1.x);
        codes[2] = quantCode<48>(d0.y); codes[3] = quantCode<48>(d1.y);
        codes[4] = quantCode<48>(d0.z); codes[5] = quantCode<48>(d1.z);
        codes[6] = quantCode<48>(a0);   codes[7] = quantCode<48>(a1);
        return packDualPlane<48, 2>(ANBC_ASTC_MODE_W2, ANBC_CEM_RGBA, 3, codes, 8, idx, idx2);
    }
    if (sseA3 < sseA) {
        float3 d0 = a3.d0, d1 = a3.d1;
        const bool swap = rgbSwapNeeded(d0, d1);
        if (swap) { const float3 t = d0; d0 = d1; d1 = t; }
        for (uint i = 0; i < 16; i++)
            idx[i] = swap ? 7 - a3.idx[i] : a3.idx[i];
        codes[0] = quantCode<256>(d0.x); codes[1] = quantCode<256>(d1.x);
        codes[2] = quantCode<256>(d0.y); codes[3] = quantCode<256>(d1.y);
        codes[4] = quantCode<256>(d0.z); codes[5] = quantCode<256>(d1.z);
        return packSinglePlane<256, 3>(ANBC_ASTC_MODE_W3, ANBC_CEM_RGB, codes, 6, idx);
    }
    float3 d0 = a.d0, d1 = a.d1;
    const bool swap = rgbSwapNeeded(d0, d1);
    if (swap) { const float3 t = d0; d0 = d1; d1 = t; }
    for (uint i = 0; i < 16; i++)
        idx[i] = swap ? 15 - a.idx[i] : a.idx[i];
    codes[0] = quantCode<192>(d0.x); codes[1] = quantCode<192>(d1.x);
    codes[2] = quantCode<192>(d0.y); codes[3] = quantCode<192>(d1.y);
    codes[4] = quantCode<192>(d0.z); codes[5] = quantCode<192>(d1.z);
    return packSinglePlane<192, 4>(ANBC_ASTC_MODE_W4, ANBC_CEM_RGB, codes, 6, idx);
}

// ---------------------------------------------------------------------------
// NORMAL: (X, Y) from the source's R, G -> CEM 4 luminance + alpha (sample
// .ra). Configs A (W4, one plane), B (W2, two planes). e0/e1: initial (X,Y).
// ---------------------------------------------------------------------------

static uint4 encodeAstcNormal(thread const float4* px, float2 e0, float2 e1, uint iters)
{
    float2 xy[16];
    float x[16], y[16];
    for (uint i = 0; i < 16; i++) { xy[i] = px[i].rg; x[i] = px[i].r; y[i] = px[i].g; }

    const LineResult<float2> a = refineLine<256>(xy, e0, e1, kAstcWeights16, 16, iters);
    const LineResult<float> bx = refineLine<256>(x, e0.x, e1.x, kAstcWeights4, 4, iters);
    const LineResult<float> by = refineLine<256>(y, e0.y, e1.y, kAstcWeights4, 4, iters);

    uint codes[4]; // L0 L1 A0 A1
    if (bx.sse + by.sse < a.sse) {
        codes[0] = quantCode<256>(bx.d0); codes[1] = quantCode<256>(bx.d1);
        codes[2] = quantCode<256>(by.d0); codes[3] = quantCode<256>(by.d1);
        return packDualPlane<256, 2>(ANBC_ASTC_MODE_W2, ANBC_CEM_LA, 3, codes, 4, bx.idx, by.idx);
    }
    codes[0] = quantCode<256>(a.d0.x); codes[1] = quantCode<256>(a.d1.x);
    codes[2] = quantCode<256>(a.d0.y); codes[3] = quantCode<256>(a.d1.y);
    return packSinglePlane<256, 4>(ANBC_ASTC_MODE_W4, ANBC_CEM_LA, codes, 4, a.idx);
}

// ---------------------------------------------------------------------------
// FLOAT: HDR RGB, CEM 11, W3, QUANT_256. px in the [0,1] LNS domain
// (lnsNorm), e0/e1 likewise.
// ---------------------------------------------------------------------------

static uint4 encodeAstcFloat(thread const float3* px, float3 e0, float3 e1, uint iters)
{
    const HdrLineResult r = refineHdr(px, e0, e1, kAstcWeights8, 8, iters);
    uint idx[16];
    for (uint i = 0; i < 16; i++)
        idx[i] = r.packed.swapped ? 7 - r.idx[i] : r.idx[i];
    uint codes[6];
    for (uint i = 0; i < 6; i++)
        codes[i] = r.packed.v[i];
    return packSinglePlane<256, 3>(ANBC_ASTC_MODE_W3, ANBC_CEM_HDR_RGB, codes, 6, idx);
}

#endif // ANBC_ASTC_COMMON_METAL
