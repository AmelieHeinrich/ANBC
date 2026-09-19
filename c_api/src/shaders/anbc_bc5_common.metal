// BC5 encoder -- refinement and packing (prepended to the kernel, see
// CMakeLists.txt). Needs anbc_common.metal first.
//
// Mirrors src/bc5_codec.py (encode_bc5_blocks). There is no network for
// BC5: a BC4 line is 1-D and the block's own min/max is already a
// near-optimal endpoint pair (min/max + 2 refinement rounds scores ~56 dB on
// the normal-map set vs ~53 dB for Compressonator; the BC5 MLP that was
// tried scored ~46 dB because its too-wide endpoint guesses collapsed
// low-contrast blocks to their mean).
// BC5 = two independent BC4 lines, R then G, 8 bytes each:
//   byte 0/1   endpoint0 / endpoint1 (8-bit)
//   byte 2..7  16 x 3-bit palette indices, LSB-first in pixel order
// With e0 > e1 the 8-entry palette, in *index* order, is
//   e0, e1, (6e0+e1)/7, ..., (e0+6e1)/7
// i.e. uniform levels k/7 whose bitstream index is 0 -> 0, 7 -> 1, else k+1.
// e0 <= e1 would select the 6-entry + {0,255} palette, which we never emit.
//
// Per line, starting from the block's min/max, we alternate
//   level_i = round(7 * clamp((p_i - e0) / (e1 - e0), 0, 1))   (exact index)
//   e0, e1  = least-squares refit for those levels, snapped to 8 bits
// P.refineIters times, exactly like the Python path.

#ifndef ANBC_BC5_COMMON_METAL
#define ANBC_BC5_COMMON_METAL

#include "anbc_common.metal"

constant uint kBc4LevelToIndex[8] = { 0, 2, 3, 4, 5, 6, 7, 1 };
// Index remap for t -> 1 - t when the endpoints get swapped.
constant uint kBc4SwapIndex[8] = { 1, 0, 7, 6, 5, 4, 3, 2 };

static inline uint quant8(float v) { return uint(clamp(round(v * 255.0f), 0.0f, 255.0f)); }
static inline float dec8(uint v) { return float(v) / 255.0f; }

// Nearest of the 8 uniform levels for pixel p on the line e0 -> e1 (t = 0
// when the endpoints coincide), as a level 0..7.
static inline uint bc4Level(float p, float e0, float e1)
{
    const float span = e1 - e0;
    if (abs(span) <= 1e-8f)
        return 0;
    return uint(round(clamp((p - e0) / span, 0.0f, 1.0f) * 7.0f));
}

struct Bc4Result { uint e0, e1; uint idx[16]; float sse; };

static Bc4Result refineBc4(thread const float* px, float e0, float e1, uint iters)
{
    Bc4Result r;
    r.e0 = quant8(e0);
    r.e1 = quant8(e1);
    float d0 = dec8(r.e0), d1 = dec8(r.e1);
    float w[16];
    for (uint it = 0; it < iters; it++) {
        for (uint i = 0; i < 16; i++)
            w[i] = float(bc4Level(px[i], d0, d1)) / 7.0f;
        lsEndpoints(px, w, e0, e1);
        r.e0 = quant8(e0);
        r.e1 = quant8(e1);
        d0 = dec8(r.e0);
        d1 = dec8(r.e1);
    }
    r.sse = 0.0f;
    for (uint i = 0; i < 16; i++) {
        const uint level = bc4Level(px[i], d0, d1);
        r.idx[i] = kBc4LevelToIndex[level];
        const float d = mix(d0, d1, float(level) / 7.0f) - px[i];
        r.sse += d * d;
    }
    return r;
}

// One BC4 block as two 32-bit words: [e0 | e1 << 8 | idx bits...].
static void packBc4(Bc4Result r, thread uint2& words)
{
    // The 8-entry palette needs e0 > e1: swap (t -> 1 - t) where needed.
    // Equal endpoints would select the 6-entry palette whose indices 6/7
    // decode to 0/255, so those blocks get index 0 everywhere (= e0 in
    // either palette).
    if (r.e0 < r.e1) {
        const uint t = r.e0; r.e0 = r.e1; r.e1 = t;
        for (uint i = 0; i < 16; i++)
            r.idx[i] = kBc4SwapIndex[r.idx[i]];
    }
    if (r.e0 == r.e1) {
        for (uint i = 0; i < 16; i++)
            r.idx[i] = 0;
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, r.e0, 8);
    bwPut(b, r.e1, 8);
    for (uint i = 0; i < 16; i++)
        bwPut(b, r.idx[i], 3);
    words = uint2(b.words[0], b.words[1]);
}

// One channel: min/max start, refine, pack into two 32-bit words.
static uint2 encodeBc4Line(thread const float* v, uint iters)
{
    float lo = v[0], hi = v[0];
    for (uint i = 1; i < 16; i++) {
        lo = min(lo, v[i]);
        hi = max(hi, v[i]);
    }
    uint2 words;
    packBc4(refineBc4(v, lo, hi, iters), words);
    return words;
}

// A 4x4 RGBA block -> packed BC5 block (R line, then G line).
static uint4 encodeBlockBc5(thread const float4* px, constant EncodeParams& P)
{
    float r[16], g[16];
    for (uint i = 0; i < 16; i++) { r[i] = px[i].r; g[i] = px[i].g; }
    const uint2 wr = encodeBc4Line(r, P.refineIters);
    const uint2 wg = encodeBc4Line(g, P.refineIters);
    return uint4(wr.x, wr.y, wg.x, wg.y);
}

#endif // ANBC_BC5_COMMON_METAL
