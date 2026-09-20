// BC5 encoder -- refinement and packing (GLSL port of anbc_bc5_common.metal).
// Needs anbc_common.glsl first.
//
// Mirrors src/bc5_codec.py (encode_bc5_blocks). No network: a BC4 line is
// 1-D and the block's own min/max is already a near-optimal endpoint pair.
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

#ifndef ANBC_BC5_COMMON_GLSL
#define ANBC_BC5_COMMON_GLSL

#include "anbc_common.glsl"

const uint kBc4LevelToIndex[8] = uint[8](0u, 2u, 3u, 4u, 5u, 6u, 7u, 1u);
// Index remap for t -> 1 - t when the endpoints get swapped.
const uint kBc4SwapIndex[8] = uint[8](1u, 0u, 7u, 6u, 5u, 4u, 3u, 2u);

uint quant8(float v) { return uint(clamp(round(v * 255.0), 0.0, 255.0)); }
float dec8(uint v) { return float(v) / 255.0; }

// Nearest of the 8 uniform levels for pixel p on the line e0 -> e1 (t = 0
// when the endpoints coincide), as a level 0..7.
uint bc4Level(float p, float e0, float e1)
{
    const float span = e1 - e0;
    if (abs(span) <= 1e-8)
        return 0u;
    return uint(round(clamp((p - e0) / span, 0.0, 1.0) * 7.0));
}

struct Bc4Result { uint e0, e1; uint idx[16]; float sse; };

Bc4Result refineBc4(in float px[16], float e0, float e1, uint iters)
{
    Bc4Result r;
    r.e0 = quant8(e0);
    r.e1 = quant8(e1);
    float d0 = dec8(r.e0), d1 = dec8(r.e1);
    float w[16];
    for (uint it = 0u; it < iters; it++) {
        for (uint i = 0u; i < 16u; i++)
            w[i] = float(bc4Level(px[i], d0, d1)) / 7.0;
        lsEndpoints(px, w, e0, e1);
        r.e0 = quant8(e0);
        r.e1 = quant8(e1);
        d0 = dec8(r.e0);
        d1 = dec8(r.e1);
    }
    r.sse = 0.0;
    for (uint i = 0u; i < 16u; i++) {
        const uint level = bc4Level(px[i], d0, d1);
        r.idx[i] = kBc4LevelToIndex[level];
        const float d = mix(d0, d1, float(level) / 7.0) - px[i];
        r.sse += d * d;
    }
    return r;
}

// One BC4 block as two 32-bit words: [e0 | e1 << 8 | idx bits...].
uvec2 packBc4(Bc4Result r)
{
    // The 8-entry palette needs e0 > e1: swap (t -> 1 - t) where needed.
    // Equal endpoints would select the 6-entry palette whose indices 6/7
    // decode to 0/255, so those blocks get index 0 everywhere (= e0 in
    // either palette).
    if (r.e0 < r.e1) {
        const uint t = r.e0; r.e0 = r.e1; r.e1 = t;
        for (uint i = 0u; i < 16u; i++)
            r.idx[i] = kBc4SwapIndex[r.idx[i]];
    }
    if (r.e0 == r.e1) {
        for (uint i = 0u; i < 16u; i++)
            r.idx[i] = 0u;
    }
    BitWriter b;
    bwInit(b);
    bwPut(b, r.e0, 8u);
    bwPut(b, r.e1, 8u);
    for (uint i = 0u; i < 16u; i++)
        bwPut(b, r.idx[i], 3u);
    return uvec2(b.words[0], b.words[1]);
}

// One channel: min/max start, refine, pack into two 32-bit words.
uvec2 encodeBc4Line(in float v[16], uint iters)
{
    float lo = v[0], hi = v[0];
    for (uint i = 1u; i < 16u; i++) {
        lo = min(lo, v[i]);
        hi = max(hi, v[i]);
    }
    return packBc4(refineBc4(v, lo, hi, iters));
}

// A 4x4 RGBA block -> packed BC5 block (R line, then G line).
uvec4 encodeBlockBc5(in vec4 px[16])
{
    float r[16], g[16];
    for (uint i = 0u; i < 16u; i++) { r[i] = px[i].r; g[i] = px[i].g; }
    const uvec2 wr = encodeBc4Line(r, P.refineIters);
    const uvec2 wg = encodeBc4Line(g, P.refineIters);
    return uvec4(wr.x, wr.y, wg.x, wg.y);
}

#endif // ANBC_BC5_COMMON_GLSL
