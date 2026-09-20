// Single-dispatch mip chain generator (FidelityFX SPD pattern).
//
// Each 256-thread threadgroup owns a 64x64 tile of mip 0 and produces mips 1..6
// for it: mips 1-2 in registers, 3-6 through a 16x16 threadgroup array. The
// last threadgroup to finish (atomic counter) then treats mip 6 (<= 64x64 for
// textures up to 4096^2) as a single tile and produces mips 7..12 the same way.
//
// Filter is a 2x2 box, edge-clamped *at every level* (an odd-sized level's
// last row/column is reused for the missing one, like the CPU references in
// anbc_compare / compare_dds.py). The in-register and threadgroup stages hold
// phantom texels past a level's edge (computed from source texels clamped one
// level up, which is not the same thing), so those reads clamp explicitly.
// With kSRGB the RGB channels are decoded to linear before averaging and
// re-encoded on store; alpha is always linear.

#include <metal_stdlib>
using namespace metal;

constant bool kSRGB [[function_constant(1)]];

#define ANBC_MIPS_MAX_LEVELS 13
#define ANBC_MIPS_TILE 64

struct MipParams {
    uint  numMips;
    uint  numGroups;   // threadgroups in phase 1 (tiles of mip 0)
    uint  pad0, pad1;
    uint4 dims[ANBC_MIPS_MAX_LEVELS]; // .xy = size of each level
};

static inline float3 srgbToLinear(float3 c)
{
    return select(pow((c + 0.055f) / 1.055f, 2.4f), c / 12.92f, c <= 0.04045f);
}

static inline float3 linearToSrgb(float3 c)
{
    return select(1.055f * pow(c, 1.0f / 2.4f) - 0.055f, 12.92f * c, c <= 0.0031308f);
}

static inline float4 loadTexel(texture2d<float, access::read> src, uint lod, uint2 p, uint2 dims)
{
    p = min(p, dims - 1);
    float4 c = src.read(p, lod);
    if (kSRGB)
        c.rgb = srgbToLinear(c.rgb);
    return c;
}

static inline void storeTexel(texture2d<float, access::write> dst, uint2 p, uint2 dims, float4 c)
{
    if (p.x >= dims.x || p.y >= dims.y)
        return;
    if (kSRGB)
        c.rgb = linearToSrgb(c.rgb);
    dst.write(c, p);
}

// Reduce the 64x64 tile at `tileOrigin` (in level `srcLod` texels) into levels
// srcLod+1 .. srcLod+6, writing into outs[level-1]. `tileIdx` is the tile's
// coordinate in the tile grid, which is also its texel coordinate at level
// srcLod+6. All early returns are on group-uniform conditions.
static void reduceTile(texture2d<float, access::read> src,
                       array<texture2d<float, access::write>, ANBC_MIPS_MAX_LEVELS - 1> outs,
                       constant MipParams& P,
                       uint srcLod, uint2 tileOrigin, uint2 tileIdx,
                       uint2 lt, uint lidx, threadgroup float4* tg)
{
    const uint2 srcDims = P.dims[srcLod].xy;
    const uint2 base = tileOrigin + lt * 4;

    float4 v[16];
    for (uint j = 0; j < 4; j++)
        for (uint i = 0; i < 4; i++)
            v[j * 4 + i] = loadTexel(src, srcLod, base + uint2(i, j), srcDims);

    // level +1: 2x2 per thread
    const uint l1 = srcLod + 1;
    if (l1 >= P.numMips)
        return;
    float4 m1[4];
    for (uint j = 0; j < 2; j++) {
        for (uint i = 0; i < 2; i++) {
            const uint r0 = (2 * j) * 4 + 2 * i;
            const uint r1 = (2 * j + 1) * 4 + 2 * i;
            m1[j * 2 + i] = 0.25f * (v[r0] + v[r0 + 1] + v[r1] + v[r1 + 1]);
            storeTexel(outs[l1 - 1], tileIdx * 32 + lt * 2 + uint2(i, j), P.dims[l1].xy, m1[j * 2 + i]);
        }
    }

    // level +2: 1 per thread. This thread's 2x2 of level +1 texels starts at
    // tileIdx * 32 + lt * 2; if the second column/row is past that level's
    // edge, reuse the first (the edge clamp).
    const uint l2 = srcLod + 2;
    if (l2 >= P.numMips)
        return;
    const uint2 g1 = tileIdx * 32 + lt * 2;
    const uint cx = (g1.x + 1 < P.dims[l1].x) ? 1 : 0;
    const uint cy = (g1.y + 1 < P.dims[l1].y) ? 2 : 0;
    const float4 m2 = 0.25f * (m1[0] + m1[cx] + m1[cy] + m1[cy + cx]);
    storeTexel(outs[l2 - 1], tileIdx * 16 + lt, P.dims[l2].xy, m2);
    tg[lidx] = m2; // 16x16, stride 16

    // levels +3..+6: 8x8, 4x4, 2x2, 1x1 through threadgroup memory
    for (uint k = 3; k <= 6; k++) {
        const uint l = srcLod + k;
        if (l >= P.numMips)
            return;
        const uint size = 16u >> (k - 2);
        const uint stride = size * 2;
        const bool active = lt.x < size && lt.y < size;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 m = 0.0f;
        if (active) {
            // Same edge clamp against level l-1 (whose texels tg holds, at
            // global coordinates tileIdx * stride + local).
            const uint2 g = tileIdx * stride + lt * 2;
            const uint dx = (g.x + 1 < P.dims[l - 1].x) ? 1 : 0;
            const uint dy = (g.y + 1 < P.dims[l - 1].y) ? stride : 0;
            const uint r0 = (2 * lt.y) * stride + 2 * lt.x;
            m = 0.25f * (tg[r0] + tg[r0 + dx] + tg[r0 + dy] + tg[r0 + dy + dx]);
            storeTexel(outs[l - 1], tileIdx * size + lt, P.dims[l].xy, m);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active)
            tg[lt.y * size + lt.x] = m;
    }
}

kernel void anbc_mips(texture2d<float, access::read> src [[texture(0)]],
                      array<texture2d<float, access::write>, ANBC_MIPS_MAX_LEVELS - 1> outs [[texture(1)]],
                      constant MipParams& P [[buffer(0)]],
                      device atomic_uint* counter [[buffer(1)]],
                      uint2 groupId [[threadgroup_position_in_grid]],
                      uint2 lt [[thread_position_in_threadgroup]],
                      uint lidx [[thread_index_in_threadgroup]])
{
    threadgroup float4 tg[256];
    threadgroup uint isLast;

    reduceTile(src, outs, P, 0, groupId * ANBC_MIPS_TILE, groupId, lt, lidx, tg);

    if (P.numMips <= 7)
        return;

    // Publish this group's mip-6 texel to the rest of the device before
    // counting ourselves done.
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_texture);
    if (lidx == 0) {
        atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);
        const uint prev = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        isLast = (prev == P.numGroups - 1) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (isLast == 0)
        return;

    atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Last group: mip 6 is one tile, produce 7..12.
    reduceTile(src, outs, P, 6, uint2(0, 0), uint2(0, 0), lt, lidx, tg);

    if (lidx == 0)
        atomic_store_explicit(counter, 0u, memory_order_relaxed); // ready for the next dispatch
}
