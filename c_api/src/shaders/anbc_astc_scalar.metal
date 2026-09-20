// ASTC 4x4 encoder kernels: one thread per 4x4 block, one per texture kind
// (anbc_astc_common.metal). No network: endpoints start from the block's
// per-channel min/max (like BC5/BC6H) and go through the refinement rounds.

#include "anbc_astc_common.metal"

kernel void anbc_astc_game_encode(texture2d<float, access::read> src [[texture(0)]],
                                  constant EncodeParams& P [[buffer(0)]],
                                  device uint4* out [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.blocksX || gid.y >= P.blocksY)
        return;
    float4 px[16];
    loadBlock(src, P, gid, px);
    float4 e0 = px[0], e1 = px[0];
    for (uint i = 1; i < 16; i++) { e0 = min(e0, px[i]); e1 = max(e1, px[i]); }
    out[gid.y * P.blocksX + gid.x] = encodeAstcGame(px, e0, e1, P.refineIters);
}

kernel void anbc_astc_normal_encode(texture2d<float, access::read> src [[texture(0)]],
                                    constant EncodeParams& P [[buffer(0)]],
                                    device uint4* out [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.blocksX || gid.y >= P.blocksY)
        return;
    float4 px[16];
    loadBlock(src, P, gid, px);
    float2 e0 = px[0].rg, e1 = px[0].rg;
    for (uint i = 1; i < 16; i++) { e0 = min(e0, px[i].rg); e1 = max(e1, px[i].rg); }
    out[gid.y * P.blocksX + gid.x] = encodeAstcNormal(px, e0, e1, P.refineIters);
}

kernel void anbc_astc_float_encode(texture2d<float, access::read> src [[texture(0)]],
                                   constant EncodeParams& P [[buffer(0)]],
                                   device uint4* out [[buffer(5)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.blocksX || gid.y >= P.blocksY)
        return;
    float4 px[16];
    loadBlock(src, P, gid, px);
    // HDR texels -> LNS-domain pixels
    float3 lns[16];
    for (uint i = 0; i < 16; i++)
        lns[i] = float3(lnsNorm(px[i].r), lnsNorm(px[i].g), lnsNorm(px[i].b));
    float3 e0 = lns[0], e1 = lns[0];
    for (uint i = 1; i < 16; i++) { e0 = min(e0, lns[i]); e1 = max(e1, lns[i]); }
    out[gid.y * P.blocksX + gid.x] = encodeAstcFloat(lns, e0, e1, P.refineIters);
}
