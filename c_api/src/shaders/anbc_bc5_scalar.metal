// BC5 encoder kernel: one thread per 4x4 block, no network (see
// anbc_bc5_common.metal). Runs on every Metal 4 GPU.

#include "anbc_bc5_common.metal"

kernel void anbc_bc5_encode(texture2d<float, access::read> src [[texture(0)]],
                            constant EncodeParams& P [[buffer(0)]],
                            device uint4* out [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.blocksX || gid.y >= P.blocksY)
        return;

    float4 px[16];
    loadBlock(src, P, gid, px);
    out[gid.y * P.blocksX + gid.x] = encodeBlockBc5(px, P);
}
