// Scalar neural BC7 encoder: one thread per 4x4 block, both MLPs evaluated
// per thread (anbc_mlp_scalar.metal). Runs on every Metal 4 GPU; the
// tensor-ops kernel (anbc_bc7_tensor.metal) replaces it on M5-class GPUs.

#include "anbc_bc7_common.metal"
#include "anbc_mlp_scalar.metal"

kernel void anbc_bc7_encode(texture2d<float, access::read> src [[texture(0)]],
                            constant EncodeParams& P [[buffer(0)]],
                            constant ModelHeader& H6 [[buffer(1)]],
                            constant float* W6 [[buffer(2)]],
                            constant ModelHeader& H5 [[buffer(3)]],
                            constant float* W5 [[buffer(4)]],
                            device uint4* out [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.blocksX || gid.y >= P.blocksY)
        return;

    float4 px[16];
    loadBlock(src, P, gid, px);

    float y6[ANBC_MLP_MAX_OUT];
    float y5[ANBC_MLP_MAX_OUT];
    runMlp(H6, W6, px, y6);
    runMlp(H5, W5, px, y5);

    out[gid.y * P.blocksX + gid.x] = finishBlock(px, y6, y5, P);
}
