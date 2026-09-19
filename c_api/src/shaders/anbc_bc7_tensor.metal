// Tensor-ops neural BC7 encoder (M5-class GPUs, MTLGPUFamilyApple10).
//
// Same pipeline as the scalar kernel, but the two MLPs run on the neural
// accelerators (anbc_mlp_tensor.metal): 32 blocks per SIMD group, one per
// lane. Compiled offline (see CMakeLists.txt).

#include "anbc_bc7_common.metal"
#include "anbc_mlp_tensor.metal"

#define ANBC_TOUT6 32 // 24 outputs padded
#define ANBC_TOUT5 64 // 40 outputs padded

using W0Tensor = W0TensorT<ANBC_MLP_IN>;
using W6OutTensor = WOutTensorT<ANBC_TOUT6>;
using W5OutTensor = WOutTensorT<ANBC_TOUT5>;

static constexpr constant auto kDescIn = matmul2d_descriptor(ANBC_TB, ANBC_TH, ANBC_MLP_IN, false, true, false);
static constexpr constant auto kDescOut6 = matmul2d_descriptor(ANBC_TB, ANBC_TOUT6, ANBC_TH, false, true, false);
static constexpr constant auto kDescOut5 = matmul2d_descriptor(ANBC_TB, ANBC_TOUT5, ANBC_TH, false, true, false);

kernel void anbc_bc7_encode_tensor(texture2d<float, access::read> src [[texture(0)]],
                                   constant EncodeParams& P [[buffer(0)]],
                                   device uint4* out [[buffer(5)]],
                                   W0Tensor W6_0 [[buffer(6)]],
                                   WHTensor W6_1 [[buffer(7)]],
                                   WHTensor W6_2 [[buffer(8)]],
                                   W6OutTensor W6_3 [[buffer(9)]],
                                   constant half* bias6 [[buffer(10)]],
                                   W0Tensor W5_0 [[buffer(11)]],
                                   WHTensor W5_1 [[buffer(12)]],
                                   WHTensor W5_2 [[buffer(13)]],
                                   W5OutTensor W5_3 [[buffer(14)]],
                                   constant half* bias5 [[buffer(15)]],
                                   uint tid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half scratchAll[ANBC_TG_SIMDGROUPS * ANBC_TB * ANBC_TH];
    threadgroup half* scratch = scratchAll + sg * (ANBC_TB * ANBC_TH);

    const uint numBlocks = P.blocksX * P.blocksY;
    const uint block = tid; // one block per lane, 32 per SIMD group
    const bool haveBlock = block < numBlocks;
    if (simd_all(!haveBlock))
        return;

    float4 px[16];
    if (haveBlock)
        loadBlock(src, P, uint2(block % P.blocksX, block / P.blocksX), px);
    float y6[ANBC_MLP_MAX_OUT];
    float y5[ANBC_MLP_MAX_OUT];
    runModelTensor<ANBC_MLP_IN, kDescIn, ANBC_TOUT6, kDescOut6>(scratch, px, haveBlock, lane, W6_0, W6_1, W6_2, W6_3, bias6, y6, 24);
    runModelTensor<ANBC_MLP_IN, kDescIn, ANBC_TOUT5, kDescOut5>(scratch, px, haveBlock, lane, W5_0, W5_1, W5_2, W5_3, bias5, y5, 40);

    if (haveBlock)
        out[block] = finishBlock(px, y6, y5, P);
}
