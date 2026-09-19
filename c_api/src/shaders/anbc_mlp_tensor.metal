// Tensor-ops MLP evaluation (M5-class GPUs, MTLGPUFamilyApple10), used by
// the BC7 tensor kernel (templated on the input width for future formats).
//
// An MLP runs as batched matrix multiplies through Metal shader tensors +
// MetalPerformancePrimitives matmul2d, which the neural accelerators execute.
// Each SIMD group encodes 32 blocks (one per lane) as a 32-row batch:
//
//   X[32][IN] (threadgroup, half)      <- the lane's block
//   H = relu(X x W0^T + b0)             matmul2d, execution_simdgroup
//   H = relu(H x W1^T + b1)             (activations stay in registers as a
//   H = relu(H x W2^T + b2)              cooperative tensor when the layout
//   Y = H x W3^T -> threadgroup          allows; else via threadgroup scratch)
//   lane reads its row of Y, sigmoid, + b3
//
// Weight tensors use the export layout, (out, in) row-major = extents
// <in, out> with `in` contiguous, hence transpose_right. The final layer is
// zero-padded to OUT_PADDED outputs by the C side. Layer geometry is fixed to
// IN -> 128 -> 128 -> 128 -> out (the C side checks).
//
// Compiled offline only (see CMakeLists.txt): the matmul2d implementations
// come from libTensorOps.rtlib, which only the offline Metal linker resolves.

#ifndef ANBC_MLP_TENSOR_METAL
#define ANBC_MLP_TENSOR_METAL

#include "anbc_common.metal"

#include <metal_tensor>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

#define ANBC_TB 32           // blocks per SIMD group (matmul M) == SIMD width
#define ANBC_TG_SIMDGROUPS 2 // SIMD groups per threadgroup
#define ANBC_TH 128          // hidden width
#define ANBC_BIAS_STRIDE 128 // halves per layer in the bias block

template <int IN>
using W0TensorT = tensor<device half, extents<int, IN, ANBC_TH>>;
using WHTensor = tensor<device half, extents<int, ANBC_TH, ANBC_TH>>;
template <int OUT_PADDED>
using WOutTensorT = tensor<device half, extents<int, ANBC_TH, OUT_PADDED>>;
template <int IN>
using XTensorT = tensor<threadgroup half, extents<int, IN, ANBC_TB>, tensor_inline>;
using HTensor = tensor<threadgroup half, extents<int, ANBC_TH, ANBC_TB>, tensor_inline>;

static constexpr constant auto kDescHidden = matmul2d_descriptor(ANBC_TB, ANBC_TH, ANBC_TH, false, true, false);

template <typename CoopTensor>
static void applyBiasRelu(thread CoopTensor& coop, constant half* bias)
{
    simdgroup_barrier(mem_flags::mem_none);
    for (uint16_t i = 0; i < coop.get_capacity(); ++i) {
        const int col = coop.get_multidimensional_index(i)[0];
        const half v = coop[i] + bias[col];
        coop[i] = max(v, 0.0h);
    }
}

// Hidden -> hidden layer whose input is the previous layer's cooperative
// result. Falls back to a threadgroup round trip when the two ops' register
// layouts are not compatible.
template <typename Op, typename PrevCoop>
static auto hiddenLayer(thread Op& op, thread PrevCoop& prev, thread WHTensor& W, constant half* bias,
                        threadgroup half* scratch)
{
    const bool compatible = op.template is_compatible_as_left_input<half, half, half>(prev);
    auto input = compatible ? op.template get_left_input_cooperative_tensor<half, half, half>(prev)
                            : op.template get_left_input_cooperative_tensor<half, half, half>();
    if (!compatible) {
        HTensor h(scratch, extents<int, ANBC_TH, ANBC_TB>());
        prev.store(h);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        input.load(h);
        simdgroup_barrier(mem_flags::mem_none);
    }
    auto dst = op.template get_destination_cooperative_tensor<HTensor, WHTensor, half>();
    op.run(input, W, dst);
    applyBiasRelu(dst, bias);
    return dst;
}

// Runs one model for this SIMD group's 32 blocks. `scratch` is 32 x 128
// halves owned by the SIMD group; `xin` is the lane's IN inputs as IN/4
// float4s -- keep it float4: reading the same array through a reinterpreted
// `thread const float*` silently produced wrong inputs (-3.5 dB on BC7).
// Returns the lane's outN outputs (sigmoid applied) in y.
template <int IN, matmul2d_descriptor DescIn, int OUT_PADDED, matmul2d_descriptor DescOut>
static void runModelTensor(threadgroup half* scratch, thread const float4* xin, bool haveBlock, uint lane,
                           thread W0TensorT<IN>& W0, thread WHTensor& W1, thread WHTensor& W2,
                           thread WOutTensorT<OUT_PADDED>& W3, constant half* bias, thread float* y, uint outN)
{
    XTensorT<IN> X(scratch, extents<int, IN, ANBC_TB>());
    for (int i = 0; i < IN / 4; i++) {
        const float4 c = haveBlock ? xin[i] : float4(0.0f);
        X.set(half(c.x), i * 4 + 0, lane);
        X.set(half(c.y), i * 4 + 1, lane);
        X.set(half(c.z), i * 4 + 2, lane);
        X.set(half(c.w), i * 4 + 3, lane);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    matmul2d<DescIn, execution_simdgroup> opIn;
    auto h0 = opIn.template get_destination_cooperative_tensor<XTensorT<IN>, W0TensorT<IN>, half>();
    opIn.run(X, W0, h0);
    applyBiasRelu(h0, bias + 0 * ANBC_BIAS_STRIDE);

    matmul2d<kDescHidden, execution_simdgroup> opHidden;
    auto h1 = hiddenLayer(opHidden, h0, W1, bias + 1 * ANBC_BIAS_STRIDE, scratch);
    auto h2 = hiddenLayer(opHidden, h1, W2, bias + 2 * ANBC_BIAS_STRIDE, scratch);

    matmul2d<DescOut, execution_simdgroup> opOut;
    const bool compatible = opOut.template is_compatible_as_left_input<half, half, half>(h2);
    auto input = compatible ? opOut.template get_left_input_cooperative_tensor<half, half, half>(h2)
                            : opOut.template get_left_input_cooperative_tensor<half, half, half>();
    if (!compatible) {
        HTensor h(scratch, extents<int, ANBC_TH, ANBC_TB>());
        h2.store(h);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        input.load(h);
        simdgroup_barrier(mem_flags::mem_none);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup); // scratch is about to be overwritten
    tensor<threadgroup half, extents<int, OUT_PADDED, ANBC_TB>, tensor_inline> Y(scratch, extents<int, OUT_PADDED, ANBC_TB>());
    opOut.run(input, W3, Y);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    constant half* b3 = bias + 3 * ANBC_BIAS_STRIDE;
    for (uint o = 0; o < outN; o++) {
        const float v = float(Y.get(int(o), int(lane))) + float(b3[o]);
        y[o] = 1.0f / (1.0f + exp(-v));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup); // before the next model reuses scratch
}

#endif // ANBC_MLP_TENSOR_METAL
