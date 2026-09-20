// Scalar MLP evaluation, one block per thread (GLSL port of
// anbc_mlp_scalar.metal). Used by the BC7 kernel; no cooperative-matrix /
// linear-algebra extensions, so it runs on any Vulkan 1.1 device.
//
// GLSL functions cannot take buffer blocks as parameters, so ANBC_DEFINE_MLP
// stamps out one evaluator per (header, weights) block pair. The weights
// are declared as vec4: every layer offset is a multiple of 4 floats
// because all layer widths are multiples of 4 (the C side checks), which is
// also what lets the loops stream float4s.

#ifndef ANBC_MLP_SCALAR_GLSL
#define ANBC_MLP_SCALAR_GLSL

#include "anbc_common.glsl"

// Same layout as the Metal ModelHeader / the host's (72 bytes, std430 packs
// uint arrays tightly).
struct ModelHeader {
    uint numLayers;
    uint dims[ANBC_MODEL_MAX_LAYERS + 1];
    uint layerOffset[ANBC_MODEL_MAX_LAYERS]; // float offset of W; bias follows W
};

float sigmoidf(float x) { return 1.0 / (1.0 + exp(-x)); }

// x: dims[0] inputs (as vec4s). y: receives dims[numLayers] outputs
// (sigmoid applied). Activations ping-pong between the two halves of `buf`.
#define ANBC_DEFINE_MLP(FN, H, W)                                                            \
    void FN(in vec4 x[16], out float y[ANBC_MLP_MAX_OUT])                                    \
    {                                                                                        \
        vec4 buf[2][ANBC_MLP_MAX_HIDDEN / 4];                                                \
        for (uint i = 0u; i < H.dims[0] / 4u; i++)                                           \
            buf[0][i] = x[i];                                                                \
        uint src = 0u;                                                                       \
        for (uint l = 0u; l < H.numLayers; l++) {                                            \
            const uint din4 = H.dims[l] / 4u;                                                \
            const uint dout = H.dims[l + 1u];                                                \
            const uint w4 = H.layerOffset[l] / 4u;                                           \
            const uint b4 = w4 + din4 * dout;                                                \
            const bool last = (l + 1u == H.numLayers);                                       \
            const uint dst = src ^ 1u;                                                       \
            for (uint o = 0u; o < dout; o += 4u) {                                           \
                vec4 acc = W[b4 + o / 4u];                                                   \
                const uint r0 = w4 + o * din4;                                               \
                const uint r1 = r0 + din4, r2 = r1 + din4, r3 = r2 + din4;                   \
                for (uint i = 0u; i < din4; i++) {                                           \
                    const vec4 v = buf[src][i];                                              \
                    acc.x += dot(W[r0 + i], v);                                              \
                    acc.y += dot(W[r1 + i], v);                                              \
                    acc.z += dot(W[r2 + i], v);                                              \
                    acc.w += dot(W[r3 + i], v);                                              \
                }                                                                            \
                buf[dst][o / 4u] = last ? vec4(sigmoidf(acc.x), sigmoidf(acc.y), sigmoidf(acc.z), sigmoidf(acc.w)) \
                                        : max(acc, vec4(0.0));                               \
            }                                                                                \
            src = dst;                                                                       \
        }                                                                                    \
        const uint dout = H.dims[H.numLayers];                                               \
        for (uint o = 0u; o < ANBC_MLP_MAX_OUT; o++)                                         \
            y[o] = (o < dout) ? buf[src][o / 4u][o & 3u] : 0.0;                              \
    }

#endif // ANBC_MLP_SCALAR_GLSL
