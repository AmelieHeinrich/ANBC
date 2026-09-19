// Scalar MLP evaluation, one block per thread, float4-streamed weights. Used
// by the BC7 scalar kernel; runs on every Metal 4 GPU. Needs anbc_common.metal
// first.

#ifndef ANBC_MLP_SCALAR_METAL
#define ANBC_MLP_SCALAR_METAL

#include "anbc_common.metal"

struct ModelHeader {
    uint numLayers;
    uint dims[ANBC_MODEL_MAX_LAYERS + 1];
    uint layerOffset[ANBC_MODEL_MAX_LAYERS]; // float offset of W; bias follows W
};

static inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }

// x: dims[0] inputs (as float4s). y: receives dims[numLayers] outputs
// (sigmoid applied). All layer widths must be multiples of 4 (the C side
// checks) so weights and activations stream as float4.
static void runMlp(constant ModelHeader& H, constant float* W, thread const float4* x, thread float* y)
{
    float4 bufA[ANBC_MLP_MAX_HIDDEN / 4];
    float4 bufB[ANBC_MLP_MAX_HIDDEN / 4];
    for (uint i = 0; i < H.dims[0] / 4; i++)
        bufA[i] = x[i];

    thread float4* in = bufA;
    thread float4* out = bufB;
    for (uint l = 0; l < H.numLayers; l++) {
        const uint din4 = H.dims[l] / 4;
        const uint dout = H.dims[l + 1];
        constant float4* w = (constant float4*)(W + H.layerOffset[l]);
        constant float* b = W + H.layerOffset[l] + H.dims[l] * dout;
        const bool last = (l + 1 == H.numLayers);
        for (uint o = 0; o < dout; o += 4) {
            float4 acc = float4(b[o], b[o + 1], b[o + 2], b[o + 3]);
            constant float4* r0 = w + (o + 0) * din4;
            constant float4* r1 = w + (o + 1) * din4;
            constant float4* r2 = w + (o + 2) * din4;
            constant float4* r3 = w + (o + 3) * din4;
            for (uint i = 0; i < din4; i++) {
                const float4 v = in[i];
                acc.x = fma(r0[i].x, v.x, fma(r0[i].y, v.y, fma(r0[i].z, v.z, fma(r0[i].w, v.w, acc.x))));
                acc.y = fma(r1[i].x, v.x, fma(r1[i].y, v.y, fma(r1[i].z, v.z, fma(r1[i].w, v.w, acc.y))));
                acc.z = fma(r2[i].x, v.x, fma(r2[i].y, v.y, fma(r2[i].z, v.z, fma(r2[i].w, v.w, acc.z))));
                acc.w = fma(r3[i].x, v.x, fma(r3[i].y, v.y, fma(r3[i].z, v.z, fma(r3[i].w, v.w, acc.w))));
            }
            out[o / 4] = last ? float4(sigmoidf(acc.x), sigmoidf(acc.y), sigmoidf(acc.z), sigmoidf(acc.w))
                              : max(acc, 0.0f);
        }
        thread float4* t = in; in = out; out = t;
    }
    const uint dout = H.dims[H.numLayers];
    for (uint o = 0; o < dout; o++)
        y[o] = in[o / 4][o & 3];
}

#endif // ANBC_MLP_SCALAR_METAL
