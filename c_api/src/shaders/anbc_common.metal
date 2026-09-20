// Shared by every encoder kernel (BC7 scalar/tensor, BC6H, BC5): dispatch
// parameters, block loading, the least-squares endpoint refit and the bit
// writer. Format-specific refinement/packing lives in anbc_bc7_common.metal /
// anbc_bc6h_common.metal / anbc_bc5_common.metal, the MLP evaluators in
// anbc_mlp_scalar.metal / anbc_mlp_tensor.metal.
//
// The runtime-compiled (scalar) libraries are built by concatenating these
// files (see CMakeLists.txt), the offline-compiled tensor kernels #include
// them, hence the include guards.

#ifndef ANBC_COMMON_METAL
#define ANBC_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

#define ANBC_MODEL_MAX_LAYERS 8
#define ANBC_MLP_MAX_IN 64     // 16 RGBA texels
#define ANBC_MLP_MAX_HIDDEN 128 // the C side rejects wider models
#define ANBC_MLP_MAX_OUT 40

struct EncodeParams {
    uint mipLevel;
    uint blocksX;
    uint blocksY;
    uint refineIters;
    uint mipWidth;
    uint mipHeight;
    uint blockBase; // first output block of this mip (the Vulkan kernels index with it; Metal binds an offset address)
    uint pad1;
};

// ---------------------------------------------------------------------------
// Refinement helpers (templated on channel count via overloads)
// ---------------------------------------------------------------------------

// dot() for the scalar case so the templates below work for float too.
static inline float vdot(float a, float b) { return a * b; }
static inline float vdot(float2 a, float2 b) { return dot(a, b); }
static inline float vdot(float3 a, float3 b) { return dot(a, b); }
static inline float vdot(float4 a, float4 b) { return dot(a, b); }

// Blend factor of the palette entry nearest to a pixel: project on the
// endpoint line (all palette entries lie on it), the caller snaps to the
// format's weights.
template <typename V>
static inline float projectT(V p, V e0, V e1)
{
    const V d = e1 - e0;
    const float dd = vdot(d, d);
    return dd > 0.0f ? clamp(vdot(p - e0, d) / dd, 0.0f, 1.0f) : 0.0f;
}

// Least-squares endpoints for fixed blend weights w[i] (see _ls_endpoints in
// bc7_codec.py). Degenerate (all pixels share one index) -> e0 = e1 = mean.
// The degeneracy test is relative: for a flat block det is mathematically 0
// but comes out as float32 cancellation noise of either sign, and an absolute
// threshold that lets it through divides noise by noise.
template <typename V>
static void lsEndpoints(thread const V* px, thread const float* w, thread V& e0, thread V& e1)
{
    float aa = 0, ab = 0, bb = 0;
    V ap = V(0), bp = V(0), sum = V(0);
    for (uint i = 0; i < 16; i++) {
        const float b = w[i];
        const float a = 1.0f - b;
        aa += a * a; ab += a * b; bb += b * b;
        ap += a * px[i]; bp += b * px[i]; sum += px[i];
    }
    const float det = aa * bb - ab * ab;
    if (det > 1e-4f * aa * bb) {
        e0 = clamp((bb * ap - ab * bp) / det, V(0), V(1));
        e1 = clamp((aa * bp - ab * ap) / det, V(0), V(1));
    } else {
        e0 = e1 = sum / 16.0f;
    }
}

// ---------------------------------------------------------------------------
// Bit packing (LSB-first, matching the Python packers)
// ---------------------------------------------------------------------------

struct BitWriter {
    uint words[4];
    uint pos;
};

static inline void bwInit(thread BitWriter& b) { b.words[0] = b.words[1] = b.words[2] = b.words[3] = 0; b.pos = 0; }

static inline void bwPut(thread BitWriter& b, uint value, uint nbits)
{
    const uint w = b.pos >> 5;
    const uint o = b.pos & 31;
    value &= (nbits == 32) ? 0xffffffffu : ((1u << nbits) - 1u);
    b.words[w] |= value << o;
    if (o + nbits > 32)
        b.words[w + 1] |= value >> (32 - o);
    b.pos += nbits;
}

// ---------------------------------------------------------------------------
// Block loading
// ---------------------------------------------------------------------------

// The 16 RGBA texels of block (bx, by) at the current mip, edge-clamped
// (RGBA8 or RGBA16F source, both read as float). They double as the BC7
// network's 64 inputs (pixel-major).
static void loadBlock(texture2d<float, access::read> src, constant EncodeParams& P, uint2 blockId, thread float4* px)
{
    const uint2 dims = uint2(P.mipWidth, P.mipHeight);
    for (uint j = 0; j < 4; j++) {
        for (uint i = 0; i < 4; i++) {
            const uint2 p = min(blockId * 4 + uint2(i, j), dims - 1);
            px[j * 4 + i] = src.read(p, P.mipLevel);
        }
    }
}

#endif // ANBC_COMMON_METAL
