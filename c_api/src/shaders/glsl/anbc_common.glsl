// Shared by every Vulkan encoder kernel (GLSL port of anbc_common.metal):
// dispatch parameters, block loading, the least-squares endpoint refit and
// the bit writer. Format-specific refinement/packing lives in
// anbc_bc7_common.glsl / anbc_bc6h_common.glsl / anbc_bc5_common.glsl /
// anbc_astc_common.glsl, the MLP evaluator in anbc_mlp_scalar.glsl.
//
// The kernels are compiled with glslc (see CMakeLists.txt) and #include
// these files; GLSL has no templates, so the Metal templates on channel
// count become macro-instantiated overloads (GLSL does overload).
//
// Bindings (set 0), matching the Metal argument-table slots:
//   0  sampler2D  source texture, full mip chain (texelFetch, no filtering)
//   1  SSBO       BC7 mode-6 model header    3  mode-5 header
//   2  SSBO       BC7 mode-6 weights (vec4)  4  mode-5 weights
//   5  SSBO       output blocks (uvec4) of the whole mip chain (P.blockBase)
// EncodeParams are push constants.

#ifndef ANBC_COMMON_GLSL
#define ANBC_COMMON_GLSL

#define ANBC_MODEL_MAX_LAYERS 8
#define ANBC_MLP_MAX_IN 64     // 16 RGBA texels
#define ANBC_MLP_MAX_HIDDEN 128 // the C side rejects wider models
#define ANBC_MLP_MAX_OUT 40

layout(push_constant) uniform EncodeParams {
    uint mipLevel;
    uint blocksX;
    uint blocksY;
    uint refineIters;
    uint mipWidth;
    uint mipHeight;
    uint blockBase; // first output block of this mip in uOut (one buffer for the whole chain)
    uint pad1;
} P;

layout(set = 0, binding = 0) uniform sampler2D uSrc;
layout(std430, set = 0, binding = 5) writeonly buffer OutBlocks { uvec4 blocks[]; } uOut;

// ---------------------------------------------------------------------------
// Refinement helpers, one overload per channel count (float .. vec4)
// ---------------------------------------------------------------------------

float vdot(float a, float b) { return a * b; }
float vdot(vec2 a, vec2 b) { return dot(a, b); }
float vdot(vec3 a, vec3 b) { return dot(a, b); }
float vdot(vec4 a, vec4 b) { return dot(a, b); }

// projectT: blend factor of the palette entry nearest to a pixel (project on
// the endpoint line, the caller snaps to the format's weights).
// lsEndpoints: least-squares endpoints for fixed blend weights w[i] (see
// _ls_endpoints in bc7_codec.py). Degenerate (all pixels share one index)
// -> e0 = e1 = mean. The degeneracy test is relative: for a flat block det
// is mathematically 0 but comes out as float32 cancellation noise of either
// sign, and an absolute threshold that lets it through divides noise by noise.
#define ANBC_LINE_FUNCS(V)                                                                   \
    float projectT(V p, V e0, V e1)                                                          \
    {                                                                                        \
        const V d = e1 - e0;                                                                 \
        const float dd = vdot(d, d);                                                         \
        return dd > 0.0 ? clamp(vdot(p - e0, d) / dd, 0.0, 1.0) : 0.0;                       \
    }                                                                                        \
    void lsEndpoints(in V px[16], in float w[16], out V e0, out V e1)                        \
    {                                                                                        \
        float aa = 0.0, ab = 0.0, bb = 0.0;                                                  \
        V ap = V(0.0), bp = V(0.0), sum = V(0.0);                                            \
        for (uint i = 0u; i < 16u; i++) {                                                    \
            const float b = w[i];                                                            \
            const float a = 1.0 - b;                                                         \
            aa += a * a; ab += a * b; bb += b * b;                                            \
            ap += a * px[i]; bp += b * px[i]; sum += px[i];                                  \
        }                                                                                    \
        const float det = aa * bb - ab * ab;                                                 \
        if (det > 1e-4 * aa * bb) {                                                          \
            e0 = clamp((bb * ap - ab * bp) / det, V(0.0), V(1.0));                           \
            e1 = clamp((aa * bp - ab * ap) / det, V(0.0), V(1.0));                           \
        } else {                                                                             \
            e0 = sum / 16.0;                                                                 \
            e1 = e0;                                                                         \
        }                                                                                    \
    }

ANBC_LINE_FUNCS(float)
ANBC_LINE_FUNCS(vec2)
ANBC_LINE_FUNCS(vec3)
ANBC_LINE_FUNCS(vec4)

// ---------------------------------------------------------------------------
// Bit packing (LSB-first, matching the Python packers)
// ---------------------------------------------------------------------------

struct BitWriter {
    uint words[4];
    uint pos;
};

void bwInit(inout BitWriter b) { b.words[0] = b.words[1] = b.words[2] = b.words[3] = 0u; b.pos = 0u; }

void bwPut(inout BitWriter b, uint value, uint nbits)
{
    const uint w = b.pos >> 5;
    const uint o = b.pos & 31u;
    value &= (nbits == 32u) ? 0xffffffffu : ((1u << nbits) - 1u);
    b.words[w] |= value << o;
    if (o + nbits > 32u)
        b.words[w + 1] |= value >> (32u - o);
    b.pos += nbits;
}

// ---------------------------------------------------------------------------
// Block loading
// ---------------------------------------------------------------------------

// The 16 RGBA texels of block (bx, by) at the current mip, edge-clamped
// (RGBA8 or RGBA16F source, both read as float). They double as the BC7
// network's 64 inputs (pixel-major).
void loadBlock(uvec2 blockId, out vec4 px[16])
{
    const ivec2 dims = ivec2(P.mipWidth, P.mipHeight);
    for (uint j = 0u; j < 4u; j++) {
        for (uint i = 0u; i < 4u; i++) {
            const ivec2 p = min(ivec2(blockId * 4u + uvec2(i, j)), dims - 1);
            px[j * 4u + i] = texelFetch(uSrc, p, int(P.mipLevel));
        }
    }
}

#endif // ANBC_COMMON_GLSL
