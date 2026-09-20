/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#ifndef ANBC_INTERNAL_H
#define ANBC_INTERNAL_H

#include "anbc.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ANBC_MAX_MIPS 16
#define ANBC_MAX_MODEL_LAYERS 8
#define ANBC_BLOCK_BYTES 16 /* BC7, BC6H, BC5 and ASTC 4x4 all use 16-byte blocks */

/* `format` field of a model file (export_weights.py): only the BC7 networks
 * (mode 5 and 6) exist; BC5, BC6H and ASTC are encoded without one. */
#define ANBC_MODEL_FORMAT_BC7 7

/* ASTC 4x4 texture kinds: the index of the device's kernel. */
enum {
    ANBC_ASTC_GAME = 0,   /* ASTC_4x4_UNORM */
    ANBC_ASTC_NORMAL = 1, /* ASTC_4x4_UNORM + ANBC_TEXTURE_FLAG_NORMAL_MAP */
    ANBC_ASTC_FLOAT = 2,  /* ASTC_4x4_FLOAT */
    ANBC_ASTC_KINDS = 3
};

static inline bool anbcIsAstc(anbcTextureFormat format)
{
    return format == ANBC_TEXTURE_FORMAT_ASTC_4x4_UNORM || format == ANBC_TEXTURE_FORMAT_ASTC_4x4_FLOAT;
}

static inline uint32_t anbcAstcKind(anbcTextureFormat format, uint32_t textureFlags)
{
    if (format == ANBC_TEXTURE_FORMAT_ASTC_4x4_FLOAT)
        return ANBC_ASTC_FLOAT;
    return (textureFlags & ANBC_TEXTURE_FLAG_NORMAL_MAP) ? ANBC_ASTC_NORMAL : ANBC_ASTC_GAME;
}

/* A multi-layer perceptron loaded from a .bin written by src/export_weights.py.
 * Layer i maps dims[i] -> dims[i+1]; weights are (out, in) row-major, then bias. */
typedef struct anbcModel {
    uint32_t format;     /* ANBC_MODEL_FORMAT_BC7 */
    uint32_t mode;       /* 5 or 6 */
    uint32_t numLayers;
    uint32_t dims[ANBC_MAX_MODEL_LAYERS + 1];
    float*   data;       /* all W/b blobs back to back, layer order */
    size_t   dataFloats;
    size_t   layerOffset[ANBC_MAX_MODEL_LAYERS]; /* float offset of layer i's W (bias follows) */
} anbcModel;

anbcResult anbcModelLoad(const char* path, anbcModel* out);
void       anbcModelFree(anbcModel* model);

typedef struct anbcMipLayout {
    uint32_t width, height, blocksX, blocksY;
    size_t   byteOffset; /* into the compressed chain */
} anbcMipLayout;

/* Fills `mips` with the chain for (w, h): one level when !generateMips, else
 * down to 1x1. Returns the level count and writes the total byte size. */
uint32_t anbcComputeMipLayout(uint32_t width, uint32_t height, bool generateMips,
                              anbcMipLayout mips[ANBC_MAX_MIPS], size_t* totalBytes);

/* What a backend has to provide. Each backend fills one of these. */
typedef struct anbcBackend {
    void       (*destroy)(anbcDevice* device);
    anbcResult (*uploadModel)(anbcDevice* device, const anbcModel* model);
    anbcResult (*createTexture)(anbcDevice* device, anbcTexture* texture, const anbcTextureDesc* desc);
    void       (*destroyTexture)(anbcDevice* device, anbcTexture* texture);
    /* Must leave texture->compressed pointing at ANBC_BLOCK_BYTES-per-block data
     * laid out per texture->mips[]. */
    anbcResult (*compress)(anbcDevice* device, anbcTexture* texture, anbcTextureFormat format,
                           const anbcCompressOptions* options);
} anbcBackend;

struct anbcDevice {
    anbcDeviceBackend backendKind;
    anbcBackend       backend;
    void*             backendData;
    anbcDeviceInfo    info; /* filled by the backend at init */

    anbcModel bc7Mode6;
    anbcModel bc7Mode5;
    bool      hasBc7Mode6;
    bool      hasBc7Mode5;
};

struct anbcTexture {
    anbcDevice*     device;
    uint32_t        width, height;
    anbcPixelFormat pixelFormat;
    uint32_t        flags;
    uint32_t      mipCount;
    anbcMipLayout mips[ANBC_MAX_MIPS];
    size_t        compressedBytes;
    const void*   compressed;  /* valid after compress(), owned by the backend */
    bool          isCompressed;
    void*         backendData;
};

/* Backend constructors. Return ANBC_ERROR_UNSUPPORTED when not available
 * (the Metal one is a stub unless ANBC_HAVE_METAL, see anbc.c). */
anbcResult anbcBackendMetalInit(anbcDevice* device);
anbcResult anbcBackendVulkanInit(anbcDevice* device);

#ifdef __cplusplus
}
#endif

#endif
