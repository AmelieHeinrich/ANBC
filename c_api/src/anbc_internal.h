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
#define ANBC_BLOCK_BYTES 16 /* BC7, BC6H and BC5 all use 16-byte blocks */

/* `format` field of a model file (export_weights.py). Only BC7 networks
 * exist; BC5 and BC6H are encoded without one. */
#define ANBC_MODEL_FORMAT_BC7 7

/* A multi-layer perceptron loaded from a .bin written by src/export_weights.py.
 * Layer i maps dims[i] -> dims[i+1]; weights are (out, in) row-major, then bias. */
typedef struct anbcModel {
    uint32_t format;     /* ANBC_MODEL_FORMAT_* */
    uint32_t mode;       /* 5 or 6 for BC7 */
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

/* Backend constructors. Return ANBC_ERROR_UNSUPPORTED when not available. */
anbcResult anbcBackendCpuInit(anbcDevice* device);
anbcResult anbcBackendMetalInit(anbcDevice* device);

#ifdef __cplusplus
}
#endif

#endif
