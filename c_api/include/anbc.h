/**
 * @ Author: Amélie Heinrich
 * @ Create Time: 2026-09-19 17:21:24
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#ifndef AMELIE_NEURAL_BLOCK_COMPRESSOR_H
#define AMELIE_NEURAL_BLOCK_COMPRESSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum anbcDeviceBackend {
    ANBC_DEVICE_BACKEND_METAL, /* macOS: Metal 4 (the shipping backend on Apple platforms) */
    ANBC_DEVICE_BACKEND_VULKAN /* Windows / Linux (also runs on MoltenVK for testing) */
} anbcDeviceBackend;

typedef enum anbcTextureFormat {
    ANBC_TEXTURE_FORMAT_BC7,
    ANBC_TEXTURE_FORMAT_BC5, /* R and G of the source (normal maps); B/A are ignored. No network. */
    ANBC_TEXTURE_FORMAT_BC6H, /* BC6H_UF16 (unsigned HDR RGB, mode 11): negative inputs clamp to 0,
                               * alpha is ignored. No network. */
    ANBC_TEXTURE_FORMAT_ASTC_4x4_UNORM, /* ASTC 4x4 LDR (RGBA). Game textures: albedo, packed metal/roughness,
                                         * emissive, with or without alpha. With ANBC_TEXTURE_FLAG_NORMAL_MAP the
                                         * source's R,G (X,Y) are stored as luminance + alpha: sample .ra. */
    ANBC_TEXTURE_FORMAT_ASTC_4x4_FLOAT  /* ASTC 4x4 HDR RGB: negative inputs clamp to 0, alpha is ignored and
                                         * decodes as 1.0. Needs an HDR-capable ASTC decoder (Apple GPUs from A13/M1). */
} anbcTextureFormat;

/* Layout of the source pixels handed to anbcCreateTexture. Any texture format
 * can be compressed from either: BC7/BC5 on RGBA16F just clamp to [0,1],
 * BC6H on RGBA8 encodes the [0,1] values as-is. */
typedef enum anbcPixelFormat {
    ANBC_PIXEL_FORMAT_RGBA8_UNORM = 0, /* 4 bytes per pixel */
    ANBC_PIXEL_FORMAT_RGBA16_FLOAT     /* 8 bytes per pixel (IEEE half) */
} anbcPixelFormat;

typedef enum anbcResult {
    ANBC_OK = 0,
    ANBC_ERROR_INVALID_ARGUMENT,
    ANBC_ERROR_UNSUPPORTED,   /* backend not compiled in / GPU lacks Metal 4 / texture too large
                               * (4096^2 with ANBC_TEXTURE_FLAG_GENERATE_MIPS, 16384^2 without) */
    ANBC_ERROR_IO,            /* model file could not be read */
    ANBC_ERROR_BAD_MODEL,     /* model file malformed or wrong format */
    ANBC_ERROR_NO_MODEL,      /* BC7 with a network missing (an embedded one failed to upload) */
    ANBC_ERROR_BACKEND        /* GPU/API failure; see stderr for details */
} anbcResult;

enum {
    ANBC_TEXTURE_FLAG_SRGB          = 1 << 0, /* mip filtering happens in linear light */
    ANBC_TEXTURE_FLAG_GENERATE_MIPS = 1 << 1, /* build the full chain down to 1x1 on the GPU */
    ANBC_TEXTURE_FLAG_NORMAL_MAP    = 1 << 2  /* ASTC_4x4_UNORM: X,Y from R,G -> luminance + alpha (see above) */
};

typedef struct anbcTextureDesc {
    uint32_t        width;
    uint32_t        height;
    uint32_t        rowPitch;    /* bytes between rows of `pixels`; 0 means width * bytes per pixel */
    const void*     pixels;      /* tightly packed R,G,B,A per `pixelFormat`, top-left origin */
    anbcPixelFormat pixelFormat; /* ANBC_PIXEL_FORMAT_* (0 = RGBA8) */
    uint32_t        flags;       /* ANBC_TEXTURE_FLAG_* */
} anbcTextureDesc;

enum {
    /* Use the scalar inference kernel even when the GPU has tensor
     * acceleration (for A/B comparisons). */
    ANBC_COMPRESS_FLAG_NO_TENSOR_OPS = 1 << 0
};

typedef struct anbcCompressOptions {
    /* Rounds of least-squares endpoint refinement applied on top of the
     * initial endpoints (the networks' for BC7, the block min/max for
     * BC5 / BC6H / ASTC; indices are always chosen exactly). 0 keeps the
     * initial endpoints as-is. Default 2; costs almost nothing. */
    uint32_t refineIterations;
    uint32_t flags; /* ANBC_COMPRESS_FLAG_* */
} anbcCompressOptions;

typedef struct anbcDeviceInfo {
    const char* name;      /* GPU name, owned by the device */
    const char* backend;   /* "Metal 4" or "Vulkan" */
    int         tensorOps; /* MLP inference runs on the tensor accelerators (Metal, M5+) */
} anbcDeviceInfo;

typedef struct anbcMipInfo {
    uint32_t    width;
    uint32_t    height;
    uint32_t    blocksX;
    uint32_t    blocksY;
    const void* data;      /* blocksX * blocksY * 16 bytes, row-major blocks */
    size_t      sizeBytes;
} anbcMipInfo;

typedef struct anbcDevice anbcDevice;
typedef struct anbcTexture anbcTexture;

/* Returns NULL if the backend is unavailable on this machine (not compiled
 * in, no capable GPU, or -- Vulkan -- no loader library found). */
anbcDevice* anbcCreateDevice(anbcDeviceBackend backend);
void        anbcDestroyDevice(anbcDevice* device);
void        anbcGetDeviceInfo(const anbcDevice* device, anbcDeviceInfo* outInfo);

/* Optional: replace one of the embedded BC7 networks (mode 6 / mode 5, the
 * file records which) with one exported by src/export_weights.py. The
 * shipped networks are built into the library, so this is only for
 * experiments. BC5, BC6H and ASTC are encoded analytically (block min/max +
 * refinement) and have no model; passing them returns ANBC_ERROR_UNSUPPORTED. */
anbcResult anbcLoadModel(anbcDevice* device, anbcTextureFormat format, const char* path);

anbcTexture* anbcCreateTexture(anbcDevice* device, const anbcTextureDesc* desc);
void         anbcDestroyTexture(anbcTexture* texture);

/* Generates mips (if requested at creation) and compresses every level.
 * `options` may be NULL for defaults. Blocking. Every format works right
 * after anbcCreateDevice (the BC7 networks are embedded). */
anbcResult anbcCompress(anbcDevice* device, anbcTexture* texture, anbcTextureFormat format,
                        const anbcCompressOptions* options);

uint32_t   anbcGetMipCount(const anbcTexture* texture);
/* Valid after a successful anbcCompress; `data` stays owned by the texture. */
anbcResult anbcGetMip(const anbcTexture* texture, uint32_t level, anbcMipInfo* outInfo);

const char* anbcResultString(anbcResult result);

#ifdef __cplusplus
}
#endif

#endif
