/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 */

#include "anbc_blob.h"
#include "anbc_internal.h"

#include <stdlib.h>
#include <string.h>

/* ANBC_HAVE_METAL: the Metal backend (backend_metal.mm) is part of this build
 * -- set by CMake on Apple platforms and by the single header when it is
 * compiled as Objective-C++. */
#ifndef ANBC_HAVE_METAL
anbcResult anbcBackendMetalInit(anbcDevice* device)
{
    (void)device;
    return ANBC_ERROR_UNSUPPORTED;
}
#endif

const char* anbcResultString(anbcResult result)
{
    switch (result) {
    case ANBC_OK:                     return "ok";
    case ANBC_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case ANBC_ERROR_UNSUPPORTED:      return "unsupported";
    case ANBC_ERROR_IO:               return "io error";
    case ANBC_ERROR_BAD_MODEL:        return "bad model file";
    case ANBC_ERROR_NO_MODEL:         return "model not loaded";
    case ANBC_ERROR_BACKEND:          return "backend error";
    }
    return "unknown";
}

uint32_t anbcComputeMipLayout(uint32_t width, uint32_t height, bool generateMips,
                              anbcMipLayout mips[ANBC_MAX_MIPS], size_t* totalBytes)
{
    uint32_t count = 0;
    size_t   offset = 0;
    uint32_t w = width, h = height;
    for (;;) {
        anbcMipLayout* m = &mips[count++];
        m->width = w;
        m->height = h;
        m->blocksX = (w + 3) / 4;
        m->blocksY = (h + 3) / 4;
        m->byteOffset = offset;
        offset += (size_t)m->blocksX * m->blocksY * ANBC_BLOCK_BYTES;
        if (!generateMips || (w == 1 && h == 1) || count == ANBC_MAX_MIPS)
            break;
        w = w > 1 ? w / 2 : 1;
        h = h > 1 ? h / 2 : 1;
    }
    *totalBytes = offset;
    return count;
}

anbcDevice* anbcCreateDevice(anbcDeviceBackend backend)
{
    anbcDevice* device = (anbcDevice*)calloc(1, sizeof(anbcDevice));
    if (!device)
        return NULL;
    device->backendKind = backend;

    anbcResult r;
    switch (backend) {
    case ANBC_DEVICE_BACKEND_METAL:  r = anbcBackendMetalInit(device); break;
    case ANBC_DEVICE_BACKEND_VULKAN: r = anbcBackendVulkanInit(device); break;
    default:                         r = ANBC_ERROR_INVALID_ARGUMENT; break;
    }
    if (r != ANBC_OK) {
        free(device);
        return NULL;
    }
    /* The shipped BC7 networks come from the blob, so BC7 works right away;
     * anbcLoadModel can still swap one in from a file. */
    static const char* const kEmbeddedModels[] = { "bc7_mode6.bin", "bc7_mode5.bin" };
    for (size_t i = 0; i < sizeof(kEmbeddedModels) / sizeof(kEmbeddedModels[0]); i++) {
        const void* data;
        size_t      size;
        anbcModel   model;
        if (anbcBlobFind(kEmbeddedModels[i], &data, &size) && anbcModelParse(data, size, &model) == ANBC_OK &&
            anbcInstallModel(device, &model) != ANBC_OK)
            anbcModelFree(&model);
    }
    return device;
}

void anbcDestroyDevice(anbcDevice* device)
{
    if (!device)
        return;
    if (device->backend.destroy)
        device->backend.destroy(device);
    anbcModelFree(&device->bc7Mode6);
    anbcModelFree(&device->bc7Mode5);
    free(device);
}

void anbcGetDeviceInfo(const anbcDevice* device, anbcDeviceInfo* outInfo)
{
    if (!outInfo)
        return;
    if (!device) {
        outInfo->name = outInfo->backend = "none";
        outInfo->tensorOps = 0;
        return;
    }
    *outInfo = device->info;
}

/* Uploads a parsed model and takes ownership of it (on failure the caller
 * still owns it). */
anbcResult anbcInstallModel(anbcDevice* device, anbcModel* model)
{
    anbcModel* slot = model->mode == 6 ? &device->bc7Mode6 : &device->bc7Mode5;
    bool*      flag = model->mode == 6 ? &device->hasBc7Mode6 : &device->hasBc7Mode5;
    anbcResult r = device->backend.uploadModel(device, model);
    if (r != ANBC_OK)
        return r;
    anbcModelFree(slot);
    *slot = *model;
    *flag = true;
    return ANBC_OK;
}

anbcResult anbcLoadModel(anbcDevice* device, anbcTextureFormat format, const char* path)
{
    if (!device || !path)
        return ANBC_ERROR_INVALID_ARGUMENT;
    if (format != ANBC_TEXTURE_FORMAT_BC7)
        return ANBC_ERROR_UNSUPPORTED; /* BC5, BC6H and ASTC have no network */

    anbcModel  model;
    anbcResult r = anbcModelLoad(path, &model);
    if (r != ANBC_OK)
        return r;
    r = anbcInstallModel(device, &model);
    if (r != ANBC_OK)
        anbcModelFree(&model);
    return r;
}

anbcTexture* anbcCreateTexture(anbcDevice* device, const anbcTextureDesc* desc)
{
    if (!device || !desc || !desc->pixels || desc->width == 0 || desc->height == 0)
        return NULL;
    if (desc->pixelFormat != ANBC_PIXEL_FORMAT_RGBA8_UNORM && desc->pixelFormat != ANBC_PIXEL_FORMAT_RGBA16_FLOAT)
        return NULL;

    anbcTexture* texture = (anbcTexture*)calloc(1, sizeof(anbcTexture));
    if (!texture)
        return NULL;
    texture->device = device;
    texture->width = desc->width;
    texture->height = desc->height;
    texture->pixelFormat = desc->pixelFormat;
    texture->flags = desc->flags;
    texture->mipCount = anbcComputeMipLayout(desc->width, desc->height,
                                             (desc->flags & ANBC_TEXTURE_FLAG_GENERATE_MIPS) != 0,
                                             texture->mips, &texture->compressedBytes);

    anbcTextureDesc d = *desc;
    if (d.rowPitch == 0)
        d.rowPitch = d.width * (d.pixelFormat == ANBC_PIXEL_FORMAT_RGBA16_FLOAT ? 8 : 4);

    if (device->backend.createTexture(device, texture, &d) != ANBC_OK) {
        free(texture);
        return NULL;
    }
    return texture;
}

void anbcDestroyTexture(anbcTexture* texture)
{
    if (!texture)
        return;
    texture->device->backend.destroyTexture(texture->device, texture);
    free(texture);
}

anbcResult anbcCompress(anbcDevice* device, anbcTexture* texture, anbcTextureFormat format,
                        const anbcCompressOptions* options)
{
    if (!device || !texture || texture->device != device)
        return ANBC_ERROR_INVALID_ARGUMENT;
    if (format == ANBC_TEXTURE_FORMAT_BC7) {
        if (!device->hasBc7Mode6 || !device->hasBc7Mode5)
            return ANBC_ERROR_NO_MODEL;
    } else if (format != ANBC_TEXTURE_FORMAT_BC5 && format != ANBC_TEXTURE_FORMAT_BC6H && !anbcIsAstc(format)) {
        return ANBC_ERROR_UNSUPPORTED; /* BC5 / BC6H / ASTC need no model */
    }

    anbcCompressOptions opts = { .refineIterations = 2, .flags = 0 };
    if (options)
        opts = *options;

    anbcResult r = device->backend.compress(device, texture, format, &opts);
    texture->isCompressed = (r == ANBC_OK);
    return r;
}

uint32_t anbcGetMipCount(const anbcTexture* texture)
{
    return texture ? texture->mipCount : 0;
}

anbcResult anbcGetMip(const anbcTexture* texture, uint32_t level, anbcMipInfo* outInfo)
{
    if (!texture || !outInfo || level >= texture->mipCount)
        return ANBC_ERROR_INVALID_ARGUMENT;
    if (!texture->isCompressed)
        return ANBC_ERROR_INVALID_ARGUMENT;

    const anbcMipLayout* m = &texture->mips[level];
    outInfo->width = m->width;
    outInfo->height = m->height;
    outInfo->blocksX = m->blocksX;
    outInfo->blocksY = m->blocksY;
    outInfo->sizeBytes = (size_t)m->blocksX * m->blocksY * ANBC_BLOCK_BYTES;
    outInfo->data = (const uint8_t*)texture->compressed + m->byteOffset;
    return ANBC_OK;
}
