/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * Single-header smoke test: includes only dist/anbc.h (as Objective-C++, so
 * both backends are in) and encodes a synthetic RGBA texture with mips on
 * each backend with every network-free format. Built by the
 * anbc_single_header_test target after anbc_amalgamate.
 */
#define ANBC_IMPLEMENTATION
#include "anbc.h"
#include <stdio.h>
#include <stdlib.h>

static int encode(anbcDeviceBackend backend, const char* label)
{
    anbcDevice* dev = anbcCreateDevice(backend);
    if (!dev) { printf("%s: no device\n", label); return 1; }
    anbcDeviceInfo info;
    anbcGetDeviceInfo(dev, &info);
    // A 256x256 gradient with alpha, mips on: exercises the mip generator and
    // every kernel family cheaply.
    const uint32_t w = 256, h = 256;
    uint8_t* px = (uint8_t*)malloc(w * h * 4);
    for (uint32_t y = 0; y < h; y++) for (uint32_t x = 0; x < w; x++) {
        uint8_t* p = px + (y * w + x) * 4; p[0] = x; p[1] = y; p[2] = (x ^ y); p[3] = 255 - (x + y) / 2;
    }
    anbcTextureDesc desc = {}; desc.width = w; desc.height = h; desc.pixels = px;
    desc.pixelFormat = ANBC_PIXEL_FORMAT_RGBA8_UNORM; desc.flags = ANBC_TEXTURE_FLAG_GENERATE_MIPS;
    anbcTexture* tex = anbcCreateTexture(dev, &desc);
    int rc = 0;
    const anbcTextureFormat formats[] = { ANBC_TEXTURE_FORMAT_BC5, ANBC_TEXTURE_FORMAT_BC6H, ANBC_TEXTURE_FORMAT_ASTC_4x4_UNORM };
    for (int i = 0; i < 3; i++) {
        anbcResult r = anbcCompress(dev, tex, formats[i], NULL);
        anbcMipInfo mip; anbcGetMip(tex, 0, &mip);
        const uint32_t* blk = (const uint32_t*)mip.data;
        printf("%s [%s, %s]: format %d -> %s, %u mips, block0 %08x %08x %08x %08x\n", label, info.name, info.backend,
               (int)formats[i], anbcResultString(r), anbcGetMipCount(tex), blk[0], blk[1], blk[2], blk[3]);
        if (r != ANBC_OK) rc = 1;
    }
    anbcDestroyTexture(tex); anbcDestroyDevice(dev); free(px);
    return rc;
}

int main(void) { return encode(ANBC_DEVICE_BACKEND_METAL, "metal") | encode(ANBC_DEVICE_BACKEND_VULKAN, "vulkan"); }
