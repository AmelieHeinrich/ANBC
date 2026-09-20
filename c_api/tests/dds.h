/* Minimal DDS writer for BC7_UNORM / BC6H_UF16 / BC5_UNORM textures with mips (DX10 header). */
#ifndef ANBC_TESTS_DDS_H
#define ANBC_TESTS_DDS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct ddsMip {
    const void* data;
    size_t      sizeBytes;
    uint32_t    width;
} ddsMip;

static void ddsPutU32(FILE* f, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, 4, f);
}

#define DDS_DXGI_FORMAT_BC5_UNORM 83
#define DDS_DXGI_FORMAT_BC6H_UF16 95
#define DDS_DXGI_FORMAT_BC7_UNORM 98

/* Returns 0 on success. `dxgiFormat` is one of the DDS_DXGI_FORMAT_* above
 * (all are 16 bytes per 4x4 block). */
static int ddsWriteBlocks(const char* path, uint32_t dxgiFormat, uint32_t width, uint32_t height, const ddsMip* mips,
                          uint32_t mipCount)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return -1;

    const uint32_t DDSD_CAPS = 0x1, DDSD_HEIGHT = 0x2, DDSD_WIDTH = 0x4, DDSD_PIXELFORMAT = 0x1000,
                   DDSD_MIPMAPCOUNT = 0x20000, DDSD_LINEARSIZE = 0x80000;
    const uint32_t DDPF_FOURCC = 0x4;
    const uint32_t DDSCAPS_COMPLEX = 0x8, DDSCAPS_TEXTURE = 0x1000, DDSCAPS_MIPMAP = 0x400000;
    const uint32_t D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3;

    fwrite("DDS ", 1, 4, f);
    ddsPutU32(f, 124);
    ddsPutU32(f, DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE |
                     (mipCount > 1 ? DDSD_MIPMAPCOUNT : 0));
    ddsPutU32(f, height);
    ddsPutU32(f, width);
    ddsPutU32(f, (uint32_t)mips[0].sizeBytes);
    ddsPutU32(f, 0); /* depth */
    ddsPutU32(f, mipCount);
    for (int i = 0; i < 11; i++)
        ddsPutU32(f, 0); /* reserved1 */
    /* DDS_PIXELFORMAT */
    ddsPutU32(f, 32);
    ddsPutU32(f, DDPF_FOURCC);
    fwrite("DX10", 1, 4, f);
    ddsPutU32(f, 0); ddsPutU32(f, 0); ddsPutU32(f, 0); ddsPutU32(f, 0); ddsPutU32(f, 0);
    ddsPutU32(f, DDSCAPS_TEXTURE | (mipCount > 1 ? (DDSCAPS_COMPLEX | DDSCAPS_MIPMAP) : 0));
    ddsPutU32(f, 0); ddsPutU32(f, 0); ddsPutU32(f, 0); ddsPutU32(f, 0);
    /* DDS_HEADER_DXT10 */
    ddsPutU32(f, dxgiFormat);
    ddsPutU32(f, D3D10_RESOURCE_DIMENSION_TEXTURE2D);
    ddsPutU32(f, 0); /* misc flag */
    ddsPutU32(f, 1); /* array size */
    ddsPutU32(f, 0); /* misc flags 2 (alpha mode unknown) */

    for (uint32_t i = 0; i < mipCount; i++)
        fwrite(mips[i].data, 1, mips[i].sizeBytes, f);
    fclose(f);
    return 0;
}

#endif
