/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * Parser for the .bin files written by src/export_weights.py:
 *   "ANBC" u32 version=2, u32 format (7 = BC7), u32 mode,
 *   u32 numLayers, u32 dims[numLayers+1],
 *   then per layer f32 W[out][in] (row-major) followed by f32 b[out].
 * Version 1 files have no `format` word and are always BC7.
 *
 * The shipped networks are embedded in the blob (c_api/models/) and
 * parsed from memory; anbcModelLoad reads a file into memory first.
 */

#include "anbc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Reader {
    const uint8_t* p;
    size_t         left;
} Reader;

static bool readU32(Reader* r, uint32_t* out)
{
    if (r->left < 4)
        return false;
    *out = (uint32_t)r->p[0] | ((uint32_t)r->p[1] << 8) | ((uint32_t)r->p[2] << 16) | ((uint32_t)r->p[3] << 24);
    r->p += 4;
    r->left -= 4;
    return true;
}

anbcResult anbcModelParse(const void* data, size_t size, anbcModel* out)
{
    memset(out, 0, sizeof(*out));
    Reader   r = { (const uint8_t*)data, size };
    uint32_t version;
    if (r.left < 4 || memcmp(r.p, "ANBC", 4) != 0)
        return ANBC_ERROR_BAD_MODEL;
    r.p += 4;
    r.left -= 4;
    if (!readU32(&r, &version) || (version != 1 && version != 2))
        return ANBC_ERROR_BAD_MODEL;
    if (version == 1)
        out->format = ANBC_MODEL_FORMAT_BC7;
    else if (!readU32(&r, &out->format))
        return ANBC_ERROR_BAD_MODEL;
    if (!readU32(&r, &out->mode) || !readU32(&r, &out->numLayers))
        return ANBC_ERROR_BAD_MODEL;
    if (out->format != ANBC_MODEL_FORMAT_BC7 || (out->mode != 5 && out->mode != 6))
        return ANBC_ERROR_BAD_MODEL;
    if (out->numLayers == 0 || out->numLayers > ANBC_MAX_MODEL_LAYERS)
        return ANBC_ERROR_BAD_MODEL;
    for (uint32_t i = 0; i <= out->numLayers; i++)
        if (!readU32(&r, &out->dims[i]) || out->dims[i] == 0)
            return ANBC_ERROR_BAD_MODEL;

    size_t total = 0;
    for (uint32_t i = 0; i < out->numLayers; i++) {
        out->layerOffset[i] = total;
        total += (size_t)out->dims[i + 1] * out->dims[i] + out->dims[i + 1];
    }
    if (r.left < total * sizeof(float))
        return ANBC_ERROR_BAD_MODEL;
    out->dataFloats = total;
    out->data = (float*)malloc(total * sizeof(float));
    if (!out->data)
        return ANBC_ERROR_IO;
    /* The file is little-endian float32; every platform we target is too. */
    memcpy(out->data, r.p, total * sizeof(float));
    return ANBC_OK;
}

anbcResult anbcModelLoad(const char* path, anbcModel* out)
{
    memset(out, 0, sizeof(*out));
    FILE* f = fopen(path, "rb");
    if (!f)
        return ANBC_ERROR_IO;
    anbcResult result = ANBC_ERROR_IO;
    uint8_t*   data = NULL;
    if (fseek(f, 0, SEEK_END) == 0) {
        const long size = ftell(f);
        if (size > 0 && fseek(f, 0, SEEK_SET) == 0 && (data = (uint8_t*)malloc((size_t)size)) != NULL &&
            fread(data, 1, (size_t)size, f) == (size_t)size)
            result = anbcModelParse(data, (size_t)size, out);
    }
    free(data);
    fclose(f);
    if (result != ANBC_OK)
        memset(out, 0, sizeof(*out));
    return result;
}

void anbcModelFree(anbcModel* model)
{
    free(model->data);
    memset(model, 0, sizeof(*model));
}
