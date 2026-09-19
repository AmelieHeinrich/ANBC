/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * Parser for the .bin files written by src/export_weights.py:
 *   "ANBC" u32 version=2, u32 format (7 = BC7), u32 mode,
 *   u32 numLayers, u32 dims[numLayers+1],
 *   then per layer f32 W[out][in] (row-major) followed by f32 b[out].
 * Version 1 files have no `format` word and are always BC7.
 */

#include "anbc_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool readU32(FILE* f, uint32_t* out)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4)
        return false;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    return true;
}

anbcResult anbcModelLoad(const char* path, anbcModel* out)
{
    memset(out, 0, sizeof(*out));

    FILE* f = fopen(path, "rb");
    if (!f)
        return ANBC_ERROR_IO;

    anbcResult result = ANBC_ERROR_BAD_MODEL;
    char       magic[4];
    uint32_t   version;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "ANBC", 4) != 0)
        goto done;
    if (!readU32(f, &version) || (version != 1 && version != 2))
        goto done;
    if (version == 1)
        out->format = ANBC_MODEL_FORMAT_BC7;
    else if (!readU32(f, &out->format))
        goto done;
    if (!readU32(f, &out->mode) || !readU32(f, &out->numLayers))
        goto done;
    if (out->format != ANBC_MODEL_FORMAT_BC7 || (out->mode != 5 && out->mode != 6))
        goto done;
    if (out->numLayers == 0 || out->numLayers > ANBC_MAX_MODEL_LAYERS)
        goto done;
    for (uint32_t i = 0; i <= out->numLayers; i++) {
        if (!readU32(f, &out->dims[i]) || out->dims[i] == 0)
            goto done;
    }

    size_t total = 0;
    for (uint32_t i = 0; i < out->numLayers; i++) {
        out->layerOffset[i] = total;
        total += (size_t)out->dims[i + 1] * out->dims[i] + out->dims[i + 1];
    }
    out->dataFloats = total;
    out->data = (float*)malloc(total * sizeof(float));
    if (!out->data) {
        result = ANBC_ERROR_IO;
        goto done;
    }
    /* The file is little-endian float32; every platform we target is too. */
    if (fread(out->data, sizeof(float), total, f) != total) {
        free(out->data);
        out->data = NULL;
        goto done;
    }
    result = ANBC_OK;

done:
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
