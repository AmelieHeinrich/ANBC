/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * Reader for the embedded shader blob (tools/pack_blob.py): the Metal
 * sources compiled at runtime, the precompiled BC7 tensor metallib and the
 * SPIR-V modules, all in one uint32_t array so every payload is 4-byte
 * aligned. Text payloads are NUL-terminated (the NUL is not part of `size`).
 */

#ifndef ANBC_BLOB_H
#define ANBC_BLOB_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ANBC_BLOB_NAME_BYTES 40

typedef struct anbcBlobHeader {
    uint32_t magic; /* 'ANBB' */
    uint32_t count;
    uint32_t tocOffset;
    uint32_t dataOffset;
} anbcBlobHeader;

typedef struct anbcBlobEntry {
    char     name[ANBC_BLOB_NAME_BYTES];
    uint32_t offset;
    uint32_t size;
} anbcBlobEntry;

extern const uint32_t anbcBlobData[];

/* Looks up `name`; returns 0 when the blob has no such entry. */
static inline int anbcBlobFind(const char* name, const void** data, size_t* size)
{
    const uint8_t*        base = (const uint8_t*)anbcBlobData;
    const anbcBlobHeader* h = (const anbcBlobHeader*)base;
    const anbcBlobEntry*  toc = (const anbcBlobEntry*)(base + h->tocOffset);
    for (uint32_t i = 0; i < h->count; i++) {
        if (strncmp(toc[i].name, name, ANBC_BLOB_NAME_BYTES) == 0) {
            *data = base + toc[i].offset;
            *size = toc[i].size;
            return 1;
        }
    }
    return 0;
}

#endif
