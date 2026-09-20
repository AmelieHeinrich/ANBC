/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * anbc_compare: neural BC7 / BC6H / BC5 / ASTC 4x4 (anbc, GPU) vs a CPU
 * reference encoder, multithreaded: AMD Compressonator CMP_Core for the BCn
 * formats, ARM astc-encoder for ASTC. Both produce the full mip chain from
 * the same source, so the timings are "RGBA in -> block chain out" for both.
 *
 *   anbc_compare <image-or-folder> [--format bc7|bc6h|bc5|astc|astc-float]
 *                [--backend metal|vulkan] [--xcheck] [--normal-map] [--out DIR] [--srgb] [--cmp-quality q]
 *                [--astc-preset fast|medium|thorough] [--refine-iters n]
 *                [--no-mips] [--no-tensor-ops] [--threads n] [--single-thread]
 *
 * BC6H and astc-float take .hdr (Radiance) inputs, uploaded as RGBA16F; an
 * 8-bit image is converted to half [0,1] instead. Their PSNR is measured in
 * the half-int domain (half bit pattern / 0x7BFF, ~relative error), see
 * src/bc6h_codec.py. --normal-map (astc only) stores the source's R,G as
 * luminance + alpha (ANBC_TEXTURE_FLAG_NORMAL_MAP) and scores those two
 * channels like BC5.
 *
 * --xcheck (single image, macOS): encodes with the Metal *and* the Vulkan
 * backend and reports each one's PSNR plus the share of bit-identical
 * blocks per mip. Metal compiles with fast math, so the two are expected to
 * agree to within a few hundredths of a dB, not bit for bit.
 *
 * Single image: also decodes both results with the reference's decoder (an
 * independent third-party decoder, so it doubles as a bitstream check) and
 * prints PSNR against the source (RGB for BC7/BC6H/ASTC, the two stored
 * channels for BC5 / normal maps), plus anbc's mips against a CPU
 * box-filtered reference chain.
 *
 * Folder: writes <stem>_anbc.dds and <stem>_cmp.dds (full chains) for every
 * image into --out, prints timings, and writes summary.csv. ASTC formats
 * additionally write <stem>_anbc.astc / <stem>_cmp.astc (mip 0; further
 * levels as <stem>_anbc_mipN.astc), and astc-float writes only those since
 * DDS has no HDR ASTC format. Quality scoring of a folder is done by
 * src/compare_dds.py.
 */

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_TGA
#define STBI_ONLY_BMP
#define STBI_ONLY_HDR
#include "stb_image.h"

#include "anbc.h"
#include "astcenc.h"
#include "cmp_core.h"
#include "dds.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------------- */
/* Image helpers                                                             */
/* ------------------------------------------------------------------------- */

/* One of `rgba` (8-bit) or `half` (RGBA16F, HDR formats) is populated. */
struct Image {
    uint32_t w = 0, h = 0;
    bool hdr = false;
    std::vector<uint8_t> rgba;
    std::vector<uint16_t> half;
};

static inline float halfToFloat(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); return (float)f; }
static inline uint16_t floatToHalf(float v) { const _Float16 f = (_Float16)v; uint16_t h; memcpy(&h, &f, 2); return h; }

/* Unsigned BC6H's own domain: the half bit pattern, negatives -> 0, capped
 * at 0x7BFF (65504.0), as a fraction of that cap. */
static inline double halfIntNorm(uint16_t h)
{
    if (h & 0x8000)
        return 0.0;
    return std::min<uint32_t>(h, 0x7BFF) / 31743.0;
}

/* Everything that differs between the formats. */
struct Format {
    anbcTextureFormat anbc;
    const char*       name;
    const char*       modelFiles[3]; /* NULL-terminated; only BC7 loads networks */
    bool              tensorKernel;  /* has a tensor-ops variant worth A/B-ing */
    bool              hdr;           /* RGBA16F source, PSNR in the half-int domain */
    uint32_t          dxgi;          /* 0: no DDS output */
    int               channels;      /* channels that are stored / scored */
    const char*       psnrLabel;
    double            floorMip0, floorMips; /* single-image sanity floors (dB) */
    bool              astc;          /* reference = astcenc */
};

static const Format kFormatBC7 = { ANBC_TEXTURE_FORMAT_BC7, "BC7", { "bc7_mode6.bin", "bc7_mode5.bin", NULL }, true, false,
                                   DDS_DXGI_FORMAT_BC7_UNORM, 3, "PSNR RGB", 25.0, 20.0, false };
static const Format kFormatBC6H = { ANBC_TEXTURE_FORMAT_BC6H, "BC6H", { NULL, NULL, NULL }, false, true,
                                    DDS_DXGI_FORMAT_BC6H_UF16, 3, "PSNR half-int", 30.0, 25.0, false };
static const Format kFormatBC5 = { ANBC_TEXTURE_FORMAT_BC5, "BC5", { NULL, NULL, NULL }, false, false,
                                   DDS_DXGI_FORMAT_BC5_UNORM, 2, "PSNR RG", 25.0, 20.0, false };
static const Format kFormatASTC = { ANBC_TEXTURE_FORMAT_ASTC_4x4_UNORM, "ASTC 4x4", { NULL, NULL, NULL }, false, false,
                                    DDS_DXGI_FORMAT_ASTC_4X4_UNORM, 3, "PSNR RGB", 25.0, 20.0, true };
/* --normal-map: the same format scored on the two stored channels (X from L, Y from A). */
static const Format kFormatASTCNormal = { ANBC_TEXTURE_FORMAT_ASTC_4x4_UNORM, "ASTC 4x4 normal map", { NULL, NULL, NULL },
                                          false, false, DDS_DXGI_FORMAT_ASTC_4X4_UNORM, 2, "PSNR XY", 25.0, 20.0, true };
static const Format kFormatASTCFloat = { ANBC_TEXTURE_FORMAT_ASTC_4x4_FLOAT, "ASTC 4x4 HDR", { NULL, NULL, NULL }, false, true,
                                         0, 3, "PSNR half-int", 30.0, 25.0, true };

/* ------------------------------------------------------------------------- */
/* astcenc (reference encoder + decoder for the ASTC formats)                */
/* ------------------------------------------------------------------------- */

static float gAstcPreset = ASTCENC_PRE_MEDIUM;
static const char* gAstcPresetName = "medium";

/* One context per (profile, normal map, decode-only, thread count), reused
 * across images/levels so its setup cost stays out of the timings. */
static astcenc_context* astcContext(const Format& fmt, unsigned threads, bool decodeOnly)
{
    static astcenc_context* cache[2][2][2] = {}; /* [hdr][normal][decodeOnly] */
    static unsigned cachedThreads[2][2][2] = {};
    const int hdr = fmt.hdr ? 1 : 0, normal = fmt.channels == 2 ? 1 : 0, dec = decodeOnly ? 1 : 0;
    astcenc_context*& ctx = cache[hdr][normal][dec];
    if (ctx && cachedThreads[hdr][normal][dec] != threads) {
        astcenc_context_free(ctx);
        ctx = NULL;
    }
    if (!ctx) {
        astcenc_config cfg;
        unsigned flags = decodeOnly ? ASTCENC_FLG_DECOMPRESS_ONLY : 0;
        if (!fmt.hdr)
            flags |= ASTCENC_FLG_USE_DECODE_UNORM8; /* the exact top-8-bits decode our decoder implements */
        if (normal && !decodeOnly)
            flags |= ASTCENC_FLG_MAP_NORMAL;
        const astcenc_profile profile = fmt.hdr ? ASTCENC_PRF_HDR_RGB_LDR_A : ASTCENC_PRF_LDR;
        astcenc_error err = astcenc_config_init(profile, 4, 4, 1, gAstcPreset, flags, &cfg);
        if (err == ASTCENC_SUCCESS)
            err = astcenc_context_alloc(&cfg, threads, &ctx, NULL);
        if (err != ASTCENC_SUCCESS) {
            fprintf(stderr, "astcenc: %s\n", astcenc_get_error_string(err));
            ctx = NULL;
        }
        cachedThreads[hdr][normal][dec] = threads;
    }
    return ctx;
}

/* Decode a block image with the reference decoder (Compressonator for BCn,
 * astcenc for ASTC) into (w, h, 4) RGBA8, or RGBA half for the HDR formats
 * (A = 1.0). BC5 and ASTC normal maps fill R and G; B = 0, A = 255. */
static Image decodeBlocks(const Format& fmt, const void* blocks, uint32_t w, uint32_t h)
{
    Image out;
    out.w = w;
    out.h = h;
    out.hdr = fmt.hdr;
    if (fmt.hdr)
        out.half.resize((size_t)w * h * 4);
    else
        out.rgba.resize((size_t)w * h * 4);
    const uint32_t bx = (w + 3) / 4, by = (h + 3) / 4;
    if (fmt.astc) {
        astcenc_context* ctx = astcContext(fmt, 1, true);
        void* slice = fmt.hdr ? (void*)out.half.data() : (void*)out.rgba.data();
        astcenc_image img = { w, h, 1, fmt.hdr ? ASTCENC_TYPE_F16 : ASTCENC_TYPE_U8, &slice };
        /* normal maps: X is in L (-> R), Y in A (-> G) */
        const astcenc_swizzle swz = fmt.channels == 2 ? astcenc_swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_A, ASTCENC_SWZ_0, ASTCENC_SWZ_1 }
                                                      : astcenc_swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
        astcenc_decompress_reset(ctx);
        const astcenc_error err = astcenc_decompress_image(ctx, (const uint8_t*)blocks, (size_t)bx * by * 16, &img, &swz, 0);
        if (err != ASTCENC_SUCCESS)
            fprintf(stderr, "astcenc decode: %s\n", astcenc_get_error_string(err));
        return out;
    }
    const uint8_t* src = (const uint8_t*)blocks;
    for (uint32_t y = 0; y < by; y++) {
        for (uint32_t x = 0; x < bx; x++) {
            uint8_t px[64];
            uint16_t pxh[64];
            if (fmt.anbc == ANBC_TEXTURE_FORMAT_BC5) {
                uint8_t r[16], g[16];
                DecompressBlockBC5(src + ((size_t)y * bx + x) * 16, r, g, NULL);
                for (int i = 0; i < 16; i++) {
                    px[i * 4 + 0] = r[i];
                    px[i * 4 + 1] = g[i];
                    px[i * 4 + 2] = 0;
                    px[i * 4 + 3] = 255;
                }
            } else if (fmt.anbc == ANBC_TEXTURE_FORMAT_BC6H) {
                uint16_t rgb[48];
                DecompressBlockBC6(src + ((size_t)y * bx + x) * 16, rgb, NULL);
                for (int i = 0; i < 16; i++) {
                    pxh[i * 4 + 0] = rgb[i * 3 + 0];
                    pxh[i * 4 + 1] = rgb[i * 3 + 1];
                    pxh[i * 4 + 2] = rgb[i * 3 + 2];
                    pxh[i * 4 + 3] = 0x3C00; /* 1.0 */
                }
            } else {
                DecompressBlockBC7(src + ((size_t)y * bx + x) * 16, px, NULL);
            }
            for (uint32_t j = 0; j < 4; j++) {
                for (uint32_t i = 0; i < 4; i++) {
                    const uint32_t px_x = x * 4 + i, px_y = y * 4 + j;
                    if (px_x < w && px_y < h) {
                        if (fmt.hdr)
                            memcpy(&out.half[((size_t)px_y * w + px_x) * 4], pxh + (j * 4 + i) * 4, 8);
                        else
                            memcpy(&out.rgba[((size_t)px_y * w + px_x) * 4], px + (j * 4 + i) * 4, 4);
                    }
                }
            }
        }
    }
    return out;
}

/* `rgb` is over the format's stored colour channels (RGB for BC7/BC6H, RG
 * for BC5), `rgba` additionally includes alpha (only meaningful for BC7;
 * equal to `rgb` for BC6H, which is scored in the half-int domain). */
struct Psnr { double rgb, rgba; };

static Psnr computePsnr(const Format& fmt, const Image& a, const Image& b)
{
    const uint32_t w = a.w, h = a.h;
    const double n = (double)w * h;
    Psnr p;
    if (fmt.hdr) {
        double se = 0;
        for (size_t i = 0; i < (size_t)w * h; i++) {
            for (int c = 0; c < fmt.channels; c++) {
                const double d = halfIntNorm(a.half[i * 4 + c]) - halfIntNorm(b.half[i * 4 + c]);
                se += d * d;
            }
        }
        const double mse = se / (n * fmt.channels);
        p.rgb = p.rgba = mse > 0 ? 10 * log10(1.0 / mse) : INFINITY;
        return p;
    }
    double seRgb = 0, seA = 0;
    for (size_t i = 0; i < (size_t)w * h; i++) {
        for (int c = 0; c < fmt.channels; c++) {
            const double d = (double)a.rgba[i * 4 + c] - b.rgba[i * 4 + c];
            seRgb += d * d;
        }
        const double d = (double)a.rgba[i * 4 + 3] - b.rgba[i * 4 + 3];
        seA += d * d;
    }
    const double mseRgb = seRgb / (n * fmt.channels), mseRgba = (seRgb + seA) / (n * (fmt.channels + 1));
    p.rgb = mseRgb > 0 ? 10 * log10(255.0 * 255.0 / mseRgb) : INFINITY;
    p.rgba = mseRgba > 0 ? 10 * log10(255.0 * 255.0 / mseRgba) : INFINITY;
    return p;
}

/* CPU twin of the GPU mip generator: 2x2 box, optionally in linear light
 * (8-bit); HDR images are averaged as floats and stored back as half. */
static double srgbToLinear(double c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
static double linearToSrgb(double c) { return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1 / 2.4) - 0.055; }

static Image downsample(const Image& src, bool srgb)
{
    Image dst;
    dst.w = src.w > 1 ? src.w / 2 : 1;
    dst.h = src.h > 1 ? src.h / 2 : 1;
    dst.hdr = src.hdr;
    if (src.hdr)
        dst.half.resize((size_t)dst.w * dst.h * 4);
    else
        dst.rgba.resize((size_t)dst.w * dst.h * 4);
    for (uint32_t y = 0; y < dst.h; y++) {
        for (uint32_t x = 0; x < dst.w; x++) {
            for (int c = 0; c < 4; c++) {
                double sum = 0;
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
                        const uint32_t sx = std::min(x * 2 + i, src.w - 1);
                        const uint32_t sy = std::min(y * 2 + j, src.h - 1);
                        const size_t idx = ((size_t)sy * src.w + sx) * 4 + c;
                        double v = src.hdr ? halfToFloat(src.half[idx]) : src.rgba[idx] / 255.0;
                        if (srgb && !src.hdr && c < 3)
                            v = srgbToLinear(v);
                        sum += v;
                    }
                }
                double v = sum / 4;
                if (src.hdr) {
                    dst.half[((size_t)y * dst.w + x) * 4 + c] = floatToHalf((float)v);
                    continue;
                }
                if (srgb && c < 3)
                    v = linearToSrgb(v);
                dst.rgba[((size_t)y * dst.w + x) * 4 + c] = (uint8_t)(v * 255.0 + 0.5);
            }
        }
    }
    return dst;
}

/* Load any supported image into the representation `fmt` wants: 8-bit RGBA
 * for the LDR formats (an .hdr source is clamped to [0,1]), RGBA half for
 * BC6H (an 8-bit source becomes [0,1] linear). */
static bool loadImage(const std::string& path, const Format& fmt, Image& out)
{
    int w, h, comp;
    const bool hdrFile = stbi_is_hdr(path.c_str()) != 0;
    if (hdrFile) {
        float* f = stbi_loadf(path.c_str(), &w, &h, &comp, 4);
        if (!f)
            return false;
        out.w = (uint32_t)w;
        out.h = (uint32_t)h;
        out.hdr = fmt.hdr;
        if (fmt.hdr) {
            out.half.resize((size_t)w * h * 4);
            for (size_t i = 0; i < (size_t)w * h * 4; i++)
                out.half[i] = floatToHalf(f[i]);
        } else {
            out.rgba.resize((size_t)w * h * 4);
            for (size_t i = 0; i < (size_t)w * h * 4; i++)
                out.rgba[i] = (uint8_t)(std::min(std::max(f[i], 0.0f), 1.0f) * 255.0f + 0.5f);
        }
        stbi_image_free(f);
        return true;
    }
    uint8_t* pixels = stbi_load(path.c_str(), &w, &h, &comp, 4);
    if (!pixels)
        return false;
    out.w = (uint32_t)w;
    out.h = (uint32_t)h;
    out.hdr = fmt.hdr;
    if (fmt.hdr) {
        out.half.resize((size_t)w * h * 4);
        for (size_t i = 0; i < (size_t)w * h * 4; i++)
            out.half[i] = floatToHalf(pixels[i] / 255.0f);
    } else {
        out.rgba.assign(pixels, pixels + (size_t)w * h * 4);
    }
    stbi_image_free(pixels);
    return true;
}

/* ------------------------------------------------------------------------- */
/* Encoders                                                                  */
/* ------------------------------------------------------------------------- */

struct Level {
    uint32_t w, h;
    std::vector<uint8_t> blocks;
};

/* Compressonator (BCn) or astcenc (ASTC): CPU mips + every level, the work
 * of a level split across `threads` worker threads. */
static std::vector<Level> encodeCmpChain(const Format& fmt, const Image& src, bool mips, bool srgb, unsigned threads,
                                         float quality)
{
    const bool bc5 = fmt.anbc == ANBC_TEXTURE_FORMAT_BC5;
    const bool bc6h = fmt.anbc == ANBC_TEXTURE_FORMAT_BC6H;
    std::vector<Level> out;
    Image level = src;
    for (;;) {
        const uint32_t w = level.w, h = level.h;
        const uint32_t bx = (w + 3) / 4, by = (h + 3) / 4;
        Level L = { w, h, std::vector<uint8_t>((size_t)bx * by * 16) };

        if (fmt.astc) {
            /* astcenc splits the image across its own thread indices. */
            const unsigned n = (bx * by < 256) ? 1 : threads;
            astcenc_context* ctx = astcContext(fmt, n, false);
            void* slice = fmt.hdr ? (void*)level.half.data() : (void*)level.rgba.data();
            astcenc_image img = { w, h, 1, fmt.hdr ? ASTCENC_TYPE_F16 : ASTCENC_TYPE_U8, &slice };
            const astcenc_swizzle swz = fmt.channels == 2 ? astcenc_swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_R, ASTCENC_SWZ_G }
                                                          : astcenc_swizzle{ ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A };
            astcenc_compress_reset(ctx);
            auto worker = [&](unsigned t) {
                const astcenc_error err = astcenc_compress_image(ctx, &img, &swz, L.blocks.data(), L.blocks.size(), t);
                if (err != ASTCENC_SUCCESS)
                    fprintf(stderr, "astcenc encode: %s\n", astcenc_get_error_string(err));
            };
            if (n == 1) {
                worker(0);
            } else {
                std::vector<std::thread> pool;
                for (unsigned t = 0; t < n; t++)
                    pool.emplace_back(worker, t);
                for (auto& th : pool)
                    th.join();
            }
            out.push_back(std::move(L));
            if (!mips || (w == 1 && h == 1))
                break;
            level = downsample(level, srgb);
            continue;
        }

        auto encodeRows = [&](uint32_t y0, uint32_t y1) {
            void* options = NULL;
            if (bc5) {
                CreateOptionsBC5(&options);
                SetQualityBC5(options, quality);
            } else if (bc6h) {
                CreateOptionsBC6(&options);
                SetQualityBC6(options, quality);
                SetSignedBC6(options, false);
            } else {
                CreateOptionsBC7(&options);
                SetQualityBC7(options, quality);
                SetAlphaOptionsBC7(options, true, false, false);
            }
            for (uint32_t y = y0; y < y1; y++) {
                for (uint32_t x = 0; x < bx; x++) {
                    /* Edge blocks: gather a clamped 4x4 so partial blocks are well-defined. */
                    uint8_t block[64], r[16], g[16];
                    uint16_t blockh[48];
                    for (uint32_t j = 0; j < 4; j++) {
                        for (uint32_t i = 0; i < 4; i++) {
                            const uint32_t sx = std::min(x * 4 + i, w - 1), sy = std::min(y * 4 + j, h - 1);
                            if (bc6h) {
                                const uint16_t* p = &level.half[((size_t)sy * w + sx) * 4];
                                memcpy(blockh + (j * 4 + i) * 3, p, 6);
                                continue;
                            }
                            const uint8_t* p = &level.rgba[((size_t)sy * w + sx) * 4];
                            memcpy(block + (j * 4 + i) * 4, p, 4);
                            r[j * 4 + i] = p[0];
                            g[j * 4 + i] = p[1];
                        }
                    }
                    uint8_t* dst = &L.blocks[((size_t)y * bx + x) * 16];
                    if (bc5)
                        CompressBlockBC5(r, 4, g, 4, dst, options);
                    else if (bc6h)
                        CompressBlockBC6(blockh, 12, dst, options);
                    else
                        CompressBlockBC7(block, 16, dst, options);
                }
            }
            if (bc5)
                DestroyOptionsBC5(options);
            else if (bc6h)
                DestroyOptionsBC6(options);
            else
                DestroyOptionsBC7(options);
        };

        /* Tiny levels aren't worth spinning threads up for. */
        const unsigned n = (bx * by < 256) ? 1 : std::max(1u, std::min(threads, by));
        if (n == 1) {
            encodeRows(0, by);
        } else {
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < n; t++)
                pool.emplace_back(encodeRows, by * t / n, by * (t + 1) / n);
            for (auto& th : pool)
                th.join();
        }
        out.push_back(std::move(L));

        if (!mips || (w == 1 && h == 1))
            break;
        level = downsample(level, srgb);
    }
    return out;
}

static std::vector<Level> collectAnbcLevels(const anbcTexture* texture)
{
    std::vector<Level> out;
    for (uint32_t i = 0; i < anbcGetMipCount(texture); i++) {
        anbcMipInfo info;
        anbcGetMip(texture, i, &info);
        Level L = { info.width, info.height, std::vector<uint8_t>((const uint8_t*)info.data, (const uint8_t*)info.data + info.sizeBytes) };
        out.push_back(std::move(L));
    }
    return out;
}

/* <base>.dds when the format has a DXGI id; ASTC formats also write
 * <base>.astc (mip 0) and <base>_mipN.astc for the other levels. */
static int writeChain(const Format& fmt, const std::string& base, const std::vector<Level>& levels)
{
    int rc = 0;
    if (fmt.dxgi) {
        std::vector<ddsMip> mips;
        for (const Level& L : levels)
            mips.push_back({ L.blocks.data(), L.blocks.size(), L.w });
        rc = ddsWriteBlocks((base + ".dds").c_str(), fmt.dxgi, levels[0].w, levels[0].h, mips.data(), (uint32_t)mips.size());
    }
    if (fmt.astc) {
        for (size_t i = 0; i < levels.size(); i++) {
            const std::string path = i == 0 ? base + ".astc" : base + "_mip" + std::to_string(i) + ".astc";
            rc |= astcWriteFile(path.c_str(), levels[i].w, levels[i].h, levels[i].blocks.data(), levels[i].blocks.size());
        }
    }
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Driver                                                                    */
/* ------------------------------------------------------------------------- */

struct Options {
    const Format* format = &kFormatBC7;
#ifdef __APPLE__
    anbcDeviceBackend backend = ANBC_DEVICE_BACKEND_METAL;
#else
    anbcDeviceBackend backend = ANBC_DEVICE_BACKEND_VULKAN;
#endif
    std::string modelDir = "checkpoints";
    std::string outDir = ".";
    float cmpQuality = 0.05f;
    uint32_t refineIters = 2;
    bool srgb = false, mips = true, allowTensor = true, xcheck = false;
    unsigned threads = std::max(1u, std::thread::hardware_concurrency());
};

static bool hasImageExt(const std::string& name)
{
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos)
        return false;
    std::string ext = name.substr(dot + 1);
    for (char& c : ext)
        c = (char)tolower(c);
    return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "tga" || ext == "bmp" || ext == "hdr";
}

static std::string stemOf(const std::string& path)
{
    const size_t slash = path.rfind('/');
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = base.rfind('.');
    return dot == std::string::npos ? base : base.substr(0, dot);
}

struct Result {
    std::string stem;
    uint32_t w, h, mips;
    double anbcMs, cmpMs;
};

static bool processImage(anbcDevice* device, const anbcDeviceInfo& devInfo, const std::string& imagePath,
                         const Options& opt, bool verbose, Result* result)
{
    const Format& fmt = *opt.format;
    Image src;
    if (!loadImage(imagePath, fmt, src)) {
        fprintf(stderr, "failed to load %s: %s\n", imagePath.c_str(), stbi_failure_reason());
        return false;
    }
    const std::string stem = stemOf(imagePath);

    /* ---- anbc ---------------------------------------------------------- */
    anbcTextureDesc desc = {};
    desc.width = src.w;
    desc.height = src.h;
    desc.pixels = fmt.hdr ? (const void*)src.half.data() : (const void*)src.rgba.data();
    desc.pixelFormat = fmt.hdr ? ANBC_PIXEL_FORMAT_RGBA16_FLOAT : ANBC_PIXEL_FORMAT_RGBA8_UNORM;
    desc.flags = (opt.srgb && !fmt.hdr ? ANBC_TEXTURE_FLAG_SRGB : 0) | (opt.mips ? ANBC_TEXTURE_FLAG_GENERATE_MIPS : 0) |
                 (fmt.astc && fmt.channels == 2 ? ANBC_TEXTURE_FLAG_NORMAL_MAP : 0);
    anbcTexture* texture = anbcCreateTexture(device, &desc);
    if (!texture) {
        fprintf(stderr, "anbcCreateTexture failed for %s\n", imagePath.c_str());
        return false;
    }

    /* Scalar first, then the tensor-ops kernel where available; the texture
     * keeps the last result, which is what gets written out and scored. */
    struct Run { const char* label; uint32_t flags; double ms; Psnr p; } runs[2] = {
        { fmt.tensorKernel ? "anbc scalar" : "anbc", ANBC_COMPRESS_FLAG_NO_TENSOR_OPS, 0, { 0, 0 } },
        { "anbc tensor ops", 0, 0, { 0, 0 } },
    };
    const bool tensor = devInfo.tensorOps && opt.allowTensor && fmt.tensorKernel;
    const bool both = tensor && verbose;
    const int firstRun = both ? 0 : (tensor ? 1 : 0);
    const int lastRun = tensor ? 1 : 0;
    const int timedIters = verbose ? 5 : 3;
    for (int k = firstRun; k <= lastRun; k++) {
        anbcCompressOptions copts = { opt.refineIters, runs[k].flags };
        anbcResult r = anbcCompress(device, texture, fmt.anbc, &copts); /* warm-up */
        if (r != ANBC_OK) {
            fprintf(stderr, "anbcCompress (%s): %s\n", runs[k].label, anbcResultString(r));
            anbcDestroyTexture(texture);
            return false;
        }
        double best = INFINITY;
        for (int i = 0; i < timedIters; i++) {
            const double t0 = now();
            anbcCompress(device, texture, fmt.anbc, &copts);
            best = std::min(best, now() - t0);
        }
        runs[k].ms = best * 1000.0;
        if (verbose) {
            anbcMipInfo mip0;
            anbcGetMip(texture, 0, &mip0);
            const Image dec = decodeBlocks(fmt, mip0.data, src.w, src.h);
            runs[k].p = computePsnr(fmt, src, dec);
        }
    }
    const std::vector<Level> anbcLevels = collectAnbcLevels(texture);
    anbcDestroyTexture(texture);

    /* ---- Compressonator ------------------------------------------------ */
    const double c0 = now();
    const std::vector<Level> cmpLevels =
        encodeCmpChain(fmt, src, opt.mips, opt.srgb && !fmt.hdr, opt.threads, opt.cmpQuality);
    const double cmpMs = (now() - c0) * 1000.0;

    /* ---- outputs --------------------------------------------------------- */
    const std::string outExt = fmt.dxgi ? ".dds" : ".astc";
    const std::string anbcPath = opt.outDir + "/" + stem + "_anbc" + outExt;
    const std::string cmpPath = opt.outDir + "/" + stem + "_cmp" + outExt;
    writeChain(fmt, opt.outDir + "/" + stem + "_anbc", anbcLevels);
    writeChain(fmt, opt.outDir + "/" + stem + "_cmp", cmpLevels);

    if (result) {
        result->stem = stem;
        result->w = src.w;
        result->h = src.h;
        result->mips = (uint32_t)anbcLevels.size();
        result->anbcMs = runs[lastRun].ms;
        result->cmpMs = cmpMs;
    }
    if (!verbose)
        return true;

    /* ---- single-image report --------------------------------------------- */
    const Image cmpDec = decodeBlocks(fmt, cmpLevels[0].blocks.data(), src.w, src.h);
    const Psnr pc = computePsnr(fmt, src, cmpDec);

    /* Alpha is only stored by BC7 and ASTC; for BC5 / normal maps / HDR the second column is left blank. */
    const bool hasAlpha = fmt.anbc == ANBC_TEXTURE_FORMAT_BC7 || (fmt.anbc == ANBC_TEXTURE_FORMAT_ASTC_4x4_UNORM && fmt.channels == 3);
    const char* refName = fmt.astc ? "astcenc" : "Compressonator CMP_Core";
    char refDetail[64];
    if (fmt.astc)
        snprintf(refDetail, sizeof(refDetail), "-%s, %u threads", gAstcPresetName, opt.threads);
    else
        snprintf(refDetail, sizeof(refDetail), "quality %.2f, %u threads", opt.cmpQuality, opt.threads);
    auto alphaCol = [&](double v) {
        static char buf[32];
        if (hasAlpha) snprintf(buf, sizeof(buf), "%11.2fdB", v);
        else snprintf(buf, sizeof(buf), "%13s", "-");
        return buf;
    };
    printf("\n%-28s %10s %14s %14s\n", "", "time", fmt.psnrLabel, hasAlpha ? "PSNR RGBA" : "");
    bool ok = true;
    for (int k = firstRun; k <= lastRun; k++) {
        printf("%-28s %8.2fms %11.2fdB %s   (%zu mips, refine %u, GPU)\n", runs[k].label,
               runs[k].ms, runs[k].p.rgb, alphaCol(runs[k].p.rgba), anbcLevels.size(), opt.refineIters);
        if (runs[k].p.rgb < fmt.floorMip0)
            ok = false;
    }
    if (both)
        printf("%-28s %8.2fx  %+10.2fdB\n", "  tensor vs scalar", runs[0].ms / runs[1].ms, runs[1].p.rgb - runs[0].p.rgb);
    printf("%-28s %8.2fms %11.2fdB %s   (%zu mips, %s)\n", refName, cmpMs, pc.rgb, alphaCol(pc.rgba), cmpLevels.size(),
           refDetail);

    if (anbcLevels.size() > 1) {
        printf("\nanbc mip chain vs CPU reference (%s):\n",
               fmt.hdr ? "float box" : opt.srgb ? "sRGB-linear box" : "8-bit box");
        Image ref = src;
        for (size_t level = 1; level < anbcLevels.size(); level++) {
            ref = downsample(ref, opt.srgb && !fmt.hdr);
            const Level& L = anbcLevels[level];
            if (L.w != ref.w || L.h != ref.h) {
                printf("  level %2zu: size mismatch (%ux%u vs %ux%u)\n", level, L.w, L.h, ref.w, ref.h);
                ok = false;
                continue;
            }
            const Image dec = decodeBlocks(fmt, L.blocks.data(), L.w, L.h);
            const Psnr p = computePsnr(fmt, ref, dec);
            if (hasAlpha)
                printf("  level %2zu  %5ux%-5u  PSNR RGB %6.2fdB  RGBA %6.2fdB\n", level, L.w, L.h, p.rgb, p.rgba);
            else
                printf("  level %2zu  %5ux%-5u  %s %6.2fdB\n", level, L.w, L.h, fmt.psnrLabel, p.rgb);
            if (p.rgb < fmt.floorMips)
                ok = false;
        }
    }
    printf("\nwrote %s\nwrote %s\n", anbcPath.c_str(), cmpPath.c_str());
    if (!ok)
        fprintf(stderr, "\nFAILED: quality below sanity floor\n");
    return ok;
}

/* Encodes `src` on `backend` (loading the format's models from opt.modelDir)
 * and returns the mip chain; empty on failure. */
static std::vector<Level> encodeOnBackend(anbcDeviceBackend backend, const Image& src, const Options& opt)
{
    const Format& fmt = *opt.format;
    std::vector<Level> levels;
    anbcDevice* device = anbcCreateDevice(backend);
    if (!device) {
        fprintf(stderr, "xcheck: anbcCreateDevice(%s) failed\n", backend == ANBC_DEVICE_BACKEND_METAL ? "METAL" : "VULKAN");
        return levels;
    }
    for (const char* const* f = fmt.modelFiles; *f; f++) {
        const std::string path = opt.modelDir + "/" + *f;
        if (anbcLoadModel(device, fmt.anbc, path.c_str()) != ANBC_OK) {
            fprintf(stderr, "xcheck: anbcLoadModel(%s) failed\n", path.c_str());
            anbcDestroyDevice(device);
            return levels;
        }
    }
    anbcTextureDesc desc = {};
    desc.width = src.w;
    desc.height = src.h;
    desc.pixels = fmt.hdr ? (const void*)src.half.data() : (const void*)src.rgba.data();
    desc.pixelFormat = fmt.hdr ? ANBC_PIXEL_FORMAT_RGBA16_FLOAT : ANBC_PIXEL_FORMAT_RGBA8_UNORM;
    desc.flags = (opt.srgb && !fmt.hdr ? ANBC_TEXTURE_FLAG_SRGB : 0) | (opt.mips ? ANBC_TEXTURE_FLAG_GENERATE_MIPS : 0) |
                 (fmt.astc && fmt.channels == 2 ? ANBC_TEXTURE_FLAG_NORMAL_MAP : 0);
    anbcTexture* texture = anbcCreateTexture(device, &desc);
    anbcCompressOptions copts = { opt.refineIters, ANBC_COMPRESS_FLAG_NO_TENSOR_OPS };
    if (texture && anbcCompress(device, texture, fmt.anbc, &copts) == ANBC_OK)
        levels = collectAnbcLevels(texture);
    else
        fprintf(stderr, "xcheck: encode failed\n");
    anbcDestroyTexture(texture);
    anbcDestroyDevice(device);
    return levels;
}

static bool crossCheck(const std::string& imagePath, const Options& opt)
{
    const Format& fmt = *opt.format;
    Image src;
    if (!loadImage(imagePath, fmt, src))
        return false;
    const std::vector<Level> metal = encodeOnBackend(ANBC_DEVICE_BACKEND_METAL, src, opt);
    const std::vector<Level> vulkan = encodeOnBackend(ANBC_DEVICE_BACKEND_VULKAN, src, opt);
    if (metal.empty() || vulkan.empty() || metal.size() != vulkan.size())
        return false;

    printf("\nMetal (scalar) vs Vulkan, %s:\n", fmt.name);
    printf("  %-5s %11s %12s %12s %10s\n", "level", "size", "Metal", "Vulkan", "identical");
    bool ok = true;
    Image ref = src;
    for (size_t level = 0; level < metal.size(); level++) {
        const Level& a = metal[level];
        const Level& b = vulkan[level];
        if (level > 0)
            ref = downsample(ref, opt.srgb && !fmt.hdr);
        const Psnr pa = computePsnr(fmt, ref, decodeBlocks(fmt, a.blocks.data(), a.w, a.h));
        const Psnr pb = computePsnr(fmt, ref, decodeBlocks(fmt, b.blocks.data(), b.w, b.h));
        size_t same = 0;
        const size_t blocks = a.blocks.size() / 16;
        for (size_t i = 0; i < blocks; i++)
            same += memcmp(&a.blocks[i * 16], &b.blocks[i * 16], 16) == 0;
        char size[32];
        snprintf(size, sizeof(size), "%ux%u", a.w, a.h);
        printf("  %-5zu %11s %10.2fdB %10.2fdB %9.1f%%\n", level, size, pa.rgb, pb.rgb, 100.0 * same / blocks);
        if (fabs(pa.rgb - pb.rgb) > 0.05)
            ok = false;
    }
    if (!ok)
        fprintf(stderr, "\nFAILED: Metal and Vulkan differ by more than 0.05 dB\n");
    return ok;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <image-or-folder> [--format bc7|bc6h|bc5|astc|astc-float] [--backend metal|vulkan]\n"
                "          [--xcheck] [--normal-map] [--out dir]\n"
                "          [--models dir] [--srgb] [--cmp-quality q] [--astc-preset fast|medium|thorough]\n"
                "          [--refine-iters n] [--no-mips] [--no-tensor-ops] [--threads n] [--single-thread]\n",
                argv[0]);
        return 2;
    }
    const std::string input = argv[1];
    Options opt;
    bool normalMap = false;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--format" && i + 1 < argc) {
            const std::string f = argv[++i];
            if (f == "bc7") opt.format = &kFormatBC7;
            else if (f == "bc6h") opt.format = &kFormatBC6H;
            else if (f == "bc5") opt.format = &kFormatBC5;
            else if (f == "astc") opt.format = &kFormatASTC;
            else if (f == "astc-float") opt.format = &kFormatASTCFloat;
            else { fprintf(stderr, "unknown format %s (bc7|bc6h|bc5|astc|astc-float)\n", f.c_str()); return 2; }
        }
        else if (a == "--backend" && i + 1 < argc) {
            const std::string b = argv[++i];
            if (b == "metal") opt.backend = ANBC_DEVICE_BACKEND_METAL;
            else if (b == "vulkan") opt.backend = ANBC_DEVICE_BACKEND_VULKAN;
            else { fprintf(stderr, "unknown backend %s (metal|vulkan)\n", b.c_str()); return 2; }
        }
        else if (a == "--xcheck") opt.xcheck = true;
        else if (a == "--normal-map") normalMap = true;
        else if (a == "--astc-preset" && i + 1 < argc) {
            const std::string q = argv[++i];
            if (q == "fast") gAstcPreset = ASTCENC_PRE_FAST;
            else if (q == "medium") gAstcPreset = ASTCENC_PRE_MEDIUM;
            else if (q == "thorough") gAstcPreset = ASTCENC_PRE_THOROUGH;
            else { fprintf(stderr, "unknown astc preset %s (fast|medium|thorough)\n", q.c_str()); return 2; }
            gAstcPresetName = q == "fast" ? "fast" : q == "medium" ? "medium" : "thorough";
        }
        else if (a == "--models" && i + 1 < argc) opt.modelDir = argv[++i];
        else if (a == "--out" && i + 1 < argc) opt.outDir = argv[++i];
        else if (a == "--cmp-quality" && i + 1 < argc) opt.cmpQuality = (float)atof(argv[++i]);
        else if (a == "--refine-iters" && i + 1 < argc) opt.refineIters = (uint32_t)atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) opt.threads = (unsigned)std::max(1, atoi(argv[++i]));
        else if (a == "--single-thread") opt.threads = 1;
        else if (a == "--srgb") opt.srgb = true;
        else if (a == "--no-mips") opt.mips = false;
        else if (a == "--no-tensor-ops") opt.allowTensor = false;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (normalMap) {
        if (opt.format != &kFormatASTC) {
            fprintf(stderr, "--normal-map only applies to --format astc\n");
            return 2;
        }
        opt.format = &kFormatASTCNormal;
    }

    /* Folder or single image? */
    struct stat st;
    if (stat(input.c_str(), &st) != 0) {
        fprintf(stderr, "cannot stat %s\n", input.c_str());
        return 1;
    }
    std::vector<std::string> images;
    const bool folder = S_ISDIR(st.st_mode);
    if (folder) {
        DIR* d = opendir(input.c_str());
        for (struct dirent* e; (e = readdir(d)) != NULL;)
            if (hasImageExt(e->d_name))
                images.push_back(input + "/" + e->d_name);
        closedir(d);
        std::sort(images.begin(), images.end());
        if (images.empty()) {
            fprintf(stderr, "no images in %s\n", input.c_str());
            return 1;
        }
    } else {
        images.push_back(input);
    }
    mkdir(opt.outDir.c_str(), 0755);

    /* ---- device + models -------------------------------------------------- */
    anbcDevice* device = anbcCreateDevice(opt.backend);
    if (!device) {
        fprintf(stderr, "anbcCreateDevice(%s) failed\n", opt.backend == ANBC_DEVICE_BACKEND_METAL ? "METAL" : "VULKAN");
        return 1;
    }
    anbcDeviceInfo devInfo;
    anbcGetDeviceInfo(device, &devInfo);
    printf("device: %s (%s), tensor ops: %s, format: %s, Compressonator threads: %u\n", devInfo.name,
           devInfo.backend, devInfo.tensorOps ? "yes" : "no", opt.format->name, opt.threads);
    for (const char* const* f = opt.format->modelFiles; *f; f++) {
        const std::string path = opt.modelDir + "/" + *f;
        const anbcResult r = anbcLoadModel(device, opt.format->anbc, path.c_str());
        if (r != ANBC_OK) {
            fprintf(stderr, "anbcLoadModel(%s): %s\n", path.c_str(), anbcResultString(r));
            return 1;
        }
        printf("loaded %s\n", path.c_str());
    }

    int rc = 0;
    if (!folder) {
        printf("%s\n", input.c_str());
        rc = processImage(device, devInfo, input, opt, true, NULL) ? 0 : 1;
        if (opt.xcheck && !crossCheck(input, opt))
            rc = 1;
    } else {
        std::vector<Result> results;
        double totalAnbc = 0, totalCmp = 0;
        printf("\n%-48s %10s %5s %10s %10s %8s\n", "texture", "size", "mips", "anbc", "cmp", "speedup");
        for (const std::string& path : images) {
            Result r;
            if (!processImage(device, devInfo, path, opt, false, &r)) {
                rc = 1;
                continue;
            }
            results.push_back(r);
            totalAnbc += r.anbcMs;
            totalCmp += r.cmpMs;
            char size[32];
            snprintf(size, sizeof(size), "%ux%u", r.w, r.h);
            printf("%-48.48s %10s %5u %8.2fms %8.1fms %7.1fx\n", r.stem.c_str(), size, r.mips, r.anbcMs, r.cmpMs,
                   r.cmpMs / r.anbcMs);
            fflush(stdout);
        }
        printf("%s\n", std::string(96, '-').c_str());
        printf("%zu textures: anbc %.1f ms total (%s), %s %.1f ms total (%u threads), speedup %.1fx\n",
               results.size(), totalAnbc,
               opt.format->tensorKernel ? (devInfo.tensorOps && opt.allowTensor ? "tensor ops" : "scalar") : "GPU",
               opt.format->astc ? "astcenc" : "Compressonator", totalCmp, opt.threads, totalCmp / totalAnbc);

        const std::string csvPath = opt.outDir + "/summary.csv";
        FILE* csv = fopen(csvPath.c_str(), "w");
        if (csv) {
            fprintf(csv, "name,width,height,mips,anbc_ms,cmp_ms,threads\n");
            for (const Result& r : results)
                fprintf(csv, "%s,%u,%u,%u,%.3f,%.3f,%u\n", r.stem.c_str(), r.w, r.h, r.mips, r.anbcMs, r.cmpMs, opt.threads);
            fclose(csv);
            printf("wrote %s\n", csvPath.c_str());
        }
    }

    anbcDestroyDevice(device);
    return rc;
}
