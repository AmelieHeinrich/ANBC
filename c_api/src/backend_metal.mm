/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * Metal 4 backend. Everything goes through the MTL4 core API (command
 * allocator + reusable command buffer, argument tables, residency set on the
 * queue, MTL4Compiler for runtime shader/pipeline creation) so the MLP can
 * later move to MTL4MachineLearningCommandEncoder / shader tensors without
 * restructuring the command flow.
 *
 * Per anbcCompress: one command buffer, one compute encoder, one commit:
 *   [anbc_mips]  -> barrier -> anbc_bc7_encode | anbc_bc6h_encode | anbc_bc5_encode x mipCount
 *                -> signal event -> CPU wait
 * BC7 runs the two MLPs (scalar or tensor-ops kernel); BC6H and BC5 have no
 * network (block min/max + refinement, one scalar kernel each).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "anbc_internal.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "anbc_bc5_metal.h"
#include "anbc_bc6h_metal.h"
#include "anbc_bc7_scalar_metal.h"
#include "anbc_mips_metal.h"
#if ANBC_HAVE_TENSOR_METALLIB
#include "anbc_bc7_tensor_metallib.h"
#endif

namespace {

// Must match the structs in the .metal sources.
constexpr uint32_t kMipsMaxLevels = 13;
constexpr uint32_t kMipsTile = 64;
constexpr uint32_t kMaxTextureDim = 4096; // with mips: mip 6 must fit one 64x64 tile
constexpr uint32_t kMlpMaxHidden = 128;
constexpr uint32_t kMlpMaxOut = 40;
constexpr uint32_t kMlpIn = 64; // 16 RGBA texels

struct ModelHeader {
    uint32_t numLayers;
    uint32_t dims[ANBC_MAX_MODEL_LAYERS + 1];
    uint32_t layerOffset[ANBC_MAX_MODEL_LAYERS];
};
constexpr size_t kModelWeightsOffset = 256; // header lives at 0, weights at 256

struct MipParams {
    uint32_t numMips;
    uint32_t numGroups;
    uint32_t pad0, pad1;
    uint32_t dims[kMipsMaxLevels][4];
};

struct EncodeParams {
    uint32_t mipLevel;
    uint32_t blocksX;
    uint32_t blocksY;
    uint32_t refineIters;
    uint32_t mipWidth;
    uint32_t mipHeight;
    uint32_t pad0, pad1;
};

constexpr size_t kParamsMipOffset = 0;
constexpr size_t kParamsEncodeOffset = 512;
constexpr size_t kParamsEncodeStride = 64;
constexpr size_t kParamsBytes = kParamsEncodeOffset + kParamsEncodeStride * ANBC_MAX_MIPS;

// Tensor-ops kernel geometry (anbc_bc7_tensor.metal): one block per thread,
// 32 per SIMD group, 2 SIMD groups per threadgroup.
constexpr uint32_t kTensorBlocksPerGroup = 64;
constexpr uint32_t kTensorThreadsPerGroup = 64;
constexpr uint32_t kTensorLayers = 4;
constexpr uint32_t kTensorBiasStride = 128; // halves per layer in the bias block
constexpr size_t kTensorAlign = 128;

// Argument table slots (see the [[buffer(N)]] / [[texture(N)]] in the shaders).
enum : NSUInteger {
    kBufParams = 0,
    kBufMipsCounter = 1,
    kBufModel6Header = 1,
    kBufModel6Weights = 2,
    kBufModel5Header = 3,
    kBufModel5Weights = 4,
    kBufOutput = 5,
    kBufTensor6 = 6,  // 4 weight tensors then the bias buffer
    kBufTensor5 = 11,
    kBufCount = 16,
    kTexSource = 0,
    kTexMipsBase = 1,
    kTexCount = kMipsMaxLevels,
};

// One model's weights as MTLTensors, sub-allocated from a single buffer.
struct TensorModel {
    id<MTLBuffer> arena;
    id<MTLTensor> weights[kTensorLayers];
    size_t biasOffset = 0;
};

struct MetalDevice {
    id<MTLDevice> device;
    id<MTL4CommandQueue> queue;
    id<MTL4CommandAllocator> allocator;
    id<MTL4CommandBuffer> commandBuffer;
    id<MTL4Compiler> compiler;
    id<MTL4ArgumentTable> argumentTable;
    id<MTLResidencySet> residencySet;
    id<MTLSharedEvent> event;
    uint64_t eventValue = 0;

    id<MTLLibrary> bc7Library;
    id<MTLLibrary> bc6hLibrary;
    id<MTLLibrary> bc5Library;
    id<MTLLibrary> mipsLibrary;
    id<MTLComputePipelineState> encodePipeline;  // BC7 scalar
    id<MTLComputePipelineState> bc6hPipeline;    // BC6H (no network, so no tensor variant)
    id<MTLComputePipelineState> bc5Pipeline;     // BC5 (same)
    id<MTLComputePipelineState> mipsPipeline[2]; // [srgb]

    // Tensor-ops path for the BC7 MLPs (M5-class GPUs only).
    bool tensorOps = false;
    id<MTLLibrary> tensorLibrary;
    id<MTLComputePipelineState> tensorPipeline;
    TensorModel tensor6;
    TensorModel tensor5;

    id<MTLBuffer> model6;
    id<MTLBuffer> model5;
    id<MTLBuffer> params;
    id<MTLBuffer> counter;
    std::string name;
};

struct MetalTexture {
    id<MTLTexture> texture;
    NSMutableArray<id<MTLTexture>>* mipViews; // level 1..N-1
    id<MTLBuffer> output;
};

MetalDevice* md(anbcDevice* d) { return static_cast<MetalDevice*>(d->backendData); }
MetalTexture* mt(anbcTexture* t) { return static_cast<MetalTexture*>(t->backendData); }

void logError(const char* what, NSError* error)
{
    fprintf(stderr, "anbc[metal]: %s%s%s\n", what, error ? ": " : "",
            error ? error.localizedDescription.UTF8String : "");
}

id<MTLLibrary> compileLibrary(MetalDevice* m, const char* source, const char* name)
{
    MTL4LibraryDescriptor* desc = [[MTL4LibraryDescriptor alloc] init];
    desc.source = [NSString stringWithUTF8String:source];
    desc.name = [NSString stringWithUTF8String:name];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion4_0;
    desc.options = opts;
    NSError* error = nil;
    id<MTLLibrary> lib = [m->compiler newLibraryWithDescriptor:desc error:&error];
    if (!lib)
        logError(name, error);
    return lib;
}

id<MTLComputePipelineState> buildPipeline(MetalDevice* m, id<MTLLibrary> lib, const char* fn,
                                          MTLFunctionConstantValues* constants)
{
    MTL4LibraryFunctionDescriptor* fdesc = [[MTL4LibraryFunctionDescriptor alloc] init];
    fdesc.library = lib;
    fdesc.name = [NSString stringWithUTF8String:fn];

    MTL4ComputePipelineDescriptor* pdesc = [[MTL4ComputePipelineDescriptor alloc] init];
    if (constants) {
        MTL4SpecializedFunctionDescriptor* sdesc = [[MTL4SpecializedFunctionDescriptor alloc] init];
        sdesc.functionDescriptor = fdesc;
        sdesc.constantValues = constants;
        pdesc.computeFunctionDescriptor = sdesc;
    } else {
        pdesc.computeFunctionDescriptor = fdesc;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pso = [m->compiler newComputePipelineStateWithDescriptor:pdesc
                                                                          compilerTaskOptions:nil
                                                                                        error:&error];
    if (!pso)
        logError(fn, error);
    return pso;
}

id<MTLComputePipelineState> mipsPipeline(MetalDevice* m, bool srgb)
{
    if (!m->mipsPipeline[srgb]) {
        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        bool v = srgb;
        [cv setConstantValue:&v type:MTLDataTypeBool atIndex:1];
        m->mipsPipeline[srgb] = buildPipeline(m, m->mipsLibrary, "anbc_mips", cv);
    }
    return m->mipsPipeline[srgb];
}

void makeResident(MetalDevice* m, id<MTLAllocation> allocation)
{
    [m->residencySet addAllocation:allocation];
    [m->residencySet commit];
    [m->residencySet requestResidency];
}

void dropResident(MetalDevice* m, id<MTLAllocation> allocation)
{
    if (!allocation)
        return;
    [m->residencySet removeAllocation:allocation];
    [m->residencySet commit];
}

// ---------------------------------------------------------------------------
// Backend entry points
// ---------------------------------------------------------------------------

void metalDestroy(anbcDevice* device)
{
    delete md(device);
    device->backendData = nullptr;
}

// The tensor kernel hardcodes the layer geometry (64 -> 128 -> 128 -> 128 -> out).
bool tensorModelSupported(const anbcModel* model)
{
    if (model->numLayers != kTensorLayers || model->dims[0] != kMlpIn)
        return false;
    for (uint32_t i = 1; i < kTensorLayers; i++)
        if (model->dims[i] != kMlpMaxHidden)
            return false;
    const uint32_t out = model->dims[kTensorLayers];
    return (model->mode == 6 && out == 24) || (model->mode == 5 && out == 40);
}

// Final layer padded to what the tensor kernel declares (anbc_bc7_tensor.metal).
uint32_t tensorPaddedOut(const anbcModel* model) { return model->mode == 6 ? 32 : 64; }

NSString* modelLabel(const anbcModel* model, NSString* suffix)
{
    return [(model->mode == 6 ? @"anbc bc7 mode6 " : @"anbc bc7 mode5 ") stringByAppendingString:suffix];
}

static inline size_t alignUp(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

// Builds fp16 weight tensors for `model` inside one shared buffer: layer
// tensors first (each at a 128-byte aligned offset sized by
// tensorSizeAndAlignWithDescriptor:), then the padded biases.
bool buildTensorModel(MetalDevice* m, const anbcModel* model, TensorModel& out)
{
    MTLTensorDescriptor* descs[kTensorLayers];
    size_t offsets[kTensorLayers];
    size_t cursor = 0;
    for (uint32_t l = 0; l < kTensorLayers; l++) {
        const NSInteger in = model->dims[l];
        const NSInteger rows = (l + 1 == kTensorLayers) ? tensorPaddedOut(model) : model->dims[l + 1];
        const NSInteger dims[2] = { in, rows };    // dimension 0 is innermost (contiguous)
        const NSInteger strides[2] = { 1, in };
        MTLTensorDescriptor* d = [[MTLTensorDescriptor alloc] init];
        d.dataType = MTLTensorDataTypeFloat16;
        d.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
        d.strides = [[MTLTensorExtents alloc] initWithRank:2 values:strides];
        d.usage = MTLTensorUsageCompute;
        d.storageMode = MTLStorageModeShared;
        const MTLSizeAndAlign sa = [m->device tensorSizeAndAlignWithDescriptor:d];
        cursor = alignUp(cursor, std::max<size_t>(sa.align, kTensorAlign));
        offsets[l] = cursor;
        cursor += sa.size;
        descs[l] = d;
    }
    cursor = alignUp(cursor, kTensorAlign);
    const size_t biasOffset = cursor;
    cursor += kTensorLayers * kTensorBiasStride * sizeof(_Float16);

    id<MTLBuffer> arena = [m->device newBufferWithLength:cursor options:MTLResourceStorageModeShared];
    if (!arena)
        return false;
    arena.label = modelLabel(model, @"tensors");
    memset(arena.contents, 0, cursor); // zero padding rows / biases

    for (uint32_t l = 0; l < kTensorLayers; l++) {
        const uint32_t in = model->dims[l], outN = model->dims[l + 1];
        const float* w = model->data + model->layerOffset[l];
        const float* b = w + (size_t)in * outN;
        _Float16* dstW = (_Float16*)((uint8_t*)arena.contents + offsets[l]);
        for (size_t i = 0; i < (size_t)in * outN; i++)
            dstW[i] = (_Float16)w[i];
        _Float16* dstB = (_Float16*)((uint8_t*)arena.contents + biasOffset) + l * kTensorBiasStride;
        for (uint32_t i = 0; i < outN; i++)
            dstB[i] = (_Float16)b[i];

        NSError* error = nil;
        out.weights[l] = [arena newTensorWithDescriptor:descs[l] offset:offsets[l] error:&error];
        if (!out.weights[l]) {
            logError("newTensorWithDescriptor", error);
            return false;
        }
    }
    out.arena = arena;
    out.biasOffset = biasOffset;
    return true;
}

// The tensor pipeline is one function for both models; (re)built once both
// weight sets exist.
id<MTLComputePipelineState> tensorPipeline(MetalDevice* m)
{
    if (!m->tensorPipeline && m->tensor6.arena && m->tensor5.arena)
        m->tensorPipeline = buildPipeline(m, m->tensorLibrary, "anbc_bc7_encode_tensor", nil);
    return m->tensorPipeline;
}

// BC5 / BC6H pipelines, built on first use (BC7-only users never pay for them).
id<MTLComputePipelineState> bc5Pipeline(MetalDevice* m)
{
    if (!m->bc5Pipeline)
        m->bc5Pipeline = buildPipeline(m, m->bc5Library, "anbc_bc5_encode", nil);
    return m->bc5Pipeline;
}

id<MTLComputePipelineState> bc6hPipeline(MetalDevice* m)
{
    if (!m->bc6hPipeline)
        m->bc6hPipeline = buildPipeline(m, m->bc6hLibrary, "anbc_bc6h_encode", nil);
    return m->bc6hPipeline;
}

anbcResult metalUploadModel(anbcDevice* device, const anbcModel* model)
{
    MetalDevice* m = md(device);
    if (model->dims[0] != kMlpIn || model->dims[model->numLayers] > kMlpMaxOut)
        return ANBC_ERROR_BAD_MODEL;
    for (uint32_t i = 0; i <= model->numLayers; i++)
        if (model->dims[i] > (i == 0 ? kMlpIn : i == model->numLayers ? kMlpMaxOut : kMlpMaxHidden) ||
            model->dims[i] % 4 != 0) // the kernel streams weights/activations as float4
            return ANBC_ERROR_BAD_MODEL;

    const size_t bytes = kModelWeightsOffset + model->dataFloats * sizeof(float);
    id<MTLBuffer> buf = [m->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf)
        return ANBC_ERROR_BACKEND;
    buf.label = modelLabel(model, @"weights");

    ModelHeader header = {};
    header.numLayers = model->numLayers;
    for (uint32_t i = 0; i <= model->numLayers; i++)
        header.dims[i] = model->dims[i];
    for (uint32_t i = 0; i < model->numLayers; i++)
        header.layerOffset[i] = (uint32_t)model->layerOffset[i];
    memcpy(buf.contents, &header, sizeof(header));
    memcpy((uint8_t*)buf.contents + kModelWeightsOffset, model->data, model->dataFloats * sizeof(float));

    if (model->mode == 6) {
        dropResident(m, m->model6);
        m->model6 = buf;
    } else {
        dropResident(m, m->model5);
        m->model5 = buf;
    }
    makeResident(m, buf);

    if (m->tensorOps) {
        TensorModel& slot = (model->mode == 6) ? m->tensor6 : m->tensor5;
        dropResident(m, slot.arena);
        slot = TensorModel();
        if (tensorModelSupported(model)) {
            if (buildTensorModel(m, model, slot))
                makeResident(m, slot.arena);
            else
                slot = TensorModel();
        } else {
            fprintf(stderr, "anbc[metal]: %s model geometry not supported by the tensor kernel, using scalar\n",
                    modelLabel(model, @"").UTF8String);
        }
    }
    return ANBC_OK;
}

anbcResult metalCreateTexture(anbcDevice* device, anbcTexture* texture, const anbcTextureDesc* desc)
{
    MetalDevice* m = md(device);
    // The size limit comes from the single-dispatch mip generator; a texture
    // encoded without mips only has to fit the GPU (16384^2 on every Apple GPU).
    const uint32_t maxDim = texture->mipCount > 1 ? kMaxTextureDim : 16384;
    if (desc->width > maxDim || desc->height > maxDim)
        return ANBC_ERROR_UNSUPPORTED;

    const MTLPixelFormat pixelFormat = desc->pixelFormat == ANBC_PIXEL_FORMAT_RGBA16_FLOAT ? MTLPixelFormatRGBA16Float
                                                                                          : MTLPixelFormatRGBA8Unorm;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                                                  width:desc->width
                                                                                 height:desc->height
                                                                              mipmapped:texture->mipCount > 1];
    td.mipmapLevelCount = texture->mipCount;
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    MetalTexture* t = new MetalTexture();
    t->texture = [m->device newTextureWithDescriptor:td];
    if (!t->texture) {
        delete t;
        return ANBC_ERROR_BACKEND;
    }
    t->texture.label = @"anbc source";
    [t->texture replaceRegion:MTLRegionMake2D(0, 0, desc->width, desc->height)
                  mipmapLevel:0
                    withBytes:desc->pixels
                  bytesPerRow:desc->rowPitch];

    t->mipViews = [NSMutableArray array];
    for (uint32_t level = 1; level < texture->mipCount; level++) {
        id<MTLTexture> view = [t->texture newTextureViewWithPixelFormat:pixelFormat
                                                            textureType:MTLTextureType2D
                                                                 levels:NSMakeRange(level, 1)
                                                                 slices:NSMakeRange(0, 1)];
        [t->mipViews addObject:view];
    }

    t->output = [m->device newBufferWithLength:texture->compressedBytes options:MTLResourceStorageModeShared];
    if (!t->output) {
        delete t;
        return ANBC_ERROR_BACKEND;
    }
    t->output.label = @"anbc compressed output";

    makeResident(m, t->texture);
    makeResident(m, t->output);

    texture->backendData = t;
    texture->compressed = t->output.contents;
    return ANBC_OK;
}

void metalDestroyTexture(anbcDevice* device, anbcTexture* texture)
{
    MetalDevice* m = md(device);
    MetalTexture* t = mt(texture);
    if (!t)
        return;
    dropResident(m, t->texture);
    dropResident(m, t->output);
    delete t;
    texture->backendData = nullptr;
}

anbcResult metalCompress(anbcDevice* device, anbcTexture* texture, anbcTextureFormat format,
                         const anbcCompressOptions* options)
{
    MetalDevice* m = md(device);
    MetalTexture* t = mt(texture);
    const bool generateMips = (texture->flags & ANBC_TEXTURE_FLAG_GENERATE_MIPS) && texture->mipCount > 1;
    const bool srgb = (texture->flags & ANBC_TEXTURE_FLAG_SRGB) != 0;

    id<MTLComputePipelineState> mipsPso = generateMips ? mipsPipeline(m, srgb) : nil;
    if (generateMips && !mipsPso)
        return ANBC_ERROR_BACKEND;

    // ---- fill the params buffer ------------------------------------------
    uint8_t* params = (uint8_t*)m->params.contents;
    if (generateMips) {
        MipParams mp = {};
        mp.numMips = texture->mipCount;
        const uint32_t gx = (texture->width + kMipsTile - 1) / kMipsTile;
        const uint32_t gy = (texture->height + kMipsTile - 1) / kMipsTile;
        mp.numGroups = gx * gy;
        for (uint32_t i = 0; i < texture->mipCount && i < kMipsMaxLevels; i++) {
            mp.dims[i][0] = texture->mips[i].width;
            mp.dims[i][1] = texture->mips[i].height;
        }
        memcpy(params + kParamsMipOffset, &mp, sizeof(mp));
    }
    for (uint32_t i = 0; i < texture->mipCount; i++) {
        EncodeParams ep = {};
        ep.mipLevel = i;
        ep.blocksX = texture->mips[i].blocksX;
        ep.blocksY = texture->mips[i].blocksY;
        ep.refineIters = options->refineIterations;
        ep.mipWidth = texture->mips[i].width;
        ep.mipHeight = texture->mips[i].height;
        memcpy(params + kParamsEncodeOffset + i * kParamsEncodeStride, &ep, sizeof(ep));
    }

    // ---- encode ----------------------------------------------------------
    [m->allocator reset];
    [m->commandBuffer beginCommandBufferWithAllocator:m->allocator];
    id<MTL4ComputeCommandEncoder> enc = [m->commandBuffer computeCommandEncoder];
    id<MTL4ArgumentTable> table = m->argumentTable;
    [enc setArgumentTable:table];

    [table setTexture:t->texture.gpuResourceID atIndex:kTexSource];

    if (generateMips) {
        [table setAddress:m->params.gpuAddress + kParamsMipOffset atIndex:kBufParams];
        [table setAddress:m->counter.gpuAddress atIndex:kBufMipsCounter];
        for (uint32_t level = 1; level < kMipsMaxLevels; level++) {
            // Levels past the chain get a harmless stand-in (level 1); the
            // kernel never writes past numMips.
            id<MTLTexture> view = level < texture->mipCount ? t->mipViews[level - 1] : t->mipViews[0];
            [table setTexture:view.gpuResourceID atIndex:kTexMipsBase + level - 1];
        }
        [enc setComputePipelineState:mipsPso];
        const MipParams* mp = (const MipParams*)(params + kParamsMipOffset);
        const uint32_t gx = (texture->width + kMipsTile - 1) / kMipsTile;
        const uint32_t gy = mp->numGroups / gx;
        [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [enc barrierAfterEncoderStages:MTLStageDispatch
                   beforeEncoderStages:MTLStageDispatch
                     visibilityOptions:MTL4VisibilityOptionDevice];
    }

    const bool analytic = format == ANBC_TEXTURE_FORMAT_BC5 || format == ANBC_TEXTURE_FORMAT_BC6H;
    const bool useTensor = !analytic && m->tensorOps && !(options->flags & ANBC_COMPRESS_FLAG_NO_TENSOR_OPS) &&
                           m->tensor6.arena && m->tensor5.arena && tensorPipeline(m);
    if (analytic) {
        id<MTLComputePipelineState> pso = format == ANBC_TEXTURE_FORMAT_BC5 ? bc5Pipeline(m) : bc6hPipeline(m);
        if (!pso) {
            [enc endEncoding];
            [m->commandBuffer endCommandBuffer];
            return ANBC_ERROR_BACKEND;
        }
        [enc setComputePipelineState:pso];
    } else if (useTensor) {
        [enc setComputePipelineState:m->tensorPipeline];
        const TensorModel* models[2] = { &m->tensor6, &m->tensor5 };
        const NSUInteger base[2] = { kBufTensor6, kBufTensor5 };
        for (int k = 0; k < 2; k++) {
            for (uint32_t l = 0; l < kTensorLayers; l++)
                [table setResource:models[k]->weights[l].gpuResourceID atBufferIndex:base[k] + l];
            [table setAddress:models[k]->arena.gpuAddress + models[k]->biasOffset atIndex:base[k] + kTensorLayers];
        }
    } else {
        [enc setComputePipelineState:m->encodePipeline];
        [table setAddress:m->model6.gpuAddress atIndex:kBufModel6Header];
        [table setAddress:m->model6.gpuAddress + kModelWeightsOffset atIndex:kBufModel6Weights];
        [table setAddress:m->model5.gpuAddress atIndex:kBufModel5Header];
        [table setAddress:m->model5.gpuAddress + kModelWeightsOffset atIndex:kBufModel5Weights];
    }
    for (uint32_t i = 0; i < texture->mipCount; i++) {
        const anbcMipLayout& mip = texture->mips[i];
        [table setAddress:m->params.gpuAddress + kParamsEncodeOffset + i * kParamsEncodeStride atIndex:kBufParams];
        [table setAddress:t->output.gpuAddress + mip.byteOffset atIndex:kBufOutput];
        if (useTensor) {
            const uint32_t blocks = mip.blocksX * mip.blocksY;
            const uint32_t groups = (blocks + kTensorBlocksPerGroup - 1) / kTensorBlocksPerGroup;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(kTensorThreadsPerGroup, 1, 1)];
        } else {
            [enc dispatchThreads:MTLSizeMake(mip.blocksX, mip.blocksY, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        }
    }
    [enc endEncoding];
    [m->commandBuffer endCommandBuffer];

    id<MTL4CommandBuffer> cmds[] = { m->commandBuffer };
    [m->queue commit:cmds count:1];
    const uint64_t value = ++m->eventValue;
    [m->queue signalEvent:m->event value:value];
    if (![m->event waitUntilSignaledValue:value timeoutMS:60000]) {
        logError("timed out waiting for the GPU", nil);
        return ANBC_ERROR_BACKEND;
    }
    return ANBC_OK;
}

} // namespace

anbcResult anbcBackendMetalInit(anbcDevice* device)
{
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev || ![dev supportsFamily:MTLGPUFamilyMetal4]) {
            fprintf(stderr, "anbc[metal]: no Metal 4 capable device\n");
            return ANBC_ERROR_UNSUPPORTED;
        }

        MetalDevice* m = new MetalDevice();
        m->device = dev;
        m->queue = [dev newMTL4CommandQueue];
        m->allocator = [dev newCommandAllocator];
        m->commandBuffer = [dev newCommandBuffer];
        m->event = [dev newSharedEvent];

        MTL4CompilerDescriptor* cdesc = [[MTL4CompilerDescriptor alloc] init];
        cdesc.label = @"anbc compiler";
        NSError* error = nil;
        m->compiler = [dev newCompilerWithDescriptor:cdesc error:&error];

        MTL4ArgumentTableDescriptor* adesc = [[MTL4ArgumentTableDescriptor alloc] init];
        adesc.maxBufferBindCount = kBufCount;
        adesc.maxTextureBindCount = kTexCount;
        adesc.initializeBindings = YES;
        adesc.label = @"anbc arguments";
        m->argumentTable = [dev newArgumentTableWithDescriptor:adesc error:&error];

        MTLResidencySetDescriptor* rdesc = [[MTLResidencySetDescriptor alloc] init];
        rdesc.label = @"anbc residency";
        rdesc.initialCapacity = 16;
        m->residencySet = [dev newResidencySetWithDescriptor:rdesc error:&error];

        if (!m->queue || !m->allocator || !m->commandBuffer || !m->event || !m->compiler ||
            !m->argumentTable || !m->residencySet) {
            logError("failed to create Metal 4 core objects", error);
            delete m;
            return ANBC_ERROR_BACKEND;
        }
        [m->queue addResidencySet:m->residencySet];

        m->bc7Library = compileLibrary(m, ANBC_BC7_SCALAR_METAL_SOURCE, "anbc_bc7_scalar");
        m->bc6hLibrary = compileLibrary(m, ANBC_BC6H_METAL_SOURCE, "anbc_bc6h");
        m->bc5Library = compileLibrary(m, ANBC_BC5_METAL_SOURCE, "anbc_bc5");
        m->mipsLibrary = compileLibrary(m, ANBC_MIPS_METAL_SOURCE, "anbc_mips");
        if (!m->bc7Library || !m->bc6hLibrary || !m->bc5Library || !m->mipsLibrary) {
            delete m;
            return ANBC_ERROR_BACKEND;
        }
        m->encodePipeline = buildPipeline(m, m->bc7Library, "anbc_bc7_encode", nil);
        if (!m->encodePipeline) {
            delete m;
            return ANBC_ERROR_BACKEND;
        }

        // Shader tensor ops are only hardware accelerated from the M5 / A19
        // generation (Apple GPU family 10). Older GPUs keep the scalar kernel.
#if ANBC_HAVE_TENSOR_METALLIB
        if ([dev supportsFamily:MTLGPUFamilyApple10]) {
            dispatch_data_t data = dispatch_data_create(ANBC_BC7_TENSOR_METALLIB, ANBC_BC7_TENSOR_METALLIB_size,
                                                        nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            m->tensorLibrary = [dev newLibraryWithData:data error:&error];
            m->tensorOps = m->tensorLibrary != nil;
            if (!m->tensorOps)
                logError("tensor kernel library failed to load, falling back to scalar", error);
        }
#endif
        m->name = dev.name.UTF8String;
        device->info.name = m->name.c_str();
        device->info.metal4 = 1;
        device->info.tensorOps = m->tensorOps ? 1 : 0;

        m->params = [dev newBufferWithLength:kParamsBytes options:MTLResourceStorageModeShared];
        m->counter = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        memset(m->counter.contents, 0, 16);
        m->params.label = @"anbc params";
        m->counter.label = @"anbc mips counter";
        makeResident(m, m->params);
        makeResident(m, m->counter);

        device->backendData = m;
        device->backend.destroy = metalDestroy;
        device->backend.uploadModel = metalUploadModel;
        device->backend.createTexture = metalCreateTexture;
        device->backend.destroyTexture = metalDestroyTexture;
        device->backend.compress = metalCompress;
        return ANBC_OK;
    }
}
