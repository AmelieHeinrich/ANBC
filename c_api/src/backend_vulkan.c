/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * Vulkan backend: Windows / Linux (and MoltenVK on macOS, for testing --
 * Apple platforms ship the Metal backend). Plain Vulkan 1.1 compute, no
 * extensions beyond the portability pair MoltenVK needs, no SDK at compile
 * time (anbc_vulkan.h is a hand-checked subset of vulkan_core.h plus a
 * dlopen loader).
 *
 * Same kernels as Metal, as GLSL -> SPIR-V from the embedded blob: BC7 runs
 * the two MLPs on a scalar kernel (no cooperative-matrix / linear-algebra
 * extensions), BC5 / BC6H / ASTC 4x4 have no network, the mip chain comes
 * from the single-dispatch generator.
 *
 * Memory is the regular staging pattern: images, model weights and the
 * output blocks are DEVICE_LOCAL; uploads go through a HOST_VISIBLE staging
 * buffer and the compressed blocks are copied into a persistently mapped
 * HOST_VISIBLE readback buffer that `texture->compressed` points at.
 *
 * Per anbcCompress: one command buffer, one submit, one fence wait:
 *   [anbc_mips] -> barrier -> encode kernel x mipCount -> barrier
 *   -> copy output -> readback -> fence
 */

#include "anbc_blob.h"
#include "anbc_internal.h"
#include "anbc_vulkan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Must match the structs in the GLSL sources. */
#define ANBC_VK_MIPS_MAX_LEVELS 13
#define ANBC_VK_MIPS_TILE 64
#define ANBC_VK_MAX_TEXTURE_DIM 4096 /* with mips: mip 6 must fit one 64x64 tile */
#define ANBC_VK_MLP_MAX_HIDDEN 128
#define ANBC_VK_MLP_MAX_OUT 40
#define ANBC_VK_MLP_IN 64
#define ANBC_VK_MODEL_WEIGHTS_OFFSET 256 /* header at 0, weights at 256, like Metal */
#define ANBC_VK_FENCE_TIMEOUT_NS (60ull * 1000000000ull)

typedef struct VkModelHeader {
    uint32_t numLayers;
    uint32_t dims[ANBC_MAX_MODEL_LAYERS + 1];
    uint32_t layerOffset[ANBC_MAX_MODEL_LAYERS];
} VkModelHeader;

typedef struct VkMipParams {
    uint32_t numMips;
    uint32_t numGroups;
    uint32_t pad0, pad1;
    uint32_t dims[ANBC_VK_MIPS_MAX_LEVELS][4];
} VkMipParams;

typedef struct VkEncodeParams {
    uint32_t mipLevel;
    uint32_t blocksX;
    uint32_t blocksY;
    uint32_t refineIters;
    uint32_t mipWidth;
    uint32_t mipHeight;
    uint32_t blockBase;
    uint32_t pad1;
} VkEncodeParams;

/* Descriptor bindings (set 0), see anbc_common.glsl / anbc_mips.comp. */
enum {
    kBindSource = 0,
    kBindModel6Header = 1,
    kBindModel6Weights = 2,
    kBindModel5Header = 3,
    kBindModel5Weights = 4,
    kBindOutput = 5,
    kBindMipsViews = 1,
    kBindMipsParams = 2,
    kBindMipsCounter = 3
};

enum { kAstcKinds = ANBC_ASTC_KINDS };

typedef struct VkBufferMem {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    VkDeviceSize   size;
} VkBufferMem;

typedef struct VulkanDevice {
    anbcVk           vk;
    VkInstance       instance;
    VkPhysicalDevice physical;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queueFamily;
    VkPhysicalDeviceMemoryProperties memProps;
    char             name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];

    VkCommandPool    commandPool;
    VkCommandBuffer  commandBuffer;
    VkFence          fence;
    VkDescriptorPool descriptorPool;
    VkSampler        sampler;

    /* Layouts: BC7 binds the two models, the analytic kernels only the
     * source and the output, the mip generator its own set. */
    VkDescriptorSetLayout bc7SetLayout, plainSetLayout, mipsSetLayout;
    VkPipelineLayout      bc7PipelineLayout, plainPipelineLayout, mipsPipelineLayout;
    VkPipeline            bc7Pipeline, bc5Pipeline, bc6hPipeline;
    VkPipeline            astcPipeline[kAstcKinds];
    VkPipeline            mipsPipeline[2][2]; /* [rgba16f][srgb], built on first use */

    VkBufferMem model6, model5; /* header + weights, DEVICE_LOCAL */
    VkBufferMem mipParams;      /* MipParams, HOST_VISIBLE */
    VkBufferMem counter;        /* mips atomic counter, DEVICE_LOCAL, zero between dispatches */
    void*       mipParamsMapped;
} VulkanDevice;

typedef struct VulkanTexture {
    VkImage        image;
    VkDeviceMemory imageMemory;
    VkFormat       format;
    VkImageView    fullView;                    /* all levels, for the sampler */
    VkImageView    levelViews[ANBC_MAX_MIPS];   /* levels 1..N-1, for the mip generator */
    VkBufferMem    output;                      /* DEVICE_LOCAL */
    VkBufferMem    readback;                    /* HOST_VISIBLE, persistently mapped */
    void*          readbackMapped;
} VulkanTexture;

static VulkanDevice* vd(anbcDevice* d) { return (VulkanDevice*)d->backendData; }
static VulkanTexture* vt(anbcTexture* t) { return (VulkanTexture*)t->backendData; }

static void vkLog(const char* what, VkResult r)
{
    if (r == VK_SUCCESS)
        fprintf(stderr, "anbc[vulkan]: %s\n", what);
    else
        fprintf(stderr, "anbc[vulkan]: %s: VkResult %d\n", what, (int)r);
}

#define VK_CHECK(expr, what)                                  \
    do {                                                      \
        const VkResult vkr_ = (expr);                         \
        if (vkr_ != VK_SUCCESS) {                             \
            vkLog(what, vkr_);                                \
            return ANBC_ERROR_BACKEND;                        \
        }                                                     \
    } while (0)

/* ------------------------------------------------------------------------- */
/* Memory / buffers                                                          */
/* ------------------------------------------------------------------------- */

static uint32_t findMemoryType(VulkanDevice* v, uint32_t typeBits, VkMemoryPropertyFlags wanted)
{
    for (uint32_t i = 0; i < v->memProps.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) && (v->memProps.memoryTypes[i].propertyFlags & wanted) == wanted)
            return i;
    return UINT32_MAX;
}

static anbcResult createBuffer(VulkanDevice* v, VkDeviceSize size, VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags props, VkBufferMem* out)
{
    memset(out, 0, sizeof(*out));
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(v->vk.CreateBuffer(v->device, &bi, NULL, &out->buffer), "vkCreateBuffer");

    VkMemoryRequirements req;
    v->vk.GetBufferMemoryRequirements(v->device, out->buffer, &req);
    uint32_t type = findMemoryType(v, req.memoryTypeBits, props);
    if (type == UINT32_MAX && (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        type = findMemoryType(v, req.memoryTypeBits, 0); /* no device-local heap: anything the buffer accepts */
    if (type == UINT32_MAX) {
        vkLog("no suitable memory type for buffer", VK_SUCCESS);
        return ANBC_ERROR_BACKEND;
    }
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    VK_CHECK(v->vk.AllocateMemory(v->device, &ai, NULL, &out->memory), "vkAllocateMemory");
    VK_CHECK(v->vk.BindBufferMemory(v->device, out->buffer, out->memory, 0), "vkBindBufferMemory");
    out->size = size;
    return ANBC_OK;
}

static void destroyBuffer(VulkanDevice* v, VkBufferMem* b)
{
    if (b->buffer)
        v->vk.DestroyBuffer(v->device, b->buffer, NULL);
    if (b->memory)
        v->vk.FreeMemory(v->device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

/* One-shot command buffer: record with `begin`, run `submitAndWait`. */
static anbcResult beginCommands(VulkanDevice* v)
{
    VK_CHECK(v->vk.ResetCommandBuffer(v->commandBuffer, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(v->vk.BeginCommandBuffer(v->commandBuffer, &bi), "vkBeginCommandBuffer");
    return ANBC_OK;
}

static anbcResult submitAndWait(VulkanDevice* v)
{
    VK_CHECK(v->vk.EndCommandBuffer(v->commandBuffer), "vkEndCommandBuffer");
    VK_CHECK(v->vk.ResetFences(v->device, 1, &v->fence), "vkResetFences");
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->commandBuffer;
    VK_CHECK(v->vk.QueueSubmit(v->queue, 1, &si, v->fence), "vkQueueSubmit");
    const VkResult r = v->vk.WaitForFences(v->device, 1, &v->fence, 1, ANBC_VK_FENCE_TIMEOUT_NS);
    if (r == VK_TIMEOUT) {
        vkLog("timed out waiting for the GPU", VK_SUCCESS);
        return ANBC_ERROR_BACKEND;
    }
    VK_CHECK(r, "vkWaitForFences");
    return ANBC_OK;
}

/* Uploads `size` bytes into a DEVICE_LOCAL buffer through a staging buffer. */
static anbcResult uploadBuffer(VulkanDevice* v, const void* data, VkDeviceSize size, VkBufferMem* dst)
{
    VkBufferMem staging;
    anbcResult  r = createBuffer(v, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging);
    if (r != ANBC_OK)
        return r;
    void* mapped;
    if (v->vk.MapMemory(v->device, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS) {
        destroyBuffer(v, &staging);
        return ANBC_ERROR_BACKEND;
    }
    memcpy(mapped, data, size);
    v->vk.UnmapMemory(v->device, staging.memory);

    if ((r = beginCommands(v)) == ANBC_OK) {
        VkBufferCopy region = { 0, 0, size };
        v->vk.CmdCopyBuffer(v->commandBuffer, staging.buffer, dst->buffer, 1, &region);
        r = submitAndWait(v);
    }
    destroyBuffer(v, &staging);
    return r;
}

/* ------------------------------------------------------------------------- */
/* Pipelines                                                                 */
/* ------------------------------------------------------------------------- */

static anbcResult createSetLayout(VulkanDevice* v, const VkDescriptorSetLayoutBinding* bindings, uint32_t count,
                                  VkDescriptorSetLayout* out)
{
    VkDescriptorSetLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    ci.bindingCount = count;
    ci.pBindings = bindings;
    VK_CHECK(v->vk.CreateDescriptorSetLayout(v->device, &ci, NULL, out), "vkCreateDescriptorSetLayout");
    return ANBC_OK;
}

static anbcResult createPipelineLayout(VulkanDevice* v, VkDescriptorSetLayout set, uint32_t pushBytes,
                                       VkPipelineLayout* out)
{
    VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes };
    VkPipelineLayoutCreateInfo ci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &set;
    ci.pushConstantRangeCount = pushBytes ? 1 : 0;
    ci.pPushConstantRanges = &range;
    VK_CHECK(v->vk.CreatePipelineLayout(v->device, &ci, NULL, out), "vkCreatePipelineLayout");
    return ANBC_OK;
}

/* Builds a compute pipeline from the blob entry `name` (SPIR-V). */
static anbcResult createPipeline(VulkanDevice* v, const char* name, VkPipelineLayout layout,
                                 const VkSpecializationInfo* spec, VkPipeline* out)
{
    const void* code;
    size_t      size;
    if (!anbcBlobFind(name, &code, &size)) {
        vkLog(name, VK_SUCCESS);
        return ANBC_ERROR_BACKEND;
    }
    VkShaderModuleCreateInfo mi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    mi.codeSize = size;
    mi.pCode = (const uint32_t*)code;
    VkShaderModule module;
    VK_CHECK(v->vk.CreateShaderModule(v->device, &mi, NULL, &module), name);

    VkComputePipelineCreateInfo ci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.stage.pSpecializationInfo = spec;
    ci.layout = layout;
    const VkResult r = v->vk.CreateComputePipelines(v->device, VK_NULL_HANDLE, 1, &ci, NULL, out);
    v->vk.DestroyShaderModule(v->device, module, NULL);
    VK_CHECK(r, name);
    return ANBC_OK;
}

static VkDescriptorSetLayoutBinding binding(uint32_t index, VkDescriptorType type, uint32_t count)
{
    VkDescriptorSetLayoutBinding b = { index, type, count, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    return b;
}

static anbcResult createLayouts(VulkanDevice* v)
{
    const VkDescriptorSetLayoutBinding bc7[] = {
        binding(kBindSource, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
        binding(kBindModel6Header, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
        binding(kBindModel6Weights, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
        binding(kBindModel5Header, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
        binding(kBindModel5Weights, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
        binding(kBindOutput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
    };
    const VkDescriptorSetLayoutBinding plain[] = {
        binding(kBindSource, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
        binding(kBindOutput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
    };
    const VkDescriptorSetLayoutBinding mips[] = {
        binding(kBindSource, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
        binding(kBindMipsViews, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ANBC_VK_MIPS_MAX_LEVELS - 1),
        binding(kBindMipsParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
        binding(kBindMipsCounter, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1),
    };
    anbcResult r;
    if ((r = createSetLayout(v, bc7, 6, &v->bc7SetLayout)) != ANBC_OK) return r;
    if ((r = createSetLayout(v, plain, 2, &v->plainSetLayout)) != ANBC_OK) return r;
    if ((r = createSetLayout(v, mips, 4, &v->mipsSetLayout)) != ANBC_OK) return r;
    if ((r = createPipelineLayout(v, v->bc7SetLayout, sizeof(VkEncodeParams), &v->bc7PipelineLayout)) != ANBC_OK) return r;
    if ((r = createPipelineLayout(v, v->plainSetLayout, sizeof(VkEncodeParams), &v->plainPipelineLayout)) != ANBC_OK) return r;
    if ((r = createPipelineLayout(v, v->mipsSetLayout, 0, &v->mipsPipelineLayout)) != ANBC_OK) return r;
    return ANBC_OK;
}

static const char* const kAstcKernels[kAstcKinds] = { "astc_game.spv", "astc_normal.spv", "astc_float.spv" };

/* BC5 / BC6H / ASTC pipelines are built on first use, like the Metal backend. */
static VkPipeline plainPipeline(VulkanDevice* v, VkPipeline* slot, const char* name)
{
    if (!*slot && createPipeline(v, name, v->plainPipelineLayout, NULL, slot) != ANBC_OK)
        *slot = VK_NULL_HANDLE;
    return *slot;
}

static VkPipeline mipsPipeline(VulkanDevice* v, bool half, bool srgb)
{
    VkPipeline* slot = &v->mipsPipeline[half][srgb];
    if (!*slot) {
        const VkBool32 value = srgb ? 1 : 0;
        const VkSpecializationMapEntry entry = { 1, 0, sizeof(value) };
        VkSpecializationInfo spec = { 1, &entry, sizeof(value), &value };
        if (createPipeline(v, half ? "mips_rgba16f.spv" : "mips_rgba8.spv", v->mipsPipelineLayout, &spec, slot) != ANBC_OK)
            *slot = VK_NULL_HANDLE;
    }
    return *slot;
}

/* ------------------------------------------------------------------------- */
/* Backend entry points                                                      */
/* ------------------------------------------------------------------------- */

static void vulkanDestroy(anbcDevice* device)
{
    VulkanDevice* v = vd(device);
    if (!v)
        return;
    if (v->device) {
        v->vk.DeviceWaitIdle(v->device);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                if (v->mipsPipeline[i][j]) v->vk.DestroyPipeline(v->device, v->mipsPipeline[i][j], NULL);
        for (int k = 0; k < kAstcKinds; k++)
            if (v->astcPipeline[k]) v->vk.DestroyPipeline(v->device, v->astcPipeline[k], NULL);
        if (v->bc7Pipeline) v->vk.DestroyPipeline(v->device, v->bc7Pipeline, NULL);
        if (v->bc5Pipeline) v->vk.DestroyPipeline(v->device, v->bc5Pipeline, NULL);
        if (v->bc6hPipeline) v->vk.DestroyPipeline(v->device, v->bc6hPipeline, NULL);
        if (v->bc7PipelineLayout) v->vk.DestroyPipelineLayout(v->device, v->bc7PipelineLayout, NULL);
        if (v->plainPipelineLayout) v->vk.DestroyPipelineLayout(v->device, v->plainPipelineLayout, NULL);
        if (v->mipsPipelineLayout) v->vk.DestroyPipelineLayout(v->device, v->mipsPipelineLayout, NULL);
        if (v->bc7SetLayout) v->vk.DestroyDescriptorSetLayout(v->device, v->bc7SetLayout, NULL);
        if (v->plainSetLayout) v->vk.DestroyDescriptorSetLayout(v->device, v->plainSetLayout, NULL);
        if (v->mipsSetLayout) v->vk.DestroyDescriptorSetLayout(v->device, v->mipsSetLayout, NULL);
        destroyBuffer(v, &v->model6);
        destroyBuffer(v, &v->model5);
        if (v->mipParamsMapped) v->vk.UnmapMemory(v->device, v->mipParams.memory);
        destroyBuffer(v, &v->mipParams);
        destroyBuffer(v, &v->counter);
        if (v->sampler) v->vk.DestroySampler(v->device, v->sampler, NULL);
        if (v->descriptorPool) v->vk.DestroyDescriptorPool(v->device, v->descriptorPool, NULL);
        if (v->fence) v->vk.DestroyFence(v->device, v->fence, NULL);
        if (v->commandPool) v->vk.DestroyCommandPool(v->device, v->commandPool, NULL);
        v->vk.DestroyDevice(v->device, NULL);
    }
    if (v->instance)
        v->vk.DestroyInstance(v->instance, NULL);
    anbcVkUnload(&v->vk);
    free(v);
    device->backendData = NULL;
}

static anbcResult vulkanUploadModel(anbcDevice* device, const anbcModel* model)
{
    VulkanDevice* v = vd(device);
    if (model->dims[0] != ANBC_VK_MLP_IN || model->dims[model->numLayers] > ANBC_VK_MLP_MAX_OUT)
        return ANBC_ERROR_BAD_MODEL;
    for (uint32_t i = 0; i <= model->numLayers; i++)
        if (model->dims[i] > (i == 0 ? ANBC_VK_MLP_IN : i == model->numLayers ? ANBC_VK_MLP_MAX_OUT : ANBC_VK_MLP_MAX_HIDDEN) ||
            model->dims[i] % 4 != 0) /* the kernel streams weights/activations as vec4 */
            return ANBC_ERROR_BAD_MODEL;

    const size_t bytes = ANBC_VK_MODEL_WEIGHTS_OFFSET + model->dataFloats * sizeof(float);
    uint8_t*     staged = (uint8_t*)calloc(1, bytes);
    if (!staged)
        return ANBC_ERROR_BACKEND;
    VkModelHeader header = { 0 };
    header.numLayers = model->numLayers;
    for (uint32_t i = 0; i <= model->numLayers; i++)
        header.dims[i] = model->dims[i];
    for (uint32_t i = 0; i < model->numLayers; i++)
        header.layerOffset[i] = (uint32_t)model->layerOffset[i];
    memcpy(staged, &header, sizeof(header));
    memcpy(staged + ANBC_VK_MODEL_WEIGHTS_OFFSET, model->data, model->dataFloats * sizeof(float));

    VkBufferMem* slot = model->mode == 6 ? &v->model6 : &v->model5;
    destroyBuffer(v, slot);
    anbcResult r = createBuffer(v, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, slot);
    if (r == ANBC_OK)
        r = uploadBuffer(v, staged, bytes, slot);
    free(staged);
    if (r != ANBC_OK)
        destroyBuffer(v, slot);
    return r;
}

static void vulkanDestroyTexture(anbcDevice* device, anbcTexture* texture)
{
    VulkanDevice*  v = vd(device);
    VulkanTexture* t = vt(texture);
    if (!t)
        return;
    v->vk.DeviceWaitIdle(v->device);
    for (uint32_t i = 0; i < ANBC_MAX_MIPS; i++)
        if (t->levelViews[i]) v->vk.DestroyImageView(v->device, t->levelViews[i], NULL);
    if (t->fullView) v->vk.DestroyImageView(v->device, t->fullView, NULL);
    if (t->image) v->vk.DestroyImage(v->device, t->image, NULL);
    if (t->imageMemory) v->vk.FreeMemory(v->device, t->imageMemory, NULL);
    if (t->readbackMapped) v->vk.UnmapMemory(v->device, t->readback.memory);
    destroyBuffer(v, &t->output);
    destroyBuffer(v, &t->readback);
    free(t);
    texture->backendData = NULL;
    texture->compressed = NULL;
}

static VkImageView createView(VulkanDevice* v, VkImage image, VkFormat format, uint32_t baseLevel, uint32_t levels)
{
    VkImageViewCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    ci.image = image;
    ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ci.format = format;
    ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ci.subresourceRange.baseMipLevel = baseLevel;
    ci.subresourceRange.levelCount = levels;
    ci.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    if (v->vk.CreateImageView(v->device, &ci, NULL, &view) != VK_SUCCESS)
        vkLog("vkCreateImageView", VK_SUCCESS);
    return view;
}

static VkImageMemoryBarrier imageBarrier(VkImage image, uint32_t levels, VkImageLayout from, VkImageLayout to,
                                         VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = levels;
    b.subresourceRange.layerCount = 1;
    return b;
}

static anbcResult vulkanCreateTexture(anbcDevice* device, anbcTexture* texture, const anbcTextureDesc* desc)
{
    VulkanDevice* v = vd(device);
    /* The size limit comes from the single-dispatch mip generator; a texture
     * encoded without mips only has to fit the GPU (checked at image creation). */
    if (texture->mipCount > 1 && (desc->width > ANBC_VK_MAX_TEXTURE_DIM || desc->height > ANBC_VK_MAX_TEXTURE_DIM))
        return ANBC_ERROR_UNSUPPORTED;

    const bool     half = desc->pixelFormat == ANBC_PIXEL_FORMAT_RGBA16_FLOAT;
    const uint32_t bpp = half ? 8 : 4;
    VulkanTexture* t = (VulkanTexture*)calloc(1, sizeof(VulkanTexture));
    if (!t)
        return ANBC_ERROR_BACKEND;
    texture->backendData = t;
    t->format = half ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;

    /* ---- image + views -------------------------------------------------- */
    VkImageCreateInfo ii = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = t->format;
    ii.extent.width = desc->width;
    ii.extent.height = desc->height;
    ii.extent.depth = 1;
    ii.mipLevels = texture->mipCount;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (v->vk.CreateImage(v->device, &ii, NULL, &t->image) != VK_SUCCESS) {
        vkLog("vkCreateImage", VK_SUCCESS);
        goto fail;
    }
    {
        VkMemoryRequirements req;
        v->vk.GetImageMemoryRequirements(v->device, t->image, &req);
        uint32_t type = findMemoryType(v, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX)
            type = findMemoryType(v, req.memoryTypeBits, 0);
        VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        if (type == UINT32_MAX || v->vk.AllocateMemory(v->device, &ai, NULL, &t->imageMemory) != VK_SUCCESS ||
            v->vk.BindImageMemory(v->device, t->image, t->imageMemory, 0) != VK_SUCCESS) {
            vkLog("image memory", VK_SUCCESS);
            goto fail;
        }
    }
    t->fullView = createView(v, t->image, t->format, 0, texture->mipCount);
    if (!t->fullView)
        goto fail;
    for (uint32_t level = 1; level < texture->mipCount; level++)
        if (!(t->levelViews[level - 1] = createView(v, t->image, t->format, level, 1)))
            goto fail;

    /* ---- output + readback ----------------------------------------------- */
    if (createBuffer(v, texture->compressedBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &t->output) != ANBC_OK)
        goto fail;
    if (createBuffer(v, texture->compressedBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &t->readback) != ANBC_OK)
        goto fail;
    if (v->vk.MapMemory(v->device, t->readback.memory, 0, VK_WHOLE_SIZE, 0, &t->readbackMapped) != VK_SUCCESS) {
        vkLog("vkMapMemory (readback)", VK_SUCCESS);
        goto fail;
    }
    texture->compressed = t->readbackMapped;

    /* ---- upload level 0 through a staging buffer, then GENERAL for life --- */
    {
        const VkDeviceSize tight = (VkDeviceSize)desc->width * desc->height * bpp;
        VkBufferMem        staging;
        if (createBuffer(v, tight, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging) != ANBC_OK)
            goto fail;
        void* mapped;
        if (v->vk.MapMemory(v->device, staging.memory, 0, tight, 0, &mapped) != VK_SUCCESS) {
            destroyBuffer(v, &staging);
            goto fail;
        }
        for (uint32_t y = 0; y < desc->height; y++)
            memcpy((uint8_t*)mapped + (size_t)y * desc->width * bpp, (const uint8_t*)desc->pixels + (size_t)y * desc->rowPitch,
                   (size_t)desc->width * bpp);
        v->vk.UnmapMemory(v->device, staging.memory);

        anbcResult r = beginCommands(v);
        if (r == ANBC_OK) {
            VkImageMemoryBarrier toDst = imageBarrier(t->image, texture->mipCount, VK_IMAGE_LAYOUT_UNDEFINED,
                                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            v->vk.CmdPipelineBarrier(v->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, NULL, 0, NULL, 1, &toDst);
            VkBufferImageCopy region = { 0 };
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent.width = desc->width;
            region.imageExtent.height = desc->height;
            region.imageExtent.depth = 1;
            v->vk.CmdCopyBufferToImage(v->commandBuffer, staging.buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            /* GENERAL is the one layout that serves both the sampler reads and
             * the mip generator's storage writes; the texture stays in it. */
            VkImageMemoryBarrier toGeneral = imageBarrier(t->image, texture->mipCount, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                          VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            v->vk.CmdPipelineBarrier(v->commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     0, NULL, 0, NULL, 1, &toGeneral);
            r = submitAndWait(v);
        }
        destroyBuffer(v, &staging);
        if (r != ANBC_OK)
            goto fail;
    }
    return ANBC_OK;

fail:
    vulkanDestroyTexture(device, texture);
    return ANBC_ERROR_BACKEND;
}

static VkWriteDescriptorSet writeBuffer(VkDescriptorSet set, uint32_t bind, VkDescriptorType type,
                                        const VkDescriptorBufferInfo* info)
{
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = set;
    w.dstBinding = bind;
    w.descriptorCount = 1;
    w.descriptorType = type;
    w.pBufferInfo = info;
    return w;
}

static VkWriteDescriptorSet writeImages(VkDescriptorSet set, uint32_t bind, VkDescriptorType type, uint32_t count,
                                        const VkDescriptorImageInfo* info)
{
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = set;
    w.dstBinding = bind;
    w.descriptorCount = count;
    w.descriptorType = type;
    w.pImageInfo = info;
    return w;
}

static anbcResult vulkanCompress(anbcDevice* device, anbcTexture* texture, anbcTextureFormat format,
                                 const anbcCompressOptions* options)
{
    VulkanDevice*  v = vd(device);
    VulkanTexture* t = vt(texture);
    const bool     generateMips = (texture->flags & ANBC_TEXTURE_FLAG_GENERATE_MIPS) && texture->mipCount > 1;
    const bool     srgb = (texture->flags & ANBC_TEXTURE_FLAG_SRGB) != 0;
    const bool     half = t->format == VK_FORMAT_R16G16B16A16_SFLOAT;

    /* ---- pipelines --------------------------------------------------------- */
    VkPipeline mipsPso = generateMips ? mipsPipeline(v, half, srgb) : VK_NULL_HANDLE;
    if (generateMips && !mipsPso)
        return ANBC_ERROR_BACKEND;
    VkPipeline       pso;
    VkPipelineLayout layout;
    VkDescriptorSetLayout setLayout;
    if (format == ANBC_TEXTURE_FORMAT_BC7) {
        pso = v->bc7Pipeline;
        layout = v->bc7PipelineLayout;
        setLayout = v->bc7SetLayout;
    } else {
        pso = format == ANBC_TEXTURE_FORMAT_BC5    ? plainPipeline(v, &v->bc5Pipeline, "bc5.spv")
              : format == ANBC_TEXTURE_FORMAT_BC6H ? plainPipeline(v, &v->bc6hPipeline, "bc6h.spv")
                                                   : plainPipeline(v, &v->astcPipeline[anbcAstcKind(format, texture->flags)],
                                                                   kAstcKernels[anbcAstcKind(format, texture->flags)]);
        layout = v->plainPipelineLayout;
        setLayout = v->plainSetLayout;
    }
    if (!pso)
        return ANBC_ERROR_BACKEND;

    /* ---- descriptor sets (pool reset per compress) --------------------------- */
    VK_CHECK(v->vk.ResetDescriptorPool(v->device, v->descriptorPool, 0), "vkResetDescriptorPool");
    VkDescriptorSetLayout layouts[2] = { setLayout, v->mipsSetLayout };
    VkDescriptorSet       sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = v->descriptorPool;
    ai.descriptorSetCount = generateMips ? 2 : 1;
    ai.pSetLayouts = layouts;
    VK_CHECK(v->vk.AllocateDescriptorSets(v->device, &ai, sets), "vkAllocateDescriptorSets");

    VkDescriptorImageInfo  source = { v->sampler, t->fullView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo output = { t->output.buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo h6 = { v->model6.buffer, 0, ANBC_VK_MODEL_WEIGHTS_OFFSET };
    VkDescriptorBufferInfo w6 = { v->model6.buffer, ANBC_VK_MODEL_WEIGHTS_OFFSET, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo h5 = { v->model5.buffer, 0, ANBC_VK_MODEL_WEIGHTS_OFFSET };
    VkDescriptorBufferInfo w5 = { v->model5.buffer, ANBC_VK_MODEL_WEIGHTS_OFFSET, VK_WHOLE_SIZE };
    VkDescriptorImageInfo  views[ANBC_VK_MIPS_MAX_LEVELS - 1];
    VkDescriptorBufferInfo mipParams = { v->mipParams.buffer, 0, sizeof(VkMipParams) };
    VkDescriptorBufferInfo counter = { v->counter.buffer, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet   writes[12];
    uint32_t               n = 0;
    writes[n++] = writeImages(sets[0], kBindSource, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &source);
    writes[n++] = writeBuffer(sets[0], kBindOutput, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &output);
    if (format == ANBC_TEXTURE_FORMAT_BC7) {
        writes[n++] = writeBuffer(sets[0], kBindModel6Header, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &h6);
        writes[n++] = writeBuffer(sets[0], kBindModel6Weights, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &w6);
        writes[n++] = writeBuffer(sets[0], kBindModel5Header, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &h5);
        writes[n++] = writeBuffer(sets[0], kBindModel5Weights, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &w5);
    }
    if (generateMips) {
        /* Levels past the chain get a harmless stand-in (level 1); the kernel
         * never writes past numMips. */
        for (uint32_t level = 1; level < ANBC_VK_MIPS_MAX_LEVELS; level++) {
            views[level - 1].sampler = VK_NULL_HANDLE;
            views[level - 1].imageView = level < texture->mipCount ? t->levelViews[level - 1] : t->levelViews[0];
            views[level - 1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        writes[n++] = writeImages(sets[1], kBindSource, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &source);
        writes[n++] = writeImages(sets[1], kBindMipsViews, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ANBC_VK_MIPS_MAX_LEVELS - 1, views);
        writes[n++] = writeBuffer(sets[1], kBindMipsParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &mipParams);
        writes[n++] = writeBuffer(sets[1], kBindMipsCounter, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &counter);

        VkMipParams* mp = (VkMipParams*)v->mipParamsMapped;
        memset(mp, 0, sizeof(*mp));
        mp->numMips = texture->mipCount;
        const uint32_t gx = (texture->width + ANBC_VK_MIPS_TILE - 1) / ANBC_VK_MIPS_TILE;
        const uint32_t gy = (texture->height + ANBC_VK_MIPS_TILE - 1) / ANBC_VK_MIPS_TILE;
        mp->numGroups = gx * gy;
        for (uint32_t i = 0; i < texture->mipCount && i < ANBC_VK_MIPS_MAX_LEVELS; i++) {
            mp->dims[i][0] = texture->mips[i].width;
            mp->dims[i][1] = texture->mips[i].height;
        }
    }
    v->vk.UpdateDescriptorSets(v->device, n, writes, 0, NULL);

    /* ---- record ------------------------------------------------------------ */
    anbcResult r = beginCommands(v);
    if (r != ANBC_OK)
        return r;
    VkCommandBuffer cmd = v->commandBuffer;
    if (generateMips) {
        const uint32_t gx = (texture->width + ANBC_VK_MIPS_TILE - 1) / ANBC_VK_MIPS_TILE;
        const uint32_t gy = (texture->height + ANBC_VK_MIPS_TILE - 1) / ANBC_VK_MIPS_TILE;
        v->vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mipsPso);
        v->vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v->mipsPipelineLayout, 0, 1, &sets[1], 0, NULL);
        v->vk.CmdDispatch(cmd, gx, gy, 1);
        VkImageMemoryBarrier written = imageBarrier(t->image, texture->mipCount, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        v->vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 0, NULL, 0, NULL, 1, &written);
    }

    v->vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pso);
    v->vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[0], 0, NULL);
    for (uint32_t i = 0; i < texture->mipCount; i++) {
        const anbcMipLayout* mip = &texture->mips[i];
        VkEncodeParams       ep = { 0 };
        ep.mipLevel = i;
        ep.blocksX = mip->blocksX;
        ep.blocksY = mip->blocksY;
        ep.refineIters = options->refineIterations;
        ep.mipWidth = mip->width;
        ep.mipHeight = mip->height;
        ep.blockBase = (uint32_t)(mip->byteOffset / ANBC_BLOCK_BYTES);
        v->vk.CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ep), &ep);
        v->vk.CmdDispatch(cmd, (mip->blocksX + 7) / 8, (mip->blocksY + 7) / 8, 1);
    }

    VkBufferMemoryBarrier encoded = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
    encoded.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    encoded.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    encoded.srcQueueFamilyIndex = encoded.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    encoded.buffer = t->output.buffer;
    encoded.size = VK_WHOLE_SIZE;
    v->vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, NULL, 1, &encoded, 0, NULL);
    VkBufferCopy region = { 0, 0, texture->compressedBytes };
    v->vk.CmdCopyBuffer(cmd, t->output.buffer, t->readback.buffer, 1, &region);
    VkBufferMemoryBarrier toHost = encoded;
    toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    toHost.buffer = t->readback.buffer;
    v->vk.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                             0, NULL, 1, &toHost, 0, NULL);
    return submitAndWait(v);
}

/* ------------------------------------------------------------------------- */
/* Device bring-up                                                           */
/* ------------------------------------------------------------------------- */

static bool hasExtension(const VkExtensionProperties* list, uint32_t count, const char* name)
{
    for (uint32_t i = 0; i < count; i++)
        if (strcmp(list[i].extensionName, name) == 0)
            return true;
    return false;
}

static anbcResult createInstance(VulkanDevice* v)
{
    /* MoltenVK is a non-conformant "portability" implementation: the loader
     * only lists it when asked to. Harmless elsewhere. */
    uint32_t               count = 0;
    VkExtensionProperties* exts = NULL;
    bool                   portability = false;
    if (v->vk.EnumerateInstanceExtensionProperties(NULL, &count, NULL) == VK_SUCCESS && count) {
        exts = (VkExtensionProperties*)calloc(count, sizeof(*exts));
        if (exts && v->vk.EnumerateInstanceExtensionProperties(NULL, &count, exts) == VK_SUCCESS)
            portability = hasExtension(exts, count, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        free(exts);
    }
    const char* const  portabilityExt = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    VkApplicationInfo  app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "anbc";
    app.pEngineName = "anbc";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ci.pApplicationInfo = &app;
    if (portability) {
        ci.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = &portabilityExt;
    }
    VK_CHECK(v->vk.CreateInstance(&ci, NULL, &v->instance), "vkCreateInstance");
    anbcVkLoadInstance(&v->vk, v->instance);
    return ANBC_OK;
}

/* Picks a discrete GPU with a compute queue, else an integrated one, else
 * whatever has one. Also checks the storage-image formats the mip generator
 * needs. */
static anbcResult pickPhysicalDevice(VulkanDevice* v)
{
    uint32_t count = 0;
    VK_CHECK(v->vk.EnumeratePhysicalDevices(v->instance, &count, NULL), "vkEnumeratePhysicalDevices");
    if (count == 0) {
        vkLog("no Vulkan devices", VK_SUCCESS);
        return ANBC_ERROR_UNSUPPORTED;
    }
    VkPhysicalDevice* devices = (VkPhysicalDevice*)calloc(count, sizeof(*devices));
    if (!devices)
        return ANBC_ERROR_BACKEND;
    v->vk.EnumeratePhysicalDevices(v->instance, &count, devices);

    int bestScore = -1;
    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties props;
        v->vk.GetPhysicalDeviceProperties(devices[i], &props);
        uint32_t families = 0;
        v->vk.GetPhysicalDeviceQueueFamilyProperties(devices[i], &families, NULL);
        VkQueueFamilyProperties* qf = (VkQueueFamilyProperties*)calloc(families, sizeof(*qf));
        if (!qf)
            continue;
        v->vk.GetPhysicalDeviceQueueFamilyProperties(devices[i], &families, qf);
        uint32_t family = UINT32_MAX;
        for (uint32_t q = 0; q < families && family == UINT32_MAX; q++)
            if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT)
                family = q;
        free(qf);
        if (family == UINT32_MAX)
            continue;
        VkFormatProperties fp8, fp16;
        v->vk.GetPhysicalDeviceFormatProperties(devices[i], VK_FORMAT_R8G8B8A8_UNORM, &fp8);
        v->vk.GetPhysicalDeviceFormatProperties(devices[i], VK_FORMAT_R16G16B16A16_SFLOAT, &fp16);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        if ((fp8.optimalTilingFeatures & need) != need || (fp16.optimalTilingFeatures & need) != need)
            continue;
        const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 2
                          : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1
                                                                                       : 0;
        if (score > bestScore) {
            bestScore = score;
            v->physical = devices[i];
            v->queueFamily = family;
            memcpy(v->name, props.deviceName, sizeof(v->name));
        }
    }
    free(devices);
    if (bestScore < 0) {
        vkLog("no Vulkan device with a compute queue and RGBA8/RGBA16F storage images", VK_SUCCESS);
        return ANBC_ERROR_UNSUPPORTED;
    }
    v->vk.GetPhysicalDeviceMemoryProperties(v->physical, &v->memProps);
    return ANBC_OK;
}

static anbcResult createDevice(VulkanDevice* v)
{
    /* VK_KHR_portability_subset must be enabled when the device advertises it. */
    uint32_t               count = 0;
    VkExtensionProperties* exts = NULL;
    bool                   subset = false;
    if (v->vk.EnumerateDeviceExtensionProperties(v->physical, NULL, &count, NULL) == VK_SUCCESS && count) {
        exts = (VkExtensionProperties*)calloc(count, sizeof(*exts));
        if (exts && v->vk.EnumerateDeviceExtensionProperties(v->physical, NULL, &count, exts) == VK_SUCCESS)
            subset = hasExtension(exts, count, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        free(exts);
    }
    const char* const       subsetExt = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
    const float             priority = 1.0f;
    VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qi.queueFamilyIndex = v->queueFamily;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo ci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    if (subset) {
        ci.enabledExtensionCount = 1;
        ci.ppEnabledExtensionNames = &subsetExt;
    }
    VK_CHECK(v->vk.CreateDevice(v->physical, &ci, NULL, &v->device), "vkCreateDevice");
    anbcVkLoadDevice(&v->vk, v->device);
    v->vk.GetDeviceQueue(v->device, v->queueFamily, 0, &v->queue);

    VkCommandPoolCreateInfo pi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pi.queueFamilyIndex = v->queueFamily;
    VK_CHECK(v->vk.CreateCommandPool(v->device, &pi, NULL, &v->commandPool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    bi.commandPool = v->commandPool;
    bi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    bi.commandBufferCount = 1;
    VK_CHECK(v->vk.AllocateCommandBuffers(v->device, &bi, &v->commandBuffer), "vkAllocateCommandBuffers");
    VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(v->vk.CreateFence(v->device, &fi, NULL, &v->fence), "vkCreateFence");

    /* Worst case per compress: the encode set (1 sampler, 5 SSBOs) and the
     * mips set (1 sampler, 12 storage images, 1 UBO, 1 SSBO). */
    const VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, ANBC_VK_MIPS_MAX_LEVELS - 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
    };
    VkDescriptorPoolCreateInfo di = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    di.maxSets = 2;
    di.poolSizeCount = 4;
    di.pPoolSizes = sizes;
    VK_CHECK(v->vk.CreateDescriptorPool(v->device, &di, NULL, &v->descriptorPool), "vkCreateDescriptorPool");

    /* texelFetch never filters; the sampler is only there for sampler2D. */
    VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = (float)ANBC_MAX_MIPS;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    VK_CHECK(v->vk.CreateSampler(v->device, &si, NULL, &v->sampler), "vkCreateSampler");
    return ANBC_OK;
}

anbcResult anbcBackendVulkanInit(anbcDevice* device)
{
    VulkanDevice* v = (VulkanDevice*)calloc(1, sizeof(VulkanDevice));
    if (!v)
        return ANBC_ERROR_BACKEND;
    device->backendData = v;

    if (!anbcVkLoad(&v->vk)) {
        fprintf(stderr, "anbc[vulkan]: no Vulkan loader library found\n");
        vulkanDestroy(device);
        return ANBC_ERROR_UNSUPPORTED;
    }
    anbcResult r;
    if ((r = createInstance(v)) != ANBC_OK || (r = pickPhysicalDevice(v)) != ANBC_OK || (r = createDevice(v)) != ANBC_OK ||
        (r = createLayouts(v)) != ANBC_OK ||
        (r = createPipeline(v, "bc7.spv", v->bc7PipelineLayout, NULL, &v->bc7Pipeline)) != ANBC_OK ||
        (r = createBuffer(v, sizeof(VkMipParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &v->mipParams)) != ANBC_OK ||
        (r = createBuffer(v, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &v->counter)) != ANBC_OK) {
        vulkanDestroy(device);
        return r;
    }
    if (v->vk.MapMemory(v->device, v->mipParams.memory, 0, VK_WHOLE_SIZE, 0, &v->mipParamsMapped) != VK_SUCCESS) {
        vulkanDestroy(device);
        return ANBC_ERROR_BACKEND;
    }
    /* The mip generator's counter starts at zero and resets itself after
     * every dispatch. */
    if ((r = beginCommands(v)) == ANBC_OK) {
        v->vk.CmdFillBuffer(v->commandBuffer, v->counter.buffer, 0, VK_WHOLE_SIZE, 0);
        r = submitAndWait(v);
    }
    if (r != ANBC_OK) {
        vulkanDestroy(device);
        return r;
    }

    device->info.name = v->name;
    device->info.backend = "Vulkan";
    device->info.tensorOps = 0;
    device->backend.destroy = vulkanDestroy;
    device->backend.uploadModel = vulkanUploadModel;
    device->backend.createTexture = vulkanCreateTexture;
    device->backend.destroyTexture = vulkanDestroyTexture;
    device->backend.compress = vulkanCompress;
    return ANBC_OK;
}
