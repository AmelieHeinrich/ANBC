/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * The subset of the Vulkan API the anbc Vulkan backend uses, extracted from
 * the Khronos vulkan_core.h (header version 341, Apache-2.0 /
 * MIT, Copyright 2015-2025 The Khronos Group Inc.) by tools/gen_vulkan_header.py
 * on 2026-09-20 -- regenerate rather than edit. Types and enum
 * values are verbatim (ABI); enums carry only the members we name. No SDK
 * is needed at compile time: `anbcVkLoad` dlopens the loader library and
 * resolves every entry point at runtime.
 */

#ifndef ANBC_VULKAN_H
#define ANBC_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define VKAPI_PTR __stdcall
#else
#define VKAPI_PTR
#endif

#define VK_DEFINE_HANDLE(object) typedef struct object##_T* object;
#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || \
    defined(__ia64) || defined(_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T* object;
#define VK_NULL_HANDLE NULL
#else
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef uint64_t object;
#define VK_NULL_HANDLE 0ULL
#endif

#define VK_MAKE_API_VERSION(variant, major, minor, patch) \
    ((((uint32_t)(variant)) << 29U) | (((uint32_t)(major)) << 22U) | (((uint32_t)(minor)) << 12U) | ((uint32_t)(patch)))
#define VK_API_VERSION_1_1 VK_MAKE_API_VERSION(0, 1, 1, 0)
#define VK_WHOLE_SIZE (~0ULL)
#define VK_QUEUE_FAMILY_IGNORED (~0U)
#define VK_MAX_PHYSICAL_DEVICE_NAME_SIZE 256U
#define VK_UUID_SIZE 16U
#define VK_MAX_MEMORY_TYPES 32U
#define VK_MAX_MEMORY_HEAPS 16U
#define VK_MAX_EXTENSION_NAME_SIZE 256U
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"

typedef uint32_t VkFlags;
typedef uint64_t VkFlags64;
typedef uint32_t VkBool32;
typedef uint64_t VkDeviceSize;
typedef uint32_t VkSampleMask;
typedef uint64_t VkDeviceAddress;
typedef struct VkAllocationCallbacks VkAllocationCallbacks; /* always NULL here */
typedef void (VKAPI_PTR* PFN_vkVoidFunction)(void);

typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_INCOMPLETE = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_LAYER_NOT_PRESENT = -6,
    VK_ERROR_EXTENSION_NOT_PRESENT = -7,
    VK_ERROR_FEATURE_NOT_PRESENT = -8,
    VK_ERROR_INCOMPATIBLE_DRIVER = -9,
    VK_RESULT_MAX_ENUM = 0x7FFFFFFF
} VkResult;
typedef enum VkStructureType {
    VK_STRUCTURE_TYPE_APPLICATION_INFO = 0,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1,
    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3,
    VK_STRUCTURE_TYPE_SUBMIT_INFO = 4,
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5,
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 8,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO = 12,
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO = 14,
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15,
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = 16,
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = 18,
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO = 29,
    VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = 30,
    VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO = 31,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = 32,
    VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = 33,
    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = 34,
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = 35,
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = 39,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = 40,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 42,
    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER = 44,
    VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER = 45,
    VK_STRUCTURE_TYPE_MEMORY_BARRIER = 46,
    VK_STRUCTURE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkStructureType;
typedef enum VkImageLayout {
    VK_IMAGE_LAYOUT_UNDEFINED = 0,
    VK_IMAGE_LAYOUT_GENERAL = 1,
    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL = 7,
    VK_IMAGE_LAYOUT_MAX_ENUM = 0x7FFFFFFF
} VkImageLayout;
typedef enum VkFormat {
    VK_FORMAT_UNDEFINED = 0,
    VK_FORMAT_R8G8B8A8_UNORM = 37,
    VK_FORMAT_R16G16B16A16_SFLOAT = 97,
    VK_FORMAT_MAX_ENUM = 0x7FFFFFFF
} VkFormat;
typedef enum VkImageTiling {
    VK_IMAGE_TILING_OPTIMAL = 0,
    VK_IMAGE_TILING_MAX_ENUM = 0x7FFFFFFF
} VkImageTiling;
typedef enum VkImageType {
    VK_IMAGE_TYPE_2D = 1,
    VK_IMAGE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkImageType;
typedef enum VkPhysicalDeviceType {
    VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2,
    VK_PHYSICAL_DEVICE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkPhysicalDeviceType;
typedef enum VkSharingMode {
    VK_SHARING_MODE_EXCLUSIVE = 0,
    VK_SHARING_MODE_MAX_ENUM = 0x7FFFFFFF
} VkSharingMode;
typedef enum VkComponentSwizzle {
    VK_COMPONENT_SWIZZLE_IDENTITY = 0,
    VK_COMPONENT_SWIZZLE_MAX_ENUM = 0x7FFFFFFF
} VkComponentSwizzle;
typedef enum VkImageViewType {
    VK_IMAGE_VIEW_TYPE_2D = 1,
    VK_IMAGE_VIEW_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkImageViewType;
typedef enum VkCommandBufferLevel {
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0,
    VK_COMMAND_BUFFER_LEVEL_MAX_ENUM = 0x7FFFFFFF
} VkCommandBufferLevel;
typedef enum VkBorderColor {
    VK_BORDER_COLOR_INT_OPAQUE_BLACK = 3,
    VK_BORDER_COLOR_MAX_ENUM = 0x7FFFFFFF
} VkBorderColor;
typedef enum VkFilter {
    VK_FILTER_NEAREST = 0,
    VK_FILTER_MAX_ENUM = 0x7FFFFFFF
} VkFilter;
typedef enum VkSamplerAddressMode {
    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2,
    VK_SAMPLER_ADDRESS_MODE_MAX_ENUM = 0x7FFFFFFF
} VkSamplerAddressMode;
typedef enum VkSamplerMipmapMode {
    VK_SAMPLER_MIPMAP_MODE_NEAREST = 0,
    VK_SAMPLER_MIPMAP_MODE_MAX_ENUM = 0x7FFFFFFF
} VkSamplerMipmapMode;
typedef enum VkCompareOp {
    VK_COMPARE_OP_ALWAYS = 7,
    VK_COMPARE_OP_MAX_ENUM = 0x7FFFFFFF
} VkCompareOp;
typedef enum VkDescriptorType {
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1,
    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3,
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7,
    VK_DESCRIPTOR_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkDescriptorType;
typedef enum VkPipelineBindPoint {
    VK_PIPELINE_BIND_POINT_COMPUTE = 1,
    VK_PIPELINE_BIND_POINT_MAX_ENUM = 0x7FFFFFFF
} VkPipelineBindPoint;
typedef enum VkSampleCountFlagBits {
    VK_SAMPLE_COUNT_1_BIT = 0x00000001,
    VK_SAMPLE_COUNT_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkSampleCountFlagBits;
typedef enum VkShaderStageFlagBits {
    VK_SHADER_STAGE_COMPUTE_BIT = 0x00000020,
    VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkShaderStageFlagBits;
typedef enum VkAccessFlagBits {
    VK_ACCESS_SHADER_READ_BIT = 0x00000020,
    VK_ACCESS_SHADER_WRITE_BIT = 0x00000040,
    VK_ACCESS_TRANSFER_READ_BIT = 0x00000800,
    VK_ACCESS_TRANSFER_WRITE_BIT = 0x00001000,
    VK_ACCESS_HOST_READ_BIT = 0x00002000,
    VK_ACCESS_HOST_WRITE_BIT = 0x00004000,
    VK_ACCESS_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkAccessFlagBits;
typedef enum VkImageAspectFlagBits {
    VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001,
    VK_IMAGE_ASPECT_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkImageAspectFlagBits;
typedef enum VkFormatFeatureFlagBits {
    VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT = 0x00000001,
    VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT = 0x00000002,
    VK_FORMAT_FEATURE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkFormatFeatureFlagBits;
typedef enum VkImageUsageFlagBits {
    VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002,
    VK_IMAGE_USAGE_SAMPLED_BIT = 0x00000004,
    VK_IMAGE_USAGE_STORAGE_BIT = 0x00000008,
    VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkImageUsageFlagBits;
typedef enum VkInstanceCreateFlagBits {
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x00000001,
    VK_INSTANCE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkInstanceCreateFlagBits;
typedef enum VkMemoryPropertyFlagBits {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002,
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004,
    VK_MEMORY_PROPERTY_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkMemoryPropertyFlagBits;
typedef enum VkQueueFlagBits {
    VK_QUEUE_COMPUTE_BIT = 0x00000002,
    VK_QUEUE_TRANSFER_BIT = 0x00000004,
    VK_QUEUE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkQueueFlagBits;
typedef enum VkPipelineStageFlagBits {
    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT = 0x00000001,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 0x00000800,
    VK_PIPELINE_STAGE_TRANSFER_BIT = 0x00001000,
    VK_PIPELINE_STAGE_HOST_BIT = 0x00004000,
    VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkPipelineStageFlagBits;
typedef enum VkBufferUsageFlagBits {
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002,
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020,
    VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkBufferUsageFlagBits;
typedef enum VkCommandPoolCreateFlagBits {
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x00000002,
    VK_COMMAND_POOL_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkCommandPoolCreateFlagBits;
typedef enum VkCommandBufferUsageFlagBits {
    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 0x00000001,
    VK_COMMAND_BUFFER_USAGE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkCommandBufferUsageFlagBits;
typedef struct VkExtent3D {
    uint32_t    width;
    uint32_t    height;
    uint32_t    depth;
} VkExtent3D;
typedef struct VkOffset3D {
    int32_t    x;
    int32_t    y;
    int32_t    z;
} VkOffset3D;
typedef struct VkExtent2D {
    uint32_t    width;
    uint32_t    height;
} VkExtent2D;
typedef struct VkApplicationInfo {
    VkStructureType    sType;
    const void*        pNext;
    const char*        pApplicationName;
    uint32_t           applicationVersion;
    const char*        pEngineName;
    uint32_t           engineVersion;
    uint32_t           apiVersion;
} VkApplicationInfo;
typedef VkFlags VkInstanceCreateFlags;
typedef struct VkInstanceCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkInstanceCreateFlags       flags;
    const VkApplicationInfo*    pApplicationInfo;
    uint32_t                    enabledLayerCount;
    const char* const*          ppEnabledLayerNames;
    uint32_t                    enabledExtensionCount;
    const char* const*          ppEnabledExtensionNames;
} VkInstanceCreateInfo;
typedef VkFlags VkSampleCountFlags;
typedef struct VkPhysicalDeviceLimits {
    uint32_t              maxImageDimension1D;
    uint32_t              maxImageDimension2D;
    uint32_t              maxImageDimension3D;
    uint32_t              maxImageDimensionCube;
    uint32_t              maxImageArrayLayers;
    uint32_t              maxTexelBufferElements;
    uint32_t              maxUniformBufferRange;
    uint32_t              maxStorageBufferRange;
    uint32_t              maxPushConstantsSize;
    uint32_t              maxMemoryAllocationCount;
    uint32_t              maxSamplerAllocationCount;
    VkDeviceSize          bufferImageGranularity;
    VkDeviceSize          sparseAddressSpaceSize;
    uint32_t              maxBoundDescriptorSets;
    uint32_t              maxPerStageDescriptorSamplers;
    uint32_t              maxPerStageDescriptorUniformBuffers;
    uint32_t              maxPerStageDescriptorStorageBuffers;
    uint32_t              maxPerStageDescriptorSampledImages;
    uint32_t              maxPerStageDescriptorStorageImages;
    uint32_t              maxPerStageDescriptorInputAttachments;
    uint32_t              maxPerStageResources;
    uint32_t              maxDescriptorSetSamplers;
    uint32_t              maxDescriptorSetUniformBuffers;
    uint32_t              maxDescriptorSetUniformBuffersDynamic;
    uint32_t              maxDescriptorSetStorageBuffers;
    uint32_t              maxDescriptorSetStorageBuffersDynamic;
    uint32_t              maxDescriptorSetSampledImages;
    uint32_t              maxDescriptorSetStorageImages;
    uint32_t              maxDescriptorSetInputAttachments;
    uint32_t              maxVertexInputAttributes;
    uint32_t              maxVertexInputBindings;
    uint32_t              maxVertexInputAttributeOffset;
    uint32_t              maxVertexInputBindingStride;
    uint32_t              maxVertexOutputComponents;
    uint32_t              maxTessellationGenerationLevel;
    uint32_t              maxTessellationPatchSize;
    uint32_t              maxTessellationControlPerVertexInputComponents;
    uint32_t              maxTessellationControlPerVertexOutputComponents;
    uint32_t              maxTessellationControlPerPatchOutputComponents;
    uint32_t              maxTessellationControlTotalOutputComponents;
    uint32_t              maxTessellationEvaluationInputComponents;
    uint32_t              maxTessellationEvaluationOutputComponents;
    uint32_t              maxGeometryShaderInvocations;
    uint32_t              maxGeometryInputComponents;
    uint32_t              maxGeometryOutputComponents;
    uint32_t              maxGeometryOutputVertices;
    uint32_t              maxGeometryTotalOutputComponents;
    uint32_t              maxFragmentInputComponents;
    uint32_t              maxFragmentOutputAttachments;
    uint32_t              maxFragmentDualSrcAttachments;
    uint32_t              maxFragmentCombinedOutputResources;
    uint32_t              maxComputeSharedMemorySize;
    uint32_t              maxComputeWorkGroupCount[3];
    uint32_t              maxComputeWorkGroupInvocations;
    uint32_t              maxComputeWorkGroupSize[3];
    uint32_t              subPixelPrecisionBits;
    uint32_t              subTexelPrecisionBits;
    uint32_t              mipmapPrecisionBits;
    uint32_t              maxDrawIndexedIndexValue;
    uint32_t              maxDrawIndirectCount;
    float                 maxSamplerLodBias;
    float                 maxSamplerAnisotropy;
    uint32_t              maxViewports;
    uint32_t              maxViewportDimensions[2];
    float                 viewportBoundsRange[2];
    uint32_t              viewportSubPixelBits;
    size_t                minMemoryMapAlignment;
    VkDeviceSize          minTexelBufferOffsetAlignment;
    VkDeviceSize          minUniformBufferOffsetAlignment;
    VkDeviceSize          minStorageBufferOffsetAlignment;
    int32_t               minTexelOffset;
    uint32_t              maxTexelOffset;
    int32_t               minTexelGatherOffset;
    uint32_t              maxTexelGatherOffset;
    float                 minInterpolationOffset;
    float                 maxInterpolationOffset;
    uint32_t              subPixelInterpolationOffsetBits;
    uint32_t              maxFramebufferWidth;
    uint32_t              maxFramebufferHeight;
    uint32_t              maxFramebufferLayers;
    VkSampleCountFlags    framebufferColorSampleCounts;
    VkSampleCountFlags    framebufferDepthSampleCounts;
    VkSampleCountFlags    framebufferStencilSampleCounts;
    VkSampleCountFlags    framebufferNoAttachmentsSampleCounts;
    uint32_t              maxColorAttachments;
    VkSampleCountFlags    sampledImageColorSampleCounts;
    VkSampleCountFlags    sampledImageIntegerSampleCounts;
    VkSampleCountFlags    sampledImageDepthSampleCounts;
    VkSampleCountFlags    sampledImageStencilSampleCounts;
    VkSampleCountFlags    storageImageSampleCounts;
    uint32_t              maxSampleMaskWords;
    VkBool32              timestampComputeAndGraphics;
    float                 timestampPeriod;
    uint32_t              maxClipDistances;
    uint32_t              maxCullDistances;
    uint32_t              maxCombinedClipAndCullDistances;
    uint32_t              discreteQueuePriorities;
    float                 pointSizeRange[2];
    float                 lineWidthRange[2];
    float                 pointSizeGranularity;
    float                 lineWidthGranularity;
    VkBool32              strictLines;
    VkBool32              standardSampleLocations;
    VkDeviceSize          optimalBufferCopyOffsetAlignment;
    VkDeviceSize          optimalBufferCopyRowPitchAlignment;
    VkDeviceSize          nonCoherentAtomSize;
} VkPhysicalDeviceLimits;
typedef struct VkPhysicalDeviceSparseProperties {
    VkBool32    residencyStandard2DBlockShape;
    VkBool32    residencyStandard2DMultisampleBlockShape;
    VkBool32    residencyStandard3DBlockShape;
    VkBool32    residencyAlignedMipSize;
    VkBool32    residencyNonResidentStrict;
} VkPhysicalDeviceSparseProperties;
typedef struct VkPhysicalDeviceProperties {
    uint32_t                            apiVersion;
    uint32_t                            driverVersion;
    uint32_t                            vendorID;
    uint32_t                            deviceID;
    VkPhysicalDeviceType                deviceType;
    char                                deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
    uint8_t                             pipelineCacheUUID[VK_UUID_SIZE];
    VkPhysicalDeviceLimits              limits;
    VkPhysicalDeviceSparseProperties    sparseProperties;
} VkPhysicalDeviceProperties;
typedef VkFlags VkQueueFlags;
typedef struct VkQueueFamilyProperties {
    VkQueueFlags    queueFlags;
    uint32_t        queueCount;
    uint32_t        timestampValidBits;
    VkExtent3D      minImageTransferGranularity;
} VkQueueFamilyProperties;
typedef VkFlags VkDeviceQueueCreateFlags;
typedef struct VkDeviceQueueCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkDeviceQueueCreateFlags    flags;
    uint32_t                    queueFamilyIndex;
    uint32_t                    queueCount;
    const float*                pQueuePriorities;
} VkDeviceQueueCreateInfo;
typedef struct VkPhysicalDeviceFeatures {
    VkBool32    robustBufferAccess;
    VkBool32    fullDrawIndexUint32;
    VkBool32    imageCubeArray;
    VkBool32    independentBlend;
    VkBool32    geometryShader;
    VkBool32    tessellationShader;
    VkBool32    sampleRateShading;
    VkBool32    dualSrcBlend;
    VkBool32    logicOp;
    VkBool32    multiDrawIndirect;
    VkBool32    drawIndirectFirstInstance;
    VkBool32    depthClamp;
    VkBool32    depthBiasClamp;
    VkBool32    fillModeNonSolid;
    VkBool32    depthBounds;
    VkBool32    wideLines;
    VkBool32    largePoints;
    VkBool32    alphaToOne;
    VkBool32    multiViewport;
    VkBool32    samplerAnisotropy;
    VkBool32    textureCompressionETC2;
    VkBool32    textureCompressionASTC_LDR;
    VkBool32    textureCompressionBC;
    VkBool32    occlusionQueryPrecise;
    VkBool32    pipelineStatisticsQuery;
    VkBool32    vertexPipelineStoresAndAtomics;
    VkBool32    fragmentStoresAndAtomics;
    VkBool32    shaderTessellationAndGeometryPointSize;
    VkBool32    shaderImageGatherExtended;
    VkBool32    shaderStorageImageExtendedFormats;
    VkBool32    shaderStorageImageMultisample;
    VkBool32    shaderStorageImageReadWithoutFormat;
    VkBool32    shaderStorageImageWriteWithoutFormat;
    VkBool32    shaderUniformBufferArrayDynamicIndexing;
    VkBool32    shaderSampledImageArrayDynamicIndexing;
    VkBool32    shaderStorageBufferArrayDynamicIndexing;
    VkBool32    shaderStorageImageArrayDynamicIndexing;
    VkBool32    shaderClipDistance;
    VkBool32    shaderCullDistance;
    VkBool32    shaderFloat64;
    VkBool32    shaderInt64;
    VkBool32    shaderInt16;
    VkBool32    shaderResourceResidency;
    VkBool32    shaderResourceMinLod;
    VkBool32    sparseBinding;
    VkBool32    sparseResidencyBuffer;
    VkBool32    sparseResidencyImage2D;
    VkBool32    sparseResidencyImage3D;
    VkBool32    sparseResidency2Samples;
    VkBool32    sparseResidency4Samples;
    VkBool32    sparseResidency8Samples;
    VkBool32    sparseResidency16Samples;
    VkBool32    sparseResidencyAliased;
    VkBool32    variableMultisampleRate;
    VkBool32    inheritedQueries;
} VkPhysicalDeviceFeatures;
typedef VkFlags VkDeviceCreateFlags;
typedef struct VkDeviceCreateInfo {
    VkStructureType                    sType;
    const void*                        pNext;
    VkDeviceCreateFlags                flags;
    uint32_t                           queueCreateInfoCount;
    const VkDeviceQueueCreateInfo*     pQueueCreateInfos;
    // enabledLayerCount is legacy and should not be used
    uint32_t                           enabledLayerCount;
    // ppEnabledLayerNames is legacy and should not be used
    const char* const*                 ppEnabledLayerNames;
    uint32_t                           enabledExtensionCount;
    const char* const*                 ppEnabledExtensionNames;
    const VkPhysicalDeviceFeatures*    pEnabledFeatures;
} VkDeviceCreateInfo;
typedef struct VkExtensionProperties {
    char        extensionName[VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t    specVersion;
} VkExtensionProperties;
typedef VkFlags VkMemoryPropertyFlags;
typedef struct VkMemoryType {
    VkMemoryPropertyFlags    propertyFlags;
    uint32_t                 heapIndex;
} VkMemoryType;
typedef VkFlags VkMemoryHeapFlags;
typedef struct VkMemoryHeap {
    VkDeviceSize         size;
    VkMemoryHeapFlags    flags;
} VkMemoryHeap;
typedef struct VkPhysicalDeviceMemoryProperties {
    uint32_t        memoryTypeCount;
    VkMemoryType    memoryTypes[VK_MAX_MEMORY_TYPES];
    uint32_t        memoryHeapCount;
    VkMemoryHeap    memoryHeaps[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryProperties;
typedef struct VkMemoryRequirements {
    VkDeviceSize    size;
    VkDeviceSize    alignment;
    uint32_t        memoryTypeBits;
} VkMemoryRequirements;
typedef struct VkMemoryAllocateInfo {
    VkStructureType    sType;
    const void*        pNext;
    VkDeviceSize       allocationSize;
    uint32_t           memoryTypeIndex;
} VkMemoryAllocateInfo;
typedef VkFlags VkBufferCreateFlags;
typedef VkFlags VkBufferUsageFlags;
typedef struct VkBufferCreateInfo {
    VkStructureType        sType;
    const void*            pNext;
    VkBufferCreateFlags    flags;
    VkDeviceSize           size;
    VkBufferUsageFlags     usage;
    VkSharingMode          sharingMode;
    uint32_t               queueFamilyIndexCount;
    const uint32_t*        pQueueFamilyIndices;
} VkBufferCreateInfo;
typedef VkFlags VkImageCreateFlags;
typedef VkFlags VkImageUsageFlags;
typedef struct VkImageCreateInfo {
    VkStructureType          sType;
    const void*              pNext;
    VkImageCreateFlags       flags;
    VkImageType              imageType;
    VkFormat                 format;
    VkExtent3D               extent;
    uint32_t                 mipLevels;
    uint32_t                 arrayLayers;
    VkSampleCountFlagBits    samples;
    VkImageTiling            tiling;
    VkImageUsageFlags        usage;
    VkSharingMode            sharingMode;
    uint32_t                 queueFamilyIndexCount;
    const uint32_t*          pQueueFamilyIndices;
    VkImageLayout            initialLayout;
} VkImageCreateInfo;
typedef VkFlags VkImageAspectFlags;
typedef struct VkImageSubresourceRange {
    VkImageAspectFlags    aspectMask;
    uint32_t              baseMipLevel;
    uint32_t              levelCount;
    uint32_t              baseArrayLayer;
    uint32_t              layerCount;
} VkImageSubresourceRange;
typedef struct VkComponentMapping {
    VkComponentSwizzle    r;
    VkComponentSwizzle    g;
    VkComponentSwizzle    b;
    VkComponentSwizzle    a;
} VkComponentMapping;
typedef VkFlags VkImageViewCreateFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkImage)
typedef struct VkImageViewCreateInfo {
    VkStructureType            sType;
    const void*                pNext;
    VkImageViewCreateFlags     flags;
    VkImage                    image;
    VkImageViewType            viewType;
    VkFormat                   format;
    VkComponentMapping         components;
    VkImageSubresourceRange    subresourceRange;
} VkImageViewCreateInfo;
typedef VkFlags VkSamplerCreateFlags;
typedef struct VkSamplerCreateInfo {
    VkStructureType         sType;
    const void*             pNext;
    VkSamplerCreateFlags    flags;
    VkFilter                magFilter;
    VkFilter                minFilter;
    VkSamplerMipmapMode     mipmapMode;
    VkSamplerAddressMode    addressModeU;
    VkSamplerAddressMode    addressModeV;
    VkSamplerAddressMode    addressModeW;
    float                   mipLodBias;
    VkBool32                anisotropyEnable;
    float                   maxAnisotropy;
    VkBool32                compareEnable;
    VkCompareOp             compareOp;
    float                   minLod;
    float                   maxLod;
    VkBorderColor           borderColor;
    VkBool32                unnormalizedCoordinates;
} VkSamplerCreateInfo;
typedef VkFlags VkShaderModuleCreateFlags;
typedef struct VkShaderModuleCreateInfo {
    VkStructureType              sType;
    const void*                  pNext;
    VkShaderModuleCreateFlags    flags;
    size_t                       codeSize;
    const uint32_t*              pCode;
} VkShaderModuleCreateInfo;
typedef VkFlags VkShaderStageFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSampler)
typedef struct VkDescriptorSetLayoutBinding {
    uint32_t              binding;
    VkDescriptorType      descriptorType;
    uint32_t              descriptorCount;
    VkShaderStageFlags    stageFlags;
    const VkSampler*      pImmutableSamplers;
} VkDescriptorSetLayoutBinding;
typedef VkFlags VkDescriptorSetLayoutCreateFlags;
typedef struct VkDescriptorSetLayoutCreateInfo {
    VkStructureType                        sType;
    const void*                            pNext;
    VkDescriptorSetLayoutCreateFlags       flags;
    uint32_t                               bindingCount;
    const VkDescriptorSetLayoutBinding*    pBindings;
} VkDescriptorSetLayoutCreateInfo;
typedef struct VkPushConstantRange {
    VkShaderStageFlags    stageFlags;
    uint32_t              offset;
    uint32_t              size;
} VkPushConstantRange;
typedef VkFlags VkPipelineLayoutCreateFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDescriptorSetLayout)
typedef struct VkPipelineLayoutCreateInfo {
    VkStructureType                 sType;
    const void*                     pNext;
    VkPipelineLayoutCreateFlags     flags;
    uint32_t                        setLayoutCount;
    const VkDescriptorSetLayout*    pSetLayouts;
    uint32_t                        pushConstantRangeCount;
    const VkPushConstantRange*      pPushConstantRanges;
} VkPipelineLayoutCreateInfo;
typedef struct VkSpecializationMapEntry {
    uint32_t    constantID;
    uint32_t    offset;
    size_t      size;
} VkSpecializationMapEntry;
typedef struct VkSpecializationInfo {
    uint32_t                           mapEntryCount;
    const VkSpecializationMapEntry*    pMapEntries;
    size_t                             dataSize;
    const void*                        pData;
} VkSpecializationInfo;
typedef VkFlags VkPipelineShaderStageCreateFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkShaderModule)
typedef struct VkPipelineShaderStageCreateInfo {
    VkStructureType                     sType;
    const void*                         pNext;
    VkPipelineShaderStageCreateFlags    flags;
    VkShaderStageFlagBits               stage;
    VkShaderModule                      module;
    const char*                         pName;
    const VkSpecializationInfo*         pSpecializationInfo;
} VkPipelineShaderStageCreateInfo;
typedef VkFlags VkPipelineCreateFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkPipelineLayout)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkPipeline)
typedef struct VkComputePipelineCreateInfo {
    VkStructureType                    sType;
    const void*                        pNext;
    VkPipelineCreateFlags              flags;
    VkPipelineShaderStageCreateInfo    stage;
    VkPipelineLayout                   layout;
    VkPipeline                         basePipelineHandle;
    int32_t                            basePipelineIndex;
} VkComputePipelineCreateInfo;
typedef struct VkDescriptorPoolSize {
    VkDescriptorType    type;
    uint32_t            descriptorCount;
} VkDescriptorPoolSize;
typedef VkFlags VkDescriptorPoolCreateFlags;
typedef struct VkDescriptorPoolCreateInfo {
    VkStructureType                sType;
    const void*                    pNext;
    VkDescriptorPoolCreateFlags    flags;
    uint32_t                       maxSets;
    uint32_t                       poolSizeCount;
    const VkDescriptorPoolSize*    pPoolSizes;
} VkDescriptorPoolCreateInfo;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDescriptorPool)
typedef struct VkDescriptorSetAllocateInfo {
    VkStructureType                 sType;
    const void*                     pNext;
    VkDescriptorPool                descriptorPool;
    uint32_t                        descriptorSetCount;
    const VkDescriptorSetLayout*    pSetLayouts;
} VkDescriptorSetAllocateInfo;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBuffer)
typedef struct VkDescriptorBufferInfo {
    VkBuffer        buffer;
    VkDeviceSize    offset;
    VkDeviceSize    range;
} VkDescriptorBufferInfo;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkImageView)
typedef struct VkDescriptorImageInfo {
    VkSampler        sampler;
    VkImageView      imageView;
    VkImageLayout    imageLayout;
} VkDescriptorImageInfo;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDescriptorSet)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkBufferView)
typedef struct VkWriteDescriptorSet {
    VkStructureType                  sType;
    const void*                      pNext;
    VkDescriptorSet                  dstSet;
    uint32_t                         dstBinding;
    uint32_t                         dstArrayElement;
    uint32_t                         descriptorCount;
    VkDescriptorType                 descriptorType;
    const VkDescriptorImageInfo*     pImageInfo;
    const VkDescriptorBufferInfo*    pBufferInfo;
    const VkBufferView*              pTexelBufferView;
} VkWriteDescriptorSet;
typedef VkFlags VkCommandPoolCreateFlags;
typedef struct VkCommandPoolCreateInfo {
    VkStructureType             sType;
    const void*                 pNext;
    VkCommandPoolCreateFlags    flags;
    uint32_t                    queueFamilyIndex;
} VkCommandPoolCreateInfo;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkCommandPool)
typedef struct VkCommandBufferAllocateInfo {
    VkStructureType         sType;
    const void*             pNext;
    VkCommandPool           commandPool;
    VkCommandBufferLevel    level;
    uint32_t                commandBufferCount;
} VkCommandBufferAllocateInfo;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkRenderPass)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkFramebuffer)
typedef VkFlags VkQueryControlFlags;
typedef VkFlags VkQueryPipelineStatisticFlags;
typedef struct VkCommandBufferInheritanceInfo {
    VkStructureType                  sType;
    const void*                      pNext;
    VkRenderPass                     renderPass;
    uint32_t                         subpass;
    VkFramebuffer                    framebuffer;
    VkBool32                         occlusionQueryEnable;
    VkQueryControlFlags              queryFlags;
    VkQueryPipelineStatisticFlags    pipelineStatistics;
} VkCommandBufferInheritanceInfo;
typedef VkFlags VkCommandBufferUsageFlags;
typedef struct VkCommandBufferBeginInfo {
    VkStructureType                          sType;
    const void*                              pNext;
    VkCommandBufferUsageFlags                flags;
    const VkCommandBufferInheritanceInfo*    pInheritanceInfo;
} VkCommandBufferBeginInfo;
typedef struct VkBufferCopy {
    VkDeviceSize    srcOffset;
    VkDeviceSize    dstOffset;
    VkDeviceSize    size;
} VkBufferCopy;
typedef struct VkImageSubresourceLayers {
    VkImageAspectFlags    aspectMask;
    uint32_t              mipLevel;
    uint32_t              baseArrayLayer;
    uint32_t              layerCount;
} VkImageSubresourceLayers;
typedef struct VkBufferImageCopy {
    VkDeviceSize                bufferOffset;
    uint32_t                    bufferRowLength;
    uint32_t                    bufferImageHeight;
    VkImageSubresourceLayers    imageSubresource;
    VkOffset3D                  imageOffset;
    VkExtent3D                  imageExtent;
} VkBufferImageCopy;
typedef VkFlags VkAccessFlags;
typedef struct VkMemoryBarrier {
    VkStructureType    sType;
    const void*        pNext;
    VkAccessFlags      srcAccessMask;
    VkAccessFlags      dstAccessMask;
} VkMemoryBarrier;
typedef struct VkBufferMemoryBarrier {
    VkStructureType    sType;
    const void*        pNext;
    VkAccessFlags      srcAccessMask;
    VkAccessFlags      dstAccessMask;
    uint32_t           srcQueueFamilyIndex;
    uint32_t           dstQueueFamilyIndex;
    VkBuffer           buffer;
    VkDeviceSize       offset;
    VkDeviceSize       size;
} VkBufferMemoryBarrier;
typedef struct VkImageMemoryBarrier {
    VkStructureType            sType;
    const void*                pNext;
    VkAccessFlags              srcAccessMask;
    VkAccessFlags              dstAccessMask;
    VkImageLayout              oldLayout;
    VkImageLayout              newLayout;
    uint32_t                   srcQueueFamilyIndex;
    uint32_t                   dstQueueFamilyIndex;
    VkImage                    image;
    VkImageSubresourceRange    subresourceRange;
} VkImageMemoryBarrier;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSemaphore)
typedef VkFlags VkPipelineStageFlags;
VK_DEFINE_HANDLE(VkCommandBuffer)
typedef struct VkSubmitInfo {
    VkStructureType                sType;
    const void*                    pNext;
    uint32_t                       waitSemaphoreCount;
    const VkSemaphore*             pWaitSemaphores;
    const VkPipelineStageFlags*    pWaitDstStageMask;
    uint32_t                       commandBufferCount;
    const VkCommandBuffer*         pCommandBuffers;
    uint32_t                       signalSemaphoreCount;
    const VkSemaphore*             pSignalSemaphores;
} VkSubmitInfo;
typedef VkFlags VkFenceCreateFlags;
typedef struct VkFenceCreateInfo {
    VkStructureType       sType;
    const void*           pNext;
    VkFenceCreateFlags    flags;
} VkFenceCreateInfo;
typedef VkFlags VkFormatFeatureFlags;
typedef struct VkFormatProperties {
    VkFormatFeatureFlags    linearTilingFeatures;
    VkFormatFeatureFlags    optimalTilingFeatures;
    VkFormatFeatureFlags    bufferFeatures;
} VkFormatProperties;
VK_DEFINE_HANDLE(VkInstance)
VK_DEFINE_HANDLE(VkPhysicalDevice)
VK_DEFINE_HANDLE(VkDevice)
VK_DEFINE_HANDLE(VkQueue)
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDeviceMemory)
typedef VkFlags VkMemoryMapFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkPipelineCache)
typedef VkFlags VkDescriptorPoolResetFlags;
typedef struct VkCopyDescriptorSet {
    VkStructureType    sType;
    const void*        pNext;
    VkDescriptorSet    srcSet;
    uint32_t           srcBinding;
    uint32_t           srcArrayElement;
    VkDescriptorSet    dstSet;
    uint32_t           dstBinding;
    uint32_t           dstArrayElement;
    uint32_t           descriptorCount;
} VkCopyDescriptorSet;
typedef VkFlags VkCommandBufferResetFlags;
typedef VkFlags VkDependencyFlags;
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkFence)

typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vkGetInstanceProcAddr)(VkInstance instance, const char* pName);
typedef VkResult (VKAPI_PTR *PFN_vkCreateInstance)(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkInstance* pInstance);
typedef void (VKAPI_PTR *PFN_vkDestroyInstance)(VkInstance instance, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateInstanceExtensionProperties)(const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties);
typedef VkResult (VKAPI_PTR *PFN_vkEnumeratePhysicalDevices)(VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceProperties)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties* pProperties);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceQueueFamilyProperties)(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceMemoryProperties)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties* pMemoryProperties);
typedef void (VKAPI_PTR *PFN_vkGetPhysicalDeviceFormatProperties)(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties* pFormatProperties);
typedef VkResult (VKAPI_PTR *PFN_vkEnumerateDeviceExtensionProperties)(VkPhysicalDevice physicalDevice, const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDevice)(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDevice* pDevice);
typedef void (VKAPI_PTR *PFN_vkDestroyDevice)(VkDevice device, const VkAllocationCallbacks* pAllocator);
typedef PFN_vkVoidFunction (VKAPI_PTR *PFN_vkGetDeviceProcAddr)(VkDevice device, const char* pName);
typedef void (VKAPI_PTR *PFN_vkGetDeviceQueue)(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue* pQueue);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateMemory)(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory);
typedef void (VKAPI_PTR *PFN_vkFreeMemory)(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkMapMemory)(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData);
typedef void (VKAPI_PTR *PFN_vkUnmapMemory)(VkDevice device, VkDeviceMemory memory);
typedef VkResult (VKAPI_PTR *PFN_vkCreateBuffer)(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer);
typedef void (VKAPI_PTR *PFN_vkDestroyBuffer)(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator);
typedef void (VKAPI_PTR *PFN_vkGetBufferMemoryRequirements)(VkDevice device, VkBuffer buffer, VkMemoryRequirements* pMemoryRequirements);
typedef VkResult (VKAPI_PTR *PFN_vkBindBufferMemory)(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset);
typedef VkResult (VKAPI_PTR *PFN_vkCreateImage)(VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage);
typedef void (VKAPI_PTR *PFN_vkDestroyImage)(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator);
typedef void (VKAPI_PTR *PFN_vkGetImageMemoryRequirements)(VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements);
typedef VkResult (VKAPI_PTR *PFN_vkBindImageMemory)(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset);
typedef VkResult (VKAPI_PTR *PFN_vkCreateImageView)(VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView);
typedef void (VKAPI_PTR *PFN_vkDestroyImageView)(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreateSampler)(VkDevice device, const VkSamplerCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSampler* pSampler);
typedef void (VKAPI_PTR *PFN_vkDestroySampler)(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreateShaderModule)(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule);
typedef void (VKAPI_PTR *PFN_vkDestroyShaderModule)(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDescriptorSetLayout)(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout);
typedef void (VKAPI_PTR *PFN_vkDestroyDescriptorSetLayout)(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreatePipelineLayout)(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout);
typedef void (VKAPI_PTR *PFN_vkDestroyPipelineLayout)(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreateComputePipelines)(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkComputePipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines);
typedef void (VKAPI_PTR *PFN_vkDestroyPipeline)(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkCreateDescriptorPool)(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool);
typedef void (VKAPI_PTR *PFN_vkDestroyDescriptorPool)(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkResetDescriptorPool)(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorPoolResetFlags flags);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateDescriptorSets)(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets);
typedef void (VKAPI_PTR *PFN_vkUpdateDescriptorSets)(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies);
typedef VkResult (VKAPI_PTR *PFN_vkCreateCommandPool)(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool);
typedef void (VKAPI_PTR *PFN_vkDestroyCommandPool)(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkAllocateCommandBuffers)(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers);
typedef VkResult (VKAPI_PTR *PFN_vkBeginCommandBuffer)(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo);
typedef VkResult (VKAPI_PTR *PFN_vkEndCommandBuffer)(VkCommandBuffer commandBuffer);
typedef VkResult (VKAPI_PTR *PFN_vkResetCommandBuffer)(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags);
typedef void (VKAPI_PTR *PFN_vkCmdBindPipeline)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline);
typedef void (VKAPI_PTR *PFN_vkCmdBindDescriptorSets)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets);
typedef void (VKAPI_PTR *PFN_vkCmdPushConstants)(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues);
typedef void (VKAPI_PTR *PFN_vkCmdDispatch)(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
typedef void (VKAPI_PTR *PFN_vkCmdPipelineBarrier)(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers);
typedef void (VKAPI_PTR *PFN_vkCmdCopyBuffer)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer, uint32_t regionCount, const VkBufferCopy* pRegions);
typedef void (VKAPI_PTR *PFN_vkCmdCopyBufferToImage)(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkBufferImageCopy* pRegions);
typedef void (VKAPI_PTR *PFN_vkCmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data);
typedef VkResult (VKAPI_PTR *PFN_vkQueueSubmit)(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence);
typedef VkResult (VKAPI_PTR *PFN_vkQueueWaitIdle)(VkQueue queue);
typedef VkResult (VKAPI_PTR *PFN_vkCreateFence)(VkDevice device, const VkFenceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFence* pFence);
typedef void (VKAPI_PTR *PFN_vkDestroyFence)(VkDevice device, VkFence fence, const VkAllocationCallbacks* pAllocator);
typedef VkResult (VKAPI_PTR *PFN_vkWaitForFences)(VkDevice device, uint32_t fenceCount, const VkFence* pFences, VkBool32 waitAll, uint64_t timeout);
typedef VkResult (VKAPI_PTR *PFN_vkResetFences)(VkDevice device, uint32_t fenceCount, const VkFence* pFences);
typedef VkResult (VKAPI_PTR *PFN_vkDeviceWaitIdle)(VkDevice device);

/* ------------------------------------------------------------------------- */
/* Loader                                                                    */
/* ------------------------------------------------------------------------- */

typedef struct anbcVk {
    void* library;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkDestroyInstance DestroyInstance;
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice CreateDevice;
    PFN_vkDestroyDevice DestroyDevice;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
    PFN_vkGetDeviceQueue GetDeviceQueue;
    PFN_vkAllocateMemory AllocateMemory;
    PFN_vkFreeMemory FreeMemory;
    PFN_vkMapMemory MapMemory;
    PFN_vkUnmapMemory UnmapMemory;
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
    PFN_vkBindBufferMemory BindBufferMemory;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
    PFN_vkBindImageMemory BindImageMemory;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;
    PFN_vkCreateSampler CreateSampler;
    PFN_vkDestroySampler DestroySampler;
    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;
    PFN_vkCreateDescriptorPool CreateDescriptorPool;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
    PFN_vkResetDescriptorPool ResetDescriptorPool;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
    PFN_vkCreateCommandPool CreateCommandPool;
    PFN_vkDestroyCommandPool DestroyCommandPool;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
    PFN_vkBeginCommandBuffer BeginCommandBuffer;
    PFN_vkEndCommandBuffer EndCommandBuffer;
    PFN_vkResetCommandBuffer ResetCommandBuffer;
    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
    PFN_vkCmdFillBuffer CmdFillBuffer;
    PFN_vkQueueSubmit QueueSubmit;
    PFN_vkQueueWaitIdle QueueWaitIdle;
    PFN_vkCreateFence CreateFence;
    PFN_vkDestroyFence DestroyFence;
    PFN_vkWaitForFences WaitForFences;
    PFN_vkResetFences ResetFences;
    PFN_vkDeviceWaitIdle DeviceWaitIdle;
} anbcVk;

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void* anbcVkOpen(const char* name) { return (void*)LoadLibraryA(name); }
static void* anbcVkSym(void* lib, const char* name) { return (void*)GetProcAddress((HMODULE)lib, name); }
static void  anbcVkClose(void* lib) { FreeLibrary((HMODULE)lib); }
#else
#include <dlfcn.h>
static void* anbcVkOpen(const char* name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
static void* anbcVkSym(void* lib, const char* name) { return dlsym(lib, name); }
static void  anbcVkClose(void* lib) { dlclose(lib); }
#endif

/* Opens the platform's Vulkan loader (or MoltenVK directly on macOS when
 * the SDK loader is absent) and resolves the global entry points. */
static int anbcVkLoad(anbcVk* vk)
{
    static const char* const kNames[] = {
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib",
        "/usr/local/lib/libvulkan.1.dylib", "/usr/local/lib/libMoltenVK.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
    };
    for (size_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]) && !vk->library; i++)
        vk->library = anbcVkOpen(kNames[i]);
    if (!vk->library)
        return 0;
    vk->GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)anbcVkSym(vk->library, "vkGetInstanceProcAddr");
    if (!vk->GetInstanceProcAddr) {
        anbcVkClose(vk->library);
        vk->library = NULL;
        return 0;
    }
    vk->CreateInstance = (PFN_vkCreateInstance)vk->GetInstanceProcAddr(NULL, "vkCreateInstance");
    vk->EnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)vk->GetInstanceProcAddr(NULL, "vkEnumerateInstanceExtensionProperties");
    return 1;
}

static void anbcVkLoadInstance(anbcVk* vk, VkInstance instance)
{
    vk->DestroyInstance = (PFN_vkDestroyInstance)vk->GetInstanceProcAddr(instance, "vkDestroyInstance");
    vk->EnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)vk->GetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices");
    vk->GetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vk->GetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties");
    vk->GetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)vk->GetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    vk->GetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vk->GetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties");
    vk->GetPhysicalDeviceFormatProperties = (PFN_vkGetPhysicalDeviceFormatProperties)vk->GetInstanceProcAddr(instance, "vkGetPhysicalDeviceFormatProperties");
    vk->EnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)vk->GetInstanceProcAddr(instance, "vkEnumerateDeviceExtensionProperties");
    vk->CreateDevice = (PFN_vkCreateDevice)vk->GetInstanceProcAddr(instance, "vkCreateDevice");
    vk->GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vk->GetInstanceProcAddr(instance, "vkGetDeviceProcAddr");
}

static void anbcVkLoadDevice(anbcVk* vk, VkDevice device)
{
    vk->DestroyDevice = (PFN_vkDestroyDevice)vk->GetDeviceProcAddr(device, "vkDestroyDevice");
    vk->GetDeviceQueue = (PFN_vkGetDeviceQueue)vk->GetDeviceProcAddr(device, "vkGetDeviceQueue");
    vk->AllocateMemory = (PFN_vkAllocateMemory)vk->GetDeviceProcAddr(device, "vkAllocateMemory");
    vk->FreeMemory = (PFN_vkFreeMemory)vk->GetDeviceProcAddr(device, "vkFreeMemory");
    vk->MapMemory = (PFN_vkMapMemory)vk->GetDeviceProcAddr(device, "vkMapMemory");
    vk->UnmapMemory = (PFN_vkUnmapMemory)vk->GetDeviceProcAddr(device, "vkUnmapMemory");
    vk->CreateBuffer = (PFN_vkCreateBuffer)vk->GetDeviceProcAddr(device, "vkCreateBuffer");
    vk->DestroyBuffer = (PFN_vkDestroyBuffer)vk->GetDeviceProcAddr(device, "vkDestroyBuffer");
    vk->GetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vk->GetDeviceProcAddr(device, "vkGetBufferMemoryRequirements");
    vk->BindBufferMemory = (PFN_vkBindBufferMemory)vk->GetDeviceProcAddr(device, "vkBindBufferMemory");
    vk->CreateImage = (PFN_vkCreateImage)vk->GetDeviceProcAddr(device, "vkCreateImage");
    vk->DestroyImage = (PFN_vkDestroyImage)vk->GetDeviceProcAddr(device, "vkDestroyImage");
    vk->GetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)vk->GetDeviceProcAddr(device, "vkGetImageMemoryRequirements");
    vk->BindImageMemory = (PFN_vkBindImageMemory)vk->GetDeviceProcAddr(device, "vkBindImageMemory");
    vk->CreateImageView = (PFN_vkCreateImageView)vk->GetDeviceProcAddr(device, "vkCreateImageView");
    vk->DestroyImageView = (PFN_vkDestroyImageView)vk->GetDeviceProcAddr(device, "vkDestroyImageView");
    vk->CreateSampler = (PFN_vkCreateSampler)vk->GetDeviceProcAddr(device, "vkCreateSampler");
    vk->DestroySampler = (PFN_vkDestroySampler)vk->GetDeviceProcAddr(device, "vkDestroySampler");
    vk->CreateShaderModule = (PFN_vkCreateShaderModule)vk->GetDeviceProcAddr(device, "vkCreateShaderModule");
    vk->DestroyShaderModule = (PFN_vkDestroyShaderModule)vk->GetDeviceProcAddr(device, "vkDestroyShaderModule");
    vk->CreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)vk->GetDeviceProcAddr(device, "vkCreateDescriptorSetLayout");
    vk->DestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)vk->GetDeviceProcAddr(device, "vkDestroyDescriptorSetLayout");
    vk->CreatePipelineLayout = (PFN_vkCreatePipelineLayout)vk->GetDeviceProcAddr(device, "vkCreatePipelineLayout");
    vk->DestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)vk->GetDeviceProcAddr(device, "vkDestroyPipelineLayout");
    vk->CreateComputePipelines = (PFN_vkCreateComputePipelines)vk->GetDeviceProcAddr(device, "vkCreateComputePipelines");
    vk->DestroyPipeline = (PFN_vkDestroyPipeline)vk->GetDeviceProcAddr(device, "vkDestroyPipeline");
    vk->CreateDescriptorPool = (PFN_vkCreateDescriptorPool)vk->GetDeviceProcAddr(device, "vkCreateDescriptorPool");
    vk->DestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)vk->GetDeviceProcAddr(device, "vkDestroyDescriptorPool");
    vk->ResetDescriptorPool = (PFN_vkResetDescriptorPool)vk->GetDeviceProcAddr(device, "vkResetDescriptorPool");
    vk->AllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)vk->GetDeviceProcAddr(device, "vkAllocateDescriptorSets");
    vk->UpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)vk->GetDeviceProcAddr(device, "vkUpdateDescriptorSets");
    vk->CreateCommandPool = (PFN_vkCreateCommandPool)vk->GetDeviceProcAddr(device, "vkCreateCommandPool");
    vk->DestroyCommandPool = (PFN_vkDestroyCommandPool)vk->GetDeviceProcAddr(device, "vkDestroyCommandPool");
    vk->AllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)vk->GetDeviceProcAddr(device, "vkAllocateCommandBuffers");
    vk->BeginCommandBuffer = (PFN_vkBeginCommandBuffer)vk->GetDeviceProcAddr(device, "vkBeginCommandBuffer");
    vk->EndCommandBuffer = (PFN_vkEndCommandBuffer)vk->GetDeviceProcAddr(device, "vkEndCommandBuffer");
    vk->ResetCommandBuffer = (PFN_vkResetCommandBuffer)vk->GetDeviceProcAddr(device, "vkResetCommandBuffer");
    vk->CmdBindPipeline = (PFN_vkCmdBindPipeline)vk->GetDeviceProcAddr(device, "vkCmdBindPipeline");
    vk->CmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)vk->GetDeviceProcAddr(device, "vkCmdBindDescriptorSets");
    vk->CmdPushConstants = (PFN_vkCmdPushConstants)vk->GetDeviceProcAddr(device, "vkCmdPushConstants");
    vk->CmdDispatch = (PFN_vkCmdDispatch)vk->GetDeviceProcAddr(device, "vkCmdDispatch");
    vk->CmdPipelineBarrier = (PFN_vkCmdPipelineBarrier)vk->GetDeviceProcAddr(device, "vkCmdPipelineBarrier");
    vk->CmdCopyBuffer = (PFN_vkCmdCopyBuffer)vk->GetDeviceProcAddr(device, "vkCmdCopyBuffer");
    vk->CmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)vk->GetDeviceProcAddr(device, "vkCmdCopyBufferToImage");
    vk->CmdFillBuffer = (PFN_vkCmdFillBuffer)vk->GetDeviceProcAddr(device, "vkCmdFillBuffer");
    vk->QueueSubmit = (PFN_vkQueueSubmit)vk->GetDeviceProcAddr(device, "vkQueueSubmit");
    vk->QueueWaitIdle = (PFN_vkQueueWaitIdle)vk->GetDeviceProcAddr(device, "vkQueueWaitIdle");
    vk->CreateFence = (PFN_vkCreateFence)vk->GetDeviceProcAddr(device, "vkCreateFence");
    vk->DestroyFence = (PFN_vkDestroyFence)vk->GetDeviceProcAddr(device, "vkDestroyFence");
    vk->WaitForFences = (PFN_vkWaitForFences)vk->GetDeviceProcAddr(device, "vkWaitForFences");
    vk->ResetFences = (PFN_vkResetFences)vk->GetDeviceProcAddr(device, "vkResetFences");
    vk->DeviceWaitIdle = (PFN_vkDeviceWaitIdle)vk->GetDeviceProcAddr(device, "vkDeviceWaitIdle");
}

static void anbcVkUnload(anbcVk* vk)
{
    if (vk->library)
        anbcVkClose(vk->library);
    vk->library = NULL;
}

#endif /* ANBC_VULKAN_H */
