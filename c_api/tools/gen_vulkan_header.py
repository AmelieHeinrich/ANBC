#!/usr/bin/env python3
"""Generates src/anbc_vulkan.h: the subset of the Vulkan API the backend
uses, extracted verbatim from the Khronos vulkan_core.h so nothing is
hand-typed (struct layouts and enum values are ABI). The library then
needs no Vulkan SDK at compile time; everything is resolved at runtime
through the loader at the bottom of the generated file.

    gen_vulkan_header.py [--sdk /usr/local/include/vulkan/vulkan_core.h]

Extension entry points (only the portability ones MoltenVK needs) are
plain #defines since they are strings / bits, not types.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STRUCTS = """
VkExtent3D VkOffset3D VkExtent2D
VkApplicationInfo VkInstanceCreateInfo
VkPhysicalDeviceLimits VkPhysicalDeviceSparseProperties VkPhysicalDeviceProperties
VkQueueFamilyProperties VkDeviceQueueCreateInfo VkPhysicalDeviceFeatures VkDeviceCreateInfo
VkExtensionProperties VkMemoryType VkMemoryHeap VkPhysicalDeviceMemoryProperties
VkMemoryRequirements VkMemoryAllocateInfo VkBufferCreateInfo VkImageCreateInfo
VkImageSubresourceRange VkComponentMapping VkImageViewCreateInfo VkSamplerCreateInfo
VkShaderModuleCreateInfo VkDescriptorSetLayoutBinding VkDescriptorSetLayoutCreateInfo
VkPushConstantRange VkPipelineLayoutCreateInfo VkSpecializationMapEntry VkSpecializationInfo
VkPipelineShaderStageCreateInfo VkComputePipelineCreateInfo VkDescriptorPoolSize
VkDescriptorPoolCreateInfo VkDescriptorSetAllocateInfo VkDescriptorBufferInfo
VkDescriptorImageInfo VkWriteDescriptorSet VkCommandPoolCreateInfo VkCommandBufferAllocateInfo
VkCommandBufferInheritanceInfo VkCommandBufferBeginInfo VkBufferCopy VkImageSubresourceLayers
VkBufferImageCopy VkMemoryBarrier VkBufferMemoryBarrier VkImageMemoryBarrier VkSubmitInfo
VkFenceCreateInfo VkFormatProperties
""".split()

FUNCTIONS = """
vkGetInstanceProcAddr vkCreateInstance vkDestroyInstance vkEnumerateInstanceExtensionProperties
vkEnumeratePhysicalDevices vkGetPhysicalDeviceProperties vkGetPhysicalDeviceQueueFamilyProperties
vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceFormatProperties
vkEnumerateDeviceExtensionProperties vkCreateDevice vkDestroyDevice vkGetDeviceProcAddr
vkGetDeviceQueue vkAllocateMemory vkFreeMemory vkMapMemory vkUnmapMemory vkCreateBuffer
vkDestroyBuffer vkGetBufferMemoryRequirements vkBindBufferMemory vkCreateImage vkDestroyImage
vkGetImageMemoryRequirements vkBindImageMemory vkCreateImageView vkDestroyImageView
vkCreateSampler vkDestroySampler vkCreateShaderModule vkDestroyShaderModule
vkCreateDescriptorSetLayout vkDestroyDescriptorSetLayout vkCreatePipelineLayout
vkDestroyPipelineLayout vkCreateComputePipelines vkDestroyPipeline vkCreateDescriptorPool
vkDestroyDescriptorPool vkResetDescriptorPool vkAllocateDescriptorSets vkUpdateDescriptorSets
vkCreateCommandPool vkDestroyCommandPool vkAllocateCommandBuffers vkBeginCommandBuffer
vkEndCommandBuffer vkResetCommandBuffer vkCmdBindPipeline vkCmdBindDescriptorSets
vkCmdPushConstants vkCmdDispatch vkCmdPipelineBarrier vkCmdCopyBuffer vkCmdCopyBufferToImage
vkCmdFillBuffer vkQueueSubmit vkQueueWaitIdle vkCreateFence vkDestroyFence vkWaitForFences
vkResetFences vkDeviceWaitIdle
""".split()

# The global entry points, resolvable from vkGetInstanceProcAddr(NULL, ...).
GLOBAL_FUNCTIONS = {"vkCreateInstance", "vkEnumerateInstanceExtensionProperties"}
# Resolved with vkGetInstanceProcAddr(instance, ...); the rest with vkGetDeviceProcAddr.
INSTANCE_FUNCTIONS = {
    "vkDestroyInstance", "vkEnumeratePhysicalDevices", "vkGetPhysicalDeviceProperties",
    "vkGetPhysicalDeviceQueueFamilyProperties", "vkGetPhysicalDeviceMemoryProperties",
    "vkGetPhysicalDeviceFormatProperties", "vkEnumerateDeviceExtensionProperties", "vkCreateDevice",
    "vkGetDeviceProcAddr",
}

# Enum values the backend names; the enums themselves are emitted with only
# these members (plus the 32-bit-forcing _MAX_ENUM), which is ABI-identical.
ENUM_VALUES = """
VK_SUCCESS VK_NOT_READY VK_TIMEOUT VK_INCOMPLETE VK_ERROR_OUT_OF_HOST_MEMORY
VK_ERROR_OUT_OF_DEVICE_MEMORY VK_ERROR_INITIALIZATION_FAILED VK_ERROR_DEVICE_LOST
VK_ERROR_LAYER_NOT_PRESENT VK_ERROR_EXTENSION_NOT_PRESENT VK_ERROR_FEATURE_NOT_PRESENT
VK_ERROR_INCOMPATIBLE_DRIVER
VK_STRUCTURE_TYPE_APPLICATION_INFO VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
VK_STRUCTURE_TYPE_SUBMIT_INFO VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO
VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO
VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER VK_STRUCTURE_TYPE_MEMORY_BARRIER
VK_IMAGE_LAYOUT_UNDEFINED VK_IMAGE_LAYOUT_GENERAL VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
VK_FORMAT_UNDEFINED VK_FORMAT_R8G8B8A8_UNORM VK_FORMAT_R16G16B16A16_SFLOAT
VK_IMAGE_TILING_OPTIMAL VK_IMAGE_TYPE_2D VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU VK_SHARING_MODE_EXCLUSIVE VK_COMPONENT_SWIZZLE_IDENTITY
VK_IMAGE_VIEW_TYPE_2D VK_COMMAND_BUFFER_LEVEL_PRIMARY VK_BORDER_COLOR_INT_OPAQUE_BLACK
VK_FILTER_NEAREST VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE VK_SAMPLER_MIPMAP_MODE_NEAREST
VK_COMPARE_OP_ALWAYS VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER VK_DESCRIPTOR_TYPE_STORAGE_BUFFER VK_PIPELINE_BIND_POINT_COMPUTE
VK_SAMPLE_COUNT_1_BIT VK_SHADER_STAGE_COMPUTE_BIT
VK_ACCESS_SHADER_READ_BIT VK_ACCESS_SHADER_WRITE_BIT VK_ACCESS_TRANSFER_READ_BIT
VK_ACCESS_TRANSFER_WRITE_BIT VK_ACCESS_HOST_READ_BIT VK_ACCESS_HOST_WRITE_BIT
VK_IMAGE_ASPECT_COLOR_BIT VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
VK_IMAGE_USAGE_TRANSFER_DST_BIT VK_IMAGE_USAGE_SAMPLED_BIT VK_IMAGE_USAGE_STORAGE_BIT
VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
VK_QUEUE_COMPUTE_BIT VK_QUEUE_TRANSFER_BIT
VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT VK_PIPELINE_STAGE_TRANSFER_BIT
VK_PIPELINE_STAGE_HOST_BIT
VK_BUFFER_USAGE_TRANSFER_SRC_BIT VK_BUFFER_USAGE_TRANSFER_DST_BIT VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
""".split()

IDENT_RE = re.compile(r"\bVk[A-Za-z0-9]+\b")


class Sdk:
    def __init__(self, text: str):
        self.text = text
        self.structs = {m.group(1): m.group(0) for m in re.finditer(
            r"typedef struct (Vk\w+) \{\n(?:[^}]*\n)*?\} \1;", text)}
        self.unions = {m.group(1): m.group(0) for m in re.finditer(
            r"typedef union (Vk\w+) \{\n(?:[^}]*\n)*?\} \1;", text)}
        self.enums = {m.group(1): m.group(0) for m in re.finditer(
            r"typedef enum (Vk\w+) \{\n(?:[^}]*\n)*?\} \1;", text)}
        self.flag_typedefs = {name: base for base, name in re.findall(r"typedef (VkFlags|VkFlags64) (Vk\w+);", text)}
        self.handles = set(re.findall(r"VK_DEFINE_HANDLE\((Vk\w+)\)", text))
        self.nd_handles = set(re.findall(r"VK_DEFINE_NON_DISPATCHABLE_HANDLE\((Vk\w+)\)", text))
        self.pfns = {m.group(1): m.group(0) for m in re.finditer(
            r"typedef \w[\w\s\*]*\(VKAPI_PTR \*PFN_(vk\w+)\)\((?:[^;]*\n)*?[^;]*\);", text)}
        self.enum_values = {}
        for name, block in self.enums.items():
            for member, value in re.findall(r"^\s+(VK_\w+) = ([^,\n]+),?$", block, re.M):
                self.enum_values.setdefault(member, (name, value.strip()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, default=Path("/usr/local/include/vulkan/vulkan_core.h"))
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "src/anbc_vulkan.h")
    args = parser.parse_args()
    sdk = Sdk(args.sdk.read_text())
    header_version = re.search(r"#define VK_HEADER_VERSION (\d+)", sdk.text).group(1)

    # Every enum type we name a value of, or that a struct field uses, is
    # emitted with just the members we use.
    wanted_enum_members: dict[str, list[tuple[str, str]]] = {}
    for member in ENUM_VALUES:
        if member not in sdk.enum_values:
            raise SystemExit(f"{member}: not found in {args.sdk}")
        enum, value = sdk.enum_values[member]
        wanted_enum_members.setdefault(enum, []).append((member, value))

    # Resolve every type the structs / functions mention, in dependency order.
    emitted: list[str] = []
    seen: set[str] = set()
    # VkAllocationCallbacks is only ever passed as NULL: kept opaque so its
    # function-pointer members need not come along.
    base_types = {"VkFlags", "VkFlags64", "VkBool32", "VkDeviceSize", "VkSampleMask", "VkDeviceAddress",
                  "VkAllocationCallbacks"}

    def emit_type(name: str) -> None:
        if name in seen or name in base_types:
            return
        seen.add(name)
        if name in sdk.handles:
            emitted.append(f"VK_DEFINE_HANDLE({name})")
        elif name in sdk.nd_handles:
            emitted.append(f"VK_DEFINE_NON_DISPATCHABLE_HANDLE({name})")
        elif name in sdk.flag_typedefs:
            emitted.append(f"typedef {sdk.flag_typedefs[name]} {name};")
        elif name in sdk.enums:
            members = wanted_enum_members.get(name, [])
            body = "".join(f"    {m} = {v},\n" for m, v in members)
            emitted.append(f"typedef enum {name} {{\n{body}    {name.upper()}_MAX_ENUM = 0x7FFFFFFF\n}} {name};"
                           .replace(name.upper(), re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()))
        elif name in sdk.structs or name in sdk.unions:
            block = sdk.structs.get(name) or sdk.unions[name]
            for dep in IDENT_RE.findall(block):
                if dep != name:
                    emit_type(dep)
            emitted.append(block)
        else:
            raise SystemExit(f"{name}: unknown type")

    for name in list(wanted_enum_members):
        emit_type(name)
    for name in STRUCTS:
        emit_type(name)
    pfn_blocks = []
    for fn in FUNCTIONS:
        if fn not in sdk.pfns:
            raise SystemExit(f"{fn}: no PFN typedef found")
        block = sdk.pfns[fn]
        for dep in IDENT_RE.findall(block):
            emit_type(dep)
        pfn_blocks.append(block)

    fields = "\n".join(f"    PFN_{fn} {fn[2:]};" for fn in FUNCTIONS)
    load_global = "".join(f'    vk->{fn[2:]} = (PFN_{fn})vk->GetInstanceProcAddr(NULL, "{fn}");\n'
                          for fn in FUNCTIONS if fn in GLOBAL_FUNCTIONS)
    load_instance = "".join(f'    vk->{fn[2:]} = (PFN_{fn})vk->GetInstanceProcAddr(instance, "{fn}");\n'
                            for fn in FUNCTIONS if fn in INSTANCE_FUNCTIONS)
    load_device = "".join(f'    vk->{fn[2:]} = (PFN_{fn})vk->GetDeviceProcAddr(device, "{fn}");\n'
                          for fn in FUNCTIONS
                          if fn not in GLOBAL_FUNCTIONS and fn not in INSTANCE_FUNCTIONS and fn != "vkGetInstanceProcAddr")

    out = f'''/**
 * @ Author: Amélie Heinrich
 * @ Copyright: Copyright (c) 2026 Amélie Heinrich. All rights reserved.
 *
 * The subset of the Vulkan API the anbc Vulkan backend uses, extracted from
 * the Khronos vulkan_core.h (header version {header_version}, Apache-2.0 /
 * MIT, Copyright 2015-2025 The Khronos Group Inc.) by tools/gen_vulkan_header.py
 * on {date.today().isoformat()} -- regenerate rather than edit. Types and enum
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
#if defined(__LP64__) || defined(_WIN64) || (defined(__x86_64__) && !defined(__ILP32__)) || defined(_M_X64) || \\
    defined(__ia64) || defined(_M_IA64) || defined(__aarch64__) || defined(__powerpc64__)
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef struct object##_T* object;
#define VK_NULL_HANDLE NULL
#else
#define VK_DEFINE_NON_DISPATCHABLE_HANDLE(object) typedef uint64_t object;
#define VK_NULL_HANDLE 0ULL
#endif

#define VK_MAKE_API_VERSION(variant, major, minor, patch) \\
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

{chr(10).join(emitted)}

{chr(10).join(pfn_blocks)}

/* ------------------------------------------------------------------------- */
/* Loader                                                                    */
/* ------------------------------------------------------------------------- */

typedef struct anbcVk {{
    void* library;
{fields}
}} anbcVk;

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void* anbcVkOpen(const char* name) {{ return (void*)LoadLibraryA(name); }}
static void* anbcVkSym(void* lib, const char* name) {{ return (void*)GetProcAddress((HMODULE)lib, name); }}
static void  anbcVkClose(void* lib) {{ FreeLibrary((HMODULE)lib); }}
#else
#include <dlfcn.h>
static void* anbcVkOpen(const char* name) {{ return dlopen(name, RTLD_NOW | RTLD_LOCAL); }}
static void* anbcVkSym(void* lib, const char* name) {{ return dlsym(lib, name); }}
static void  anbcVkClose(void* lib) {{ dlclose(lib); }}
#endif

/* Opens the platform's Vulkan loader (or MoltenVK directly on macOS when
 * the SDK loader is absent) and resolves the global entry points. */
static int anbcVkLoad(anbcVk* vk)
{{
    static const char* const kNames[] = {{
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib",
        "/usr/local/lib/libvulkan.1.dylib", "/usr/local/lib/libMoltenVK.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
    }};
    for (size_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]) && !vk->library; i++)
        vk->library = anbcVkOpen(kNames[i]);
    if (!vk->library)
        return 0;
    vk->GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)anbcVkSym(vk->library, "vkGetInstanceProcAddr");
    if (!vk->GetInstanceProcAddr) {{
        anbcVkClose(vk->library);
        vk->library = NULL;
        return 0;
    }}
{load_global}    return 1;
}}

static void anbcVkLoadInstance(anbcVk* vk, VkInstance instance)
{{
{load_instance}}}

static void anbcVkLoadDevice(anbcVk* vk, VkDevice device)
{{
{load_device}}}

static void anbcVkUnload(anbcVk* vk)
{{
    if (vk->library)
        anbcVkClose(vk->library);
    vk->library = NULL;
}}

#endif /* ANBC_VULKAN_H */
'''
    args.output.write_text(out)
    print(f"wrote {args.output} ({len(emitted)} types, {len(pfn_blocks)} entry points)")


if __name__ == "__main__":
    main()
