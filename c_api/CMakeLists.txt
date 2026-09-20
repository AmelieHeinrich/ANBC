cmake_minimum_required(VERSION 3.21)
project(anbc LANGUAGES C CXX)

option(ANBC_BUILD_TESTS "Build the anbc_compare test tool (needs Compressonator + stb)" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Shader blob: every artefact the backends load at runtime (Metal sources,
# the BC7 tensor metallib, SPIR-V) packed into one embedded array by
# tools/pack_blob.py. The same blob goes into the single-header build.
# ---------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(ANBC_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${ANBC_GENERATED_DIR})
set(SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
set(ANBC_BLOB_ENTRIES "")  # NAME=FILE[+FILE...] arguments for pack_blob.py
set(ANBC_BLOB_DEPS "")

# anbc_blob_entry(<name> <file>...) adds one blob entry; several files are
# concatenated in order (the runtime-compiled Metal libraries).
function(anbc_blob_entry name)
    string(JOIN "+" joined ${ARGN})
    list(APPEND ANBC_BLOB_ENTRIES "${name}=${joined}")
    list(APPEND ANBC_BLOB_DEPS ${ARGN})
    set(ANBC_BLOB_ENTRIES ${ANBC_BLOB_ENTRIES} PARENT_SCOPE)
    set(ANBC_BLOB_DEPS ${ANBC_BLOB_DEPS} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(ANBC_SOURCES
    src/anbc.c
    src/anbc_model.c
    src/anbc_blob.c
    src/backend_vulkan.c)

if(APPLE)
    enable_language(OBJCXX)

    # The BC7 tensor-ops kernel uses MetalPerformancePrimitives, whose matmul
    # implementations live in libTensorOps.rtlib and are only linked by the
    # offline Metal toolchain -- so it is precompiled to a .metallib and
    # shipped in the blob. Without the toolchain (xcodebuild -downloadComponent
    # MetalToolchain) the blob has no metallib and the backend uses the scalar
    # kernel only.
    execute_process(COMMAND xcrun -sdk macosx metal --version
                    RESULT_VARIABLE ANBC_METAL_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(ANBC_METAL_RESULT EQUAL 0)
        set(metallib ${ANBC_GENERATED_DIR}/anbc_bc7_tensor.metallib)
        add_custom_command(
            OUTPUT ${metallib}
            COMMAND xcrun -sdk macosx metal -std=metal4.0 -O2 -I ${SHADERS}
                    -o ${metallib} ${SHADERS}/anbc_bc7_tensor.metal
            DEPENDS ${SHADERS}/anbc_bc7_tensor.metal ${SHADERS}/anbc_common.metal
                    ${SHADERS}/anbc_bc7_common.metal ${SHADERS}/anbc_mlp_tensor.metal
            COMMENT "Compiling anbc_bc7_tensor (offline, links MetalPerformancePrimitives)"
            VERBATIM)
        anbc_blob_entry(bc7_tensor.metallib ${metallib})
    else()
        message(WARNING "Metal toolchain not found: building without the tensor-ops kernel "
                        "(run `xcodebuild -downloadComponent MetalToolchain` to enable it)")
    endif()

    # Runtime-compiled kernels: concatenated in dependency order.
    anbc_blob_entry(bc7_scalar.metal ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc7_common.metal
        ${SHADERS}/anbc_mlp_scalar.metal ${SHADERS}/anbc_bc7_scalar.metal)
    anbc_blob_entry(bc6h.metal ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc6h_common.metal
        ${SHADERS}/anbc_bc6h_scalar.metal)
    anbc_blob_entry(bc5.metal ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc5_common.metal
        ${SHADERS}/anbc_bc5_scalar.metal)
    anbc_blob_entry(astc_scalar.metal ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_astc_tables.metal
        ${SHADERS}/anbc_astc_common.metal ${SHADERS}/anbc_astc_scalar.metal)
    anbc_blob_entry(mips.metal ${SHADERS}/anbc_mips.metal)
    list(APPEND ANBC_SOURCES src/backend_metal.mm)
endif()

# Vulkan kernels: GLSL -> SPIR-V with glslc (Vulkan SDK / shaderc). The
# backend itself needs no SDK -- see src/anbc_vulkan.h -- only the build does.
find_program(ANBC_GLSLC glslc)
if(NOT ANBC_GLSLC)
    message(FATAL_ERROR "glslc not found: install the Vulkan SDK (or shaderc) to compile the Vulkan kernels")
endif()
set(GLSL ${SHADERS}/glsl)
set(ANBC_GLSL_COMMON ${GLSL}/anbc_common.glsl ${GLSL}/anbc_mlp_scalar.glsl ${GLSL}/anbc_bc7_common.glsl
    ${GLSL}/anbc_bc5_common.glsl ${GLSL}/anbc_bc6h_common.glsl ${GLSL}/anbc_astc_common.glsl
    ${GLSL}/anbc_astc_tables.glsl)

# anbc_spirv(<entry name> <kernel.comp> [glslc flags...]) compiles one kernel
# and adds it to the blob.
function(anbc_spirv name kernel)
    set(spv ${ANBC_GENERATED_DIR}/${name})
    add_custom_command(
        OUTPUT ${spv}
        COMMAND ${ANBC_GLSLC} -O --target-env=vulkan1.1 -fshader-stage=compute -I ${GLSL} ${ARGN}
                -o ${spv} ${kernel}
        DEPENDS ${kernel} ${ANBC_GLSL_COMMON}
        COMMENT "Compiling ${name}"
        VERBATIM)
    anbc_blob_entry(${name} ${spv})
    set(ANBC_BLOB_ENTRIES ${ANBC_BLOB_ENTRIES} PARENT_SCOPE)
    set(ANBC_BLOB_DEPS ${ANBC_BLOB_DEPS} PARENT_SCOPE)
endfunction()
anbc_spirv(bc7.spv ${GLSL}/anbc_bc7.comp)
anbc_spirv(bc5.spv ${GLSL}/anbc_bc5.comp)
anbc_spirv(bc6h.spv ${GLSL}/anbc_bc6h.comp)
anbc_spirv(astc_game.spv ${GLSL}/anbc_astc_game.comp)
anbc_spirv(astc_normal.spv ${GLSL}/anbc_astc_normal.comp)
anbc_spirv(astc_float.spv ${GLSL}/anbc_astc_float.comp)
anbc_spirv(mips_rgba8.spv ${GLSL}/anbc_mips.comp -DANBC_MIPS_FORMAT=rgba8)
anbc_spirv(mips_rgba16f.spv ${GLSL}/anbc_mips.comp -DANBC_MIPS_FORMAT=rgba16f)

set(ANBC_BLOB_HEADER ${ANBC_GENERATED_DIR}/anbc_blob_data.h)
add_custom_command(
    OUTPUT ${ANBC_BLOB_HEADER}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/pack_blob.py
            -o ${ANBC_BLOB_HEADER} ${ANBC_BLOB_ENTRIES}
    DEPENDS ${ANBC_BLOB_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/tools/pack_blob.py
    COMMENT "Packing the shader blob"
    VERBATIM)
list(APPEND ANBC_SOURCES ${ANBC_BLOB_HEADER})

add_library(anbc STATIC ${ANBC_SOURCES})
target_include_directories(anbc
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE src ${ANBC_GENERATED_DIR})
target_compile_options(anbc PRIVATE -Wall -Wextra)
# `{ VK_STRUCTURE_TYPE_X }` partial initializers are the Vulkan idiom.
set_source_files_properties(src/backend_vulkan.c PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")

if(APPLE)
    set_source_files_properties(src/backend_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_compile_definitions(anbc PRIVATE ANBC_HAVE_METAL=1)
    target_link_libraries(anbc PUBLIC "-framework Metal" "-framework Foundation")
elseif(UNIX)
    target_link_libraries(anbc PUBLIC ${CMAKE_DL_LIBS}) # dlopen for the Vulkan loader
endif()

# ---------------------------------------------------------------------------
# Single-header distribution: `cmake --build build --target anbc_amalgamate`
# regenerates dist/anbc.h (committed) from the sources and the blob.
# ---------------------------------------------------------------------------
add_custom_target(anbc_amalgamate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py
            -o ${CMAKE_CURRENT_SOURCE_DIR}/dist/anbc.h --blob ${ANBC_BLOB_HEADER}
    DEPENDS ${ANBC_BLOB_HEADER} ${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py
    COMMENT "Writing dist/anbc.h"
    VERBATIM)

# Dev check: src/anbc_vulkan.h (generated by tools/gen_vulkan_header.py from
# the SDK's vulkan_core.h) must stay ABI-identical to it. Only offered when
# an SDK header is around.
find_path(ANBC_VULKAN_SDK_INCLUDE vulkan/vulkan_core.h HINTS $ENV{VULKAN_SDK}/include /usr/local/include /usr/include)
if(ANBC_VULKAN_SDK_INCLUDE)
    add_custom_target(anbc_check_vulkan_abi
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_vulkan_abi.py
                --sdk-include ${ANBC_VULKAN_SDK_INCLUDE} --cc ${CMAKE_C_COMPILER}
        COMMENT "Checking anbc_vulkan.h against the SDK header"
        VERBATIM)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ANBC_BUILD_TESTS)
    add_subdirectory(third_party/compressonator)

    # ARM astc-encoder: the ASTC reference (encoder baseline + decoder for
    # PSNR) in anbc_compare, and the astcenc-neon CLI that src/astc_codec.py
    # validates the Python packer against. See third_party/astcenc/README.md.
    set(ASTCENC_UNIVERSAL_BUILD OFF CACHE BOOL "" FORCE)
    set(ASTCENC_ISA_NEON ON CACHE BOOL "" FORCE)
    set(ASTCENC_CLI ON CACHE BOOL "" FORCE)
    set(ASTCENC_WERROR OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/astcenc EXCLUDE_FROM_ALL)
    set_target_properties(astcenc-neon PROPERTIES EXCLUDE_FROM_ALL FALSE)

    if(APPLE)
        # Builds tests/single_header_test.mm against dist/anbc.h alone.
        add_executable(anbc_single_header_test tests/single_header_test.mm)
        target_include_directories(anbc_single_header_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dist)
        target_compile_options(anbc_single_header_test PRIVATE -fobjc-arc -Wall -Wextra -Wno-missing-field-initializers)
        target_link_libraries(anbc_single_header_test PRIVATE "-framework Metal" "-framework Foundation")
        add_dependencies(anbc_single_header_test anbc_amalgamate)
    endif()

    add_executable(anbc_compare tests/anbc_compare.cpp)
    target_include_directories(anbc_compare PRIVATE third_party/stb tests third_party/astcenc/Source)
    target_link_libraries(anbc_compare PRIVATE anbc cmp_core astcenc-neon-static m)
    target_compile_options(anbc_compare PRIVATE -Wall -Wextra)
endif()
