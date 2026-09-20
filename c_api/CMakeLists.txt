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
# Shader sources -> generated headers (compiled at runtime by MTL4Compiler)
# ---------------------------------------------------------------------------
set(ANBC_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${ANBC_GENERATED_DIR})

# anbc_embed_shader(<name> <symbol> <out_var> <input>...) concatenates the
# inputs into generated/<name>_metal.h as the string constant <symbol>.
function(anbc_embed_shader name symbol output_var)
    set(inputs ${ARGN})
    set(output ${ANBC_GENERATED_DIR}/${name}_metal.h)
    add_custom_command(
        OUTPUT ${output}
        COMMAND ${CMAKE_COMMAND}
                "-DINPUTS=${inputs}" -DOUTPUT=${output} -DSYMBOL=${symbol}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_file.cmake
        DEPENDS ${inputs} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_file.cmake
        COMMENT "Embedding ${name}"
        VERBATIM)
    set(${output_var} ${output} PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(ANBC_SOURCES
    src/anbc.c
    src/anbc_model.c
    src/backend_cpu.c)

if(APPLE)
    enable_language(OBJCXX)
    set(SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)

    # The tensor-ops kernels use MetalPerformancePrimitives, whose matmul
    # implementations live in libTensorOps.rtlib and are only linked by the
    # offline Metal toolchain -- so those kernels are precompiled to .metallibs
    # and embedded as bytes. Without the toolchain (xcodebuild
    # -downloadComponent MetalToolchain) the library builds with the scalar
    # kernels only.
    execute_process(COMMAND xcrun -sdk macosx metal --version
                    RESULT_VARIABLE ANBC_METAL_RESULT OUTPUT_QUIET ERROR_QUIET)
    set(ANBC_TENSOR_HEADERS "")
    if(ANBC_METAL_RESULT EQUAL 0)
        set(ANBC_HAVE_METAL_TOOLCHAIN ON)
        # anbc_tensor_metallib(<name> <SYMBOL> <kernel.metal> <dep.metal>...)
        # compiles one tensor kernel offline and embeds the .metallib bytes.
        function(anbc_tensor_metallib name symbol kernel)
            set(metallib ${ANBC_GENERATED_DIR}/${name}.metallib)
            set(header ${ANBC_GENERATED_DIR}/${name}_metallib.h)
            add_custom_command(
                OUTPUT ${metallib}
                COMMAND xcrun -sdk macosx metal -std=metal4.0 -O2 -I ${SHADERS}
                        -o ${metallib} ${kernel}
                DEPENDS ${kernel} ${ARGN}
                COMMENT "Compiling ${name} (offline, links MetalPerformancePrimitives)"
                VERBATIM)
            add_custom_command(
                OUTPUT ${header}
                COMMAND ${CMAKE_COMMAND}
                        -DINPUT=${metallib} -DOUTPUT=${header} -DSYMBOL=${symbol}
                        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake
                DEPENDS ${metallib} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake
                COMMENT "Embedding ${name}.metallib"
                VERBATIM)
            list(APPEND ANBC_TENSOR_HEADERS ${header})
            set(ANBC_TENSOR_HEADERS ${ANBC_TENSOR_HEADERS} PARENT_SCOPE)
        endfunction()
        anbc_tensor_metallib(anbc_bc7_tensor ANBC_BC7_TENSOR_METALLIB ${SHADERS}/anbc_bc7_tensor.metal
            ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc7_common.metal ${SHADERS}/anbc_mlp_tensor.metal)
    else()
        message(WARNING "Metal toolchain not found: building without the tensor-ops kernels "
                        "(run `xcodebuild -downloadComponent MetalToolchain` to enable them)")
        set(ANBC_HAVE_METAL_TOOLCHAIN OFF)
    endif()

    # Runtime-compiled kernels: concatenated in dependency order.
    anbc_embed_shader(anbc_bc7_scalar ANBC_BC7_SCALAR_METAL_SOURCE ANBC_BC7_SCALAR_HEADER
        ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc7_common.metal
        ${SHADERS}/anbc_mlp_scalar.metal ${SHADERS}/anbc_bc7_scalar.metal)
    anbc_embed_shader(anbc_bc6h ANBC_BC6H_METAL_SOURCE ANBC_BC6H_HEADER
        ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc6h_common.metal ${SHADERS}/anbc_bc6h_scalar.metal)
    anbc_embed_shader(anbc_bc5 ANBC_BC5_METAL_SOURCE ANBC_BC5_HEADER
        ${SHADERS}/anbc_common.metal ${SHADERS}/anbc_bc5_common.metal ${SHADERS}/anbc_bc5_scalar.metal)
    anbc_embed_shader(anbc_mips ANBC_MIPS_METAL_SOURCE ANBC_MIPS_HEADER
        ${SHADERS}/anbc_mips.metal)
    list(APPEND ANBC_SOURCES src/backend_metal.mm
         ${ANBC_BC7_SCALAR_HEADER} ${ANBC_BC6H_HEADER} ${ANBC_BC5_HEADER} ${ANBC_TENSOR_HEADERS}
         ${ANBC_MIPS_HEADER})
endif()

add_library(anbc STATIC ${ANBC_SOURCES})
target_include_directories(anbc
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE src ${ANBC_GENERATED_DIR})
target_compile_options(anbc PRIVATE -Wall -Wextra)

if(APPLE)
    set_source_files_properties(src/backend_metal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    if(ANBC_HAVE_METAL_TOOLCHAIN)
        target_compile_definitions(anbc PRIVATE ANBC_HAVE_TENSOR_METALLIB=1)
    endif()
    target_link_libraries(anbc PUBLIC "-framework Metal" "-framework Foundation")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(ANBC_BUILD_TESTS)
    add_subdirectory(third_party/compressonator)

    add_executable(anbc_compare tests/anbc_compare.cpp)
    target_include_directories(anbc_compare PRIVATE third_party/stb tests)
    target_link_libraries(anbc_compare PRIVATE anbc cmp_core m)
    target_compile_options(anbc_compare PRIVATE -Wall -Wextra)
endif()
