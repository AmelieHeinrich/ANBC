# Minimal build of AMD Compressonator's CMP_Core BC7 + BC6H + BC4/BC5 codecs
# (MIT, see LICENSE).
#
# Vendored from GPUOpen-Tools/compressonator @ f4b53d79 (cmp_core/shaders,
# cmp_core/source). Only the plain C++ kernels are compiled: the upstream
# CMakeLists also pulls in BC1-5 SIMD paths built with hardcoded x86 -march
# flags, which is why we don't use it. No source patches were needed on arm64.
# The BC6H kernel's external encoders (bc6h_unity/betsy/msc) are behind
# USE_UNITY/USE_BETSY/USE_MSC and not vendored.
add_library(cmp_core STATIC
    shaders/bc7_encode_kernel.cpp
    shaders/bc6_encode_kernel.cpp
    shaders/bc4_encode_kernel.cpp
    shaders/bc5_encode_kernel.cpp)
target_include_directories(cmp_core PUBLIC source PRIVATE shaders)
target_compile_options(cmp_core PRIVATE -w)
