# astc-encoder (vendored)

ARM's reference ASTC codec, vendored from
https://github.com/ARM-software/astc-encoder tag `5.7.0`
(commit baff485b0ff36d2f95d28961605106502c653966), Apache-2.0 (LICENSE.txt).

Only `Source/` (core library + CLI + ThirdParty image loaders) and the
top-level CMakeLists are kept; Test/, Utils/, Docs/, the fuzzers, GoogleTest
and the unit tests are not. No source patches. Built by `c_api/CMakeLists.txt`
with `ASTCENC_UNIVERSAL_BUILD=OFF ASTCENC_ISA_NEON=ON ASTCENC_CLI=ON
ASTCENC_WERROR=OFF`, giving `astcenc-neon-static` (linked into
`anbc_compare` as the ASTC reference encoder/decoder) and the `astcenc-neon`
CLI (used by `src/astc_codec.py` to validate the Python packer, in particular
for HDR which texture2ddecoder cannot decode).
