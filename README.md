# NeuralBlockCompressor

GPU texture block compression (BC7, BC5, BC6H, ASTC 4x4) with full mip chains, shipped as a single-header C library. BC7 endpoints are predicted by small per-block MLPs; every other format is encoded analytically. Metal 4 on macOS, Vulkan on Windows and Linux.

The repo has two halves:

- [c_api/](c_api/) — the library (`anbc`), its test/compare tool, and the build that packs shaders and networks into one embedded blob.
- [src/](src/) — the Python side: training the BC7 networks, reference codecs for every format (bit-exact against the reference decoders), benchmarks and a compare UI.

## AI notice

This thing was entirely vibed

## How it works

Every format runs the same pipeline on the GPU, one dispatch per mip:

1. Pick initial endpoints for each 4x4 block.
   - **BC7** (mode 6 for RGB, mode 5 for RGBA, chosen per block by squared error): a 64 → 128 × 3 → 24 MLP predicts the endpoints from the raw block ([src/model.py](src/model.py)).
   - **BC5, BC6H (mode 11), ASTC 4x4**: the block's per-channel min/max. A network was trained for each of these too and none beat the bounding box after refinement, so they were removed (details in [notes.txt](notes.txt)).
2. Run `refineIterations` rounds (default 2) of exact index search + least-squares endpoint refit.
3. Pack the block.

The refinement, not the initialiser, does most of the work. Mips are box-filtered on the GPU (in linear light with `ANBC_TEXTURE_FLAG_SRGB`).

### Formats

| `anbcTextureFormat` | Input | Notes |
|---|---|---|
| `BC7` | RGBA8 | Modes 6 + 5 only. Embedded networks; `anbcLoadModel` swaps in an experimental one. |
| `BC5` | RGBA8 (R, G used) | Normal maps. Two independent BC4 lines. |
| `BC6H` | RGBA16F | `BC6H_UF16`, mode 11. Negative inputs clamp to 0. |
| `ASTC_4x4_UNORM` | RGBA8 | Single-partition configs picked per block. `ANBC_TEXTURE_FLAG_NORMAL_MAP` stores X,Y as luminance + alpha (sample `.ra`, astcenc's `-normal` layout). |
| `ASTC_4x4_FLOAT` | RGBA16F | HDR RGB. Needs an HDR-capable ASTC decoder (Apple A13/M1+). |

Max texture size is 4096² with mips, 16384² without.

### Backends

- **Metal 4** (macOS 26+, Metal 4 GPU) — what ships on Apple platforms. Kernels are compiled from source at runtime. On M5-class GPUs (`MTLGPUFamilyApple10`) the BC7 MLPs additionally run as `matmul2d` on the tensor accelerators via MetalPerformancePrimitives (~8× faster, identical output); that kernel is precompiled to a `.metallib` at build time and needs the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), otherwise CMake falls back to the scalar kernel.
- **Vulkan** (Windows, Linux) — plain Vulkan 1.1 compute, GLSL ports of the same kernels compiled with `glslc` at build time, scalar MLP only. No Vulkan SDK is needed to *compile* the library: [anbc_vulkan.h](c_api/src/anbc_vulkan.h) is the subset of `vulkan_core.h` in use plus a `dlopen`/`LoadLibrary` loader. On macOS it exists for testing through MoltenVK.

## Using the library

The single header [c_api/dist/anbc.h](c_api/dist/anbc.h) is committed and contains the public API plus, under `ANBC_IMPLEMENTATION`, every source file and the embedded shader/network blob. Define `ANBC_IMPLEMENTATION` in exactly one translation unit:

| Platform | Implementation file | Links |
|---|---|---|
| macOS | `anbc_impl.mm` (Objective-C++) | Metal, Foundation |
| Windows / Linux | `anbc_impl.c` (C11) | `-ldl` on old glibc |

```c
#define ANBC_IMPLEMENTATION
#include "anbc.h"

anbcDevice* dev = anbcCreateDevice(ANBC_DEVICE_BACKEND_METAL); /* or _VULKAN */

anbcTextureDesc desc = {
    .width = w, .height = h,
    .pixels = rgba8,                        /* top-left origin, rowPitch 0 = tightly packed */
    .pixelFormat = ANBC_PIXEL_FORMAT_RGBA8_UNORM,
    .flags = ANBC_TEXTURE_FLAG_SRGB | ANBC_TEXTURE_FLAG_GENERATE_MIPS,
};
anbcTexture* tex = anbcCreateTexture(dev, &desc);

anbcCompress(dev, tex, ANBC_TEXTURE_FORMAT_BC7, NULL); /* blocking; NULL = 2 refinement rounds */

for (uint32_t i = 0; i < anbcGetMipCount(tex); ++i) {
    anbcMipInfo mip;
    anbcGetMip(tex, i, &mip);              /* mip.data: blocksX * blocksY * 16 bytes, owned by tex */
}

anbcDestroyTexture(tex);
anbcDestroyDevice(dev);
```

`anbcGetDeviceInfo` reports the GPU, backend and whether tensor ops are active; `ANBC_COMPRESS_FLAG_NO_TENSOR_OPS` forces the scalar kernel per call. `anbcCreateDevice` returns `NULL` if the backend is unavailable (not compiled in, no capable GPU, no Vulkan loader). See [c_api/include/anbc.h](c_api/include/anbc.h) for the full API.

## Building the library

Requirements: CMake; on macOS, macOS 26+ and a Metal 4 GPU (plus the Metal toolchain for the tensor-ops kernel); `glslc` from the Vulkan SDK on `PATH` for the Vulkan kernels.

```sh
cmake -S c_api -B c_api/build && cmake --build c_api/build -j
```

This also builds `anbc_compare` and regenerates the blob. After touching sources, shaders or the networks, regenerate the single header:

```sh
cmake --build c_api/build --target anbc_amalgamate
```

The reference encoders in the test tool (Compressonator and ARM astcenc) are vendored in [c_api/third_party/](c_api/third_party/). astcenc is configured for NEON in [c_api/CMakeLists.txt](c_api/CMakeLists.txt); an x86 host needs the SSE variant there. The library itself has no such dependency.

### `anbc_compare`

Encodes with the library, encodes with the reference (Compressonator for BCn, astcenc `-medium` for ASTC), writes `.dds` (and `.astc`) files with full mip chains, and prints PSNR / timings. A folder run also writes `summary.csv`.

```sh
c_api/build/anbc_compare data/bistro/Antenna_Metal_BaseColor.png --out /tmp                 # BC7
c_api/build/anbc_compare data/normal/2299742237651021498.jpg --format bc5 --out /tmp
c_api/build/anbc_compare data/hdr/valid/0801.hdr --format bc6h --out /tmp
c_api/build/anbc_compare data/normal/2299742237651021498.jpg --format astc --normal-map --out /tmp
c_api/build/anbc_compare data/hdr/valid/0801.hdr --format astc-float --out /tmp
c_api/build/anbc_compare data/bistro --format astc --out out/astc_bistro                     # folder
```

Useful flags: `--format bc7|bc6h|bc5|astc|astc-float`, `--backend metal|vulkan`, `--xcheck` (run both backends and compare blocks), `--srgb`, `--no-mips`, `--refine-iters N`, `--no-tensor-ops`, `--models DIR`, `--cmp-quality Q`, `--astc-preset fast|medium|thorough`, `--threads N`.

### Results (M5, with mips)

| Format | Dataset | Ours | Reference | Speed-up |
|---|---|---|---|---|
| BC5 | 1024² normal map | ~1 ms, 43.7 dB | Compressonator (10 threads) ~3.5 ms, 42.8 dB | ~3.5× |
| BC6H | 8192×4096 HDRI (no mips) | ~4.3 ms | Compressonator 8.5 s | ~2000×, ~1 dB lower |
| ASTC 4x4 | bistro, 537 textures | 1.1 s, 55.0 dB | astcenc 101 s, 60.9 dB | 89× |
| ASTC 4x4 normal | 152 normal maps | 0.42 s, 44.4 dB | astcenc 205 s, 48.1 dB | 490× |
| ASTC 4x4 HDR | 100 images | 0.14 s, 44.1 dB | astcenc 15.7 s, 48.9 dB | 111× |

The quality gap to the references is the usual one for a single-subset / single-partition encoder: no partitions, no decimated weight grids, and only one or two block modes per format.

## Python side

```sh
./setup.sh                      # venv, deps, DIV2K + ambientCG downloads (+ HDR conversion if oiiotool is installed)
source .venv/bin/activate
```

Run everything from the repo root. `oiiotool` (`brew install openimageio`) is needed for the HDR dataset; the ASTC self-tests need `c_api/build` for the astcenc CLI.

**Train the BC7 networks** (checkpoints go to `checkpoints/bc7_<mode>_mlp.pt`):

```sh
python src/train_bc7.py --mode mode6 --epochs 15 --batch-size 32768
python src/train_bc7.py --mode mode5 --epochs 60 --val-images 4 --batch-size 4096
```

**Export** them to the `.bin` format the library embeds, then rebuild so the blob picks them up:

```sh
python src/export_weights.py    # -> c_api/models/bc7_mode6.bin, bc7_mode5.bin
```

**Compare UI** (original vs reference vs ours, synced zoom/pan):

```sh
python src/compare_ui.py --image data/ambientcg_alpha/Fence001.png \
    --model checkpoints/bc7_mode6_mlp.pt --model checkpoints/bc7_mode5_mlp.pt
python src/compare_ui.py --image data/normal/2299742237651021498.jpg --bc5
python src/compare_ui.py --image data/hdr/valid/0801.hdr --bc6h
python src/compare_ui.py --image data/hdr/valid/0801.hdr --astc float   # or: game, normal
```

**Benchmark** a folder (PSNR / FLIP vs ISPC or astcenc):

```sh
python src/benchmark.py --folder data/div2k/train --model checkpoints/bc7_mode6_mlp.pt
python src/benchmark.py --folder data/normal --bc5
python src/benchmark.py --folder data/hdr/valid --bc6h
python src/benchmark.py --folder data/bistro --astc game
```

**Score the DDS output** of an `anbc_compare` folder run against the originals:

```sh
python src/compare_dds.py --dds-folder out/bistro --original-folder data/bistro --flip --csv out/bistro/quality.csv
```

Each Python codec ([bc5_codec.py](src/bc5_codec.py), [bc6h_codec.py](src/bc6h_codec.py), [astc_codec.py](src/astc_codec.py)) has a `--image` self-test that round-trips encode → pack → reference decode and must match bit-exactly.

[notes.txt](notes.txt) is the full command cheat sheet and records the experiments behind each design decision.

## Layout

```
c_api/
  include/anbc.h       public API
  dist/anbc.h          committed single header (API + implementation + blob)
  src/                 anbc.c, backend_metal.mm, backend_vulkan.c, shaders/
  models/              exported BC7 networks (embedded into the blob)
  tests/               anbc_compare, single-header build test
  tools/               pack_blob.py, amalgamate.py, gen_vulkan_header.py, check_vulkan_abi.py
  third_party/         astcenc, compressonator, stb
src/                   training, codecs, benchmark, compare UI
checkpoints/           .pt checkpoints (gitignored)
data/                  datasets (gitignored)
```
