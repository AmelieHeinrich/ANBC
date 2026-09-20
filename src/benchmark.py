"""Batch benchmark: run our encoder (neural BC7 mode6/mode5, or the
network-free BC6H / BC5 encoders with --bc6h / --bc5) against ISPC over
every texture in a folder, and report averaged speed/quality numbers.
--astc game|normal|float does the same for the ASTC 4x4 encoder against the
vendored astcenc CLI (-medium).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import bc6h_codec as bc6h
from compare_ui import (
    REFINE_ITERS,
    compute_flip,
    compute_flip_hdr,
    compute_psnr,
    ispc_encode_decode,
    encode_decode_bc5,
    encode_decode_bc6h,
    ispc_encode_decode_bc5,
    ispc_encode_decode_bc6h,
    load_models,
    neural_encode_decode,
    normal_map_rgb01,
)
from data import load_image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}
HDR_EXTS = {".hdr"}


def benchmark_astc(
    folder: Path,
    variant: str,
    device_str: str,
    refine_iters: int,
    config_names: list[str] | None,
    preset: str,
    limit: int | None,
) -> None:
    """ASTC 4x4: our encoder (min/max init + refinement) vs astcenc. Scores
    GAME on RGBA, NORMAL on the two stored channels (FLIP on the
    reconstructed normal), FLOAT in the half-int domain + HDR-FLIP -- the
    same metrics as the BC7/BC5/BC6H rows."""
    import astc_codec as astc

    device = torch.device(device_str)
    configs = [c for c in astc.ALL_CONFIGS if c.name in config_names] if config_names else astc.VARIANT_CONFIGS[variant]
    if config_names:
        missing = set(config_names) - {c.name for c in configs}
        assert not missing, f"unknown configs {missing}; known: {[c.name for c in astc.ALL_CONFIGS]}"
    print(f"ASTC 4x4 {variant.upper()}: block min/max init, {refine_iters} refinement rounds, "
          f"configs {[c.name for c in configs]}; reference astcenc {preset}")
    for c in configs:
        print("  " + c.describe())

    exts = HDR_EXTS if variant == "float" else IMAGE_EXTS
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if variant == "game":  # bistro dumps: skip the normal maps
        files = [f for f in files if "_Normal" not in f.name]
    if limit:
        files = files[:limit]
    if not files:
        print(f"No images found in {folder}")
        return

    rows = []
    for f in files:
        img = load_image(f)
        ref_img, ref_time = astc.astcenc_encode_decode(variant, f, preset)
        _, our_img, chosen, w_crop, h_crop, our_time = astc.encode_image(variant, img, device, refine_iters, configs)
        ref_img = ref_img[:h_crop, :w_crop]

        if variant == "float":
            orig = img[:h_crop, :w_crop]
            ref_psnr, our_psnr = bc6h.hdr_psnr(ref_img, orig), bc6h.hdr_psnr(our_img, orig)
            ref_flip, our_flip = compute_flip_hdr(orig, ref_img), compute_flip_hdr(orig, our_img)
        else:
            channels = 2 if variant == "normal" else 4
            orig = img[:h_crop, :w_crop, :channels]
            ref_psnr = compute_psnr(ref_img.astype(np.float64), orig.astype(np.float64))
            our_psnr = compute_psnr(our_img.astype(np.float64), orig.astype(np.float64))
            to_rgb01 = normal_map_rgb01 if variant == "normal" else (lambda im: im[..., :3].astype(np.float32) / 255.0)
            ref_flip, our_flip = compute_flip(to_rgb01(orig), to_rgb01(ref_img)), compute_flip(to_rgb01(orig), to_rgb01(our_img))

        share = " ".join(f"{c.name}:{100 * np.mean(chosen == i):.0f}%" for i, c in enumerate(configs))
        rows.append({"ref_ms": ref_time * 1000, "our_ms": our_time * 1000, "ref_psnr": ref_psnr, "our_psnr": our_psnr,
                     "ref_flip": ref_flip, "our_flip": our_flip})
        print(f"{f.name:40s} {w_crop:>5d}x{h_crop:<5d} astcenc {ref_time * 1000:8.1f}ms  ours {our_time * 1000:8.1f}ms  "
              f"PSNR astcenc/ours {ref_psnr:5.2f}/{our_psnr:5.2f}dB  FLIP {ref_flip:.4f}/{our_flip:.4f}  [{share}]")

    n = len(rows)
    avg = lambda key: sum(r[key] for r in rows) / n  # noqa: E731
    print("-" * 100)
    print(f"Averaged over {n} images:")
    print(f"  astcenc {preset} encode (CLI, all cores): {avg('ref_ms'):.1f}ms")
    print(f"  Ours ({device_str}): {avg('our_ms'):.1f}ms")
    print(f"  PSNR   astcenc/ours: {avg('ref_psnr'):.2f}dB / {avg('our_psnr'):.2f}dB")
    print(f"  FLIP   astcenc/ours: {avg('ref_flip'):.4f} / {avg('our_flip'):.4f}")


def benchmark_folder(
    folder: Path,
    model_paths: list[Path],
    device_str: str,
    refine_iters: int = REFINE_ITERS,
    is_bc5: bool = False,
    is_bc6h: bool = False,
) -> None:
    device = torch.device(device_str)
    if is_bc5:
        models = {}
        print("BC5 (min/max + refinement, no network)")
    elif is_bc6h:
        models = {}
        print("BC6H mode 11 (min/max + refinement, no network)")
    else:
        models = load_models(model_paths, device)
        print(f"Neural modes loaded: {sorted(models.keys())}")

    exts = HDR_EXTS if is_bc6h else IMAGE_EXTS
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not files:
        print(f"No images found in {folder}")
        return

    rows = []
    for f in files:
        rgba = load_image(f)

        if is_bc6h:
            # HDR: PSNR in the half-int domain, HDR-FLIP on linear RGB.
            ispc_img, ispc_time = ispc_encode_decode_bc6h(rgba)
            neural_img, neural_time = encode_decode_bc6h(rgba, device, refine_iters)
            h_crop, w_crop = ispc_img.shape[:2]
            original_crop = rgba[:h_crop, :w_crop]
            ispc_psnr, neural_psnr = bc6h.hdr_psnr(ispc_img, original_crop), bc6h.hdr_psnr(neural_img, original_crop)
            ispc_flip, neural_flip = compute_flip_hdr(original_crop, ispc_img), compute_flip_hdr(original_crop, neural_img)
        elif is_bc5:
            # Score the two stored channels; FLIP on the normal map with reconstructed Z.
            ispc_img, ispc_time = ispc_encode_decode_bc5(rgba)
            neural_img, neural_time = encode_decode_bc5(rgba, device, refine_iters)
            h_crop, w_crop = ispc_img.shape[:2]
            original_crop = rgba[:h_crop, :w_crop, :2]
            to_rgb01 = normal_map_rgb01
        else:
            ispc_img, ispc_time = ispc_encode_decode(rgba)
            neural_img, neural_time = neural_encode_decode(models, rgba, device, refine_iters)
            h_crop, w_crop = ispc_img.shape[:2]
            original_crop = rgba[:h_crop, :w_crop, :3]
            ispc_img, neural_img = ispc_img[..., :3], neural_img[..., :3]
            to_rgb01 = lambda img: img.astype(np.float32) / 255.0

        if not is_bc6h:
            ispc_psnr = compute_psnr(ispc_img.astype(np.float64), original_crop.astype(np.float64))
            neural_psnr = compute_psnr(neural_img.astype(np.float64), original_crop.astype(np.float64))

            orig01 = to_rgb01(original_crop)
            ispc_flip = compute_flip(orig01, to_rgb01(ispc_img))
            neural_flip = compute_flip(orig01, to_rgb01(neural_img))

        speedup = ispc_time / neural_time if neural_time > 0 else float("nan")

        rows.append(
            {
                "name": f.name,
                "size": f"{w_crop}x{h_crop}",
                "ispc_ms": ispc_time * 1000,
                "neural_ms": neural_time * 1000,
                "speedup": speedup,
                "ispc_psnr": ispc_psnr,
                "neural_psnr": neural_psnr,
                "ispc_flip": ispc_flip,
                "neural_flip": neural_flip,
            }
        )

        print(
            f"{f.name:40s} {rows[-1]['size']:>10s}  "
            f"ISPC {rows[-1]['ispc_ms']:7.2f}ms  Neural {rows[-1]['neural_ms']:7.2f}ms  "
            f"speedup {speedup:5.2f}x  "
            f"PSNR ispc/neural {ispc_psnr:5.2f}/{neural_psnr:5.2f}dB  "
            f"FLIP ispc/neural {ispc_flip:.4f}/{neural_flip:.4f}"
        )

    n = len(rows)
    avg = lambda key: sum(r[key] for r in rows) / n
    print("-" * 100)
    print(f"Averaged over {n} images:")
    print(f"  ISPC encode:   {avg('ispc_ms'):.2f}ms")
    print(f"  Neural encode: {avg('neural_ms'):.2f}ms")
    print(f"  Speedup:       {avg('speedup'):.2f}x")
    print(f"  PSNR   ISPC/Neural: {avg('ispc_psnr'):.2f}dB / {avg('neural_psnr'):.2f}dB")
    print(f"  FLIP   ISPC/Neural: {avg('ispc_flip'):.4f} / {avg('neural_flip'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("data/sponza_albedo"))
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=[],
        help="Checkpoint path; pass twice to enable mode6/mode5 mode selection.",
    )
    parser.add_argument("--bc5", action="store_true", help="benchmark BC5 (normal maps: R/G) instead of BC7; needs no --model")
    parser.add_argument("--bc6h", action="store_true", help="benchmark BC6H over a folder of .hdr files; needs no --model")
    parser.add_argument("--astc", choices=["game", "normal", "float"], default=None,
                        help="benchmark the ASTC 4x4 encoder of this variant against astcenc (needs c_api/build); needs no --model")
    parser.add_argument("--configs", default=None,
                        help="comma-separated astc_codec config names to select from per block (default: the variant's shipped set)")
    parser.add_argument("--preset", default="-medium", help="astcenc quality preset for the reference (-fast/-medium/-thorough)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images of the folder")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument(
        "--refine-iters",
        type=int,
        default=REFINE_ITERS,
        help="Exact-index/least-squares refinement rounds on the model output before packing (0 = raw model output).",
    )
    args = parser.parse_args()

    if args.astc:
        benchmark_astc(args.folder, args.astc, args.device, args.refine_iters,
                       args.configs.split(",") if args.configs else None, args.preset, args.limit)
    else:
        model_paths = args.model or ([] if args.bc6h else [Path("checkpoints/bc7_mode6_mlp.pt")])
        benchmark_folder(args.folder, model_paths, args.device, args.refine_iters, args.bc5, args.bc6h)
