"""Batch benchmark: run our encoder (neural BC7 mode6/mode5, neural BC6H
with --bc6h on a folder of .hdr files, or the network-free BC5 encoder for
normal maps with --bc5) against ISPC over every texture in a folder, and
report averaged speed/quality numbers.
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
        models = load_models(model_paths, device) if model_paths else {}
        print("BC6H mode 11: " + ("MLP init (experiment)" if "bc6h" in models else "block min/max init (no network)"))
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
            model = models["bc6h"][0] if "bc6h" in models else None
            ispc_img, ispc_time = ispc_encode_decode_bc6h(rgba)
            neural_img, neural_time = encode_decode_bc6h(rgba, model, device, refine_iters)
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
    parser.add_argument(
        "--bc6h",
        action="store_true",
        help="benchmark BC6H over a folder of .hdr files (block min/max init like the C library; "
        "--model checkpoints/bc6h_mode11_mlp.pt for the superseded MLP init)",
    )
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

    model_paths = args.model or ([] if args.bc6h else [Path("checkpoints/bc7_mode6_mlp.pt")])
    benchmark_folder(args.folder, model_paths, args.device, args.refine_iters, args.bc5, args.bc6h)
