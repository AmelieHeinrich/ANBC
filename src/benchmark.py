"""Batch benchmark: run our encoder (neural BC7 mode6/mode5, or the
network-free BC5 encoder for normal maps with --bc5) against ISPC over every
texture in a folder, and report averaged speed/quality numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from compare_ui import (
    REFINE_ITERS,
    compute_flip,
    compute_psnr,
    ispc_encode_decode,
    encode_decode_bc5,
    ispc_encode_decode_bc5,
    load_models,
    neural_encode_decode,
    normal_map_rgb01,
)
from data import load_rgba

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}


def benchmark_folder(
    folder: Path, model_paths: list[Path], device_str: str, refine_iters: int = REFINE_ITERS, is_bc5: bool = False
) -> None:
    device = torch.device(device_str)
    if is_bc5:
        models = {}
        print("BC5 (min/max + refinement, no network)")
    else:
        models = load_models(model_paths, device)
        print(f"Neural modes loaded: {sorted(models.keys())}")

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {folder}")
        return

    rows = []
    for f in files:
        rgba = load_rgba(f)

        if is_bc5:
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

    model_paths = args.model or [Path("checkpoints/bc7_mode6_mlp.pt")]
    benchmark_folder(args.folder, model_paths, args.device, args.refine_iters, args.bc5)
