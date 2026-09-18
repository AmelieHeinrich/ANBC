"""Batch benchmark: run the neural BC7 mode-6 model against ISPC over every
texture in a folder, and report averaged speed/quality numbers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from compare_ui import (
    compute_flip,
    compute_psnr,
    ispc_encode_decode,
    load_model,
    neural_encode_decode,
    warmup_model,
)
from data import load_rgba

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}


def benchmark_folder(folder: Path, model_paths: list[Path], device_str: str) -> None:
    device = torch.device(device_str)
    models = {}
    for p in model_paths:
        model, mode, arch = load_model(p, device)
        models[mode] = (model, arch)
    warmup_model(models, device)
    print(f"Neural modes loaded: {sorted(models.keys())}")

    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images found in {folder}")
        return

    rows = []
    for f in files:
        rgba = load_rgba(f)

        ispc_img, ispc_time = ispc_encode_decode(rgba)
        neural_img, neural_time = neural_encode_decode(models, rgba, device)

        h_crop, w_crop = ispc_img.shape[:2]
        original_crop = rgba[:h_crop, :w_crop]

        ispc_psnr = compute_psnr(ispc_img[..., :3].astype(np.float64), original_crop[..., :3].astype(np.float64))
        neural_psnr = compute_psnr(
            neural_img[..., :3].astype(np.float64), original_crop[..., :3].astype(np.float64)
        )

        orig01 = original_crop[..., :3].astype(np.float32) / 255.0
        ispc_flip = compute_flip(orig01, ispc_img[..., :3].astype(np.float32) / 255.0)
        neural_flip = compute_flip(orig01, neural_img[..., :3].astype(np.float32) / 255.0)

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
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = parser.parse_args()

    model_paths = args.model or [Path("checkpoints/bc7_mode6_mlp.pt")]
    benchmark_folder(args.folder, model_paths, args.device)
