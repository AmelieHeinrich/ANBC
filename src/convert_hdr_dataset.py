"""Build the BC6H training set: DIV2K PNGs -> Radiance .hdr with oiiotool.

DIV2K is LDR, so every image is linearised (sRGB -> linear) and pushed through
a random exposure (log-uniform in [--exposure-min, --exposure-max], seeded by
the file name so re-runs are reproducible) -- the result holds values well
above 1.0, which is what the BC6H network has to learn to encode. Images are
also resized (aspect preserved) so the long side is --max-size (1024) and the
full training set fits in RAM as half-int blocks (full-resolution DIV2K would
be ~13 GB).

    data/div2k/train/*.png -> data/hdr/train/*.hdr
    data/div2k/valid/*.png -> data/hdr/valid/*.hdr

Needs oiiotool on PATH (brew install openimageio). ImageMagick would do the
same job (`magick in.png -colorspace RGB -resize 1024x1024 -evaluate Multiply E out.hdr`).

Usage: python src/convert_hdr_dataset.py [--splits train valid] [--jobs N]
           [--max-size 1024] [--exposure-min 2] [--exposure-max 32]
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from data import HDR_DIR, download_div2k, load_hdr


def exposure_for(stem: str, lo: float, hi: float) -> float:
    """Log-uniform exposure in [lo, hi], deterministic per file name."""
    u = random.Random(stem).random()
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def convert_one(src: Path, dst: Path, exposure: float, max_size: int) -> str:
    if dst.exists():
        return f"{dst.name}: exists"
    tmp = dst.with_suffix(".part.hdr")
    # --resize with one side 0 keeps the aspect ratio (--fit pads to the full
    # window with black, which would poison the block statistics).
    with Image.open(src) as img:
        w, h = img.size
    geom = f"{max_size}x0" if w >= h else f"0x{max_size}"
    cmd = [
        "oiiotool", str(src),
        "--colorconvert", "sRGB", "linear",
        "--resize", geom,
        "--mulc", f"{exposure:.6f}",
        "--ch", "R,G,B",
        "-o", str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"{src.name}: oiiotool failed: {proc.stderr.strip()}"
    tmp.rename(dst)
    img = load_hdr(dst)
    return f"{dst.name}: {img.shape[1]}x{img.shape[0]} exposure {exposure:.2f} max {img.max():.2f}"


def convert_split(split: str, jobs: int, max_size: int, lo: float, hi: float) -> None:
    src_dir = download_div2k(split)
    dst_dir = HDR_DIR / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.png"))
    print(f"=== {split}: {len(files)} images -> {dst_dir}")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(convert_one, f, dst_dir / f"{f.stem}.hdr", exposure_for(f.stem, lo, hi), max_size)
            for f in files
        ]
        for fut in futures:
            print("  " + fut.result())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "valid"], choices=["train", "valid"])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-size", type=int, default=1024, help="fit the long side to this many pixels")
    parser.add_argument("--exposure-min", type=float, default=2.0)
    parser.add_argument("--exposure-max", type=float, default=32.0)
    args = parser.parse_args()

    if shutil.which("oiiotool") is None:
        raise SystemExit("oiiotool not found on PATH (brew install openimageio)")
    for split in args.splits:
        convert_split(split, args.jobs, args.max_size, args.exposure_min, args.exposure_max)
