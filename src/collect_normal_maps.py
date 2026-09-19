"""Gather normal-map textures from the scene dumps into data/normal for BC5
training.

data/normal starts out with the sponza normals; this copies every file whose
name contains "normal" (case-insensitive) from data/bistro and
data/intel_sponza next to them. Files are copied (not moved) and kept under
their original names, so re-running is a no-op.

Bistro ships 1x1 placeholder "normal maps" for materials without one, plus a
few full-size but perfectly flat ones; both are skipped (no 4x4 blocks /
nothing to learn, and a constant image makes PSNR infinite in the benchmark).

Usage: python src/collect_normal_maps.py [--sources DIR ...] [--dest DIR]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = [ROOT / "data" / "bistro", ROOT / "data" / "intel_sponza"]
DEFAULT_DEST = ROOT / "data" / "normal"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}


def is_useful_normal_map(path: Path) -> bool:
    """False for images smaller than one 4x4 block or with constant RG."""
    img = Image.open(path)
    if min(img.size) < 4:
        return False
    rg = np.asarray(img.convert("RGB"))[..., :2]
    return bool((rg != rg[0, 0]).any())


def collect(sources: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    copied, skipped, useless = 0, 0, 0
    seen: dict[str, Path] = {}
    for src_dir in sources:
        if not src_dir.is_dir():
            print(f"warning: {src_dir} does not exist, skipping")
            continue
        files = sorted(
            p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and "normal" in p.stem.lower()
        )
        for f in files:
            if f.name in seen:
                print(f"warning: {f} collides with {seen[f.name]}, skipping")
                skipped += 1
                continue
            seen[f.name] = f
            out = dest / f.name
            if not is_useful_normal_map(f):
                useless += 1
                if out.exists():  # copied by an earlier version of this script
                    out.unlink()
                continue
            if out.exists():
                skipped += 1
                continue
            shutil.copy2(f, out)
            copied += 1
        print(f"{src_dir}: {len(files)} normal maps")

    total = sum(1 for p in dest.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(
        f"copied {copied}, skipped {skipped} (already present / collisions), "
        f"ignored {useless} placeholder/flat images; {dest} now holds {total} images"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    collect(args.sources, args.dest)
