"""Download a curated set of ambientCG (CC0) PBR materials that ship a real
Opacity map, and merge each asset's Color.png + Opacity.png into one RGBA
PNG for training the mode-5 (alpha-aware) model.

ambientCG textures with alpha are the leaf/fence/grate/foliage categories
(cutout-style transparency), which is a much closer match to real game
texture alpha usage than a photographic alpha-matting dataset.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# Candidate asset IDs pulled from the ambientCG API for q=leaf/leaves/fence/grate,
# restricted to categories that plausibly ship a real (non-flat) Opacity map.
CANDIDATE_ASSET_IDS = [
    "Leaf001", "Leaf002", "Leaf003",
    "ScatteredLeaves001", "ScatteredLeaves002", "ScatteredLeaves003",
    "ScatteredLeaves004", "ScatteredLeaves005", "ScatteredLeaves006",
    "ScatteredLeaves007", "ScatteredLeaves008", "ScatteredLeaves009",
    "Fence001", "Fence002", "Fence003", "Fence004", "Fence005", "Fence006",
    "Fence007A", "Fence007B", "Fence007C",
    "Fence008A", "Fence008B", "Fence008C",
    "Grate001", "Grate002",
]

RESOLUTION = "1K"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ambientcg_alpha"


def download_asset_rgba(asset_id: str) -> np.ndarray | None:
    zip_name = f"{asset_id}_{RESOLUTION}-PNG.zip"
    url = f"https://ambientcg.com/get?file={zip_name}"
    print(f"Fetching {asset_id}...")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"  FAILED to download: {e}")
        return None

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        print("  not a valid zip, skipping")
        return None

    color_name = next((n for n in zf.namelist() if n.endswith("_Color.png")), None)
    opacity_name = next((n for n in zf.namelist() if n.endswith("_Opacity.png")), None)

    if color_name is None or opacity_name is None:
        print(f"  no Opacity map for {asset_id}, skipping")
        return None

    color = np.array(Image.open(io.BytesIO(zf.read(color_name))).convert("RGB"), dtype=np.uint8)
    opacity = np.array(Image.open(io.BytesIO(zf.read(opacity_name))).convert("L"), dtype=np.uint8)

    if opacity.std() < 5:
        print(f"  opacity map is essentially flat (std={opacity.std():.2f}), skipping")
        return None

    rgba = np.dstack([color, opacity])
    return rgba


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept, skipped = 0, 0

    for asset_id in CANDIDATE_ASSET_IDS:
        out_path = OUT_DIR / f"{asset_id}.png"
        if out_path.exists():
            kept += 1
            continue

        rgba = download_asset_rgba(asset_id)
        if rgba is None:
            skipped += 1
            continue

        Image.fromarray(rgba, mode="RGBA").save(out_path)
        print(f"  saved {out_path} ({rgba.shape[1]}x{rgba.shape[0]}, alpha std={rgba[...,3].std():.1f})")
        kept += 1

    print(f"\nDone. kept={kept} skipped={skipped}. Output dir: {OUT_DIR}")
