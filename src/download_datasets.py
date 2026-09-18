"""One-shot setup: download every dataset the training scripts need.

  - DIV2K (train + valid)      -> data/div2k/          (mode6 training)
  - ambientCG alpha textures   -> data/ambientcg_alpha/ (mode5 training)

data/sponza_albedo/ is NOT fetched here -- it's a user-provided texture set
(copy it over separately if you want the benchmark script to use it).

Usage: python src/download_datasets.py
"""

from __future__ import annotations

from data import download_div2k
from download_ambientcg_alpha import CANDIDATE_ASSET_IDS, OUT_DIR, download_asset_rgba
from PIL import Image


def download_div2k_datasets() -> None:
    for split in ("train", "valid"):
        directory = download_div2k(split)
        n = len(list(directory.glob("*.png")))
        print(f"DIV2K {split}: {n} images in {directory}")


def download_ambientcg_dataset() -> None:
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
        print(f"  saved {out_path} ({rgba.shape[1]}x{rgba.shape[0]}, alpha std={rgba[..., 3].std():.1f})")
        kept += 1

    print(f"ambientCG alpha: kept={kept} skipped={skipped}. Output dir: {OUT_DIR}")


if __name__ == "__main__":
    print("=== DIV2K (mode6 training set) ===")
    download_div2k_datasets()
    print("\n=== ambientCG alpha textures (mode5 training set) ===")
    download_ambientcg_dataset()
    print("\nDone. data/sponza_albedo/ is not auto-downloaded -- copy it over yourself if needed for benchmark.py.")
