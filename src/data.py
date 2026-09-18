"""DIV2K download + 4x4 RGBA block extraction."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

DIV2K_BASE_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K"
DIV2K_SETS = {
    "train": "DIV2K_train_HR.zip",
    "valid": "DIV2K_valid_HR.zip",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "div2k"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        pass

    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
    tmp.rename(dest)


def download_div2k(split: str = "train") -> Path:
    """Download and extract a DIV2K split ('train' or 'valid').
    Returns the directory containing extracted PNGs."""
    zip_name = DIV2K_SETS[split]
    zip_path = DATA_DIR / zip_name
    extract_dir = DATA_DIR / split

    if extract_dir.exists() and any(extract_dir.glob("*.png")):
        return extract_dir

    _download(f"{DIV2K_BASE_URL}/{zip_name}", zip_path)

    print(f"Extracting {zip_path} -> {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".png"):
                data = zf.read(info)
                out_name = Path(info.filename).name
                (extract_dir / out_name).write_bytes(data)
    return extract_dir


def load_rgba(path: Path) -> np.ndarray:
    """Load an image file as an (H, W, 4) uint8 RGBA array."""
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def extract_blocks(rgba: np.ndarray) -> np.ndarray:
    """Slice an (H, W, 4) RGBA image into non-overlapping 4x4 blocks,
    dropping any partial blocks at the right/bottom edges.
    Returns (N, 4, 4, 4) uint8 array (N, block_h, block_w, channels)."""
    h, w = rgba.shape[:2]
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    cropped = rgba[:h_crop, :w_crop]
    blocks = (
        cropped.reshape(h_crop // 4, 4, w_crop // 4, 4, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 4, 4, 4)
    )
    return blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid", "both"], default="both")
    args = parser.parse_args()

    splits = ["train", "valid"] if args.split == "both" else [args.split]
    for split in splits:
        directory = download_div2k(split)
        files = sorted(directory.glob("*.png"))
        print(f"{split}: {len(files)} images in {directory}")
        if files:
            sample = load_rgba(files[0])
            blocks = extract_blocks(sample)
            print(f"  sample image {files[0].name}: shape={sample.shape}, blocks={blocks.shape}")
