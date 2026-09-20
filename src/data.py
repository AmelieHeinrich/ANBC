"""DIV2K download, image loading (8-bit RGBA and Radiance .hdr) + 4x4 block
extraction."""

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
# DIV2K converted to HDR by src/convert_hdr_dataset.py (BC6H training set).
HDR_DIR = Path(__file__).resolve().parent.parent / "data" / "hdr"


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


def load_hdr(path: Path) -> np.ndarray:
    """Load a Radiance RGBE (.hdr) file as an (H, W, 3) float32 linear RGB
    array. Handles the new-style RLE scanlines oiiotool/ImageMagick write and
    flat (uncompressed) scanlines; the per-scanline loop costs ~0.3s on a
    1024^2 image, which is fine for a one-off dataset load."""
    raw = path.read_bytes()
    if not raw.startswith(b"#?"):
        raise ValueError(f"{path}: not a Radiance HDR file")
    # Header: lines until an empty one, then the resolution line.
    pos = raw.index(b"\n\n") + 2
    end = raw.index(b"\n", pos)
    res = raw[pos:end].split()
    if len(res) != 4 or res[0] != b"-Y" or res[2] != b"+X":
        raise ValueError(f"{path}: unsupported orientation {raw[pos:end]!r}")
    h, w = int(res[1]), int(res[3])
    pos = end + 1

    data = np.frombuffer(raw, dtype=np.uint8)
    rgbe = np.empty((h, w, 4), dtype=np.uint8)
    for y in range(h):
        if 8 <= w < 32768 and data[pos] == 2 and data[pos + 1] == 2 and (int(data[pos + 2]) << 8 | int(data[pos + 3])) == w:
            pos += 4
            for c in range(4):
                x = 0
                while x < w:
                    count = int(data[pos])
                    pos += 1
                    if count > 128:  # run
                        count -= 128
                        rgbe[y, x : x + count, c] = data[pos]
                        pos += 1
                    else:  # literal
                        rgbe[y, x : x + count, c] = data[pos : pos + count]
                        pos += count
                    x += count
        else:  # flat scanline (old-style RLE is not produced by any modern writer)
            rgbe[y] = data[pos : pos + w * 4].reshape(w, 4)
            pos += w * 4

    e = rgbe[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)  # 2^(e-128) / 256
    return rgbe[..., :3].astype(np.float32) * scale[..., None]


def load_image(path: Path) -> np.ndarray:
    """load_hdr for .hdr files ((H, W, 3) float32), load_rgba for everything
    else ((H, W, 4) uint8)."""
    return load_hdr(path) if path.suffix.lower() == ".hdr" else load_rgba(path)


def extract_blocks(img: np.ndarray) -> np.ndarray:
    """Slice an (H, W, C) image into non-overlapping 4x4 blocks, dropping
    any partial blocks at the right/bottom edges.
    Returns (N, 4, 4, C) array (N, block_h, block_w, channels), same dtype."""
    h, w, c = img.shape
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    cropped = img[:h_crop, :w_crop]
    blocks = (
        cropped.reshape(h_crop // 4, 4, w_crop // 4, 4, c)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 4, 4, c)
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
