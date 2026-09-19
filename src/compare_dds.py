"""Score and view the BC7 / BC5 .dds files written by c_api/build/anbc_compare.

For every <stem>_anbc.dds / <stem>_cmp.dds pair in --dds-folder whose original
<stem>.<ext> exists in --original-folder, every mip level is decoded
(texture2ddecoder, an independent decoder) and compared against a CPU
box-filtered reference chain of the original (same 2x2 filter the encoders
used; pass --srgb if the DDS were generated with --srgb). The format is read
from the DDS header: BC7 is scored on RGB (and RGBA), BC5 on the two stored
channels and viewed as a normal map with Z reconstructed.

    python src/compare_dds.py --dds-folder out/bistro --original-folder data/bistro [--flip] [--csv q.csv]
    python src/compare_dds.py --dds-folder out/bistro --original-folder data/bistro --image <stem> [--mip 2]
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

import numpy as np

import bc5_codec as bc5
import bc7_codec as bc7
from compare_ui import SyncedViewer, compute_flip, compute_psnr, normal_map_rgb8, quality_label, quality_label_bc5
from data import load_rgba

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp")
DXGI_FORMAT_BC5_UNORM = 83
DXGI_FORMAT_BC7_UNORM = 98
DDS_DX10_HEADER_SIZE = 4 + 124 + 20
# Channels that are stored / scored per format.
FORMAT_CHANNELS = {DXGI_FORMAT_BC7_UNORM: 3, DXGI_FORMAT_BC5_UNORM: 2}


def read_dds(path: Path) -> tuple[list[np.ndarray], int]:
    """Returns (decoded RGBA8 image of every mip level, DXGI format). BC5
    levels have R/G decoded, B = 0 and A = 255."""
    raw = path.read_bytes()
    if raw[:4] != b"DDS " or raw[84:88] != b"DX10":
        raise ValueError(f"{path}: not a DX10 DDS")
    height, width = struct.unpack_from("<II", raw, 12)
    mip_count = max(1, struct.unpack_from("<I", raw, 28)[0])
    dxgi_format = struct.unpack_from("<I", raw, 128)[0]
    if dxgi_format not in FORMAT_CHANNELS:
        raise ValueError(f"{path}: DXGI format {dxgi_format} is not BC7_UNORM or BC5_UNORM")

    levels = []
    offset = DDS_DX10_HEADER_SIZE
    w, h = width, height
    for _ in range(mip_count):
        size = ((w + 3) // 4) * ((h + 3) // 4) * 16
        if dxgi_format == DXGI_FORMAT_BC7_UNORM:
            levels.append(bc7.decode_bc7(raw[offset : offset + size], w, h))
        else:
            rg = bc5.decode_bc5(raw[offset : offset + size], w, h)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :2] = rg
            rgba[..., 3] = 255
            levels.append(rgba)
        offset += size
        w, h = max(1, w // 2), max(1, h // 2)
    return levels, dxgi_format


def to_display_rgb(img: np.ndarray, dxgi_format: int) -> np.ndarray:
    """RGB8 for viewing / FLIP: BC5 gets its Z reconstructed."""
    return normal_map_rgb8(img[..., :2]) if dxgi_format == DXGI_FORMAT_BC5_UNORM else img[..., :3]


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def downsample(img: np.ndarray, srgb: bool) -> np.ndarray:
    """CPU twin of the GPU mip generator: 2x2 box, edge-clamped for odd sizes,
    optionally in linear light for RGB."""
    h, w = img.shape[:2]
    dw, dh = max(1, w // 2), max(1, h // 2)
    src = img.astype(np.float64) / 255.0
    if srgb:
        src[..., :3] = _srgb_to_linear(src[..., :3])
    ys = np.minimum(np.arange(dh * 2), h - 1)
    xs = np.minimum(np.arange(dw * 2), w - 1)
    src = src[ys][:, xs]
    out = src.reshape(dh, 2, dw, 2, 4).mean(axis=(1, 3))
    if srgb:
        out[..., :3] = _linear_to_srgb(out[..., :3])
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def reference_chain(original: np.ndarray, levels: int, srgb: bool) -> list[np.ndarray]:
    chain = [original]
    for _ in range(levels - 1):
        chain.append(downsample(chain[-1], srgb))
    return chain


def psnr_rgb_rgba(a: np.ndarray, b: np.ndarray, channels: int = 3) -> tuple[float, float]:
    """(PSNR over the first `channels` channels, PSNR over those plus alpha)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    keep = list(range(channels)) + [3]
    return compute_psnr(a64[..., :channels], b64[..., :channels]), compute_psnr(a64[..., keep], b64[..., keep])


def find_original(original_folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = original_folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def score_folder(dds_folder: Path, original_folder: Path, srgb: bool, use_flip: bool, csv_path: Path | None) -> None:
    stems = sorted(p.name[: -len("_anbc.dds")] for p in dds_folder.glob("*_anbc.dds"))
    rows = []
    print(f"{'texture':48s} {'size':>10s} {'anbc mip0':>10s} {'cmp mip0':>10s} {'anbc mips':>10s} {'cmp mips':>10s}"
          + (f" {'anbc flip':>10s} {'cmp flip':>10s}" if use_flip else ""))
    for stem in stems:
        orig_path = find_original(original_folder, stem)
        cmp_path = dds_folder / f"{stem}_cmp.dds"
        if orig_path is None or not cmp_path.exists():
            print(f"{stem:48.48s}  (skipped: missing original or _cmp.dds)")
            continue
        original = load_rgba(orig_path)
        anbc_levels, fmt = read_dds(dds_folder / f"{stem}_anbc.dds")
        cmp_levels, _ = read_dds(cmp_path)
        channels = FORMAT_CHANNELS[fmt]
        refs = reference_chain(original, max(len(anbc_levels), len(cmp_levels)), srgb)

        def per_level(levels: list[np.ndarray]) -> list[tuple[float, float]]:
            return [psnr_rgb_rgba(refs[i][: lv.shape[0], : lv.shape[1]], lv, channels) for i, lv in enumerate(levels)]

        pa, pc = per_level(anbc_levels), per_level(cmp_levels)
        finite = lambda xs: [x for x in xs if np.isfinite(x)]
        row = {
            "name": stem,
            "width": original.shape[1],
            "height": original.shape[0],
            "mips": len(anbc_levels),
            "anbc_mip0_rgb": pa[0][0],
            "anbc_mip0_rgba": pa[0][1],
            "cmp_mip0_rgb": pc[0][0],
            "cmp_mip0_rgba": pc[0][1],
            "anbc_mean_rgb": float(np.mean(finite([p[0] for p in pa]))),
            "cmp_mean_rgb": float(np.mean(finite([p[0] for p in pc]))),
        }
        if use_flip:
            orig01 = to_display_rgb(original, fmt).astype(np.float32) / 255.0
            row["anbc_flip"] = compute_flip(orig01, to_display_rgb(anbc_levels[0], fmt).astype(np.float32) / 255.0)
            row["cmp_flip"] = compute_flip(orig01, to_display_rgb(cmp_levels[0], fmt).astype(np.float32) / 255.0)
        rows.append(row)
        line = (f"{stem:48.48s} {row['width']:>5d}x{row['height']:<4d} {row['anbc_mip0_rgb']:8.2f}dB {row['cmp_mip0_rgb']:8.2f}dB "
                f"{row['anbc_mean_rgb']:8.2f}dB {row['cmp_mean_rgb']:8.2f}dB")
        if use_flip:
            line += f" {row['anbc_flip']:10.4f} {row['cmp_flip']:10.4f}"
        print(line, flush=True)

    if not rows:
        print("nothing scored")
        return

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    # Averages weighted towards what matters: only textures with >= 1 block row
    # of content (the 1x1 / 16x16 placeholders all hit ~inf/identical scores).
    big = [r for r in rows if r["width"] * r["height"] >= 64 * 64]
    print("-" * 100)
    print(f"{len(rows)} textures scored ({len(big)} of at least 64x64):")
    for label, subset in (("all", rows), (">=64x64", big)):
        if not subset:
            continue
        a0 = float(np.mean([r["anbc_mip0_rgb"] for r in subset if np.isfinite(r["anbc_mip0_rgb"])]))
        c0 = float(np.mean([r["cmp_mip0_rgb"] for r in subset if np.isfinite(r["cmp_mip0_rgb"])]))
        am = float(np.mean([r["anbc_mean_rgb"] for r in subset]))
        cm = float(np.mean([r["cmp_mean_rgb"] for r in subset]))
        print(f"  [{label:>7s}] mip0 PSNR RGB  anbc {a0:6.2f}dB  cmp {c0:6.2f}dB  (gap {c0 - a0:+.2f}dB)   "
              f"mean over mips  anbc {am:6.2f}dB  cmp {cm:6.2f}dB")
        if use_flip:
            af = float(np.mean([r["anbc_flip"] for r in subset]))
            cf = float(np.mean([r["cmp_flip"] for r in subset]))
            print(f"  [{label:>7s}] mip0 FLIP      anbc {af:.4f}  cmp {cf:.4f}")
    gaps = np.array([r["cmp_mip0_rgb"] - r["anbc_mip0_rgb"] for r in big if np.isfinite(r["cmp_mip0_rgb"]) and np.isfinite(r["anbc_mip0_rgb"])])
    if gaps.size:
        for t in (1, 2, 3, 5):
            print(f"  anbc within {t} dB of Compressonator (mip 0, >=64x64): {int((gaps <= t).sum())}/{gaps.size}")
        worst = sorted(big, key=lambda r: r["cmp_mip0_rgb"] - r["anbc_mip0_rgb"], reverse=True)[:5]
        print("  largest gaps:")
        for r in worst:
            print(f"    {r['name']:48.48s} anbc {r['anbc_mip0_rgb']:6.2f}dB  cmp {r['cmp_mip0_rgb']:6.2f}dB")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}")


def view_image(dds_folder: Path, original_folder: Path, stem: str, mip: int, srgb: bool) -> None:
    orig_path = find_original(original_folder, stem)
    if orig_path is None:
        raise SystemExit(f"no original for {stem} in {original_folder}")
    anbc_levels, fmt = read_dds(dds_folder / f"{stem}_anbc.dds")
    cmp_levels, _ = read_dds(dds_folder / f"{stem}_cmp.dds")
    if mip >= len(anbc_levels):
        raise SystemExit(f"{stem} has {len(anbc_levels)} mips")
    ref = reference_chain(load_rgba(orig_path), mip + 1, srgb)[mip]
    a, c = anbc_levels[mip], cmp_levels[mip]
    if fmt == DXGI_FORMAT_BC5_UNORM:
        label = lambda title, img: quality_label_bc5(title, img[..., :2], ref[..., :2])
        name = "BC5"
    else:
        label = lambda title, img: quality_label(title, img[..., :3], ref[..., :3])
        name = "BC7"
    panels = [
        (f"Original (mip {mip}, {ref.shape[1]}x{ref.shape[0]})", to_display_rgb(ref, fmt)),
        (label(f"Compressonator {name}", c), to_display_rgb(c, fmt)),
        (label(f"anbc neural {name}", a), to_display_rgb(a, fmt)),
    ]
    SyncedViewer(panels, stem).show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dds-folder", required=True, type=Path)
    parser.add_argument("--original-folder", required=True, type=Path)
    parser.add_argument("--srgb", action="store_true", help="the DDS were generated with --srgb")
    parser.add_argument("--flip", action="store_true", help="also compute FLIP on mip 0 (slow on big folders)")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--image", default=None, help="open the synced-zoom viewer for this texture stem")
    parser.add_argument("--mip", type=int, default=0)
    args = parser.parse_args()

    if args.image:
        view_image(args.dds_folder, args.original_folder, args.image, args.mip, args.srgb)
    else:
        score_folder(args.dds_folder, args.original_folder, args.srgb, args.flip, args.csv)
