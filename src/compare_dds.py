"""Score and view the BC7 / BC5 / BC6H / ASTC .dds (and .astc) files written
by c_api/build/anbc_compare.

For every <stem>_anbc.dds / <stem>_cmp.dds pair in --dds-folder whose original
<stem>.<ext> exists in --original-folder, every mip level is decoded
(texture2ddecoder, an independent decoder; our own all-mode decoder for BC6H;
the astcenc CLI for HDR ASTC) and compared against a CPU box-filtered
reference chain of the original (same 2x2 filter the encoders used; pass
--srgb if the DDS were generated with --srgb). The format is read from the
DDS header: BC7 and ASTC are scored on RGB (and RGBA), BC5 -- and ASTC with
--normal-map, whose X,Y live in the decoded R and A -- on the two stored
channels and viewed as a normal map with Z reconstructed, BC6H and HDR ASTC
(originals are .hdr; the latter comes as <stem>_anbc.astc + _mipN.astc
since DDS has no HDR ASTC id) in the half-int domain and viewed tone-mapped.

    python src/compare_dds.py --dds-folder out/bistro --original-folder data/bistro [--flip] [--csv q.csv]
    python src/compare_dds.py --dds-folder out/bistro --original-folder data/bistro --image <stem> [--mip 2]
    python src/compare_dds.py --dds-folder out/normal --original-folder data/normal --normal-map
"""

from __future__ import annotations

import argparse
import csv
import struct
from pathlib import Path

import numpy as np

import astc_codec as astc
import bc5_codec as bc5
import bc6h_codec as bc6h
import bc7_codec as bc7
from compare_ui import (
    SyncedViewer,
    compute_flip,
    compute_flip_hdr,
    compute_psnr,
    normal_map_rgb8,
    quality_label,
    quality_label_bc5,
    quality_label_bc6h,
)
from data import load_image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".hdr")
DXGI_FORMAT_BC5_UNORM = 83
DXGI_FORMAT_BC6H_UF16 = 95
DXGI_FORMAT_BC7_UNORM = 98
DXGI_FORMAT_ASTC_4X4_UNORM = 134
FORMAT_ASTC_4X4_NORMAL = -134  # pseudo format: ASTC_4X4_UNORM read with --normal-map (X,Y in R,A -> RG)
FORMAT_ASTC_4X4_HDR = -135     # pseudo format: the .astc chains of astc-float
DDS_DX10_HEADER_SIZE = 4 + 124 + 20
# Channels that are stored / scored per format.
FORMAT_CHANNELS = {DXGI_FORMAT_BC7_UNORM: 3, DXGI_FORMAT_BC5_UNORM: 2, DXGI_FORMAT_BC6H_UF16: 3,
                   DXGI_FORMAT_ASTC_4X4_UNORM: 3, FORMAT_ASTC_4X4_NORMAL: 2, FORMAT_ASTC_4X4_HDR: 3}
FORMAT_NAMES = {DXGI_FORMAT_BC7_UNORM: "BC7", DXGI_FORMAT_BC5_UNORM: "BC5", DXGI_FORMAT_BC6H_UF16: "BC6H",
                DXGI_FORMAT_ASTC_4X4_UNORM: "ASTC 4x4", FORMAT_ASTC_4X4_NORMAL: "ASTC 4x4 normal map",
                FORMAT_ASTC_4X4_HDR: "ASTC 4x4 HDR"}


def read_dds(path: Path, normal_map: bool = False) -> tuple[list[np.ndarray], int]:
    """Returns (decoded image of every mip level, DXGI format): RGBA8 for
    BC7/BC5/ASTC (BC5 and normal-map ASTC levels have R/G decoded, B = 0
    and A = 255), (H, W, 3) float32 linear RGB for BC6H."""
    raw = path.read_bytes()
    if raw[:4] != b"DDS " or raw[84:88] != b"DX10":
        raise ValueError(f"{path}: not a DX10 DDS")
    height, width = struct.unpack_from("<II", raw, 12)
    mip_count = max(1, struct.unpack_from("<I", raw, 28)[0])
    dxgi_format = struct.unpack_from("<I", raw, 128)[0]
    if dxgi_format not in FORMAT_CHANNELS:
        raise ValueError(f"{path}: DXGI format {dxgi_format} is not BC7_UNORM, BC5_UNORM, BC6H_UF16 or ASTC_4X4_UNORM")
    if normal_map:
        if dxgi_format != DXGI_FORMAT_ASTC_4X4_UNORM:
            raise ValueError(f"{path}: --normal-map only applies to ASTC_4X4_UNORM")
        dxgi_format = FORMAT_ASTC_4X4_NORMAL

    levels = []
    offset = DDS_DX10_HEADER_SIZE
    w, h = width, height
    for _ in range(mip_count):
        size = ((w + 3) // 4) * ((h + 3) // 4) * 16
        if dxgi_format == DXGI_FORMAT_BC7_UNORM:
            levels.append(bc7.decode_bc7(raw[offset : offset + size], w, h))
        elif dxgi_format == DXGI_FORMAT_BC6H_UF16:
            levels.append(bc6h.decode_bc6h(raw[offset : offset + size], w, h))
        elif dxgi_format == DXGI_FORMAT_ASTC_4X4_UNORM:
            levels.append(astc.decode_astc_t2d(raw[offset : offset + size], w, h))
        elif dxgi_format == FORMAT_ASTC_4X4_NORMAL:
            rgba = astc.decode_astc_t2d(raw[offset : offset + size], w, h)
            rgba[..., 1] = rgba[..., 3]  # Y is stored in alpha
            rgba[..., 2] = 0
            rgba[..., 3] = 255
            levels.append(rgba)
        else:
            rg = bc5.decode_bc5(raw[offset : offset + size], w, h)
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., :2] = rg
            rgba[..., 3] = 255
            levels.append(rgba)
        offset += size
        w, h = max(1, w // 2), max(1, h // 2)
    return levels, dxgi_format


def read_astc_chain(base: Path) -> tuple[list[np.ndarray], int]:
    """<base>.astc, <base>_mip1.astc, ... (astc-float output) -> (float RGB levels, FORMAT_ASTC_4X4_HDR)."""
    levels = []
    path = Path(str(base) + ".astc")
    while path.exists():
        levels.append(astc.astcenc_decode_hdr(path))
        path = Path(f"{base}_mip{len(levels)}.astc")
    if not levels:
        raise FileNotFoundError(f"{base}.astc")
    return levels, FORMAT_ASTC_4X4_HDR


def read_levels(folder: Path, base: str, normal_map: bool) -> tuple[list[np.ndarray], int]:
    """Whatever anbc_compare wrote for <base>: a .dds chain, else the .astc files."""
    dds = folder / f"{base}.dds"
    return read_dds(dds, normal_map) if dds.exists() else read_astc_chain(folder / base)


def to_display_rgb(img: np.ndarray, dxgi_format: int) -> np.ndarray:
    """RGB8 for viewing / FLIP: BC5 / normal maps get their Z reconstructed, HDR is tone-mapped."""
    if dxgi_format in (DXGI_FORMAT_BC5_UNORM, FORMAT_ASTC_4X4_NORMAL):
        return normal_map_rgb8(img[..., :2])
    if dxgi_format in (DXGI_FORMAT_BC6H_UF16, FORMAT_ASTC_4X4_HDR):
        return bc6h.tonemap_for_display(img)
    return img[..., :3]


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def downsample(img: np.ndarray, srgb: bool) -> np.ndarray:
    """CPU twin of the GPU mip generator: 2x2 box, edge-clamped for odd sizes,
    optionally in linear light for RGB. Float (HDR) images are averaged as-is
    and rounded to half, like the RGBA16F GPU texture stores them."""
    h, w, c = img.shape
    dw, dh = max(1, w // 2), max(1, h // 2)
    hdr = img.dtype != np.uint8
    src = img.astype(np.float64) if hdr else img.astype(np.float64) / 255.0
    if srgb and not hdr:
        src[..., :3] = _srgb_to_linear(src[..., :3])
    ys = np.minimum(np.arange(dh * 2), h - 1)
    xs = np.minimum(np.arange(dw * 2), w - 1)
    src = src[ys][:, xs]
    out = src.reshape(dh, 2, dw, 2, c).mean(axis=(1, 3))
    if hdr:
        return np.clip(out, 0.0, 65504.0).astype(np.float16).astype(np.float32)
    if srgb:
        out[..., :3] = _linear_to_srgb(out[..., :3])
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def reference_chain(original: np.ndarray, levels: int, srgb: bool) -> list[np.ndarray]:
    chain = [original]
    for _ in range(levels - 1):
        chain.append(downsample(chain[-1], srgb))
    return chain


def psnr_rgb_rgba(a: np.ndarray, b: np.ndarray, channels: int = 3) -> tuple[float, float]:
    """(PSNR over the first `channels` channels, PSNR over those plus alpha).
    HDR (float) images are scored in the half-int domain, with no alpha."""
    if a.dtype != np.uint8:
        p = bc6h.hdr_psnr(a, b)
        return p, p
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    keep = list(range(channels)) + [3]
    return compute_psnr(a64[..., :channels], b64[..., :channels]), compute_psnr(a64[..., keep], b64[..., keep])


def find_original(original_folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = original_folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def score_folder(dds_folder: Path, original_folder: Path, srgb: bool, use_flip: bool, csv_path: Path | None,
                 normal_map: bool = False) -> None:
    stems = sorted({p.name[: -len("_anbc.dds")] for p in dds_folder.glob("*_anbc.dds")}
                   | {p.name[: -len("_anbc.astc")] for p in dds_folder.glob("*_anbc.astc")})
    rows = []
    print(f"{'texture':48s} {'size':>10s} {'anbc mip0':>10s} {'cmp mip0':>10s} {'anbc mips':>10s} {'cmp mips':>10s}"
          + (f" {'anbc flip':>10s} {'cmp flip':>10s}" if use_flip else ""))
    for stem in stems:
        orig_path = find_original(original_folder, stem)
        if orig_path is None or not ((dds_folder / f"{stem}_cmp.dds").exists() or (dds_folder / f"{stem}_cmp.astc").exists()):
            print(f"{stem:48.48s}  (skipped: missing original or _cmp output)")
            continue
        original = load_image(orig_path)
        anbc_levels, fmt = read_levels(dds_folder, f"{stem}_anbc", normal_map)
        cmp_levels, _ = read_levels(dds_folder, f"{stem}_cmp", normal_map)
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
        if use_flip and fmt in (DXGI_FORMAT_BC6H_UF16, FORMAT_ASTC_4X4_HDR):
            row["anbc_flip"] = compute_flip_hdr(original, anbc_levels[0])
            row["cmp_flip"] = compute_flip_hdr(original, cmp_levels[0])
        elif use_flip:
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


def view_image(dds_folder: Path, original_folder: Path, stem: str, mip: int, srgb: bool, normal_map: bool = False) -> None:
    orig_path = find_original(original_folder, stem)
    if orig_path is None:
        raise SystemExit(f"no original for {stem} in {original_folder}")
    anbc_levels, fmt = read_levels(dds_folder, f"{stem}_anbc", normal_map)
    cmp_levels, _ = read_levels(dds_folder, f"{stem}_cmp", normal_map)
    if mip >= len(anbc_levels):
        raise SystemExit(f"{stem} has {len(anbc_levels)} mips")
    ref = reference_chain(load_image(orig_path), mip + 1, srgb)[mip]
    a, c = anbc_levels[mip], cmp_levels[mip]
    name = FORMAT_NAMES[fmt]
    if fmt in (DXGI_FORMAT_BC5_UNORM, FORMAT_ASTC_4X4_NORMAL):
        label = lambda title, img: quality_label_bc5(title, img[..., :2], ref[..., :2])
    elif fmt in (DXGI_FORMAT_BC6H_UF16, FORMAT_ASTC_4X4_HDR):
        label = lambda title, img: quality_label_bc6h(title, img, ref)
    else:
        label = lambda title, img: quality_label(title, img[..., :3], ref[..., :3])
    reference = "astcenc" if fmt in (DXGI_FORMAT_ASTC_4X4_UNORM, FORMAT_ASTC_4X4_NORMAL, FORMAT_ASTC_4X4_HDR) else "Compressonator"
    panels = [
        (f"Original (mip {mip}, {ref.shape[1]}x{ref.shape[0]})", to_display_rgb(ref, fmt)),
        (label(f"{reference} {name}", c), to_display_rgb(c, fmt)),
        (label(f"anbc {name}", a), to_display_rgb(a, fmt)),
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
    parser.add_argument("--normal-map", action="store_true",
                        help="the ASTC DDS were made with anbc_compare --normal-map: score X,Y from the decoded R and A")
    args = parser.parse_args()

    if args.image:
        view_image(args.dds_folder, args.original_folder, args.image, args.mip, args.srgb, args.normal_map)
    else:
        score_folder(args.dds_folder, args.original_folder, args.srgb, args.flip, args.csv, args.normal_map)
