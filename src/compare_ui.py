"""Visual comparison + benchmark tool: original vs ISPC vs ours (neural BC7,
or the network-free BC5 / BC6H / ASTC encoders), with synchronized zoom/pan
across panels and timing/quality overlays.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import bc5_codec as bc5
import bc6h_codec as bc6h
import bc7_codec as bc7
from data import load_image
from model import BC7Mode5MLP, BC7Mode6MLP


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load a checkpoint saved by train_bc7.py. Returns (model, mode), where
    mode is 'mode6' or 'mode5' (older checkpoints saved before mode5 existed
    default to 'mode6' for backward compatibility)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    mode = ckpt.get("mode", "mode6")
    hidden_dim = ckpt["hidden_dim"]
    if mode == "mode5":
        model = BC7Mode5MLP(hidden_dim=hidden_dim)
    elif mode == "mode6":
        model = BC7Mode6MLP(hidden_dim=hidden_dim)
    else:
        raise ValueError(f"{checkpoint_path}: {mode} checkpoints are not supported (only BC7 mode6/mode5 ship a network)")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, mode


# Number of exact-index / least-squares refinement rounds applied to the
# model's raw output before packing (see bc7_codec.refine_mode5/6). 0 packs
# the network's own blend factors as-is (the old behaviour).
REFINE_ITERS = 2


def _run_mode6(model, flat01: torch.Tensor, refine_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (recon (N,16,4), packed_arr (N,16) uint8) for the mode6 model."""
    endpoint0, endpoint1, interp = model(flat01)
    if refine_iters > 0:
        endpoint0, endpoint1, interp, recon = bc7.refine_mode6(flat01.view(-1, 16, 4), endpoint0, endpoint1, refine_iters)
    else:
        recon = bc7.soft_decode_block(endpoint0, endpoint1, interp)
    packed = bc7.pack_mode6_blocks_batch_torch_arr(endpoint0, endpoint1, interp)
    return recon, packed


def _run_mode5(model, flat01: torch.Tensor, refine_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (recon (N,16,4), packed_arr (N,16) uint8) for the mode5 model."""
    e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a = model(flat01)
    if refine_iters > 0:
        e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a, recon = bc7.refine_mode5(
            flat01.view(-1, 16, 4), e0_rgb, e1_rgb, e0_a, e1_a, refine_iters
        )
    else:
        recon = bc7.soft_decode_mode5(e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a)
    packed = bc7.pack_mode5_blocks_batch_torch_arr(e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a)
    return recon, packed


def _tile_blocks(rgba: np.ndarray, channels: int) -> tuple[np.ndarray, int, int]:
    """Crop to a multiple of 4 and tile into (N, 16 * channels) raster-order
    blocks keeping the first `channels` channels. Returns (blocks, h_crop, w_crop)."""
    h, w = rgba.shape[:2]
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    blocks = (
        rgba[:h_crop, :w_crop, :channels]
        .reshape(h_crop // 4, 4, w_crop // 4, 4, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 16 * channels)
    )
    return blocks, h_crop, w_crop


@torch.no_grad()
def encode_decode_bc5(rgba: np.ndarray, device: torch.device, refine_iters: int = REFINE_ITERS) -> tuple[np.ndarray, float]:
    """BC5-encode the R/G channels of `rgba` (bc5_codec.encode_bc5_blocks: block
    min/max + refinement, no network) and decode back.
    Returns ((h_crop, w_crop, 2) uint8 RG, encode_seconds)."""
    blocks, h_crop, w_crop = _tile_blocks(rgba, 2)
    flat = torch.from_numpy(blocks.astype(np.float32) / 255.0).to(device)

    t0 = time.perf_counter()
    packed_arr, _ = bc5.encode_bc5_blocks(flat.view(-1, 16, 2), refine_iters)
    packed_bytes = packed_arr.cpu().numpy().tobytes()
    if device.type == "cuda":
        torch.cuda.synchronize()
    encode_time = time.perf_counter() - t0

    return bc5.decode_bc5(packed_bytes, w_crop, h_crop), encode_time


@torch.no_grad()
def encode_decode_bc6h(rgb: np.ndarray, device: torch.device, refine_iters: int = REFINE_ITERS) -> tuple[np.ndarray, float]:
    """BC6H-encode a float RGB image (bc6h_codec.encode_bc6h_blocks: block
    min/max + refinement, no network) and decode back.
    Returns ((h_crop, w_crop, 3) float32, encode_seconds)."""
    blocks, h_crop, w_crop = _tile_blocks(bc6h.float_to_norm(rgb), 3)
    flat = torch.from_numpy(blocks).to(device)

    t0 = time.perf_counter()
    packed_arr, _ = bc6h.encode_bc6h_blocks(flat.view(-1, 16, 3), refine_iters)
    packed_bytes = packed_arr.cpu().numpy().tobytes()
    if device.type == "cuda":
        torch.cuda.synchronize()
    encode_time = time.perf_counter() - t0

    return bc6h.decode_bc6h(packed_bytes, w_crop, h_crop), encode_time


def ispc_encode_decode_bc6h(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """ISPC BC6H reference (full mode search). Returns ((h_crop, w_crop, 3) float32, encode_seconds)."""
    h, w = rgb.shape[:2]
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    cropped = np.ascontiguousarray(rgb[:h_crop, :w_crop])

    t0 = time.perf_counter()
    encoded = bc6h.ispc_encode_bc6h(cropped)
    encode_time = time.perf_counter() - t0

    return bc6h.decode_bc6h(encoded, w_crop, h_crop), encode_time


def ispc_encode_decode_bc5(rgba: np.ndarray) -> tuple[np.ndarray, float]:
    """ISPC BC5 reference. Returns ((h_crop, w_crop, 2) uint8 RG, encode_seconds)."""
    h, w = rgba.shape[:2]
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    cropped = rgba[:h_crop, :w_crop]

    t0 = time.perf_counter()
    encoded = bc5.ispc_encode_bc5(cropped)
    encode_time = time.perf_counter() - t0

    return bc5.decode_bc5(encoded, w_crop, h_crop), encode_time


@torch.no_grad()
def neural_encode_decode(
    models: dict[str, torch.nn.Module],
    rgba: np.ndarray,
    device: torch.device,
    refine_iters: int = REFINE_ITERS,
) -> tuple[np.ndarray, float]:
    """Run the loaded neural model(s) over every 4x4 block of `rgba`.

    `models` maps mode name ('mode6'/'mode5') to the model for however
    many modes were loaded. If both are present, this does real per-block
    *mode selection*: each block is reconstructed with both, the one with
    lower L1 error against the original block wins (mirroring how a real
    BC7 encoder tries several modes per block and keeps the cheapest-good
    one), and the final texture is packed by mixing whichever mode's bytes
    won for that block. If only one mode is loaded, that one is used for
    every block.

    Returns (reconstructed_rgba, encode_seconds)."""
    h, w = rgba.shape[:2]
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    cropped = rgba[:h_crop, :w_crop]

    blocks = (
        cropped.reshape(h_crop // 4, 4, w_crop // 4, 4, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(-1, 4, 4, 4)
    )
    n = blocks.shape[0]
    flat = torch.from_numpy(blocks.reshape(n, 64).astype(np.float32) / 255.0).to(device)
    target = flat.view(n, 16, 4)

    t0 = time.perf_counter()
    with torch.no_grad():
        results = {}
        if "mode6" in models:
            results["mode6"] = _run_mode6(models["mode6"], flat, refine_iters)
        if "mode5" in models:
            results["mode5"] = _run_mode5(models["mode5"], flat, refine_iters)

        if len(results) == 2:
            recon6, packed6 = results["mode6"]
            recon5, packed5 = results["mode5"]
            err6 = torch.mean(torch.abs(recon6 - target), dim=(1, 2))
            err5 = torch.mean(torch.abs(recon5 - target), dim=(1, 2))
            use_mode5 = (err5 < err6).unsqueeze(-1)  # (N,1)
            packed_arr = torch.where(use_mode5, packed5, packed6)
        else:
            _, packed_arr = next(iter(results.values()))

        packed_bytes = packed_arr.cpu().numpy().tobytes()
    if device.type == "cuda":
        torch.cuda.synchronize()
    encode_time = time.perf_counter() - t0

    # decode_bc7 takes the packed block bytes (in row-major block order,
    # matching how `blocks` was tiled above) and reconstructs the full
    # (h_crop, w_crop, 4) image directly.
    out = bc7.decode_bc7(packed_bytes, w_crop, h_crop)

    return out, encode_time


def warmup_model(models: dict[str, torch.nn.Module], device: torch.device, num_iters: int = 3) -> None:
    """Run a few dummy batches through the model(s) + GPU bit-packer before
    timing anything for real. The first CUDA call in a process pays a
    one-time cuDNN/kernel JIT + context-init cost (seen empirically as
    ~150-190ms), which otherwise completely swamps the ~10ms of actual
    steady-state inference+packing and makes single-shot timings meaningless."""
    dummy = torch.rand(4096, 64, device=device)
    with torch.no_grad():
        for _ in range(num_iters):
            if "mode6" in models:
                _run_mode6(models["mode6"], dummy, REFINE_ITERS)
            if "mode5" in models:
                _run_mode5(models["mode5"], dummy, REFINE_ITERS)
    if device.type == "cuda":
        torch.cuda.synchronize()


def ispc_encode_decode(rgba: np.ndarray) -> tuple[np.ndarray, float]:
    """ISPC's full mode search (all BC7 modes, alpha-aware) -- the real
    encoder's actual quality/speed bar, used since the neural side now also
    picks between modes (mode6 vs mode5) per block instead of being locked
    to one mode."""
    h, w = rgba.shape[:2]
    h_crop, w_crop = (h // 4) * 4, (w // 4) * 4
    cropped = rgba[:h_crop, :w_crop]

    t0 = time.perf_counter()
    encoded = bc7.ispc_encode_best(cropped)
    encode_time = time.perf_counter() - t0

    decoded = bc7.decode_bc7(encoded, w_crop, h_crop)
    return decoded, encode_time


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def compute_flip(a_rgb01: np.ndarray, b_rgb01: np.ndarray) -> float:
    try:
        import flip_evaluator as flip
    except ImportError:
        return float("nan")
    _, mean_flip, _ = flip.evaluate(a_rgb01, b_rgb01, "LDR")
    return float(mean_flip)


class SyncedViewer:
    """Side-by-side panels with synchronized scroll-to-zoom (the "magnifying
    glass"). `panels` is a list of (title, rgb uint8 image)."""

    def __init__(self, panels: list[tuple[str, np.ndarray]], suptitle: str = ""):
        self.fig, self.axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), sharex=True, sharey=True)
        if len(panels) == 1:
            self.axes = [self.axes]
        for ax, (title, img) in zip(self.axes, panels):
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
        if suptitle:
            self.fig.suptitle(suptitle)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        plt.tight_layout()

    def _on_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        for ax in self.axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x, y = event.xdata, event.ydata
            new_w = (xlim[1] - xlim[0]) * scale
            new_h = (ylim[1] - ylim[0]) * scale
            ax.set_xlim(x - new_w / 2, x + new_w / 2)
            ax.set_ylim(y - new_h / 2, y + new_h / 2)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def quality_label(title: str, img_rgb: np.ndarray, original_rgb: np.ndarray) -> str:
    """Panel title with PSNR/FLIP of `img_rgb` against `original_rgb` (both uint8)."""
    psnr = compute_psnr(img_rgb.astype(np.float64), original_rgb.astype(np.float64))
    flip_err = compute_flip(original_rgb.astype(np.float32) / 255.0, img_rgb.astype(np.float32) / 255.0)
    return f"{title}\nPSNR={psnr:.2f}dB  FLIP={flip_err:.4f}"


def quality_label_bc5(title: str, img_rg: np.ndarray, original_rg: np.ndarray) -> str:
    """BC5 variant: PSNR over the two stored channels, FLIP on the normal
    map as it would be rendered (Z reconstructed from RG)."""
    psnr = compute_psnr(img_rg.astype(np.float64), original_rg.astype(np.float64))
    flip_err = compute_flip(normal_map_rgb01(original_rg), normal_map_rgb01(img_rg))
    return f"{title}\nPSNR={psnr:.2f}dB  FLIP={flip_err:.4f}"


def quality_label_bc6h(title: str, img_rgb: np.ndarray, original_rgb: np.ndarray) -> str:
    """BC6H variant: PSNR in the half-int domain, HDR-FLIP on linear RGB."""
    psnr = bc6h.hdr_psnr(img_rgb, original_rgb)
    flip_err = compute_flip_hdr(original_rgb, img_rgb)
    return f"{title}\nPSNR(half-int)={psnr:.2f}dB  HDR-FLIP={flip_err:.4f}"


def compute_flip_hdr(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    try:
        import flip_evaluator as flip
    except ImportError:
        return float("nan")
    _, mean_flip, _ = flip.evaluate(a_rgb.astype(np.float32), b_rgb.astype(np.float32), "HDR")
    return float(mean_flip)


def normal_map_rgb01(rg: np.ndarray) -> np.ndarray:
    """(H,W,2) uint8 RG -> (H,W,3) float32 RGB in [0,1] with reconstructed Z."""
    return bc5.rg_to_rgb(rg.astype(np.float32) / 255.0)


def normal_map_rgb8(rg: np.ndarray) -> np.ndarray:
    return np.clip(np.round(normal_map_rgb01(rg) * 255.0), 0, 255).astype(np.uint8)


def load_models(model_paths: list[Path], device: torch.device) -> dict[str, torch.nn.Module]:
    """Load BC7 checkpoints into a {mode: model} dict and warm them up."""
    models = {}
    for p in model_paths:
        model, mode = load_model(p, device)
        models[mode] = model
    warmup_model(models, device)
    return models


class CompareApp(SyncedViewer):
    def __init__(
        self,
        image_path: Path,
        model_paths: list[Path],
        device_str: str,
        refine_iters: int = REFINE_ITERS,
        bc5: bool = False,
        is_bc6h: bool = False,
        astc_variant: str | None = None,
    ):
        self.device = torch.device(device_str)
        self.rgba = load_image(image_path)
        self.models = load_models(model_paths, self.device) if model_paths else {}
        self.neural_img, self.neural_time = None, None

        if astc_variant:
            # ASTC 4x4 vs the vendored astcenc CLI (-medium). Same scoring /
            # display rules as the BCn panels: game on RGB(A), normal on the
            # two stored channels with Z reconstructed, float half-int + tone-mapped.
            import astc_codec as astc

            self.ispc_img, self.ispc_time = astc.astcenc_encode_decode(astc_variant, image_path)
            astc.encode_image(astc_variant, self.rgba, self.device, refine_iters)  # warm-up
            _, self.neural_img, chosen, w_crop, h_crop, self.neural_time = astc.encode_image(
                astc_variant, self.rgba, self.device, refine_iters)
            self.ispc_img = self.ispc_img[:h_crop, :w_crop]
            configs = astc.VARIANT_CONFIGS[astc_variant]
            share = ", ".join(f"{c.name} {100 * np.mean(chosen == i):.0f}%" for i, c in enumerate(configs))
            ours = f"anbc ASTC 4x4 {astc_variant} (min/max + refine; {share})"
            ref = "astcenc -medium"
            if astc_variant == "float":
                original = self.rgba[:h_crop, :w_crop]
                panels = [
                    ("Original (tone-mapped)", bc6h.tonemap_for_display(original)),
                    (quality_label_bc6h(ref, self.ispc_img, original), bc6h.tonemap_for_display(self.ispc_img)),
                    (quality_label_bc6h(ours, self.neural_img, original), bc6h.tonemap_for_display(self.neural_img)),
                ]
            elif astc_variant == "normal":
                original = self.rgba[:h_crop, :w_crop, :2]
                panels = [
                    ("Original", normal_map_rgb8(original)),
                    (quality_label_bc5(ref + " (-normal)", self.ispc_img, original), normal_map_rgb8(self.ispc_img)),
                    (quality_label_bc5(ours, self.neural_img, original), normal_map_rgb8(self.neural_img)),
                ]
            else:
                original = self.rgba[:h_crop, :w_crop, :3]
                panels = [
                    ("Original", original),
                    (quality_label(ref, self.ispc_img[..., :3], original), self.ispc_img[..., :3]),
                    (quality_label(ours, self.neural_img[..., :3], original), self.neural_img[..., :3]),
                ]
        elif is_bc6h:
            # HDR: score in the half-int domain, display tone-mapped.
            self.ispc_img, self.ispc_time = ispc_encode_decode_bc6h(self.rgba)
            encode_decode_bc6h(self.rgba, self.device, refine_iters)  # warm-up
            self.neural_img, self.neural_time = encode_decode_bc6h(self.rgba, self.device, refine_iters)

            h_crop, w_crop = self.ispc_img.shape[:2]
            original = self.rgba[:h_crop, :w_crop]
            panels = [
                ("Original (tone-mapped)", bc6h.tonemap_for_display(original)),
                (quality_label_bc6h("ISPC BC6H (full search)", self.ispc_img, original), bc6h.tonemap_for_display(self.ispc_img)),
                (quality_label_bc6h("anbc BC6H (min/max + refine, mode 11)", self.neural_img, original),
                 bc6h.tonemap_for_display(self.neural_img)),
            ]
        elif bc5:
            # Normal map: compare the two stored channels, display with reconstructed Z.
            self.ispc_img, self.ispc_time = ispc_encode_decode_bc5(self.rgba)
            encode_decode_bc5(self.rgba, self.device, refine_iters)  # warm-up
            self.neural_img, self.neural_time = encode_decode_bc5(self.rgba, self.device, refine_iters)

            h_crop, w_crop = self.ispc_img.shape[:2]
            original = self.rgba[:h_crop, :w_crop, :2]
            panels = [
                ("Original", normal_map_rgb8(original)),
                (quality_label_bc5("ISPC BC5", self.ispc_img, original), normal_map_rgb8(self.ispc_img)),
                (quality_label_bc5("anbc BC5 (min/max + refine)", self.neural_img, original), normal_map_rgb8(self.neural_img)),
            ]
        else:
            self.ispc_img, self.ispc_time = ispc_encode_decode(self.rgba)
            if self.models:
                self.neural_img, self.neural_time = neural_encode_decode(self.models, self.rgba, self.device, refine_iters)

            h_crop, w_crop = self.ispc_img.shape[:2]
            original = self.rgba[:h_crop, :w_crop, :3]
            panels = [
                ("Original", original),
                (quality_label("ISPC BC7 (full search)", self.ispc_img[..., :3], original), self.ispc_img[..., :3]),
            ]
            if self.neural_img is not None:
                neural_label = "Neural BC7 (" + "+".join(sorted(self.models.keys())) + ")"
                panels.append((quality_label(neural_label, self.neural_img[..., :3], original), self.neural_img[..., :3]))

        super().__init__(
            panels,
            f"{'astcenc' if astc_variant else 'ISPC'} encode: {self.ispc_time * 1000:.2f}ms"
            + (f"   |   Neural encode: {self.neural_time * 1000:.2f}ms" if self.neural_time else ""),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        default=[],
        help="Checkpoint path; pass twice (--model bc7_mode6_mlp.pt --model bc7_mode5_mlp.pt) "
        "to enable real per-block mode selection between mode6 and mode5.",
    )
    parser.add_argument("--bc5", action="store_true", help="compare BC5 (normal maps: R/G) instead of BC7; needs no --model")
    parser.add_argument("--bc6h", action="store_true", help="compare BC6H on a .hdr image; needs no --model")
    parser.add_argument("--astc", choices=["game", "normal", "float"], default=None,
                        help="compare ASTC 4x4 of this texture kind against astcenc; needs no --model")
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

    app = CompareApp(args.image, args.model, args.device, args.refine_iters, args.bc5, args.bc6h, args.astc)
    app.show()
