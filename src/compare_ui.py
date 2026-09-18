"""Visual comparison + benchmark tool: original vs ISPC mode-6 vs neural
mode-6, with synchronized zoom/pan across panels and timing/quality overlays.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import bc7_codec as bc7
from data import load_rgba
from model import BC7Mode5MLP, BC7Mode6CNN, BC7Mode6MLP


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a checkpoint saved by train_bc7.py. Returns (model, mode, arch),
    where mode is 'mode6' or 'mode5' (older checkpoints saved before mode5
    existed default to 'mode6' for backward compatibility)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    mode = ckpt.get("mode", "mode6")
    arch = ckpt["arch"]
    hidden_dim = ckpt["hidden_dim"]
    if mode == "mode5":
        model = BC7Mode5MLP(hidden_dim=hidden_dim)
    elif arch == "mlp":
        model = BC7Mode6MLP(hidden_dim=hidden_dim)
    else:
        model = BC7Mode6CNN(hidden_channels=hidden_dim // 4)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, mode, arch


def _run_mode6(model, arch: str, flat01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (recon (N,16,4), packed_arr (N,16) uint8) for the mode6 model."""
    model_input = flat01 if arch == "mlp" else flat01.view(-1, 4, 4, 4).permute(0, 3, 1, 2).contiguous()
    endpoint0, endpoint1, interp = model(model_input)
    recon = bc7.soft_decode_block(endpoint0, endpoint1, interp)
    packed = bc7.pack_mode6_blocks_batch_torch_arr(endpoint0, endpoint1, interp)
    return recon, packed


def _run_mode5(model, flat01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (recon (N,16,4), packed_arr (N,16) uint8) for the mode5 model."""
    e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a = model(flat01)
    recon = bc7.soft_decode_mode5(e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a)
    packed = bc7.pack_mode5_blocks_batch_torch_arr(e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a)
    return recon, packed


@torch.no_grad()
def neural_encode_decode(
    models: dict[str, tuple[torch.nn.Module, str]], rgba: np.ndarray, device: torch.device
) -> tuple[np.ndarray, float]:
    """Run the loaded neural model(s) over every 4x4 block of `rgba`.

    `models` maps mode name ('mode6'/'mode5') to (model, arch) for however
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
            model, arch = models["mode6"]
            results["mode6"] = _run_mode6(model, arch, flat)
        if "mode5" in models:
            model, _ = models["mode5"]
            results["mode5"] = _run_mode5(model, flat)

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


def warmup_model(models: dict[str, tuple[torch.nn.Module, str]], device: torch.device, num_iters: int = 3) -> None:
    """Run a few dummy batches through the model(s) + GPU bit-packer before
    timing anything for real. The first CUDA call in a process pays a
    one-time cuDNN/kernel JIT + context-init cost (seen empirically as
    ~150-190ms), which otherwise completely swamps the ~10ms of actual
    steady-state inference+packing and makes single-shot timings meaningless."""
    dummy = torch.rand(4096, 64, device=device)
    with torch.no_grad():
        for _ in range(num_iters):
            if "mode6" in models:
                model, arch = models["mode6"]
                _run_mode6(model, arch, dummy)
            if "mode5" in models:
                model, _ = models["mode5"]
                _run_mode5(model, dummy)
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


class CompareApp:
    def __init__(self, image_path: Path, model_paths: list[Path], device_str: str):
        self.device = torch.device(device_str)
        self.rgba = load_rgba(image_path)

        self.ispc_img, self.ispc_time = ispc_encode_decode(self.rgba)

        if model_paths:
            self.models = {}
            for p in model_paths:
                model, mode, arch = load_model(p, self.device)
                self.models[mode] = (model, arch)
            warmup_model(self.models, self.device)
            self.neural_img, self.neural_time = neural_encode_decode(self.models, self.rgba, self.device)
        else:
            self.models, self.neural_img, self.neural_time = {}, None, None

        h_crop, w_crop = self.ispc_img.shape[:2]
        self.original_crop = self.rgba[:h_crop, :w_crop]

        self._build_figure()

    def _build_figure(self) -> None:
        n_panels = 3 if self.neural_img is not None else 2
        self.fig, self.axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharex=True, sharey=True)
        panels = [
            ("Original", self.original_crop[..., :3]),
            ("ISPC BC7 (full search)", self.ispc_img[..., :3]),
        ]
        if self.neural_img is not None:
            neural_label = "Neural BC7 (" + "+".join(sorted(self.models.keys())) + ")"
            panels.append((neural_label, self.neural_img[..., :3]))

        orig01 = self.original_crop[..., :3].astype(np.float32) / 255.0
        for ax, (title, img) in zip(self.axes, panels):
            ax.imshow(img)
            ax.set_title(self._panel_label(title, img, orig01))
            ax.axis("off")

        self.fig.suptitle(
            f"ISPC encode: {self.ispc_time * 1000:.2f}ms"
            + (f"   |   Neural encode: {self.neural_time * 1000:.2f}ms" if self.neural_time else "")
        )

        # Synchronized scroll-to-zoom across all panels (the "magnifying glass").
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        plt.tight_layout()

    def _panel_label(self, title: str, img: np.ndarray, orig01: np.ndarray) -> str:
        if title == "Original":
            return title
        psnr = compute_psnr(img.astype(np.float64), self.original_crop[..., :3].astype(np.float64))
        flip_err = compute_flip(orig01, img.astype(np.float32) / 255.0)
        return f"{title}\nPSNR={psnr:.2f}dB  FLIP={flip_err:.4f}"

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
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    args = parser.parse_args()

    app = CompareApp(args.image, args.model, args.device)
    app.show()
