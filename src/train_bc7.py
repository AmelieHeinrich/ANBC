"""Train a small MLP/CNN to predict BC7 (or BC6H / BC5) blocks from raw 4x4 blocks.

Four modes are supported:
  mode6 -- single shared RGBA index (fast/simple, but can't decorrelate
           alpha from color), trained on DIV2K (all-opaque photos).
  mode5 -- independent RGB and alpha indices, trained on real alpha-cutout
           textures (data/ambientcg_alpha) so it actually learns to predict
           varying alpha instead of the constant-255 DIV2K taught it.
  bc6h  -- BC6H mode 11 (HDR, RGB half-float) trained on data/hdr (DIV2K
           linearised + random exposure, see convert_hdr_dataset.py). Blocks
           are stored as int16 half bit patterns and normalised by 0x7BFF,
           the domain BC6H interpolates in (see bc6h_codec.py).
           SUPERSEDED like BC5: after refinement the block min/max init is
           as good (45.7 vs 45.7 dB), so the shipped encoder has no network;
           kept for experiments (benchmark.py / compare_ui.py --bc6h --model).
  bc5   -- two independent BC4 lines (R and G) for normal maps, trained on
           data/normal (sponza + bistro + intel_sponza normal textures, see
           collect_normal_maps.py). Only the RG channels are loaded.
           SUPERSEDED: the shipped BC5 encoder uses no network (block
           min/max + refinement beats this MLP by ~10 dB on every texture,
           see bc5_codec.py); the mode is kept for reference/experiments.

FLIP is used as the reported perceptual error metric.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import bc5_codec as bc5
import bc6h_codec as bc6h
import bc7_codec as bc7
from data import HDR_DIR, download_div2k, extract_blocks, load_hdr, load_rgba
from model import BC5MLP, BC6HMLP, BC7Mode5MLP, BC7Mode6CNN, BC7Mode6MLP

def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
AMBIENTCG_ALPHA_DIR = Path(__file__).resolve().parent.parent / "data" / "ambientcg_alpha"
NORMAL_DIR = Path(__file__).resolve().parent.parent / "data" / "normal"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}

# Channels per pixel the model sees / reconstructs: RGBA for BC7, RG for BC5, RGB for BC6H.
MODE_CHANNELS = {"mode6": 4, "mode5": 4, "bc5": 2, "bc6h": 3}
# Blocks are kept as integers in RAM and normalised to [0,1] per batch: 8-bit
# values for the LDR formats, half bit patterns (0..0x7BFF) for BC6H.
MODE_SCALE = {"mode6": 255.0, "mode5": 255.0, "bc5": 255.0, "bc6h": float(bc6h.HALF_MAX)}


def _local_split(
    directory: Path, split: str, limit_images: int | None, val_images: int, shuffle_seed: int | None = None
) -> list[Path]:
    """Train/valid split of a flat local folder with no subfolders: the last
    `val_images` files are held out as validation. The order is sorted
    (deterministic); with `shuffle_seed` it is additionally shuffled once
    with that seed so the hold-out mixes sources instead of taking whatever
    sorts last."""
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(files) <= val_images:
        raise ValueError(f"only {len(files)} images in {directory}, need more than val_images={val_images}")
    if shuffle_seed is not None:
        files = [files[i] for i in np.random.RandomState(shuffle_seed).permutation(len(files))]
    train_files, val_files = files[:-val_images], files[-val_images:]
    files = train_files if split == "train" else val_files
    if limit_images is not None and split == "train":
        files = files[:limit_images]
    return files


def get_dataset_files(mode: str, split: str, limit_images: int | None, val_images: int) -> list[Path]:
    """Return the list of image files for `split` ('train' or 'valid')."""
    if mode == "mode6":
        directory = download_div2k(split)
        files = sorted(directory.glob("*.png"))
        if split == "train":
            if limit_images is not None:
                files = files[:limit_images]
        else:
            files = files[:val_images]
        return files

    if mode == "mode5":
        return _local_split(AMBIENTCG_ALPHA_DIR, split, limit_images, val_images)

    if mode == "bc6h":
        directory = HDR_DIR / ("train" if split == "train" else "valid")
        files = sorted(directory.glob("*.hdr"))
        if not files:
            raise FileNotFoundError(f"no .hdr files in {directory}: run src/convert_hdr_dataset.py first")
        if split == "train":
            if limit_images is not None:
                files = files[:limit_images]
        else:
            files = files[:val_images]
        return files

    # bc5: sorted order would hold out only 4096^2 intel_sponza files (they
    # sort last), so shuffle deterministically to mix the three sources.
    return _local_split(NORMAL_DIR, split, limit_images, val_images, shuffle_seed=0)


def build_block_dataset(
    mode: str, split: str, limit_images: int | None, val_images: int, shuffle: bool = False
) -> torch.Tensor:
    """Load images and extract all 4x4 blocks into one big tensor, shape
    (N, 16 * channels) uint8 (RGBA channel order; RG only for bc5, which
    never stores B) -- or int16 half bit patterns for bc6h. Kept as
    integers (not float32) so the full DIV2K block set (~35GB as float32)
    fits comfortably in RAM (~9GB as uint8); normalization to [0,1] happens
    per-batch instead (MODE_SCALE).

    `shuffle=True` (training set only) does ONE full random permutation
    here, so the training loop can shuffle each epoch just by permuting
    *chunk order* and slicing contiguously, instead of doing a full random
    gather over 139M rows every epoch (see train() for why that mattered)."""
    files = get_dataset_files(mode, split, limit_images, val_images)
    channels = MODE_CHANNELS[mode]

    all_blocks = []
    for f in tqdm(files, desc=f"loading {mode} {split} blocks"):
        if mode == "bc6h":
            img = bc6h.float_to_halfint(load_hdr(f)).astype(np.int16)  # (H, W, 3), 0..0x7BFF
        else:
            img = load_rgba(f)
        blocks = extract_blocks(img)[..., :channels]  # (N, 4, 4, channels)
        all_blocks.append(np.ascontiguousarray(blocks).reshape(-1, 16 * channels))

    blocks = np.concatenate(all_blocks, axis=0)
    blocks_t = torch.from_numpy(blocks)
    if shuffle:
        blocks_t = blocks_t[torch.randperm(blocks_t.shape[0])]
    return blocks_t


def run_model(model: nn.Module, mode: str, arch: str, batch01: torch.Tensor) -> torch.Tensor:
    """Run the model on a (B, 16 * channels) float batch in [0,1] and return
    the reconstructed (B, 16, channels) pixels via the appropriate soft-decode."""
    if mode == "mode6":
        if arch == "mlp":
            model_input = batch01
        else:
            model_input = batch01.view(-1, 4, 4, 4).permute(0, 3, 1, 2).contiguous()
        endpoint0, endpoint1, interp = model(model_input)
        return bc7.soft_decode_block(endpoint0, endpoint1, interp)

    if mode == "bc5":
        return bc5.soft_decode_bc5(*model(batch01))

    if mode == "bc6h":
        # Exact indices, not the net's own blend factors: at inference the net
        # only supplies endpoints (see bc6h_codec.exact_decode_bc6h).
        pixels = batch01.view(-1, 16, 3)
        e0, e1, _ = bc6h.predict_endpoints(model, pixels)
        return bc6h.exact_decode_bc6h(e0, e1, pixels)

    e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a = model(batch01)
    return bc7.soft_decode_mode5(e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a)


def train(
    mode: str,
    epochs: int,
    limit_images: int | None,
    val_images: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    arch: str,
    device: str,
) -> None:
    blocks = build_block_dataset(mode, "train", limit_images, val_images, shuffle=True)
    print(f"Total training blocks: {blocks.shape[0]} ({blocks.numel() * blocks.element_size() / 1e9:.2f} GB in RAM)")

    val_blocks = build_block_dataset(mode, "valid", limit_images=None, val_images=val_images)

    device_t = torch.device(device)
    channels = MODE_CHANNELS[mode]
    scale = MODE_SCALE[mode]
    if mode == "mode6":
        model = (BC7Mode6MLP(hidden_dim=hidden_dim) if arch == "mlp" else BC7Mode6CNN(hidden_channels=hidden_dim // 4)).to(device_t)
    elif mode == "mode5":
        arch = "mlp"  # mode5 only has an MLP variant for now
        model = BC7Mode5MLP(hidden_dim=hidden_dim).to(device_t)
    elif mode == "bc6h":
        arch = "mlp"
        model = BC6HMLP(hidden_dim=hidden_dim).to(device_t)
    else:
        arch = "mlp"
        model = BC5MLP(hidden_dim=hidden_dim).to(device_t)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    n = blocks.shape[0]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # blocks was already fully shuffled once above. From here on, shuffling
    # each epoch is done by permuting *chunk order* and slicing contiguously
    # -- a full torch.randperm(n)-then-gather over 139M rows every epoch
    # (the original approach) is a random-access memory gather that
    # dominated wall-clock time far more than the actual GPU compute, which
    # is sub-millisecond for a model this small. Chunk-level shuffling keeps
    # nearly all of the statistical benefit (rows within a chunk are already
    # in random order from the one-time full shuffle) at a fraction of the
    # cost -- each chunk is a plain contiguous memcpy, not a random gather.
    num_chunks = (n + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        chunk_order = torch.randperm(num_chunks)
        loss_sum = torch.zeros((), device=device_t)
        num_batches = 0
        pbar = tqdm(chunk_order.tolist(), desc=f"epoch {epoch}")
        for chunk_idx in pbar:
            start = chunk_idx * batch_size
            batch = blocks[start : start + batch_size].to(device_t).float() / scale

            recon = run_model(model, mode, arch, batch)
            target = batch.view(-1, 16, channels)
            loss = loss_fn(recon, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_sum += loss.detach()
            num_batches += 1
            # Only sync (via .item()) every 50 steps for the progress display
            # -- syncing every single step (the original approach) forces a
            # CPU/GPU round trip 17000 times/epoch and was the single
            # biggest contributor to the reported slowdown.
            if num_batches % 50 == 0:
                pbar.set_postfix(loss=(loss_sum / num_batches).item())

        epoch_loss = (loss_sum / num_batches).item()
        val_metrics = evaluate(model, mode, arch, val_blocks, device_t)
        print(
            f"epoch {epoch}: train_loss={epoch_loss:.5f} "
            f"val_l1={val_metrics['l1']:.5f} val_psnr={val_metrics['psnr']:.2f}dB "
            f"val_flip={val_metrics['flip']:.5f}"
        )

        ckpt_names = {"bc5": f"bc5_{arch}.pt", "bc6h": f"bc6h_mode11_{arch}.pt"}
        ckpt_path = CHECKPOINT_DIR / ckpt_names.get(mode, f"bc7_{mode}_{arch}.pt")
        torch.save(
            {"model_state": model.state_dict(), "mode": mode, "arch": arch, "hidden_dim": hidden_dim},
            ckpt_path,
        )

    print(f"Saved checkpoint to {ckpt_path}")


@torch.no_grad()
def evaluate(
    model: nn.Module, mode: str, arch: str, val_blocks: torch.Tensor, device_t: torch.device, chunk: int = 65536
) -> dict:
    """Validation metrics. Runs in chunks: the val set is millions of blocks
    once the images are 2048^2/4096^2 normal maps, which doesn't fit on the
    GPU as one float32 batch plus its reconstruction."""
    model.eval()
    channels = MODE_CHANNELS[mode]
    n = val_blocks.shape[0]

    # Per-chunk sums are accumulated in float64 on the host (MPS has no float64).
    abs_sum, sq_sum = 0.0, 0.0
    first_target = first_recon = None
    for start in range(0, n, chunk):
        batch = val_blocks[start : start + chunk].to(device_t).float() / MODE_SCALE[mode]
        recon = run_model(model, mode, arch, batch)
        target = batch.view(-1, 16, channels)
        diff = recon - target
        abs_sum += float(diff.abs().sum())
        sq_sum += float((diff * diff).sum())
        if first_target is None:
            first_target, first_recon = target, recon

    count = n * 16 * channels
    l1 = abs_sum / count
    mse = sq_sum / count
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    flip_score = compute_flip_on_subset(first_target, first_recon, channels)

    return {"l1": l1, "psnr": psnr, "flip": flip_score}


def compute_flip_on_subset(target: torch.Tensor, recon: torch.Tensor, channels: int, num_blocks: int = 64) -> float:
    """Arrange a handful of validation blocks into a small synthetic image
    and compute FLIP error between target and reconstruction, since FLIP
    operates on full images rather than isolated 4x4 blocks. 2-channel (BC5)
    blocks are shown as normal maps with the reconstructed Z; 3-channel
    (BC6H) blocks are in the half-int domain and go through HDR-FLIP as
    linear float RGB."""
    try:
        import flip_evaluator as flip
    except ImportError:
        return float("nan")

    n = min(num_blocks, target.shape[0])
    grid = int(np.ceil(np.sqrt(n)))
    hdr = channels == 3

    def blocks_to_image(x: torch.Tensor) -> np.ndarray:
        arr = x[:n].detach().cpu().numpy().reshape(n, 4, 4, channels)
        pad = grid * grid - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, 4, 4, channels), dtype=arr.dtype)], axis=0)
        arr = arr.reshape(grid, grid, 4, 4, channels).transpose(0, 2, 1, 3, 4).reshape(grid * 4, grid * 4, channels)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        if hdr:
            return bc6h.norm_to_float(arr)
        return bc5.rg_to_rgb(arr) if channels == 2 else arr[..., :3]

    img_a = blocks_to_image(target)
    img_b = blocks_to_image(recon)
    result = flip.evaluate(img_a, img_b, "HDR" if hdr else "LDR")
    mean_flip = result[1] if isinstance(result, tuple) else result.get("mean", float("nan"))
    return float(mean_flip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mode6", "mode5", "bc6h", "bc5"], default="mode6")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--arch", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()

    # The local-folder datasets (mode5, bc5) are small; cap the hold-out.
    val_images = args.val_images if args.mode in ("mode6", "bc6h") else min(args.val_images, 4)

    t0 = time.time()
    train(
        mode=args.mode,
        epochs=args.epochs,
        limit_images=args.limit_images,
        val_images=val_images,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        arch=args.arch,
        device=args.device,
    )
    print(f"Total time: {time.time() - t0:.1f}s")
