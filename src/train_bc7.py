"""Train a small MLP/CNN to predict BC7 blocks from raw RGBA8 4x4 blocks.

Two modes are supported:
  mode6 -- single shared RGBA index (fast/simple, but can't decorrelate
           alpha from color), trained on DIV2K (all-opaque photos).
  mode5 -- independent RGB and alpha indices, trained on real alpha-cutout
           textures (data/ambientcg_alpha) so it actually learns to predict
           varying alpha instead of the constant-255 DIV2K taught it.

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

import bc7_codec as bc7
from data import download_div2k, extract_blocks, load_rgba
from model import BC7Mode5MLP, BC7Mode6CNN, BC7Mode6MLP

def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
AMBIENTCG_ALPHA_DIR = Path(__file__).resolve().parent.parent / "data" / "ambientcg_alpha"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tga"}


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

    # mode5: small local folder, no separate train/valid subfolders -- hold
    # out the last `val_images` files (sorted, deterministic) as validation.
    files = sorted(p for p in AMBIENTCG_ALPHA_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(files) <= val_images:
        raise ValueError(
            f"only {len(files)} images in {AMBIENTCG_ALPHA_DIR}, need more than val_images={val_images}"
        )
    train_files, val_files = files[:-val_images], files[-val_images:]
    files = train_files if split == "train" else val_files
    if limit_images is not None and split == "train":
        files = files[:limit_images]
    return files


def build_block_dataset(
    mode: str, split: str, limit_images: int | None, val_images: int, shuffle: bool = False
) -> torch.Tensor:
    """Load images and extract all 4x4 blocks into one big tensor, shape
    (N, 64) uint8 (RGBA channel order). Kept as uint8 (not float32) so the
    full DIV2K block set (~35GB as float32) fits comfortably in RAM
    (~9GB as uint8); normalization to [0,1] happens per-batch instead.

    `shuffle=True` (training set only) does ONE full random permutation
    here, so the training loop can shuffle each epoch just by permuting
    *chunk order* and slicing contiguously, instead of doing a full random
    gather over 139M rows every epoch (see train() for why that mattered)."""
    files = get_dataset_files(mode, split, limit_images, val_images)

    all_blocks = []
    for f in tqdm(files, desc=f"loading {mode} {split} blocks"):
        rgba = load_rgba(f)
        blocks = extract_blocks(rgba)  # (N, 4, 4, 4) uint8
        all_blocks.append(blocks.reshape(-1, 64))

    blocks = np.concatenate(all_blocks, axis=0)
    blocks_t = torch.from_numpy(blocks)
    if shuffle:
        blocks_t = blocks_t[torch.randperm(blocks_t.shape[0])]
    return blocks_t


def run_model(model: nn.Module, mode: str, arch: str, batch01: torch.Tensor) -> torch.Tensor:
    """Run the model on a (B, 64) float batch in [0,1] and return the
    reconstructed (B, 16, 4) RGBA via the appropriate soft-decode."""
    if mode == "mode6":
        if arch == "mlp":
            model_input = batch01
        else:
            model_input = batch01.view(-1, 4, 4, 4).permute(0, 3, 1, 2).contiguous()
        endpoint0, endpoint1, interp = model(model_input)
        return bc7.soft_decode_block(endpoint0, endpoint1, interp)

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
    print(f"Total training blocks: {blocks.shape[0]} ({blocks.numel() / 1e9:.2f} GB as uint8)")

    val_blocks = build_block_dataset(mode, "valid", limit_images=None, val_images=val_images)

    device_t = torch.device(device)
    if mode == "mode6":
        model = (BC7Mode6MLP(hidden_dim=hidden_dim) if arch == "mlp" else BC7Mode6CNN(hidden_channels=hidden_dim // 4)).to(device_t)
    else:
        arch = "mlp"  # mode5 only has an MLP variant for now
        model = BC7Mode5MLP(hidden_dim=hidden_dim).to(device_t)

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
            batch = blocks[start : start + batch_size].to(device_t).float() / 255.0

            recon = run_model(model, mode, arch, batch)
            target = batch.view(-1, 16, 4)
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

        ckpt_path = CHECKPOINT_DIR / f"bc7_{mode}_{arch}.pt"
        torch.save(
            {"model_state": model.state_dict(), "mode": mode, "arch": arch, "hidden_dim": hidden_dim},
            ckpt_path,
        )

    print(f"Saved checkpoint to {ckpt_path}")


@torch.no_grad()
def evaluate(model: nn.Module, mode: str, arch: str, val_blocks: torch.Tensor, device_t: torch.device) -> dict:
    model.eval()
    batch = val_blocks.to(device_t).float() / 255.0

    recon = run_model(model, mode, arch, batch)
    target = batch.view(-1, 16, 4)

    l1 = torch.mean(torch.abs(recon - target)).item()
    mse = torch.mean((recon - target) ** 2).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    flip_score = compute_flip_on_subset(target, recon)

    return {"l1": l1, "psnr": psnr, "flip": flip_score}


def compute_flip_on_subset(target: torch.Tensor, recon: torch.Tensor, num_blocks: int = 64) -> float:
    """Arrange a handful of validation blocks into a small synthetic image
    and compute FLIP error between target and reconstruction, since FLIP
    operates on full images rather than isolated 4x4 blocks."""
    try:
        import flip_evaluator as flip
    except ImportError:
        return float("nan")

    n = min(num_blocks, target.shape[0])
    grid = int(np.ceil(np.sqrt(n)))

    def blocks_to_image(x: torch.Tensor) -> np.ndarray:
        arr = x[:n].detach().cpu().numpy().reshape(n, 4, 4, 4)
        pad = grid * grid - n
        if pad > 0:
            arr = np.concatenate([arr, np.zeros((pad, 4, 4, 4), dtype=arr.dtype)], axis=0)
        arr = arr.reshape(grid, grid, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(grid * 4, grid * 4, 4)
        return np.clip(arr[..., :3], 0.0, 1.0).astype(np.float32)

    img_a = blocks_to_image(target)
    img_b = blocks_to_image(recon)
    result = flip.evaluate(img_a, img_b, "LDR")
    mean_flip = result[1] if isinstance(result, tuple) else result.get("mean", float("nan"))
    return float(mean_flip)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mode6", "mode5"], default="mode6")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--arch", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args()

    val_images = args.val_images if args.mode == "mode6" else min(args.val_images, 4)

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
