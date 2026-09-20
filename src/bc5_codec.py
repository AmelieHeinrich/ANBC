"""BC5 codec: encoder (block min/max endpoints + exact-index / least-squares
refinement), bit-exact packer and wrappers around ispc-texcomp (ground
truth) and texture2ddecoder (reference decoder). Built from the same generic
pieces as bc7_codec.

No network: a BC4 line is 1-D, and the block's own min/max is already a
near-optimal endpoint pair -- min/max + 2 refinement rounds scores ~56 dB on
the normal-map set vs ~53 dB for Compressonator and ~46 dB for a BC5 MLP
(since removed), whose too-wide endpoint guesses collapsed low-contrast
blocks to their mean.

BC5 = two independent BC4 blocks, one for R and one for G (8 bytes each,
R block first). BC4 block layout:
  byte 0      : endpoint0 (8-bit)
  byte 1      : endpoint1 (8-bit)
  byte 2..7   : 16 palette indices, 3 bits each, LSB-first in pixel order
When endpoint0 > endpoint1 the 8-entry palette is, *in index order*,
  e0, e1, (6e0+e1)/7, (5e0+2e1)/7, ..., (e0+6e1)/7
i.e. index -> blend factor t (weight of e1) = [0, 1, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7].
endpoint0 <= endpoint1 selects the 6-entry + {0, 255} palette, which we never
emit (the packer swaps endpoints to keep e0 > e1, and collapses e0 == e1 to
index 0).
"""

from __future__ import annotations

import numpy as np
import torch

import ispc_texcomp as _it
import texture2ddecoder as _t2d
from bc7_codec import _decoded_endpoint_torch, _ls_endpoints

# The BC4 palette in *value* order is uniform: t = k/7 for k = 0..7. This LUT
# maps that level k to the bitstream index (0 -> 0, 7 -> 1, k -> k + 1).
_LEVEL_TO_INDEX = np.array([0, 2, 3, 4, 5, 6, 7, 1], dtype=np.int32)

# Index remap for t -> 1 - t (used when the endpoints have to be swapped so
# that e0 > e1): index 0 <-> 1, and k in 2..7 (t = (k-1)/7) -> 9 - k.
_SWAP_INDEX_LUT = np.array([1, 0, 7, 6, 5, 4, 3, 2], dtype=np.int32)


# ---------------------------------------------------------------------------
# Reference encoder / decoder.
# ---------------------------------------------------------------------------


def ispc_encode_bc5(rgba: np.ndarray) -> bytes:
    """Encode the R and G channels of an (H, W, 4) uint8 image to BC5 with
    ispc-texcomp. Its BC5 kernel reads a *2-channel* RG-interleaved surface
    (stride = 2 * width); handing it the RGBA buffer, as the BC7 path does,
    silently produces garbage (~9 dB)."""
    h, w = rgba.shape[:2]
    rg = np.ascontiguousarray(rgba[..., :2])
    surf = _it.RGBASurface(rg.tobytes(), w, h, w * 2)
    return _it.compress_blocks_bc5(surf)


def decode_bc5(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode BC5 bytes to an (H, W, 2) uint8 RG array. texture2ddecoder
    returns BGRA (R in channel 2, G in channel 1, B = 0)."""
    raw = _t2d.decode_bc5(data, width, height)
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return bgra[..., [2, 1]].copy()


def rg_to_rgb(rg01: np.ndarray) -> np.ndarray:
    """Reconstruct a displayable tangent-space normal map (float RGB in
    [0,1]) from its stored RG: z = sqrt(1 - x^2 - y^2), like a shader would
    after sampling a BC5 normal map. Used for viewing and for FLIP, which
    needs 3 channels."""
    xy = rg01.astype(np.float32) * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - np.sum(xy * xy, axis=-1, keepdims=True), 0.0, 1.0))
    return np.concatenate([rg01.astype(np.float32), z * 0.5 + 0.5], axis=-1)


# ---------------------------------------------------------------------------
# Refinement (see bc7_codec for the rationale).
#
# Unlike the BC7 lines, a BC4 line is 1-D and its palette is uniformly spaced
# in value, so the exact index search is a projection onto the segment
# followed by rounding -- no palette matmul, no argmin over K entries (on MPS
# those two alone cost ~25ms per 262k blocks per iteration). This is also
# exactly what the GPU kernel does.
# ---------------------------------------------------------------------------


def _best_level(pixels: torch.Tensor, e0: torch.Tensor, e1: torch.Tensor) -> torch.Tensor:
    """pixels (N,16), e0/e1 (N,) -> (N,16) float blend factor snapped to the
    nearest of the 8 uniform palette levels k/7 (t = 0 when e0 == e1)."""
    span = e1 - e0
    safe = torch.where(span.abs() > 1e-8, span, torch.ones_like(span))
    t = ((pixels - e0[:, None]) / safe[:, None]).clamp(0.0, 1.0)
    t = torch.where((span.abs() > 1e-8)[:, None], t, torch.zeros_like(t))
    return torch.round(t * 7.0) / 7.0


def _refine_bc4_line(
    pixels: torch.Tensor, e0: torch.Tensor, e1: torch.Tensor, iters: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pixels (N,16), e0/e1 (N,1) in [0,1]. Alternates projection-based index
    search with the least-squares endpoint refit. Returns (e0_decoded,
    e1_decoded, interp (N,16) = exact palette weights, recon (N,16))."""
    e0q, e1q = _decoded_endpoint_torch(e0[:, 0], 8), _decoded_endpoint_torch(e1[:, 0], 8)
    for _ in range(iters):
        t = _best_level(pixels, e0q, e1q)
        e0n, e1n = _ls_endpoints(pixels[..., None], t)
        e0q, e1q = _decoded_endpoint_torch(e0n[:, 0], 8), _decoded_endpoint_torch(e1n[:, 0], 8)
    t = _best_level(pixels, e0q, e1q)
    recon = e0q[:, None] + t * (e1q - e0q)[:, None]
    return e0q[:, None], e1q[:, None], t, recon


@torch.no_grad()
def refine_bc5(
    pixels: torch.Tensor,
    endpoint0_r: torch.Tensor,
    endpoint1_r: torch.Tensor,
    endpoint0_g: torch.Tensor,
    endpoint1_g: torch.Tensor,
    iters: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pixels (N,16,2) in [0,1]; initial endpoints (N,1) per channel.
    Returns (e0_r, e1_r, interp_r, e0_g, e1_g, interp_g, recon (N,16,2)) ready
    for pack_bc5_blocks_batch_torch_arr; the blend factors are exact palette
    weights and recon is what the decoder will produce."""
    e0_r, e1_r, t_r, recon_r = _refine_bc4_line(pixels[..., 0], endpoint0_r, endpoint1_r, iters)
    e0_g, e1_g, t_g, recon_g = _refine_bc4_line(pixels[..., 1], endpoint0_g, endpoint1_g, iters)
    return e0_r, e1_r, t_r, e0_g, e1_g, t_g, torch.stack([recon_r, recon_g], dim=-1)


@torch.no_grad()
def encode_bc5_blocks(pixels: torch.Tensor, iters: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """The BC5 encoder: pixels (N,16,2) in [0,1] -> (packed (N,16) uint8
    on-device, recon (N,16,2)). Endpoints start at the block's per-channel
    min/max and go through `iters` refinement rounds (see refine_bc5)."""
    lo, hi = pixels.amin(dim=1), pixels.amax(dim=1)  # (N,2)
    out = refine_bc5(pixels, lo[:, 0:1], hi[:, 0:1], lo[:, 1:2], hi[:, 1:2], iters)
    return pack_bc5_blocks_batch_torch_arr(*out[:6]), out[-1]


# ---------------------------------------------------------------------------
# Bit-exact packer (torch, on-device).
# ---------------------------------------------------------------------------


def _pack_bc4_torch(endpoint0: torch.Tensor, endpoint1: torch.Tensor, interp: torch.Tensor) -> torch.Tensor:
    """endpoint0/endpoint1 (N,1) in [0,1], interp (N,16) in [0,1] -> (N,8) uint8."""
    device = endpoint0.device
    e0 = torch.round(endpoint0[:, 0] * 255.0).clamp(0, 255).to(torch.int32)
    e1 = torch.round(endpoint1[:, 0] * 255.0).clamp(0, 255).to(torch.int32)
    # Blend factor -> uniform level k/7 -> bitstream index.
    level = torch.round(interp.clamp(0.0, 1.0) * 7.0).to(torch.int64)  # (N,16)
    idx = torch.as_tensor(_LEVEL_TO_INDEX, device=device)[level]

    # The 8-entry interpolated palette requires e0 > e1: swap the endpoints
    # (t -> 1 - t, remapped through the LUT) where needed. Equal endpoints
    # would select the 6-entry palette whose indices 6/7 decode to 0/255, so
    # force those blocks to index 0 (= e0 in either palette).
    lut = torch.as_tensor(_SWAP_INDEX_LUT, device=device)
    swap = e0 < e1
    idx = torch.where(swap[:, None], lut[idx.long()], idx)
    e0, e1 = torch.where(swap, e1, e0), torch.where(swap, e0, e1)
    idx = torch.where((e0 == e1)[:, None], torch.zeros_like(idx), idx)

    # 16 x 3-bit indices = 48 bits, LSB-first: pack pixels 0..7 and 8..15 into
    # two 24-bit words and split those into bytes.
    shifts = (3 * torch.arange(8, device=device, dtype=torch.int32))[None, :]
    lo = (idx[:, :8] << shifts).sum(dim=1)
    hi = (idx[:, 8:] << shifts).sum(dim=1)
    byte_shifts = (8 * torch.arange(3, device=device, dtype=torch.int32))[None, :]
    lo_bytes = (lo[:, None] >> byte_shifts) & 0xFF
    hi_bytes = (hi[:, None] >> byte_shifts) & 0xFF
    return torch.cat([e0[:, None], e1[:, None], lo_bytes, hi_bytes], dim=1).to(torch.uint8)


def pack_bc5_blocks_batch_torch_arr(
    endpoint0_r: torch.Tensor,
    endpoint1_r: torch.Tensor,
    interp_r: torch.Tensor,
    endpoint0_g: torch.Tensor,
    endpoint1_g: torch.Tensor,
    interp_g: torch.Tensor,
) -> torch.Tensor:
    """Pack N BC5 blocks; returns the (N, 16) uint8 tensor on-device
    (R block in bytes 0..7, G block in bytes 8..15)."""
    return torch.cat(
        [_pack_bc4_torch(endpoint0_r, endpoint1_r, interp_r), _pack_bc4_torch(endpoint0_g, endpoint1_g, interp_g)],
        dim=1,
    )


# ---------------------------------------------------------------------------
# Self-test: refine -> pack -> reference decode must reproduce `recon`.
# ---------------------------------------------------------------------------


def _self_test(image_path: str | None) -> None:
    from pathlib import Path

    from data import extract_blocks, load_rgba

    torch.manual_seed(0)
    sets = {"random": torch.rand(4096, 16, 2)}
    if image_path is not None:
        rgba = load_rgba(Path(image_path))
        blocks = extract_blocks(rgba)[..., :2]
        sets["image"] = torch.from_numpy(blocks.reshape(-1, 16, 2).astype(np.float32) / 255.0)

        h, w = (rgba.shape[0] // 4) * 4, (rgba.shape[1] // 4) * 4
        ref = decode_bc5(ispc_encode_bc5(rgba[:h, :w]), w, h).astype(np.float64)
        mse = np.mean((ref - rgba[:h, :w, :2]) ** 2)
        print(f"ISPC BC5 on {image_path}: RG PSNR {10 * np.log10(255.0**2 / mse):.2f}dB")

    for name, pixels in sets.items():
        n = pixels.shape[0]
        packed_arr, recon = encode_bc5_blocks(pixels, 2)
        packed = packed_arr.cpu().numpy().tobytes()

        grid = int(np.ceil(np.sqrt(n)))
        pad = grid * grid - n
        packed_np = np.frombuffer(packed, dtype=np.uint8).reshape(n, 16)
        packed_np = np.concatenate([packed_np, np.zeros((pad, 16), np.uint8)], axis=0)
        decoded = decode_bc5(packed_np.tobytes(), grid * 4, grid * 4)
        decoded_blocks = extract_blocks(np.concatenate([decoded, np.zeros((*decoded.shape[:2], 2), np.uint8)], -1))
        decoded_blocks = decoded_blocks[:n, ..., :2].reshape(n, 16, 2).astype(np.float64)

        recon255 = recon.numpy().astype(np.float64) * 255.0
        max_err = np.max(np.abs(decoded_blocks - recon255))
        mse = np.mean((decoded_blocks - pixels.numpy() * 255.0) ** 2)
        print(f"{name}: max |decoder - recon| = {max_err:.3f}, PSNR vs source {10 * np.log10(255.0**2 / max(mse, 1e-10)):.2f}dB")
        assert max_err <= 1.0, "packer does not reproduce the refined reconstruction"
    print("bc5 packer self-test OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BC5 packer self-test")
    parser.add_argument("--image", default=None, help="optional normal map to also round-trip through ISPC BC5")
    args = parser.parse_args()
    _self_test(args.image)
