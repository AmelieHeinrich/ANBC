"""BC7 mode-6 / mode-5 codec: differentiable soft-decode for training,
bit-exact packers (torch, on-device) and refinement for inference, and thin
wrappers around ispc-texcomp (ground truth encoder) and texture2ddecoder
(reference decoder).

Mode 6 layout (single subset, no partition, no rotation):
  bit 0..5   : 0 (unary mode prefix)
  bit 6      : 1 (mode terminator)
  bit 7..62  : endpoints, order R0 R1 G0 G1 B0 B1 A0 A1, 7 bits each (56 bits)
  bit 63..64 : p-bits P0 P1 (1 bit each)
  bit 65..127: 16 palette indices, anchor (index 0) uses 3 bits, the other
               15 use 4 bits, LSB-first bit-packed in pixel order (63 bits)
Total = 7 + 56 + 2 + 63 = 128 bits.
"""

from __future__ import annotations

import numpy as np
import torch

import ispc_texcomp as _it
import texture2ddecoder as _t2d

# BC7 4-bit index interpolation weights (standard ALPHA/COLOR weight table).
_WEIGHTS_4BIT = np.array(
    [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.float64
) / 64.0


_ISPC_BEST_SETTINGS = _it.BC7EncSettings.from_profile("alpha_basic")


def ispc_encode_best(rgba: np.ndarray) -> bytes:
    """Encode with ISPC's full mode search (all modes, alpha-aware), using
    the 'alpha_basic' profile -- a real-world quality/speed bar, not the
    exhaustive 'alpha_slow' research-grade search (which costs tens of
    seconds per 1024x1024 image). Used now that the neural side also does
    per-block mode selection (mode6 vs mode5) rather than being forced into
    a single mode."""
    h, w = rgba.shape[:2]
    surf = _it.RGBASurface(np.ascontiguousarray(rgba).tobytes(), w, h)
    return _it.compress_blocks_bc7(surf, _ISPC_BEST_SETTINGS)


def decode_bc7(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode BC7 bytes to an (H, W, 4) uint8 RGBA array. texture2ddecoder
    returns BGRA, so channels are swapped back to RGBA here."""
    raw = _t2d.decode_bc7(data, width, height)
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return bgra[..., [2, 1, 0, 3]].copy()


# ---------------------------------------------------------------------------
# Index quantization helpers shared by the packers and the soft-decoders.
#
# The packers used to map a blend factor t to an index with round(t * K); the
# soft-decoders used t as-is (continuous). That mismatch let the model learn
# blend factors that only work at continuous precision -- fatal for mode 5,
# whose 2-bit indices only reach 4 palette weights, so at pack time pixels
# snapped to whichever endpoint was nearest and showed up as bright speckles.
# Both sides now snap to the *nearest actual BC7 weight*, and training uses a
# straight-through estimator so gradients still flow through the snap.
# ---------------------------------------------------------------------------

_WEIGHTS_2BIT = np.array([0, 21, 43, 64], dtype=np.float64) / 64.0


def nearest_weight_index_torch(t: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    w = torch.as_tensor(weights, dtype=t.dtype, device=t.device)
    return torch.argmin(torch.abs(t.unsqueeze(-1) - w), dim=-1).to(torch.int32)


def _ste(x: torch.Tensor, x_quantized: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward value is x_quantized, gradient is
    passed through as if the quantizer were the identity."""
    return x + (x_quantized - x).detach()


def quantize_interp_ste(t: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    """Snap blend factors to the nearest BC7 palette weight (STE)."""
    w = torch.as_tensor(weights, dtype=t.dtype, device=t.device)
    idx = torch.argmin(torch.abs(t.unsqueeze(-1) - w), dim=-1)
    return _ste(t, w[idx])


def quantize_endpoint_ste(v: torch.Tensor, nbits: int) -> torch.Tensor:
    """Snap a [0,1] endpoint to what an nbits value (no p-bit) expands to on
    decode (MSB replication), matching the decoder. nbits=8 is plain 8-bit
    rounding."""
    max_v = (1 << nbits) - 1
    q = torch.round(torch.round(v * 255.0).clamp(0, 255) / 255.0 * max_v).clamp(0, max_v)
    if nbits < 8:
        q = q * (1 << (8 - nbits)) + torch.floor(q / (1 << (2 * nbits - 8)))
    return _ste(v, q / 255.0)


# ---------------------------------------------------------------------------
# Bit-exact mode-6 packer (torch, on-device) and the differentiable
# soft-decode used during training.
# ---------------------------------------------------------------------------


def quantize_endpoints_and_pbits_batch_torch(rgba_0to1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize float RGBA endpoints in [0,1] to 7-bit values + one shared
    p-bit per endpoint (mode 6): the achievable 8-bit color is (v7 << 1) | p,
    so p is chosen per endpoint to minimise the squared error over its 4
    channels. rgba_0to1: (N, 4). Returns (v7 (N,4) int32, p (N,) int32)."""
    y = torch.round(rgba_0to1 * 255.0).clamp(0, 255).to(torch.int32)  # (N,4)

    errs = []
    v7s = []
    for p in (0, 1):
        v7 = torch.round((y - p) / 2.0).clamp(0, 127).to(torch.int32)
        recon = (v7 << 1) | p
        errs.append(torch.sum((recon - y) ** 2, dim=-1))
        v7s.append(v7)

    errs_t = torch.stack(errs, dim=0)  # (2, N)
    v7s_t = torch.stack(v7s, dim=0)  # (2, N, 4)
    best_p = torch.argmin(errs_t, dim=0)  # (N,)
    best_v7 = torch.take_along_dim(v7s_t, best_p[None, :, None].expand(1, -1, 4), dim=0)[0]
    return best_v7, best_p.to(torch.int32)


def _bits_from_values_2d_torch(values: torch.Tensor, nbits: int) -> torch.Tensor:
    """values: (N, K) int32 tensor -> (N, K*nbits) uint8 tensor, LSB first."""
    n, k = values.shape
    shifts = torch.arange(nbits, device=values.device, dtype=torch.int32)
    bits = (values.unsqueeze(-1).to(torch.int64) >> shifts) & 1  # (N, K, nbits)
    return bits.reshape(n, k * nbits).to(torch.uint8)


def pack_mode6_blocks_batch_torch_arr(
    endpoint0_0to1: torch.Tensor,
    endpoint1_0to1: torch.Tensor,
    interp_0to1: torch.Tensor,
) -> torch.Tensor:
    """Pack N BC7 mode-6 blocks on-device; returns the (N, 16) uint8 tensor
    so the caller can mix mode6/mode5 blocks per block before the final
    tobytes() (see compare_ui's per-block mode selection). The anchor
    (pixel 0) index only has 3 storage bits: blocks whose anchor wants an
    index > 7 get their endpoints swapped and every index inverted, which
    reconstructs identically."""
    device = endpoint0_0to1.device
    n = endpoint0_0to1.shape[0]

    e0_7, p0 = quantize_endpoints_and_pbits_batch_torch(endpoint0_0to1)
    e1_7, p1 = quantize_endpoints_and_pbits_batch_torch(endpoint1_0to1)
    indices = nearest_weight_index_torch(interp_0to1, _WEIGHTS_4BIT)  # (N,16)

    swap_mask = indices[:, 0] > 7
    e0_7_swapped = torch.where(swap_mask[:, None], e1_7, e0_7)
    e1_7_swapped = torch.where(swap_mask[:, None], e0_7, e1_7)
    p0_swapped = torch.where(swap_mask, p1, p0)
    p1_swapped = torch.where(swap_mask, p0, p1)
    indices = torch.where(swap_mask[:, None], 15 - indices, indices)
    e0_7, e1_7, p0, p1 = e0_7_swapped, e1_7_swapped, p0_swapped, p1_swapped

    mode_bits = torch.zeros((n, 7), dtype=torch.uint8, device=device)
    mode_bits[:, 6] = 1

    endpoints_interleaved = torch.empty((n, 8), dtype=torch.int32, device=device)
    endpoints_interleaved[:, 0::2] = e0_7
    endpoints_interleaved[:, 1::2] = e1_7
    endpoint_bits = _bits_from_values_2d_torch(endpoints_interleaved, 7)

    pbits = _bits_from_values_2d_torch(torch.stack([p0, p1], dim=1), 1)
    anchor_bits = _bits_from_values_2d_torch(indices[:, 0:1], 3)
    rest_bits = _bits_from_values_2d_torch(indices[:, 1:], 4)

    bits = torch.cat([mode_bits, endpoint_bits, pbits, anchor_bits, rest_bits], dim=1)  # (N,128)
    byte_weights = (1 << torch.arange(8, device=device, dtype=torch.int32)).to(torch.uint8)
    packed = (bits.reshape(n, 16, 8).to(torch.int32) * byte_weights.to(torch.int32)).sum(dim=-1).to(torch.uint8)

    return packed


def soft_decode_block(
    endpoint0: torch.Tensor, endpoint1: torch.Tensor, interp: torch.Tensor
) -> torch.Tensor:
    """Differentiable BC7 mode-6-shaped reconstruction.

    endpoint0, endpoint1: (..., 4) in [0,1] (RGBA).
    interp: (..., 16) in [0,1], per-pixel blend factor.
    Returns (..., 16, 4) reconstructed RGBA in [0,1].

    Quantization-aware: indices snap to the 16 real 4-bit weights and
    endpoints to 8-bit (the shared p-bit makes every 8-bit value reachable
    to within 1/255), with straight-through gradients.
    """
    e0 = quantize_endpoint_ste(endpoint0, 8).unsqueeze(-2)  # (..., 1, 4)
    e1 = quantize_endpoint_ste(endpoint1, 8).unsqueeze(-2)  # (..., 1, 4)
    t = quantize_interp_ste(interp, _WEIGHTS_4BIT).unsqueeze(-1)  # (..., 16, 1)
    return e0 * (1.0 - t) + e1 * t


# ---------------------------------------------------------------------------
# BC7 mode 5: like mode 6, but RGB and A get *separate* index sets (their own
# per-pixel interpolation factor each), instead of one shared index for all
# 4 channels. This is what actually lets alpha decorrelate from color --
# mode 6 cannot represent a block where alpha and RGB vary independently,
# which is exactly the failure seen on real alpha-cutout textures (leaves,
# fences). We always emit rotation=0 (no channel swap); that's a legal mode
# 5 subset, just not using the rotation bits' full generality.
#
# Mode 5 layout (rotation=0, single subset, no partition, no p-bits):
#   bit 0..4    : 0 (unary mode prefix)
#   bit 5       : 1 (mode terminator)
#   bit 6..7    : rotation (00 = no rotation, always emitted here)
#   bit 8..49   : color endpoints, order R0 R1 G0 G1 B0 B1, 7 bits each (42 bits)
#   bit 50..65  : alpha endpoints A0 A1, 8 bits each (16 bits)
#   bit 66..96  : color indices, anchor 1 bit + 15x2 bits (31 bits)
#   bit 97..127 : alpha indices, anchor 1 bit + 15x2 bits (31 bits)
# Total = 6 + 2 + 42 + 16 + 31 + 31 = 128 bits.
# ---------------------------------------------------------------------------


def soft_decode_mode5(
    endpoint0_rgb: torch.Tensor,
    endpoint1_rgb: torch.Tensor,
    interp_rgb: torch.Tensor,
    endpoint0_a: torch.Tensor,
    endpoint1_a: torch.Tensor,
    interp_a: torch.Tensor,
) -> torch.Tensor:
    """Differentiable BC7 mode-5-shaped reconstruction with independent
    RGB/alpha blend factors.

    endpoint0_rgb/endpoint1_rgb: (..., 3) in [0,1].
    interp_rgb: (..., 16) in [0,1].
    endpoint0_a/endpoint1_a: (..., 1) in [0,1].
    interp_a: (..., 16) in [0,1].
    Returns (..., 16, 4) reconstructed RGBA in [0,1].

    Quantization-aware: indices snap to the 4 real 2-bit weights, RGB
    endpoints to 7-bit (MSB-replicated) and alpha endpoints to 8-bit, all
    with straight-through gradients. This matters far more here than in
    mode 6 -- with only 4 palette levels, a model trained on a continuous
    blend factor learns wide endpoints + fine-grained t, which the packer
    then snaps to the wrong endpoint (bright speckles / "holes").
    """
    e0_rgb = quantize_endpoint_ste(endpoint0_rgb, 7).unsqueeze(-2)
    e1_rgb = quantize_endpoint_ste(endpoint1_rgb, 7).unsqueeze(-2)
    t_rgb = quantize_interp_ste(interp_rgb, _WEIGHTS_2BIT).unsqueeze(-1)
    rgb = e0_rgb * (1.0 - t_rgb) + e1_rgb * t_rgb  # (..., 16, 3)

    e0_a = quantize_endpoint_ste(endpoint0_a, 8).unsqueeze(-2)
    e1_a = quantize_endpoint_ste(endpoint1_a, 8).unsqueeze(-2)
    t_a = quantize_interp_ste(interp_a, _WEIGHTS_2BIT).unsqueeze(-1)
    a = e0_a * (1.0 - t_a) + e1_a * t_a  # (..., 16, 1)

    return torch.cat([rgb, a], dim=-1)


def pack_mode5_blocks_batch_torch_arr(
    endpoint0_rgb: torch.Tensor,
    endpoint1_rgb: torch.Tensor,
    interp_rgb: torch.Tensor,
    endpoint0_a: torch.Tensor,
    endpoint1_a: torch.Tensor,
    interp_a: torch.Tensor,
) -> torch.Tensor:
    """Pack N BC7 mode-5 (rotation = 0) blocks on-device; returns the (N, 16)
    uint8 tensor. RGB endpoints are 7-bit and alpha 8-bit, no p-bits; the
    2-bit index sets' anchors only have 1 storage bit, so an anchor wanting
    index 2 or 3 swaps that set's endpoints and inverts its indices."""
    device = endpoint0_rgb.device
    n = endpoint0_rgb.shape[0]

    def quant_no_pbit(v01: torch.Tensor, nbits: int) -> torch.Tensor:
        y = torch.round(v01 * 255.0).clamp(0, 255)
        max_v = (1 << nbits) - 1
        return torch.round(y / 255.0 * max_v).clamp(0, max_v).to(torch.int32)

    e0_rgb = quant_no_pbit(endpoint0_rgb, 7)
    e1_rgb = quant_no_pbit(endpoint1_rgb, 7)
    e0_a = quant_no_pbit(endpoint0_a, 8)
    e1_a = quant_no_pbit(endpoint1_a, 8)

    color_idx = nearest_weight_index_torch(interp_rgb, _WEIGHTS_2BIT)
    alpha_idx = nearest_weight_index_torch(interp_a, _WEIGHTS_2BIT)

    color_swap = color_idx[:, 0] > 1
    color_idx = torch.where(color_swap[:, None], 3 - color_idx, color_idx)
    e0_rgb_s = torch.where(color_swap[:, None], e1_rgb, e0_rgb)
    e1_rgb_s = torch.where(color_swap[:, None], e0_rgb, e1_rgb)
    e0_rgb, e1_rgb = e0_rgb_s, e1_rgb_s

    alpha_swap = alpha_idx[:, 0] > 1
    alpha_idx = torch.where(alpha_swap[:, None], 3 - alpha_idx, alpha_idx)
    e0_a_s = torch.where(alpha_swap[:, None], e1_a, e0_a)
    e1_a_s = torch.where(alpha_swap[:, None], e0_a, e1_a)
    e0_a, e1_a = e0_a_s, e1_a_s

    mode_bits = torch.zeros((n, 6), dtype=torch.uint8, device=device)
    mode_bits[:, 5] = 1
    rotation_bits = torch.zeros((n, 2), dtype=torch.uint8, device=device)

    color_interleaved = torch.empty((n, 6), dtype=torch.int32, device=device)
    color_interleaved[:, 0::2] = e0_rgb
    color_interleaved[:, 1::2] = e1_rgb
    color_endpoint_bits = _bits_from_values_2d_torch(color_interleaved, 7)

    alpha_interleaved = torch.empty((n, 2), dtype=torch.int32, device=device)
    alpha_interleaved[:, 0] = e0_a[:, 0]
    alpha_interleaved[:, 1] = e1_a[:, 0]
    alpha_endpoint_bits = _bits_from_values_2d_torch(alpha_interleaved, 8)

    color_anchor_bits = _bits_from_values_2d_torch(color_idx[:, 0:1], 1)
    color_rest_bits = _bits_from_values_2d_torch(color_idx[:, 1:], 2)
    alpha_anchor_bits = _bits_from_values_2d_torch(alpha_idx[:, 0:1], 1)
    alpha_rest_bits = _bits_from_values_2d_torch(alpha_idx[:, 1:], 2)

    bits = torch.cat(
        [
            mode_bits,
            rotation_bits,
            color_endpoint_bits,
            alpha_endpoint_bits,
            color_anchor_bits,
            color_rest_bits,
            alpha_anchor_bits,
            alpha_rest_bits,
        ],
        dim=1,
    )
    byte_weights = (1 << torch.arange(8, device=device, dtype=torch.int32)).to(torch.uint8)
    packed = (bits.reshape(n, 16, 8).to(torch.int32) * byte_weights.to(torch.int32)).sum(dim=-1).to(torch.uint8)
    return packed


# ---------------------------------------------------------------------------
# Inference-time refinement.
#
# The network predicts endpoints *and* per-pixel blend factors in one shot,
# but given a pair of endpoints the optimal index for each pixel is a trivial
# exact search over the 4 (mode 5) or 16 (mode 6) palette entries -- no
# learning needed, and the net's sigmoid guess frequently lands on the wrong
# side of a snapping boundary, which is the per-pixel speckle you see when
# zooming in. Conversely, given the indices, the optimal endpoints are a
# closed-form least-squares fit. So we treat the net's output purely as an
# initial guess and run a couple of rounds of the classic encoder loop:
#
#     indices   = nearest palette entry to each pixel (exact)
#     endpoints = least-squares refit given those indices
#     endpoints = snap to what the bitstream can actually store
#
# Everything is batched torch so it stays on-device and costs a few small
# kernels per iteration. The returned blend factors are *exactly* the BC7
# palette weights, so the packers' nearest_weight_index_torch() recovers the
# chosen indices bit-for-bit, and the returned reconstruction is what the decoder
# will produce (up to its integer rounding), which is what per-block mode
# selection should be comparing.
# ---------------------------------------------------------------------------


def _decoded_endpoint_torch(v01: torch.Tensor, nbits: int) -> torch.Tensor:
    """Forward half of quantize_endpoint_ste: the [0,1] value the decoder
    will actually see for an nbits (no p-bit) endpoint."""
    max_v = (1 << nbits) - 1
    q = torch.round(torch.round(v01 * 255.0).clamp(0, 255) / 255.0 * max_v).clamp(0, max_v)
    if nbits < 8:
        q = q * (1 << (8 - nbits)) + torch.floor(q / (1 << (2 * nbits - 8)))
    return q / 255.0


def _decoded_endpoint_pbit_torch(rgba01: torch.Tensor) -> torch.Tensor:
    """Same for a mode-6 endpoint (7 bits + shared p-bit)."""
    v7, p = quantize_endpoints_and_pbits_batch_torch(rgba01)
    return ((v7 << 1) | p[:, None]).to(rgba01.dtype) / 255.0


def _palette(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """e0/e1 (N,C), w (K,) -> (N,K,C)."""
    return e0[:, None, :] * (1.0 - w)[None, :, None] + e1[:, None, :] * w[None, :, None]


def _best_indices(pixels: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """pixels (N,16,C), palette (N,K,C) -> (N,16) int64 index of the closest
    palette entry per pixel. Uses ||p-q||^2 = ||p||^2 - 2p.q + ||q||^2 and
    drops the ||p||^2 term (constant per pixel) so the only big intermediate
    is the (N,16,K) score tensor."""
    dots = torch.bmm(pixels, palette.transpose(1, 2))  # (N,16,K)
    norms = (palette * palette).sum(-1)  # (N,K)
    return torch.argmin(norms[:, None, :] - 2.0 * dots, dim=-1)


def _ls_endpoints(pixels: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares endpoints for fixed per-pixel blend factors.

    pixels (N,16,C), w (N,16) in [0,1]. Minimizes
    sum_i ||(1-w_i) e0 + w_i e1 - p_i||^2 via the 2x2 normal equations
    (shared across channels). Blocks whose pixels all share one index are
    degenerate (any e0/e1 pair with the right blend gives the same color),
    so those get e0 = e1 = mean.
    """
    a = 1.0 - w
    b = w
    aa = (a * a).sum(1)
    ab = (a * b).sum(1)
    bb = (b * b).sum(1)
    ap = (a[..., None] * pixels).sum(1)  # (N,C)
    bp = (b[..., None] * pixels).sum(1)
    det = aa * bb - ab * ab
    # Degeneracy test must be *relative*: for a flat block (all pixels share
    # one index) det is mathematically 0 but comes out as float32
    # cancellation noise of either sign (~1e-6 with aa*bb ~ 1e2). An absolute
    # 1e-8 threshold then depends on the rounding of the device (CPU vs
    # MPS/CUDA gave different signs) and, when it passes, divides noise by
    # noise -> endpoints like 2.0. det/(aa*bb) = 1 - cos^2(a, b) is ~1 for
    # any block with two distinct blend factors and ~0 only when degenerate.
    ok = det > 1e-4 * aa * bb
    det_safe = torch.where(ok, det, torch.ones_like(det))[:, None]
    e0 = (bb[:, None] * ap - ab[:, None] * bp) / det_safe
    e1 = (aa[:, None] * bp - ab[:, None] * ap) / det_safe
    mean = pixels.mean(1)
    e0 = torch.where(ok[:, None], e0, mean).clamp(0.0, 1.0)
    e1 = torch.where(ok[:, None], e1, mean).clamp(0.0, 1.0)
    return e0, e1


def _refine_line(
    pixels: torch.Tensor,
    e0: torch.Tensor,
    e1: torch.Tensor,
    weights: np.ndarray,
    quantize,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alternate exact index search / LS endpoint refit for one index set.
    Returns (e0_decoded, e1_decoded, interp (N,16) = exact palette weights,
    recon (N,16,C))."""
    w = torch.as_tensor(weights, dtype=pixels.dtype, device=pixels.device)
    e0q, e1q = quantize(e0), quantize(e1)
    for _ in range(iters):
        idx = _best_indices(pixels, _palette(e0q, e1q, w))
        e0, e1 = _ls_endpoints(pixels, w[idx])
        e0q, e1q = quantize(e0), quantize(e1)
    palette = _palette(e0q, e1q, w)
    idx = _best_indices(pixels, palette)
    recon = torch.gather(palette, 1, idx[..., None].expand(-1, -1, palette.shape[-1]))
    return e0q, e1q, w[idx], recon


@torch.no_grad()
def refine_mode6(
    pixels: torch.Tensor,
    endpoint0: torch.Tensor,
    endpoint1: torch.Tensor,
    iters: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pixels (N,16,4) in [0,1]; endpoint0/1 (N,4) from the model.
    Returns (endpoint0, endpoint1, interp (N,16), recon (N,16,4)) ready for
    pack_mode6_blocks_batch_torch_arr."""
    return _refine_line(pixels, endpoint0, endpoint1, _WEIGHTS_4BIT, _decoded_endpoint_pbit_torch, iters)


@torch.no_grad()
def refine_mode5(
    pixels: torch.Tensor,
    endpoint0_rgb: torch.Tensor,
    endpoint1_rgb: torch.Tensor,
    endpoint0_a: torch.Tensor,
    endpoint1_a: torch.Tensor,
    iters: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pixels (N,16,4) in [0,1]; endpoints from the model.
    Returns (e0_rgb, e1_rgb, interp_rgb, e0_a, e1_a, interp_a, recon (N,16,4))
    ready for pack_mode5_blocks_batch_torch_arr."""
    rgb, a = pixels[..., :3], pixels[..., 3:]
    e0_rgb, e1_rgb, t_rgb, recon_rgb = _refine_line(
        rgb, endpoint0_rgb, endpoint1_rgb, _WEIGHTS_2BIT, lambda v: _decoded_endpoint_torch(v, 7), iters
    )
    e0_a, e1_a, t_a, recon_a = _refine_line(
        a, endpoint0_a, endpoint1_a, _WEIGHTS_2BIT, lambda v: _decoded_endpoint_torch(v, 8), iters
    )
    return e0_rgb, e1_rgb, t_rgb, e0_a, e1_a, t_a, torch.cat([recon_rgb, recon_a], dim=-1)
