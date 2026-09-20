"""BC6H (unsigned, UF16) codec: mode-11 encoder (block min/max endpoints +
exact-index / least-squares refinement, bit-exact packer), a full 14-mode
reference decoder and a wrapper around ispc-texcomp (ground truth). Built
from the same generic pieces as bc7_codec.

No network, like BC5. BC6H mode 11 (single subset, 10-bit RGB endpoints,
4-bit indices, no partition) is structurally BC7 mode 6 -- fit one 3-D line
through the 16 pixels -- so a mode-11 MLP was tried (per-block normalised
input, exact-index loss). Its endpoints *were* better than the block's
bounding box before refinement (43.7 vs 42.6 dB over 20 HDR validation
images), but after the 2 standard least-squares refinement rounds both
landed in the same place (45.67 vs 45.72 dB), so the encoder starts from the
per-channel min/max -- what the fast GPU BC6H encoders do -- and the network
was removed. The two-subset modes (32 partitions x 9 delta layouts) are out
of scope, as they are for the BC7 side; ISPC's full mode search is ~4 dB
ahead mostly because of the higher-precision single-subset modes 12/13.

Encoding domain: BC6H interpolates in the *half-float bit pattern as an
integer* (unsigned: 0..0x7BFF = 65504.0), which is piecewise-log. Every pixel
here is x = halfbits(v) / 0x7BFF in [0,1]; negative inputs clamp to 0 (the
format is unsigned), inf/NaN to 0x7BFF. An L1/L2 loss in this domain is
~relative error, which is the right metric for HDR. Since it is a linear
rescale of the codec's own integer space, bc7_codec's refinement machinery
(_refine_line / _ls_endpoints) is reused unchanged.

Mode 11 reference (LSB-first, 128 bits):
  bit 0..4    : mode 00011 (value 3)
  bit 5..64   : endpoints e0.r e0.g e0.b e1.r e1.g e1.b, 10 bits each (NOT
                interleaved like BC7)
  bit 65..127 : 16 indices, anchor (pixel 0) 3 bits, the rest 4 bits
Decoder (D3D11.3 spec / DirectXTex):
  quantize   q   = (x << 10) / 0x7C00              (x = half bits, floor)
  unquantize unq = 0 if q == 0, 0xFFFF if q == 1023, else (q << 16 | 0x8000) >> 10
  palette    v   = (unq0 * (64 - w) + unq1 * w + 32) >> 6, w = BC7 4-bit weights
  output     halfbits = (v * 31) >> 6
"""

from __future__ import annotations

import numpy as np
import torch

import ispc_texcomp as _it
from bc6h_tables import FIXUP2, MODE_DESC, MODE_INFO, PARTITION2
from bc7_codec import _WEIGHTS_4BIT, _bits_from_values_2d_torch, _refine_line, nearest_weight_index_torch

HALF_MAX = 0x7BFF  # largest finite positive half (65504.0) as bits
# Integer palette weights (x64) of the 3-bit (two-subset modes) and 4-bit index sets.
_WEIGHTS_INT = {3: np.array([0, 9, 18, 27, 37, 46, 55, 64]), 4: np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64])}

# BC6H mode bits -> DirectXTex mode index (row of MODE_DESC / MODE_INFO); 10 is mode 11.
_MODE_TO_INFO = {0x00: 0, 0x01: 1, 0x02: 2, 0x06: 3, 0x0A: 4, 0x0E: 5, 0x12: 6, 0x16: 7, 0x1A: 8, 0x1E: 9,
                 0x03: 10, 0x07: 11, 0x0B: 12, 0x0F: 13}


# ---------------------------------------------------------------------------
# Half-int domain conversions.
# ---------------------------------------------------------------------------


def float_to_halfint(v: np.ndarray) -> np.ndarray:
    """float RGB -> half bit pattern as int32 in [0, 0x7BFF] (unsigned
    BC6H: negatives clamp to 0, values above 65504 / inf / NaN to 0x7BFF).
    Clipped before the half cast only to keep numpy's overflow warning quiet
    (inf would clamp to the same 0x7BFF)."""
    clipped = np.clip(np.asarray(v, dtype=np.float32), 0.0, 65504.0)
    bits = clipped.astype(np.float16).view(np.int16).astype(np.int32)
    return np.clip(bits, 0, HALF_MAX)


def halfint_to_float(h: np.ndarray) -> np.ndarray:
    return np.asarray(h).astype(np.int16).view(np.float16).astype(np.float32)


def float_to_norm(v: np.ndarray) -> np.ndarray:
    """float RGB -> the [0,1] half-int domain everything here works in."""
    return float_to_halfint(v).astype(np.float32) / HALF_MAX


# ---------------------------------------------------------------------------
# Endpoint quantization (what the decoder will actually see).
# ---------------------------------------------------------------------------


def _quantize10_torch(v01: torch.Tensor) -> torch.Tensor:
    """[0,1] half-int domain -> 10-bit mode-11 endpoint (int32)."""
    x = torch.round(v01 * HALF_MAX).clamp(0, HALF_MAX).to(torch.int32)
    return torch.div(x * 1024, 0x7C00, rounding_mode="floor")


def _decoded10_torch(q: torch.Tensor) -> torch.Tensor:
    """10-bit endpoint -> the [0,1] value the decoder reconstructs for it."""
    unq = torch.div(q * 65536 + 0x8000, 1024, rounding_mode="floor")
    unq = torch.where(q == 0, torch.zeros_like(unq), torch.where(q == 1023, torch.full_like(unq, 0xFFFF), unq))
    return torch.div(unq * 31, 64, rounding_mode="floor").float() / HALF_MAX


def _decoded_endpoint_bc6h_torch(v01: torch.Tensor) -> torch.Tensor:
    return _decoded10_torch(_quantize10_torch(v01))


# ---------------------------------------------------------------------------
# Refinement + encoder (see bc7_codec for the rationale).
# ---------------------------------------------------------------------------


@torch.no_grad()
def refine_bc6h(
    pixels: torch.Tensor, endpoint0: torch.Tensor, endpoint1: torch.Tensor, iters: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pixels (N,16,3) in the half-int domain; endpoint0/1 (N,3) initial
    guesses. Returns (e0, e1, interp (N,16) = exact palette weights, recon
    (N,16,3)) ready for pack_mode11_blocks_batch_torch_arr."""
    return _refine_line(pixels, endpoint0, endpoint1, _WEIGHTS_4BIT, _decoded_endpoint_bc6h_torch, iters)


@torch.no_grad()
def encode_bc6h_blocks(pixels: torch.Tensor, iters: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """The BC6H encoder: pixels (N,16,3) in the half-int domain -> (packed
    (N,16) uint8 on-device, recon (N,16,3)). Endpoints start at the block's
    per-channel min/max (bounding-box diagonal; this is what the GPU kernel
    does) and go through `iters` refinement rounds (see refine_bc6h)."""
    e0, e1 = pixels.amin(dim=1), pixels.amax(dim=1)
    e0, e1, interp, recon = refine_bc6h(pixels, e0, e1, iters)
    return pack_mode11_blocks_batch_torch_arr(e0, e1, interp), recon


# ---------------------------------------------------------------------------
# Bit-exact mode-11 packer (torch, on-device).
# ---------------------------------------------------------------------------


def pack_mode11_blocks_batch_torch_arr(
    endpoint0: torch.Tensor, endpoint1: torch.Tensor, interp: torch.Tensor
) -> torch.Tensor:
    """endpoint0/1 (N,3) in [0,1], interp (N,16) in [0,1] -> (N,16) uint8."""
    device = endpoint0.device
    n = endpoint0.shape[0]
    q0, q1 = _quantize10_torch(endpoint0), _quantize10_torch(endpoint1)
    indices = nearest_weight_index_torch(interp, _WEIGHTS_4BIT)  # (N,16)

    # The anchor index only stores 3 bits: swap endpoints and invert if needed.
    swap = indices[:, 0] > 7
    q0, q1 = torch.where(swap[:, None], q1, q0), torch.where(swap[:, None], q0, q1)
    indices = torch.where(swap[:, None], 15 - indices, indices)

    mode_bits = torch.zeros((n, 5), dtype=torch.uint8, device=device)
    mode_bits[:, 0] = 1
    mode_bits[:, 1] = 1
    endpoint_bits = _bits_from_values_2d_torch(torch.cat([q0, q1], dim=1), 10)  # (N,60)
    anchor_bits = _bits_from_values_2d_torch(indices[:, 0:1], 3)
    rest_bits = _bits_from_values_2d_torch(indices[:, 1:], 4)
    bits = torch.cat([mode_bits, endpoint_bits, anchor_bits, rest_bits], dim=1)  # (N,128)
    byte_weights = (1 << torch.arange(8, device=device, dtype=torch.int32))
    return (bits.reshape(n, 16, 8).to(torch.int32) * byte_weights).sum(dim=-1).to(torch.uint8)


# ---------------------------------------------------------------------------
# Reference decoder (all 14 modes, unsigned) and reference encoder.
#
# texture2ddecoder.decode_bc6 only returns clamped 8-bit BGRA, so this is
# our own port of the DirectXTex decoder: the bit layouts live in
# bc6h_tables.py (generated from BC6HBC7.cpp's ms_aDesc / partition / fix-up
# tables). Blocks are decoded vectorised per (mode, shape) group.
# ---------------------------------------------------------------------------


def _sign_extend(x: np.ndarray, nbits: np.ndarray | int) -> np.ndarray:
    return np.where(x & (1 << (nbits - 1)), x - (1 << nbits), x)


def _unquantize(comp: np.ndarray, prec: int) -> np.ndarray:
    if prec >= 15:
        return comp
    unq = ((comp << 16) + 0x8000) >> prec
    return np.where(comp == 0, 0, np.where(comp == (1 << prec) - 1, 0xFFFF, unq))


def decode_bc6h_blocks(blocks: np.ndarray) -> np.ndarray:
    """(N,16) uint8 BC6H blocks -> (N,16,3) int32 half bits (unsigned)."""
    n = blocks.shape[0]
    bits = np.unpackbits(blocks.reshape(n, 16), axis=1, bitorder="little").astype(np.int32)  # (N,128)
    out = np.zeros((n, 16, 3), dtype=np.int32)

    mode2 = bits[:, 0] | (bits[:, 1] << 1)
    mode5 = mode2 | (bits[:, 2] << 2) | (bits[:, 3] << 3) | (bits[:, 4] << 4)
    mode = np.where(mode2 < 2, mode2, mode5)

    for mode_bits, m in _MODE_TO_INFO.items():
        sel = np.nonzero(mode == mode_bits)[0]
        if sel.size == 0:
            continue
        b = bits[sel]
        _, partitions, transformed, index_bits, base_prec, delta_prec = MODE_INFO[m]
        header_bits = 82 if partitions else 65

        # Gather every header field: fields[f] = sum of bit p << position.
        fields = np.zeros((15, sel.size), dtype=np.int32)
        for p in range(header_bits):
            f, pos = MODE_DESC[m][p]
            if f >= 2:
                fields[f] |= b[:, p] << pos
        shape = fields[2]
        # endpoints[subset][A/B][channel]; RW RX RY RZ = 3..6 etc.
        ep = np.zeros((2, 2, 3, sel.size), dtype=np.int32)
        for c in range(3):
            ep[0, 0, c] = fields[3 + 4 * c]
            ep[0, 1, c] = fields[4 + 4 * c]
            ep[1, 0, c] = fields[5 + 4 * c]
            ep[1, 1, c] = fields[6 + 4 * c]
        if transformed:
            for c in range(3):
                mask = (1 << base_prec[c]) - 1
                for s, e in ((0, 1), (1, 0), (1, 1)):
                    d = _sign_extend(ep[s, e, c], delta_prec[c])
                    ep[s, e, c] = (ep[0, 0, c] + d) & mask
        unq = np.stack([_unquantize(ep[..., c, :], base_prec[c]) for c in range(3)], axis=2)  # (2,2,3,n)

        # Indices: bit positions depend on the anchors, i.e. on the shape.
        weights = _WEIGHTS_INT[index_bits]
        shapes = np.unique(shape) if partitions else np.array([0])
        for sh in shapes:
            rows = np.nonzero(shape == sh)[0] if partitions else np.arange(sel.size)
            anchors = {0, FIXUP2[sh][1]} if partitions else {0}
            subset = np.array(PARTITION2[sh]) if partitions else np.zeros(16, dtype=np.int32)
            pos = header_bits
            for i in range(16):
                nb = index_bits - (1 if i in anchors else 0)
                idx = np.zeros(rows.size, dtype=np.int32)
                for k in range(nb):
                    idx |= b[rows, pos + k] << k
                pos += nb
                w = weights[np.clip(idx, 0, len(weights) - 1)]
                a = unq[subset[i], 0][:, rows]  # (3, r)
                c = unq[subset[i], 1][:, rows]
                v = (a * (64 - w) + c * w + 32) >> 6
                out[sel[rows], i] = ((v * 31) >> 6).T
    return out


def decode_bc6h(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode BC6H (UF16) bytes to an (H, W, 3) float32 linear RGB image."""
    bx, by = (width + 3) // 4, (height + 3) // 4
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(bx * by, 16)
    px = halfint_to_float(decode_bc6h_blocks(blocks))  # (N,16,3)
    img = px.reshape(by, bx, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(by * 4, bx * 4, 3)
    return np.ascontiguousarray(img[:height, :width])


_ISPC_BC6H_SETTINGS = _it.BC6HEncSettings.from_profile("basic")


def ispc_encode_bc6h(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) float32 image to BC6H with ispc-texcomp's full
    mode search ('basic' profile, the same speed/quality bar as the BC7
    side's 'alpha_basic'). ISPC reads an RGBA *half* surface, stride w*8."""
    h, w = rgb.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float16)
    rgba[..., :3] = np.clip(rgb, 0.0, 65504.0).astype(np.float16)
    surf = _it.RGBASurface(rgba.tobytes(), w, h, w * 8)
    return _it.compress_blocks_bc6h(surf, _ISPC_BC6H_SETTINGS)


# ---------------------------------------------------------------------------
# Metrics / display.
# ---------------------------------------------------------------------------


def hdr_psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR between two float RGB images in the normalised half-int domain
    (0..0x7BFF -> 0..1), i.e. ~relative error in dB."""
    mse = np.mean((float_to_norm(a).astype(np.float64) - float_to_norm(b).astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def tonemap_for_display(rgb: np.ndarray) -> np.ndarray:
    """Linear HDR -> uint8 sRGB for viewers: Reinhard x/(1+x) + sRGB curve."""
    x = np.clip(rgb, 0.0, None).astype(np.float32)
    x = x / (1.0 + x)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Self-test: refine -> pack -> reference decode must reproduce `recon`.
# ---------------------------------------------------------------------------


def _self_test(image_path: str | None) -> None:
    from pathlib import Path

    from data import extract_blocks, load_hdr

    torch.manual_seed(0)
    sets = {"random": torch.rand(4096, 16, 3)}
    if image_path is not None:
        rgb = load_hdr(Path(image_path))
        h, w = (rgb.shape[0] // 4) * 4, (rgb.shape[1] // 4) * 4
        rgb = rgb[:h, :w]
        sets["image"] = torch.from_numpy(extract_blocks(float_to_norm(rgb)).reshape(-1, 16, 3))

        ref = decode_bc6h(ispc_encode_bc6h(rgb), w, h)
        print(f"ISPC BC6H on {image_path}: PSNR (half-int) {hdr_psnr(ref, rgb):.2f}dB")

    for name, pixels in sets.items():
        packed_arr, recon = encode_bc6h_blocks(pixels, 2)
        decoded = decode_bc6h_blocks(packed_arr.cpu().numpy()).astype(np.float64)  # (N,16,3) half bits
        recon_bits = recon.numpy().astype(np.float64) * HALF_MAX
        max_err = np.max(np.abs(decoded - recon_bits))
        mse = np.mean((decoded / HALF_MAX - pixels.numpy()) ** 2)
        print(f"{name}: max |decoder - recon| = {max_err:.2f} half units, PSNR (half-int) vs source {10 * np.log10(1.0 / max(mse, 1e-12)):.2f}dB")
        assert max_err <= 3.0, "packer does not reproduce the refined reconstruction"
    print("bc6h packer self-test OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BC6H packer self-test")
    parser.add_argument("--image", default=None, help="optional .hdr to also round-trip through ISPC BC6H")
    args = parser.parse_args()
    _self_test(args.image)
