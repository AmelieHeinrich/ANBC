"""ASTC 4x4 codec: a generic single-partition block configuration
(colour endpoint mode x weight range x optional dual plane), the exact-index /
least-squares refinement and per-block config selection of the BCn codecs,
a bit-exact packer (torch, batched), a reference decoder for what we emit,
and wrappers around texture2ddecoder (LDR) and the vendored astcenc CLI
(reference encoder, exact decoder incl. HDR). No network: the per-kind MLPs
did not beat the block min/max init after refinement (see notes.txt).

Three texture kinds ("variants"), each with a shortlist of configs the
encoder picks from per block by squared error (like BC7 mode 6 vs 5):
  GAME   RGBA8 pixels (N,16,4). GAME_A = CEM 8 (RGB direct) with 4-bit
         weights -> 47 endpoint bits -> QUANT_192 (trit) endpoints; GAME_B =
         CEM 12 (RGBA direct) for blocks with alpha.
  NORMAL (X,Y) pixels (N,16,2) stored as CEM 4 luminance + alpha (astcenc's
         -normal convention, sample .ra): X in L, Y in A. NORMAL_B puts the
         Y weights in a second plane (like BC5's two independent lines).
  FLOAT  HDR RGB (N,16,3) as CEM 11 (HDR RGB) with 3-bit weights so the six
         endpoint values get the full 8 bits QUANT_256 (astcenc's HDR RGB
         packing works on 8-bit values). Everything happens in ASTC's LNS
         domain: x = lns16 / 65535, a piecewise-linear log2 (2048 per
         octave) that the decoder converts to half floats.

Block layout (LSB-first, bit 0 = byte 0 bit 0), single partition:
  bits 0..10   block mode (4x4 weight grid: bits 6:5 = 10, bits 8:7 = 00,
               bits 3:2 = 00; R2 R1 = bits 1:0, R0 = bit 4 select the weight
               range; bit 9 = H, bit 10 = D (dual plane))
  bits 11..12  partition count - 1 = 0
  bits 13..16  colour endpoint mode (4, 8, 11 or 12)
  bit 17..     endpoint ISE (value 0 first, LSB first), QUANT level derived
               by the decoder from the bits left (astc_ise.derive_endpoint_quant)
  128-W-2..    dual plane only: 2-bit colour component of plane 2 (CCS)
  bit 127 down weight ISE, stream bit k stored at block bit 127 - k
Decode: e16 = e8 * 257 (LDR) or the CEM 11 LNS value; c16 = (e0_16 * (64-w)
+ e1_16 * w + 32) >> 6; unorm8 = c16 >> 8, HDR: lns_to_sf16(c16). CEM 8/12
decoders swap endpoints (and blue-contract) when sum(rgb0) > sum(rgb1), so
the packer keeps sum(rgb0) <= sum(rgb1) by swapping + inverting weights.
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

import texture2ddecoder as _t2d

import astc_ise as ise
from bc7_codec import _best_indices, _ls_endpoints, _palette
from bc6h_codec import HALF_MAX, float_to_halfint, halfint_to_float

ROOT = Path(__file__).resolve().parent.parent
LNS_MAX = 65535
LNS_MAX_FINITE = 63487  # LNS of the largest finite half (0x7BFF); 65535 is infinity
CEM_LA, CEM_RGB, CEM_HDR_RGB, CEM_RGBA = 4, 8, 11, 12


# ---------------------------------------------------------------------------
# Block configuration.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockConfig:
    name: str
    cem: int
    weight_levels: int  # bits-only weight range: 4, 8, 16 or 32
    dual_plane: bool = False
    ccs: int = 0  # ASTC colour component of the second weight plane (3 = alpha)

    @cached_property
    def weight_bits(self) -> int:
        return ise.QUANT_INFO[self.weight_levels][0]

    @cached_property
    def n_weights(self) -> int:
        return 32 if self.dual_plane else 16

    @cached_property
    def weight_stream_bits(self) -> int:
        bits = self.n_weights * self.weight_bits
        assert 24 <= bits <= 96, f"{self.name}: {bits} weight bits (ASTC allows 24..96)"
        return bits

    @cached_property
    def color_bits(self) -> int:
        return 128 - 17 - self.weight_stream_bits - (2 if self.dual_plane else 0)

    @cached_property
    def n_values(self) -> int:
        return {CEM_LA: 4, CEM_RGB: 6, CEM_HDR_RGB: 6, CEM_RGBA: 8}[self.cem]

    @cached_property
    def endpoint_levels(self) -> int:
        return ise.derive_endpoint_quant(self.n_values, self.color_bits)

    @cached_property
    def hdr(self) -> bool:
        return self.cem == CEM_HDR_RGB

    @cached_property
    def channels(self) -> list[int]:
        """Indices into the variant's pixel tensor of the channels this CEM
        stores: GAME (RGBA) -> RGB or RGBA, NORMAL (X,Y) -> (L,A), FLOAT -> RGB."""
        return {CEM_LA: [0, 1], CEM_RGB: [0, 1, 2], CEM_HDR_RGB: [0, 1, 2], CEM_RGBA: [0, 1, 2, 3]}[self.cem]

    @cached_property
    def plane2_channel(self) -> int:
        """Position of the CCS channel within `channels` (dual plane only)."""
        assert self.dual_plane
        astc_to_local = {CEM_LA: {0: 0, 3: 1}, CEM_RGBA: {0: 0, 1: 1, 2: 2, 3: 3}, CEM_RGB: {0: 0, 1: 1, 2: 2}}
        return astc_to_local[self.cem][self.ccs]

    @cached_property
    def block_mode(self) -> int:
        h, r = {4: (0, 4), 8: (0, 7), 16: (1, 4), 32: (1, 7)}[self.weight_levels]
        r0, r1, r2 = r & 1, (r >> 1) & 1, (r >> 2) & 1
        return (r1 | (r2 << 1)) | (r0 << 4) | (2 << 5) | (h << 9) | (int(self.dual_plane) << 10)

    @cached_property
    def weights01(self) -> np.ndarray:
        return ise.weight_unquant_table(self.weight_levels).astype(np.float64) / 64.0

    @cached_property
    def endpoint_lut(self) -> np.ndarray:
        """(256,) the decoded 8-bit value each 8-bit value snaps to (LDR)."""
        unq, code_of_value = ise.endpoint_unquant_table(self.endpoint_levels)
        return unq[code_of_value].astype(np.int64)

    def describe(self) -> str:
        return (f"{self.name}: CEM {self.cem}, {self.weight_bits}-bit weights"
                f"{' x2 planes (CCS ' + str(self.ccs) + ')' if self.dual_plane else ''}, "
                f"{self.color_bits} endpoint bits -> QUANT_{self.endpoint_levels}, block mode 0x{self.block_mode:03x}")


GAME_A = BlockConfig("GAME_A", CEM_RGB, 16)
GAME_A3 = BlockConfig("GAME_A3", CEM_RGB, 8)
GAME_B = BlockConfig("GAME_B", CEM_RGBA, 4, dual_plane=True, ccs=3)
GAME_B1 = BlockConfig("GAME_B1", CEM_RGBA, 16)
GAME_B3 = BlockConfig("GAME_B3", CEM_RGBA, 8)
NORMAL_A = BlockConfig("NORMAL_A", CEM_LA, 16)
NORMAL_B = BlockConfig("NORMAL_B", CEM_LA, 4, dual_plane=True, ccs=3)
NORMAL_C = BlockConfig("NORMAL_C", CEM_LA, 32)
# CEM 11 packs 8-bit values with mode bits in fixed positions, so HDR needs
# QUANT_256 endpoints: 4-bit weights (QUANT_192) would need astcenc's
# "re-quantize but keep the top bits" dance and are not supported.
FLOAT_A = BlockConfig("FLOAT_A", CEM_HDR_RGB, 8)

ALL_CONFIGS = [GAME_A, GAME_A3, GAME_B, GAME_B1, GAME_B3, NORMAL_A, NORMAL_B, NORMAL_C, FLOAT_A]
# The shipped shortlists (what the Metal kernels implement), chosen on the
# validation subsets with min/max init (mean PSNR over 20 GAME / 8 NORMAL /
# 8 FLOAT images, astcenc -medium in brackets): GAME A+B 52.3, A+A3+B 54.3
# (A3's 8-bit endpoints win on smooth textures) [58.0]; NORMAL B alone
# 44.1, A+B 45.1 [47.7]; FLOAT 43.0 [47.7]. The 4x4 candidates not kept:
# GAME_B1/B3 (single-plane RGBA) lose to the dual-plane B on cutouts and to
# A/A3 elsewhere, NORMAL_C's 5-bit weights starve the endpoints.
VARIANT_CONFIGS: dict[str, list[BlockConfig]] = {
    "game": [GAME_A, GAME_A3, GAME_B],
    "normal": [NORMAL_A, NORMAL_B],
    "float": [FLOAT_A],
}
VARIANT_CHANNELS = {"game": 4, "normal": 2, "float": 3}


# ---------------------------------------------------------------------------
# HDR: LNS <-> half conversions (astcenc lns_to_sf16 and its exact inverse).
# ---------------------------------------------------------------------------


def lns_to_half(p: np.ndarray) -> np.ndarray:
    """16-bit LNS values (int) -> half bit patterns (astcenc lns_to_sf16)."""
    p = np.asarray(p).astype(np.int64)
    mc = p & 0x7FF
    ec = p >> 11
    mt = np.where(mc < 512, mc * 3, np.where(mc < 1536, mc * 4 - 512, mc * 5 - 2048))
    return np.minimum((ec << 10) | (mt >> 3), HALF_MAX)


def _build_half_to_lns() -> np.ndarray:
    """(0x7C00,) the smallest LNS value decoding to each finite half."""
    halves = lns_to_half(np.arange(LNS_MAX + 1))
    table = np.full(HALF_MAX + 1, -1, dtype=np.int64)
    # iterate from the top so the lowest lns wins
    for lns in range(LNS_MAX, -1, -1):
        table[halves[lns]] = lns
    assert (table >= 0).all(), "lns_to_half is not surjective onto the finite halves"
    return table


HALF_TO_LNS = _build_half_to_lns()


def halfint_to_lns(h: np.ndarray) -> np.ndarray:
    return HALF_TO_LNS[np.clip(np.asarray(h).astype(np.int64), 0, HALF_MAX)]


def float_to_lns_norm(v: np.ndarray) -> np.ndarray:
    """float RGB -> the [0,1] LNS domain the FLOAT variant works in."""
    return halfint_to_lns(float_to_halfint(v)).astype(np.float32) / LNS_MAX


# ---------------------------------------------------------------------------
# HDR RGB (CEM 11) endpoint packing, port of astcenc quantize_hdr_rgb with
# QUANT_256 (all the "quantize and retain top bits" steps are the identity)
# and hdr_rgb_unpack. Batched torch integer maths; colours are LNS 0..65535.
# ---------------------------------------------------------------------------

_HDR_MODE_BITS = torch.tensor([[9, 7, 6, 7], [9, 8, 6, 6], [10, 6, 7, 7], [10, 7, 7, 6],
                               [11, 8, 6, 5], [11, 6, 8, 6], [12, 7, 7, 5], [12, 6, 7, 6]])
_HDR_MODE_CUTOFFS = torch.tensor([[16384, 8192, 8192], [32768, 8192, 4096], [4096, 8192, 4096], [8192, 8192, 2048],
                                  [8192, 2048, 512], [2048, 8192, 1024], [2048, 2048, 256], [1024, 2048, 512]],
                                 dtype=torch.float32)
_HDR_MODE_SHIFT = [7, 7, 6, 6, 5, 5, 4, 4]  # log2 of astcenc's mode_rscales


def _rtn(x: torch.Tensor) -> torch.Tensor:
    """astcenc flt2int_rtn: (int)(x + 0.5), i.e. truncation toward zero."""
    return torch.trunc(x + 0.5).to(torch.int64)


def pack_hdr_endpoints(e0: torch.Tensor, e1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """e0/e1 (N,3) float LNS colours in [0, 65535] -> ((N,6) int64 CEM 11
    values, swapped (N,) bool). The encoding stores e1's major component
    and non-negative deltas down to e0, so blocks whose e0 is brighter on
    the major axis are stored swapped; the caller inverts the weights."""
    n = e0.shape[0]
    dev = e0.device
    # Finite range only: a = 65535 would overflow mode 7's 12-bit field.
    c0 = e0.clamp(0.0, float(LNS_MAX_FINITE))
    c1 = e1.clamp(0.0, float(LNS_MAX_FINITE))

    hi = torch.maximum(c0, c1)
    majcomp = torch.where((hi[:, 0] >= hi[:, 1]) & (hi[:, 0] >= hi[:, 2]), 0, torch.where(hi[:, 1] >= hi[:, 2], 1, 2))
    swapped = torch.gather(c0, 1, majcomp[:, None])[:, 0] > torch.gather(c1, 1, majcomp[:, None])[:, 0]
    c0, c1 = torch.where(swapped[:, None], c1, c0), torch.where(swapped[:, None], c0, c1)
    swz = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0]], device=dev)[majcomp]  # (N,3)
    c0s = torch.gather(c0, 1, swz)
    c1s = torch.gather(c1, 1, swz)

    a_base = c1s[:, 0]
    b0_base = a_base - c1s[:, 1]
    b1_base = a_base - c1s[:, 2]
    c_base = a_base - c0s[:, 0]
    d0_base = a_base - b0_base - c_base - c0s[:, 1]
    d1_base = a_base - b1_base - c_base - c0s[:, 2]

    out = torch.zeros((n, 6), dtype=torch.int64, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    cutoffs = _HDR_MODE_CUTOFFS.to(dev)
    mode_bits = _HDR_MODE_BITS.to(dev)
    for mode in range(7, -1, -1):
        b_cut, c_cut, d_cut = cutoffs[mode]
        ok = ~done & ~((b0_base > b_cut) | (b1_base > b_cut) | (c_base > c_cut) | (d0_base.abs() > d_cut) | (d1_base.abs() > d_cut))
        sh = _HDR_MODE_SHIFT[mode]
        scale = 1.0 / (1 << sh)
        b_intcut = 1 << int(mode_bits[mode, 1])
        c_intcut = 1 << int(mode_bits[mode, 2])
        d_intcut = 1 << (int(mode_bits[mode, 3]) - 1)

        a_int = _rtn(a_base * scale)
        a_f = (a_int << sh).to(torch.float32)

        c_f = (a_f - c0s[:, 0]).clamp(0.0, 65535.0)
        c_int = _rtn(c_f * scale)
        ok &= c_int < c_intcut
        c_f = (c_int << sh).to(torch.float32)

        b0_f = (a_f - c1s[:, 1]).clamp(0.0, 65535.0)
        b1_f = (a_f - c1s[:, 2]).clamp(0.0, 65535.0)
        b0_int = _rtn(b0_f * scale)
        b1_int = _rtn(b1_f * scale)
        ok &= (b0_int < b_intcut) & (b1_int < b_intcut)
        b0_f = (b0_int << sh).to(torch.float32)
        b1_f = (b1_int << sh).to(torch.float32)

        d0_f = (a_f - b0_f - c_f - c0s[:, 1]).clamp(-65535.0, 65535.0)
        d1_f = (a_f - b1_f - c_f - c0s[:, 2]).clamp(-65535.0, 65535.0)
        d0_int = _rtn(d0_f * scale)
        d1_int = _rtn(d1_f * scale)
        ok &= (d0_int.abs() < d_intcut) & (d1_int.abs() < d_intcut)

        # Assemble the six 8-bit values (see hdr_rgb_unpack for the layout).
        a_low = a_int & 0xFF
        c_low = (c_int & 0x3F) | ((mode & 1) << 7) | ((a_int & 0x100) >> 2)
        if mode in (0, 1, 3, 4, 6):
            bit0 = (b0_int >> 6) & 1
            bit1 = (b1_int >> 6) & 1
        elif mode == 2:
            bit0 = (a_int >> 9) & 1
            bit1 = (c_int >> 6) & 1
        else:  # 5, 7
            bit0 = (a_int >> 9) & 1
            bit1 = (a_int >> 10) & 1
        b0_low = (b0_int & 0x3F) | (bit0 << 6) | (((mode >> 1) & 1) << 7)
        b1_low = (b1_int & 0x3F) | (bit1 << 6) | (((mode >> 2) & 1) << 7)
        if mode in (0, 2):
            bit2 = (d0_int >> 6) & 1
            bit3 = (d1_int >> 6) & 1
        elif mode in (1, 4):
            bit2 = (b0_int >> 7) & 1
            bit3 = (b1_int >> 7) & 1
        elif mode == 3:
            bit2 = (a_int >> 9) & 1
            bit3 = (c_int >> 6) & 1
        elif mode == 5:
            bit2 = (c_int >> 7) & 1
            bit3 = (c_int >> 6) & 1
        else:  # 6, 7
            bit2 = (a_int >> 11) & 1
            bit3 = (c_int >> 6) & 1
        if mode in (4, 6):
            bit4 = (a_int >> 9) & 1
            bit5 = (a_int >> 10) & 1
        else:
            bit4 = (d0_int >> 5) & 1
            bit5 = (d1_int >> 5) & 1
        d0_low = (d0_int & 0x1F) | (bit2 << 6) | (bit4 << 5) | ((majcomp & 1) << 7)
        d1_low = (d1_int & 0x1F) | (bit3 << 6) | (bit5 << 5) | (((majcomp >> 1) & 1) << 7)

        vals = torch.stack([a_low, c_low, b0_low, b1_low, d0_low, d1_low], dim=1)
        out = torch.where(ok[:, None], vals, out)
        done |= ok

    # Flat fallback (majcomp 3): 8 bits for R and G, 7 for B.
    flat0 = c0.clamp(0.0, 65020.0)
    flat1 = c1.clamp(0.0, 65020.0)
    fv = torch.stack([
        _rtn(flat0[:, 0] / 256.0), _rtn(flat1[:, 0] / 256.0),
        _rtn(flat0[:, 1] / 256.0), _rtn(flat1[:, 1] / 256.0),
        _rtn(flat0[:, 2] / 512.0) + 128, _rtn(flat1[:, 2] / 512.0) + 128,
    ], dim=1)
    return torch.where(done[:, None], out, fv), swapped


def unpack_hdr_endpoints(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,6) int64 CEM 11 values -> (e0, e1) (N,3) int64 16-bit LNS colours."""
    v0, v1, v2, v3, v4, v5 = (v[:, i] for i in range(6))
    modeval = ((v1 >> 7) & 1) | (((v2 >> 7) & 1) << 1) | (((v3 >> 7) & 1) << 2)
    majcomp = ((v4 >> 7) & 1) | (((v5 >> 7) & 1) << 1)

    a = v0 | ((v1 & 0x40) << 2)
    b0 = v2 & 0x3F
    b1 = v3 & 0x3F
    c = v1 & 0x3F
    d0 = v4 & 0x7F
    d1 = v5 & 0x7F
    dbits = torch.tensor([7, 6, 7, 6, 5, 6, 5, 6], device=v.device)[modeval]
    bit0 = (v2 >> 6) & 1
    bit1 = (v3 >> 6) & 1
    bit2 = (v4 >> 6) & 1
    bit3 = (v5 >> 6) & 1
    bit4 = (v4 >> 5) & 1
    bit5 = (v5 >> 5) & 1
    ohmod = 1 << modeval

    def has(mask: int) -> torch.Tensor:
        return (ohmod & mask) != 0

    a = a | torch.where(has(0xA4), bit0 << 9, 0)
    a = a | torch.where(has(0x8), bit2 << 9, 0)
    a = a | torch.where(has(0x50), bit4 << 9, 0)
    a = a | torch.where(has(0x50), bit5 << 10, 0)
    a = a | torch.where(has(0xA0), bit1 << 10, 0)
    a = a | torch.where(has(0xC0), bit2 << 11, 0)
    c = c | torch.where(has(0x4), bit1 << 6, 0)
    c = c | torch.where(has(0xE8), bit3 << 6, 0)
    c = c | torch.where(has(0x20), bit2 << 7, 0)
    b0 = b0 | torch.where(has(0x5B), bit0 << 6, 0)
    b1 = b1 | torch.where(has(0x5B), bit1 << 6, 0)
    b0 = b0 | torch.where(has(0x12), bit2 << 7, 0)
    b1 = b1 | torch.where(has(0x12), bit3 << 7, 0)
    d0 = d0 | torch.where(has(0xAF), bit4 << 5, 0)
    d1 = d1 | torch.where(has(0xAF), bit5 << 5, 0)
    d0 = d0 | torch.where(has(0x5), bit2 << 6, 0)
    d1 = d1 | torch.where(has(0x5), bit3 << 6, 0)

    # sign-extend d0/d1 from dbits (astcenc: lsh by 32-dbits, arithmetic rsh -- bits above dbits are dropped)
    sign = 1 << (dbits - 1)
    d0 = d0 & ((sign << 1) - 1)
    d1 = d1 & ((sign << 1) - 1)
    d0 = torch.where((d0 & sign) != 0, d0 - (sign << 1), d0)
    d1 = torch.where((d1 & sign) != 0, d1 - (sign << 1), d1)

    shamt = (modeval >> 1) ^ 3
    a, b0, b1, c, d0, d1 = (x << shamt for x in (a, b0, b1, c, d0, d1))
    red1 = a
    green1 = a - b0
    blue1 = a - b1
    red0 = a - c
    green0 = a - b0 - c - d0
    blue0 = a - b1 - c - d1
    e0 = torch.stack([red0, green0, blue0], dim=1).clamp(0, 4095)
    e1 = torch.stack([red1, green1, blue1], dim=1).clamp(0, 4095)
    unswz = torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 1, 2]], device=v.device)[majcomp]
    e0 = torch.gather(e0, 1, unswz) << 4
    e1 = torch.gather(e1, 1, unswz) << 4

    flat = majcomp == 3
    f0 = torch.stack([v0 << 8, v2 << 8, (v4 & 0x7F) << 9], dim=1)
    f1 = torch.stack([v1 << 8, v3 << 8, (v5 & 0x7F) << 9], dim=1)
    return torch.where(flat[:, None], f0, e0), torch.where(flat[:, None], f1, e1)


def _decoded_hdr_endpoints_torch(
    e0: torch.Tensor, e1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """[0,1] LNS-domain endpoints (N,3) -> what the decoder will see after
    CEM 11 packing, plus the packed (values, swapped) themselves. The packer
    must reuse those rather than re-pack the decoded endpoints: decoding can
    leave two channels tied for the major component, and a re-pack that
    picks the other one lands in a different mode."""
    values, swapped = pack_hdr_endpoints(e0 * LNS_MAX, e1 * LNS_MAX)
    d0, d1 = unpack_hdr_endpoints(values)
    d0, d1 = torch.where(swapped[:, None], d1, d0), torch.where(swapped[:, None], d0, d1)
    return d0.to(e0.dtype) / LNS_MAX, d1.to(e0.dtype) / LNS_MAX, (values, swapped)


# ---------------------------------------------------------------------------
# Endpoint quantizers (the decoded value, as [0,1] floats) + STE versions.
# ---------------------------------------------------------------------------


def _decoded_endpoint_ldr_torch(cfg: BlockConfig, v01: torch.Tensor) -> torch.Tensor:
    lut = torch.as_tensor(cfg.endpoint_lut, device=v01.device)
    q = torch.round(v01 * 255.0).clamp(0, 255).to(torch.int64)
    return lut[q].to(v01.dtype) / 255.0


def decoded_endpoints(cfg: BlockConfig, e0: torch.Tensor, e1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, object]:
    """(N,C) [0,1] endpoints -> (e0q, e1q, aux): the pair the decoder
    reconstructs, and for HDR the packed values pack_blocks must use."""
    if cfg.hdr:
        return _decoded_hdr_endpoints_torch(e0, e1)
    return _decoded_endpoint_ldr_torch(cfg, e0), _decoded_endpoint_ldr_torch(cfg, e1), None


# ---------------------------------------------------------------------------
# Refinement (exact index search / LS refit, see bc7_codec).
# ---------------------------------------------------------------------------


def _refine_plane(
    pixels: torch.Tensor,
    e0: torch.Tensor,
    e1: torch.Tensor,
    weights01: np.ndarray,
    quantize_pair,
    iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, object]:
    """bc7_codec._refine_line with a *pair* quantizer (CEM 11 packs the two
    endpoints jointly). Returns (e0q, e1q, idx (N,16) int64, recon (N,16,C),
    aux from the last quantization)."""
    w = torch.as_tensor(weights01, dtype=pixels.dtype, device=pixels.device)
    e0q, e1q, aux = quantize_pair(e0, e1)
    for _ in range(iters):
        idx = _best_indices(pixels, _palette(e0q, e1q, w))
        e0, e1 = _ls_endpoints(pixels, w[idx])
        e0q, e1q, aux = quantize_pair(e0, e1)
    palette = _palette(e0q, e1q, w)
    idx = _best_indices(pixels, palette)
    recon = torch.gather(palette, 1, idx[..., None].expand(-1, -1, palette.shape[-1]))
    return e0q, e1q, idx, recon, aux


@torch.no_grad()
def refine(
    cfg: BlockConfig, pixels: torch.Tensor, e0: torch.Tensor, e1: torch.Tensor, iters: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, object]:
    """pixels (N,16,C) and initial endpoints (N,C) for the config's channels.
    Returns (e0, e1, idx, idx2 | None, recon (N,16,C), aux); the endpoints
    are the decoded values, idx/idx2 the weight indices of plane 1 / plane
    2, aux the HDR packed values (None for LDR) -- all for pack_blocks."""
    quant = lambda a, b: decoded_endpoints(cfg, a, b)  # noqa: E731
    if not cfg.dual_plane:
        e0q, e1q, idx, recon, aux = _refine_plane(pixels, e0, e1, cfg.weights01, quant, iters)
        return e0q, e1q, idx, None, recon, aux
    assert not cfg.hdr
    c = len(cfg.channels)
    p2 = cfg.plane2_channel
    p1 = [i for i in range(c) if i != p2]
    # the LDR quantizer is per channel, so the two planes can be refined independently
    e0a, e1a, idx1, recon1, _ = _refine_plane(pixels[..., p1], e0[:, p1], e1[:, p1], cfg.weights01, quant, iters)
    e0b, e1b, idx2, recon2, _ = _refine_plane(pixels[..., p2:p2 + 1], e0[:, p2:p2 + 1], e1[:, p2:p2 + 1], cfg.weights01, quant, iters)
    e0q = torch.empty_like(e0)
    e1q = torch.empty_like(e1)
    recon = torch.empty_like(pixels)
    e0q[:, p1], e0q[:, p2] = e0a, e0b[:, 0]
    e1q[:, p1], e1q[:, p2] = e1a, e1b[:, 0]
    recon[..., p1], recon[..., p2] = recon1, recon2[..., 0]
    return e0q, e1q, idx1, idx2, recon, None


# ---------------------------------------------------------------------------
# Bit-exact packer (torch, batched).
# ---------------------------------------------------------------------------


def _bits_cols(values: torch.Tensor, nbits: int) -> torch.Tensor:
    """(N,) int64 -> (N, nbits) uint8 LSB first."""
    shifts = torch.arange(nbits, device=values.device, dtype=torch.int64)
    return ((values[:, None] >> shifts) & 1).to(torch.uint8)


def pack_blocks(
    cfg: BlockConfig,
    e0: torch.Tensor,
    e1: torch.Tensor,
    idx: torch.Tensor,
    idx2: torch.Tensor | None = None,
    aux: object = None,
) -> torch.Tensor:
    """e0/e1 (N,C) decoded [0,1] endpoints (from `refine`), idx/idx2 (N,16)
    weight indices, aux = refine's HDR packed values (re-packed from e0/e1
    when None) -> (N,16) uint8 ASTC blocks."""
    n = e0.shape[0]
    dev = e0.device
    idx = idx.to(torch.int64)
    idx2 = None if idx2 is None else idx2.to(torch.int64)
    maxw = cfg.weight_levels - 1

    if cfg.hdr:
        assert cfg.endpoint_levels == 256
        # (N,6) 8-bit values = the QUANT_256 codes
        codes, swap = aux if aux is not None else pack_hdr_endpoints(e0 * LNS_MAX, e1 * LNS_MAX)
        idx = torch.where(swap[:, None], maxw - idx, idx)
    else:
        v0 = torch.round(e0 * 255.0).clamp(0, 255).to(torch.int64)
        v1 = torch.round(e1 * 255.0).clamp(0, 255).to(torch.int64)
        if cfg.cem in (CEM_RGB, CEM_RGBA):
            # decoder swaps (and blue-contracts) when sum(rgb0) > sum(rgb1)
            swap = v0[:, :3].sum(1) > v1[:, :3].sum(1)
            v0, v1 = torch.where(swap[:, None], v1, v0), torch.where(swap[:, None], v0, v1)
            idx = torch.where(swap[:, None], maxw - idx, idx)
            if idx2 is not None:
                idx2 = torch.where(swap[:, None], maxw - idx2, idx2)
        values = torch.stack([v0, v1], dim=2).reshape(n, -1)  # v0[c], v1[c] interleaved per channel
        _, code_of_value = ise.endpoint_unquant_table(cfg.endpoint_levels)
        codes = torch.as_tensor(code_of_value, device=dev)[values]
    assert codes.shape[1] == cfg.n_values

    bits = torch.zeros((n, 128), dtype=torch.uint8, device=dev)
    bits[:, 0:11] = _bits_cols(torch.full((n,), cfg.block_mode, dtype=torch.int64, device=dev), 11)
    bits[:, 13:17] = _bits_cols(torch.full((n,), cfg.cem, dtype=torch.int64, device=dev), 4)
    ep = ise.encode_ise_torch(codes, cfg.endpoint_levels)
    bits[:, 17:17 + ep.shape[1]] = ep

    if cfg.dual_plane:
        assert idx2 is not None
        wcodes = torch.stack([idx, idx2], dim=2).reshape(n, 32)
        pos = 128 - cfg.weight_stream_bits - 2
        bits[:, pos:pos + 2] = _bits_cols(torch.full((n,), cfg.ccs, dtype=torch.int64, device=dev), 2)
    else:
        wcodes = idx
    wbits = ise.encode_ise_torch(wcodes, cfg.weight_levels)  # (N, W) stream order
    bits[:, 127 - torch.arange(wbits.shape[1], device=dev)] = wbits

    byte_weights = (1 << torch.arange(8, device=dev, dtype=torch.int32))
    return (bits.reshape(n, 16, 8).to(torch.int32) * byte_weights).sum(dim=-1).to(torch.uint8)


# ---------------------------------------------------------------------------
# The encoder: per-block config selection.
# ---------------------------------------------------------------------------


def _full_recon(cfg: BlockConfig, recon: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Config-channel reconstruction -> variant channels (alpha = 1 when not stored)."""
    if len(cfg.channels) == n_channels:
        return recon
    out = torch.ones((recon.shape[0], 16, n_channels), dtype=recon.dtype, device=recon.device)
    out[..., cfg.channels] = recon
    return out


@torch.no_grad()
def encode_config(cfg: BlockConfig, pixels: torch.Tensor, iters: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """pixels (N,16,Cv) in the variant's domain. Endpoints start at the
    per-channel block min/max. Returns (packed (N,16) uint8, recon (N,16,Cv))."""
    px = pixels[..., cfg.channels]
    e0c, e1c = px.amin(dim=1), px.amax(dim=1)
    e0q, e1q, idx, idx2, recon, aux = refine(cfg, px, e0c, e1c, iters)
    return pack_blocks(cfg, e0q, e1q, idx, idx2, aux), _full_recon(cfg, recon, pixels.shape[-1])


@torch.no_grad()
def encode_blocks(
    pixels: torch.Tensor, configs: list[BlockConfig], iters: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every block with each config and keep the one with the lowest
    squared error. Returns (packed (N,16) uint8, recon (N,16,Cv), chosen
    (N,) int64 index into `configs`)."""
    best_sse = best_packed = best_recon = None
    chosen = torch.zeros(pixels.shape[0], dtype=torch.int64, device=pixels.device)
    for i, cfg in enumerate(configs):
        packed, recon = encode_config(cfg, pixels, iters)
        sse = ((recon - pixels) ** 2).sum(dim=(1, 2))
        if best_sse is None:
            best_sse, best_packed, best_recon = sse, packed, recon
            continue
        better = sse < best_sse
        best_sse = torch.where(better, sse, best_sse)
        best_packed = torch.where(better[:, None], packed, best_packed)
        best_recon = torch.where(better[:, None, None], recon, best_recon)
        chosen = torch.where(better, i, chosen)
    return best_packed, best_recon, chosen


@torch.no_grad()
def encode_image(
    variant: str,
    img: np.ndarray,
    device: torch.device,
    iters: int = 2,
    configs: list[BlockConfig] | None = None,
) -> tuple[bytes, np.ndarray, np.ndarray, int, int, float]:
    """Encode a whole image (uint8 RGBA for game/normal, float RGB for
    float) -> (blocks bytes, decoded image in the variant's presentation
    ((H,W,4) uint8 / (H,W,2) uint8 / (H,W,3) float32), chosen config per
    block, w, h, encode seconds (refinement + packing, excluding the
    image -> block conversion and the decode))."""
    import time

    configs = configs or VARIANT_CONFIGS[variant]
    pixels = image_to_pixels(variant, img).to(device)
    h, w = (img.shape[0] // 4) * 4, (img.shape[1] // 4) * 4
    t0 = time.perf_counter()
    packed, _, chosen = encode_blocks(pixels, configs, iters)
    blocks = packed.cpu().numpy()
    seconds = time.perf_counter() - t0
    decoded = decode_blocks(blocks)
    if variant == "float":
        out = halfint_to_float(blocks_to_image(decoded[..., :3], w, h))
    elif variant == "normal":
        out = blocks_to_image(decoded[..., [0, 3]], w, h).astype(np.uint8)
    else:
        out = blocks_to_image(decoded, w, h).astype(np.uint8)
    return blocks.tobytes(), out, chosen.cpu().numpy(), w, h, seconds


def astcenc_encode_decode(variant: str, image_path: Path, preset: str = "-medium") -> tuple[np.ndarray, float]:
    """astcenc reference: encode the file, decode the result. Returns (decoded
    image in the variant's presentation, encode wall time in seconds incl.
    the CLI's image load, all cores)."""
    import time

    with tempfile.TemporaryDirectory() as tmp:
        astc = Path(tmp) / "ref.astc"
        t0 = time.perf_counter()
        astcenc_encode(image_path, astc, variant, preset)
        dt = time.perf_counter() - t0
        if variant == "float":
            return astcenc_decode_hdr(astc), dt
        img = astcenc_decode_ldr(astc)
        if variant == "normal":  # -dl does not apply -normal's output swizzle: Y is in A
            return img[..., [0, 3]], dt
        return img, dt


# ---------------------------------------------------------------------------
# Reference decoder for the blocks we emit (numpy + torch, CPU).
# ---------------------------------------------------------------------------

_CONFIG_BY_MODE = {(c.block_mode, c.cem): c for c in ALL_CONFIGS}


def decode_blocks(blocks: np.ndarray) -> np.ndarray:
    """(N,16) uint8 blocks -> (N,16,4) int64: RGBA8 for LDR blocks, RGB half
    bits (A = 0x3C00) for HDR blocks. Only the block modes / CEMs of
    ALL_CONFIGS are understood; anything else raises."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, 16)
    n = blocks.shape[0]
    bits = np.unpackbits(blocks, axis=1, bitorder="little").astype(np.int64)  # (N,128), column = bit index

    def field(lo: int, count: int) -> np.ndarray:
        return (bits[:, lo:lo + count] * (1 << np.arange(count))).sum(axis=1)

    mode = field(0, 11)
    cem = field(13, 4)
    if (field(11, 2) != 0).any():
        raise ValueError("multi-partition block")
    out = np.zeros((n, 16, 4), dtype=np.int64)
    keys = np.stack([mode, cem], axis=1)
    for key in np.unique(keys, axis=0):
        cfg = _CONFIG_BY_MODE.get((int(key[0]), int(key[1])))
        if cfg is None:
            raise ValueError(f"block mode 0x{int(key[0]):03x} / CEM {int(key[1])} is not one of ours")
        sel = np.flatnonzero((keys == key).all(axis=1))
        b = bits[sel]
        w_stream = b[:, 127 - np.arange(cfg.weight_stream_bits)]
        wcodes = ise.decode_ise(w_stream, cfg.n_weights, cfg.weight_levels)
        wunq = ise.weight_unquant_table(cfg.weight_levels)[wcodes]  # 0..64
        codes = ise.decode_ise(b[:, 17:17 + cfg.color_bits], cfg.n_values, cfg.endpoint_levels)

        if cfg.hdr:
            e0, e1 = unpack_hdr_endpoints(torch.from_numpy(codes))
            e0, e1 = e0.numpy(), e1.numpy()  # (M,3) 16-bit LNS
        else:
            unq, _ = ise.endpoint_unquant_table(cfg.endpoint_levels)
            v = unq[codes].astype(np.int64)  # (M, n_values)
            if cfg.cem == CEM_LA:
                e0 = np.stack([v[:, 0], v[:, 0], v[:, 0], v[:, 2]], axis=1)
                e1 = np.stack([v[:, 1], v[:, 1], v[:, 1], v[:, 3]], axis=1)
            else:
                e0 = v[:, 0::2]
                e1 = v[:, 1::2]
                if cfg.cem == CEM_RGB:
                    e0 = np.concatenate([e0, np.full((len(sel), 1), 255)], axis=1)
                    e1 = np.concatenate([e1, np.full((len(sel), 1), 255)], axis=1)
                swap = e0[:, :3].sum(1) > e1[:, :3].sum(1)
                if swap.any():  # never for our packer, but keep the decoder honest
                    def contract(e):
                        return np.stack([(e[:, 0] + e[:, 2]) >> 1, (e[:, 1] + e[:, 2]) >> 1, e[:, 2], e[:, 3]], axis=1)
                    c0, c1 = contract(e0), contract(e1)
                    e0, e1 = np.where(swap[:, None], c1, e0), np.where(swap[:, None], c0, e1)
            e0, e1 = e0 * 257, e1 * 257

        if cfg.dual_plane:
            ccs = (b[:, 128 - cfg.weight_stream_bits - 2:128 - cfg.weight_stream_bits] * np.array([1, 2])).sum(axis=1)
            assert (ccs == cfg.ccs).all()
            w = np.repeat(wunq[:, 0::2, None], 4, axis=2)  # (M,16,4)
            w[np.arange(len(sel)), :, ccs] = wunq[:, 1::2]
        else:
            w = np.repeat(wunq[:, :, None], 4, axis=2)

        if cfg.hdr:
            w = w[..., :3]
        c16 = (e0[:, None, :] * (64 - w) + e1[:, None, :] * w + 32) >> 6
        if cfg.hdr:
            rgb = lns_to_half(c16[..., :3])
            out[sel, :, :3] = rgb
            out[sel, :, 3] = 0x3C00
        else:
            out[sel] = c16 >> 8
    return out


# ---------------------------------------------------------------------------
# Containers, third-party decoders, astcenc CLI.
# ---------------------------------------------------------------------------


def write_astc_file(path: Path, blocks: bytes, width: int, height: int) -> None:
    """The .astc container: magic, block dims, then x/y/z size as 24-bit LE."""
    header = bytes([0x13, 0xAB, 0xA1, 0x5C, 4, 4, 1])
    header += struct.pack("<I", width)[:3] + struct.pack("<I", height)[:3] + struct.pack("<I", 1)[:3]
    Path(path).write_bytes(header + blocks)


def blocks_to_image(blocks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(N,16,C) block pixels -> (H,W,C) image (H, W multiples of 4)."""
    c = blocks.shape[-1]
    return blocks.reshape(height // 4, width // 4, 4, 4, c).transpose(0, 2, 1, 3, 4).reshape(height, width, c)


def decode_astc_t2d(data: bytes, width: int, height: int) -> np.ndarray:
    """texture2ddecoder (LDR only, 8-bit): (H,W,4) uint8 RGBA."""
    raw = _t2d.decode_astc(data, width, height, 4, 4)
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return bgra[..., [2, 1, 0, 3]].copy()


def astcenc_path() -> Path | None:
    p = Path(os.environ.get("ASTCENC", ROOT / "c_api" / "build" / "third_party" / "astcenc" / "Source" / "astcenc-neon"))
    return p if p.exists() else None


def _run(cmd: list[str]) -> None:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr or r.stdout}")


def astcenc_decode_ldr(astc_file: Path) -> np.ndarray:
    """(H,W,4) uint8 via `astcenc -dl -decode_unorm8` (the exact top-8-bits decode)."""
    from PIL import Image

    exe = astcenc_path()
    assert exe is not None, "astcenc CLI not built: cmake --build c_api/build"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dec.png"
        _run([exe, "-dl", astc_file, out, "-decode_unorm8"])
        return np.array(Image.open(out).convert("RGBA"), dtype=np.uint8)


def astcenc_decode_hdr(astc_file: Path) -> np.ndarray:
    """(H,W,3) float32 via `astcenc -dh` -> .exr -> oiiotool -> .pfm."""
    exe = astcenc_path()
    assert exe is not None, "astcenc CLI not built: cmake --build c_api/build"
    with tempfile.TemporaryDirectory() as tmp:
        exr = Path(tmp) / "dec.exr"
        pfm = Path(tmp) / "dec.pfm"
        _run([exe, "-dh", astc_file, exr])
        _run(["oiiotool", exr, "-d", "float", "-o", pfm])
        return _read_pfm(pfm)


def _read_pfm(path: Path) -> np.ndarray:
    raw = Path(path).read_bytes()
    parts = raw.split(b"\n", 3)
    assert parts[0] == b"PF", "expected a colour PFM"
    w, h = (int(x) for x in parts[1].split())
    scale = float(parts[2])
    data = np.frombuffer(parts[3], dtype="<f4" if scale < 0 else ">f4", count=w * h * 3).reshape(h, w, 3)
    return np.ascontiguousarray(data[::-1]).astype(np.float32)  # PFM is bottom-up


def astcenc_encode(image_path: Path, out_astc: Path, mode: str, preset: str = "-medium") -> None:
    """mode: 'game' (-cl), 'normal' (-cl -normal, X,Y in R,G), 'float' (-ch)."""
    exe = astcenc_path()
    assert exe is not None, "astcenc CLI not built: cmake --build c_api/build"
    cmd = [exe, "-ch" if mode == "float" else "-cl", image_path, out_astc, "4x4", preset]
    if mode == "normal":
        cmd.append("-normal")
    _run(cmd)


# ---------------------------------------------------------------------------
# Metrics / image <-> block helpers per variant.
# ---------------------------------------------------------------------------


def psnr(a: np.ndarray, b: np.ndarray, peak: float = 1.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


def image_to_pixels(variant: str, img: np.ndarray) -> torch.Tensor:
    """Image (uint8 RGBA for game/normal, float RGB for float) -> (N,16,Cv) float32 in [0,1]."""
    from data import extract_blocks

    h, w = (img.shape[0] // 4) * 4, (img.shape[1] // 4) * 4
    img = img[:h, :w]
    if variant == "float":
        x = float_to_lns_norm(img)
    elif variant == "normal":
        x = img[..., :2].astype(np.float32) / 255.0
    else:
        x = img.astype(np.float32) / 255.0
    return torch.from_numpy(extract_blocks(x).reshape(-1, 16, x.shape[-1]).astype(np.float32))


def decoded_to_variant(variant: str, decoded: np.ndarray) -> np.ndarray:
    """decode_blocks output (N,16,4) -> (N,16,Cv) in the variant's [0,1] domain."""
    if variant == "float":
        return halfint_to_lns(decoded[..., :3]).astype(np.float32) / LNS_MAX
    if variant == "normal":
        return decoded[..., [0, 3]].astype(np.float32) / 255.0
    return decoded.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Self-test: refine -> pack -> own decoder == recon, == astcenc's decode.
# ---------------------------------------------------------------------------


def _self_test(image: str | None, normal: str | None, hdr: str | None, iters: int) -> None:
    from data import load_hdr, load_rgba

    torch.manual_seed(0)
    for cfg in ALL_CONFIGS:
        print(cfg.describe())

    # CEM 11 pack/unpack fuzz: unpack(pack(e)) must be within the mode's precision of e.
    e0 = torch.rand(20000, 3) * LNS_MAX_FINITE
    e1 = e0 + (torch.rand(20000, 3) - 0.5) * torch.rand(20000, 1) * 8000
    e1 = e1.clamp(0, LNS_MAX_FINITE)
    p0, p1, _ = _decoded_hdr_endpoints_torch(e0 / LNS_MAX, e1 / LNS_MAX)
    err = torch.maximum((p0 * LNS_MAX - e0).abs(), (p1 * LNS_MAX - e1).abs()).amax(1)
    # worst case is the flat fallback (blue at 9-bit steps = 512) -- anything beyond it is a packing bug
    assert err.max() <= 512, f"CEM 11 pack/unpack: max error {err.max()} LNS units"
    print(f"CEM 11 pack/unpack fuzz OK: median error {err.median():.1f}, max {err.max():.0f} LNS units")

    jobs: list[tuple[str, dict[str, torch.Tensor], np.ndarray | None, int, int, Path | None]] = []
    sets = {"random": torch.rand(4096, 16, 4)}
    src = None
    w = h = 0
    if image:
        src = load_rgba(Path(image))
        h, w = (src.shape[0] // 4) * 4, (src.shape[1] // 4) * 4
        src = src[:h, :w]
        sets["image"] = image_to_pixels("game", src)
    jobs.append(("game", sets, src, w, h, Path(image) if image else None))

    sets = {"random": torch.rand(4096, 16, 2)}
    src = None
    if normal:
        src = load_rgba(Path(normal))
        h, w = (src.shape[0] // 4) * 4, (src.shape[1] // 4) * 4
        src = src[:h, :w]
        sets["image"] = image_to_pixels("normal", src)
    jobs.append(("normal", sets, src, w, h, Path(normal) if normal else None))

    sets = {"random": torch.rand(4096, 16, 3) * 0.6 + 0.2}
    src = None
    if hdr:
        src = load_hdr(Path(hdr))
        h, w = (src.shape[0] // 4) * 4, (src.shape[1] // 4) * 4
        src = src[:h, :w]
        sets["image"] = image_to_pixels("float", src)
    jobs.append(("float", sets, src, w, h, Path(hdr) if hdr else None))

    exe = astcenc_path()
    for variant, sets, src, w, h, src_path in jobs:
        configs = [c for c in ALL_CONFIGS if c.name.lower().startswith(variant)]
        for name, pixels in sets.items():
            for cfg in configs:
                packed, recon = encode_config(cfg, pixels, iters)
                decoded = decoded_to_variant(variant, decode_blocks(packed.numpy()))
                scale = LNS_MAX if variant == "float" else 255.0
                diff = np.abs(decoded * scale - recon.numpy() * scale).max()
                tol = 40.0 if variant == "float" else 1.0  # LNS->half->LNS loses the low 3-4 bits
                assert diff <= tol, f"{variant}/{name}/{cfg.name}: own decoder differs from recon by {diff:.2f}"
                if name == "image":
                    print(f"  {variant} {cfg.name:9s} analytic PSNR {psnr(decoded, pixels.numpy()):.2f}dB")
            packed, recon, chosen = encode_blocks(pixels, VARIANT_CONFIGS[variant], iters)
            decoded = decoded_to_variant(variant, decode_blocks(packed.numpy()))
            share = " ".join(f"{c.name} {100 * (chosen == i).float().mean():.0f}%" for i, c in enumerate(VARIANT_CONFIGS[variant]))
            print(f"{variant}/{name}: {len(pixels)} blocks, own decoder OK, PSNR {psnr(decoded, pixels.numpy()):.2f}dB [{share}]")

            if name != "image":
                continue
            blocks = packed.numpy().tobytes()
            with tempfile.TemporaryDirectory() as tmp:
                astc = Path(tmp) / "ours.astc"
                write_astc_file(astc, blocks, w, h)
                dec_img = blocks_to_image(decode_blocks(packed.numpy()), w, h)
                if variant == "float":
                    if exe:
                        ref = astcenc_decode_hdr(astc)
                        ours = halfint_to_float(dec_img[..., :3])
                        assert np.array_equal(float_to_halfint(ref), float_to_halfint(ours)), "astcenc HDR decode differs from ours"
                        print("  astcenc -dh decode == ours (bit-exact half floats)")
                else:
                    t2d = decode_astc_t2d(blocks, w, h)
                    d = np.abs(t2d.astype(int) - dec_img).max()
                    assert d <= 1, f"texture2ddecoder differs from ours by {d}"
                    print(f"  texture2ddecoder decode within {d} LSB of ours")
                    if exe:
                        ref = astcenc_decode_ldr(astc)
                        assert np.array_equal(ref.astype(int), dec_img), "astcenc LDR decode differs from ours"
                        print("  astcenc -dl -decode_unorm8 decode == ours (bit-exact)")
                if exe and src_path is not None:
                    ref_astc = Path(tmp) / "ref.astc"
                    astcenc_encode(src_path, ref_astc, variant)
                    if variant == "float":
                        ref_img = astcenc_decode_hdr(ref_astc)[:h, :w]
                    else:
                        ref_img = astcenc_decode_ldr(ref_astc)[:h, :w]
                        if variant == "normal":  # -dl does not apply -normal's output swizzle: Y is in A
                            ref_img = ref_img[..., [0, 3, 2, 3]]
                    ref_px = image_to_pixels(variant, ref_img)
                    print(f"  astcenc -medium reference: PSNR {psnr(ref_px.numpy(), pixels.numpy()):.2f}dB")
    print("astc codec self-test OK")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASTC 4x4 codec self-test")
    parser.add_argument("--image", default=None, help="RGBA texture for the GAME configs")
    parser.add_argument("--normal", default=None, help="normal map (X,Y in R,G) for the NORMAL configs")
    parser.add_argument("--hdr", default=None, help=".hdr image for the FLOAT config")
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()
    _self_test(args.image, args.normal, args.hdr, args.iters)
