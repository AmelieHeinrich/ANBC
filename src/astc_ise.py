"""ASTC integer-sequence encoding (BISE) + quantization tables, the bit-level
primitives shared by astc_codec.py (Python packer/decoder) and the Metal
kernels (`--emit-metal` prints the constant tables).

Only what the 4x4 encoder emits is covered: bits-only weight ranges
(QUANT_4/8/16/32) and bits-only or trit-based endpoint ranges (QUANT_48,
QUANT_192, QUANT_256). Quint ranges are known to the bit-count table (the
endpoint quant derivation needs every level) but have no encoder/decoder.

Facts (from the Khronos Data Format spec / astcenc, verified in the
self-test against the vendored astcenc source tables):
  * A quant level is (bits, trits, quints); a value in 0..levels-1 is the
    code `trit * 2^bits + bits` (astcenc's "scrambled pquant").
  * ISE bit count: n*bits + ceil(8n/5) with trits, + ceil(7n/3) with quints.
  * Trit ISE: 5 values per block, T (8 bits) = integer_of_trits[t4..t0],
    layout [v0][T1:0][v1][T3:2][v2][T4][v3][T6:5][v4][T7], LSB-first; a
    partial trailing block only emits the elements present.
  * Endpoint unquantization (trit ranges, bit field with LSB a):
    A = a ? 0x1FF : 0; T = D*C + B; T ^= A; T = (A & 0x80) | (T >> 2)
    with (C, B pattern) = 2 bits (93, b000b0bb0), 4 bits (22, dcb000dcb),
    6 bits (5, fedcb000f). Bits-only ranges are plain bit replication.
  * Weight unquantization (bits-only): replicate to 6 bits, +1 if > 32.
  * The endpoint quant level is NOT stored in the block: the decoder takes
    the largest level (>= QUANT_6) whose ISE size for the value count fits
    the bits left after the header, the weights and the dual-plane CCS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# levels -> (bits, trits, quints), in ASTC quant_method order.
QUANT_LEVELS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]
QUANT_INFO: dict[int, tuple[int, int, int]] = {
    2: (1, 0, 0), 3: (0, 1, 0), 4: (2, 0, 0), 5: (0, 0, 1), 6: (1, 1, 0), 8: (3, 0, 0), 10: (1, 0, 1),
    12: (2, 1, 0), 16: (4, 0, 0), 20: (2, 0, 1), 24: (3, 1, 0), 32: (5, 0, 0), 40: (3, 0, 1), 48: (4, 1, 0),
    64: (6, 0, 0), 80: (4, 0, 1), 96: (5, 1, 0), 128: (7, 0, 0), 160: (5, 0, 1), 192: (6, 1, 0), 256: (8, 0, 0),
}


def ise_bitcount(n_values: int, levels: int) -> int:
    bits, trits, quints = QUANT_INFO[levels]
    total = n_values * bits
    if trits:
        total += (8 * n_values + 4) // 5
    if quints:
        total += (7 * n_values + 2) // 3
    return total


def derive_endpoint_quant(n_values: int, color_bits: int) -> int:
    """The endpoint quant level (levels) a decoder derives for `n_values`
    endpoint integers stored in `color_bits` bits."""
    for levels in reversed(QUANT_LEVELS):
        if levels >= 6 and ise_bitcount(n_values, levels) <= color_bits:
            return levels
    raise ValueError(f"{n_values} endpoint values do not fit in {color_bits} bits")


# ---------------------------------------------------------------------------
# Trits.
# ---------------------------------------------------------------------------


def _bits(v: int, hi: int, lo: int) -> int:
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def _trits_of_integer(t: int) -> tuple[int, int, int, int, int]:
    """Spec decode of one 8-bit trit block T into (t0..t4)."""
    if _bits(t, 4, 2) == 0b111:
        c = (_bits(t, 7, 5) << 2) | _bits(t, 1, 0)
        t4 = t3 = 2
    else:
        c = _bits(t, 4, 0)
        if _bits(t, 6, 5) == 0b11:
            t4 = 2
            t3 = _bits(t, 7, 7)
        else:
            t4 = _bits(t, 7, 7)
            t3 = _bits(t, 6, 5)
    if _bits(c, 1, 0) == 0b11:
        t2 = 2
        t1 = _bits(c, 4, 4)
        t0 = (_bits(c, 3, 3) << 1) | (_bits(c, 2, 2) & ~_bits(c, 3, 3) & 1)
    elif _bits(c, 3, 2) == 0b11:
        t2 = 2
        t1 = 2
        t0 = _bits(c, 1, 0)
    else:
        t2 = _bits(c, 4, 4)
        t1 = _bits(c, 3, 2)
        t0 = (_bits(c, 1, 1) << 1) | (_bits(c, 0, 0) & ~_bits(c, 1, 1) & 1)
    return t0, t1, t2, t3, t4


TRITS_OF_INTEGER = np.array([_trits_of_integer(t) for t in range(256)], dtype=np.int64)  # (256, 5)


def _build_integer_of_trits() -> np.ndarray:
    """(3,3,3,3,3) indexed [t4][t3][t2][t1][t0] -> the T the encoder writes.
    The decode is not a bijection (13 trit tuples have two T encodings,
    e.g. (2,0,2,0,0) <- 11 and 15, because a "don't care" bit is left in
    C); any decoder accepts both, and astcenc's table -- which the
    self-test checks against -- holds the *highest* T for each tuple."""
    table = np.full((3, 3, 3, 3, 3), -1, dtype=np.int64)
    for t in range(256):
        t0, t1, t2, t3, t4 = TRITS_OF_INTEGER[t]
        table[t4, t3, t2, t1, t0] = t
    assert (table >= 0).all()
    return table


INTEGER_OF_TRITS = _build_integer_of_trits()
# Flat (243,) version indexed by t4*81 + t3*27 + t2*9 + t1*3 + t0 (what the Metal kernel uses).
INTEGER_OF_TRITS_FLAT = INTEGER_OF_TRITS.reshape(-1)

# Per element of a 5-value trit block: how many T bits follow it and from which T bit they start.
_TRIT_TBITS = (2, 2, 1, 2, 1)
_TRIT_TSHIFT = (0, 2, 4, 5, 7)


# ---------------------------------------------------------------------------
# Unquantization tables.
# ---------------------------------------------------------------------------

_TRIT_C = {1: 204, 2: 93, 3: 44, 4: 22, 5: 11, 6: 5}
# 9-bit B pattern, MSB first, as a function of the field bits named a (LSB), b, c, ...
_TRIT_B = {
    1: "000000000",
    2: "b000b0bb0",
    3: "cb000cbcb",
    4: "dcb000dcb",
    5: "edcb000ed",
    6: "fedcb000f",
}


def _unquant_endpoint_code(levels: int, code: int) -> int:
    bits, trits, quints = QUANT_INFO[levels]
    assert quints == 0, "quint endpoint ranges are not supported"
    if not trits:
        if bits == 8:
            return code
        # bit replication until 8 bits are filled (3 bits: 001 -> 00100100)
        v, n = 0, 0
        while n < 8:
            v = (v << bits) | code
            n += bits
        return v >> (n - 8)
    d = code >> bits
    field = code & ((1 << bits) - 1)
    a = field & 1
    A = 0x1FF if a else 0
    names = "abcdef"
    B = 0
    for ch in _TRIT_B[bits]:
        B <<= 1
        if ch != "0":
            B |= (field >> names.index(ch)) & 1
    T = d * _TRIT_C[bits] + B
    T ^= A
    T = (A & 0x80) | (T >> 2)
    return T


def endpoint_unquant_table(levels: int) -> tuple[np.ndarray, np.ndarray]:
    """(unq_of_code (levels,) uint8, code_of_value (256,) int64): the decoded
    8-bit value of each ISE code, and the code whose decoded value is
    nearest to each 8-bit value (ties -> lower value, like astcenc's
    'negative residual' entry)."""
    unq = np.array([_unquant_endpoint_code(levels, c) for c in range(levels)], dtype=np.int64)
    values = np.arange(256)[:, None]
    dist = np.abs(values - unq[None, :])
    best = np.empty(256, dtype=np.int64)
    for v in range(256):
        d = dist[v]
        cands = np.flatnonzero(d == d.min())
        best[v] = cands[np.argmin(unq[cands])]
    return unq.astype(np.uint8), best


def weight_unquant_table(levels: int) -> np.ndarray:
    """(levels,) int64 unquantized weights 0..64 of a bits-only weight range."""
    bits, trits, quints = QUANT_INFO[levels]
    assert trits == 0 and quints == 0, "only bits-only weight ranges are used"
    out = []
    for v in range(levels):
        # replicate to 6 bits
        r = 0
        n = 0
        while n < 6:
            r = (r << bits) | v
            n += bits
        r >>= n - 6
        if r > 32:
            r += 1
        out.append(r)
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# ISE encode / decode.
# ---------------------------------------------------------------------------


def _bits_lsb_first(values: torch.Tensor, nbits: int) -> torch.Tensor:
    """(N,) int64 -> (N, nbits) uint8, LSB first."""
    shifts = torch.arange(nbits, device=values.device, dtype=torch.int64)
    return ((values[:, None] >> shifts) & 1).to(torch.uint8)


def encode_ise_torch(codes: torch.Tensor, levels: int) -> torch.Tensor:
    """codes (N, K) int64 ISE codes (0..levels-1) -> (N, ise_bitcount(K)) uint8
    bit matrix, stream order (bit 0 first)."""
    bits, trits, quints = QUANT_INFO[levels]
    assert quints == 0, "quint ranges are not supported"
    n, k = codes.shape
    device = codes.device
    if not trits:
        if bits == 0:
            return torch.zeros((n, 0), dtype=torch.uint8, device=device)
        return _bits_lsb_first(codes.reshape(-1), bits).reshape(n, k * bits)

    mask = (1 << bits) - 1
    table = torch.as_tensor(INTEGER_OF_TRITS_FLAT, device=device)
    columns: list[torch.Tensor] = []
    for start in range(0, k, 5):
        count = min(5, k - start)
        tr = [codes[:, start + j] >> bits if j < count else torch.zeros(n, dtype=torch.int64, device=device)
              for j in range(5)]
        t_val = table[tr[4] * 81 + tr[3] * 27 + tr[2] * 9 + tr[1] * 3 + tr[0]]
        for j in range(count):
            if bits:
                columns.append(_bits_lsb_first(codes[:, start + j] & mask, bits))
            tb = (t_val >> _TRIT_TSHIFT[j]) & ((1 << _TRIT_TBITS[j]) - 1)
            columns.append(_bits_lsb_first(tb, _TRIT_TBITS[j]))
    out = torch.cat(columns, dim=1)
    assert out.shape[1] == ise_bitcount(k, levels)
    return out


def decode_ise(bits_stream: np.ndarray, n_values: int, levels: int) -> np.ndarray:
    """bits_stream (N, >= ise_bitcount) uint8/bool in stream order -> (N, n_values) int64 codes."""
    bits, trits, quints = QUANT_INFO[levels]
    assert quints == 0, "quint ranges are not supported"
    b = np.asarray(bits_stream).astype(np.int64)
    n = b.shape[0]

    def read(pos: int, count: int) -> np.ndarray:
        if count == 0:
            return np.zeros(n, dtype=np.int64)
        w = 1 << np.arange(count)
        return (b[:, pos:pos + count] * w).sum(axis=1)

    out = np.zeros((n, n_values), dtype=np.int64)
    pos = 0
    if not trits:
        for i in range(n_values):
            out[:, i] = read(pos, bits)
            pos += bits
        return out

    for start in range(0, n_values, 5):
        count = min(5, n_values - start)
        low = []
        t_val = np.zeros(n, dtype=np.int64)
        for j in range(count):
            low.append(read(pos, bits))
            pos += bits
            t_val |= read(pos, _TRIT_TBITS[j]) << _TRIT_TSHIFT[j]
            pos += _TRIT_TBITS[j]
        trit = TRITS_OF_INTEGER[t_val]  # (N, 5)
        for j in range(count):
            out[:, start + j] = (trit[:, j] << bits) | low[j]
    return out


# ---------------------------------------------------------------------------
# Self-test against the vendored astcenc tables + Metal table emitter.
# ---------------------------------------------------------------------------

_ASTCENC_SRC = Path(__file__).resolve().parent.parent / "c_api" / "third_party" / "astcenc" / "Source"


def _parse_c_array(text: str, name: str) -> list[int]:
    start = text.index(name)
    start = text.index("{", start)
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    body = text[start + 1:end]
    body = "".join(ch if (ch.isdigit() or ch == "-") else " " for ch in body)
    return [int(x) for x in body.split()]


def _self_test() -> None:
    # Trit bijection + canonical T table vs astcenc.
    assert len(set(INTEGER_OF_TRITS_FLAT.tolist())) == 243
    iseq = (_ASTCENC_SRC / "astcenc_integer_sequence.cpp").read_text()
    ref_trits = np.array(_parse_c_array(iseq, "trits_of_integer[256][5]")).reshape(256, 5)
    assert np.array_equal(ref_trits, TRITS_OF_INTEGER), "trits_of_integer differs from astcenc"
    ref_iot = np.array(_parse_c_array(iseq, "integer_of_trits[3][3][3][3][3]")).reshape(3, 3, 3, 3, 3)
    assert np.array_equal(ref_iot, INTEGER_OF_TRITS), "integer_of_trits differs from astcenc"

    # Endpoint unquant tables vs astcenc's scrambled pquant -> uquant tables.
    quant = (_ASTCENC_SRC / "astcenc_quantization.cpp").read_text()
    for levels in (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        ref = np.array(_parse_c_array(quant, f"color_scrambled_pquant_to_uquant_q{levels}[{levels}]"))
        unq, _ = endpoint_unquant_table(levels)
        assert np.array_equal(ref, unq), f"QUANT_{levels} endpoint unquant differs from astcenc"

    # Endpoint quant derivation vs astcenc's quant_mode_table[n/2][bits].
    qmt = np.array(_parse_c_array(quant, "quant_mode_table[10][128]")).reshape(10, 128)
    for n_values in (4, 6, 8):
        for color_bits in range(1, 128):
            ref_idx = qmt[n_values // 2, color_bits]
            if ref_idx < QUANT_LEVELS.index(6):
                ref_idx = -1  # physical_to_symbolic rejects blocks below QUANT_6
            try:
                mine = QUANT_LEVELS.index(derive_endpoint_quant(n_values, color_bits))
            except ValueError:
                mine = -1
            ref_levels = QUANT_LEVELS[ref_idx] if ref_idx >= 0 else -1
            mine_levels = QUANT_LEVELS[mine] if mine >= 0 else -1
            assert ref_idx == mine, f"quant derivation: n={n_values} bits={color_bits}: astcenc {ref_levels}, ours {mine_levels}"
    assert derive_endpoint_quant(6, 47) == 192 and derive_endpoint_quant(8, 45) == 48
    assert derive_endpoint_quant(4, 47) == 256 and derive_endpoint_quant(6, 63) == 256
    assert ise_bitcount(6, 192) == 46 and ise_bitcount(8, 48) == 45 and ise_bitcount(4, 256) == 32

    # Weight unquant tables vs astcenc.
    xfer = (_ASTCENC_SRC / "astcenc_weight_quant_xfer_tables.cpp").read_text()
    for levels, marker in ((4, "// QUANT_4, range 0..3"), (8, "// QUANT_8, range 0..7"),
                           (16, "// QUANT_16, range 0..15"), (32, "// QUANT_32, range 0..31")):
        idx = xfer.index(marker)
        ref = np.array(_parse_c_array(xfer[idx:], "{"))[:levels]
        assert np.array_equal(ref, weight_unquant_table(levels)), f"QUANT_{levels} weight unquant differs from astcenc"

    # ISE round trips.
    g = torch.Generator().manual_seed(0)
    for levels in (4, 8, 16, 32, 48, 192, 256):
        for k in (4, 6, 8, 16, 32):
            codes = torch.randint(0, levels, (1000, k), generator=g)
            bits = encode_ise_torch(codes, levels)
            back = decode_ise(bits.numpy(), k, levels)
            assert np.array_equal(back, codes.numpy()), f"ISE round trip failed for QUANT_{levels}, {k} values"
    print("astc_ise self-test OK (tables match vendored astcenc)")


def _emit_tables(lang: str) -> None:
    """Prints the constant tables for the GPU kernels: `metal` for
    c_api/src/shaders/anbc_astc_tables.metal, `glsl` for
    c_api/src/shaders/glsl/anbc_astc_tables.glsl (GLSL has no 8-bit type, so
    the byte tables become uint arrays)."""
    glsl = lang == "glsl"

    def arr(name: str, values, per_line: int = 16, ty: str = "uchar") -> str:
        vals = [int(v) for v in values]
        lines = [", ".join(str(v) for v in vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
        if glsl:
            ty = "uint" if ty == "uchar" else ty
            return f"const {ty} {name}[{len(vals)}] = {ty}[{len(vals)}](\n    " + ",\n    ".join(lines) + "\n);\n"
        return f"constant {ty} {name}[{len(vals)}] = {{\n    " + ",\n    ".join(lines) + "\n};\n"

    guard = "ANBC_ASTC_TABLES_GLSL" if glsl else "ANBC_ASTC_TABLES_METAL"
    out = [f"// ASTC constant tables. Generated by `python src/astc_ise.py --emit-{lang}`, do not edit.\n"
           "// Verified against the vendored astcenc sources by `python src/astc_ise.py`.\n"
           f"#ifndef {guard}\n#define {guard}\n\n"
           "// T byte of a 5-trit ISE block, indexed t4*81 + t3*27 + t2*9 + t1*3 + t0.\n"]
    out.append(arr("kAstcIntegerOfTrits", INTEGER_OF_TRITS_FLAT))
    for levels in (48, 192):
        unq, code_of_value = endpoint_unquant_table(levels)
        out.append(f"// QUANT_{levels}: decoded 8-bit value of each ISE code, and the code nearest to each 8-bit value.\n")
        out.append(arr(f"kAstcUnquant{levels}", unq))
        out.append(arr(f"kAstcCodeOfValue{levels}", code_of_value))
    for levels in (4, 8, 16):
        out.append(f"// QUANT_{levels} weights (unquantized 0..64, as blend factors).\n")
        w = weight_unquant_table(levels)
        body = ", ".join(f"{v}.0 / 64.0" if glsl else f"{v}.0f / 64" for v in w)
        if glsl:
            out.append(f"const float kAstcWeights{levels}[{levels}] = float[{levels}]({body});\n")
        else:
            out.append(f"constant float kAstcWeights{levels}[{levels}] = {{ {body} }};\n")
    out.append(f"\n#endif // {guard}\n")
    print("".join(out), end="")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASTC ISE self-test / table emitter")
    parser.add_argument("--emit-metal", action="store_true", help="print the constant tables for the Metal kernels")
    parser.add_argument("--emit-glsl", action="store_true", help="print the constant tables for the Vulkan (GLSL) kernels")
    args = parser.parse_args()
    if args.emit_metal:
        _emit_tables("metal")
    elif args.emit_glsl:
        _emit_tables("glsl")
    else:
        _self_test()
