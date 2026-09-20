"""Export trained MLP checkpoints to the flat binary format the C library
(c_api/) loads at runtime via anbcLoadModel().

Layout (little-endian):
    char[4]  magic       "ANBC"
    u32      version     2
    u32      format      7 = BC7 (the texture format the network encodes)
    u32      mode        5 or 6 (the BC7 mode the network predicts)
    u32      numLayers   number of Linear layers
    u32      dims[numLayers + 1]   dims[0] = input, dims[i] = output of layer i-1
    then for each layer i:
        f32 W[dims[i+1]][dims[i]]   (row-major, torch's (out, in) layout)
        f32 b[dims[i+1]]

(Version 1 files had no `format` field and were always BC7; the C loader
still reads them.)

Only the BC7 mode 6 / mode 5 checkpoints exist: BC5, BC6H and ASTC are
encoded without a network (see notes.txt).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
MODELS_DIR = Path(__file__).resolve().parent.parent / "c_api" / "models"  # embedded in the library's blob
FORMAT_BC7 = 7
# checkpoint mode -> (mode word, output file name)
EXPORT_INFO = {
    "mode6": (6, "bc7_mode6.bin"),
    "mode5": (5, "bc7_mode5.bin"),
}


def export(checkpoint: Path, output: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    mode = ckpt.get("mode", "mode6")
    if mode not in EXPORT_INFO:
        raise ValueError(f"{checkpoint}: unknown mode {mode}")

    state = ckpt["model_state"]
    # Linear layers are net.0, net.2, net.4, ... (ReLUs in between have no params).
    layer_ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("net.")})
    weights = [(state[f"net.{i}.weight"].numpy(), state[f"net.{i}.bias"].numpy()) for i in layer_ids]

    dims = [weights[0][0].shape[1]] + [w.shape[0] for w, _ in weights]
    for (w, b), din, dout in zip(weights, dims[:-1], dims[1:]):
        assert w.shape == (dout, din) and b.shape == (dout,)

    mode_num, _ = EXPORT_INFO[mode]
    with open(output, "wb") as f:
        f.write(b"ANBC")
        f.write(struct.pack("<IIII", 2, FORMAT_BC7, mode_num, len(weights)))
        f.write(struct.pack(f"<{len(dims)}I", *dims))
        for w, b in weights:
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())

    print(f"{checkpoint.name} -> {output} (BC7 mode {mode_num}, dims {dims}, {output.stat().st_size} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Checkpoint to export (repeatable). Default: checkpoints/bc7_mode6_mlp.pt and bc7_mode5_mlp.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    checkpoints = args.checkpoint or [CHECKPOINT_DIR / "bc7_mode6_mlp.pt", CHECKPOINT_DIR / "bc7_mode5_mlp.pt"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in checkpoints:
        mode = torch.load(ckpt, map_location="cpu").get("mode", "mode6")
        export(ckpt, args.out_dir / EXPORT_INFO.get(mode, (0, f"{mode}.bin"))[1])
