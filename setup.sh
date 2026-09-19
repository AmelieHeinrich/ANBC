#!/usr/bin/env bash
# One-shot setup: create the venv, install dependencies, download datasets.
# After this, you're ready to run src/train_bc7.py, src/compare_ui.py, etc.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python

echo "=== Creating venv (.venv) with $PYTHON ==="
"$PYTHON" -m venv .venv

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate   # macOS/Linux
else
    source .venv/Scripts/activate  # Windows (git-bash)
fi

echo "=== Installing dependencies ==="
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt

echo "=== Downloading datasets (DIV2K + ambientCG alpha) ==="
python src/download_datasets.py

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate  (or .venv/Scripts/activate on Windows)"
echo "Then train with:"
echo "  python src/train_bc7.py --mode mode6 --epochs 15 --batch-size 32768"
echo "  python src/train_bc7.py --mode mode5 --epochs 60 --val-images 4 --batch-size 4096"
echo "  python src/train_bc7.py --mode bc5 --epochs 15 --batch-size 32768   (needs data/normal, see notes.txt)"
