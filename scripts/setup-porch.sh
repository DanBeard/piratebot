#!/usr/bin/env bash
#
# setup-porch.sh — one-time prep for the Halloween porch PC.
#
# Run this on the gaming PC (RTX 3080) before the kids show up.  It:
#   1. Ensures uv is installed
#   2. Syncs Python dependencies
#   3. Downloads/caches YOLO and Moondream2 models
#   4. (Optional) Mirrors the parrotts voice line cache from the cluster
#   5. Checks for Godot, Rhubarb, and a webcam
#
# Usage:
#     ./scripts/setup-porch.sh
#     ./scripts/setup-porch.sh --skip-voice-mirror   # if parrotts isn't up yet
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

SKIP_VOICE_MIRROR=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-voice-mirror) SKIP_VOICE_MIRROR=1; shift ;;
        -h|--help) sed -n '3,17p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

echo "==> Checking environment"
missing=()

if ! command -v godot &>/dev/null && ! command -v /usr/bin/godot &>/dev/null; then
    missing+=("Godot 4 (download from https://godotengine.org)")
fi

if ! command -v rhubarb &>/dev/null; then
    missing+=("Rhubarb lip-sync (https://github.com/DanielSWolf/rhubarb-lip-sync)")
fi

if ! ls /dev/video* &>/dev/null; then
    echo "WARN: no /dev/video* devices found. Plug in a webcam before showtime."
fi

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing required tools:"
    for m in "${missing[@]}"; do echo "  - $m"; done
    exit 1
fi

if ! command -v uv &>/dev/null; then
    echo "==> Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv is installed to ~/.local/bin by default
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Syncing porch dependencies"
uv sync --extra porch

echo "==> Caching YOLO model"
uv run python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("yolov8n.pt ready")
PY

echo "==> Caching Moondream2 (this will download ~5GB on first run)"
uv run python - <<'PY'
import torch
from services.moondream_vlm import MoondreamVLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
vlm = MoondreamVLM(device=device)
vlm.warmup()
vlm.cleanup()
print("Moondream2 ready")
PY

if (( SKIP_VOICE_MIRROR )); then
    echo "==> Skipping parrotts voice mirror (--skip-voice-mirror)"
else
    echo "==> Mirroring parrotts voice cache"
    echo "    Make sure the cluster parrotts service is available. If not, run:"
    echo "      kubectl port-forward -n default svc/parrotts 18003:8000"
    uv run python tools/mirror_parrotts_cache.py --base-url http://localhost:18003
fi

echo
echo "==> Porch setup complete."
echo "    Next step: ./scripts/run-porch.sh"
