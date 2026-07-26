#!/usr/bin/env bash
#
# run-porch.sh — start the Halloween show on the porch PC.
#
# This launches the Godot avatar in fullscreen, waits for its WebSocket server,
# then starts the PirateBot orchestrator.  When the script exits (Ctrl+C),
# both processes are cleaned up.
#
# Usage:
#     ./scripts/run-porch.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="${PIRATEBOT_CONFIG:-config.porch.yaml}"
GODOT_BIN="${GODOT_BIN:-godot}"

echo "==> Starting Godot avatar"
"$GODOT_BIN" --path ./godot_project --fullscreen &
GODOT_PID=$!

cleanup() {
    echo "==> Shutting down (Ctrl+C received)"
    if kill -0 "$GODOT_PID" 2>/dev/null; then
        kill "$GODOT_PID" 2>/dev/null || true
        wait "$GODOT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "==> Waiting for Godot WebSocket server on port 9876"
for i in $(seq 1 60); do
    if nc -z localhost 9876 2>/dev/null; then
        echo "==> Godot ready"
        break
    fi
    sleep 1
done

if ! nc -z localhost 9876 2>/dev/null; then
    echo "ERROR: Godot WebSocket server did not start" >&2
    exit 1
fi

echo "==> Starting PirateBot orchestrator with config: $CONFIG"
exec uv run python main.py --config "$CONFIG"
