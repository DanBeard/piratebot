#!/usr/bin/env bash
#
# run-portrait.sh — start the "haunted painting" show on the porch PC.
#
# Launches Chrome in fullscreen pointing at the local portrait player, then
# starts the PirateBot orchestrator in portrait mode.
#
# Usage:
#     ./scripts/run-portrait.sh
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="${PIRATEBOT_CONFIG:-config.portrait.yaml}"
CHROME_BIN="${CHROME_BIN:-google-chrome}"
PLAYER_URL="http://localhost:9877/"

echo "==> Starting portrait player in Chrome"
"$CHROME_BIN" \
  --kiosk \
  --app="$PLAYER_URL" \
  --disable-infobars \
  --disable-features=Translate \
  --autoplay-policy=no-user-gesture-required \
  &
CHROME_PID=$!

cleanup() {
  echo "==> Shutting down"
  if kill -0 "$CHROME_PID" 2>/dev/null; then
    kill "$CHROME_PID" 2>/dev/null || true
    wait "$CHROME_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "==> Waiting for portrait server"
for i in $(seq 1 30); do
  if curl -sf "$PLAYER_URL" >/dev/null 2>&1; then
    echo "==> Portrait server ready"
    break
  fi
  sleep 1
done

echo "==> Starting PirateBot orchestrator with config: $CONFIG"
exec uv run python main.py --config "$CONFIG"
