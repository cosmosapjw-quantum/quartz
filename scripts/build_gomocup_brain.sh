#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_NAME="${1:-pbrain-quartz}"
SOURCE_BIN="target/release/mcts_demo"
OUTPUT_BIN="target/release/${TARGET_NAME}"

echo "[gomocup] building release binary..."
cargo build --release

cp "$SOURCE_BIN" "$OUTPUT_BIN"
chmod +x "$OUTPUT_BIN"

echo "[gomocup] wrote ${OUTPUT_BIN}"
echo "[gomocup] launch with: ./${OUTPUT_BIN}"
