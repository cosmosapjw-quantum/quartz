#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_NAME="pbrain-quartz"
BUNDLE_DIR=""
USE_ONNX=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-name)
      TARGET_NAME="${2:?missing value for --target-name}"
      shift 2
      ;;
    --bundle-dir)
      BUNDLE_DIR="${2:?missing value for --bundle-dir}"
      shift 2
      ;;
    --no-onnx)
      USE_ONNX=0
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      echo "usage: $0 [--target-name NAME] [--bundle-dir DIR] [--no-onnx]" >&2
      exit 2
      ;;
  esac
done

SOURCE_BIN="target/release/mcts_demo"
OUTPUT_BIN="target/release/${TARGET_NAME}"

echo "[gomocup] building release binary..."
if [[ "$USE_ONNX" -eq 1 ]]; then
  cargo build --release --features onnx
else
  cargo build --release
fi

cp "$SOURCE_BIN" "$OUTPUT_BIN"
chmod +x "$OUTPUT_BIN"

echo "[gomocup] wrote ${OUTPUT_BIN}"
if [[ -n "$BUNDLE_DIR" ]]; then
  mkdir -p "$BUNDLE_DIR"
  cp "$OUTPUT_BIN" "$BUNDLE_DIR/${TARGET_NAME}"
  chmod +x "$BUNDLE_DIR/${TARGET_NAME}"
  echo "[gomocup] copied bundle binary to ${BUNDLE_DIR}/${TARGET_NAME}"
fi
echo "[gomocup] launch with: ./${OUTPUT_BIN}"
