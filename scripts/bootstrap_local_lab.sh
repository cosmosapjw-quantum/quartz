#!/usr/bin/env bash
# Bootstrap a local QUARTZ experiment environment without guessing GPU wheels.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="cpu"
VENV_DIR="venv"
PYTHON_BIN="python3"
TORCH_INDEX_URL=""
SKIP_RUST_TESTS=0
WITH_SMOKE=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_local_lab.sh [options]

Options:
  --profile cpu|cuda|rocm   Runtime profile (default: cpu)
  --venv PATH               Virtual environment directory (default: venv)
  --python EXE              Python interpreter used to create the venv
  --torch-index-url URL     Explicit PyTorch wheel index for CUDA/ROCm
  --skip-rust-tests         Build Rust but skip cargo test --release
  --with-smoke              Run the local idea-lab smoke suite after setup
  -h, --help                Show this help

The script only auto-installs the CPU PyTorch wheel.  For CUDA/ROCm, either
install a matching torch wheel before running this script or pass the exact
--torch-index-url appropriate for the local driver/runtime.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:?missing profile}"; shift 2 ;;
    --venv)
      VENV_DIR="${2:?missing venv path}"; shift 2 ;;
    --python)
      PYTHON_BIN="${2:?missing python executable}"; shift 2 ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:?missing torch index URL}"; shift 2 ;;
    --skip-rust-tests)
      SKIP_RUST_TESTS=1; shift ;;
    --with-smoke)
      WITH_SMOKE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

case "$PROFILE" in
  cpu|cuda|rocm) ;;
  *) echo "invalid --profile: $PROFILE" >&2; exit 2 ;;
esac

command -v "$PYTHON_BIN" >/dev/null || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
command -v cargo >/dev/null || { echo "missing cargo; install Rust first" >&2; exit 2; }
command -v rustc >/dev/null || { echo "missing rustc; install Rust first" >&2; exit 2; }

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 required, found {sys.version.split()[0]}")
PY

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    "$PY" -m pip install torch --index-url "$TORCH_INDEX_URL"
  elif [[ "$PROFILE" == "cpu" ]]; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  else
    cat >&2 <<EOF
PyTorch is not installed in $VENV_DIR.
For $PROFILE, install the wheel matching the local runtime, or rerun with:
  --torch-index-url <official PyTorch wheel index>
No GPU wheel is guessed automatically because a mismatched CUDA/ROCm wheel can
silently invalidate the experiment environment.
EOF
    exit 2
  fi
fi

# QUARTZ + test/analysis dependencies. Torch was handled above deliberately.
"$PY" -m pip install -e '.[dev]'

cargo build --release
if [[ "$SKIP_RUST_TESTS" -eq 0 ]]; then
  cargo test --release
fi

"$PY" scripts/idea_lab.py doctor --profile "$PROFILE" --strict

if [[ "$WITH_SMOKE" -eq 1 ]]; then
  "$PY" scripts/idea_lab.py run \
    --suite smoke \
    --profile "$PROFILE" \
    --output-root results/idea_lab_local
fi

cat <<EOF

Local lab setup complete.

Next commands:
  $PY scripts/idea_lab.py list
  $PY scripts/idea_lab.py plan --suite all-available --profile $PROFILE
  $PY scripts/idea_lab.py run --suite synthetic --profile $PROFILE
EOF
