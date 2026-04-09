#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[install] error: ${PYTHON_BIN} not found. Install Python 3.11 first." >&2
  exit 1
fi

if [[ -d "$VENV_DIR" && "${FORCE_RECREATE_VENV:-0}" == "1" ]]; then
  echo "[install] removing existing virtualenv: ${VENV_DIR}"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[install] creating virtualenv with ${PYTHON_BIN}: ${VENV_DIR}"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="${ROOT_DIR}/${VENV_DIR}/bin/python"
VENV_PIP="${ROOT_DIR}/${VENV_DIR}/bin/pip"

PY_VER="$("$VENV_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" != "3.11" ]]; then
  echo "[install] error: expected Python 3.11 in ${VENV_DIR}, got ${PY_VER}." >&2
  exit 1
fi

echo "[install] repo: $ROOT_DIR"
echo "[install] venv: ${ROOT_DIR}/${VENV_DIR}"
echo "[install] python: $PY_VER"
echo "[install] note: AMD Radeon Linux currently documents JAX as inference-only."
echo "[install] note: QUARTZ --backend jax training on Radeon is experimental."

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "[install] warning: rocm-smi not found. Install ROCm first." >&2
fi

export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
echo "[install] HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"

echo "[install] upgrading build tooling..."
"$VENV_PIP" install --upgrade pip setuptools wheel setuptools_scm

echo "[install] installing core Python deps..."
"$VENV_PIP" install \
  "numpy<2" \
  scipy \
  tqdm \
  "matplotlib>=3.8" \
  pytest \
  ruff \
  "onnx>=1.14" \
  "onnxruntime-rocm>=1.22.1"

echo "[install] installing ROCm PyTorch 2.5.1..."
"$VENV_PIP" install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/rocm6.2

echo "[install] removing incompatible JAX packages..."
"$VENV_PIP" uninstall -y \
  jax-rocm7-pjrt \
  jax-rocm7-plugin \
  jaxlib \
  jax \
  ml_dtypes || true

echo "[install] installing AMD official JAX wheels for Ubuntu 22.04 / Python 3.11..."
"$VENV_PIP" install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_pjrt-0.8.2%2Brocm7.2.1-py3-none-manylinux_2_28_x86_64.whl
"$VENV_PIP" install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_plugin-0.8.2%2Brocm7.2.1-cp311-cp311-manylinux_2_28_x86_64.whl
"$VENV_PIP" install jaxlib==0.8.2
"$VENV_PIP" install jax==0.8.2

echo "[install] installing Flax/Optax..."
"$VENV_PIP" install flax optax

echo "[install] installing QUARTZ package..."
"$VENV_PIP" install -e . --no-deps

echo "[install] building Rust engine..."
cargo build --release

echo "[install] done. activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "[install] verification commands:"
echo "  python -c \"import torch; print(torch.cuda.is_available(), torch.version.hip)\""
echo "  python -c \"import onnxruntime as ort; print(ort.get_available_providers())\""
echo "  python -c \"import jax; print(jax.devices())\""
echo "  python -m quartz.train --game gomoku7 --backend jax --iterations 2 --retune"

rm -rf models/*
