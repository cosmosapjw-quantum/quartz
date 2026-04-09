# Setup Guide

## Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default 1.82.0  # for ONNX support; 1.75+ works without ONNX
cargo build --release
cargo test --release
```

## Python
```bash
python3.11 -m venv venv && source venv/bin/activate

# Native AMD ROCm + JAX path for Ubuntu 22.04 / Python 3.11:
# AMD's current Radeon Linux JAX support is inference-only, so QUARTZ
# training with --backend jax on Radeon should be treated as experimental.
pip install --upgrade pip setuptools wheel setuptools_scm
pip install "numpy<2" scipy tqdm "matplotlib>=3.8" pytest ruff "onnx>=1.14" "onnxruntime-rocm>=1.22.1"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
pip uninstall -y jax-rocm7-pjrt jax-rocm7-plugin jaxlib jax ml_dtypes || true
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_pjrt-0.8.2%2Brocm7.2.1-py3-none-manylinux_2_28_x86_64.whl
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_plugin-0.8.2%2Brocm7.2.1-cp311-cp311-manylinux_2_28_x86_64.whl
pip install jaxlib==0.8.2 jax==0.8.2 flax optax
pip install -e . --no-deps
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# One-shot installer:
bash scripts/install_rocm_jax_py311.sh
```

## Docker
```bash
docker compose -f docker/docker-compose.yml run pytorch
```

## Verify
```bash
venv/bin/python -c "import torch; print(torch.cuda.is_available())"
venv/bin/python -m quartz.train --help
venv/bin/python -m quartz.train --game gomoku7 --iterations 2 --device auto
```

`--device auto` resolution order:

1. CUDA
2. Apple MPS
3. CPU
