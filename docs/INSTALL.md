# Installation Guide

## Prerequisites

- **Rust** ≥ 1.70 (recommended: 1.94+)
- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0 (CPU, CUDA, or ROCm)
- **Optional**: JAX (for JAX backend), onnxruntime (for ONNX inference)

Notes:

- `pip install -e .[jax]` is now runnable with the current train entrypoint
  because the `jax` extra includes `torch`.
- `--device auto` now prefers CUDA, then Apple MPS (if available), then CPU.

---

## Option A: Native Install (NVIDIA / CPU)

### 1. Build Rust Engine

```bash
git clone <repo-url> quartz && cd quartz

# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Build
cargo build --release
cargo test --release    # ~270+ tests should pass
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv && source .venv/bin/activate

# NVIDIA GPU (CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install QUARTZ package
pip install -e .
```

### 3. Verify

```bash
./target/release/mcts_demo --server <<< '{"cmd":"quit"}'
venv/bin/python -c "from quartz.evaluation import _run_all; _run_all()"
venv/bin/python -m quartz.train --help
```

---

## Option B: AMD ROCm (Native)

ROCm requires Linux with a supported AMD GPU. For local JAX, prefer AMD's
current official Radeon path:

- Ubuntu 22.04
- Python 3.11
- AMD official JAX wheels (`pjrt -> plugin -> jaxlib -> jax`)

This is the most maintainable local JAX route today. For older RDNA2 cards
such as `RX 6950 XT (gfx1030)`, treat it as a best-effort setup rather than
fully guaranteed support.

Important:

- AMD's current Radeon Linux documentation describes JAX support as
  `inference only`.
- QUARTZ's JAX backend is a training backend, so local JAX training on
  Radeon GPUs should be treated as experimental and unsupported.
- For reliable training on Radeon, prefer the PyTorch backend.

### 1. Install ROCm

```bash
# Ubuntu 22.04.5 — install ROCm 6.2.4 using AMD's official docs/packages

# Verify ROCm installation
rocm-smi              # should list your GPU
hipcc --version       # HIP compiler version
```

### 2. Create Python 3.11 virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip setuptools wheel setuptools_scm
pip install \
  "numpy<2" scipy tqdm "matplotlib>=3.8" \
  pytest ruff "onnx>=1.14" "onnxruntime-rocm>=1.22.1"

# ROCm PyTorch 2.5.1
pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/rocm6.2

# AMD official JAX install order for Ubuntu 22.04 / Python 3.11
pip uninstall -y jax-rocm7-pjrt jax-rocm7-plugin jaxlib jax ml_dtypes || true
pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_pjrt-0.8.2%2Brocm7.2.1-py3-none-manylinux_2_28_x86_64.whl
pip install \
  https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_plugin-0.8.2%2Brocm7.2.1-cp311-cp311-manylinux_2_28_x86_64.whl
pip install jaxlib==0.8.2
pip install jax==0.8.2
pip install flax optax
```

### 4. Install QUARTZ + build Rust engine

```bash
pip install -e . --no-deps
cargo build --release
```

QUARTZ auto-detects ROCm via `rocm-smi` and selects the GPU device
automatically. Override with `--device cpu` if needed.

If you use RDNA2 and ROCm reports an unsupported GPU target, set:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### 5. Verify

```bash
venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
venv/bin/python -c "import onnxruntime as ort; print(ort.get_available_providers())"
venv/bin/python -c "import jax; print(jax.devices())"
venv/bin/python -m quartz.train --game gomoku7 --backend jax --iterations 2 --retune
```

### 6. One-shot installer

This script creates `venv` with Python 3.11, installs all Python
dependencies, installs AMD's official JAX wheel set, and builds Rust.

```bash
bash scripts/install_rocm_jax_py311.sh
```

---

## Option C: Docker (Recommended for ROCm / Reproducibility)

Docker provides the most reliable environment, especially for AMD GPUs
where driver/library version mismatches are common.

### Dockerfile (ROCm + PyTorch + JAX + ONNX)

```dockerfile
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# System dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python dependencies
RUN pip install --no-cache-dir \
    numpy scipy tqdm \
    onnxruntime-rocm \
    && pip install --no-cache-dir \
    jax[rocm] -f https://storage.googleapis.com/jax-releases/rocm/jaxlib-0.4.30+rocm620.find_links.html \
    || echo "JAX ROCm install failed (optional)"

WORKDIR /workspace/quartz
COPY . .

# Build Rust engine
RUN cargo build --release

# Install Python package
RUN pip install -e .

# Verify
RUN cargo test --release 2>&1 | tail -1
RUN python3 -c "from quartz.evaluation import _run_all; _run_all()"

CMD ["python3", "-m", "quartz.train", "--help"]
```

### Dockerfile (CUDA + PyTorch)

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir numpy scipy tqdm onnxruntime-gpu

WORKDIR /workspace/quartz
COPY . .
RUN cargo build --release && pip install -e .

CMD ["python3", "-m", "quartz.train", "--help"]
```

### Build and Run

```bash
# Build image
docker build -t quartz-rocm -f Dockerfile.rocm .

# Run training (AMD GPU passthrough)
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    --shm-size=8g \
    -v $(pwd)/models:/workspace/quartz/models \
    quartz-rocm \
    python3 -m quartz.train --game gomoku7 --iterations 30

# Run training (NVIDIA GPU)
docker run --rm -it --gpus all \
    -v $(pwd)/models:/workspace/quartz/models \
    quartz-cuda \
    python3 -m quartz.train --game gomoku7 --iterations 30

# Run ablation
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
    quartz-rocm \
    cargo test --release -- ablation_vl --ignored --nocapture
```

### Docker Compose (optional)

```yaml
# docker-compose.yml
services:
  quartz-train:
    build:
      context: .
      dockerfile: Dockerfile.rocm
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    shm_size: '8g'
    volumes:
      - ./models:/workspace/quartz/models
    command: python3 -m quartz.train --game gomoku7 --iterations 50
```

---

## ONNX Export and Inference

ONNX enables deployment without PyTorch/JAX. Supported providers:
CPU, CUDA, ROCm, TensorRT, DirectML, CoreML.

### Export

```python
from quartz.onnx_support import export_onnx
from python.alphazero_train import AlphaZeroNet

cfg = {'board': 7, 'actions': 49, 'channels': 32, 'blocks': 2, 'in_channels': 3}
model = AlphaZeroNet(cfg)
model.load_state_dict(torch.load("models/gomoku7_train/best.pt"))

export_onnx(model, cfg, "model.onnx")
```

### Inference

```python
from quartz.onnx_support import OnnxPredictor

pred = OnnxPredictor("model.onnx")
# pred auto-selects best provider: ROCm > CUDA > CPU

policy, value = pred.predict(board_tensor)
```

### Provider Selection

| Provider | Install | Auto-detected |
|----------|---------|---------------|
| CPU | `pip install onnxruntime` | Always |
| CUDA | `pip install onnxruntime-gpu` | If CUDA available |
| ROCm | `pip install onnxruntime-rocm` | If ROCm available |
| TensorRT | `pip install onnxruntime-gpu` + TensorRT | If TensorRT installed |
| DirectML | `pip install onnxruntime-directml` | Windows + DX12 GPU |

---

## JAX Backend

QUARTZ supports JAX as an alternative training backend. JAX is auto-detected
at runtime; no code changes needed.

### Install

```bash
# CPU
pip install jax jaxlib

# CUDA
pip install jax[cuda12]

# ROCm (use Docker for reliability)
pip install jax[rocm]
```

### Usage

```bash
# JAX is auto-selected when available and PyTorch is not
# Or force via backend config in training script
python3 -m quartz.train --game gomoku7 --iterations 30
```

### Limitations

- JAX backend handles the **training loop** (forward/backward/update)
- Self-play NN evaluation currently uses **PyTorch direct path**
- For full JAX end-to-end, the eval responder would need JAX inference
  (not yet implemented)

---

## Hardware Reference

| Setup | Self-play | NN Eval | Notes |
|-------|-----------|---------|-------|
| CPU only | Rust (rayon) | PyTorch CPU | Works, slow for large boards |
| NVIDIA GPU | Rust (rayon) | PyTorch CUDA | Recommended |
| AMD GPU (ROCm) | Rust (rayon) | PyTorch ROCm | Docker recommended |
| Apple Silicon | Rust (rayon) | PyTorch MPS | Auto-detected |

QUARTZ auto-detects your hardware via `quartz.gpu_detect`. The detection
checks (in order): `nvidia-smi`, `rocm-smi`, macOS `sysctl`, and falls
back to CPU.

```python
from quartz.gpu_detect import detect_gpu, install_commands
gpu = detect_gpu()
print(gpu)          # GpuInfo(vendor='amd', arch='gfx1030', driver='rocm-6.2', vram_mb=8192)
print(install_commands(gpu))  # ['pip install torch --index-url ...rocm6.2']
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cargo build` fails with network error | crates.io unreachable | `cargo build --locked` or vendor deps |
| `RuntimeError: Rust binary not found` | Binary not built | `cargo build --release` |
| `ModuleNotFoundError` on import | Package not installed | `pip install -e .` or set PYTHONPATH |
| CUDA out of memory | Model too large | Reduce `--channels` or `--device cpu` |
| ROCm: `hip error` at runtime | Driver mismatch | Match PyTorch ROCm version to installed ROCm |
| ROCm: `hipErrorNoBinaryForGpu` | Unsupported GPU arch | Check GPU is RDNA2+ or set `HSA_OVERRIDE_GFX_VERSION` |
| JAX: `No GPU/TPU found` | Wrong jaxlib build | Reinstall with correct CUDA/ROCm suffix |
| ONNX: `ROCmExecutionProvider not found` | Wrong onnxruntime | `pip install onnxruntime-rocm` |
| Docker: `/dev/kfd permission denied` | User not in video group | `--group-add video` in docker run |
