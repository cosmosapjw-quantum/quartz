# Setup Guide

## Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo build --release
cargo test --release
```

## Python (ROCm 7.2 — AMD GPU)

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip setuptools wheel setuptools_scm
pip install "numpy<2" scipy tqdm "matplotlib>=3.8" pytest ruff onnxscript

# PyTorch (ROCm 7.2)
pip install torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

# JAX (ROCm 7.2, optional — eval stall known issue, use --backend torch)
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_pjrt-0.8.2%2Brocm7.2.1-py3-none-manylinux_2_28_x86_64.whl
pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/jax_rocm7_plugin-0.8.2%2Brocm7.2.1-cp311-cp311-manylinux_2_28_x86_64.whl
pip install jaxlib==0.8.2 jax==0.8.2 flax optax

pip install -e . --no-deps
```

### Environment variables (ROCm)

```bash
export MIOPEN_DEBUG_CONV_GEMM=0          # avoid MIOpen gemm errors on RDNA2
export TORCHINDUCTOR_FX_GRAPH_CACHE=1    # cache torch.compile kernels across processes
# export XLA_FLAGS='--xla_gpu_autotune_level=0'  # needed for JAX backend only
# export QUARTZ_NO_COMPILE=1             # disable torch.compile if segfaults occur
```

## Verify
```bash
venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
venv/bin/python -m quartz.train --help
venv/bin/python -m quartz.train --game gomoku7 --iterations 2 --device auto
venv/bin/python scripts/ablation_study.py \
  --study search_vl \
  --game gomoku7 \
  --iterations 2 \
  --eval-games 4 \
  --eval-interval 1 \
  --seeds 11 \
  --paired-seed-eval \
  --include-strict-reference \
  --resident-session \
  --timeout-hours 1 \
  --output /tmp/quartz_ablation_smoke
```

`--device auto` resolution order: CUDA → Apple MPS → CPU

## Next steps

- Training and ablation flow: [QUICKSTART.md](./QUICKSTART.md)
- Full install notes: [INSTALL.md](./INSTALL.md)
- Ablation studies: [ABLATION_GUIDE.md](./ABLATION_GUIDE.md)
