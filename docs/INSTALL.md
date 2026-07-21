# QUARTZ installation

The supported local research baseline is Ubuntu 24.04, CPython 3.12, and the
NVIDIA CUDA PyTorch wheels locked in `uv.lock`. The current host target is an
RTX 3080 Ti (compute capability 8.6). JAX is not required by the active
training and Idea Foundry paths and is intentionally absent from the baseline.
Docker installation artifacts are likewise not maintained.

## Canonical uv install

Install `uv` once if it is not already available:

```bash
python3.12 -m pip install --user uv
```

Then recreate the repo-local environment from the lock file:

```bash
scripts/setup_cuda_venv.sh --recreate
```

The script replaces only `./venv`, uses `/usr/bin/python3.12` by default, and
runs the equivalent of:

```bash
uv venv --python /usr/bin/python3.12 --clear venv
UV_PROJECT_ENVIRONMENT="$PWD/venv" uv sync --locked --all-groups
```

Use `uv add`, `uv remove`, and `uv lock` when changing dependencies. Do not
manually edit resolved versions in `uv.lock`.

## pip-compatible install

`pip` users must install the CUDA wheel family from the PyTorch CUDA 12.8
index before installing the project. The NVIDIA driver may report CUDA 13.x;
that is the driver's maximum API compatibility, not a requirement that the
PyTorch wheel use the same toolkit minor version.

```bash
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip 'setuptools>=77,<82' wheel
venv/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
venv/bin/python -m pip install -e '.[dev,onnx-gpu,search]'
```

The pip route is compatible but not fully lock-reproducible. Use the uv route
for campaign evidence.

## Runtime proof

Environment installation is not proof that CUDA execution works. Run this
after installation and after driver changes:

```bash
venv/bin/python - <<'PY'
import torch

assert torch.version.cuda is not None
assert torch.version.hip is None
assert torch.cuda.is_available()
device = torch.device("cuda:0")
x = torch.randn(2048, 2048, device=device)
y = x @ x
torch.cuda.synchronize()
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(float(y.norm()))
del x, y
torch.cuda.empty_cache()
PY
```

Also inspect ownership rather than memory totals alone:

```bash
nvidia-smi
venv/bin/python scripts/idea_foundry_preflight.py \
  --run-id local-python312-cuda-preflight \
  --python venv/bin/python --mode quick
```

PyTorch's caching allocator can retain process-owned VRAM between steps. It is
released when the worker exits; `torch.cuda.empty_cache()` only returns unused
cached blocks while the process remains alive. Persistent multi-step workers
must explicitly delete step-local models/tensors, run Python garbage
collection, synchronize the device, and clear the cache at step boundaries.

## CPU isolation and load contract

A15 CPU/CUDA wall-clock comparisons are sensitive to unrelated CPU work. The
runner now samples per-CPU utilization, audits SMT siblings, records the
process affinity before and after pinning, and inventories overlapping
high-CPU processes. The `full` profile fails closed when the preregistered
thresholds in `configs/a15_matched_service_curve.v1.json` are exceeded.

Affinity to one logical CPU is only pinning. It is reported as kernel-level
isolation only when `/sys/devices/system/cpu/isolated` contains that CPU and
the load guard passes. Stop or separately pin unrelated campaigns before a
full A15 run; diagnostic runs preserve contention evidence but cannot support
a controlled timing comparison.

## Verification

```bash
venv/bin/python -m pytest -q tests/test_host_resources.py \
  tests/test_a15_matched_service_curve.py
venv/bin/ruff format --check .
venv/bin/ruff check .
cargo test --release --locked
```

Use `venv/bin/python` or `UV_PROJECT_ENVIRONMENT="$PWD/venv" uv run ...` so a
shell-level environment from another project cannot silently select a
different interpreter.
