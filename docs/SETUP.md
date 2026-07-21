# Local setup

The canonical setup is defined in `docs/INSTALL.md`, `pyproject.toml`, and
`uv.lock`.

```bash
cd /home/cosmosapjw/Dropbox/personal_projects/quartz
scripts/setup_cuda_venv.sh --recreate
cargo build --release --locked
venv/bin/python -m quartz.train --help
```

Expected host contract:

- Ubuntu 24.04
- `/usr/bin/python3.12`
- NVIDIA RTX 3080 Ti
- NVIDIA driver new enough for the locked CUDA 12.8 PyTorch wheel
- no JAX dependency
- no Docker-based runtime

Before a long ablation campaign, verify the interpreter and live device:

```bash
venv/bin/python -c \
  'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi
```

For a pip-only installation, use the ordered commands in `docs/INSTALL.md` so
the CUDA wheel source is explicit. For dependency changes, prefer `uv add` or
`uv remove` and commit the resulting `pyproject.toml` and `uv.lock` together.
