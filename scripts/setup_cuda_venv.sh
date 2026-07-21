#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_path="${repo_root}/venv"
python_path="${QUARTZ_PYTHON:-/usr/bin/python3.12}"
uv_path="${QUARTZ_UV:-$(command -v uv || true)}"

if [[ "${1:-}" != "--recreate" && -e "${env_path}" ]]; then
  echo "${env_path} already exists; pass --recreate to replace only this environment" >&2
  exit 2
fi
if [[ ! -x "${python_path}" ]]; then
  echo "Python 3.12 interpreter not found at ${python_path}" >&2
  exit 2
fi
if [[ -z "${uv_path}" || ! -x "${uv_path}" ]]; then
  echo "uv is required; install it with: python3.12 -m pip install --user uv" >&2
  exit 2
fi
if [[ -L "${env_path}" ]]; then
  echo "refusing to replace symlinked environment: ${env_path}" >&2
  exit 2
fi

"${uv_path}" venv --python "${python_path}" --clear "${env_path}"
UV_PROJECT_ENVIRONMENT="${env_path}" "${uv_path}" sync --locked --all-groups

"${env_path}/bin/python" - <<'PY'
import platform
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda_build", torch.version.cuda)
print("torch_hip_build", torch.version.hip)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
    print("cuda_capability", torch.cuda.get_device_capability(0))
PY
