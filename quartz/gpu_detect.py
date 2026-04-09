"""
QUARTZ GPU Detection & Install Helper
======================================

Auto-detects GPU hardware (ROCm, CUDA, Apple Metal) and recommends
or installs the correct PyTorch/JAX build.

Usage:
  # CLI
  python -m quartz.gpu_detect
  python -m quartz.gpu_detect --install torch
  python -m quartz.gpu_detect --install jax

  # Python API
  from quartz.gpu_detect import detect_gpu, recommend_install
  gpu = detect_gpu()
  print(gpu)  # GpuInfo(vendor='amd', arch='gfx1030', driver='rocm-6.2', vram_mb=8192)
  cmds = recommend_install(gpu)
  print(cmds)  # ['pip install torch --index-url ...rocm6.2']
"""

import os
import re
import sys
import shlex
import shutil
import subprocess
import platform
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class GpuInfo:
    vendor: str = "none"          # "nvidia", "amd", "apple", "intel", "none"
    arch: str = ""                # "sm_86", "gfx1030", "apple_m1", etc.
    driver: str = ""              # "cuda-12.4", "rocm-6.2", etc.
    vram_mb: int = 0
    device_name: str = ""
    compute_capability: str = ""  # CUDA compute capability
    hsa_override: str = ""        # HSA_OVERRIDE_GFX_VERSION if needed
    details: dict = field(default_factory=dict)


# ════════════════════════════════════════════
# § Detection
# ════════════════════════════════════════════

def detect_gpu() -> GpuInfo:
    """Auto-detect GPU hardware. Tries nvidia-smi, rocm-smi, sysctl (macOS)."""
    # Try NVIDIA first
    info = _detect_nvidia()
    if info.vendor != "none":
        return info

    # Try AMD ROCm
    info = _detect_rocm()
    if info.vendor != "none":
        return info

    # Try Apple Metal
    info = _detect_apple()
    if info.vendor != "none":
        return info

    return GpuInfo()


def _detect_nvidia() -> GpuInfo:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
        if not out:
            return GpuInfo()
        parts = out.split(",")
        name = parts[0].strip()
        vram = int(float(parts[1].strip())) if len(parts) > 1 else 0
        driver = parts[2].strip() if len(parts) > 2 else ""
        cc = parts[3].strip() if len(parts) > 3 else ""

        # Detect CUDA toolkit version
        cuda_ver = ""
        try:
            nvcc = subprocess.check_output(["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL, timeout=5)
            m = re.search(r"release (\d+\.\d+)", nvcc)
            if m:
                cuda_ver = m.group(1)
        except Exception:
            # Try from nvidia-smi
            try:
                smi = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL, timeout=5)
                m = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi)
                if m:
                    cuda_ver = m.group(1)
            except Exception:
                pass

        return GpuInfo(
            vendor="nvidia",
            arch=f"sm_{cc.replace('.', '')}" if cc else "",
            driver=f"cuda-{cuda_ver}" if cuda_ver else f"driver-{driver}",
            vram_mb=vram,
            device_name=name,
            compute_capability=cc,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return GpuInfo()


def _detect_rocm() -> GpuInfo:
    # Check if ROCm is installed
    rocm_path = Path("/opt/rocm")
    if not rocm_path.exists() and not shutil.which("rocm-smi"):
        return GpuInfo()

    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )

        name = ""
        vram = 0
        for line in out.splitlines():
            if "GPU" in line and ":" in line:
                name = line.split(":")[-1].strip()
            if "Total" in line and "vram" in line.lower():
                m = re.search(r"(\d+)", line.split(":")[-1])
                if m:
                    vram = int(m.group(1)) // (1024 * 1024)  # bytes to MB

        # Get GFX version
        arch = ""
        try:
            agent_out = subprocess.check_output(
                ["rocm-smi", "--showuniqueid"], text=True, stderr=subprocess.DEVNULL, timeout=5
            )
        except Exception:
            pass

        try:
            agent_out = subprocess.check_output(
                ["rocminfo"], text=True, stderr=subprocess.DEVNULL, timeout=5
            )
            for line in agent_out.splitlines():
                if "gfx" in line.lower():
                    m = re.search(r"(gfx\w+)", line)
                    if m:
                        arch = m.group(1)
                        break
        except Exception:
            pass

        # ROCm version
        rocm_ver = ""
        ver_file = rocm_path / "include" / "rocm-core" / "rocm_version.h"
        if not ver_file.exists():
            ver_file = rocm_path / ".info" / "version"
        if ver_file.exists():
            try:
                content = ver_file.read_text()
                m = re.search(r"(\d+\.\d+)", content)
                if m:
                    rocm_ver = m.group(1)
            except Exception:
                pass

        # HSA override needed for some GPUs
        hsa_override = ""
        if arch:
            # Known overrides: gfx1031→10.3.0, gfx1032→10.3.0, etc.
            overrides = {
                "gfx1031": "10.3.0", "gfx1032": "10.3.0", "gfx1033": "10.3.0",
                "gfx1034": "10.3.0", "gfx1035": "10.3.0", "gfx1036": "10.3.0",
                "gfx1100": "11.0.0", "gfx1101": "11.0.0", "gfx1102": "11.0.0",
                "gfx1103": "11.0.0",
            }
            hsa_override = overrides.get(arch, "")

        return GpuInfo(
            vendor="amd",
            arch=arch,
            driver=f"rocm-{rocm_ver}" if rocm_ver else "rocm",
            vram_mb=vram,
            device_name=name,
            hsa_override=hsa_override,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return GpuInfo()


def _detect_apple() -> GpuInfo:
    if platform.system() != "Darwin":
        return GpuInfo()

    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, timeout=5
        ).strip()
        is_apple_silicon = "Apple" in out

        if is_apple_silicon:
            chip = "m1"
            m = re.search(r"(M\d+\s*(Pro|Max|Ultra)?)", out, re.IGNORECASE)
            if m:
                chip = m.group(1).strip().lower().replace(" ", "_")

            return GpuInfo(
                vendor="apple",
                arch=f"apple_{chip}",
                driver="mps",
                device_name=out,
            )
    except Exception:
        pass

    return GpuInfo()


# ════════════════════════════════════════════
# § Install Recommendations
# ════════════════════════════════════════════

def recommend_install(gpu: GpuInfo, framework: str = "auto") -> dict:
    """Recommend pip install commands based on detected GPU.

    Args:
        gpu: detected GPU info
        framework: "torch", "jax", "onnx", or "auto" (all applicable)

    Returns:
        dict with keys: torch, jax, onnx, env (environment variables)
    """
    result = {"torch": [], "jax": [], "onnx": [], "env": {}}

    if gpu.vendor == "nvidia":
        cuda_major = ""
        if gpu.driver:
            m = re.search(r"cuda-(\d+)", gpu.driver)
            if m:
                cuda_major = m.group(1)

        # PyTorch
        if cuda_major and int(cuda_major) >= 12:
            result["torch"] = [f"pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_major}1"]
        elif cuda_major:
            result["torch"] = [f"pip install torch --index-url https://download.pytorch.org/whl/cu{cuda_major}0"]
        else:
            result["torch"] = ["pip install torch"]

        # JAX
        result["jax"] = ["pip install 'jax[cuda12]'"]

        # ONNX Runtime
        result["onnx"] = ["pip install onnxruntime-gpu"]

    elif gpu.vendor == "amd":
        rocm_ver = ""
        if gpu.driver:
            m = re.search(r"rocm-(\d+\.\d+)", gpu.driver)
            if m:
                rocm_ver = m.group(1)
        rocm_short = rocm_ver.replace(".", "") if rocm_ver else "62"

        # PyTorch
        result["torch"] = [
            f"pip install torch --index-url https://download.pytorch.org/whl/rocm{rocm_ver or '6.2'}"
        ]

        # JAX (ROCm)
        result["jax"] = [
            "pip install jax[rocm] -f https://storage.googleapis.com/jax-releases/rocm/jaxlib-*.whl"
        ]

        # ONNX Runtime ROCm
        result["onnx"] = ["pip install onnxruntime-rocm"]

        # Environment
        if gpu.hsa_override:
            result["env"]["HSA_OVERRIDE_GFX_VERSION"] = gpu.hsa_override

    elif gpu.vendor == "apple":
        result["torch"] = ["pip install torch"]  # MPS built-in
        result["jax"] = ["pip install 'jax[metal]'"]
        result["onnx"] = ["pip install onnxruntime"]

    else:
        result["torch"] = ["pip install torch --index-url https://download.pytorch.org/whl/cpu"]
        result["jax"] = ["pip install jax"]
        result["onnx"] = ["pip install onnxruntime"]

    # Always include quartz extras
    if framework == "auto" or framework == "torch":
        result["torch"].append("pip install numpy tqdm")
    if framework == "auto" or framework == "jax":
        result["jax"].append("pip install flax optax")

    return result


def install_deps(gpu: GpuInfo = None, framework: str = "torch", dry_run: bool = False):
    """Auto-install dependencies based on detected GPU.

    Args:
        gpu: GpuInfo (auto-detected if None)
        framework: "torch", "jax", "onnx"
        dry_run: print commands without executing
    """
    if gpu is None:
        gpu = detect_gpu()

    recs = recommend_install(gpu, framework)

    # Set environment variables
    for k, v in recs.get("env", {}).items():
        if dry_run:
            print(f"export {k}={v}")
        else:
            os.environ[k] = v

    # Run pip installs
    cmds = recs.get(framework, recs.get("torch", []))
    for cmd in cmds:
        if dry_run:
            print(cmd)
        else:
            print(f"Running: {cmd}")
            subprocess.check_call(shlex.split(cmd))


# ════════════════════════════════════════════
# § Dockerfile Generator
# ════════════════════════════════════════════

def generate_dockerfile(gpu: GpuInfo = None, framework: str = "torch") -> str:
    """Generate a Dockerfile for the detected GPU environment."""
    if gpu is None:
        gpu = detect_gpu()

    if gpu.vendor == "nvidia":
        cuda_tag = "12.4.0" if "12" in gpu.driver else "11.8.0"
        base = f"nvidia/cuda:{cuda_tag}-devel-ubuntu22.04"
    elif gpu.vendor == "amd":
        rocm_ver = "6.2" if "6.2" in gpu.driver else "6.1"
        base = f"rocm/dev-ubuntu-22.04:{rocm_ver}-complete"
    else:
        base = "ubuntu:22.04"

    recs = recommend_install(gpu, framework)
    pip_cmds = recs.get(framework, recs.get("torch", []))
    env_lines = "\n".join(f"ENV {k}={v}" for k, v in recs.get("env", {}).items())
    pip_lines = "\n".join(f"RUN {cmd}" for cmd in pip_cmds)

    ort_suffix = "-gpu" if gpu.vendor == "nvidia" else "-rocm" if gpu.vendor == "amd" else ""
    ort_pkg = f"onnxruntime{ort_suffix}"

    lines = [
        f"# QUARTZ AlphaZero — Auto-generated Dockerfile",
        f"# GPU: {gpu.vendor} {gpu.device_name} ({gpu.arch})",
        f"FROM {base}",
        "",
        "RUN apt-get update && apt-get install -y \\",
        "    python3 python3-pip git curl build-essential && \\",
        "    rm -rf /var/lib/apt/lists/*",
        "",
        "# Rust toolchain",
        "RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        'ENV PATH="/root/.cargo/bin:${PATH}"',
        "",
    ]
    if env_lines:
        lines.append(env_lines)
        lines.append("")
    lines += [
        "WORKDIR /app",
        "COPY . .",
        "",
        "# Python dependencies",
        pip_lines,
        f"RUN pip install {ort_pkg}",
        "",
        "# Build Rust MCTS engine",
        "RUN cargo build --release && cargo test --release",
        "",
        "# Install Python package",
        "RUN pip install -e .",
        "",
        'ENTRYPOINT ["quartz-train"]',
        'CMD ["--game", "gomoku15", "--iterations", "50"]',
    ]
    return "\n".join(lines) + "\n"


# ════════════════════════════════════════════
# § CLI
# ════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="QUARTZ GPU Detection & Install Helper")
    parser.add_argument("--install", choices=["torch", "jax", "onnx", "all"],
                        help="Install dependencies for specified framework")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--dockerfile", action="store_true",
                        help="Generate Dockerfile for detected GPU")
    parser.add_argument("--json", action="store_true",
                        help="Output GPU info as JSON")
    args = parser.parse_args()

    gpu = detect_gpu()

    if args.json:
        import json
        print(json.dumps(gpu.__dict__, indent=2))
        return

    # Print detection results
    print(f"GPU Detection:")
    print(f"  Vendor:  {gpu.vendor}")
    print(f"  Device:  {gpu.device_name or '(none)'}")
    print(f"  Arch:    {gpu.arch or '(unknown)'}")
    print(f"  Driver:  {gpu.driver or '(unknown)'}")
    print(f"  VRAM:    {gpu.vram_mb} MB" if gpu.vram_mb else "  VRAM:    (unknown)")
    if gpu.hsa_override:
        print(f"  HSA:     export HSA_OVERRIDE_GFX_VERSION={gpu.hsa_override}")
    print()

    if args.dockerfile:
        print(generate_dockerfile(gpu))
        return

    if args.install:
        frameworks = ["torch", "jax", "onnx"] if args.install == "all" else [args.install]
        for fw in frameworks:
            install_deps(gpu, fw, dry_run=args.dry_run)
        return

    # Default: show recommendations
    recs = recommend_install(gpu)
    print("Recommended install commands:")
    for fw in ["torch", "jax", "onnx"]:
        cmds = recs.get(fw, [])
        if cmds:
            print(f"\n  {fw}:")
            for cmd in cmds:
                print(f"    {cmd}")
    if recs.get("env"):
        print(f"\n  Environment:")
        for k, v in recs["env"].items():
            print(f"    export {k}={v}")


if __name__ == "__main__":
    main()
