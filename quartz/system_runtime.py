"""Hardware detection and runtime autotune helpers."""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass


def _torch_module():
    import torch

    return torch


@dataclass
class HardwareSpec:
    logical_cpus: int
    physical_cpus: int
    memory_mb: int
    gpu_vendor: str = "none"
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    gpu_count: int = 0
    torch_cuda: bool = False
    device_kind: str = "cpu"


def _detect_cpu_counts():
    logical = 0
    try:
        logical = len(os.sched_getaffinity(0))
    except Exception:
        logical = os.cpu_count() or 1
    logical = max(1, logical)

    physical = 0
    if sys.platform.startswith("linux"):
        try:
            pairs = set()
            current_phys = None
            current_core = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        current_phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        current_core = line.split(":", 1)[1].strip()
                    elif not line.strip():
                        if current_phys is not None and current_core is not None:
                            pairs.add((current_phys, current_core))
                        current_phys = None
                        current_core = None
            physical = len(pairs)
        except Exception:
            physical = 0
    if physical <= 0:
        physical = max(1, logical // 2)
    return logical, physical


def _detect_memory_mb():
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except Exception:
            pass
    return 0


def detect_hardware_spec(device, detect_gpu_fn=None):
    torch = _torch_module()
    logical, physical = _detect_cpu_counts()
    memory_mb = _detect_memory_mb()

    gpu_vendor = "none"
    gpu_name = ""
    gpu_vram_mb = 0
    gpu_count = 0
    torch_cuda = bool(torch.cuda.is_available())
    device_kind = getattr(device, "type", str(device))

    if detect_gpu_fn is not None:
        try:
            gpu_info = detect_gpu_fn()
            gpu_vendor = gpu_info.vendor or gpu_vendor
            gpu_name = gpu_info.device_name or gpu_name
            gpu_vram_mb = gpu_info.vram_mb or gpu_vram_mb
        except Exception:
            pass

    if torch_cuda:
        try:
            gpu_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            gpu_name = props.name or gpu_name
            gpu_vram_mb = max(gpu_vram_mb, int(props.total_memory // (1024 * 1024)))
            if gpu_vendor == "none":
                gpu_vendor = "cuda"
        except Exception:
            gpu_count = max(gpu_count, 1)

    return HardwareSpec(
        logical_cpus=logical,
        physical_cpus=physical,
        memory_mb=memory_mb,
        gpu_vendor=gpu_vendor,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        gpu_count=gpu_count,
        torch_cuda=torch_cuda,
        device_kind=device_kind,
    )


def configure_torch_rocm_runtime(hw):
    torch = _torch_module()
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        return
    gpu_name = (hw.gpu_name or "").lower()
    unsupported_lt = (
        "gfx1030" in gpu_name
        or "rx 6950" in gpu_name
        or "rx 6900" in gpu_name
        or "rx 6800" in gpu_name
    )
    if not unsupported_lt:
        return
    preferred_blas = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if preferred_blas is None:
        return
    downgraded = False
    try:
        preferred_blas("hipblas")
        downgraded = True
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore",
        message="Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas",
        category=UserWarning,
    )
    # Print an explicit one-line banner so users on a known-degraded
    # RDNA2 / gfx1030 GPU understand which BLAS backend is actually
    # running. This is documented in docs/INSTALL.md under "Known-degraded
    # GPUs". The environment variable QUARTZ_SILENCE_ROCM_BANNER=1 can
    # opt out for scripted runs where stderr chatter is undesirable.
    if downgraded and not os.environ.get("QUARTZ_SILENCE_ROCM_BANNER"):
        label = hw.gpu_name or "unknown AMD GPU"
        print(
            f"[quartz] ROCm on {label}: hipBLASLt unsupported — "
            f"downgraded to hipBLAS. Expect 30-50% throughput vs CUDA-equivalent; "
            f"see docs/INSTALL.md § Known-degraded GPUs.",
            file=sys.stderr,
            flush=True,
        )


def recommend_eval_parallel_workers(hw, cfg, eval_games, rust_ok):
    if eval_games <= 1:
        return 1
    thread_cost = max(1, int(cfg.get("n_threads", 1)))
    cpu_capacity = max(1, hw.physical_cpus // thread_cost)
    return max(1, min(cpu_capacity, int(eval_games)))


EVAL_AUTOTUNE_PROFILE_VERSION = 4


def hardware_signature(hw):
    return {
        "logical_cpus": hw.logical_cpus,
        "physical_cpus": hw.physical_cpus,
        "memory_mb": hw.memory_mb,
        "gpu_vendor": hw.gpu_vendor,
        "gpu_name": hw.gpu_name,
        "gpu_vram_mb": hw.gpu_vram_mb,
        "device_kind": hw.device_kind,
    }


def eval_autotune_signature(hw, cfg, eval_games):
    return {
        "hardware": hardware_signature(hw),
        "game": cfg.get("_name"),
        "eval_games": int(eval_games),
        "iters": int(cfg.get("iters", 0)),
        "n_threads": int(cfg.get("n_threads", 1)),
        "batch_size": int(cfg.get("batch_size", 8)),
        "backend": str(cfg.get("_backend_name", "torch")),
        "search_profile": str(cfg.get("search_profile", "quartz")),
        "penalty_mode": str(cfg.get("penalty_mode", "GatedRefresh")),
        "batch_timeout_us": int(cfg.get("batch_timeout_us", 0) or 0),
        "eval_runner_mode": str(cfg.get("_eval_runner_mode", "python_batched")),
        "shared_eval_session": bool(cfg.get("_shared_eval_session", False)),
        "broker_enabled": bool(cfg.get("_broker_enabled", False)),
        "eval_topology_version": 4,
    }


def load_eval_autotune_profile(profile_path, hw, cfg, eval_games):
    if not os.path.exists(profile_path):
        return None
    try:
        with open(profile_path) as f:
            data = json.load(f)
        if data.get("version") != EVAL_AUTOTUNE_PROFILE_VERSION:
            return None
        if data.get("signature") != eval_autotune_signature(hw, cfg, eval_games):
            return None
        workers = int(data.get("workers", 0) or 0)
        return workers if workers > 0 else None
    except Exception:
        return None


def save_eval_autotune_profile(profile_path, hw, cfg, eval_games, workers, benchmarks):
    payload = {
        "version": EVAL_AUTOTUNE_PROFILE_VERSION,
        "signature": eval_autotune_signature(hw, cfg, eval_games),
        "workers": int(workers),
        "benchmarks": benchmarks,
        "saved_at": int(time.time()),
    }
    with open(profile_path, "w") as f:
        json.dump(payload, f, indent=2)


def eval_worker_candidates(hw, cfg, eval_games):
    thread_cost = max(1, int(cfg.get("n_threads", 1)))
    cap = max(1, min(int(eval_games), hw.physical_cpus // thread_cost))
    seeds = [
        1,
        2,
        3,
        max(1, cap // 2),
        max(1, (cap * 2) // 3),
        cap,
    ]
    return [w for w in sorted(set(int(x) for x in seeds)) if 1 <= w <= cap]


def compute_eval_collect_policy(base_target_items, base_timeout_s, batch_items_ema=None, wait_ema_s=None):
    target = max(1, int(base_target_items))
    timeout_s = max(0.0005, float(base_timeout_s))
    items_ema = float(batch_items_ema if batch_items_ema is not None else target)
    wait_ema_s = max(0.0, float(wait_ema_s or 0.0))
    fill_ratio = max(0.0, min(2.0, items_ema / max(float(target), 1.0)))

    if fill_ratio < 0.55:
        timeout_s *= min(4.0, 1.0 / max(fill_ratio, 0.25))
    elif fill_ratio > 0.90:
        timeout_s *= 0.85
        target = min(64, max(target, int(round(items_ema * 1.15))))

    if wait_ema_s > timeout_s * 1.25:
        timeout_s *= 0.8
        target = max(1, min(target, int(round(max(1.0, items_ema)))))

    timeout_s = min(0.02, max(0.0005, timeout_s))
    return target, timeout_s


def max_supported_threads(hw):
    return max(1, int(getattr(hw, "logical_cpus", 1) or 1))


def gpu_host_thread_cap(hw):
    logical = max(1, int(getattr(hw, "logical_cpus", 1) or 1))
    physical = max(1, int(getattr(hw, "physical_cpus", logical) or logical))
    return max(1, min(logical, physical))


def gpu_interop_thread_cap(hw):
    logical = max(1, int(getattr(hw, "logical_cpus", 1) or 1))
    physical = max(1, int(getattr(hw, "physical_cpus", logical) or logical))
    return max(1, min(logical, max(1, physical // 2)))


def auto_device_name():
    torch = _torch_module()
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if sys.platform == "darwin" and mps_backend is not None:
        try:
            if bool(mps_backend.is_available()):
                return "mps"
        except Exception:
            pass
    return "cpu"


def clamp_thread_count(value, hw):
    return max(1, min(int(value), max_supported_threads(hw)))


def clamp_runtime_cfg_to_hardware(cfg, hw):
    out = dict(cfg)
    thread_cap = max_supported_threads(hw)
    if "n_threads" in out:
        out["n_threads"] = max(1, min(int(out["n_threads"]), thread_cap))
    return out


__all__ = [
    "EVAL_AUTOTUNE_PROFILE_VERSION",
    "HardwareSpec",
    "auto_device_name",
    "clamp_runtime_cfg_to_hardware",
    "clamp_thread_count",
    "compute_eval_collect_policy",
    "configure_torch_rocm_runtime",
    "detect_hardware_spec",
    "eval_autotune_signature",
    "eval_worker_candidates",
    "gpu_host_thread_cap",
    "gpu_interop_thread_cap",
    "hardware_signature",
    "load_eval_autotune_profile",
    "max_supported_threads",
    "recommend_eval_parallel_workers",
    "save_eval_autotune_profile",
]
