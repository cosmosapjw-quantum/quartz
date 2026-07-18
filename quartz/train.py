"""QUARTZ training entrypoint."""

from __future__ import annotations

import os
import sys


def _detect_logical_threads() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def _detect_physical_threads(logical: int) -> int:
    try:
        import psutil  # type: ignore

        return max(1, int(psutil.cpu_count(logical=False) or logical))
    except Exception:
        return max(1, logical // 2)


def _arg_value(argv: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _should_prewarm_jax(argv: list[str]) -> bool:
    backend = (_arg_value(argv, "--backend") or "").strip().lower()
    device = (_arg_value(argv, "--device") or "").strip().lower()
    return backend == "jax" or device == "jax"


def _runtime_module_name(argv: list[str]) -> str:
    return "quartz.jax_runtime" if _should_prewarm_jax(argv) else "quartz.torch_runtime"


def _configure_host_thread_env(argv: list[str]) -> None:
    device = (_arg_value(argv, "--device") or "auto").strip().lower()
    logical = _detect_logical_threads()
    physical = _detect_physical_threads(logical)
    # CPU runs may benefit from all logical threads. GPU/JAX runs should keep
    # host-side BLAS/OpenMP pools bounded, but not pinned to 1. Using roughly
    # physical-core capacity prevents helper-thread explosions while still
    # allowing NN preprocessing and host-side batching to parallelize.
    thread_cap = logical if device == "cpu" else max(1, min(logical, physical))
    value = str(max(1, thread_cap))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(key, value)


def _prewarm_jax_rocm(argv: list[str]) -> None:
    if not _should_prewarm_jax(argv):
        return
    if any(arg in {"-h", "--help"} for arg in argv):
        return
    os.environ.setdefault("JAX_PLATFORMS", "rocm,cpu")
    try:
        import jax

        devices = jax.devices()
        print(f"  JAX prewarm devices: {devices}")
    except Exception as exc:
        print(f"  JAX prewarm failed ({exc})", file=sys.stderr)


def main() -> None:
    argv = sys.argv[1:]
    _configure_host_thread_env(argv)
    _prewarm_jax_rocm(argv)
    runtime_module = __import__(_runtime_module_name(argv), fromlist=["main"])
    train_main = runtime_module.main

    train_main()


if __name__ == "__main__":
    main()
