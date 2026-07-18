"""Shared PyTorch inference helpers for QUARTZ runtime paths."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np


# A3-a audit fix: this migrated from the old runtime_support.py design,
# which cached the compiled model as an attribute ON the model object
# itself (object.__setattr__(model, "_quartz_compiled", compiled)). The
# migration replaced that with a separate module-level dict keyed by
# id(model) — a real regression: (a) entries were never evicted,
# leaking one compiled model per distinct model object ever seen for
# the life of the process; (b) id() is a memory address, which CPython
# reuses once an object is freed, so an unrelated later model could
# alias a freed model's id and silently receive a stale compiled
# artifact from a different model.
#
# A WeakKeyDictionary looks like the obvious fix, but is NOT one here:
# `torch.compile(model)` returns an OptimizedModule that stores the
# original module as `self._orig_mod = model` — a STRONG reference
# back to the key. Storing that as a WeakKeyDictionary's value means
# the module-level dict (always GC-reachable) strongly holds
# `compiled`, which strongly holds `model` via `_orig_mod`, regardless
# of how weakly the dict treats the key side. The entry can never be
# evicted — verified by writing the eviction test first
# (test_get_compiled_model_cache_evicts_when_model_is_garbage_collected
# in tests/test_torch_inference_runtime.py) and watching it fail
# against a WeakKeyDictionary-based implementation.
#
# The attribute-based design is the one that actually works: it forms
# an explicit model<->compiled reference CYCLE (attribute one way,
# _orig_mod the other) with no external anchor beyond however many
# places hold `model` itself. Python's cycle-collecting GC (not mere
# refcounting) reclaims exactly this pattern once the caller's own
# reference to `model` is dropped — this is why the pre-migration code
# was correct, and it also sidesteps the id-reuse hazard for free
# (a new object at a reused address starts with a fresh __dict__ and
# no cached-compile attribute, so no aliasing is possible).
_COMPILED_MODEL_ATTR = "_quartz_compiled"
_PINNED_BUFS: dict[tuple[str, int, int, int], tuple[Any, Any, int]] = {}


def _import_torch():
    import torch

    return torch


def _resolve_torch(
    torch_module: Any | None = None,
    torch_module_factory: Callable[[], Any] | None = None,
):
    if torch_module is not None:
        return torch_module
    if torch_module_factory is not None:
        return torch_module_factory()
    return _import_torch()


def _device_type(device: Any) -> str:
    return str(getattr(device, "type", device)).lower()


def _is_cpu_device(device: Any) -> bool:
    return _device_type(device).startswith("cpu")


def _compile_disabled() -> bool:
    return (
        bool(os.environ.get("QUARTZ_DISABLE_COMPILE"))
        or os.environ.get("QUARTZ_NO_COMPILE") == "1"
    )


def coerce_batch_features(batch_features: Any) -> np.ndarray:
    """Return a contiguous, writable float32 batch array."""
    batch_np = np.asarray(batch_features, dtype=np.float32)
    if not batch_np.flags.c_contiguous:
        batch_np = np.ascontiguousarray(batch_np)
    if not batch_np.flags.writeable:
        batch_np = batch_np.copy()
    return batch_np


def get_compiled_model(
    model: Any,
    *,
    device: Any = None,
    torch_module: Any | None = None,
    torch_module_factory: Callable[[], Any] | None = None,
    compile_on_cpu: bool = False,
) -> Any:
    """Lazily compile a model while preserving eager fallback semantics.

    NOTE (A3-a, deferred): the compile call below uses
    `mode="default", dynamic=True`, introduced by the migration from
    the old runtime_support.py's `torch.compile(model,
    backend="inductor")`. `dynamic=True` may exist specifically to
    avoid a recompilation storm across phase15's varying per-budget
    batch sizes — reverting to `backend="inductor")` without measuring
    "batch-size distribution x recompilation count" on real GPU
    hardware could reintroduce that storm just as easily as it could
    fix a real regression. This environment has no GPU
    (`torch.cuda.is_available()` is False here), so that benchmark
    cannot be run in this session; the mode/backend choice is
    deliberately left as-is pending hardware access, per this plan's
    own adversarial-review finding
    (~/.claude/plans/parallel-popping-owl.md §A3-a). Only the cache
    lifetime bug below is fixed now — it doesn't depend on GPU
    benchmarking.
    """
    if _compile_disabled() or (
        device is not None and _is_cpu_device(device) and not compile_on_cpu
    ):
        return model
    compiled = getattr(model, _COMPILED_MODEL_ATTR, None)
    if compiled is not None:
        return compiled
    torch = _resolve_torch(torch_module, torch_module_factory)
    if not hasattr(torch, "compile"):
        return model
    try:
        compiled = torch.compile(model, mode="default", dynamic=True)
    except Exception:
        compiled = model
    try:
        object.__setattr__(model, _COMPILED_MODEL_ATTR, compiled)
    except (AttributeError, TypeError):
        # Model doesn't support arbitrary attribute assignment (e.g. a
        # __slots__ type without this attr, or a non-object-like
        # callable) — skip caching rather than crash; every call
        # recompiles, matching the pre-cache eager fallback
        # correctness (just without the speed benefit).
        pass
    return compiled


def get_inference_buffers(
    device: Any,
    batch_np: np.ndarray,
    *,
    torch_module: Any | None = None,
    torch_module_factory: Callable[[], Any] | None = None,
):
    """Return a reusable pinned-host/GPU tensor slice for 4D CUDA batches."""
    if batch_np.ndim != 4:
        return None
    torch = _resolve_torch(torch_module, torch_module_factory)
    bs, channels, height, width = batch_np.shape
    key = (str(device), channels, height, width)
    entry = _PINNED_BUFS.get(key)
    if entry is not None:
        pinned, gpu, max_bs = entry
        if bs <= max_bs:
            pinned[:bs].copy_(torch.from_numpy(batch_np))
            gpu[:bs].copy_(pinned[:bs], non_blocking=True)
            return gpu[:bs]

    max_bs = max(bs * 2, 64)
    pinned = torch.zeros(
        max_bs, channels, height, width, dtype=torch.float32
    ).pin_memory()
    gpu = torch.zeros(
        max_bs, channels, height, width, dtype=torch.float32, device=device
    )
    _PINNED_BUFS[key] = (pinned, gpu, max_bs)
    pinned[:bs].copy_(torch.from_numpy(batch_np))
    gpu[:bs].copy_(pinned[:bs], non_blocking=True)
    return gpu[:bs]


def _tensor_to_device(
    torch, batch_np: np.ndarray, device: Any, *, use_pinned_transfer: bool
):
    if _is_cpu_device(device):
        return torch.from_numpy(batch_np).to(device)
    if use_pinned_transfer:
        buffered = get_inference_buffers(device, batch_np, torch_module=torch)
        if buffered is not None:
            return buffered
        try:
            return torch.from_numpy(batch_np).pin_memory().to(device, non_blocking=True)
        except Exception:
            pass
    return torch.from_numpy(batch_np).to(device)


def run_model_batch(
    model: Any,
    device: Any,
    batch_features: Any,
    *,
    torch_module: Any | None = None,
    torch_module_factory: Callable[[], Any] | None = None,
    compile_on_cpu: bool = False,
    use_pinned_transfer: bool = True,
):
    """Run a batch through either a predictor object or a PyTorch module."""
    batch_np = coerce_batch_features(batch_features)
    if hasattr(model, "predict"):
        probs_batch, vals_np = model.predict(batch_np)
        return np.asarray(probs_batch, dtype=np.float32), np.asarray(
            vals_np, dtype=np.float32
        ).reshape(-1)

    torch = _resolve_torch(torch_module, torch_module_factory)
    x_batch = _tensor_to_device(
        torch, batch_np, device, use_pinned_transfer=use_pinned_transfer
    )
    run_model = get_compiled_model(
        model,
        device=device,
        torch_module=torch,
        compile_on_cpu=compile_on_cpu,
    )
    with torch.inference_mode():
        logits_batch, vals_batch = run_model(x_batch)
        probs_batch = torch.softmax(logits_batch, dim=-1).cpu().numpy()
        vals_np = vals_batch.cpu().numpy()
    return probs_batch, vals_np
