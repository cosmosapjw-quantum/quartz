"""Eval-safe runtime overrides derived from saved autotune profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path


SAFE_EVAL_RUNTIME_KEYS = ("batch_size",)


def _normalize_device_kind(device_name: str | None) -> str:
    name = str(device_name or "").lower()
    if name.startswith("cuda") or name.startswith("hip"):
        return "cuda"
    if name.startswith("mps"):
        return "mps"
    if name.startswith("cpu"):
        return "cpu"
    return name


def load_eval_runtime_overrides_from_model(
    model_path: str | None, device_name: str | None
) -> dict:
    if not model_path or os.environ.get("QUARTZ_DISABLE_EVAL_AUTOTUNE_PROFILE"):
        return {}

    device_kind = _normalize_device_kind(device_name)
    if device_kind in {"", "cpu"}:
        return {}

    profile_path = Path(model_path).with_name("autotune_profile.json")
    if not profile_path.exists():
        return {}

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    signature = payload.get("signature") or {}
    hardware = signature.get("hardware") or {}
    profile_device_kind = _normalize_device_kind(hardware.get("device_kind"))
    if profile_device_kind and profile_device_kind != device_kind:
        return {}

    overrides = payload.get("overrides") or {}
    benchmarks = payload.get("benchmarks") or {}
    heuristic = benchmarks.get("heuristic") or {}
    applied = {}
    optimal_batch_size = heuristic.get("optimal_batch_size")
    if optimal_batch_size is not None:
        try:
            applied["batch_size"] = int(optimal_batch_size)
        except Exception:
            pass
    elif overrides.get("batch_size") is not None:
        try:
            applied["batch_size"] = int(overrides["batch_size"])
        except Exception:
            pass
    return applied
