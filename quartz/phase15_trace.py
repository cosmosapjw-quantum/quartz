"""Trace artifacts and disk cache helpers for phase 1.5 assays."""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TRACE_CACHE_SCHEMA_VERSION = 6  # Stage 7: bundles carry per-chunk p_flip + checkpoint/position identity
TRACE_CACHE_RELEVANT_PATHS = (
    "configs/phase15_systems.default.json",
    "quartz/phase15_ablation.py",
    "quartz/phase15_online.py",
    "quartz/phase15_suite.py",
    "quartz/phase15_trace.py",
    "quartz/search_manifest.py",
    "quartz/selfplay_runtime.py",
    "scripts/phase15_ablation_study.py",
    "scripts/phase15_online_ablation.py",
    "scripts/phase15_benchmark.py",
    "scripts/phase15_mine_suite.py",
    "src/mcts/mod.rs",
    "src/mcts/root.rs",
    "src/mcts_server.rs",
)


@functools.lru_cache(maxsize=1)
def trace_cache_salt() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(f"phase15-trace-schema:{TRACE_CACHE_SCHEMA_VERSION}".encode("utf-8"))
    for rel_path in TRACE_CACHE_RELEVANT_PATHS:
        path = repo_root / rel_path
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def trace_cache_key(
    checkpoint_id: str,
    checkpoint_path: str,
    position_id: str,
    search_signature: tuple[Any, ...],
    trace_budgets: list[int],
    *,
    code_salt: str | None = None,
) -> str:
    """Cache key for a (checkpoint, position, search config, budgets) trace.

    `search_signature` must identify only what actually reaches the
    Rust search (see `quartz.phase15_ablation.search_relevant_signature`).
    It deliberately excludes any system id/label/readout-operator identity:
    two systems that only differ in post-hoc readout must collide onto
    the same cache entry so they provably share one search trace
    (A0-b audit fix — see docs/CLAIM_LEDGER.md).
    """
    payload = {
        "trace_cache_schema_version": TRACE_CACHE_SCHEMA_VERSION,
        "trace_code_salt": str(code_salt or trace_cache_salt()),
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": checkpoint_path,
        "position_id": position_id,
        "search_signature": list(search_signature),
        "trace_budgets": [int(x) for x in trace_budgets],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trace_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.json"


def load_cached_trace(cache_dir: Path | None, cache_key: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = trace_cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def store_cached_trace(cache_dir: Path | None, cache_key: str, payload: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    trace_cache_path(cache_dir, cache_key).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def build_trace_artifact(
    trace_budgets: list[int],
    trace_policies: list[np.ndarray],
    trace_latencies_ms: list[float],
    *,
    source: str,
    code_salt: str | None = None,
    trace_p_flips: list[float | None] | None = None,
    checkpoint_id: str | None = None,
    position_id: str | None = None,
) -> dict[str, Any]:
    # Stage 7 / C6: the engine's own per-chunk P_flip is recorded alongside the
    # policy so the flip-calibration lane (C8) can compare H1 argmax-stability
    # against the incumbent P_flip at IDENTICAL chunk boundaries. Missing values
    # are stored as None (back-compat: pre-C6 bundles simply omit the field).
    if trace_p_flips is None:
        p_flips: list[float | None] = [None for _ in trace_budgets]
    else:
        p_flips = [None if v is None else float(v) for v in trace_p_flips]
    return {
        "trace_cache_schema_version": TRACE_CACHE_SCHEMA_VERSION,
        "trace_code_salt": str(code_salt or trace_cache_salt()),
        "trace_budgets": [int(x) for x in trace_budgets],
        "trace_policies": [np.asarray(policy, dtype=np.float32).tolist() for policy in trace_policies],
        "trace_latencies_ms": [float(x) for x in trace_latencies_ms],
        "trace_p_flips": p_flips,
        "trace_acquire_ms": float(sum(float(x) for x in trace_latencies_ms)),
        "trace_source": str(source),
        # Stage 7: self-identify so the O6 join (C9) and any bundle-level analysis
        # can key by (checkpoint, position) without re-deriving the cache key.
        "checkpoint_id": None if checkpoint_id is None else str(checkpoint_id),
        "position_id": None if position_id is None else str(position_id),
    }


__all__ = [
    "build_trace_artifact",
    "load_cached_trace",
    "store_cached_trace",
    "trace_cache_salt",
    "trace_cache_key",
    "trace_cache_path",
]
