"""Suite preparation and mining helpers for phase 1.5 assays."""

from __future__ import annotations

import base64
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .phase15_ablation import classify_position_buckets, policy_argmax


SUITE_POLICY_KEYS = ("prior_policy", "low_budget_policy", "reference_policy", "oracle_policy")


def _pack_row_id(row_id: str) -> str:
    encoded = base64.urlsafe_b64encode(str(row_id).encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _unpack_row_id(packed: str) -> str:
    pad = "=" * ((4 - (len(packed) % 4)) % 4)
    return base64.urlsafe_b64decode(f"{packed}{pad}".encode("ascii")).decode("utf-8")


def bucket_thresholds(
    *,
    confident_threshold: float,
    ambiguous_margin: float,
    root_conflict_topk: int,
    deep_conflict_topk: int,
) -> dict[str, Any]:
    return {
        "confident_threshold": float(confident_threshold),
        "ambiguous_margin": float(ambiguous_margin),
        "root_conflict_topk": int(root_conflict_topk),
        "deep_conflict_topk": int(deep_conflict_topk),
    }


def annotate_position_suite(
    positions: list[dict[str, Any]],
    *,
    prior_policy_fn: Callable[[dict[str, Any]], np.ndarray],
    low_policy_fn: Callable[[dict[str, Any]], np.ndarray],
    reference_policy_fn: Callable[[dict[str, Any]], np.ndarray],
    oracle_policy_fn: Callable[[dict[str, Any]], np.ndarray],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    suite = []
    for row in positions:
        prior = np.asarray(prior_policy_fn(row), dtype=np.float32)
        low = np.asarray(low_policy_fn(row), dtype=np.float32)
        reference = np.asarray(reference_policy_fn(row), dtype=np.float32)
        oracle = np.asarray(oracle_policy_fn(row), dtype=np.float32)
        item = dict(row)
        item["bucket_tags"] = classify_position_buckets(prior, low, oracle, thresholds=thresholds)
        item["prior_argmax"] = policy_argmax(prior)
        item["low_budget_best"] = policy_argmax(low)
        item["reference_best"] = policy_argmax(reference)
        item["oracle_best"] = policy_argmax(oracle)
        item["prior_policy"] = prior.astype(np.float32, copy=False).tolist()
        item["low_budget_policy"] = low.astype(np.float32, copy=False).tolist()
        item["reference_policy"] = reference.astype(np.float32, copy=False).tolist()
        item["oracle_policy"] = oracle.astype(np.float32, copy=False).tolist()
        suite.append(item)
    return suite


def bucket_counts(suite: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in suite:
        for tag in row.get("bucket_tags", []):
            counter[str(tag)] += 1
    return dict(sorted(counter.items()))


def mine_balanced_suite(
    annotated_positions: list[dict[str, Any]],
    *,
    suite_size: int,
    bucket_min_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    generic: list[dict[str, Any]] = []
    for row in annotated_positions:
        tags = [str(tag) for tag in row.get("bucket_tags", []) if str(tag) != "generic"]
        if not tags:
            generic.append(row)
        for tag in tags:
            by_bucket[tag].append(row)

    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []
    for bucket in sorted(by_bucket):
        bucket_rows = list(by_bucket[bucket])
        rng.shuffle(bucket_rows)
        for row in bucket_rows[:bucket_min_count]:
            row_id = str(row.get("id"))
            if row_id in selected_ids:
                continue
            selected_ids.add(row_id)
            selected.append(row)

    leftovers = list(annotated_positions)
    rng.shuffle(leftovers)
    for row in leftovers:
        if len(selected) >= int(suite_size):
            break
        row_id = str(row.get("id"))
        if row_id in selected_ids:
            continue
        selected_ids.add(row_id)
        selected.append(row)

    selected.sort(key=lambda row: str(row.get("id")))
    return selected[: int(suite_size)]


def split_suite_policy_artifacts(suite: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]]]:
    compact_suite: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, list[float]]] = {}
    for row in suite:
        item = dict(row)
        row_id = str(item.get("id"))
        artifact_entry: dict[str, list[float]] = {}
        for key in SUITE_POLICY_KEYS:
            raw = item.pop(key, None)
            if isinstance(raw, list) and raw:
                artifact_entry[key] = [float(x) for x in raw]
        if artifact_entry:
            artifacts[row_id] = artifact_entry
            item["policy_artifact_ref"] = row_id
        compact_suite.append(item)
    return compact_suite, artifacts


def merge_suite_policy_artifacts(
    suite: list[dict[str, Any]],
    artifacts: dict[str, dict[str, list[float]]] | None,
) -> list[dict[str, Any]]:
    if not artifacts:
        return [dict(row) for row in suite]
    merged: list[dict[str, Any]] = []
    for row in suite:
        item = dict(row)
        ref = str(item.get("policy_artifact_ref") or item.get("id"))
        payload = artifacts.get(ref, {})
        for key in SUITE_POLICY_KEYS:
            if key in payload and key not in item:
                item[key] = list(payload[key])
        merged.append(item)
    return merged


def write_suite_policy_artifacts(path: Path, artifacts: dict[str, dict[str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for row_id, entry in artifacts.items():
        safe_row = _pack_row_id(str(row_id))
        for key, values in entry.items():
            payload[f"{safe_row}__{key}"] = np.asarray(values, dtype=np.float32)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def read_suite_policy_artifacts(path: Path | None) -> dict[str, dict[str, list[float]]] | None:
    if path is None or not path.exists():
        return None
    out: dict[str, dict[str, list[float]]] = {}
    with np.load(path, allow_pickle=False) as payload:
        for packed_key in payload.files:
            if "__" not in packed_key:
                continue
            packed_row_id, policy_key = packed_key.split("__", 1)
            if policy_key not in SUITE_POLICY_KEYS:
                continue
            values = payload[packed_key]
            if not isinstance(values, np.ndarray):
                continue
            row_id = _unpack_row_id(str(packed_row_id))
            entry = out.setdefault(str(row_id), {})
            entry[policy_key] = values.astype(np.float32, copy=False).tolist()
    return out or None


__all__ = [
    "annotate_position_suite",
    "bucket_counts",
    "bucket_thresholds",
    "merge_suite_policy_artifacts",
    "mine_balanced_suite",
    "read_suite_policy_artifacts",
    "split_suite_policy_artifacts",
    "write_suite_policy_artifacts",
]
