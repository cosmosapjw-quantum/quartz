"""Prior revision operators and frozen-checkpoint assay helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PriorRevisionSystem:
    id: str
    label: str
    operator: str
    search_overrides: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


def normalize_policy(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        if arr.size == 0:
            return arr
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    return (arr / total).astype(np.float32, copy=False)


def policy_argmax(policy: np.ndarray) -> int:
    if policy.size == 0:
        return -1
    return int(np.argmax(policy))


def entropy(policy: np.ndarray) -> float:
    probs = normalize_policy(policy)
    probs = np.clip(probs, 1e-8, 1.0)
    return float(-(probs * np.log(probs)).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_norm = np.clip(normalize_policy(p), 1e-8, 1.0)
    q_norm = np.clip(normalize_policy(q), 1e-8, 1.0)
    return float((p_norm * (np.log(p_norm) - np.log(q_norm))).sum())


def topk_indices(policy: np.ndarray, k: int) -> list[int]:
    probs = normalize_policy(policy)
    k = max(1, min(int(k), int(probs.size)))
    if probs.size == 0:
        return []
    order = np.argsort(-probs, kind="stable")
    return [int(idx) for idx in order[:k]]


def topk_recall(policy: np.ndarray, oracle_policy: np.ndarray, k: int = 3) -> float:
    oracle_best = policy_argmax(oracle_policy)
    if oracle_best < 0:
        return 0.0
    return float(oracle_best in topk_indices(policy, k))


def candidate_undercoverage(candidate_set: list[int], oracle_best: int) -> int:
    if oracle_best < 0:
        return 0
    return int(int(oracle_best) not in {int(idx) for idx in candidate_set})


def compute_argmax_path_metrics(argmax_path: list[int]) -> dict[str, Any]:
    if not argmax_path:
        return {"num_revisions": 0, "oscillation_count": 0, "rank_swap_count": 0}
    num_revisions = 0
    oscillation_count = 0
    last = argmax_path[0]
    seen = {last}
    for cur in argmax_path[1:]:
        if cur != last:
            num_revisions += 1
            if cur in seen:
                oscillation_count += 1
            seen.add(cur)
            last = cur
    return {
        "num_revisions": int(num_revisions),
        "oscillation_count": int(oscillation_count),
        "rank_swap_count": int(num_revisions),
    }


def first_revision_budget(argmax_path: list[int], budgets: list[int], argmax_prior: int) -> int | None:
    for budget, argmax_eff in zip(budgets, argmax_path):
        if int(argmax_eff) != int(argmax_prior):
            return int(budget)
    return None


def make_default_systems(base_cfg: dict[str, Any]) -> list[PriorRevisionSystem]:
    base_temp = float(base_cfg.get("prior_refresh_temp", 1.0) or 1.0)
    return [
        PriorRevisionSystem(
            id="B0",
            label="no-refresh baseline",
            operator="search",
            search_overrides={
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.0,
            },
        ),
        PriorRevisionSystem(
            id="B1",
            label="current refresh",
            operator="search",
            search_overrides={
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
            },
        ),
        PriorRevisionSystem(
            id="N1",
            label="dual-channel refresh",
            operator="dual_channel",
            search_overrides={
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.0,
            },
            params={
                "gate_epsilon": 0.05,
                "gate_scale": 1.0,
                "posterior_mix": 1.0,
                "fallback_temp": base_temp,
            },
        ),
        PriorRevisionSystem(
            id="N2",
            label="root-only posterior snapshot",
            operator="root_snapshot",
            search_overrides={
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.0,
            },
            params={
                "challenger_k": 4,
                "candidate_score_mix": 0.5,
                "snapshot_alpha": 0.75,
                "adaptive_k": True,
            },
        ),
    ]


def load_systems_config(path_str: str | None, base_cfg: dict[str, Any]) -> list[PriorRevisionSystem]:
    if not path_str:
        return make_default_systems(base_cfg)
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    rows = payload.get("systems", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid systems config: {path_str}")
    systems = []
    for row in rows:
        systems.append(
            PriorRevisionSystem(
                id=str(row["id"]),
                label=str(row.get("label", row["id"])),
                operator=str(row["operator"]),
                search_overrides=dict(row.get("search_overrides") or {}),
                params=dict(row.get("params") or {}),
            )
        )
    return systems


def apply_prior_corruption(prior: np.ndarray, corruption: str, oracle_best: int | None = None,
                           strength: float = 1.75) -> np.ndarray:
    probs = normalize_policy(prior)
    if probs.size < 2:
        return probs
    order = topk_indices(probs, min(4, probs.size))
    oracle_best = None if oracle_best is None else int(oracle_best)
    out = probs.copy()

    if corruption == "swap_top12":
        if oracle_best is not None:
            if int(order[0]) == oracle_best:
                alt = next((idx for idx in order[1:] if idx != oracle_best), int(order[1]))
                out[int(order[0])], out[int(alt)] = out[int(alt)], out[int(order[0])]
            else:
                alt = next((idx for idx in order[1:] if idx != oracle_best), int(order[0]))
                if alt == int(order[0]):
                    out[int(order[0])] *= float(max(1.01, strength))
                    out[oracle_best] *= float(max(0.05, 1.0 / max(1.01, strength)))
                else:
                    out[int(order[0])], out[int(alt)] = out[int(alt)], out[int(order[0])]
        else:
            out[int(order[0])], out[int(order[1])] = out[int(order[1])], out[int(order[0])]
        return normalize_policy(out)

    if corruption == "inflate_wrong_confidence":
        wrong_idx = int(order[0])
        if oracle_best is not None and wrong_idx == oracle_best:
            wrong_idx = next((idx for idx in order[1:] if idx != oracle_best), int(order[1]))
        out[wrong_idx] *= float(max(1.01, strength))
        if oracle_best is not None and 0 <= oracle_best < out.size:
            out[oracle_best] *= float(max(0.05, 1.0 / max(1.01, strength)))
        return normalize_policy(out)

    raise ValueError(f"unsupported corruption: {corruption}")


def dual_channel_gate(prior_base: np.ndarray, posterior_search: np.ndarray,
                      gate_epsilon: float = 0.05, gate_scale: float = 1.0) -> float:
    divergence = kl_divergence(prior_base, posterior_search)
    gate_epsilon = max(1e-6, float(gate_epsilon))
    if divergence <= gate_epsilon:
        return 0.0
    raw = (divergence - gate_epsilon) / max(divergence, gate_epsilon)
    return float(np.clip(raw * float(gate_scale), 0.0, 1.0))


def infer_search_posterior(prior_base: np.ndarray, search_policy: np.ndarray,
                           posterior_mix: float = 1.0) -> np.ndarray:
    prior_norm = normalize_policy(prior_base)
    search_norm = normalize_policy(search_policy)
    mix = float(np.clip(posterior_mix, 0.0, 1.0))
    return normalize_policy((1.0 - mix) * prior_norm + mix * search_norm)


def apply_dual_channel_refresh(prior_base: np.ndarray, search_policy: np.ndarray,
                               params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    posterior = infer_search_posterior(
        prior_base,
        search_policy,
        posterior_mix=float(params.get("posterior_mix", 1.0)),
    )
    gate = dual_channel_gate(
        prior_base,
        posterior,
        gate_epsilon=float(params.get("gate_epsilon", 0.05)),
        gate_scale=float(params.get("gate_scale", 1.0)),
    )
    effective = normalize_policy((1.0 - gate) * normalize_policy(prior_base) + gate * posterior)
    return effective, {
        "posterior_search": posterior.tolist(),
        "dual_gate": float(gate),
        "posterior_norm": float(np.linalg.norm(posterior, ord=1)),
    }


def adaptive_challenger_k(challenger_k: int, budget: int, adaptive: bool = True) -> int:
    challenger_k = max(2, int(challenger_k))
    if not adaptive:
        return challenger_k
    return max(2, min(challenger_k, int(math.ceil(max(1.0, float(budget)) ** 0.5))))


def build_root_challenger_set(prior_base: np.ndarray, search_policy: np.ndarray, budget: int,
                              params: dict[str, Any]) -> tuple[list[int], list[float]]:
    prior_norm = normalize_policy(prior_base)
    search_norm = normalize_policy(search_policy)
    mix = float(np.clip(params.get("candidate_score_mix", 0.5), 0.0, 1.0))
    score = mix * prior_norm + (1.0 - mix) * search_norm
    k = adaptive_challenger_k(
        int(params.get("challenger_k", 4)),
        int(budget),
        adaptive=bool(params.get("adaptive_k", True)),
    )
    candidates = topk_indices(score, k)
    return candidates, [float(score[idx]) for idx in candidates]


def compute_root_posterior(prior_base: np.ndarray, search_policy: np.ndarray, challenger_set: list[int],
                           params: dict[str, Any]) -> np.ndarray:
    prior_norm = normalize_policy(prior_base)
    search_norm = normalize_policy(search_policy)
    effective = prior_norm.copy()
    if not challenger_set:
        return effective
    alpha = float(np.clip(params.get("snapshot_alpha", 0.75), 0.0, 1.0))
    challenger_idx = np.asarray(challenger_set, dtype=np.int64)
    effective[challenger_idx] = normalize_policy(
        alpha * prior_norm[challenger_idx] + (1.0 - alpha) * search_norm[challenger_idx]
    )
    return normalize_policy(effective)


def apply_root_only_snapshot(prior_base: np.ndarray, search_policy: np.ndarray, budget: int,
                             params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    challenger_set, candidate_scores = build_root_challenger_set(prior_base, search_policy, budget, params)
    root_posterior = compute_root_posterior(prior_base, search_policy, challenger_set, params)
    return root_posterior, {
        "root_candidate_set": list(challenger_set),
        "root_candidate_scores": list(candidate_scores),
        "root_posterior": root_posterior.tolist(),
    }


def apply_revision_operator(system: PriorRevisionSystem, prior_base: np.ndarray, search_policy: np.ndarray,
                            budget: int) -> tuple[np.ndarray, dict[str, Any]]:
    if system.operator == "search":
        effective = normalize_policy(search_policy)
        return effective, {}
    if system.operator == "dual_channel":
        return apply_dual_channel_refresh(prior_base, search_policy, system.params)
    if system.operator == "root_snapshot":
        return apply_root_only_snapshot(prior_base, search_policy, budget, system.params)
    raise ValueError(f"unsupported prior revision operator: {system.operator}")


def classify_position_buckets(prior_base: np.ndarray, low_budget_policy: np.ndarray, oracle_policy: np.ndarray,
                              thresholds: dict[str, Any] | None = None) -> list[str]:
    thresholds = thresholds or {}
    confident_threshold = float(thresholds.get("confident_threshold", 0.55))
    ambiguous_margin = float(thresholds.get("ambiguous_margin", 0.10))
    root_topk = int(thresholds.get("root_conflict_topk", 2))
    deep_topk = int(thresholds.get("deep_conflict_topk", 2))

    prior = normalize_policy(prior_base)
    low = normalize_policy(low_budget_policy)
    oracle = normalize_policy(oracle_policy)
    prior_best = policy_argmax(prior)
    low_best = policy_argmax(low)
    oracle_best = policy_argmax(oracle)
    prior_top2 = topk_indices(prior, 2)
    low_topk = topk_indices(low, root_topk)

    prior_sorted = np.sort(prior)[::-1]
    prior_margin = float(prior_sorted[0] - prior_sorted[1]) if prior.size >= 2 else 1.0
    buckets: set[str] = {"generic"}

    if prior_best != oracle_best:
        buckets.add("wrong_top1")
        if float(prior.max()) >= confident_threshold:
            buckets.add("wrong_confident")
        if oracle_best in prior_top2:
            buckets.add("wrong_top2swap")
    else:
        if float(prior.max()) >= confident_threshold:
            buckets.add("easy_good_prior")

    if prior_margin <= ambiguous_margin:
        buckets.add("ambiguous")

    if low_best != oracle_best:
        buckets.add("late_evidence")
        if prior_best == low_best and float(prior.max()) >= confident_threshold:
            buckets.add("shallow_trap")

    if oracle_best in low_topk and low_best != oracle_best:
        buckets.add("root_conflict")

    if low_best != oracle_best and oracle_best not in topk_indices(prior, deep_topk):
        buckets.add("deep_conflict")

    return sorted(buckets)


def summarize_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        bucket = str(row.get("position_bucket", "all"))
        key = (str(row["experiment"]), str(row["system"]), int(row["budget"]))
        acc = grouped.setdefault(
            key,
            {
                "experiment": key[0],
                "system": key[1],
                "budget": key[2],
                "rows": 0,
            },
        )
        acc["rows"] += 1
        for metric_key in metric_keys:
            value = row.get(metric_key)
            if value is None:
                continue
            acc.setdefault(metric_key, 0.0)
            acc[metric_key] += float(value)
        acc.setdefault("_bucket_counts", {})
        acc["_bucket_counts"][bucket] = acc["_bucket_counts"].get(bucket, 0) + 1

    summary = []
    for acc in grouped.values():
        rows_n = max(1, int(acc["rows"]))
        item = {k: v for k, v in acc.items() if not k.startswith("_")}
        for metric_key in metric_keys:
            if metric_key in item:
                item[metric_key] = item[metric_key] / rows_n
        item["bucket_counts"] = acc["_bucket_counts"]
        summary.append(item)
    summary.sort(key=lambda row: (row["experiment"], row["budget"], row["system"]))
    return summary
