"""Phase 1.5 clean-split ablation helpers.

This module replaces the older prior-revision operator study with the
phase-1.5 matrix described in ``phase15_strategy_revision_v2.md``:

- Group A: substrate/controller sanity
- Group B: refresh isolated on top of a clean substrate
- Group C: legacy anchor comparison only
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Phase15System:
    id: str
    label: str
    group: str
    substrate: str
    controller: str
    refresh_operator: str
    search_overrides: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "posthoc"
    report_alias: str | None = None


LEGACY_PENALTY_MODES = {"GatedRefreshLegacy", "PFlipMixture", "SelfAdaptive"}
POSTHOC_OPERATORS = {"none", "dual_channel_commit", "root_challenger", "budget_routing"}
ONLINE_OPERATORS = {"none", "dual_channel_commit", "root_challenger", "budget_routing"}


def system_semantic_signature(system: Phase15System) -> tuple[Any, ...]:
    return (
        system.group,
        system.substrate,
        system.controller,
        system.refresh_operator,
        system.execution_mode,
        json.dumps(system.search_overrides, sort_keys=True, separators=(",", ":")),
        json.dumps(system.params, sort_keys=True, separators=(",", ":")),
    )


def validate_phase15_systems(systems: list[Phase15System]) -> None:
    if not systems:
        raise ValueError("phase15 systems list is empty")
    by_id: dict[str, Phase15System] = {}
    alias_targets = set()
    for system in systems:
        if system.id in by_id:
            raise ValueError(f"duplicate phase15 system id: {system.id}")
        by_id[system.id] = system
        alias_targets.add(system.id)

    for system in systems:
        if system.group not in {"A", "B", "C"}:
            raise ValueError(f"{system.id}: unsupported group {system.group!r}")
        if system.execution_mode not in {"posthoc", "online"}:
            raise ValueError(f"{system.id}: unsupported execution_mode {system.execution_mode!r}")
        root_only = system.search_overrides.get("root_only_shaping")
        penalty_mode = str(system.search_overrides.get("penalty_mode", "None"))
        if system.group in {"A", "B"}:
            if root_only is not True:
                raise ValueError(f"{system.id}: clean A/B systems require root_only_shaping=true")
            if penalty_mode in LEGACY_PENALTY_MODES:
                raise ValueError(f"{system.id}: clean A/B systems may not use legacy penalty mode {penalty_mode}")
            if system.substrate == "legacy":
                raise ValueError(f"{system.id}: clean A/B systems may not declare legacy substrate")
        if system.group == "A":
            if system.refresh_operator != "none":
                raise ValueError(f"{system.id}: Group A must not enable refresh_operator")
            if system.execution_mode != "posthoc":
                raise ValueError(f"{system.id}: Group A should remain posthoc/readout-only")
        elif system.group == "B":
            if system.refresh_operator not in POSTHOC_OPERATORS | ONLINE_OPERATORS:
                raise ValueError(f"{system.id}: unsupported Group B refresh operator {system.refresh_operator!r}")
        elif system.group == "C":
            if penalty_mode not in LEGACY_PENALTY_MODES:
                raise ValueError(f"{system.id}: Group C must remain legacy-anchor only")
        if system.report_alias is not None and system.report_alias not in alias_targets:
            raise ValueError(f"{system.id}: unknown report_alias target {system.report_alias}")
        if system.execution_mode == "posthoc" and system.refresh_operator not in POSTHOC_OPERATORS:
            raise ValueError(f"{system.id}: posthoc mode cannot run {system.refresh_operator!r}")
        if system.execution_mode == "online" and system.refresh_operator not in ONLINE_OPERATORS:
            raise ValueError(f"{system.id}: online mode cannot run {system.refresh_operator!r}")


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
    probs = np.clip(normalize_policy(policy), 1e-8, 1.0)
    return float(-(probs * np.log(probs)).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_norm = np.clip(normalize_policy(p), 1e-8, 1.0)
    q_norm = np.clip(normalize_policy(q), 1e-8, 1.0)
    return float((p_norm * (np.log(p_norm) - np.log(q_norm))).sum())


def topk_indices(policy: np.ndarray, k: int) -> list[int]:
    probs = normalize_policy(policy)
    if probs.size == 0:
        return []
    k = max(1, min(int(k), int(probs.size)))
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


def top2_margin(policy: np.ndarray) -> float:
    probs = normalize_policy(policy)
    if probs.size < 2:
        return 1.0
    order = np.sort(probs)[::-1]
    return float(order[0] - order[1])


def _suffix_stability(argmax_path: list[int]) -> float:
    if not argmax_path:
        return 0.0
    final = argmax_path[-1]
    suffix_len = 0
    for item in reversed(argmax_path):
        if item != final:
            break
        suffix_len += 1
    return float(suffix_len) / float(len(argmax_path))


def jaccard_overlap(lhs: list[int], rhs: list[int]) -> float:
    left = {int(x) for x in lhs}
    right = {int(x) for x in rhs}
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right)) / float(len(union))


def compute_argmax_path_metrics(argmax_path: list[int]) -> dict[str, Any]:
    if not argmax_path:
        return {
            "num_revisions": 0,
            "oscillation_count": 0,
            "rank_swap_count": 0,
            "argmax_persistence": 0.0,
        }
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
        "argmax_persistence": _suffix_stability(argmax_path),
    }


def first_revision_budget(argmax_path: list[int], budgets: list[int], argmax_prior: int) -> int | None:
    for budget, argmax_eff in zip(budgets, argmax_path):
        if int(argmax_eff) != int(argmax_prior):
            return int(budget)
    return None


def classify_position_buckets(
    prior_base: np.ndarray,
    low_budget_policy: np.ndarray,
    oracle_policy: np.ndarray,
    thresholds: dict[str, Any] | None = None,
) -> list[str]:
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
    elif float(prior.max()) >= confident_threshold:
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


def _trace_signal_metrics(policies: list[np.ndarray], budgets: list[int], challenger_k: int = 4) -> dict[str, Any]:
    if not policies:
        return {
            "argmax_path": [],
            "entropy_path": [],
            "margin_path": [],
            "argmax_persistence": 0.0,
            "top2_margin_stability": 0.0,
            "challenger_overlap": 1.0,
            "posterior_entropy_slope": 0.0,
            "revision_flip_flop_count": 0,
        }

    argmax_path = [policy_argmax(p) for p in policies]
    entropy_path = [entropy(p) for p in policies]
    margin_path = [top2_margin(p) for p in policies]
    metrics = compute_argmax_path_metrics(argmax_path)
    overlap = 1.0
    if len(policies) >= 2:
        overlap = jaccard_overlap(topk_indices(policies[-2], challenger_k), topk_indices(policies[-1], challenger_k))
    margin_std = float(np.std(np.asarray(margin_path, dtype=np.float32))) if len(margin_path) > 1 else 0.0
    margin_mean = float(np.mean(np.asarray(margin_path, dtype=np.float32))) if margin_path else 0.0
    margin_stability = 1.0 / (1.0 + (margin_std / max(margin_mean, 1e-6)))
    return {
        "argmax_path": argmax_path,
        "entropy_path": [float(x) for x in entropy_path],
        "margin_path": [float(x) for x in margin_path],
        "argmax_persistence": float(metrics["argmax_persistence"]),
        "top2_margin_stability": float(np.clip(margin_stability, 0.0, 1.0)),
        "challenger_overlap": float(np.clip(overlap, 0.0, 1.0)),
        "posterior_entropy_slope": float(entropy_path[-1] - entropy_path[0]) if len(entropy_path) >= 2 else 0.0,
        "revision_flip_flop_count": int(metrics["oscillation_count"]),
        "num_revisions": int(metrics["num_revisions"]),
        "rank_swap_count": int(metrics["rank_swap_count"]),
        "trace_budgets": [int(x) for x in budgets],
    }


def _clipped_unit(value: float, scale: float) -> float:
    return float(np.clip(float(value) / max(float(scale), 1e-6), 0.0, 1.0))


def compute_commit_confidence(
    prior_base: np.ndarray,
    posterior_prev: np.ndarray | None,
    posterior_now: np.ndarray,
    signal_metrics: dict[str, Any],
    params: dict[str, Any],
) -> float:
    divergence = kl_divergence(prior_base, posterior_now)
    entropy_gain = max(0.0, entropy(prior_base) - entropy(posterior_now))
    posterior_shift = 0.0 if posterior_prev is None else kl_divergence(posterior_prev, posterior_now)
    posterior_stability = 1.0 - _clipped_unit(posterior_shift, float(params.get("posterior_shift_scale", 0.35)))

    score = (
        0.35 * _clipped_unit(divergence, float(params.get("divergence_scale", 0.40)))
        + 0.25 * float(signal_metrics.get("argmax_persistence", 0.0))
        + 0.20 * float(signal_metrics.get("top2_margin_stability", 0.0))
        + 0.10 * posterior_stability
        + 0.10 * _clipped_unit(entropy_gain, float(params.get("entropy_scale", 0.35)))
    )
    return float(np.clip(score * float(params.get("gate_scale", 1.0)), 0.0, 1.0))


def apply_dual_channel_commit(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    posterior_now = normalize_policy(trace_policies[-1])
    posterior_prev = normalize_policy(trace_policies[-2]) if len(trace_policies) >= 2 else None
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets, challenger_k=int(params.get("challenger_k", 4)))
    commit_confidence = compute_commit_confidence(prior_base, posterior_prev, posterior_now, signal_metrics, params)
    commit_threshold = float(params.get("commit_threshold", 0.55))
    if commit_confidence >= commit_threshold:
        effective = normalize_policy(
            (1.0 - commit_confidence) * normalize_policy(prior_base) + commit_confidence * posterior_now
        )
    else:
        effective = normalize_policy(prior_base)
    return effective, {
        **signal_metrics,
        "posterior_search": posterior_now.tolist(),
        "commit_confidence": float(commit_confidence),
        "commit_threshold": float(commit_threshold),
        "commit_applied": int(commit_confidence >= commit_threshold),
        "commit_latency": first_revision_budget(
            signal_metrics["argmax_path"], signal_metrics["trace_budgets"], policy_argmax(prior_base)
        ),
    }


def build_root_challenger_set(
    prior_base: np.ndarray,
    posterior_now: np.ndarray,
    params: dict[str, Any],
) -> tuple[list[int], list[float]]:
    prior = normalize_policy(prior_base)
    posterior = normalize_policy(posterior_now)
    mix = float(np.clip(params.get("candidate_score_mix", 0.5), 0.0, 1.0))
    challenger_k = max(2, int(params.get("challenger_k", 4)))
    tie_eps = max(0.0, float(params.get("candidate_tie_eps", 5e-4)))
    max_extra = max(0, int(params.get("challenger_max_extra", 4)))
    prior_anchor_k = max(0, int(params.get("prior_anchor_k", min(2, challenger_k))))
    posterior_anchor_k = max(0, int(params.get("posterior_anchor_k", challenger_k)))
    score = normalize_policy(mix * prior + (1.0 - mix) * posterior)
    order = np.argsort(-score, kind="stable")
    cutoff_rank = min(challenger_k, int(score.size)) - 1
    cutoff = float(score[order[cutoff_rank]]) if score.size else 0.0
    limit = min(int(score.size), challenger_k + max_extra)
    anchor_ids = {
        *topk_indices(prior, prior_anchor_k),
        *topk_indices(posterior, posterior_anchor_k),
    }
    candidates = []
    for idx in order[:limit]:
        idx_int = int(idx)
        if (
            len(candidates) < challenger_k
            or float(score[idx_int]) >= cutoff - tie_eps
            or idx_int in anchor_ids
        ):
            candidates.append(idx_int)
        else:
            break
    return candidates, [float(score[idx]) for idx in candidates]


def apply_root_challenger_refresh(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    posterior_now = normalize_policy(trace_policies[-1])
    candidates, candidate_scores = build_root_challenger_set(prior_base, posterior_now, params)
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets, challenger_k=len(candidates))
    alpha = float(np.clip(params.get("snapshot_alpha", 0.75), 0.0, 1.0))
    effective = normalize_policy(prior_base)
    if candidates:
        idx = np.asarray(candidates, dtype=np.int64)
        effective[idx] = normalize_policy(alpha * normalize_policy(prior_base)[idx] + (1.0 - alpha) * posterior_now[idx])
        effective = normalize_policy(effective)
    return effective, {
        **signal_metrics,
        "root_candidate_set": list(candidates),
        "root_candidate_scores": list(candidate_scores),
        "challenger_k": int(len(candidates)),
        "challenger_recall_k": int(policy_argmax(posterior_now) in candidates),
    }


def apply_budget_routing(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    target_budget: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    base_trace = [p for p, b in zip(trace_policies, trace_budgets) if int(b) <= int(target_budget)]
    if not base_trace:
        base_trace = [trace_policies[0]]
    base_budgets = [int(b) for b in trace_budgets if int(b) <= int(target_budget)] or [int(target_budget)]
    signal_metrics = _trace_signal_metrics(base_trace, base_budgets, challenger_k=int(params.get("challenger_k", 4)))
    base_policy = normalize_policy(base_trace[-1])
    burst_idx = next((idx for idx, budget in enumerate(trace_budgets) if int(budget) > int(target_budget)), None)
    burst_trigger = (
        burst_idx is not None
        and (
            float(signal_metrics["argmax_persistence"]) < float(params.get("persistence_floor", 0.60))
            or float(signal_metrics["top2_margin_stability"]) < float(params.get("margin_stability_floor", 0.72))
        )
    )
    effective = normalize_policy(trace_policies[burst_idx]) if burst_trigger else base_policy
    burst_budget = int(trace_budgets[burst_idx]) if burst_trigger and burst_idx is not None else int(target_budget)
    return effective, {
        **signal_metrics,
        "budget_burst_triggered": int(burst_trigger),
        "extra_budget_used": int(max(0, burst_budget - int(target_budget))),
        "burst_budget": int(burst_budget),
        "burst_reason": (
            "instability"
            if burst_trigger
            else "none"
        ),
    }


def apply_system_readout(
    system: Phase15System,
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    target_budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    operator = system.refresh_operator
    if operator == "none":
        signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets, challenger_k=int(system.params.get("challenger_k", 4)))
        return normalize_policy(trace_policies[-1]), signal_metrics
    if operator == "dual_channel_commit":
        return apply_dual_channel_commit(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "root_challenger":
        return apply_root_challenger_refresh(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "budget_routing":
        return apply_budget_routing(prior_base, trace_policies, trace_budgets, target_budget, system.params)
    raise ValueError(f"unsupported phase15 refresh operator: {operator}")


def _clean_overrides(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = dict(base)
    out.update(updates)
    return out


def make_default_systems(_base_cfg: dict[str, Any]) -> list[Phase15System]:
    a0 = _clean_overrides({}, search_profile="baseline", vl_mode="disabled", root_only_shaping=True)
    a1 = _clean_overrides({}, search_profile="quartz", vl_mode="disabled", penalty_mode="None", root_only_shaping=True)
    a2 = _clean_overrides({}, search_profile="quartz", vl_mode="adaptive", penalty_mode="None", root_only_shaping=True)
    a3 = _clean_overrides({}, search_profile="quartz", vl_mode="disabled", penalty_mode="EffectiveV2", root_only_shaping=True)
    a4 = _clean_overrides({}, search_profile="quartz", vl_mode="adaptive", penalty_mode="EffectiveV2", root_only_shaping=True)

    return [
        Phase15System("A0", "S0 substrate only", "A", "S0", "none", "none", search_overrides=a0),
        Phase15System("A1", "S0 + Quartz stop-only", "A", "S0", "QuartzStopOnly", "none", search_overrides=a1),
        Phase15System("A2", "S0 + Quartz + VL", "A", "S0", "QuartzVL", "none", search_overrides=a2),
        Phase15System("A3", "S1 + Quartz stop-only", "A", "S1", "QuartzStopOnly", "none", search_overrides=a3),
        Phase15System("A4", "S1 + Quartz + VL", "A", "S1", "QuartzVL", "none", search_overrides=a4),
        Phase15System(
            "B0",
            "A4 report alias (posthoc no-refresh baseline)",
            "B",
            "S1",
            "QuartzVL",
            "none",
            search_overrides=a4,
            report_alias="A4",
        ),
        Phase15System(
            "B1",
            "A4 + dual-channel commit (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "dual_channel_commit",
            search_overrides=a4,
            params={"commit_threshold": 0.55, "gate_scale": 1.0, "divergence_scale": 0.40, "entropy_scale": 0.35},
        ),
        Phase15System(
            "B2",
            "A4 + root challenger (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "root_challenger",
            search_overrides=a4,
            params={
                "challenger_k": 4,
                "candidate_score_mix": 0.50,
                "snapshot_alpha": 0.75,
                "candidate_tie_eps": 5e-4,
                "challenger_max_extra": 4,
                "prior_anchor_k": 2,
                "posterior_anchor_k": 4,
            },
        ),
        Phase15System(
            "B3",
            "A4 + budget routing (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "budget_routing",
            search_overrides=a4,
            params={"persistence_floor": 0.60, "margin_stability_floor": 0.72},
        ),
        Phase15System(
            "C0",
            "legacy GatedRefreshLegacy",
            "C",
            "legacy",
            "hybrid",
            "none",
            search_overrides={
                "search_profile": "quartz",
                "vl_mode": "adaptive",
                "penalty_mode": "GatedRefreshLegacy",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
            },
            execution_mode="online",
        ),
        Phase15System(
            "C1",
            "legacy PFlipMixture",
            "C",
            "legacy",
            "hybrid",
            "none",
            search_overrides={
                "search_profile": "quartz",
                "vl_mode": "adaptive",
                "penalty_mode": "PFlipMixture",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
            },
            execution_mode="online",
        ),
        Phase15System(
            "C2",
            "legacy SelfAdaptive",
            "C",
            "legacy",
            "hybrid",
            "none",
            search_overrides={
                "search_profile": "quartz",
                "vl_mode": "adaptive",
                "penalty_mode": "SelfAdaptive",
                "root_only_shaping": False,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
            },
            execution_mode="online",
        ),
    ]


def load_systems_config(path_str: str | None, base_cfg: dict[str, Any]) -> list[Phase15System]:
    if not path_str:
        systems = make_default_systems(base_cfg)
        validate_phase15_systems(systems)
        return systems
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    rows = payload.get("systems", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid systems config: {path_str}")
    systems = []
    for row in rows:
        systems.append(
            Phase15System(
                id=str(row["id"]),
                label=str(row.get("label", row["id"])),
                group=str(row["group"]),
                substrate=str(row.get("substrate", "")),
                controller=str(row.get("controller", "")),
                refresh_operator=str(row.get("refresh_operator", "none")),
                search_overrides=dict(row.get("search_overrides") or {}),
                params=dict(row.get("params") or {}),
                execution_mode=str(row.get("execution_mode", "posthoc")),
                report_alias=str(row["report_alias"]) if row.get("report_alias") is not None else None,
            )
        )
    validate_phase15_systems(systems)
    return systems


def summarize_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["group"]), str(row["system"]), int(row["budget"]))
        acc = grouped.setdefault(
            key,
            {"group": key[0], "system": key[1], "budget": key[2], "rows": 0, "_bucket_counts": {}},
        )
        acc["rows"] += 1
        bucket = str(row.get("position_bucket", "all"))
        acc["_bucket_counts"][bucket] = acc["_bucket_counts"].get(bucket, 0) + 1
        for metric_key in metric_keys:
            value = row.get(metric_key)
            if value is None:
                continue
            acc.setdefault(metric_key, 0.0)
            acc[metric_key] += float(value)

    summary = []
    for acc in grouped.values():
        rows_n = max(1, int(acc["rows"]))
        item = {k: v for k, v in acc.items() if not k.startswith("_")}
        for metric_key in metric_keys:
            if metric_key in item:
                item[metric_key] = item[metric_key] / rows_n
        item["bucket_counts"] = acc["_bucket_counts"]
        summary.append(item)
    summary.sort(key=lambda row: (row["group"], row["budget"], row["system"]))
    return summary


__all__ = [
    "Phase15System",
    "apply_system_readout",
    "candidate_undercoverage",
    "classify_position_buckets",
    "compute_argmax_path_metrics",
    "entropy",
    "first_revision_budget",
    "jaccard_overlap",
    "kl_divergence",
    "load_systems_config",
    "make_default_systems",
    "normalize_policy",
    "policy_argmax",
    "system_semantic_signature",
    "summarize_rows",
    "top2_margin",
    "topk_indices",
    "topk_recall",
    "validate_phase15_systems",
]
