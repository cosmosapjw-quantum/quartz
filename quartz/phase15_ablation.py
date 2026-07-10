"""Phase 1.5 clean-split ablation helpers.

This module replaces the older prior-revision operator study with the
phase-1.5 matrix described in ``phase15_strategy_revision_v2.md``:

- Group A: substrate/controller sanity
- Group B: refresh isolated on top of a clean substrate
- Group C: legacy anchor comparison only
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quartz.phase15_one_loop import apply_one_loop_readout


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
POSTHOC_OPERATORS = {
    "none",
    "dual_channel_commit",
    "root_challenger",
    "root_dual_posterior",
    "root_posterior_snapshot",
    "confidence_bound_posterior",
    "robust_valley_posterior",
    "entropy_annealed_posterior",
    "guarded_root_dual_posterior",
    "snapshot_trace_stabilized_posterior",
    "adaptive_snapshot_trace_stabilized_posterior",
    "entropy_expansion_stabilized_posterior",
    "budget_routing",
    "one_loop_finite_n",
}
ONLINE_OPERATORS = set(POSTHOC_OPERATORS)

PHASE15_GROUP_A_SYSTEMS = ("A0", "A1", "A2", "A3", "A4")
PHASE15_BASELINE_SYSTEMS = ("A4",)
PHASE15_GROUP_B_ALIAS_SYSTEMS = ("B0",)
PHASE15_CANDIDATE_SYSTEMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12")
# Part B experimental readouts. Deliberately kept OUT of the CANDIDATE /
# SMALL / CI battery (so default paired ablations and count assertions are
# unchanged) but included in FULL so they are selectable and registered.
PHASE15_PARTB_SYSTEMS = ("B13",)
PHASE15_LEGACY_ANCHOR_SYSTEMS = ("C0", "C1", "C2")
PHASE15_CI_SMOKE_SYSTEMS = ("A4", "B1", "B2")
PHASE15_SMALL_ABLATION_SYSTEMS = PHASE15_BASELINE_SYSTEMS + PHASE15_CANDIDATE_SYSTEMS
PHASE15_ONLINE_EXPLORATORY_SYSTEMS = PHASE15_CANDIDATE_SYSTEMS + PHASE15_LEGACY_ANCHOR_SYSTEMS
PHASE15_FULL_SYSTEMS = (
    PHASE15_GROUP_A_SYSTEMS
    + PHASE15_GROUP_B_ALIAS_SYSTEMS
    + PHASE15_CANDIDATE_SYSTEMS
    + PHASE15_PARTB_SYSTEMS
    + PHASE15_LEGACY_ANCHOR_SYSTEMS
)
PHASE15_SYSTEM_PRESETS = {
    "ci": PHASE15_CI_SMOKE_SYSTEMS,
    "ci_smoke": PHASE15_CI_SMOKE_SYSTEMS,
    "smoke": PHASE15_CI_SMOKE_SYSTEMS,
    "candidate": PHASE15_CANDIDATE_SYSTEMS,
    "candidates": PHASE15_CANDIDATE_SYSTEMS,
    "small": PHASE15_SMALL_ABLATION_SYSTEMS,
    "small_ablation": PHASE15_SMALL_ABLATION_SYSTEMS,
    "toy": PHASE15_SMALL_ABLATION_SYSTEMS,
    "toy_ablation": PHASE15_SMALL_ABLATION_SYSTEMS,
    "online": PHASE15_ONLINE_EXPLORATORY_SYSTEMS,
    "online_exploratory": PHASE15_ONLINE_EXPLORATORY_SYSTEMS,
    "full": PHASE15_FULL_SYSTEMS,
    "all": PHASE15_FULL_SYSTEMS,
}


def resolve_phase15_systems_arg(value: str | None) -> tuple[str, ...]:
    raw = str(value or "").strip()
    if not raw:
        return ()
    preset_key = raw.lower().replace("-", "_")
    if preset_key in PHASE15_SYSTEM_PRESETS:
        return tuple(PHASE15_SYSTEM_PRESETS[preset_key])
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def phase15_systems_csv(value: str | None) -> str:
    return ",".join(resolve_phase15_systems_arg(value))


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


def search_relevant_signature(system: Phase15System) -> tuple[Any, ...]:
    """Search-only identity for trace-cache keying (A0-b).

    Deliberately narrower than `system_semantic_signature`: it omits
    `refresh_operator` and `params`, which drive `apply_system_readout`
    — a pure post-hoc transform of an already-built trace — and never
    reach the Rust engine (see `_client_key`/`apply_runtime_overrides`,
    which reduce to exactly `search_overrides`). Systems that only
    differ in readout (e.g. the B1..B12 posthoc family, which share
    identical `search_overrides`) must share one search trace. Keying
    the trace cache on `system.id` + the full semantic signature
    fragmented them into distinct cache entries; a partially-warm
    cache dir then mixed cached and freshly-searched (nondeterministic
    n_threads>1) traces across candidates, breaking the paired-delta
    comparison the whole ablation harness depends on. Use this
    signature for `trace_cache_key`; use `system_semantic_signature`
    for provenance/manifest rows, where the full identity is correct.
    """
    return (json.dumps(system.search_overrides, sort_keys=True, separators=(",", ":")),)


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


def _trace_policy_matrix(trace_policies: list[np.ndarray]) -> np.ndarray:
    if not trace_policies:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack([normalize_policy(policy) for policy in trace_policies], axis=0).astype(np.float32, copy=False)


def _candidate_policy_from_scores(
    prior: np.ndarray,
    scores: np.ndarray,
    candidates: list[int],
    *,
    prior_weight: float,
) -> np.ndarray:
    effective = normalize_policy(prior).copy()
    if not candidates:
        return effective
    idx = np.asarray(candidates, dtype=np.int64)
    local_scores = normalize_policy(np.asarray(scores, dtype=np.float32)[idx])
    local_prior = normalize_policy(effective[idx])
    effective[idx] = normalize_policy((1.0 - prior_weight) * local_scores + prior_weight * local_prior)
    return normalize_policy(effective)


def _tempered_policy(policy: np.ndarray, temperature: float) -> np.ndarray:
    probs = np.clip(normalize_policy(policy), 1e-8, 1.0)
    temp = max(1e-3, float(temperature))
    logits = np.log(probs) / temp
    logits -= float(np.max(logits))
    out = np.exp(logits)
    return normalize_policy(out)


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


def apply_root_posterior_snapshot(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    del params
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets)
    return posterior_now, {
        **signal_metrics,
        "belief_revision_operator": "root_posterior_snapshot",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_channel": "search_trace_final",
        "posterior_search": posterior_now.tolist(),
        "effective_policy": posterior_now.tolist(),
        "revision_occurred": int(policy_argmax(prior) != policy_argmax(posterior_now)),
        "posterior_weight": 1.0,
        "kl_prior_to_effective": kl_divergence(prior, posterior_now),
        "revision_step": first_revision_budget(
            signal_metrics["argmax_path"], signal_metrics["trace_budgets"], policy_argmax(prior)
        ),
    }


def apply_root_dual_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    posterior_prev = normalize_policy(trace_policies[-2]) if len(trace_policies) >= 2 else None
    candidates, candidate_scores = build_root_challenger_set(prior, posterior_now, params)
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets, challenger_k=max(1, len(candidates)))
    gate_confidence = compute_commit_confidence(prior, posterior_prev, posterior_now, signal_metrics, params)
    gate_threshold = float(params.get("revision_threshold", params.get("commit_threshold", 0.50)))
    revision_applied = gate_confidence >= gate_threshold
    max_weight = float(np.clip(params.get("max_posterior_weight", 0.85), 0.0, 1.0))
    posterior_weight = min(max_weight, gate_confidence) if revision_applied else 0.0

    effective = prior.copy()
    if revision_applied and candidates:
        idx = np.asarray(candidates, dtype=np.int64)
        effective[idx] = (1.0 - posterior_weight) * prior[idx] + posterior_weight * posterior_now[idx]
        effective = normalize_policy(effective)

    return effective, {
        **signal_metrics,
        "belief_revision_operator": "root_dual_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_channel": "search_trace_final",
        "posterior_search": posterior_now.tolist(),
        "effective_policy": effective.tolist(),
        "root_candidate_set": list(candidates),
        "root_candidate_scores": list(candidate_scores),
        "revision_confidence": float(gate_confidence),
        "revision_threshold": float(gate_threshold),
        "revision_occurred": int(revision_applied),
        "posterior_weight": float(posterior_weight),
        "prior_entropy": entropy(prior),
        "posterior_entropy": entropy(posterior_now),
        "effective_entropy": entropy(effective),
        "kl_prior_to_posterior": kl_divergence(prior, posterior_now),
        "kl_prior_to_effective": kl_divergence(prior, effective),
        "kl_posterior_to_effective": kl_divergence(posterior_now, effective),
        "revision_step": first_revision_budget(
            signal_metrics["argmax_path"], signal_metrics["trace_budgets"], policy_argmax(prior)
        ),
    }


def apply_confidence_bound_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    matrix = _trace_policy_matrix(trace_policies)
    volatility = np.std(matrix, axis=0) if matrix.size else np.zeros_like(posterior_now)
    penalty = float(params.get("volatility_penalty", 1.0))
    bound_scores = normalize_policy(np.maximum(posterior_now - penalty * volatility, 0.0))
    candidates, candidate_scores = build_root_challenger_set(prior, bound_scores, params)
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets, challenger_k=max(1, len(candidates)))
    confidence = float(
        np.clip(
            0.45 * signal_metrics.get("argmax_persistence", 0.0)
            + 0.35 * signal_metrics.get("top2_margin_stability", 0.0)
            + 0.20 * (1.0 / (1.0 + float(np.mean(volatility)))),
            0.0,
            1.0,
        )
    )
    threshold = float(params.get("confidence_threshold", 0.45))
    prior_weight = 1.0 - float(np.clip(params.get("posterior_weight", confidence), 0.0, 1.0))
    effective = (
        _candidate_policy_from_scores(prior, bound_scores, candidates, prior_weight=prior_weight)
        if confidence >= threshold
        else prior
    )
    return effective, {
        **signal_metrics,
        "belief_revision_operator": "confidence_bound_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "root_candidate_set": list(candidates),
        "root_candidate_scores": list(candidate_scores),
        "posterior_search": posterior_now.tolist(),
        "confidence_bound_scores": bound_scores.tolist(),
        "revision_confidence": float(confidence),
        "revision_threshold": float(threshold),
        "revision_occurred": int(confidence >= threshold and policy_argmax(effective) != policy_argmax(prior)),
        "mean_trace_volatility": float(np.mean(volatility)) if volatility.size else 0.0,
        "kl_prior_to_effective": kl_divergence(prior, effective),
    }


def apply_robust_valley_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    matrix = _trace_policy_matrix(trace_policies)
    if matrix.size:
        weights = np.linspace(float(params.get("early_weight", 0.6)), 1.0, matrix.shape[0], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        mean_policy = normalize_policy((matrix * weights[:, None]).sum(axis=0))
        volatility = np.std(matrix, axis=0)
    else:
        mean_policy = prior
        volatility = np.zeros_like(prior)
    stability = 1.0 / (1.0 + volatility / np.maximum(mean_policy, 1e-6))
    robust_scores = normalize_policy(mean_policy * (float(params.get("stability_floor", 0.20)) + stability))
    candidates, candidate_scores = build_root_challenger_set(prior, robust_scores, params)
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets, challenger_k=max(1, len(candidates)))
    prior_weight = float(np.clip(params.get("prior_weight", 0.25), 0.0, 1.0))
    effective = _candidate_policy_from_scores(prior, robust_scores, candidates, prior_weight=prior_weight)
    return effective, {
        **signal_metrics,
        "belief_revision_operator": "robust_valley_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "root_candidate_set": list(candidates),
        "root_candidate_scores": list(candidate_scores),
        "posterior_search": normalize_policy(trace_policies[-1]).tolist(),
        "robust_valley_scores": robust_scores.tolist(),
        "mean_trace_volatility": float(np.mean(volatility)) if volatility.size else 0.0,
        "revision_occurred": int(policy_argmax(effective) != policy_argmax(prior)),
        "posterior_weight": float(1.0 - prior_weight),
        "kl_prior_to_effective": kl_divergence(prior, effective),
    }


def apply_entropy_annealed_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets)
    stability = float(signal_metrics.get("argmax_persistence", 0.0))
    temp_min = float(params.get("temperature_min", 0.70))
    temp_max = float(params.get("temperature_max", 1.60))
    temperature = temp_max - np.clip(stability, 0.0, 1.0) * (temp_max - temp_min)
    annealed = _tempered_policy(posterior_now, temperature)
    posterior_weight = float(np.clip(params.get("posterior_weight", 0.75), 0.0, 1.0))
    effective = normalize_policy((1.0 - posterior_weight) * prior + posterior_weight * annealed)
    return effective, {
        **signal_metrics,
        "belief_revision_operator": "entropy_annealed_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_search": posterior_now.tolist(),
        "annealed_posterior": annealed.tolist(),
        "annealing_temperature": float(temperature),
        "revision_occurred": int(policy_argmax(effective) != policy_argmax(prior)),
        "posterior_weight": float(posterior_weight),
        "prior_entropy": entropy(prior),
        "posterior_entropy": entropy(posterior_now),
        "effective_entropy": entropy(effective),
        "kl_prior_to_effective": kl_divergence(prior, effective),
    }


def apply_guarded_root_dual_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    proposal, proposal_meta = apply_root_dual_posterior(prior, trace_policies, trace_budgets, params)
    signal_metrics = {
        key: proposal_meta[key]
        for key in (
            "argmax_path",
            "entropy_path",
            "margin_path",
            "argmax_persistence",
            "top2_margin_stability",
            "challenger_overlap",
            "posterior_entropy_slope",
            "revision_flip_flop_count",
            "num_revisions",
            "rank_swap_count",
            "trace_budgets",
        )
        if key in proposal_meta
    }

    posterior_argmax = policy_argmax(posterior_now)
    proposal_argmax = policy_argmax(proposal)
    guard_margin = top2_margin(posterior_now)
    persistence = float(proposal_meta.get("argmax_persistence", 0.0))
    margin_stability = float(proposal_meta.get("top2_margin_stability", 0.0))
    flip_flops = int(proposal_meta.get("revision_flip_flop_count", 0))
    reasons: list[str] = []

    if bool(params.get("guard_require_revision", True)) and int(proposal_meta.get("revision_occurred", 0)) == 0:
        reasons.append("proposal_inactive")
    if proposal_argmax != posterior_argmax:
        reasons.append("argmax_mismatch")
    if persistence < float(params.get("guard_persistence_floor", 0.67)):
        reasons.append("unstable_suffix")
    if guard_margin < float(params.get("guard_margin_floor", 0.03)):
        reasons.append("thin_margin")
    if margin_stability < float(params.get("guard_margin_stability_floor", 0.45)):
        reasons.append("unstable_margin_path")
    if flip_flops > int(params.get("guard_max_flip_flops", 0)):
        reasons.append("flip_flop")

    vetoed = bool(reasons)
    effective = posterior_now if vetoed else normalize_policy(proposal)
    meta = {
        **signal_metrics,
        "belief_revision_operator": "guarded_root_dual_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_channel": "search_trace_final",
        "posterior_search": posterior_now.tolist(),
        "effective_policy": effective.tolist(),
        "root_candidate_set": list(proposal_meta.get("root_candidate_set", [])),
        "root_candidate_scores": list(proposal_meta.get("root_candidate_scores", [])),
        "revision_confidence": float(proposal_meta.get("revision_confidence", 0.0)),
        "revision_threshold": float(proposal_meta.get("revision_threshold", params.get("revision_threshold", 0.0))),
        "revision_occurred": int(not vetoed and int(proposal_meta.get("revision_occurred", 0)) == 1),
        "posterior_weight": float(0.0 if vetoed else proposal_meta.get("posterior_weight", 0.0)),
        "prior_entropy": entropy(prior),
        "posterior_entropy": entropy(posterior_now),
        "effective_entropy": entropy(effective),
        "kl_prior_to_posterior": kl_divergence(prior, posterior_now),
        "kl_prior_to_effective": kl_divergence(prior, effective),
        "kl_posterior_to_effective": kl_divergence(posterior_now, effective),
        "revision_step": proposal_meta.get("revision_step"),
        "guard_operator": "argmax_tie_guard",
        "guard_vetoed": int(vetoed),
        "guard_reason": ";".join(reasons) if reasons else "passed",
        "fallback_operator": "root_posterior_snapshot" if vetoed else "none",
        "guard_margin": float(guard_margin),
        "guard_margin_floor": float(params.get("guard_margin_floor", 0.03)),
        "guard_argmax_persistence": float(persistence),
        "guard_persistence_floor": float(params.get("guard_persistence_floor", 0.67)),
        "guard_margin_stability": float(margin_stability),
        "guard_margin_stability_floor": float(params.get("guard_margin_stability_floor", 0.45)),
        "guard_flip_flop_count": int(flip_flops),
        "guard_max_flip_flops": int(params.get("guard_max_flip_flops", 0)),
        "proposal_argmax": int(proposal_argmax),
        "posterior_argmax": int(posterior_argmax),
    }
    return effective, meta


def apply_snapshot_trace_stabilized_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets)
    trace_mean = _weighted_trace_mean(trace_policies, params, fallback=posterior_now)

    stabilization_weight = float(np.clip(params.get("stabilization_weight", 0.20), 0.0, 1.0))
    proposal = normalize_policy((1.0 - stabilization_weight) * posterior_now + stabilization_weight * trace_mean)
    reasons, anchor_k, anchor_preserved, snapshot_argmax, proposal_argmax = _snapshot_stabilization_reasons(
        posterior_now,
        proposal,
        signal_metrics,
        params,
    )

    applied = not reasons
    effective = proposal if applied else posterior_now
    return effective, {
        **signal_metrics,
        "belief_revision_operator": "snapshot_trace_stabilized_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_channel": "search_trace_final",
        "posterior_search": posterior_now.tolist(),
        "trace_mean_policy": trace_mean.tolist(),
        "effective_policy": effective.tolist(),
        "stabilizer_applied": int(applied),
        "stabilization_reason": "passed" if applied else ";".join(reasons),
        "fallback_operator": "none" if applied else "root_posterior_snapshot",
        "stabilization_weight": float(stabilization_weight),
        "snapshot_anchor_k": int(anchor_k),
        "snapshot_anchor_preserved": int(anchor_preserved),
        "snapshot_argmax": int(snapshot_argmax),
        "stabilized_argmax": int(proposal_argmax),
        "revision_occurred": int(policy_argmax(effective) != policy_argmax(prior)),
        "prior_entropy": entropy(prior),
        "posterior_entropy": entropy(posterior_now),
        "effective_entropy": entropy(effective),
        "kl_prior_to_posterior": kl_divergence(prior, posterior_now),
        "kl_prior_to_effective": kl_divergence(prior, effective),
        "kl_posterior_to_effective": kl_divergence(posterior_now, effective),
    }


def _weighted_trace_mean(
    trace_policies: list[np.ndarray],
    params: dict[str, Any],
    *,
    fallback: np.ndarray,
) -> np.ndarray:
    matrix = _trace_policy_matrix(trace_policies)
    if not matrix.size:
        return normalize_policy(fallback)
    early_weight = float(params.get("trace_early_weight", 0.60))
    weights = np.linspace(early_weight, 1.0, matrix.shape[0], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)
    return normalize_policy((matrix * weights[:, None]).sum(axis=0))


def _snapshot_stabilization_reasons(
    posterior_now: np.ndarray,
    proposal: np.ndarray,
    signal_metrics: dict[str, Any],
    params: dict[str, Any],
) -> tuple[list[str], int, bool, int, int]:
    anchor_k = max(1, int(params.get("snapshot_anchor_k", 3)))
    snapshot_topk = set(topk_indices(posterior_now, anchor_k))
    proposal_topk = set(topk_indices(proposal, anchor_k))
    snapshot_argmax = policy_argmax(posterior_now)
    proposal_argmax = policy_argmax(proposal)
    anchor_preserved = snapshot_topk == proposal_topk
    reasons: list[str] = []

    if proposal_argmax != snapshot_argmax:
        reasons.append("argmax_break")
    if not anchor_preserved:
        reasons.append("topk_anchor_break")
    if float(signal_metrics.get("argmax_persistence", 0.0)) < float(params.get("stability_floor", 0.50)):
        reasons.append("unstable_suffix")
    if float(signal_metrics.get("top2_margin_stability", 0.0)) < float(params.get("margin_stability_floor", 0.50)):
        reasons.append("unstable_margin_path")
    return reasons, int(anchor_k), bool(anchor_preserved), int(snapshot_argmax), int(proposal_argmax)


def _adaptive_stabilization_weights(params: dict[str, Any]) -> list[float]:
    max_weight = float(np.clip(params.get("max_stabilization_weight", params.get("stabilization_weight", 0.45)), 0.0, 1.0))
    step = float(params.get("candidate_weight_step", 0.05))
    if step <= 0.0:
        step = max(max_weight, 1.0)
    min_weight = float(params.get("min_stabilization_weight", step))
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    if max_weight <= 0.0 or min_weight <= 0.0:
        return []

    values: list[float] = []
    current = max_weight
    while current + 1e-9 >= min_weight:
        values.append(float(round(current, 10)))
        current -= step
    return values


def apply_adaptive_snapshot_trace_stabilized_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets)
    trace_mean = _weighted_trace_mean(trace_policies, params, fallback=posterior_now)
    candidate_weights = _adaptive_stabilization_weights(params)

    first_reasons: list[str] = ["no_candidate_weight"]
    first_anchor_k = max(1, int(params.get("snapshot_anchor_k", 3)))
    first_anchor_preserved = True
    first_snapshot_argmax = policy_argmax(posterior_now)
    first_proposal_argmax = first_snapshot_argmax
    selected_weight = 0.0
    selected_proposal: np.ndarray | None = None
    selected_anchor_k = first_anchor_k
    selected_anchor_preserved = first_anchor_preserved
    selected_snapshot_argmax = first_snapshot_argmax
    selected_proposal_argmax = first_proposal_argmax

    for idx, weight in enumerate(candidate_weights):
        proposal = normalize_policy((1.0 - weight) * posterior_now + weight * trace_mean)
        reasons, anchor_k, anchor_preserved, snapshot_argmax, proposal_argmax = _snapshot_stabilization_reasons(
            posterior_now,
            proposal,
            signal_metrics,
            params,
        )
        if idx == 0:
            first_reasons = list(reasons)
            first_anchor_k = anchor_k
            first_anchor_preserved = anchor_preserved
            first_snapshot_argmax = snapshot_argmax
            first_proposal_argmax = proposal_argmax
        if reasons:
            continue
        selected_weight = float(weight)
        selected_proposal = proposal
        selected_anchor_k = anchor_k
        selected_anchor_preserved = anchor_preserved
        selected_snapshot_argmax = snapshot_argmax
        selected_proposal_argmax = proposal_argmax
        break

    applied = selected_proposal is not None
    effective = selected_proposal if applied else posterior_now
    if not applied:
        selected_anchor_k = first_anchor_k
        selected_anchor_preserved = first_anchor_preserved
        selected_snapshot_argmax = first_snapshot_argmax
        selected_proposal_argmax = first_proposal_argmax
        if "no_safe_weight" not in first_reasons:
            first_reasons.append("no_safe_weight")

    max_weight = float(np.clip(params.get("max_stabilization_weight", params.get("stabilization_weight", 0.45)), 0.0, 1.0))
    step = float(params.get("candidate_weight_step", 0.05))
    min_weight = float(params.get("min_stabilization_weight", step if step > 0.0 else max_weight))
    return effective, {
        **signal_metrics,
        "belief_revision_operator": "adaptive_snapshot_trace_stabilized_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_channel": "search_trace_final",
        "posterior_search": posterior_now.tolist(),
        "trace_mean_policy": trace_mean.tolist(),
        "effective_policy": effective.tolist(),
        "adaptive_stabilizer": 1,
        "stabilizer_applied": int(applied),
        "stabilization_reason": "passed" if applied else ";".join(first_reasons),
        "fallback_operator": "none" if applied else "root_posterior_snapshot",
        "stabilization_weight": float(selected_weight),
        "selected_stabilization_weight": float(selected_weight),
        "max_stabilization_weight": float(max_weight),
        "candidate_weight_step": float(step),
        "min_stabilization_weight": float(min_weight),
        "candidate_weight_count": int(len(candidate_weights)),
        "snapshot_anchor_k": int(selected_anchor_k),
        "snapshot_anchor_preserved": int(selected_anchor_preserved),
        "snapshot_argmax": int(selected_snapshot_argmax),
        "stabilized_argmax": int(selected_proposal_argmax),
        "revision_occurred": int(policy_argmax(effective) != policy_argmax(prior)),
        "prior_entropy": entropy(prior),
        "posterior_entropy": entropy(posterior_now),
        "effective_entropy": entropy(effective),
        "kl_prior_to_posterior": kl_divergence(prior, posterior_now),
        "kl_prior_to_effective": kl_divergence(prior, effective),
        "kl_posterior_to_effective": kl_divergence(posterior_now, effective),
    }


def apply_entropy_expansion_stabilized_posterior(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prior = normalize_policy(prior_base)
    posterior_now = normalize_policy(trace_policies[-1])
    signal_metrics = _trace_signal_metrics(trace_policies, trace_budgets)
    entropy_slope_floor = float(params.get("entropy_slope_floor", 0.25))
    entropy_slope = float(signal_metrics.get("posterior_entropy_slope", 0.0))
    gate_passed = entropy_slope >= entropy_slope_floor

    if gate_passed:
        effective, meta = apply_adaptive_snapshot_trace_stabilized_posterior(
            prior_base,
            trace_policies,
            trace_budgets,
            params,
        )
        meta["belief_revision_operator"] = "entropy_expansion_stabilized_posterior"
        meta["entropy_expansion_gate_passed"] = 1
        meta["entropy_slope_floor"] = float(entropy_slope_floor)
        return effective, meta

    trace_mean = _weighted_trace_mean(trace_policies, params, fallback=posterior_now)
    candidate_weights = _adaptive_stabilization_weights(params)
    snapshot_argmax = policy_argmax(posterior_now)
    anchor_k = max(1, int(params.get("snapshot_anchor_k", 3)))
    max_weight = float(np.clip(params.get("max_stabilization_weight", params.get("stabilization_weight", 0.45)), 0.0, 1.0))
    step = float(params.get("candidate_weight_step", 0.05))
    min_weight = float(params.get("min_stabilization_weight", step if step > 0.0 else max_weight))
    return posterior_now, {
        **signal_metrics,
        "belief_revision_operator": "entropy_expansion_stabilized_posterior",
        "belief_revision_scope": "root_only",
        "prior_base_mutated": 0,
        "posterior_channel": "search_trace_final",
        "posterior_search": posterior_now.tolist(),
        "trace_mean_policy": trace_mean.tolist(),
        "effective_policy": posterior_now.tolist(),
        "adaptive_stabilizer": 1,
        "entropy_expansion_gate_passed": 0,
        "entropy_slope_floor": float(entropy_slope_floor),
        "stabilizer_applied": 0,
        "stabilization_reason": "entropy_slope_below_floor",
        "fallback_operator": "root_posterior_snapshot",
        "stabilization_weight": 0.0,
        "selected_stabilization_weight": 0.0,
        "max_stabilization_weight": float(max_weight),
        "candidate_weight_step": float(step),
        "min_stabilization_weight": float(min_weight),
        "candidate_weight_count": int(len(candidate_weights)),
        "snapshot_anchor_k": int(anchor_k),
        "snapshot_anchor_preserved": 1,
        "snapshot_argmax": int(snapshot_argmax),
        "stabilized_argmax": int(snapshot_argmax),
        "revision_occurred": int(policy_argmax(posterior_now) != policy_argmax(prior)),
        "prior_entropy": entropy(prior),
        "posterior_entropy": entropy(posterior_now),
        "effective_entropy": entropy(posterior_now),
        "kl_prior_to_posterior": kl_divergence(prior, posterior_now),
        "kl_prior_to_effective": kl_divergence(prior, posterior_now),
        "kl_posterior_to_effective": 0.0,
    }


def budget_routing_signal(
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    target_budget: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Instability signal for budget routing, computed from the
    sub-target trace ONLY — it never looks at (or requires) a
    supra-target chunk.

    A2-a: this is split out of `apply_budget_routing` specifically so
    the online runner (`quartz.phase15_online.run_online_readout`) can
    decide whether to burst BEFORE paying for the extra tier's search.
    Before this split, the online path always evaluated
    `apply_budget_routing` mid-loop with only the sub-target trace
    accumulated so far, so `burst_idx` (which needs a supra-target
    entry) was always `None` and the burst branch was unreachable
    dead code. Reading `signal["unstable"]` here first lets the
    online runner fetch the extra chunk ONLY when actually indicated —
    the genuinely adaptive (not always-pay) version of the same
    decision `apply_budget_routing` makes when the full trace bundle
    is already available (the posthoc path).
    """
    base_trace = [p for p, b in zip(trace_policies, trace_budgets) if int(b) <= int(target_budget)]
    if not base_trace:
        base_trace = [trace_policies[0]]
    base_budgets = [int(b) for b in trace_budgets if int(b) <= int(target_budget)] or [int(target_budget)]
    signal_metrics = _trace_signal_metrics(base_trace, base_budgets, challenger_k=int(params.get("challenger_k", 4)))
    unstable = (
        float(signal_metrics["argmax_persistence"]) < float(params.get("persistence_floor", 0.60))
        or float(signal_metrics["top2_margin_stability"]) < float(params.get("margin_stability_floor", 0.72))
    )
    return {**signal_metrics, "unstable": unstable}


def apply_budget_routing(
    prior_base: np.ndarray,
    trace_policies: list[np.ndarray],
    trace_budgets: list[int],
    target_budget: int,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    signal = budget_routing_signal(trace_policies, trace_budgets, target_budget, params)
    base_trace = [p for p, b in zip(trace_policies, trace_budgets) if int(b) <= int(target_budget)]
    if not base_trace:
        base_trace = [trace_policies[0]]
    base_policy = normalize_policy(base_trace[-1])
    burst_idx = next((idx for idx, budget in enumerate(trace_budgets) if int(budget) > int(target_budget)), None)
    burst_trigger = burst_idx is not None and bool(signal["unstable"])
    effective = normalize_policy(trace_policies[burst_idx]) if burst_trigger else base_policy
    burst_budget = int(trace_budgets[burst_idx]) if burst_trigger and burst_idx is not None else int(target_budget)
    return effective, {
        **{key: value for key, value in signal.items() if key != "unstable"},
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
    if operator == "root_dual_posterior":
        return apply_root_dual_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "root_posterior_snapshot":
        return apply_root_posterior_snapshot(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "confidence_bound_posterior":
        return apply_confidence_bound_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "robust_valley_posterior":
        return apply_robust_valley_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "entropy_annealed_posterior":
        return apply_entropy_annealed_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "guarded_root_dual_posterior":
        return apply_guarded_root_dual_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "snapshot_trace_stabilized_posterior":
        return apply_snapshot_trace_stabilized_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "adaptive_snapshot_trace_stabilized_posterior":
        return apply_adaptive_snapshot_trace_stabilized_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "entropy_expansion_stabilized_posterior":
        return apply_entropy_expansion_stabilized_posterior(prior_base, trace_policies, trace_budgets, system.params)
    if operator == "budget_routing":
        return apply_budget_routing(prior_base, trace_policies, trace_budgets, target_budget, system.params)
    if operator == "one_loop_finite_n":
        return apply_one_loop_readout(prior_base, trace_policies, trace_budgets, system.params)
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
            params={
                "persistence_floor": 0.60,
                "margin_stability_floor": 0.72,
                "comparison_role": "budget_scheduler",
                "budget_confounded": True,
            },
        ),
        Phase15System(
            "B4",
            "A4 + root dual posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "root_dual_posterior",
            search_overrides=a4,
            params={
                "revision_threshold": 0.50,
                "max_posterior_weight": 0.85,
                "challenger_k": 4,
                "candidate_score_mix": 0.35,
                "candidate_tie_eps": 5e-4,
                "challenger_max_extra": 4,
                "prior_anchor_k": 2,
                "posterior_anchor_k": 4,
                "divergence_scale": 0.40,
                "entropy_scale": 0.35,
            },
        ),
        Phase15System(
            "B5",
            "A4-equivalent root posterior snapshot anchor (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "root_posterior_snapshot",
            search_overrides=a4,
            params={"comparison_role": "a4_equivalence_anchor", "equivalence_anchor": "A4"},
            report_alias="A4",
        ),
        Phase15System(
            "B6",
            "A4 + confidence-bound posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "confidence_bound_posterior",
            search_overrides=a4,
            params={
                "confidence_threshold": 0.45,
                "posterior_weight": 0.75,
                "volatility_penalty": 1.0,
                "challenger_k": 4,
                "candidate_score_mix": 0.20,
                "candidate_tie_eps": 5e-4,
                "challenger_max_extra": 4,
                "prior_anchor_k": 2,
                "posterior_anchor_k": 4,
            },
        ),
        Phase15System(
            "B7",
            "A4 + robust-valley posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "robust_valley_posterior",
            search_overrides=a4,
            params={
                "prior_weight": 0.25,
                "early_weight": 0.6,
                "stability_floor": 0.20,
                "challenger_k": 4,
                "candidate_score_mix": 0.20,
                "candidate_tie_eps": 5e-4,
                "challenger_max_extra": 4,
                "prior_anchor_k": 2,
                "posterior_anchor_k": 4,
            },
        ),
        Phase15System(
            "B8",
            "A4 + entropy-annealed posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "entropy_annealed_posterior",
            search_overrides=a4,
            params={
                "posterior_weight": 0.75,
                "temperature_min": 0.70,
                "temperature_max": 1.60,
            },
        ),
        Phase15System(
            "B9",
            "A4 + argmax/tie-guarded dual posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "guarded_root_dual_posterior",
            search_overrides=a4,
            params={
                "revision_threshold": 0.50,
                "max_posterior_weight": 0.85,
                "challenger_k": 4,
                "candidate_score_mix": 0.35,
                "candidate_tie_eps": 5e-4,
                "challenger_max_extra": 4,
                "prior_anchor_k": 2,
                "posterior_anchor_k": 4,
                "divergence_scale": 0.40,
                "entropy_scale": 0.35,
                "guard_require_revision": True,
                "guard_persistence_floor": 0.67,
                "guard_margin_floor": 0.03,
                "guard_margin_stability_floor": 0.45,
                "guard_max_flip_flops": 0,
                "comparison_role": "argmax_tie_guarded_readout",
            },
        ),
        Phase15System(
            "B10",
            "A4 + snapshot-safe trace-stabilized posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "snapshot_trace_stabilized_posterior",
            search_overrides=a4,
            params={
                "stabilization_weight": 0.20,
                "trace_early_weight": 0.60,
                "snapshot_anchor_k": 3,
                "stability_floor": 0.50,
                "margin_stability_floor": 0.50,
                "comparison_role": "snapshot_safe_stabilized_readout",
            },
        ),
        Phase15System(
            "B11",
            "A4 + adaptive snapshot-safe trace-stabilized posterior (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "adaptive_snapshot_trace_stabilized_posterior",
            search_overrides=a4,
            params={
                "max_stabilization_weight": 0.45,
                "candidate_weight_step": 0.05,
                "min_stabilization_weight": 0.05,
                "trace_early_weight": 0.60,
                "snapshot_anchor_k": 3,
                "stability_floor": 0.50,
                "margin_stability_floor": 0.50,
                "comparison_role": "adaptive_snapshot_safe_stabilized_readout",
            },
        ),
        Phase15System(
            "B12",
            "A4 + entropy-expansion-gated snapshot stabilizer (posthoc)",
            "B",
            "S1",
            "QuartzVL",
            "entropy_expansion_stabilized_posterior",
            search_overrides=a4,
            params={
                "max_stabilization_weight": 0.45,
                "candidate_weight_step": 0.05,
                "min_stabilization_weight": 0.05,
                "trace_early_weight": 0.60,
                "snapshot_anchor_k": 3,
                "stability_floor": 0.50,
                "margin_stability_floor": 0.50,
                "entropy_slope_floor": 0.25,
                "comparison_role": "entropy_expansion_stabilized_readout",
            },
        ),
        Phase15System(
            "B13",
            "A4 + one-loop finite-N curvature readout (posthoc, Part B H2)",
            "B",
            "S1",
            "QuartzVL",
            "one_loop_finite_n",
            # Same search substrate as A4/B-series so B13 shares the SAME
            # trace per (checkpoint, position) — search_relevant_signature
            # is identical, preserving same-trace paired-delta pairing.
            search_overrides=a4,
            params={
                "one_loop_curvature": 1.0,
                "one_loop_n_floor": 1.0,
                "comparison_role": "one_loop_finite_n_readout",
            },
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
    "PHASE15_BASELINE_SYSTEMS",
    "PHASE15_CANDIDATE_SYSTEMS",
    "PHASE15_CI_SMOKE_SYSTEMS",
    "PHASE15_FULL_SYSTEMS",
    "PHASE15_GROUP_A_SYSTEMS",
    "PHASE15_GROUP_B_ALIAS_SYSTEMS",
    "PHASE15_LEGACY_ANCHOR_SYSTEMS",
    "PHASE15_ONLINE_EXPLORATORY_SYSTEMS",
    "PHASE15_PARTB_SYSTEMS",
    "PHASE15_SMALL_ABLATION_SYSTEMS",
    "PHASE15_SYSTEM_PRESETS",
    "Phase15System",
    "apply_confidence_bound_posterior",
    "apply_entropy_annealed_posterior",
    "apply_entropy_expansion_stabilized_posterior",
    "apply_guarded_root_dual_posterior",
    "apply_adaptive_snapshot_trace_stabilized_posterior",
    "apply_robust_valley_posterior",
    "apply_root_dual_posterior",
    "apply_root_posterior_snapshot",
    "apply_snapshot_trace_stabilized_posterior",
    "apply_system_readout",
    "budget_routing_signal",
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
    "phase15_systems_csv",
    "policy_argmax",
    "resolve_phase15_systems_arg",
    "system_semantic_signature",
    "summarize_rows",
    "top2_margin",
    "topk_indices",
    "topk_recall",
    "validate_phase15_systems",
]
