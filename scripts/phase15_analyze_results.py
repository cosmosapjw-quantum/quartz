#!/usr/bin/env python3
"""Analyze Phase15 posthoc and continuation-benchmark result artifacts.

This script is intentionally analysis-only. It does not promote candidates or
turn small rehearsals into validation evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


CLAIM_STATUS = "ANALYSIS-ONLY; candidate quality remains ABLATION-PENDING"
# A2-b audit fix: paired_posthoc_deltas previously reported only a bare
# mean of per-pair deltas with no confidence interval and no
# multiple-comparison correction. With up to 12 candidates x 3
# baselines screened in one run (36 comparisons), and effect sizes as
# small as the ledger's own delta_kl_to_oracle=-0.00079, that is
# structurally primed to surface false champions on noise-scale
# effects. BOOTSTRAP_RESAMPLES/BASE_ALPHA are the defaults for the
# nonparametric paired bootstrap CI added below; nonparametric because
# Phase15 rows are not iid (adaptive search, shared checkpoints/
# positions across systems) — the same "trust nothing about the
# sampling distribution" posture the project already uses elsewhere
# (e.g. src/ablation_refresh.rs's bootstrap CI for flip rate), not a
# parametric t-interval.
BOOTSTRAP_RESAMPLES = 2000
BASE_ALPHA = 0.05
AUTO_TARGET_TOKEN = "auto"
CANDIDATE_TELEMETRY_KEYS = frozenset(
    {
        "guard_vetoed",
        "guard_reason",
        "stabilizer_applied",
        "stabilization_reason",
        "snapshot_anchor_preserved",
        "selected_stabilization_weight",
        "candidate_weight_count",
        "adaptive_stabilizer",
        "entropy_expansion_gate_passed",
        "entropy_slope_floor",
        "continuation_guard_vetoed",
        "continuation_guard_reason",
        "restart_guard_vetoed",
        "restart_guard_reason",
        "continuation_stabilizer_applied",
        "continuation_stabilization_reason",
        "restart_stabilizer_applied",
        "restart_stabilization_reason",
        "continuation_selected_stabilization_weight",
        "restart_selected_stabilization_weight",
        "continuation_entropy_expansion_gate_passed",
        "restart_entropy_expansion_gate_passed",
        "budget_confounded",
        "budget_burst_triggered",
        "extra_budget_used",
        "burst_budget",
        "burst_reason",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Phase15 candidate result artifacts"
    )
    parser.add_argument(
        "--posthoc-rows",
        default="results/phase15_ablation_b9_aligned_best/gomoku7/assays/phase15_rows.jsonl",
    )
    parser.add_argument(
        "--benchmark-rows",
        default="results/phase15_benchmarks_b9_aligned_best/gomoku7/phase15_continuation_benchmark_rows.jsonl",
    )
    parser.add_argument("--targets", default="B9,B10,B11,B12")
    parser.add_argument("--baselines", default="A4,B4,B5")
    parser.add_argument(
        "--research-grade",
        action="store_true",
        help="enforce the full phase15 research-grade gate on the rows/manifest/report",
    )
    parser.add_argument(
        "--manifest", default=None, help="run manifest json (for artifact-hash gate)"
    )
    parser.add_argument("--min-seed-families", type=int, default=3)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def parse_csv(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        return 0.0
    return float(mean(rows))


def _round(value: float) -> float:
    return float(round(float(value), 12))


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    alpha: float = BASE_ALPHA,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a list of paired deltas.

    Fewer than 2 paired observations can't support a bootstrap; returns
    (0.0, 0.0) in that case (callers should treat that as "no CI",
    not as a real interval around zero).

    Deterministic: uses a local `random.Random(seed)` instance, not the
    global RNG state, so repeated calls (or calls interleaved with
    other randomness elsewhere in the process) always reproduce the
    same CI for the same input — required for a research-grade
    artifact.
    """
    n = len(deltas)
    if n < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    resample_means = []
    for _ in range(n_resamples):
        resample_means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    resample_means.sort()
    lower_idx = max(0, min(int((alpha / 2.0) * n_resamples), n_resamples - 1))
    upper_idx = max(0, min(int((1.0 - alpha / 2.0) * n_resamples) - 1, n_resamples - 1))
    return (resample_means[lower_idx], resample_means[upper_idx])


def _system_sort_key(system: str) -> tuple[str, int, str]:
    split_at = len(system)
    while split_at > 0 and system[split_at - 1].isdigit():
        split_at -= 1
    suffix = system[split_at:]
    if suffix:
        return (system[:split_at], int(suffix), "")
    return (system, -1, system)


def _sorted_systems(systems: Iterable[str]) -> list[str]:
    return sorted(systems, key=_system_sort_key)


def _systems(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {str(row["system"]) for row in rows if "system" in row}


def _has_candidate_telemetry(row: dict[str, Any]) -> bool:
    return any(key in row for key in CANDIDATE_TELEMETRY_KEYS)


def telemetry_candidate_systems(
    posthoc_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]] | None = None,
    *,
    baselines: tuple[str, ...] = (),
) -> list[str]:
    baseline_set = set(baselines)
    rows = list(posthoc_rows)
    if benchmark_rows is not None:
        rows.extend(benchmark_rows)
    return _sorted_systems(
        {
            str(row["system"])
            for row in rows
            if "system" in row
            and str(row["system"]) not in baseline_set
            and _has_candidate_telemetry(row)
        }
    )


def expand_analysis_targets(
    posthoc_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    *,
    targets: tuple[str, ...],
    baselines: tuple[str, ...],
) -> tuple[str, ...]:
    if len(targets) == 1 and targets[0].lower() == AUTO_TARGET_TOKEN:
        inferred = telemetry_candidate_systems(
            posthoc_rows, benchmark_rows, baselines=baselines
        )
        if inferred:
            return tuple(inferred)
        baseline_set = set(baselines)
        return tuple(
            _sorted_systems(
                (_systems(posthoc_rows) | _systems(benchmark_rows)) - baseline_set
            )
        )
    return targets


def analysis_coverage(
    *,
    posthoc_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    targets: tuple[str, ...],
    requested_targets: tuple[str, ...],
    baselines: tuple[str, ...],
) -> dict[str, Any]:
    posthoc_systems = _sorted_systems(_systems(posthoc_rows))
    benchmark_systems = _sorted_systems(_systems(benchmark_rows))
    available_systems = _sorted_systems(set(posthoc_systems) | set(benchmark_systems))
    telemetry_systems = telemetry_candidate_systems(
        posthoc_rows, benchmark_rows, baselines=baselines
    )
    target_set = set(targets)
    available_set = set(available_systems)
    return {
        "requested_targets": list(requested_targets),
        "auto_targets": len(requested_targets) == 1
        and requested_targets[0].lower() == AUTO_TARGET_TOKEN,
        "targets": list(targets),
        "baselines": list(baselines),
        "available_systems": available_systems,
        "posthoc_systems": posthoc_systems,
        "benchmark_systems": benchmark_systems,
        "telemetry_candidate_systems": telemetry_systems,
        "untargeted_telemetry_systems": _sorted_systems(
            set(telemetry_systems) - target_set
        ),
        "missing_target_systems": _sorted_systems(target_set - available_set),
    }


def summarize_posthoc_by_system(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    out = []
    for system, group in sorted(grouped.items()):
        out.append(
            {
                "system": system,
                "rows": len(group),
                "accuracy_to_oracle": _round(
                    _mean(row.get("accuracy_to_oracle", 0.0) for row in group)
                ),
                "accuracy_to_reference": _round(
                    _mean(row.get("accuracy_to_reference", 0.0) for row in group)
                ),
                "topk_recall_oracle": _round(
                    _mean(row.get("topk_recall_oracle", 0.0) for row in group)
                ),
                "kl_to_oracle": _round(
                    _mean(row.get("kl_to_oracle", 0.0) for row in group)
                ),
                "revision_occurred": _round(
                    _mean(row.get("revision_occurred", 0.0) for row in group)
                ),
                "guard_vetoed": _round(
                    _mean(row.get("guard_vetoed", 0.0) for row in group)
                ),
                "stabilizer_applied": _round(
                    _mean(row.get("stabilizer_applied", 0.0) for row in group)
                ),
                "entropy_expansion_gate_passed": _round(
                    _mean(
                        row.get("entropy_expansion_gate_passed", 0.0) for row in group
                    )
                ),
                "selected_stabilization_weight": _round(
                    _mean(
                        row.get(
                            "selected_stabilization_weight",
                            row.get("stabilization_weight", 0.0),
                        )
                        for row in group
                    )
                ),
            }
        )
    return out


def summarize_benchmark_by_system(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    out = []
    for system, group in sorted(grouped.items()):
        out.append(
            {
                "system": system,
                "rows": len(group),
                "tie_aware_match_rate": _round(
                    _mean(row.get("tie_aware_match", 0.0) for row in group)
                ),
                "argmax_match_rate": _round(
                    _mean(row.get("argmax_match", 0.0) for row in group)
                ),
                "ambiguous_top1_rate": _round(
                    _mean(row.get("ambiguous_top1_case", 0.0) for row in group)
                ),
                "policy_kl_restart_vs_continuation": _round(
                    _mean(
                        row.get("policy_kl_restart_vs_continuation", 0.0)
                        for row in group
                    )
                ),
                "continuation_guard_vetoed": _round(
                    _mean(row.get("continuation_guard_vetoed", 0.0) for row in group)
                ),
                "restart_guard_vetoed": _round(
                    _mean(row.get("restart_guard_vetoed", 0.0) for row in group)
                ),
                "continuation_stabilizer_applied": _round(
                    _mean(
                        row.get("continuation_stabilizer_applied", 0.0) for row in group
                    )
                ),
                "restart_stabilizer_applied": _round(
                    _mean(row.get("restart_stabilizer_applied", 0.0) for row in group)
                ),
                "continuation_entropy_expansion_gate_passed": _round(
                    _mean(
                        row.get("continuation_entropy_expansion_gate_passed", 0.0)
                        for row in group
                    )
                ),
                "restart_entropy_expansion_gate_passed": _round(
                    _mean(
                        row.get("restart_entropy_expansion_gate_passed", 0.0)
                        for row in group
                    )
                ),
                "continuation_selected_stabilization_weight": _round(
                    _mean(
                        row.get(
                            "continuation_selected_stabilization_weight",
                            row.get("continuation_stabilization_weight", 0.0),
                        )
                        for row in group
                    )
                ),
                "restart_selected_stabilization_weight": _round(
                    _mean(
                        row.get(
                            "restart_selected_stabilization_weight",
                            row.get("restart_stabilization_weight", 0.0),
                        )
                        for row in group
                    )
                ),
            }
        )
    return out


def paired_posthoc_deltas(
    rows: list[dict[str, Any]],
    *,
    targets: tuple[str, ...] = ("B9",),
    baselines: tuple[str, ...] = ("A4", "B4", "B5"),
    base_alpha: float = BASE_ALPHA,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[dict[str, Any]]:
    index = {
        (
            str(row["checkpoint_id"]),
            str(row["position_id"]),
            int(row["budget"]),
            str(row["system"]),
        ): row
        for row in rows
    }
    # A2-b: Bonferroni-correct the CI's confidence level for the full
    # family of (target, baseline) comparisons this call screens, so a
    # single call with e.g. 12 targets x 3 baselines = 36 comparisons
    # doesn't silently run 36 independent 95%-CI tests (~1-(0.95^36)
    # ~= 84% chance of at least one false "distinguishable" claim by
    # chance alone).
    n_comparisons = max(1, len(targets) * len(baselines))
    corrected_alpha = base_alpha / n_comparisons
    out = []
    for target in targets:
        target_rows = [row for row in rows if str(row.get("system")) == target]
        for baseline in baselines:
            acc_deltas: list[float] = []
            topk_deltas: list[float] = []
            kl_deltas: list[float] = []
            win_count = 0
            loss_count = 0
            tie_count = 0
            for target_row in target_rows:
                key = (
                    str(target_row["checkpoint_id"]),
                    str(target_row["position_id"]),
                    int(target_row["budget"]),
                    baseline,
                )
                baseline_row = index.get(key)
                if baseline_row is None:
                    continue
                acc_delta = float(target_row.get("accuracy_to_oracle", 0.0)) - float(
                    baseline_row.get("accuracy_to_oracle", 0.0)
                )
                topk_delta = float(target_row.get("topk_recall_oracle", 0.0)) - float(
                    baseline_row.get("topk_recall_oracle", 0.0)
                )
                kl_delta = float(target_row.get("kl_to_oracle", 0.0)) - float(
                    baseline_row.get("kl_to_oracle", 0.0)
                )
                acc_deltas.append(acc_delta)
                topk_deltas.append(topk_delta)
                kl_deltas.append(kl_delta)
                if acc_delta > 0:
                    win_count += 1
                elif acc_delta < 0:
                    loss_count += 1
                else:
                    tie_count += 1
            acc_ci = paired_bootstrap_ci(
                acc_deltas, alpha=corrected_alpha, n_resamples=n_resamples
            )
            topk_ci = paired_bootstrap_ci(
                topk_deltas, alpha=corrected_alpha, n_resamples=n_resamples
            )
            kl_ci = paired_bootstrap_ci(
                kl_deltas, alpha=corrected_alpha, n_resamples=n_resamples
            )
            out.append(
                {
                    "target": target,
                    "baseline": baseline,
                    "pairs": len(acc_deltas),
                    "delta_accuracy_to_oracle": _round(_mean(acc_deltas)),
                    "delta_accuracy_to_oracle_ci": [
                        _round(acc_ci[0]),
                        _round(acc_ci[1]),
                    ],
                    "delta_accuracy_to_oracle_ci_excludes_zero": bool(
                        acc_ci[0] > 0.0 or acc_ci[1] < 0.0
                    ),
                    "delta_topk_recall_oracle": _round(_mean(topk_deltas)),
                    "delta_topk_recall_oracle_ci": [
                        _round(topk_ci[0]),
                        _round(topk_ci[1]),
                    ],
                    "delta_topk_recall_oracle_ci_excludes_zero": bool(
                        topk_ci[0] > 0.0 or topk_ci[1] < 0.0
                    ),
                    "delta_kl_to_oracle": _round(_mean(kl_deltas)),
                    "delta_kl_to_oracle_ci": [_round(kl_ci[0]), _round(kl_ci[1])],
                    "delta_kl_to_oracle_ci_excludes_zero": bool(
                        kl_ci[0] > 0.0 or kl_ci[1] < 0.0
                    ),
                    "accuracy_win_count": int(win_count),
                    "accuracy_loss_count": int(loss_count),
                    "accuracy_tie_count": int(tie_count),
                }
            )
    return out


def guard_summary(
    rows: list[dict[str, Any]], *, system: str, prefix: str = ""
) -> dict[str, Any]:
    veto_key = f"{prefix}_guard_vetoed" if prefix else "guard_vetoed"
    reason_key = f"{prefix}_guard_reason" if prefix else "guard_reason"
    group = [
        row
        for row in rows
        if str(row.get("system")) == system and (veto_key in row or reason_key in row)
    ]
    reason_counts = Counter(str(row.get(reason_key, "missing")) for row in group)
    by_budget_out = []
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        if "budget" in row:
            by_budget[int(row["budget"])].append(row)
    for budget, budget_rows in sorted(by_budget.items()):
        by_budget_out.append(
            {
                "budget": int(budget),
                "rows": len(budget_rows),
                "veto_rate": _round(
                    _mean(row.get(veto_key, 0.0) for row in budget_rows)
                ),
            }
        )
    return {
        "system": system,
        "prefix": prefix,
        "rows": len(group),
        "veto_rate": _round(_mean(row.get(veto_key, 0.0) for row in group)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_budget": by_budget_out,
    }


def stabilizer_summary(
    rows: list[dict[str, Any]], *, system: str, prefix: str = ""
) -> dict[str, Any]:
    applied_key = f"{prefix}_stabilizer_applied" if prefix else "stabilizer_applied"
    reason_key = f"{prefix}_stabilization_reason" if prefix else "stabilization_reason"
    group = [
        row
        for row in rows
        if str(row.get("system")) == system
        and (applied_key in row or reason_key in row)
    ]
    reason_counts = Counter(str(row.get(reason_key, "missing")) for row in group)
    by_budget_out = []
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        if "budget" in row:
            by_budget[int(row["budget"])].append(row)
    for budget, budget_rows in sorted(by_budget.items()):
        by_budget_out.append(
            {
                "budget": int(budget),
                "rows": len(budget_rows),
                "applied_rate": _round(
                    _mean(row.get(applied_key, 0.0) for row in budget_rows)
                ),
            }
        )
    return {
        "system": system,
        "prefix": prefix,
        "rows": len(group),
        "applied_rate": _round(_mean(row.get(applied_key, 0.0) for row in group)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "by_budget": by_budget_out,
    }


def budget_scheduler_summary(
    rows: list[dict[str, Any]], *, system: str
) -> dict[str, Any]:
    group = [
        row
        for row in rows
        if str(row.get("system")) == system
        and (
            "budget_confounded" in row
            or "budget_burst_triggered" in row
            or "extra_budget_used" in row
            or "burst_budget" in row
        )
    ]
    reason_counts = Counter(str(row.get("burst_reason", "missing")) for row in group)
    positive_extra = [
        float(row.get("extra_budget_used", 0.0))
        for row in group
        if float(row.get("extra_budget_used", 0.0)) > 0.0
    ]
    by_budget_out = []
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in group:
        if "budget" in row:
            by_budget[int(row["budget"])].append(row)
    for budget, budget_rows in sorted(by_budget.items()):
        by_budget_out.append(
            {
                "budget": int(budget),
                "rows": len(budget_rows),
                "budget_burst_rate": _round(
                    _mean(row.get("budget_burst_triggered", 0.0) for row in budget_rows)
                ),
                "extra_budget_used_mean": _round(
                    _mean(row.get("extra_budget_used", 0.0) for row in budget_rows)
                ),
                "burst_budget_mean": _round(
                    _mean(
                        row.get("burst_budget", row.get("budget", 0.0))
                        for row in budget_rows
                    )
                ),
            }
        )
    return {
        "system": system,
        "rows": len(group),
        "budget_confounded_rate": _round(
            _mean(row.get("budget_confounded", 0.0) for row in group)
        ),
        "budget_burst_rate": _round(
            _mean(row.get("budget_burst_triggered", 0.0) for row in group)
        ),
        "extra_budget_used_mean": _round(
            _mean(row.get("extra_budget_used", 0.0) for row in group)
        ),
        "positive_extra_budget_used_mean": _round(_mean(positive_extra)),
        "burst_reason_counts": dict(sorted(reason_counts.items())),
        "by_budget": by_budget_out,
    }


def interpretation_flags(
    *,
    posthoc_deltas: list[dict[str, Any]],
    posthoc_summary: list[dict[str, Any]],
    benchmark_summary: list[dict[str, Any]],
    coverage: dict[str, Any] | None = None,
    budget_scheduler_summaries: list[dict[str, Any]] | None = None,
) -> list[str]:
    flags: set[str] = set()
    for row in posthoc_deltas:
        target = str(row["target"])
        baseline = str(row["baseline"])
        # A2-b: a positive-looking accuracy delta whose Bonferroni-
        # corrected bootstrap CI still straddles zero is not a quality
        # claim, no matter how the point estimate reads.
        if float(row.get("delta_accuracy_to_oracle", 0.0)) > 0.0 and not bool(
            row.get("delta_accuracy_to_oracle_ci_excludes_zero", False)
        ):
            flags.add(
                f"{target}_vs_{baseline}_accuracy_gain_not_distinguishable_from_zero"
            )
        if (
            baseline == "A4"
            and float(row["delta_accuracy_to_oracle"]) <= 0.0
            and float(row["delta_kl_to_oracle"]) > 0.0
        ):
            flags.add(f"{target}_no_accuracy_gain_vs_A4_with_higher_kl")
        if (
            baseline == "A4"
            and float(row["delta_accuracy_to_oracle"]) >= 0.0
            and float(row["delta_topk_recall_oracle"]) >= 0.0
            and float(row["delta_kl_to_oracle"]) < 0.0
        ):
            flags.add(
                f"{target}_rehearsal_lower_kl_without_accuracy_or_topk_loss_vs_A4"
            )
        if (
            baseline == "B4"
            and float(row["delta_accuracy_to_oracle"]) > 0.0
            and float(row["delta_kl_to_oracle"]) < 0.0
        ):
            flags.add(f"{target}_rehearsal_better_than_B4_on_accuracy_and_kl")
    by_system_post = {str(row["system"]): row for row in posthoc_summary}
    by_system_bench = {str(row["system"]): row for row in benchmark_summary}
    if float(by_system_post.get("B9", {}).get("guard_vetoed", 0.0)) >= 0.75:
        flags.add("B9_high_posthoc_guard_veto_rate")
    if (
        float(by_system_bench.get("B9", {}).get("continuation_guard_vetoed", 0.0))
        >= 0.95
    ):
        flags.add("B9_high_continuation_guard_veto_rate")
    if (
        "B4" in by_system_bench
        and "A4" in by_system_bench
        and float(by_system_bench["B4"].get("policy_kl_restart_vs_continuation", 0.0))
        < float(by_system_bench["A4"].get("policy_kl_restart_vs_continuation", 0.0))
        and float(by_system_bench["B4"].get("tie_aware_match_rate", 0.0)) < 0.25
    ):
        flags.add("B4_low_kl_but_argmax_unsafe")
    if coverage is not None and coverage.get("untargeted_telemetry_systems"):
        flags.add("analysis_targets_omit_available_telemetry_candidates")
    if coverage is not None and coverage.get("missing_target_systems"):
        flags.add("analysis_targets_missing_from_artifacts")
    budget_confounded_systems = {
        str(row["system"])
        for row in budget_scheduler_summaries or []
        if float(row.get("budget_confounded_rate", 0.0)) > 0.0
        or float(row.get("extra_budget_used_mean", 0.0)) > 0.0
    }
    for row in posthoc_deltas:
        if (
            str(row.get("target")) in budget_confounded_systems
            and str(row.get("baseline")) == "A4"
            and float(row.get("delta_accuracy_to_oracle", 0.0)) > 0.0
        ):
            flags.add(f"{row['target']}_budget_confounded_quality_signal")
    return sorted(flags)


def build_analysis_report(
    *,
    posthoc_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    targets: tuple[str, ...] = ("B9",),
    baselines: tuple[str, ...] = ("A4", "B4", "B5"),
    base_alpha: float = BASE_ALPHA,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    requested_targets = targets
    targets = expand_analysis_targets(
        posthoc_rows, benchmark_rows, targets=targets, baselines=baselines
    )
    posthoc_summary = summarize_posthoc_by_system(posthoc_rows)
    benchmark_summary = summarize_benchmark_by_system(benchmark_rows)
    deltas = paired_posthoc_deltas(
        posthoc_rows,
        targets=targets,
        baselines=baselines,
        base_alpha=base_alpha,
        n_resamples=n_resamples,
    )
    n_comparisons = max(1, len(targets) * len(baselines))
    coverage = analysis_coverage(
        posthoc_rows=posthoc_rows,
        benchmark_rows=benchmark_rows,
        targets=targets,
        requested_targets=requested_targets,
        baselines=baselines,
    )
    posthoc_guard_systems = sorted(
        {
            str(row["system"])
            for row in posthoc_rows
            if "guard_vetoed" in row or "guard_reason" in row
        }
    )
    benchmark_guard_systems = sorted(
        {
            str(row["system"])
            for row in benchmark_rows
            if "continuation_guard_vetoed" in row or "restart_guard_vetoed" in row
        }
    )
    posthoc_stabilizer_systems = sorted(
        {
            str(row["system"])
            for row in posthoc_rows
            if "stabilizer_applied" in row or "stabilization_reason" in row
        }
    )
    posthoc_budget_scheduler_systems = _sorted_systems(
        {
            str(row["system"])
            for row in posthoc_rows
            if (
                "budget_confounded" in row
                or "budget_burst_triggered" in row
                or "extra_budget_used" in row
                or "burst_budget" in row
            )
        }
    )
    benchmark_stabilizer_systems = sorted(
        {
            str(row["system"])
            for row in benchmark_rows
            if "continuation_stabilizer_applied" in row
            or "restart_stabilizer_applied" in row
        }
    )
    budget_scheduler_summaries = [
        budget_scheduler_summary(posthoc_rows, system=system)
        for system in posthoc_budget_scheduler_systems
    ]
    return {
        "format_version": 1,
        "claim_status": CLAIM_STATUS,
        "posthoc_rows": len(posthoc_rows),
        "benchmark_rows": len(benchmark_rows),
        "targets": list(targets),
        "baselines": list(baselines),
        # A2-b: how many (target, baseline) comparisons this report
        # screens in one call, and the confidence level actually used
        # for each delta's CI after Bonferroni-correcting for that
        # count. Every paired_posthoc_deltas row's *_ci_excludes_zero
        # flags are computed AT corrected_alpha, not base_alpha —
        # readers should not re-interpret them as plain 95% CIs.
        "screening_multiplicity": {
            "n_candidates_screened": len(targets),
            "n_baselines": len(baselines),
            "n_comparisons": n_comparisons,
            "base_alpha": base_alpha,
            "bonferroni_corrected_alpha": base_alpha / n_comparisons,
            "bootstrap_resamples": n_resamples,
        },
        "analysis_coverage": coverage,
        "posthoc_by_system": posthoc_summary,
        "benchmark_by_system": benchmark_summary,
        "paired_posthoc_deltas": deltas,
        "posthoc_guard_summary": [
            guard_summary(posthoc_rows, system=system)
            for system in posthoc_guard_systems
        ],
        "benchmark_guard_summary": [
            guard_summary(benchmark_rows, system=system, prefix=prefix)
            for system in benchmark_guard_systems
            for prefix in ("continuation", "restart")
        ],
        "posthoc_stabilizer_summary": [
            stabilizer_summary(posthoc_rows, system=system)
            for system in posthoc_stabilizer_systems
        ],
        "posthoc_budget_scheduler_summary": budget_scheduler_summaries,
        "benchmark_stabilizer_summary": [
            stabilizer_summary(benchmark_rows, system=system, prefix=prefix)
            for system in benchmark_stabilizer_systems
            for prefix in ("continuation", "restart")
        ],
        "interpretation_flags": interpretation_flags(
            posthoc_deltas=deltas,
            posthoc_summary=posthoc_summary,
            benchmark_summary=benchmark_summary,
            coverage=coverage,
            budget_scheduler_summaries=budget_scheduler_summaries,
        ),
    }


def main() -> None:
    args = parse_args()
    posthoc_rows = load_jsonl(args.posthoc_rows)
    benchmark_rows = load_jsonl(args.benchmark_rows)
    report = build_analysis_report(
        posthoc_rows=posthoc_rows,
        benchmark_rows=benchmark_rows,
        targets=parse_csv(args.targets),
        baselines=parse_csv(args.baselines),
    )
    if getattr(args, "research_grade", False):
        # Stage 7 / C10: enforce the full research-grade gate on the actual rows.
        from quartz.phase15_research_grade import (
            check_research_grade,
            enforce_research_grade,
        )

        manifest_path = Path(args.manifest).resolve() if args.manifest else None
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path
            else {}
        )
        systems = sorted({str(r.get("system")) for r in posthoc_rows if "system" in r})
        checkpoints = sorted(
            {str(r.get("checkpoint_id")) for r in posthoc_rows if "checkpoint_id" in r}
        )
        positions = sorted(
            {str(r.get("position_id")) for r in posthoc_rows if "position_id" in r}
        )
        budgets = sorted({int(r["budget"]) for r in posthoc_rows if "budget" in r})
        rg = check_research_grade(
            checkpoints=checkpoints,
            rows=posthoc_rows,
            manifest=manifest,
            systems=systems,
            n_positions=len(positions),
            n_budgets=len(budgets),
            analyzer_report=report,
            min_seed_families=int(args.min_seed_families),
            artifact_root=manifest_path.parent if manifest_path else None,
        )
        report["research_grade_gate"] = rg
        enforce_research_grade(rg)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
