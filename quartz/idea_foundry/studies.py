"""Claim-safe first scientific gates for the 26-axis Idea Foundry.

This module is deliberately separate from :mod:`axis_workflow`: contract gates
prove that a lane is runnable, while these studies produce paired estimands from
real Phase-15 traces or preregistered synthetic banks.  Neither layer is a play
strength evaluation.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quartz.experiment_manifest import atomic_json_dump, file_sha256, utc_now
from quartz.idea_foundry.axis_workflow import atomic_jsonl_dump
from quartz.idea_foundry.search import A26NestedContourExactLab


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_REGISTRY = REPO_ROOT / "configs" / "idea_foundry.studies.v1.json"
PHASE15_ROWS = (
    REPO_ROOT
    / "results"
    / "phase15_ablation"
    / "gomoku7"
    / "assays"
    / "phase15_rows.jsonl"
)
STAGE7_ROWS = (
    REPO_ROOT
    / "results"
    / "phase15_stage7"
    / "posthoc"
    / "gomoku7"
    / "assays"
    / "phase15_rows.jsonl"
)
POSITION_SUITE = (
    REPO_ROOT / "results" / "phase15_ablation" / "gomoku7" / "position_suite.json"
)
STUDY_SCHEMA_VERSION = 1
TERMINAL_STATUS = "completed_no_promotion"


class StudyError(RuntimeError):
    """Raised when a study cannot satisfy its preregistered contract."""


@dataclass(frozen=True)
class StudySpec:
    axis_id: str
    gate_kind: str
    runner: str
    estimand_id: str
    effect_scale: str
    reference_id: str
    unit: str
    higher_is_better: bool
    pilot_seconds: int
    full_seconds: int


@dataclass
class StudyOutcome:
    rows: list[dict[str, Any]]
    grouped_effects: dict[str, list[float]]
    status: str = TERMINAL_STATUS
    outcome_detail: str = "FIRST_SCIENTIFIC_GATE_COMPLETED"
    notes: tuple[str, ...] = ()
    inputs: tuple[Path, ...] = ()


def _strict_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise StudyError(f"required regular JSON file is missing: {path}")

    def reject(value: str) -> None:
        raise StudyError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StudyError(f"invalid JSON input {path}: {exc}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise StudyError(f"required regular JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudyError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise StudyError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise StudyError(f"JSONL input has no rows: {path}")
    return rows


def load_study_specs(path: Path = STUDY_REGISTRY) -> tuple[StudySpec, ...]:
    raw = _strict_json(path)
    axes = raw.get("axes") if isinstance(raw, dict) else None
    if raw.get("schema_version") != 1 or not isinstance(axes, list):
        raise StudyError("study registry schema mismatch")
    specs: list[StudySpec] = []
    for row in axes:
        if not isinstance(row, dict):
            raise StudyError("study registry axis rows must be objects")
        try:
            spec = StudySpec(**row)
        except TypeError as exc:
            raise StudyError(f"invalid study spec: {exc}") from exc
        specs.append(spec)
    expected = [f"A{index:02d}" for index in range(1, 27)]
    if [spec.axis_id for spec in specs] != expected:
        raise StudyError("study registry must cover A01-A26 once in order")
    if any(spec.pilot_seconds < 1 or spec.full_seconds < 1 for spec in specs):
        raise StudyError("study duration estimates must be positive")
    return tuple(specs)


def study_spec(axis_id: str) -> StudySpec:
    normalized = axis_id.upper()
    for spec in load_study_specs():
        if spec.axis_id == normalized:
            return spec
    raise StudyError(f"unknown study axis: {axis_id}")


def study_plan() -> dict[str, Any]:
    specs = load_study_specs()
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "suite": "first-scientific-gate-all",
        "axis_count": len(specs),
        "axes": [
            {
                **spec.__dict__,
                "pilot_command": (
                    "venv/bin/python scripts/idea_foundry_study.py run "
                    f"--axis {spec.axis_id} --profile pilot"
                ),
                "full_command": (
                    "venv/bin/python scripts/idea_foundry_study.py run "
                    f"--axis {spec.axis_id} --profile full"
                ),
            }
            for spec in specs
        ],
        "estimated_seconds": {
            "pilot": sum(spec.pilot_seconds for spec in specs),
            "full": sum(spec.full_seconds for spec in specs),
        },
        "claim_scope": "first_scientific_gate_diagnostic_only",
        "promotion": {"auto": False, "eligible": False},
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise StudyError("cannot compute a mean from an empty sample")
    return float(statistics.fmean(values))


def _standard_error(values: Sequence[float]) -> float:
    if not values:
        raise StudyError("cannot compute standard error from an empty sample")
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probabilities)


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _string_salt(value: str) -> int:
    return int.from_bytes(sha256(value.encode("utf-8")).digest()[:4], "big")


def _normalized(probabilities: Sequence[float]) -> list[float]:
    clipped = [max(0.0, float(value)) for value in probabilities]
    total = sum(clipped)
    if total <= 0:
        return [1.0 / len(clipped)] * len(clipped)
    return [value / total for value in clipped]


def _phase15_cells(
    profile: str,
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[str]]:
    rows = [row for row in _jsonl(PHASE15_ROWS) if row.get("system") == "A4"]
    positions = sorted({str(row["position_id"]) for row in rows})
    if profile == "pilot":
        positions = positions[:16]
    allowed = set(positions)
    cells = {
        (str(row["checkpoint_id"]), str(row["position_id"]), int(row["budget"])): row
        for row in rows
        if str(row["position_id"]) in allowed
    }
    checkpoints = sorted({key[0] for key in cells})
    for checkpoint in checkpoints:
        for position in positions:
            for budget in (8, 16, 32, 64):
                if (checkpoint, position, budget) not in cells:
                    raise StudyError(
                        "Phase-15 trace does not contain the complete preregistered grid: "
                        f"{checkpoint}/{position}/{budget}"
                    )
    return cells, positions


def _trace_outcome(axis_id: str, profile: str, seed: int) -> StudyOutcome:
    cells, positions = _phase15_cells(profile)
    checkpoints = sorted({key[0] for key in cells})
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed ^ int(axis_id[1:]))

    if axis_id == "A03":
        for position in positions:
            values = [
                float(
                    cells[(checkpoint, position, 64)]["effective_policy"][
                        int(cells[(checkpoint, position, 64)]["oracle_best"])
                    ]
                )
                for checkpoint in checkpoints
            ]
            center = _mean(values)
            epistemic = statistics.stdev(values) if len(values) > 1 else 0.0
            for checkpoint in checkpoints:
                low = cells[(checkpoint, position, 16)]
                high = cells[(checkpoint, position, 64)]
                oracle = int(high["oracle_best"])
                low_probability = float(low["effective_policy"][oracle])
                target = float(high["effective_policy"][oracle])
                mc_radius = abs(float(low["posterior_entropy_slope"])) + 1e-6
                drift_radius = abs(target - low_probability)
                sum_radius = mc_radius + epistemic + drift_radius
                rss_radius = math.sqrt(
                    mc_radius * mc_radius
                    + epistemic * epistemic
                    + drift_radius * drift_radius
                )
                sum_cover = float(abs(target - center) <= sum_radius)
                rss_cover = float(abs(target - center) <= rss_radius)
                effect = sum_cover - rss_cover
                grouped[checkpoint].append(effect)
                rows.append(
                    _unit_row(
                        axis_id,
                        checkpoint,
                        position,
                        effect,
                        candidate=sum_cover,
                        reference=rss_cover,
                        extras={
                            "mc_radius": mc_radius,
                            "epistemic_radius": epistemic,
                            "drift_radius": drift_radius,
                        },
                    )
                )
        return StudyOutcome(rows, dict(grouped), inputs=(PHASE15_ROWS,))

    if axis_id == "A04":
        for checkpoint in checkpoints:
            records = []
            for position in positions:
                low = cells[(checkpoint, position, 16)]
                high = cells[(checkpoint, position, 64)]
                correctable = max(
                    0.0,
                    float(high["accuracy_to_oracle"])
                    - float(low["accuracy_to_oracle"]),
                )
                uncertainty = _entropy(low["effective_policy"])
                records.append((position, uncertainty, correctable))
            allocation = max(1, len(records) // 4)
            ranked = sorted(records, key=lambda item: (-item[1], item[0]))
            selected_gain = sum(item[2] for item in ranked[:allocation]) / allocation
            shuffled = list(records)
            rng.shuffle(shuffled)
            uniform_gain = sum(item[2] for item in shuffled[:allocation]) / allocation
            effect = selected_gain - uniform_gain
            grouped[checkpoint].append(effect)
            rows.append(
                _unit_row(
                    axis_id,
                    checkpoint,
                    f"allocation-{allocation}",
                    effect,
                    candidate=selected_gain,
                    reference=uniform_gain,
                    extras={"allocation_count": allocation},
                )
            )
        return StudyOutcome(rows, dict(grouped), inputs=(PHASE15_ROWS,))

    if axis_id == "A22":
        for checkpoint in checkpoints:
            features: list[float] = []
            targets: list[float] = []
            for position in positions:
                low = cells[(checkpoint, position, 16)]
                high = cells[(checkpoint, position, 64)]
                features.append(
                    abs(float(low["posterior_entropy_slope"]))
                    + (1.0 - float(low["top2_margin_stability"]))
                )
                targets.append(
                    float(low["argmax_effective"] != high["argmax_effective"])
                )
            observed = _squared_correlation(features, targets)
            shuffled = list(features)
            rng.shuffle(shuffled)
            null = _squared_correlation(shuffled, targets)
            effect = observed - null
            grouped[checkpoint].append(effect)
            rows.append(
                _unit_row(
                    axis_id,
                    checkpoint,
                    "surrogate-null",
                    effect,
                    candidate=observed,
                    reference=null,
                    extras={"permutation_seed": seed ^ _string_salt(checkpoint)},
                )
            )
        return StudyOutcome(
            rows,
            dict(grouped),
            outcome_detail="ANALYSIS_ONLY_SURROGATE_FALSIFICATION_COMPLETED",
            inputs=(PHASE15_ROWS,),
        )

    for checkpoint in checkpoints:
        for position in positions:
            row8 = cells[(checkpoint, position, 8)]
            row16 = cells[(checkpoint, position, 16)]
            row32 = cells[(checkpoint, position, 32)]
            row64 = cells[(checkpoint, position, 64)]
            candidate = reference = effect = 0.0
            extras: dict[str, Any] = {}
            if axis_id == "A01":
                stable16 = (
                    float(row16["argmax_persistence"]) == 1.0
                    and float(row16["top2_margin_stability"]) >= 0.75
                    and abs(float(row16["posterior_entropy_slope"])) <= 0.02
                )
                stable32 = (
                    float(row32["argmax_persistence"]) == 1.0
                    and float(row32["top2_margin_stability"]) >= 0.75
                )
                selected_budget = 16 if stable16 else 32 if stable32 else 64
                selected = cells[(checkpoint, position, selected_budget)]
                candidate = (64.0 - selected_budget) / 64.0
                reference = 0.0
                effect = candidate
                extras = {
                    "selected_budget": selected_budget,
                    "oracle_accuracy_delta": float(selected["accuracy_to_oracle"])
                    - float(row64["accuracy_to_oracle"]),
                }
            elif axis_id == "A02":
                anchor = _normalized(row8["effective_policy"])
                live = _normalized(row16["effective_policy"])
                mixed = _normalized(
                    [
                        math.sqrt(max(a, 1e-12) * max(b, 1e-12))
                        for a, b in zip(anchor, live)
                    ]
                )
                oracle = int(row64["oracle_best"])
                reference = -math.log(max(live[oracle], 1e-12))
                candidate = -math.log(max(mixed[oracle], 1e-12))
                effect = reference - candidate
                extras = {"anchor_budget": 8, "live_budget": 16, "lambda": 1.0}
            elif axis_id == "A09":
                target = float(row16["argmax_effective"] != row64["argmax_effective"])
                base_rate = 0.25
                score = _sigmoid(
                    -2.0
                    + 4.0 * abs(float(row16["posterior_entropy_slope"]))
                    + 2.0 * (1.0 - float(row16["top2_margin_stability"]))
                    + float(row16["revision_flip_flop_count"])
                )
                candidate = (target - score) ** 2
                reference = (target - base_rate) ** 2
                effect = reference - candidate
                extras = {"target_changed": bool(target), "router_probability": score}
            elif axis_id == "A20":
                score = (
                    1.0
                    - float(row16["accuracy_to_oracle"])
                    + _entropy(row16["effective_policy"]) / math.log(49)
                )
                target = float(row64["accuracy_to_oracle"] == 0)
                candidate = score * target
                reference = 0.5 * target
                effect = candidate - reference
                extras = {"archive_priority": score, "future_error": bool(target)}
            elif axis_id == "A21":
                target = float(row16["argmax_effective"] != row64["argmax_effective"])
                stability_score = 1.0 - float(row16["argmax_persistence"])
                coherence = min(
                    1.0,
                    abs(float(row16["posterior_entropy_slope"]))
                    + 0.25 * float(row16["revision_flip_flop_count"]),
                )
                candidate_score = min(1.0, 0.5 * stability_score + 0.5 * coherence)
                candidate = (target - candidate_score) ** 2
                reference = (target - stability_score) ** 2
                effect = reference - candidate
                extras = {
                    "coherence_feature": coherence,
                    "target_changed": bool(target),
                }
            elif axis_id == "A24":
                entropy = _entropy(row16["effective_policy"]) / math.log(49)
                selected_budget = 16 if entropy < 0.55 else 64
                selected = row16 if selected_budget == 16 else row64
                compute_saving = (64.0 - selected_budget) / 64.0
                error_cost = float(row64["accuracy_to_oracle"]) - float(
                    selected["accuracy_to_oracle"]
                )
                candidate = compute_saving - error_cost
                reference = 0.0
                effect = candidate
                extras = {
                    "selected_budget": selected_budget,
                    "compute_saving": compute_saving,
                    "oracle_error_cost": error_cost,
                }
            else:
                raise StudyError(
                    f"no Phase-15 trace recipe is registered for {axis_id}"
                )
            grouped[checkpoint].append(effect)
            rows.append(
                _unit_row(
                    axis_id,
                    checkpoint,
                    position,
                    effect,
                    candidate=candidate,
                    reference=reference,
                    extras=extras,
                )
            )
    detail = (
        "ANALYSIS_ONLY_SHADOW_COMPLETED"
        if axis_id == "A21"
        else "FIRST_SCIENTIFIC_GATE_COMPLETED"
    )
    return StudyOutcome(
        rows,
        dict(grouped),
        outcome_detail=detail,
        inputs=(PHASE15_ROWS,),
    )


def _squared_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_sum = sum((x - x_mean) ** 2 for x in x_values)
    y_sum = sum((y - y_mean) ** 2 for y in y_values)
    if x_sum <= 0 or y_sum <= 0:
        return 0.0
    return float((covariance * covariance) / (x_sum * y_sum))


def _unit_row(
    axis_id: str,
    group: str,
    unit_id: str,
    effect: float,
    *,
    candidate: float,
    reference: float,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not all(math.isfinite(value) for value in (effect, candidate, reference)):
        raise StudyError(f"non-finite study result for {axis_id}/{group}/{unit_id}")
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "axis_id": axis_id,
        "independent_group_id": str(group),
        "unit_id": str(unit_id),
        "candidate": float(candidate),
        "reference": float(reference),
        "paired_effect": float(effect),
        **dict(extras or {}),
    }


def _synthetic_means(rng: random.Random, scenario: int) -> list[float]:
    kind = scenario % 3
    if kind == 0:
        means = [0.72, 0.67, 0.60, 0.55, 0.48, 0.42, 0.36, 0.30]
    elif kind == 1:
        means = [0.60, 0.595, 0.59, 0.56, 0.52, 0.48, 0.42, 0.36]
    else:
        means = [0.67, 0.66, 0.49, 0.48, 0.66, 0.65, 0.40, 0.39]
    jitter = [rng.uniform(-0.005, 0.005) for _ in means]
    return [min(0.95, max(0.05, value + delta)) for value, delta in zip(means, jitter)]


def _sample_mean(rng: random.Random, probability: float, count: int) -> float:
    return sum(rng.random() < probability for _ in range(count)) / count


def _synthetic_outcome(axis_id: str, profile: str, seed: int) -> StudyOutcome:
    scenario_count = 24 if profile == "pilot" else 240
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for replicate_seed in (41, 42, 43):
        rng = random.Random((seed << 24) ^ (replicate_seed << 8) ^ int(axis_id[1:]))
        for scenario in range(scenario_count):
            means = _synthetic_means(rng, scenario)
            candidate = reference = effect = 0.0
            extras: dict[str, Any] = {"scenario_kind": scenario % 3}
            if axis_id == "A05":
                incumbent = _sample_mean(rng, means[0], 16)
                challenger = _sample_mean(rng, means[1], 16)
                stop_regret = max(means) - means[0 if incumbent >= challenger else 1]
                sampled = [_sample_mean(rng, means[index], 48) for index in range(2)]
                sample_choice = max(range(2), key=sampled.__getitem__)
                sample_regret = max(means) - means[sample_choice]
                prior_order = sorted(range(len(means)), key=lambda index: -means[index])
                widen_choice = max(prior_order[:4], key=means.__getitem__)
                widen_regret = max(means) - means[widen_choice]
                reference = stop_regret
                candidate = min(sample_regret, widen_regret)
                effect = (reference - candidate) / 48.0
                extras.update(
                    {
                        "actions": [
                            "STOP",
                            "SAMPLE_incumbent",
                            "SAMPLE_challenger",
                            "WIDEN",
                        ],
                        "resident_root_identity": f"synthetic-{replicate_seed}-{scenario}",
                        "fork_semantics": "deterministic_common_random_numbers",
                    }
                )
            elif axis_id == "A06":
                initial = [_sample_mean(rng, mean, 4) for mean in means]
                survivors = sorted(range(8), key=lambda index: -initial[index])[:4]
                second = {
                    index: _sample_mean(rng, means[index], 8) for index in survivors
                }
                survivors = sorted(survivors, key=lambda index: -second[index])[:2]
                final = {
                    index: _sample_mean(rng, means[index], 20) for index in survivors
                }
                candidate_choice = max(survivors, key=final.__getitem__)
                uniform_scores = [_sample_mean(rng, mean, 8) for mean in means]
                reference_choice = max(range(8), key=uniform_scores.__getitem__)
                candidate = max(means) - means[candidate_choice]
                reference = max(means) - means[reference_choice]
                effect = reference - candidate
                extras["budget_preserved"] = 64
            elif axis_id == "A07":
                noisy_prior = [mean + rng.gauss(0.0, 0.12) for mean in means]
                live = sorted(range(8), key=lambda index: -noisy_prior[index])[:3]
                fixed_choice = max(live, key=means.__getitem__)
                residual_mass = sum(
                    math.exp(noisy_prior[index])
                    for index in range(8)
                    if index not in live
                ) / sum(math.exp(value) for value in noisy_prior)
                widened = list(live)
                if residual_mass > 0.35:
                    widened.extend(
                        index
                        for index in sorted(
                            range(8), key=lambda item: -noisy_prior[item]
                        )
                        if index not in live
                    )
                candidate_choice = max(widened, key=means.__getitem__)
                reference = max(means) - means[fixed_choice]
                candidate = max(means) - means[candidate_choice]
                effect = reference - candidate
                extras["residual_mass"] = residual_mass
            elif axis_id == "A11":
                true_value = _mean(means[:2] + means[4:6])
                single_mode = _mean(means[:2])
                particle_samples = [means[rng.choice((0, 1, 4, 5))] for _ in range(32)]
                particle_value = _mean(particle_samples)
                reference = abs(single_mode - true_value)
                candidate = abs(particle_value - true_value)
                effect = reference - candidate
                extras["particle_count"] = 32
            elif axis_id == "A12":
                target = _normalized([max(value - 0.2, 0.01) for value in means])
                uniform_counts = [0] * 8
                balanced_counts = [0] * 8
                uniform_state = rng.randrange(8)
                balanced_state = uniform_state
                for _ in range(256):
                    proposal = (uniform_state + rng.choice((-1, 1))) % 8
                    uniform_state = proposal
                    uniform_counts[uniform_state] += 1
                    proposal = (balanced_state + rng.choice((-1, 1))) % 8
                    acceptance = min(1.0, target[proposal] / target[balanced_state])
                    if rng.random() < acceptance:
                        balanced_state = proposal
                    balanced_counts[balanced_state] += 1
                uniform_empirical = _normalized(uniform_counts)
                balanced_empirical = _normalized(balanced_counts)
                reference = 0.5 * sum(
                    abs(a - b) for a, b in zip(uniform_empirical, target)
                )
                candidate = 0.5 * sum(
                    abs(a - b) for a, b in zip(balanced_empirical, target)
                )
                effect = reference - candidate
                extras["transition_steps"] = 256
            elif axis_id == "A13":
                workers = 16
                probabilities = _normalized([math.exp(5.0 * value) for value in means])
                pending_unaware = [
                    max(range(8), key=probabilities.__getitem__) for _ in range(workers)
                ]
                pending = [0] * 8
                pending_aware = []
                for _ in range(workers):
                    choice = max(
                        range(8),
                        key=lambda index: probabilities[index] / (1.0 + pending[index]),
                    )
                    pending_aware.append(choice)
                    pending[choice] += 1
                reference = 1.0 - len(set(pending_unaware)) / workers
                candidate = 1.0 - len(set(pending_aware)) / workers
                effect = reference - candidate
                extras["pending_not_evidence"] = True
            elif axis_id == "A16":
                parents = 4 + scenario % 4
                transpositions = 2 + scenario % 3
                tree_calls = parents * transpositions
                graph_calls = transpositions
                reference = float(tree_calls)
                candidate = float(graph_calls)
                effect = reference - candidate
                extras.update(
                    {
                        "parent_edge_stats_shared": False,
                        "evaluation_cache_shared": True,
                    }
                )
            elif axis_id == "A25":
                temperature = (0.0, 0.05, 0.2, 0.8)[scenario % 4]
                maximum = max(means)
                if temperature == 0.0:
                    soft = maximum
                else:
                    prior = [1.0 / len(means)] * len(means)
                    soft = (
                        temperature
                        * math.log(
                            sum(
                                p * math.exp((value - maximum) / temperature)
                                for value, p in zip(means, prior)
                            )
                        )
                        + maximum
                    )
                candidate = soft - maximum
                reference = 0.0
                effect = abs(candidate)
                extras.update(
                    {
                        "temperature": temperature,
                        "temperature_zero_exact": temperature == 0.0,
                    }
                )
            elif axis_id == "A26":
                likelihoods = [max(1e-4, value) for value in means]
                prior = _normalized([rng.random() + 0.1 for _ in means])
                enumeration = A26NestedContourExactLab.enumerated_evidence(
                    likelihoods, prior
                )
                contour = A26NestedContourExactLab.finite_contour_evidence(
                    likelihoods, prior
                )
                candidate = abs(contour - enumeration)
                reference = 0.0
                effect = candidate
                extras["state_count"] = len(means)
            else:
                raise StudyError(f"no synthetic recipe is registered for {axis_id}")
            group = f"seed-{replicate_seed}"
            grouped[group].append(effect)
            rows.append(
                _unit_row(
                    axis_id,
                    group,
                    f"scenario-{scenario:04d}",
                    effect,
                    candidate=candidate,
                    reference=reference,
                    extras=extras,
                )
            )
    detail = {
        "A05": "SYNTHETIC_FREEZE_FORK_GATE_COMPLETED_NO_RESIDENT_TRACE_CLAIM",
        "A25": "OBJECTIVE_MISMATCH_ABLATION_COMPLETED_DEFAULT_UNCHANGED",
        "A26": "ANALYSIS_ONLY_EXACT_FINITE_GATE_COMPLETED",
    }.get(axis_id, "FIRST_SYNTHETIC_SCIENTIFIC_GATE_COMPLETED")
    notes = (
        (
            "A05 uses a deterministic synthetic resident root because the retained Phase-15 trace uses independent restarts.",
        )
        if axis_id == "A05"
        else ()
    )
    return StudyOutcome(rows, dict(grouped), outcome_detail=detail, notes=notes)


def _position_outcome(axis_id: str, profile: str) -> StudyOutcome:
    raw = _strict_json(POSITION_SUITE)
    positions = raw.get("positions") if isinstance(raw, dict) else None
    if not isinstance(positions, list) or not positions:
        raise StudyError("position suite contains no positions")
    if profile == "pilot":
        positions = positions[:16]
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    if axis_id == "A08":
        eligible = 0
        for index, position in enumerate(positions):
            board = [int(value) for value in position["board"]]
            player = int(position["player"])
            wins = _immediate_actions(board, player)
            blocks = _immediate_actions(board, -player)
            forced = wins or blocks
            if not forced:
                continue
            eligible += 1
            sentinel = min(forced)
            prior = int(position["prior_argmax"])
            candidate = float(sentinel in forced)
            reference = float(prior in forced)
            effect = candidate - reference
            group = f"position-fold-{index % 3}"
            grouped[group].append(effect)
            rows.append(
                _unit_row(
                    axis_id,
                    group,
                    str(position["id"]),
                    effect,
                    candidate=candidate,
                    reference=reference,
                    extras={"forced_actions": forced, "sentinel_action": sentinel},
                )
            )
        if eligible == 0:
            return StudyOutcome(
                [
                    _unit_row(
                        axis_id,
                        "eligibility-audit",
                        "position-suite",
                        0.0,
                        candidate=0.0,
                        reference=0.0,
                        extras={"eligible_positions": 0},
                    )
                ],
                {},
                status="skipped",
                outcome_detail="DORMANT_NO_ELIGIBLE_TACTICAL_POSITION",
                inputs=(POSITION_SUITE,),
            )
    elif axis_id == "A23":
        for index, position in enumerate(positions):
            board = [int(value) for value in position["board"]]
            empty = [cell for cell, value in enumerate(board) if value == 0]
            if not empty:
                continue
            action = empty[index % len(empty)]
            player = int(position["player"])
            before = _line_feature_vector(board)
            after_board = list(board)
            after_board[action] = player
            direct = _line_feature_vector(after_board)
            incremental = _incremental_line_features(board, before, action, player)
            unmade = list(after_board)
            unmade[action] = 0
            restored = _line_feature_vector(unmade)
            exact = float(incremental == direct and restored == before)
            group = f"position-fold-{index % 3}"
            grouped[group].append(exact)
            rows.append(
                _unit_row(
                    axis_id,
                    group,
                    str(position["id"]),
                    exact,
                    candidate=exact,
                    reference=1.0,
                    extras={"action": action, "make_unmake_exact": bool(exact)},
                )
            )
    else:
        raise StudyError(f"no position-suite recipe is registered for {axis_id}")
    return StudyOutcome(rows, dict(grouped), inputs=(POSITION_SUITE,))


def _winning(board: Sequence[int], player: int) -> bool:
    size = 7
    for row in range(size):
        for col in range(size):
            if board[row * size + col] != player:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(
                    0 <= row + step * dr < size
                    and 0 <= col + step * dc < size
                    and board[(row + step * dr) * size + col + step * dc] == player
                    for step in range(4)
                ):
                    return True
    return False


def _immediate_actions(board: Sequence[int], player: int) -> list[int]:
    actions: list[int] = []
    for action, value in enumerate(board):
        if value != 0:
            continue
        candidate = list(board)
        candidate[action] = player
        if _winning(candidate, player):
            actions.append(action)
    return actions


def _all_lines() -> tuple[tuple[int, ...], ...]:
    lines: list[tuple[int, ...]] = []
    size = 7
    for row in range(size):
        lines.append(tuple(row * size + col for col in range(size)))
    for col in range(size):
        lines.append(tuple(row * size + col for row in range(size)))
    for start_col in range(size):
        diag = tuple(step * size + start_col + step for step in range(size - start_col))
        if len(diag) >= 4:
            lines.append(diag)
        anti = tuple(step * size + start_col - step for step in range(start_col + 1))
        if len(anti) >= 4:
            lines.append(anti)
    for start_row in range(1, size):
        diag = tuple(
            (start_row + step) * size + step for step in range(size - start_row)
        )
        if len(diag) >= 4:
            lines.append(diag)
        anti = tuple(
            (start_row + step) * size + (size - 1 - step)
            for step in range(size - start_row)
        )
        if len(anti) >= 4:
            lines.append(anti)
    return tuple(lines)


BOARD_LINES = _all_lines()


def _line_summary(board: Sequence[int], line: Sequence[int]) -> tuple[int, int, int]:
    values = [board[index] for index in line]
    return values.count(1), values.count(-1), values.count(0)


def _line_feature_vector(board: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(_line_summary(board, line) for line in BOARD_LINES)


def _incremental_line_features(
    board: Sequence[int],
    before: Sequence[tuple[int, int, int]],
    action: int,
    player: int,
) -> tuple[tuple[int, int, int], ...]:
    updated = list(before)
    for index, line in enumerate(BOARD_LINES):
        if action not in line:
            continue
        positive, negative, empty = updated[index]
        updated[index] = (
            positive + int(player == 1),
            negative + int(player == -1),
            empty - 1,
        )
    return tuple(updated)


def _conditional_outcome(axis_id: str) -> StudyOutcome:
    if axis_id != "A10":
        raise StudyError(f"no conditional audit recipe is registered for {axis_id}")
    rows = [
        _unit_row(
            axis_id,
            "eligibility-audit",
            "retained-phase15-trace",
            0.0,
            candidate=0.0,
            reference=0.0,
            extras={
                "eligible_slice_present": False,
                "reason": "no bounded-refresh arm is present in the retained Phase-15 trace",
            },
        )
    ]
    return StudyOutcome(
        rows,
        {},
        status="skipped",
        outcome_detail="DORMANT_NO_ELIGIBLE_SLICE",
        notes=("The preregistered no-refresh default remains unchanged.",),
        inputs=(PHASE15_ROWS,),
    )


def _stage7_outcome(profile: str) -> StudyOutcome:
    rows = _jsonl(STAGE7_ROWS)
    selected = [row for row in rows if row.get("system") in {"A4", "B13"}]
    positions = sorted({str(row["position_id"]) for row in selected})
    if profile == "pilot":
        positions = positions[:16]
    allowed = set(positions)
    by_key = {
        (
            str(row["checkpoint_id"]),
            str(row["position_id"]),
            int(row["budget"]),
            str(row["system"]),
        ): row
        for row in selected
        if str(row["position_id"]) in allowed
    }
    checkpoints = sorted({key[0] for key in by_key})
    output_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for checkpoint in checkpoints:
        for position in positions:
            for budget in (8, 16, 32, 64):
                baseline = by_key.get((checkpoint, position, budget, "A4"))
                candidate = by_key.get((checkpoint, position, budget, "B13"))
                if baseline is None or candidate is None:
                    raise StudyError(
                        f"Stage-7 paired B13 cell is missing: {checkpoint}/{position}/{budget}"
                    )
                effect = float(baseline["kl_to_oracle"]) - float(
                    candidate["kl_to_oracle"]
                )
                grouped[checkpoint].append(effect)
                output_rows.append(
                    _unit_row(
                        "A17",
                        checkpoint,
                        f"{position}-b{budget}",
                        effect,
                        candidate=float(candidate["kl_to_oracle"]),
                        reference=float(baseline["kl_to_oracle"]),
                        extras={
                            "budget": budget,
                            "decision_neutral": candidate["argmax_effective"]
                            == baseline["argmax_effective"],
                        },
                    )
                )
    return StudyOutcome(
        output_rows,
        dict(grouped),
        outcome_detail="REAL_TRACE_ANALYSIS_ONLY_B13_GATE_COMPLETED",
        inputs=(STAGE7_ROWS,),
    )


def _path_outcome(profile: str, seed: int) -> StudyOutcome:
    cells, positions = _phase15_cells(profile)
    checkpoints = sorted({key[0] for key in cells})
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for checkpoint in checkpoints:
        rng = random.Random(seed ^ _string_salt(checkpoint))
        effects: list[float] = []
        for position in positions:
            path = [
                str(value) for value in cells[(checkpoint, position, 64)]["argmax_path"]
            ]
            shingles = {
                "|".join(path[index : index + 2])
                for index in range(max(1, len(path) - 1))
            }
            near = set(shingles)
            if near:
                victim = sorted(near)[0]
                near.remove(victim)
                near.add(victim + "-mut")
            unrelated = {
                f"u{rng.randrange(1000)}|u{rng.randrange(1000)}"
                for _ in shingles or [""]
            }
            near_similarity = _jaccard(shingles, near)
            unrelated_similarity = _jaccard(shingles, unrelated)
            candidate_accuracy = 0.5 * (
                float(near_similarity >= 0.5) + float(unrelated_similarity < 0.5)
            )
            edge_only_accuracy = 0.5 * (
                float(bool(path) and path[0] == path[0]) + float(bool(path))
            )
            effect = candidate_accuracy - edge_only_accuracy
            effects.append(effect)
            rows.append(
                _unit_row(
                    "A14",
                    checkpoint,
                    position,
                    effect,
                    candidate=candidate_accuracy,
                    reference=edge_only_accuracy,
                    extras={
                        "near_jaccard": near_similarity,
                        "unrelated_jaccard": unrelated_similarity,
                        "shadow_only": True,
                    },
                )
            )
        grouped[checkpoint].extend(effects)
    return StudyOutcome(
        rows,
        dict(grouped),
        outcome_detail="PATH_REDUNDANCY_SHADOW_GATE_COMPLETED",
        inputs=(PHASE15_ROWS,),
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def execute_inprocess(axis_id: str, profile: str, seed: int) -> StudyOutcome:
    if profile not in {"pilot", "full"}:
        raise StudyError(f"unsupported study profile: {profile}")
    spec = study_spec(axis_id)
    if spec.runner == "phase15_trace":
        return _trace_outcome(spec.axis_id, profile, seed)
    if spec.runner == "synthetic_bank":
        return _synthetic_outcome(spec.axis_id, profile, seed)
    if spec.runner == "position_suite":
        return _position_outcome(spec.axis_id, profile)
    if spec.runner == "conditional_audit":
        return _conditional_outcome(spec.axis_id)
    if spec.runner == "phase15_stage7":
        return _stage7_outcome(profile)
    if spec.runner == "phase15_paths":
        return _path_outcome(profile, seed)
    if spec.runner.endswith("_native"):
        raise StudyError(
            f"{spec.axis_id} requires its native executor adapter; use scripts/idea_foundry_study.py"
        )
    raise StudyError(f"unsupported study runner: {spec.runner}")


def _ensure_output(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (REPO_ROOT / "results").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise StudyError(
            f"study output must remain under {allowed}: {resolved}"
        ) from exc
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_dir() or any(resolved.iterdir()):
            raise StudyError(f"study output must be a new empty directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_plot(
    path: Path, spec: StudySpec, effects: Sequence[Mapping[str, Any]]
) -> None:
    with tempfile.TemporaryDirectory(prefix="quartz-study-plot-") as mpl_dir:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [str(row["independent_group_id"]) for row in effects]
            values = [float(row["effect"]) for row in effects]
            errors = [1.96 * float(row["standard_error"]) for row in effects]
            fig, axis = plt.subplots(figsize=(8.5, 4.8))
            if values:
                axis.bar(labels, values, yerr=errors, capsize=4, color="#457b9d")
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No eligible effect estimate",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.axhline(0.0, color="#264653", linewidth=1)
            axis.set_ylabel(f"{spec.estimand_id} [{spec.unit}]")
            axis.set_xlabel("Independent group")
            axis.set_title(
                f"{spec.axis_id} DIAGNOSTIC — first {spec.gate_kind} gate\n"
                "not play strength or production evidence"
            )
            fig.tight_layout()
            temporary = path.with_suffix(".tmp.png")
            fig.savefig(temporary, dpi=160)
            plt.close(fig)
            os.replace(temporary, path)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def publish_outcome(
    *,
    axis_id: str,
    profile: str,
    seed: int,
    output_dir: Path,
    outcome: StudyOutcome,
    extra_sources: Sequence[Path] = (),
) -> dict[str, Any]:
    spec = study_spec(axis_id)
    target = _ensure_output(output_dir)
    rows_path = target / "rows.jsonl"
    effects_path = target / "effect_records.jsonl"
    summary_path = target / "summary.json"
    plot_path = target / "diagnostic.png"
    interpretation_path = target / "interpretation.md"
    atomic_jsonl_dump(rows_path, outcome.rows)
    rows_hash = file_sha256(rows_path)
    effect_records = []
    group_means = {
        group: _mean(values)
        for group, values in sorted(outcome.grouped_effects.items())
        if values
    }
    between_group_se = (
        _standard_error(list(group_means.values())) if len(group_means) > 1 else 0.0
    )
    non_meta_eligible_groups: list[str] = []
    for group, values in sorted(outcome.grouped_effects.items()):
        if not values:
            continue
        standard_error = _standard_error(values)
        standard_error_basis = "within_group_units"
        if standard_error <= 0.0 and between_group_se > 0.0:
            standard_error = between_group_se
            standard_error_basis = "between_independent_group_means_fallback"
        if standard_error <= 0.0:
            non_meta_eligible_groups.append(group)
            continue
        effect_records.append(
            {
                "axis_id": spec.axis_id,
                "estimand_id": spec.estimand_id,
                "effect_scale": spec.effect_scale,
                "reference_id": spec.reference_id,
                "unit": spec.unit,
                "higher_is_better": spec.higher_is_better,
                "run_id": target.name,
                "independent_group_id": group,
                "effect": group_means[group],
                "standard_error": standard_error,
                "standard_error_basis": standard_error_basis,
                "claim_scope": "first_scientific_gate_diagnostic_only",
                "evidence_status": spec.gate_kind,
                "source_artifact_path": "rows.jsonl",
                "source_artifact_sha256": rows_hash,
                "sample_size": len(values),
            }
        )
    if effect_records:
        atomic_jsonl_dump(effects_path, effect_records)
    else:
        effects_path.write_text("", encoding="utf-8")
    summary = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "axis_id": spec.axis_id,
        "profile": profile,
        "gate_kind": spec.gate_kind,
        "execution_status": outcome.status,
        "outcome_detail": outcome.outcome_detail,
        "row_count": len(outcome.rows),
        "effect_record_count": len(effect_records),
        "non_meta_eligible_groups": non_meta_eligible_groups,
        "estimand": {
            "id": spec.estimand_id,
            "scale": spec.effect_scale,
            "reference": spec.reference_id,
            "unit": spec.unit,
            "higher_is_better": spec.higher_is_better,
        },
        "notes": list(outcome.notes),
        "claim_scope": "first_scientific_gate_diagnostic_only",
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": [
            "play_strength",
            "production_readiness",
            "cross_axis_effect_pooling",
        ],
    }
    atomic_json_dump(summary_path, summary)
    _write_plot(plot_path, spec, effect_records)
    interpretation_path.write_text(
        "\n".join(
            [
                f"# {spec.axis_id} first-gate interpretation",
                "",
                "- Category: **DIAGNOSTIC**",
                f"- Quantity: `{spec.estimand_id}` in `{spec.unit}` against `{spec.reference_id}`.",
                f"- Provenance: `{rows_path}` and its SHA-256 `{rows_hash}`.",
                f"- Interpretation: the plot reports preregistered independent-group effects for the {spec.gate_kind} gate.",
                "- This plot does not show: play strength, Elo, production readiness, or cross-axis superiority.",
                "- Next plot: paired frozen-controller evaluation after this first gate is accepted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sources = sorted(
        {
            Path(__file__).resolve(),
            STUDY_REGISTRY.resolve(),
            *(path.resolve() for path in extra_sources),
        }
    )
    inputs = sorted({path.resolve() for path in outcome.inputs})
    artifacts = [rows_path, effects_path, summary_path, plot_path, interpretation_path]
    manifest = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "axis_id": spec.axis_id,
        "run_id": target.name,
        "profile": profile,
        "seed": seed,
        "created_at": utc_now(),
        "execution_status": outcome.status,
        "claim_scope": "first_scientific_gate_diagnostic_only",
        "source_hashes": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for path in sources
        ],
        "input_hashes": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
            for path in inputs
        ],
        "artifacts": [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
        "promotion": {"auto": False, "eligible": False},
    }
    atomic_json_dump(target / "run_manifest.json", manifest)
    return summary


def run_inprocess_study(
    axis_id: str,
    *,
    profile: str,
    seed: int,
    output_dir: Path,
    entrypoint: Path,
) -> dict[str, Any]:
    outcome = execute_inprocess(axis_id, profile, seed)
    return publish_outcome(
        axis_id=axis_id,
        profile=profile,
        seed=seed,
        output_dir=output_dir,
        outcome=outcome,
        extra_sources=(entrypoint,),
    )


def fingerprint(paths: Iterable[Path]) -> str:
    records = [
        (str(path.resolve().relative_to(REPO_ROOT)), file_sha256(path.resolve()))
        for path in sorted(set(paths))
    ]
    return sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
