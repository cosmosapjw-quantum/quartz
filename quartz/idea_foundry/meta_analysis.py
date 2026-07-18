"""Campaign aggregation and conservative meta-analysis for Idea Foundry.

Contract-gate diagnostics are summarized but never converted into scientific
effect estimates.  Meta-analysis accepts only explicit, schema-checked effect
records sharing an exact estimand contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quartz.experiment_manifest import atomic_json_dump, file_sha256
from quartz.idea_foundry.axis_workflow import (
    REPO_ROOT,
    AxisWorkflowError,
    atomic_jsonl_dump,
    load_json_strict,
    load_jsonl_strict,
    load_workflow_specs,
    validate_axis_analysis,
)

ANALYSIS_SCHEMA_VERSION = 1
CAMPAIGN_ANALYSIS_FILENAMES = (
    "campaign_analysis.json",
    "campaign_axis_rows.jsonl",
    "diagnostic.png",
)
META_ANALYSIS_FILENAMES = (
    "meta_analysis.json",
    "meta_rows.jsonl",
    "diagnostic.png",
)


class MetaAnalysisError(RuntimeError):
    """Raised when analysis inputs are incomplete or statistically incompatible."""


def _ensure_new_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
            raise MetaAnalysisError(
                f"analysis output must be a new empty directory: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)


def _safe_child(root: Path, raw_path: str, *, label: str) -> Path:
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise MetaAnalysisError(f"{label} escapes campaign root: {raw_path}") from exc
    return path


def _write_campaign_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pass_rates = [
        100.0 * float(row["contract_pass_rate"])
        if row["contract_pass_rate"] is not None
        else 0.0
        for row in rows
    ]
    check_counts = [int(row["contract_check_count"]) for row in rows]
    with tempfile.TemporaryDirectory(prefix="quartz-campaign-plot-") as mpl_dir:
        old_mpl = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            x_values = list(range(1, len(rows) + 1))
            fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
            axes[0].bar(x_values, pass_rates, color="#2a9d8f")
            axes[0].axhline(100.0, color="#264653", linewidth=1, linestyle="--")
            axes[0].set_ylabel("Contract pass rate [%]")
            axes[0].set_ylim(0, 105)
            axes[1].bar(x_values, check_counts, color="#457b9d")
            axes[1].set_ylabel("Checks [count]")
            axes[1].set_xlabel("Axis in preregistered sequential order")
            axes[1].set_xticks(x_values)
            axes[1].set_xticklabels([str(row["axis_id"]) for row in rows], rotation=90)
            fig.suptitle(
                "IDEA FOUNDRY DIAGNOSTIC ONLY\n"
                "first-gate contract coverage; not efficacy or play strength"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            _atomic_figure(fig, path)
            plt.close(fig)
        finally:
            if old_mpl is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mpl


def _atomic_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(fd)
    try:
        fig.savefig(tmp_name, dpi=160)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _artifact_manifest(
    *,
    kind: str,
    inputs: Sequence[Path],
    artifacts: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "artifact_kind": kind,
        "claim_scope": "analysis_only",
        "inputs": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in sorted(set(inputs))
        ],
        "sources": [
            {
                "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": file_sha256(__file__),
            }
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
        "promotion": {"auto": False, "eligible": False},
    }


def analyze_campaign(
    campaign_dir: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    campaign_root = campaign_dir.resolve()
    state_path = campaign_root / "campaign_state.json"
    summary_path = campaign_root / "campaign_summary.json"
    state = load_json_strict(state_path)
    summary = load_json_strict(summary_path)
    if not isinstance(state, dict) or not isinstance(summary, dict):
        raise MetaAnalysisError("campaign state and summary must be JSON objects")
    if state.get("schema_version") != 1 or summary.get("schema_version") != 1:
        raise MetaAnalysisError("campaign state or summary schema mismatch")
    if state.get("run_id") != summary.get("run_id"):
        raise MetaAnalysisError("campaign state/summary run identity mismatch")
    if state.get("status") != "completed_no_promotion":
        raise MetaAnalysisError(f"campaign is not complete: {state.get('status')!r}")
    if summary.get("status") != state.get("status"):
        raise MetaAnalysisError("campaign state/summary status mismatch")
    specs = load_workflow_specs()
    axis_rows = state.get("axes")
    if not isinstance(axis_rows, list) or [row.get("axis_id") for row in axis_rows] != [
        spec.axis_id for spec in specs
    ]:
        raise MetaAnalysisError(
            "campaign must cover the registered 26-axis order exactly"
        )
    summary_axes = summary.get("axes")
    if not isinstance(summary_axes, list) or len(summary_axes) != len(axis_rows):
        raise MetaAnalysisError("campaign summary axis inventory is incomplete")
    expected_summary_rows = [
        (
            row.get("axis_id"),
            row.get("lane_id"),
            row.get("role"),
            row.get("status"),
            row.get("current_attempt"),
            len(row.get("attempts", [])),
        )
        for row in axis_rows
    ]
    observed_summary_rows = [
        (
            row.get("axis_id"),
            row.get("lane_id"),
            row.get("role"),
            row.get("status"),
            row.get("current_attempt"),
            row.get("attempt_count"),
        )
        for row in summary_axes
        if isinstance(row, dict)
    ]
    if observed_summary_rows != expected_summary_rows:
        raise MetaAnalysisError("campaign summary axis state mismatch")
    if summary.get("axis_count") != len(axis_rows):
        raise MetaAnalysisError("campaign summary axis count mismatch")
    expected_status_counts = dict(
        sorted(Counter(str(row.get("status")) for row in axis_rows).items())
    )
    if summary.get("status_counts") != expected_status_counts:
        raise MetaAnalysisError("campaign summary status counts mismatch")
    for label, payload in (("state", state), ("summary", summary)):
        promotion = payload.get("promotion")
        if not isinstance(promotion, dict) or promotion.get("eligible") is not False:
            raise MetaAnalysisError(f"campaign {label} may not be promotion eligible")

    rows: list[dict[str, Any]] = []
    effect_records: list[dict[str, Any]] = []
    analysis_inputs: list[Path] = [state_path, summary_path]
    for spec, axis_state in zip(specs, axis_rows, strict=True):
        if axis_state.get("status") != "completed_no_promotion":
            raise MetaAnalysisError(f"axis is not complete: {spec.axis_id}")
        raw_attempt = axis_state.get("current_attempt")
        if not isinstance(raw_attempt, str):
            raise MetaAnalysisError(f"axis attempt path is missing: {spec.axis_id}")
        attempt_dir = _safe_child(campaign_root, raw_attempt, label="axis attempt")
        analysis = validate_axis_analysis(
            spec.axis_id,
            input_dir=attempt_dir,
            analysis_dir=attempt_dir / "analysis",
        )
        aggregate = analysis["aggregate"]
        rows.append(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "run_id": state["run_id"],
                "order_index": spec.order_index,
                "axis_id": spec.axis_id,
                "axis_slug": spec.slug,
                "plane": spec.plane,
                "lane_id": spec.lane_id,
                "role": spec.role,
                "execution_status": axis_state["status"],
                "analysis_status": analysis["analysis_status"],
                "evidence_status": analysis["source_evidence_status"],
                "outcome_detail": analysis["outcome_detail"],
                "contract_check_count": aggregate["contract_check_count"],
                "contract_checks_passed": aggregate["contract_checks_passed"],
                "contract_checks_failed": aggregate["contract_checks_failed"],
                "contract_pass_rate": aggregate["contract_pass_rate"],
                "fixture_count": aggregate["fixture_count"],
                "promotion_eligible": False,
            }
        )
        for record in analysis.get("effect_records", []):
            effect_records.append(validate_effect_record(record))
        analysis_inputs.extend(
            [
                attempt_dir / "analysis" / "analysis_manifest.json",
                attempt_dir / "analysis" / "analysis.json",
                attempt_dir / "analysis" / "analysis_rows.jsonl",
            ]
        )

    target = (output_dir or campaign_root / "campaign_analysis").resolve()
    _ensure_new_directory(target)
    rows_path = target / "campaign_axis_rows.jsonl"
    analysis_path = target / "campaign_analysis.json"
    plot_path = target / "diagnostic.png"
    manifest_path = target / "analysis_manifest.json"
    atomic_jsonl_dump(rows_path, rows)
    status_counts = dict(
        sorted(Counter(row["execution_status"] for row in rows).items())
    )
    evidence_counts = dict(
        sorted(Counter(row["evidence_status"] for row in rows).items())
    )
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "idea_foundry_campaign_analysis",
        "run_id": state["run_id"],
        "axis_count": len(rows),
        "status": "ANALYZED_CONTRACT_ONLY",
        "status_counts": status_counts,
        "evidence_status_counts": evidence_counts,
        "contract_check_count": sum(row["contract_check_count"] for row in rows),
        "contract_checks_passed": sum(row["contract_checks_passed"] for row in rows),
        "contract_checks_failed": sum(row["contract_checks_failed"] for row in rows),
        "axes_with_failed_contract_checks": [
            row["axis_id"] for row in rows if row["contract_checks_failed"]
        ],
        "effect_records": effect_records,
        "meta_analysis_eligibility": (
            "EXPLICIT_EFFECT_RECORDS_AVAILABLE"
            if effect_records
            else "NO_COMPARABLE_EFFECT_ESTIMATES"
        ),
        "claim_scope": "synthetic_contract_analysis_only",
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": [
            "play_strength",
            "efficacy",
            "production_readiness",
            "cross_axis_effect_pooling_without_a_shared_estimand",
        ],
    }
    atomic_json_dump(analysis_path, payload)
    _write_campaign_plot(plot_path, rows)
    manifest = _artifact_manifest(
        kind="idea_foundry_campaign_analysis_manifest",
        inputs=analysis_inputs,
        artifacts=[analysis_path, rows_path, plot_path],
        output_dir=target,
    )
    atomic_json_dump(manifest_path, manifest)
    return payload


EFFECT_KEYS = (
    "axis_id",
    "estimand_id",
    "effect_scale",
    "reference_id",
    "unit",
    "higher_is_better",
)


def validate_effect_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MetaAnalysisError("effect record must be an object")
    required = set(EFFECT_KEYS) | {
        "run_id",
        "independent_group_id",
        "effect",
        "standard_error",
        "claim_scope",
        "evidence_status",
        "source_artifact_path",
        "source_artifact_sha256",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise MetaAnalysisError(f"effect record is missing fields: {missing}")
    record = dict(raw)
    axis_id = record["axis_id"]
    if axis_id not in {f"A{index:02d}" for index in range(1, 27)}:
        raise MetaAnalysisError(f"invalid effect axis: {axis_id!r}")
    for key in (
        "estimand_id",
        "effect_scale",
        "reference_id",
        "unit",
        "run_id",
        "independent_group_id",
        "claim_scope",
        "evidence_status",
        "source_artifact_path",
        "source_artifact_sha256",
    ):
        if not isinstance(record[key], str) or not record[key]:
            raise MetaAnalysisError(f"effect field must be a non-empty string: {key}")
    if not isinstance(record["higher_is_better"], bool):
        raise MetaAnalysisError("higher_is_better must be boolean")
    for key in ("effect", "standard_error"):
        value = record[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise MetaAnalysisError(f"effect field must be finite numeric: {key}")
        record[key] = float(value)
    if record["standard_error"] <= 0:
        raise MetaAnalysisError("standard_error must be positive")
    variance = record["standard_error"] ** 2
    if not math.isfinite(variance) or variance <= 0:
        raise MetaAnalysisError(
            "standard_error variance must be finite and representable"
        )
    digest = record["source_artifact_sha256"]
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest.lower()
    ):
        raise MetaAnalysisError("source_artifact_sha256 must be a SHA-256 hex digest")
    if record["claim_scope"] in {
        "synthetic_contract_gate_only",
        "synthetic_contract_analysis_only",
    }:
        raise MetaAnalysisError(
            "contract diagnostics are not admissible effect records"
        )
    return record


def _group_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[key] for key in EFFECT_KEYS)


def pool_effect_group(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validated = [validate_effect_record(record) for record in records]
    if not validated:
        raise MetaAnalysisError("cannot pool an empty effect group")
    key = _group_key(validated[0])
    if any(_group_key(record) != key for record in validated[1:]):
        raise MetaAnalysisError("effect group contains incompatible estimand contracts")
    independent_ids = [record["independent_group_id"] for record in validated]
    if len(independent_ids) != len(set(independent_ids)):
        raise MetaAnalysisError(
            "duplicate independent_group_id would double-count evidence"
        )
    base = {name: value for name, value in zip(EFFECT_KEYS, key, strict=True)}
    if len(validated) < 2:
        return {
            **base,
            "k": len(validated),
            "status": "INSUFFICIENT_INDEPENDENT_EFFECTS",
            "run_ids": sorted({record["run_id"] for record in validated}),
        }
    try:
        effects = [record["effect"] for record in validated]
        variances = [record["standard_error"] ** 2 for record in validated]
        fixed_weights = [1.0 / variance for variance in variances]
        weight_sum = math.fsum(fixed_weights)
        fixed_effect = math.fsum(
            (weight / weight_sum) * effect
            for weight, effect in zip(fixed_weights, effects)
        )
        fixed_se = math.sqrt(1.0 / weight_sum)
        q = math.fsum(
            weight * (effect - fixed_effect) ** 2
            for weight, effect in zip(fixed_weights, effects)
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise MetaAnalysisError("fixed-effect arithmetic overflowed") from exc
    if not all(
        math.isfinite(value) for value in (weight_sum, fixed_effect, fixed_se, q)
    ):
        raise MetaAnalysisError("fixed-effect arithmetic produced a non-finite value")
    df = len(effects) - 1
    normalized_weight_square_sum = math.fsum(
        (weight / weight_sum) ** 2 for weight in fixed_weights
    )
    c_value = weight_sum * (1.0 - normalized_weight_square_sum)
    tau_squared = max(0.0, (q - df) / c_value) if c_value > 0 else 0.0
    random_weights = [1.0 / (variance + tau_squared) for variance in variances]
    random_weight_sum = math.fsum(random_weights)
    random_effect = math.fsum(
        (weight / random_weight_sum) * effect
        for weight, effect in zip(random_weights, effects)
    )
    random_se = math.sqrt(1.0 / random_weight_sum)
    i_squared = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0
    fixed_ci = [fixed_effect - 1.96 * fixed_se, fixed_effect + 1.96 * fixed_se]
    random_ci = [random_effect - 1.96 * random_se, random_effect + 1.96 * random_se]
    numerical_outputs = [
        c_value,
        tau_squared,
        random_weight_sum,
        random_effect,
        random_se,
        i_squared,
        *fixed_ci,
        *random_ci,
    ]
    if not all(math.isfinite(value) for value in numerical_outputs):
        raise MetaAnalysisError("random-effects arithmetic produced a non-finite value")
    return {
        **base,
        "k": len(validated),
        "status": "POOLED_ANALYSIS_ONLY",
        "run_ids": sorted({record["run_id"] for record in validated}),
        "fixed_effect": fixed_effect,
        "fixed_standard_error": fixed_se,
        "fixed_ci95": fixed_ci,
        "random_effect": random_effect,
        "random_standard_error": random_se,
        "random_ci95": random_ci,
        "cochran_q": q,
        "heterogeneity_df": df,
        "tau_squared_dl": tau_squared,
        "i_squared_percent": i_squared,
    }


def pool_effect_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw in records:
        record = validate_effect_record(raw)
        groups[_group_key(record)].append(record)
    return [
        pool_effect_group(groups[key])
        for key in sorted(groups, key=lambda row: tuple(map(str, row)))
    ]


def _load_effect_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix == ".jsonl":
            payloads: Iterable[Mapping[str, Any]] = load_jsonl_strict(path)
        else:
            payload = load_json_strict(path)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("effect_records"), list
            ):
                raise MetaAnalysisError(
                    f"JSON input has no effect_records list: {path}"
                )
            payloads = payload["effect_records"]
        for payload in payloads:
            record = validate_effect_record(payload)
            raw_source_path = Path(record["source_artifact_path"])
            if raw_source_path.is_absolute():
                raise MetaAnalysisError("effect source artifact path must be relative")
            unresolved_source = path.resolve().parent
            for component in raw_source_path.parts:
                unresolved_source = unresolved_source / component
                if unresolved_source.is_symlink():
                    raise MetaAnalysisError(
                        f"effect source artifact may not traverse a symlink: {unresolved_source}"
                    )
            source_path = _safe_child(
                path.resolve().parent,
                record["source_artifact_path"],
                label="effect source artifact",
            )
            if not source_path.is_file() or source_path.is_symlink():
                raise MetaAnalysisError(
                    f"effect source artifact is missing: {source_path}"
                )
            if file_sha256(source_path) != record["source_artifact_sha256"]:
                raise MetaAnalysisError(
                    f"effect source artifact hash mismatch: {source_path}"
                )
            records.append(record)
    return records


def _write_meta_plot(path: Path, groups: Sequence[Mapping[str, Any]]) -> None:
    pooled = [row for row in groups if row["status"] == "POOLED_ANALYSIS_ONLY"]
    with tempfile.TemporaryDirectory(prefix="quartz-meta-plot-") as mpl_dir:
        old_mpl = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(
                figsize=(11, max(4.5, 0.7 * max(1, len(pooled)) + 2))
            )
            if pooled:
                y_values = list(range(len(pooled)))
                effects = [float(row["random_effect"]) for row in pooled]
                errors = [1.96 * float(row["random_standard_error"]) for row in pooled]
                labels = [f"{row['axis_id']}:{row['estimand_id']}" for row in pooled]
                axis.errorbar(
                    effects, y_values, xerr=errors, fmt="o", color="#457b9d", capsize=4
                )
                axis.axvline(0.0, color="#6c757d", linestyle="--", linewidth=1)
                axis.set_yticks(y_values)
                axis.set_yticklabels(labels)
                axis.set_xlabel("Random-effects estimate [declared effect unit]")
            else:
                axis.text(
                    0.5,
                    0.5,
                    "NO COMPARABLE EFFECTS\ncontract diagnostics were not pooled",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
            axis.set_title("IDEA FOUNDRY META-ANALYSIS — ANALYSIS ONLY")
            fig.tight_layout()
            _atomic_figure(fig, path)
            plt.close(fig)
        finally:
            if old_mpl is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mpl


def run_meta_analysis(input_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    if not input_paths:
        raise MetaAnalysisError(
            "at least one campaign analysis or effect-record input is required"
        )
    target = output_dir.resolve()
    records = _load_effect_records(input_paths)
    groups = pool_effect_records(records)
    _ensure_new_directory(target)
    pooled_count = sum(row["status"] == "POOLED_ANALYSIS_ONLY" for row in groups)
    status = "COMPLETED_ANALYSIS_ONLY" if pooled_count else "NO_COMPARABLE_EFFECTS"
    rows = groups or [
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "status": "NO_COMPARABLE_EFFECTS",
            "reason": "no explicit admissible effect records were supplied",
        }
    ]
    rows_path = target / "meta_rows.jsonl"
    analysis_path = target / "meta_analysis.json"
    plot_path = target / "diagnostic.png"
    manifest_path = target / "analysis_manifest.json"
    atomic_jsonl_dump(rows_path, rows)
    payload = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_kind": "idea_foundry_meta_analysis",
        "status": status,
        "effect_record_count": len(records),
        "estimand_group_count": len(groups),
        "pooled_group_count": pooled_count,
        "groups": groups,
        "method": {
            "fixed": "inverse_variance",
            "random": "dersimonian_laird",
            "confidence_interval": "normal_95_percent",
            "compatibility_key": list(EFFECT_KEYS),
            "minimum_independent_effects": 2,
        },
        "claim_scope": "analysis_only",
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": [
            "causality_without_randomized_or_valid_counterfactual_design",
            "cross_estimand_pooling",
            "double_counting_correlated_runs",
            "automatic_claim_promotion",
        ],
    }
    atomic_json_dump(analysis_path, payload)
    _write_meta_plot(plot_path, groups)
    manifest = _artifact_manifest(
        kind="idea_foundry_meta_analysis_manifest",
        inputs=[path.resolve() for path in input_paths],
        artifacts=[analysis_path, rows_path, plot_path],
        output_dir=target,
    )
    atomic_json_dump(manifest_path, manifest)
    return payload


def campaign_analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze one completed 26-axis campaign"
    )
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        payload = analyze_campaign(args.campaign_dir, args.output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (AxisWorkflowError, MetaAnalysisError) as exc:
        print(f"CAMPAIGN ANALYSIS BLOCKED: {exc}", file=os.sys.stderr)
        return 2


def meta_analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Meta-analyze compatible Idea Foundry effects"
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = run_meta_analysis(args.input, args.output_dir)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (AxisWorkflowError, MetaAnalysisError) as exc:
        print(f"META-ANALYSIS BLOCKED: {exc}", file=os.sys.stderr)
        return 2


__all__ = [
    "MetaAnalysisError",
    "analyze_campaign",
    "campaign_analysis_main",
    "meta_analysis_main",
    "pool_effect_group",
    "pool_effect_records",
    "run_meta_analysis",
    "validate_effect_record",
]
