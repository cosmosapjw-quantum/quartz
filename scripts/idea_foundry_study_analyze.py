#!/usr/bin/env python3
"""Validate, summarize, and meta-analyze a completed 26-axis study campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quartz.experiment_manifest import atomic_json_dump, file_sha256, utc_now  # noqa: E402
from quartz.idea_foundry.axis_workflow import atomic_jsonl_dump  # noqa: E402
from quartz.idea_foundry.meta_analysis import validate_effect_record  # noqa: E402
from quartz.idea_foundry.studies import StudyError, load_study_specs  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyError(f"JSON artifact must be an object: {path}")
    return payload


def _jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StudyError(f"cannot read JSONL artifact {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StudyError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise StudyError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    if not rows and not allow_empty:
        raise StudyError(f"JSONL artifact has no rows: {path}")
    return rows


def _verify_axis(
    axis_dir: Path, expected_axis: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(axis_dir / "run_manifest.json")
    summary = _json(axis_dir / "summary.json")
    if (
        manifest.get("axis_id") != expected_axis
        or summary.get("axis_id") != expected_axis
    ):
        raise StudyError(f"axis identity mismatch in {axis_dir}")
    if summary.get("execution_status") not in {"completed_no_promotion", "skipped"}:
        raise StudyError(f"axis is not scientifically terminal: {expected_axis}")
    if summary.get("promotion", {}).get("eligible") is not False:
        raise StudyError(f"axis may not be promotion eligible: {expected_axis}")
    for record in manifest.get("artifacts", []):
        artifact = axis_dir / record.get("path", "")
        if not artifact.is_file() or file_sha256(artifact) != record.get("sha256"):
            raise StudyError(f"artifact hash drift: {artifact}")
    rows = _jsonl(axis_dir / "effect_records.jsonl", allow_empty=True)
    normalized = [validate_effect_record(row) for row in rows]
    return summary, normalized


def _axis_meta(axis_id: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    effects = [float(row["effect"]) for row in records]
    standard_errors = [float(row["standard_error"]) for row in records]
    positive_variance = all(error > 0 for error in standard_errors)
    if positive_variance:
        weights = [1.0 / (error * error) for error in standard_errors]
        pooled = sum(weight * effect for weight, effect in zip(weights, effects)) / sum(
            weights
        )
        pooled_se = math.sqrt(1.0 / sum(weights))
        q_statistic = sum(
            weight * (effect - pooled) ** 2 for weight, effect in zip(weights, effects)
        )
        degrees = len(effects) - 1
        i_squared = (
            max(0.0, (q_statistic - degrees) / q_statistic) if q_statistic > 0 else 0.0
        )
        method = "fixed_effect_inverse_variance_within_axis"
    else:
        pooled = statistics.fmean(effects)
        pooled_se = (
            statistics.stdev(effects) / math.sqrt(len(effects))
            if len(effects) > 1
            else 0.0
        )
        q_statistic = sum((effect - pooled) ** 2 for effect in effects)
        i_squared = None
        method = "unweighted_independent_group_summary_zero_se_present"
    first = records[0]
    return {
        "schema_version": 1,
        "axis_id": axis_id,
        "estimand_id": first["estimand_id"],
        "effect_scale": first["effect_scale"],
        "reference_id": first["reference_id"],
        "unit": first["unit"],
        "higher_is_better": first["higher_is_better"],
        "independent_group_count": len(records),
        "pooled_effect": pooled,
        "pooled_standard_error": pooled_se,
        "ci95_low": pooled - 1.96 * pooled_se,
        "ci95_high": pooled + 1.96 * pooled_se,
        "q_statistic": q_statistic,
        "i_squared": i_squared,
        "method": method,
        "claim_scope": "within_axis_diagnostic_meta_analysis_only",
        "promotion_eligible": False,
    }


def _plot(path: Path, meta_rows: Sequence[Mapping[str, Any]]) -> None:
    with tempfile.TemporaryDirectory(prefix="quartz-study-meta-plot-") as mpl_dir:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels = [str(row["axis_id"]) for row in meta_rows]
            values = [float(row["pooled_effect"]) for row in meta_rows]
            errors = [1.96 * float(row["pooled_standard_error"]) for row in meta_rows]
            figure, axis = plt.subplots(figsize=(14, 6))
            x_values = list(range(len(labels)))
            axis.errorbar(
                x_values,
                values,
                yerr=errors,
                fmt="o",
                capsize=3,
                color="#457b9d",
            )
            axis.axhline(0.0, color="#264653", linewidth=1)
            axis.set_xticks(x_values)
            axis.set_xticklabels(labels, rotation=90)
            axis.set_ylabel("Within-axis standardized direction (native units differ)")
            axis.set_title(
                "IDEA FOUNDRY DIAGNOSTIC OVERVIEW\n"
                "points are not pooled across axes; units and estimands differ"
            )
            figure.tight_layout()
            temporary = path.with_suffix(".tmp.png")
            figure.savefig(temporary, dpi=160)
            plt.close(figure)
            os.replace(temporary, path)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous


def analyze_campaign(campaign_dir: Path, output_dir: Path | None) -> dict[str, Any]:
    campaign = campaign_dir.resolve()
    state = _json(campaign / "campaign_state.json")
    summary = _json(campaign / "campaign_summary.json")
    if state.get("status") != "completed_no_promotion" or summary.get(
        "status"
    ) != state.get("status"):
        raise StudyError("campaign is not complete")
    expected_axes = [spec.axis_id for spec in load_study_specs()]
    if [row.get("axis_id") for row in state.get("axes", [])] != expected_axes:
        raise StudyError("campaign axis order or coverage drift")
    target = (output_dir or campaign / "campaign_analysis").resolve()
    if target.exists() and (
        target.is_symlink() or not target.is_dir() or any(target.iterdir())
    ):
        raise StudyError(f"analysis output must be a new empty directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    axis_summaries: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    source_rows_dir = target / "source_rows"
    source_rows_dir.mkdir()
    for axis_id in expected_axes:
        axis_summary, axis_effects = _verify_axis(campaign / "axes" / axis_id, axis_id)
        axis_summaries.append(axis_summary)
        if axis_effects:
            copied_rows = source_rows_dir / f"{axis_id}.rows.jsonl"
            shutil.copyfile(campaign / "axes" / axis_id / "rows.jsonl", copied_rows)
            expected_hash = file_sha256(copied_rows)
            for record in axis_effects:
                if record["source_artifact_sha256"] != expected_hash:
                    raise StudyError(f"effect source hash drift: {axis_id}")
                record["source_artifact_path"] = f"source_rows/{axis_id}.rows.jsonl"
        effects.extend(axis_effects)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in effects:
        grouped[str(record["axis_id"])].append(record)
    effect_axis_ids = set(grouped)
    meta_rows = [
        _axis_meta(axis_id, grouped[axis_id])
        for axis_id in expected_axes
        if axis_id in effect_axis_ids
    ]
    atomic_jsonl_dump(target / "effect_records.jsonl", effects)
    atomic_jsonl_dump(target / "within_axis_meta_rows.jsonl", meta_rows)
    analysis = {
        "schema_version": 1,
        "analysis_kind": "idea_foundry_first_scientific_gate_campaign",
        "run_id": state["run_id"],
        "profile": state["profile"],
        "axis_count": len(axis_summaries),
        "effect_axis_count": len(effect_axis_ids),
        "effect_record_count": len(effects),
        "execution_status_counts": dict(
            sorted(Counter(row["execution_status"] for row in axis_summaries).items())
        ),
        "dormant_axes": [
            row["axis_id"]
            for row in axis_summaries
            if row["execution_status"] == "skipped"
        ],
        "meta_analysis_scope": "within_axis_only",
        "cross_axis_pooling_performed": False,
        "claim_scope": "analysis_only",
        "promotion": {"auto": False, "eligible": False},
        "created_at": utc_now(),
    }
    atomic_json_dump(target / "campaign_analysis.json", analysis)
    _plot(target / "diagnostic.png", meta_rows)
    (target / "interpretation.md").write_text(
        "\n".join(
            [
                "# Idea Foundry campaign analysis",
                "",
                "- Category: **DIAGNOSTIC**",
                "- Quantity: one point per axis, pooled only across that axis's declared independent groups.",
                "- Provenance: validated per-axis effect records and artifact SHA-256 values.",
                "- Interpretation: confidence intervals summarize repeat-group variability within each first gate.",
                "- This plot does not show: a common cross-axis effect, play strength, Elo, or production readiness.",
                "- Next plot: estimand-specific frozen-controller comparisons for axes selected after preregistered review.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts = [
        target / "campaign_analysis.json",
        target / "effect_records.jsonl",
        target / "within_axis_meta_rows.jsonl",
        target / "diagnostic.png",
        target / "interpretation.md",
        *sorted(source_rows_dir.glob("*.jsonl")),
    ]
    atomic_json_dump(
        target / "analysis_manifest.json",
        {
            "schema_version": 1,
            "run_id": state["run_id"],
            "inputs": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for path in (
                    campaign / "campaign_state.json",
                    campaign / "campaign_summary.json",
                )
            ],
            "artifacts": [
                {
                    "path": str(path.relative_to(target)),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in artifacts
            ],
            "cross_axis_pooling_performed": False,
            "promotion": {"auto": False, "eligible": False},
        },
    )
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_campaign(args.campaign_dir, args.output_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, StudyError) as exc:
        print(f"IDEA FOUNDRY ANALYSIS BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
