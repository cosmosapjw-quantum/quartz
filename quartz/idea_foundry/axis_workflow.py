"""Per-axis execution and analysis workflows for the 26-axis idea foundry.

The workflows intentionally stop at the first registered contract gate.  They
make every axis independently runnable and analyzable without converting a
contract check into an efficacy or play-strength estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quartz.experiment_manifest import atomic_json_dump, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
AXIS_REGISTRY_PATH = REPO_ROOT / "configs" / "idea_foundry.axes.v1.json"
LAB_REGISTRY_PATH = REPO_ROOT / "configs" / "idea_lab.local.v2.json"
WORKFLOW_SCHEMA_VERSION = 1
ANALYSIS_CLAIM_SCOPE = "synthetic_contract_analysis_only"
ANALYSIS_FILENAMES = (
    "analysis.json",
    "analysis_rows.jsonl",
    "diagnostic.png",
)


class AxisWorkflowError(RuntimeError):
    """Raised when an axis workflow or artifact contract is invalid."""


@dataclass(frozen=True)
class AxisWorkflowSpec:
    axis_id: str
    slug: str
    plane: str
    registry_status: str
    description: str
    lane_id: str
    role: str
    evidence_status: str
    claim_scope: str
    order_index: int

    @property
    def script_name(self) -> str:
        return f"{self.axis_id.lower()}_{self.slug}.py"

    @property
    def script_path(self) -> Path:
        return REPO_ROOT / "scripts" / "idea_foundry" / self.script_name


def _reject_nonfinite_constant(value: str) -> None:
    raise AxisWorkflowError(f"non-finite JSON constant is forbidden: {value}")


def load_json_strict(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise AxisWorkflowError(f"required regular JSON file is missing: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AxisWorkflowError(f"invalid JSON artifact {path}: {exc}") from exc


def load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise AxisWorkflowError(f"required regular JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line, parse_constant=_reject_nonfinite_constant)
        except json.JSONDecodeError as exc:
            raise AxisWorkflowError(
                f"invalid JSONL row {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise AxisWorkflowError(
                f"JSONL row must be an object: {path}:{line_number}"
            )
        rows.append(payload)
    if not rows:
        raise AxisWorkflowError(f"JSONL artifact has no rows: {path}")
    return rows


def atomic_jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_workflow_specs() -> tuple[AxisWorkflowSpec, ...]:
    axes_payload = load_json_strict(AXIS_REGISTRY_PATH)
    lab_payload = load_json_strict(LAB_REGISTRY_PATH)
    axes = axes_payload.get("axes") if isinstance(axes_payload, dict) else None
    lanes = lab_payload.get("lanes") if isinstance(lab_payload, dict) else None
    suite = (
        lab_payload.get("suites", {}).get("first-gate-all")
        if isinstance(lab_payload, dict)
        else None
    )
    if (
        not isinstance(axes, list)
        or not isinstance(lanes, list)
        or not isinstance(suite, list)
    ):
        raise AxisWorkflowError(
            "axis and first-gate registries must contain list contracts"
        )
    if not all(isinstance(row, dict) for row in axes):
        raise AxisWorkflowError("every axis registry entry must be an object")
    if not all(isinstance(row, dict) for row in lanes):
        raise AxisWorkflowError("every lane registry entry must be an object")
    if not all(isinstance(row, str) and row for row in suite):
        raise AxisWorkflowError("every first-gate lane id must be a non-empty string")
    raw_axis_ids = [str(row.get("id")) for row in axes]
    raw_lane_ids = [str(row.get("id")) for row in lanes]
    if len(raw_axis_ids) != len(set(raw_axis_ids)):
        raise AxisWorkflowError("axis registry contains duplicate ids")
    if len(raw_lane_ids) != len(set(raw_lane_ids)):
        raise AxisWorkflowError("lane registry contains duplicate ids")
    axis_by_id = {str(row.get("id")): row for row in axes if isinstance(row, dict)}
    lane_by_id = {str(row.get("id")): row for row in lanes if isinstance(row, dict)}
    expected_axis_ids = {f"A{index:02d}" for index in range(1, 27)}
    if set(axis_by_id) != expected_axis_ids:
        raise AxisWorkflowError("axis registry must cover A01 through A26 exactly")
    if len(suite) != 26 or len(set(map(str, suite))) != 26:
        raise AxisWorkflowError("first-gate-all must contain exactly 26 unique lanes")

    specs: list[AxisWorkflowSpec] = []
    seen_axes: set[str] = set()
    for order_index, raw_lane_id in enumerate(suite):
        lane_id = str(raw_lane_id)
        lane = lane_by_id.get(lane_id)
        if lane is None:
            raise AxisWorkflowError(f"first-gate lane is not registered: {lane_id}")
        axis_id = str(lane.get("axis_id"))
        axis = axis_by_id.get(axis_id)
        if axis is None or axis_id in seen_axes:
            raise AxisWorkflowError(f"first-gate axis coverage is invalid at {lane_id}")
        if lane.get("execution_status") != "available":
            raise AxisWorkflowError(f"first-gate lane must be available: {lane_id}")
        seen_axes.add(axis_id)
        specs.append(
            AxisWorkflowSpec(
                axis_id=axis_id,
                slug=str(axis["slug"]),
                plane=str(axis["plane"]),
                registry_status=str(axis["status"]),
                description=str(axis["description"]),
                lane_id=lane_id,
                role=str(lane["role"]),
                evidence_status=str(lane["evidence_status"]),
                claim_scope=str(lane["claim_scope"]),
                order_index=order_index,
            )
        )
    if seen_axes != expected_axis_ids:
        raise AxisWorkflowError("first-gate-all does not cover A01 through A26 exactly")
    return tuple(specs)


def workflow_spec(axis_id: str) -> AxisWorkflowSpec:
    normalized = axis_id.upper()
    for spec in load_workflow_specs():
        if spec.axis_id == normalized:
            return spec
    raise AxisWorkflowError(f"unknown idea-foundry axis: {axis_id}")


def _ensure_new_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or path.is_symlink():
            raise AxisWorkflowError(f"output path must be a new directory: {path}")
        if any(path.iterdir()):
            raise AxisWorkflowError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _append_source_hash(manifest: dict[str, Any], path: Path) -> None:
    rows = list(manifest.get("source_hashes") or [])
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        display = str(resolved)
    record = {"path": display, "sha256": file_sha256(resolved)}
    rows = [row for row in rows if row.get("path") != display]
    rows.append(record)
    manifest["source_hashes"] = sorted(rows, key=lambda row: str(row["path"]))


def run_axis_gate(
    axis_id: str,
    *,
    role: str,
    output_dir: Path,
    seed: int,
    entrypoint_path: Path,
) -> int:
    spec = workflow_spec(axis_id)
    if role != spec.role:
        raise AxisWorkflowError(
            f"{spec.axis_id} first gate role is {spec.role!r}, not {role!r}"
        )
    _ensure_new_directory(output_dir)
    from scripts.idea_foundry_axis_gate import run as run_contract_gate

    returncode = run_contract_gate(spec.axis_id, role, output_dir, seed)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        manifest = load_json_strict(manifest_path)
        if not isinstance(manifest, dict):
            raise AxisWorkflowError("run manifest must be a JSON object")
        _append_source_hash(manifest, Path(__file__))
        _append_source_hash(manifest, entrypoint_path)
        manifest["axis_entrypoint"] = str(
            entrypoint_path.resolve().relative_to(REPO_ROOT)
        )
        manifest["workflow_schema_version"] = WORKFLOW_SCHEMA_VERSION
        atomic_json_dump(manifest_path, manifest)
    return returncode


def _validate_hash_record(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    raw_path = record.get("path")
    expected = record.get("sha256")
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise AxisWorkflowError(f"invalid {label} hash record")
    if not raw_path or Path(raw_path).is_absolute():
        raise AxisWorkflowError(
            f"{label} path must be non-empty and relative: {raw_path!r}"
        )
    path = root / raw_path
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AxisWorkflowError(
            f"{label} escapes its artifact root: {raw_path}"
        ) from exc
    if not path.is_file() or path.is_symlink():
        raise AxisWorkflowError(f"{label} is missing or not a regular file: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise AxisWorkflowError(
            f"{label} hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path


def _validate_run_artifacts(
    axis_id: str, input_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    spec = workflow_spec(axis_id)
    manifest = load_json_strict(input_dir / "run_manifest.json")
    summary = load_json_strict(input_dir / "summary.json")
    rows = load_jsonl_strict(input_dir / "rows.jsonl")
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise AxisWorkflowError("run manifest and summary must be JSON objects")
    if manifest.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise AxisWorkflowError("run manifest schema version mismatch")
    for label, payload in (("manifest", manifest), ("summary", summary)):
        if payload.get("axis_id") != spec.axis_id or payload.get("role") != spec.role:
            raise AxisWorkflowError(
                f"{label} axis/role identity mismatch for {spec.axis_id}"
            )
    if summary.get("execution_status") != "completed_no_promotion":
        raise AxisWorkflowError(
            f"axis run is not scientifically terminal: {summary.get('execution_status')!r}"
        )
    promotion = summary.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("eligible") is not False:
        raise AxisWorkflowError("contract-gate summary must prohibit promotion")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != 2
        or not all(isinstance(record, dict) for record in artifacts)
    ):
        raise AxisWorkflowError("run manifest artifact inventory is missing")
    artifact_names = [str(record.get("path")) for record in artifacts]
    if len(artifact_names) != len(set(artifact_names)):
        raise AxisWorkflowError("run manifest artifact inventory contains duplicates")
    artifact_paths = {
        str(record.get("path")): _validate_hash_record(
            input_dir, record, label="run artifact"
        )
        for record in artifacts
    }
    if set(artifact_paths) != {"rows.jsonl", "summary.json"}:
        raise AxisWorkflowError(
            "run artifact inventory must contain rows.jsonl and summary.json exactly"
        )
    for index, row in enumerate(rows):
        if row.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
            raise AxisWorkflowError(f"row {index} schema version mismatch")
        if row.get("axis_id") != spec.axis_id or row.get("role") != spec.role:
            raise AxisWorkflowError(f"row {index} axis/role identity mismatch")
        metric = row.get("metric")
        fixture_id = row.get("fixture_id")
        if not isinstance(metric, str) or not metric or not isinstance(fixture_id, str):
            raise AxisWorkflowError(
                f"row {index} has an invalid metric or fixture identity"
            )
        value = row.get("value")
        if isinstance(value, float) and not math.isfinite(value):
            raise AxisWorkflowError(f"row {index} contains a non-finite value")
        if not isinstance(value, (bool, int, float, str)):
            raise AxisWorkflowError(f"row {index} contains an unsupported value type")
    return manifest, summary, rows


def normalize_analysis_rows(
    axis_id: str,
    role: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for source_index, raw in enumerate(rows):
        metric = str(raw["metric"])
        value = raw["value"]
        if isinstance(value, bool):
            family = "contract_check"
        elif metric.endswith("_hash") and isinstance(value, str):
            family = "identity_hash"
        elif metric.endswith("_count") and isinstance(value, (int, float)):
            family = "count"
        elif isinstance(value, (int, float)):
            family = "scalar_diagnostic"
        else:
            family = "categorical_diagnostic"
        normalized.append(
            {
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "axis_id": axis_id,
                "role": role,
                "fixture_id": str(raw["fixture_id"]),
                "metric": metric,
                "metric_family": family,
                "value": value,
                "source_index": source_index,
            }
        )
    return sorted(
        normalized,
        key=lambda row: (
            str(row["fixture_id"]),
            str(row["metric"]),
            int(row["source_index"]),
        ),
    )


def summarize_analysis_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = [row for row in rows if row.get("metric_family") == "contract_check"]
    passed = sum(row.get("value") is True for row in checks)
    failed = sum(row.get("value") is False for row in checks)
    fixture_ids = sorted({str(row["fixture_id"]) for row in rows})
    counts = {
        str(row["metric"]): row["value"]
        for row in rows
        if row.get("metric_family") == "count"
    }
    return {
        "row_count": len(rows),
        "contract_check_count": len(checks),
        "contract_checks_passed": passed,
        "contract_checks_failed": failed,
        "contract_pass_rate": (passed / len(checks)) if checks else None,
        "fixture_ids": fixture_ids,
        "fixture_count": len(fixture_ids),
        "count_metrics": counts,
        "metric_family_counts": {
            family: sum(row.get("metric_family") == family for row in rows)
            for family in sorted({str(row.get("metric_family")) for row in rows})
        },
    }


def _write_diagnostic_plot(
    path: Path, spec: AxisWorkflowSpec, summary: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quartz-axis-plot-") as mpl_dir:
        old_mpl = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = mpl_dir
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            passed = int(summary["contract_checks_passed"])
            failed = int(summary["contract_checks_failed"])
            proposal_count = float(summary["count_metrics"].get("proposal_count", 0.0))
            fixture_count = int(summary["fixture_count"])
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
            axes[0].bar(
                ["passed", "failed"], [passed, failed], color=["#2a9d8f", "#e76f51"]
            )
            axes[0].set_ylabel("Contract checks [count]")
            axes[0].set_title("First-gate invariant checks")
            axes[1].bar(
                ["fixtures", "proposals"],
                [fixture_count, proposal_count],
                color=["#457b9d", "#f4a261"],
            )
            axes[1].set_ylabel("Recorded items [count]")
            axes[1].set_title("Diagnostic coverage")
            fig.suptitle(
                f"{spec.axis_id} DIAGNOSTIC ONLY — {spec.slug}\n"
                "contract execution, not efficacy or play strength"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.90))
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
            finally:
                plt.close(fig)
        finally:
            if old_mpl is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mpl


def analyze_axis(
    axis_id: str,
    *,
    input_dir: Path,
    analysis_dir: Path | None = None,
    entrypoint_path: Path | None = None,
) -> dict[str, Any]:
    spec = workflow_spec(axis_id)
    target = analysis_dir or input_dir / "analysis"
    if target.exists() and (not target.is_dir() or target.is_symlink()):
        raise AxisWorkflowError(
            f"analysis output must be a regular directory: {target}"
        )
    if target.exists() and any(target.iterdir()):
        return validate_axis_analysis(
            spec.axis_id, input_dir=input_dir, analysis_dir=target
        )
    _ensure_new_directory(target)
    _, run_summary, source_rows = _validate_run_artifacts(spec.axis_id, input_dir)
    normalized = normalize_analysis_rows(spec.axis_id, spec.role, source_rows)
    aggregate = summarize_analysis_rows(normalized)
    analysis_rows_path = target / "analysis_rows.jsonl"
    analysis_path = target / "analysis.json"
    diagnostic_path = target / "diagnostic.png"
    manifest_path = target / "analysis_manifest.json"
    atomic_jsonl_dump(analysis_rows_path, normalized)
    payload = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "analysis_id": f"{spec.axis_id.lower()}_first_gate_analysis_v1",
        "axis_id": spec.axis_id,
        "axis_slug": spec.slug,
        "plane": spec.plane,
        "lane_id": spec.lane_id,
        "role": spec.role,
        "execution_status": "completed_no_promotion",
        "analysis_status": "ANALYZED_CONTRACT_ONLY",
        "claim_scope": ANALYSIS_CLAIM_SCOPE,
        "source_execution_status": run_summary["execution_status"],
        "source_evidence_status": run_summary["evidence_status"],
        "outcome_detail": run_summary["outcome_detail"],
        "aggregate": aggregate,
        "effect_records": [],
        "meta_analysis_eligibility": "NO_COMPARABLE_EFFECT_ESTIMATES",
        "promotion": {
            "auto": False,
            "eligible": False,
            "reason": "contract analysis is not scientific efficacy evidence",
        },
        "prohibited_inferences": [
            "play_strength",
            "efficacy",
            "production_readiness",
            "cross_axis_effect_pooling_without_a_shared_estimand",
        ],
    }
    atomic_json_dump(analysis_path, payload)
    _write_diagnostic_plot(diagnostic_path, spec, aggregate)
    source_paths = [Path(__file__)]
    if entrypoint_path is not None:
        source_paths.append(entrypoint_path)
    manifest = {
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "artifact_kind": "idea_foundry_axis_analysis_manifest",
        "axis_id": spec.axis_id,
        "role": spec.role,
        "claim_scope": ANALYSIS_CLAIM_SCOPE,
        "inputs": [
            {
                "path": str(path.relative_to(input_dir)),
                "sha256": file_sha256(path),
            }
            for path in (
                input_dir / "run_manifest.json",
                input_dir / "rows.jsonl",
                input_dir / "summary.json",
            )
        ],
        "sources": [
            {
                "path": str(path.resolve().relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for path in sorted(set(source_paths))
        ],
        "artifacts": [
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in (analysis_path, analysis_rows_path, diagnostic_path)
        ],
        "promotion": {"auto": False, "eligible": False},
    }
    atomic_json_dump(manifest_path, manifest)
    return validate_axis_analysis(
        spec.axis_id, input_dir=input_dir, analysis_dir=target
    )


def validate_axis_analysis(
    axis_id: str,
    *,
    input_dir: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    spec = workflow_spec(axis_id)
    _validate_run_artifacts(spec.axis_id, input_dir)
    manifest = load_json_strict(analysis_dir / "analysis_manifest.json")
    payload = load_json_strict(analysis_dir / "analysis.json")
    rows = load_jsonl_strict(analysis_dir / "analysis_rows.jsonl")
    if not isinstance(manifest, dict) or not isinstance(payload, dict):
        raise AxisWorkflowError("analysis manifest and payload must be objects")
    if (
        manifest.get("schema_version") != WORKFLOW_SCHEMA_VERSION
        or payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION
    ):
        raise AxisWorkflowError("analysis schema version mismatch")
    if (
        manifest.get("axis_id") != spec.axis_id
        or payload.get("axis_id") != spec.axis_id
    ):
        raise AxisWorkflowError("analysis axis identity mismatch")
    if payload.get("analysis_status") != "ANALYZED_CONTRACT_ONLY":
        raise AxisWorkflowError("analysis status is outside the first-gate contract")
    promotion = payload.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("eligible") is not False:
        raise AxisWorkflowError("axis analysis may not be promotion eligible")
    if payload.get("effect_records") != []:
        raise AxisWorkflowError("contract analysis may not manufacture effect records")
    input_records = manifest.get("inputs")
    if (
        not isinstance(input_records, list)
        or len(input_records) != 3
        or not all(isinstance(record, dict) for record in input_records)
    ):
        raise AxisWorkflowError("analysis input inventory is incomplete")
    input_names = [str(record.get("path")) for record in input_records]
    if set(input_names) != {"run_manifest.json", "rows.jsonl", "summary.json"}:
        raise AxisWorkflowError("analysis input inventory has unexpected paths")
    if len(input_names) != len(set(input_names)):
        raise AxisWorkflowError("analysis input inventory contains duplicates")
    for record in input_records:
        _validate_hash_record(input_dir, record, label="analysis input")
    artifact_records = manifest.get("artifacts")
    if (
        not isinstance(artifact_records, list)
        or len(artifact_records) != len(ANALYSIS_FILENAMES)
        or not all(isinstance(record, dict) for record in artifact_records)
    ):
        raise AxisWorkflowError("analysis artifact inventory is missing")
    artifact_names = [str(record.get("path")) for record in artifact_records]
    if len(artifact_names) != len(set(artifact_names)):
        raise AxisWorkflowError("analysis artifact inventory contains duplicates")
    for record in artifact_records:
        _validate_hash_record(analysis_dir, record, label="analysis artifact")
    if set(artifact_names) != set(ANALYSIS_FILENAMES):
        raise AxisWorkflowError("analysis artifact inventory is incomplete")
    if any(row.get("axis_id") != spec.axis_id for row in rows):
        raise AxisWorkflowError("normalized analysis row axis mismatch")
    if any(row.get("schema_version") != WORKFLOW_SCHEMA_VERSION for row in rows):
        raise AxisWorkflowError("normalized analysis row schema mismatch")
    recomputed = summarize_analysis_rows(rows)
    if payload.get("aggregate") != recomputed:
        raise AxisWorkflowError("analysis aggregate does not match normalized rows")
    return payload


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False))


def axis_main(
    axis_id: str,
    entrypoint_path: str | Path,
    argv: Sequence[str] | None = None,
) -> int:
    spec = workflow_spec(axis_id)
    entrypoint = Path(entrypoint_path).resolve()
    parser = argparse.ArgumentParser(
        description=f"Run or analyze {spec.axis_id} {spec.slug} first-gate experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser(
        "describe", help="Print the registered axis contract"
    )
    describe.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run the first registered gate")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, default=20260718)
    run_parser.add_argument("--role", default=spec.role)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze an existing axis run"
    )
    analyze_parser.add_argument("--input-dir", type=Path, required=True)
    analyze_parser.add_argument("--analysis-dir", type=Path, default=None)

    both_parser = subparsers.add_parser(
        "run-and-analyze",
        help="Run the first gate and immediately analyze its artifacts",
    )
    both_parser.add_argument("--output-dir", type=Path, required=True)
    both_parser.add_argument("--seed", type=int, default=20260718)
    both_parser.add_argument("--role", default=spec.role)

    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            payload = {
                "schema_version": WORKFLOW_SCHEMA_VERSION,
                "axis_id": spec.axis_id,
                "slug": spec.slug,
                "plane": spec.plane,
                "lane_id": spec.lane_id,
                "role": spec.role,
                "registry_status": spec.registry_status,
                "evidence_status": spec.evidence_status,
                "claim_scope": spec.claim_scope,
                "description": spec.description,
                "order_index": spec.order_index,
                "script": str(entrypoint.relative_to(REPO_ROOT)),
            }
            _emit(payload)
            return 0
        if args.command == "run":
            return run_axis_gate(
                spec.axis_id,
                role=args.role,
                output_dir=args.output_dir,
                seed=args.seed,
                entrypoint_path=entrypoint,
            )
        if args.command == "analyze":
            payload = analyze_axis(
                spec.axis_id,
                input_dir=args.input_dir,
                analysis_dir=args.analysis_dir,
                entrypoint_path=entrypoint,
            )
            _emit(payload)
            return 0
        returncode = run_axis_gate(
            spec.axis_id,
            role=args.role,
            output_dir=args.output_dir,
            seed=args.seed,
            entrypoint_path=entrypoint,
        )
        if returncode != 0:
            return returncode
        payload = analyze_axis(
            spec.axis_id,
            input_dir=args.output_dir,
            entrypoint_path=entrypoint,
        )
        _emit(payload)
        return 0
    except AxisWorkflowError as exc:
        print(f"{spec.axis_id} WORKFLOW BLOCKED: {exc}", file=sys.stderr)
        return 2


__all__ = [
    "ANALYSIS_CLAIM_SCOPE",
    "ANALYSIS_FILENAMES",
    "AxisWorkflowError",
    "AxisWorkflowSpec",
    "analyze_axis",
    "axis_main",
    "atomic_jsonl_dump",
    "load_json_strict",
    "load_jsonl_strict",
    "load_workflow_specs",
    "normalize_analysis_rows",
    "run_axis_gate",
    "summarize_analysis_rows",
    "validate_axis_analysis",
    "workflow_spec",
]
