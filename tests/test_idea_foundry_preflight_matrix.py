from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

import quartz.idea_foundry.axis_workflow as workflow
from quartz.experiment_manifest import file_sha256
from quartz.idea_foundry.axis_workflow import (
    REPO_ROOT,
    AxisWorkflowError,
    analyze_axis,
    load_workflow_specs,
    validate_axis_analysis,
)
from quartz.idea_foundry.meta_analysis import (
    EFFECT_KEYS,
    MetaAnalysisError,
    pool_effect_group,
    pool_effect_records,
    run_meta_analysis,
    validate_effect_record,
)
from quartz.idea_foundry.sequential import (
    SequentialCampaignError,
    resolve_run_root,
)


def _load_preflight_script():
    path = REPO_ROOT / "scripts" / "idea_foundry_preflight.py"
    spec = importlib.util.spec_from_file_location("idea_foundry_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dump(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _axis_run(
    tmp_path: Path, *, analyze: bool
) -> tuple[workflow.AxisWorkflowSpec, Path]:
    spec = load_workflow_specs()[0]
    output = tmp_path / "axis"
    command = "run-and-analyze" if analyze else "run"
    proc = subprocess.run(
        [
            sys.executable,
            str(spec.script_path),
            command,
            "--output-dir",
            str(output),
            "--seed",
            "101",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return spec, output


def _effect(**updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "axis_id": "A15",
        "estimand_id": "paired_win_rate_delta",
        "effect_scale": "risk_difference",
        "reference_id": "fixed_controller_v1",
        "unit": "fraction",
        "higher_is_better": True,
        "run_id": "run-1",
        "independent_group_id": "group-1",
        "effect": 0.2,
        "standard_error": 0.1,
        "claim_scope": "paired_ablation_analysis_only",
        "evidence_status": "preregistered_ablation",
        "source_artifact_path": "source.json",
        "source_artifact_sha256": "0" * 64,
    }
    record.update(updates)
    return record


def test_release_preflight_inventory_and_import_receipt_are_complete(
    tmp_path: Path,
) -> None:
    preflight = _load_preflight_script()
    receipt = preflight.verify_import_receipt()
    assert receipt["status"] == "VALIDATED_IMPORT_PROVENANCE_ONLY"
    assert receipt["payload_count"] == 21
    quick = preflight.build_steps(
        python=Path(sys.executable), run_root=tmp_path / "quick", mode="quick"
    )
    release = preflight.build_steps(
        python=Path(sys.executable), run_root=tmp_path / "release", mode="release"
    )
    quick_names = [step.name for step in quick]
    release_names = [step.name for step in release]
    assert len(release_names) == len(set(release_names))
    assert release_names[: len(quick_names)] == quick_names
    assert {
        "cargo-default",
        "cargo-idea-foundry",
        "full-python-regression",
        "eager-real-loop",
        "phase15-ci-smoke",
        "first-gate-sequential-run",
        "first-gate-sequential-resume",
        "a15-a19-readiness-run",
        "a15-a19-readiness-resume",
        "a18-study-input-inspect",
    } <= set(release_names)
    blocked = {
        step.name: step.expected_returncodes
        for step in release
        if "remains-blocked" in step.name
    }
    assert blocked == {
        "live-promotion-remains-blocked": (2,),
        "accelerator-promotion-remains-blocked": (2,),
    }


@pytest.mark.parametrize("duplicate_kind", ["axis", "lane"])
def test_direct_workflow_registry_rejects_every_duplicate_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate_kind: str
) -> None:
    axes = json.loads(workflow.AXIS_REGISTRY_PATH.read_text(encoding="utf-8"))
    lab = json.loads(workflow.LAB_REGISTRY_PATH.read_text(encoding="utf-8"))
    if duplicate_kind == "axis":
        axes["axes"].append(copy.deepcopy(axes["axes"][0]))
    else:
        lab["lanes"].append(copy.deepcopy(lab["lanes"][0]))
    axis_path = tmp_path / "axes.json"
    lab_path = tmp_path / "lab.json"
    _dump(axis_path, axes)
    _dump(lab_path, lab)
    monkeypatch.setattr(workflow, "AXIS_REGISTRY_PATH", axis_path)
    monkeypatch.setattr(workflow, "LAB_REGISTRY_PATH", lab_path)
    with pytest.raises(AxisWorkflowError, match="duplicate ids"):
        load_workflow_specs()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload, root: payload.update(schema_version=2),
            "schema version mismatch",
        ),
        (
            lambda payload, root: payload["artifacts"].append(
                copy.deepcopy(payload["artifacts"][0])
            ),
            "artifact inventory is missing",
        ),
        (
            lambda payload, root: payload["artifacts"][0].update(
                path=str((root / "rows.jsonl").resolve())
            ),
            "must be non-empty and relative",
        ),
    ],
)
def test_run_manifest_inventory_variants_fail_closed(
    tmp_path: Path, mutator: object, message: str
) -> None:
    spec, output = _axis_run(tmp_path, analyze=False)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutator(manifest, output)  # type: ignore[operator]
    _dump(manifest_path, manifest)
    with pytest.raises(AxisWorkflowError, match=message):
        analyze_axis(spec.axis_id, input_dir=output)


@pytest.mark.parametrize("inventory", ["inputs", "artifacts"])
def test_analysis_manifest_rejects_duplicate_inventory(
    tmp_path: Path, inventory: str
) -> None:
    spec, output = _axis_run(tmp_path, analyze=True)
    manifest_path = output / "analysis" / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[inventory].append(copy.deepcopy(manifest[inventory][0]))
    _dump(manifest_path, manifest)
    with pytest.raises(AxisWorkflowError, match="inventory"):
        validate_axis_analysis(
            spec.axis_id,
            input_dir=output,
            analysis_dir=output / "analysis",
        )


def test_analysis_rejects_non_directory_output_and_row_schema_drift(
    tmp_path: Path,
) -> None:
    spec, output = _axis_run(tmp_path, analyze=False)
    analysis_target = tmp_path / "analysis-file"
    analysis_target.write_text("not a directory\n", encoding="utf-8")
    with pytest.raises(AxisWorkflowError, match="regular directory"):
        analyze_axis(spec.axis_id, input_dir=output, analysis_dir=analysis_target)

    analyze_axis(spec.axis_id, input_dir=output)
    rows_path = output / "analysis" / "analysis_rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0]["schema_version"] = 2
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = output / "analysis" / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(row for row in manifest["artifacts"] if row["path"] == rows_path.name)[
        "sha256"
    ] = file_sha256(rows_path)
    _dump(manifest_path, manifest)
    with pytest.raises(AxisWorkflowError, match="row schema mismatch"):
        validate_axis_analysis(
            spec.axis_id,
            input_dir=output,
            analysis_dir=output / "analysis",
        )


def test_run_id_domain_accepts_only_confined_ascii_tokens() -> None:
    root = REPO_ROOT / "results" / "idea_foundry-preflight-run-id"
    valid = ["a", "A01", "run_1", "run-1.2", "z" * 96]
    invalid = [
        "",
        ".",
        "..",
        "a..b",
        " leading",
        "trailing ",
        "slash/name",
        "back\\slash",
        "한글",
        "z" * 97,
    ]
    for run_id in valid:
        assert resolve_run_root(root, run_id).parent == root.resolve()
    for run_id in invalid:
        with pytest.raises(SequentialCampaignError, match="unsafe run id"):
            resolve_run_root(root, run_id)


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("axis_id", "A16"),
        ("estimand_id", "latency_delta"),
        ("effect_scale", "log_ratio"),
        ("reference_id", "fixed_controller_v2"),
        ("unit", "milliseconds"),
        ("higher_is_better", False),
    ],
)
def test_every_meta_compatibility_key_blocks_cross_contract_pooling(
    field: str, different: object
) -> None:
    first = _effect()
    second = _effect(run_id="run-2", independent_group_id="group-2")
    second[field] = different
    groups = pool_effect_records([first, second])
    assert len(groups) == 2
    assert {group["status"] for group in groups} == {"INSUFFICIENT_INDEPENDENT_EFFECTS"}


def test_effect_record_required_fields_are_exhaustively_enforced() -> None:
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
    for field in sorted(required):
        record = _effect()
        del record[field]
        with pytest.raises(MetaAnalysisError, match="missing fields"):
            validate_effect_record(record)


@pytest.mark.parametrize(
    ("field", "malformed", "message"),
    [
        ("effect", True, "finite numeric"),
        ("effect", "0.1", "finite numeric"),
        ("effect", math.nan, "finite numeric"),
        ("effect", math.inf, "finite numeric"),
        ("standard_error", 0.0, "positive"),
        ("standard_error", -0.1, "positive"),
        ("standard_error", 1e-200, "representable"),
        ("higher_is_better", 1, "must be boolean"),
        ("estimand_id", "", "non-empty string"),
        ("source_artifact_sha256", "xyz", "SHA-256"),
        ("claim_scope", "synthetic_contract_gate_only", "not admissible"),
    ],
)
def test_effect_numeric_and_claim_boundaries_fail_closed(
    field: str, malformed: object, message: str
) -> None:
    with pytest.raises(MetaAnalysisError, match=message):
        validate_effect_record(_effect(**{field: malformed}))


def test_all_effect_record_permutations_produce_identical_groups() -> None:
    records = [
        _effect(run_id="r1", independent_group_id="g1", effect=0.1),
        _effect(run_id="r2", independent_group_id="g2", effect=0.2),
        _effect(
            axis_id="A16",
            run_id="r3",
            independent_group_id="g3",
            effect=-0.1,
        ),
        _effect(
            axis_id="A16",
            run_id="r4",
            independent_group_id="g4",
            effect=0.3,
        ),
    ]
    expected = pool_effect_records(records)
    for permutation in itertools.permutations(records):
        assert pool_effect_records(permutation) == expected


def test_extreme_finite_effect_arithmetic_fails_as_domain_error() -> None:
    records = [
        _effect(run_id="r1", independent_group_id="g1", effect=1e308),
        _effect(run_id="r2", independent_group_id="g2", effect=-1e308),
    ]
    with pytest.raises(MetaAnalysisError, match="arithmetic"):
        pool_effect_group(records)


def test_effect_source_path_escape_absolute_missing_symlink_and_hash_matrix(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    source = input_root / "source.json"
    source.write_text('{"paired": true}\n', encoding="utf-8")
    valid_hash = file_sha256(source)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    symlink = input_root / "source-link.json"
    symlink.symlink_to(source)
    cases = [
        (str(source.resolve()), valid_hash, "must be relative"),
        (f"../{outside.name}", file_sha256(outside), "escapes"),
        ("missing.json", "0" * 64, "missing"),
        (symlink.name, valid_hash, "symlink"),
        (source.name, "0" * 64, "hash mismatch"),
    ]
    for index, (source_path, source_hash, message) in enumerate(cases):
        input_path = input_root / f"effects-{index}.json"
        _dump(
            input_path,
            {
                "effect_records": [
                    _effect(
                        source_artifact_path=source_path,
                        source_artifact_sha256=source_hash,
                    )
                ]
            },
        )
        with pytest.raises(MetaAnalysisError, match=message):
            run_meta_analysis([input_path], input_root / f"meta-{index}")


def test_single_verified_effect_remains_unpooled(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    input_path = tmp_path / "effects.json"
    _dump(
        input_path,
        {"effect_records": [_effect(source_artifact_sha256=file_sha256(source))]},
    )
    payload = run_meta_analysis([input_path], tmp_path / "meta")
    assert payload["status"] == "NO_COMPARABLE_EFFECTS"
    assert payload["pooled_group_count"] == 0
    assert payload["groups"][0]["status"] == "INSUFFICIENT_INDEPENDENT_EFFECTS"
