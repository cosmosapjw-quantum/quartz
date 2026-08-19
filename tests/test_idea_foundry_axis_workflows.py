from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from quartz.experiment_manifest import file_sha256
from quartz.idea_foundry.axis_workflow import (
    REPO_ROOT,
    AxisWorkflowError,
    analyze_axis,
    load_workflow_specs,
    normalize_analysis_rows,
    run_axis_gate,
    summarize_analysis_rows,
    validate_axis_analysis,
    workflow_spec,
)
from quartz.idea_foundry.meta_analysis import (
    MetaAnalysisError,
    analyze_campaign,
    run_meta_analysis,
)
from quartz.idea_foundry.sequential import (
    SequentialCampaignError,
    _new_state,
    _validate_state,
    _validated_attempt,
    run_campaign,
)
from quartz.idea_foundry.status_schema import (
    ExecutionStatus,
    StatusSchemaError,
    first_gate_status,
    is_poolable,
    transition_status,
    validate_status_v2,
)


# fmt: off
def _rewrite_run_status(output: Path, status: object) -> None:
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["status"] = status
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    next(row for row in manifest["artifacts"] if row["path"] == "summary.json")["sha256"] = file_sha256(summary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _run_gate(spec: object, output: Path, seed: int) -> None:
    assert run_axis_gate(spec.axis_id, role=spec.role, output_dir=output, seed=seed, entrypoint_path=spec.script_path) == 0


def test_axis_entrypoints_cover_registered_first_gate_order_exactly() -> None:
    specs = load_workflow_specs()
    assert len(specs) == 26
    assert {spec.axis_id for spec in specs} == {
        f"A{index:02d}" for index in range(1, 27)
    }
    assert len({spec.script_name for spec in specs}) == 26
    for spec in specs:
        assert spec.script_path.is_file()
        source = spec.script_path.read_text(encoding="utf-8")
        assert f'axis_main("{spec.axis_id}", __file__)' in source


def test_contract_summary_is_invariant_to_input_row_permutations() -> None:
    source_rows = [
        {"fixture_id": "f2", "metric": "guard", "value": True},
        {"fixture_id": "f1", "metric": "proposal_count", "value": 3},
        {"fixture_id": "f1", "metric": "category", "value": "stable"},
        {"fixture_id": "f2", "metric": "cost", "value": 1.25},
    ]
    expected = summarize_analysis_rows(
        normalize_analysis_rows("A01", "trace", source_rows)
    )
    for seed in range(25):
        shuffled = list(source_rows)
        random.Random(seed).shuffle(shuffled)
        actual = summarize_analysis_rows(
            normalize_analysis_rows("A01", "trace", shuffled)
        )
        assert actual == expected


def test_axis_script_runs_analyzes_and_detects_hash_tampering(tmp_path: Path) -> None:
    spec = load_workflow_specs()[0]
    output_dir = tmp_path / "axis"
    proc = subprocess.run(
        [
            sys.executable,
            str(spec.script_path),
            "run-and-analyze",
            "--output-dir",
            str(output_dir),
            "--seed",
            "17",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = validate_axis_analysis(
        spec.axis_id,
        input_dir=output_dir,
        analysis_dir=output_dir / "analysis",
    )
    assert payload["status"] == first_gate_status("A01")
    assert payload["effect_records"] == []
    assert is_poolable(payload["status"]) is False

    rows_path = output_dir / "rows.jsonl"
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(AxisWorkflowError, match="hash mismatch"):
        validate_axis_analysis(
            spec.axis_id,
            input_dir=output_dir,
            analysis_dir=output_dir / "analysis",
        )


def test_axis_analysis_reuses_only_a_valid_existing_analysis(tmp_path: Path) -> None:
    spec = load_workflow_specs()[1]
    output_dir = tmp_path / "axis"
    subprocess.run(
        [
            sys.executable,
            str(spec.script_path),
            "run",
            "--output-dir",
            str(output_dir),
            "--seed",
            "23",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    first = analyze_axis(spec.axis_id, input_dir=output_dir)
    first_hash = file_sha256(output_dir / "analysis" / "analysis.json")
    second = analyze_axis(spec.axis_id, input_dir=output_dir)
    assert second == first
    assert file_sha256(output_dir / "analysis" / "analysis.json") == first_hash


def test_v2_status_round_trips_through_axis_analysis_and_resume(tmp_path: Path) -> None:
    spec = load_workflow_specs()[0]
    output_dir = tmp_path / "axis"
    _run_gate(spec, output_dir, 31)
    status = first_gate_status("A01")
    analysis = analyze_axis(spec.axis_id, input_dir=output_dir)
    assert analysis["status"] == status
    assert _validated_attempt(spec.axis_id, tmp_path, {"status": status, "current_attempt": "axis"})
    assert _validated_attempt(spec.axis_id, tmp_path, {"status": first_gate_status("A10"), "current_attempt": "axis"}) is False
    _rewrite_run_status(output_dir, "completed_no_promotion")
    with pytest.raises(AxisWorkflowError, match="legacy status schema v1"):
        analyze_axis(spec.axis_id, input_dir=output_dir, analysis_dir=tmp_path / "legacy")
    assert not (tmp_path / "legacy").exists()


@pytest.mark.parametrize("axis_id, wrong_axis", [("A01", "A10"), ("A10", "A01")])
def test_axis_context_rejects_the_other_terminal_status(tmp_path: Path, axis_id: str, wrong_axis: str) -> None:
    spec = workflow_spec(axis_id)
    output = tmp_path / spec.axis_id
    _run_gate(spec, output, 37)
    _rewrite_run_status(output, first_gate_status(wrong_axis))
    target = tmp_path / "analysis"
    with pytest.raises(AxisWorkflowError, match="axis status"):
        analyze_axis(spec.axis_id, input_dir=output, analysis_dir=target)
    assert not target.exists()


@pytest.mark.parametrize("bad_axes", [None, 7, [None], ["bad"], [], [{}] * 27, "missing-status", "attempts"])
def test_malformed_axis_containers_raise_controlled_error(bad_axes: object) -> None:
    entrypoint = REPO_ROOT / "scripts" / "idea_foundry_run_all.py"
    state = _new_state("bad-state", 41, entrypoint)
    if bad_axes == "missing-status":
        bad_axes = state["axes"]
        bad_axes[0].pop("status")
    if bad_axes == "attempts":
        bad_axes = state["axes"]
        bad_axes[0]["attempts"] = None
    state["axes"] = bad_axes
    with pytest.raises(SequentialCampaignError, match="campaign ax|status schema-v2"):
        _validate_state(state, "bad-state", 41, entrypoint)


@pytest.mark.parametrize("campaign, axis", [(ExecutionStatus.RUNNING, ExecutionStatus.PLANNED), (ExecutionStatus.FAILED, ExecutionStatus.RUNNING), (ExecutionStatus.RUNNING, ExecutionStatus.FAILED)])
def test_partial_v2_state_is_admissible(campaign: ExecutionStatus, axis: ExecutionStatus) -> None:
    entrypoint = REPO_ROOT / "scripts" / "idea_foundry_run_all.py"
    state = _new_state("partial", 43, entrypoint)
    state["status"], state["axes"][0]["status"] = transition_status(campaign), transition_status(axis)
    assert _validate_state(state, "partial", 43, entrypoint) is state
    state["status"] = first_gate_status("campaign")
    with pytest.raises(SequentialCampaignError):
        _validate_state(state, "partial", 43, entrypoint)
# fmt: on


def test_status_v2_rejects_invalid_lattice_and_preserves_special_states() -> None:
    invalid = first_gate_status("A01")
    invalid["contract"] = "failed"
    with pytest.raises(StatusSchemaError, match="evidence lattice"):
        validate_status_v2(invalid)
    assert first_gate_status("A10")["execution"] == "skipped"
    assert transition_status(ExecutionStatus.FAILED)["effect"] == "invalidated"


# fmt: off
def test_full_sequential_campaign_and_resume_skip_validated_axes() -> None:
    results_root = REPO_ROOT / "results"
    results_root.mkdir(exist_ok=True)
    entrypoint = REPO_ROOT / "scripts" / "idea_foundry_run_all.py"
    with tempfile.TemporaryDirectory(
        prefix="idea-foundry-test-", dir=results_root
    ) as raw_root:
        campaign_root = Path(raw_root)
        first = run_campaign(
            campaign_root=campaign_root,
            run_id="sequential-smoke",
            seed=29,
            timeout_seconds=30,
            resume=False,
            entrypoint=entrypoint,
        )
        assert first["status"] == first_gate_status("campaign")
        assert first["axis_count"] == 26
        assert first["status_counts"] == {"skipped": 1, "success": 25}

        state_path = campaign_root / "sequential-smoke" / "campaign_state.json"
        partial = json.loads(state_path.read_text(encoding="utf-8"))
        partial["status"] = partial["axes"][-1]["status"] = transition_status(ExecutionStatus.FAILED)
        state_path.write_text(json.dumps(partial), encoding="utf-8")
        resumed = run_campaign(
            campaign_root=campaign_root,
            run_id="sequential-smoke",
            seed=29,
            timeout_seconds=30,
            resume=True,
            entrypoint=entrypoint,
        )
        assert resumed["status_counts"] == {"skipped": 1, "success": 25}
        state = json.loads(
            (campaign_root / "sequential-smoke" / "campaign_state.json").read_text(
                encoding="utf-8"
            )
        )
        assert all(row["resume_action"] == "verified_skip" for row in state["axes"][:-1])
        assert [len(row["attempts"]) for row in state["axes"][-2:]] == [1, 2]
        assert all("process_outcome" in row["attempts"][0] for row in state["axes"])
        source = next(row for row in state["fingerprint"]["sources"] if row["path"] == "quartz/idea_foundry/status_schema.py")
        source["sha256"] = "0" * 64
        with pytest.raises(SequentialCampaignError, match="source"):
            _validate_state(state, "sequential-smoke", 29, entrypoint)

        run_root = campaign_root / "sequential-smoke"
        original = state["axes"][0]
        for case, bad in (("row", None), ("attempts", {**original, "attempts": None})):
            state["axes"][0] = bad
            state_path.write_text(json.dumps(state), encoding="utf-8")
            target = run_root / f"malformed-{case}"
            with pytest.raises(MetaAnalysisError):
                analyze_campaign(run_root, target)
            assert not target.exists()
        state["axes"][0] = original
        state_path.write_text(json.dumps(state), encoding="utf-8")
        campaign_analysis = analyze_campaign(run_root)
        assert campaign_analysis["axis_count"] == 26
        assert campaign_analysis["contract_checks_failed"] == 0
        assert campaign_analysis["effect_records"] == []
        meta = run_meta_analysis(
            [run_root / "campaign_analysis" / "campaign_analysis.json"],
            run_root / "meta_analysis",
        )
        assert meta["status"] == first_gate_status("meta")
        assert meta["effect_record_count"] == 0
        campaign_path = run_root / "campaign_analysis" / "campaign_analysis.json"
        legacy_path = campaign_root / "legacy-campaign-analysis.json"
        legacy = json.loads(campaign_path.read_text(encoding="utf-8"))
        legacy["status"] = "ANALYZED_CONTRACT_ONLY"
        legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
        frozen_legacy = legacy_path.read_bytes()
        with pytest.raises(MetaAnalysisError, match="legacy status schema v1"):
            run_meta_analysis([legacy_path], campaign_root / "legacy-meta")
        assert legacy_path.read_bytes() == frozen_legacy
        assert not (campaign_root / "legacy-meta").exists()

        summary_path = run_root / "campaign_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        state["status"] = summary["status"] = first_gate_status("A10")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        skipped_output = run_root / "skipped-campaign-analysis"
        with pytest.raises(MetaAnalysisError, match="campaign status"):
            analyze_campaign(run_root, skipped_output)
        assert not skipped_output.exists()
        state["status"] = summary["status"] = first_gate_status("campaign")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        summary["axes"][0]["current_attempt"] = "axes/A03/attempt-999"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(MetaAnalysisError, match="axis state mismatch"):
            analyze_campaign(run_root, run_root / "tampered-campaign-analysis")

        state["status"] = "completed_no_promotion"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        frozen = state_path.read_bytes()
        with pytest.raises(SequentialCampaignError, match="legacy status schema v1"):
            run_campaign(campaign_root=campaign_root, run_id="sequential-smoke", seed=29, timeout_seconds=30, resume=True, entrypoint=entrypoint)
        with pytest.raises(MetaAnalysisError, match="legacy status schema v1"):
            analyze_campaign(run_root, run_root / "legacy-analysis")
        assert state_path.read_bytes() == frozen
        assert not (run_root / "legacy-analysis").exists()
# fmt: on
