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
    summarize_analysis_rows,
    validate_axis_analysis,
)
from quartz.idea_foundry.meta_analysis import (
    MetaAnalysisError,
    analyze_campaign,
    run_meta_analysis,
)
from quartz.idea_foundry.sequential import run_campaign


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
    assert payload["analysis_status"] == "ANALYZED_CONTRACT_ONLY"
    assert payload["effect_records"] == []
    assert payload["promotion"]["eligible"] is False

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
        assert first["status"] == "completed_no_promotion"
        assert first["axis_count"] == 26
        assert first["status_counts"] == {"completed_no_promotion": 26}

        resumed = run_campaign(
            campaign_root=campaign_root,
            run_id="sequential-smoke",
            seed=29,
            timeout_seconds=30,
            resume=True,
            entrypoint=entrypoint,
        )
        assert resumed["status_counts"] == {"completed_no_promotion": 26}
        state = json.loads(
            (campaign_root / "sequential-smoke" / "campaign_state.json").read_text(
                encoding="utf-8"
            )
        )
        assert all(row["resume_action"] == "verified_skip" for row in state["axes"])
        assert all(len(row["attempts"]) == 1 for row in state["axes"])

        run_root = campaign_root / "sequential-smoke"
        campaign_analysis = analyze_campaign(run_root)
        assert campaign_analysis["axis_count"] == 26
        assert campaign_analysis["contract_checks_failed"] == 0
        assert campaign_analysis["effect_records"] == []
        meta = run_meta_analysis(
            [run_root / "campaign_analysis" / "campaign_analysis.json"],
            run_root / "meta_analysis",
        )
        assert meta["status"] == "NO_COMPARABLE_EFFECTS"
        assert meta["effect_record_count"] == 0

        summary_path = run_root / "campaign_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["axes"][0]["current_attempt"] = "axes/A03/attempt-999"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        with pytest.raises(MetaAnalysisError, match="axis state mismatch"):
            analyze_campaign(run_root, run_root / "tampered-campaign-analysis")
