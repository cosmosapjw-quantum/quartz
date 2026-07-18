import importlib
import importlib.util
import argparse
import copy
import json
import sys
import types
import tomllib
from pathlib import Path

import numpy as np
import pytest


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_ablation_module():
    root = Path(__file__).resolve().parents[1]
    return load_module("ablation_study_script", root / "scripts" / "ablation_study.py")


def load_evaluator_calibration_module():
    root = Path(__file__).resolve().parents[1]
    return load_module(
        "evaluator_calibration_script", root / "scripts" / "evaluator_calibration.py"
    )


def load_smoke_module():
    root = Path(__file__).resolve().parents[1]
    return load_module("smoke_e2e_script", root / "scripts" / "smoke_e2e.py")


def load_gomocup_export_module():
    return importlib.import_module("quartz.gomocup_export")


def load_audit_bundle_script():
    root = Path(__file__).resolve().parents[1]
    return load_module(
        "build_audit_bundle_script", root / "scripts" / "build_audit_bundle.py"
    )


def write_condition_run(
    run_dir: Path, condition: str, seed: int | None, elo: float, loss: float
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "condition": condition,
        "game": "gomoku15",
        "seed": seed,
        "train_cfg": {"search_profile": "quartz", "vl_mode": "adaptive"},
        "elapsed_s": 120.0,
        "returncode": 0,
    }
    (run_dir / "condition.json").write_text(json.dumps(meta), encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"ckpt")
    rows = [
        {"loss": loss},
        {
            "_type": "eval",
            "published_elo": elo,
            "eval_verdict": "promote",
            "score_rate": 0.61,
        },
    ]
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_discover_model_runs_handles_flat_and_seeded_layout(tmp_path):
    ablation = load_ablation_module()
    base_dir = tmp_path / "results" / "gomoku15"

    write_condition_run(
        base_dir / "models" / "T1_noS_noVL", "T1_noS_noVL", None, 1510.0, 1.2
    )
    write_condition_run(
        base_dir / "models" / "T4_S_VL" / "seed_41", "T4_S_VL", 41, 1640.0, 0.9
    )
    write_condition_run(
        base_dir / "models" / "T4_S_VL" / "seed_42", "T4_S_VL", 42, 1655.0, 0.8
    )

    runs = sorted(ablation.discover_model_runs(base_dir), key=lambda row: row["id"])

    assert [row["id"] for row in runs] == ["T1_noS_noVL", "T4_S_VL_s41", "T4_S_VL_s42"]
    assert runs[0]["metrics"]["published_elo"] == 1510.0
    assert runs[2]["seed"] == 42
    assert runs[2]["model_path"].endswith("best.pt")


def test_build_study_manifest_uses_controller_preset():
    ablation = load_ablation_module()
    args = type(
        "Args",
        (),
        {
            "study": "controller",
            "game": "gomoku7",
            "iterations": 4,
            "eval_games": 12,
            "quick": True,
            "rust_binary": "./target/release/mcts_demo",
            "backend": "torch",
            "device": "cpu",
            "seeds": "7",
            "conditions": None,
            "eval_conditions": None,
            "include_strict_reference": False,
        },
    )()

    manifest = ablation.build_study_manifest(args)

    assert manifest["study"] == "controller"
    assert manifest["conditions"] == ["C1_impl_legacy", "C2_theory_doc"]
    assert (
        manifest["train_conditions"]["C1_impl_legacy"]["penalty_mode"]
        == "GatedRefreshLegacy"
    )
    assert manifest["eval_conditions"]["E2_theory_doc"]["root_only_shaping"] is True
    assert (
        manifest["train_condition_surfaces"]["C1_impl_legacy"]["penalty_mode"]
        == "GatedRefreshLegacy"
    )


def test_build_study_manifest_uses_controller_factorial_preset():
    ablation = load_ablation_module()
    args = type(
        "Args",
        (),
        {
            "study": "controller_factorial",
            "game": "gomoku7",
            "iterations": 3,
            "eval_games": 8,
            "quick": False,
            "rust_binary": "./target/release/mcts_demo",
            "backend": "torch",
            "device": "cpu",
            "seeds": "41,42",
            "conditions": None,
            "eval_conditions": "E1_legacy_base,E4_theory_krefresh",
            "include_strict_reference": False,
        },
    )()

    manifest = ablation.build_study_manifest(args)

    assert manifest["study"] == "controller_factorial"
    assert manifest["conditions"] == [
        "F1_legacy_base",
        "F2_legacy_krefresh",
        "F3_theory_base",
        "F4_theory_krefresh",
    ]
    assert manifest["eval_conditions_selected"] == [
        "E1_legacy_base",
        "E4_theory_krefresh",
    ]
    assert (
        manifest["train_conditions"]["F2_legacy_krefresh"]["prior_refresh_rate"] == 0.5
    )
    assert set(manifest["eval_conditions"]) == {"E1_legacy_base", "E4_theory_krefresh"}
    assert (
        manifest["eval_conditions"]["E4_theory_krefresh"]["prior_refresh_temp"] == 0.0
    )


def test_build_study_manifest_uses_controller_axes_preset():
    ablation = load_ablation_module()
    args = type(
        "Args",
        (),
        {
            "study": "controller_axes",
            "game": "gomoku7",
            "iterations": 3,
            "eval_games": 8,
            "quick": False,
            "rust_binary": "./target/release/mcts_demo",
            "backend": "torch",
            "device": "cpu",
            "seeds": "41,42",
            "conditions": None,
            "eval_conditions": None,
            "include_strict_reference": False,
        },
    )()

    manifest = ablation.build_study_manifest(args)

    assert manifest["study"] == "controller_axes"
    assert manifest["conditions"] == [
        "A1_legacy_tree_norefresh",
        "A2_legacy_root_norefresh",
        "A3_theory_root_norefresh",
        "A4_theory_root_refresh",
    ]
    assert (
        manifest["train_conditions"]["A2_legacy_root_norefresh"]["root_only_shaping"]
        is True
    )
    assert (
        manifest["train_conditions"]["A3_theory_root_norefresh"]["penalty_mode"]
        == "GatedRefresh"
    )
    assert (
        manifest["train_conditions"]["A4_theory_root_refresh"]["prior_refresh_rate"]
        == 0.5
    )
    assert (
        manifest["train_condition_surfaces"]["A4_theory_root_refresh"][
            "prior_refresh_rate"
        ]
        == 0.5
    )
    assert manifest["runtime_contract"]["config_layout"] == "repo_top_level_configs"
    assert isinstance(manifest["runtime_contract_hash"], str)
    assert len(manifest["runtime_contract_hash"]) == 16
    assert manifest["search_options_schema_version"] >= 1
    assert "penalty_mode" in manifest["search_options_keys"]


def test_build_study_manifest_rejects_unknown_search_option_key(monkeypatch):
    ablation = load_ablation_module()
    args = type(
        "Args",
        (),
        {
            "study": "search_vl",
            "game": "gomoku7",
            "iterations": 1,
            "eval_games": 2,
            "quick": True,
            "rust_binary": "./target/release/mcts_demo",
            "backend": "torch",
            "device": "cpu",
            "seeds": "42",
            "conditions": None,
            "eval_conditions": None,
            "include_strict_reference": False,
        },
    )()
    bad = copy.deepcopy(ablation.STUDY_PRESETS["search_vl"])
    bad["train_conditions"] = copy.deepcopy(bad["train_conditions"])
    bad["train_conditions"]["T1_noS_noVL"]["penalty_mod"] = "typo"
    monkeypatch.setitem(ablation.STUDY_PRESETS, "search_vl", bad)

    with pytest.raises(SystemExit, match="unknown search-option keys"):
        ablation.build_study_manifest(args)


def test_p04_research_grade_blocks_single_seed():
    """P04: --research-grade with fewer than min_seeds_for_research_grade
    seeds is a hard SystemExit BEFORE training starts. Default min is 3
    matching RESEARCH_READINESS.md."""
    ablation = load_ablation_module()
    args = argparse.Namespace(
        seeds="42",
        research_grade=True,
        min_seeds_for_research_grade=3,
        paired_seed_eval=False,
    )
    with pytest.raises(SystemExit, match="at least 3 seeds"):
        ablation.enforce_research_grade(args, None)


def test_p04_research_grade_passes_with_three_seeds_and_ready_report():
    """P04: three seeds + ready readiness ⇒ no SystemExit."""
    ablation = load_ablation_module()
    args = argparse.Namespace(
        seeds="11,22,33",
        research_grade=True,
        min_seeds_for_research_grade=3,
        paired_seed_eval=False,
    )
    report = {"research_readiness": {"research_grade_ready": True}}
    ablation.enforce_research_grade(args, report)  # no raise


def test_p04_research_grade_paired_seed_mismatch_blocks():
    """P04: under --paired-seed-eval, conditions A and B must share the
    same seed set. Mismatch ⇒ SystemExit. Triggers the second hard gate
    in enforce_research_grade."""
    ablation = load_ablation_module()
    args = argparse.Namespace(
        seeds="11,22,33",
        research_grade=True,
        min_seeds_for_research_grade=3,
        paired_seed_eval=True,
    )
    report = {
        "research_readiness": {"research_grade_ready": True},
        "runs": [
            {"condition": "A", "seed": 11},
            {"condition": "A", "seed": 22},
            {"condition": "A", "seed": 33},
            {"condition": "B", "seed": 11},
            {"condition": "B", "seed": 22},
            # missing seed 33 for B → mismatch
        ],
    }
    with pytest.raises(SystemExit, match="conditions disagree on seeds"):
        ablation.enforce_research_grade(args, report)


def test_p04_soft_warning_on_single_seed_without_research_grade(capsys):
    """P04: single-seed runs without --research-grade do NOT raise; a
    soft warning is printed to stderr so users learn the protocol
    convention but iteration is not blocked."""
    ablation = load_ablation_module()
    args = argparse.Namespace(
        seeds="42",
        research_grade=False,
        min_seeds_for_research_grade=3,
        paired_seed_eval=False,
    )
    ablation.enforce_research_grade(args, None)  # no raise
    captured = capsys.readouterr()
    assert "1 seed(s)" in captured.err
    assert "RESEARCH_READINESS" in captured.err


def test_research_grade_gate_fails_on_incomplete_report():
    ablation = load_ablation_module()
    # P04 added the multi-seed gate which fires BEFORE the readiness gate;
    # provide 3 seeds so this test continues to exercise the readiness
    # path it was originally written for.
    args = argparse.Namespace(
        seeds="11,22,33",
        research_grade=True,
        min_seeds_for_research_grade=3,
        paired_seed_eval=False,
    )
    report = {
        "research_readiness": {
            "research_grade_ready": False,
            "unmet_criteria": ["multi_seed_per_condition"],
        }
    }

    with pytest.raises(SystemExit, match="multi_seed_per_condition"):
        ablation.enforce_research_grade(args, report)


def test_seed_protocol_requires_paired_seed_eval_rows_for_claims():
    ablation = load_ablation_module()
    runs = [
        {"condition": "A", "seed": seed, "id": f"A_s{seed}"} for seed in (1, 2, 3)
    ] + [{"condition": "B", "seed": seed, "id": f"B_s{seed}"} for seed in (1, 2, 3)]
    eval_payload = {
        "runtime_contract": {"paired_seed_eval": False},
        "matches": [
            {"a_id": "A_s1", "b_id": "B_s1"},
            {"a_id": "A_s2", "b_id": "B_s2"},
            {"a_id": "A_s3", "b_id": "B_s3"},
        ],
    }

    summary = ablation.summarize_seed_protocol(runs, eval_payload)

    assert summary["common_seed_count"] == 3
    assert summary["eval_pairs"]["same_seed_pair_frac"] == 1.0
    assert summary["paired_seed_claim_ready"] is False


def test_p02_sha256_checkpoint_full_digest(tmp_path):
    """P02: sha256_checkpoint must return the FULL 64-char digest (not the
    16-char prefix `sha256_file_prefix` returns) so cross-checkpoint
    fingerprinting in the eval matrix is collision-resistant for the
    ~10^5 checkpoint scale a long campaign produces.
    """
    ablation = load_ablation_module()
    p = tmp_path / "ckpt.pt"
    p.write_bytes(b"hello")
    full = ablation.sha256_checkpoint(p)
    assert full is not None
    assert len(full) == 64
    # known SHA256 of "hello"
    assert full == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    # missing path returns None, no exception
    assert ablation.sha256_checkpoint(tmp_path / "nope.pt") is None
    assert ablation.sha256_checkpoint(None) is None
    # `sha256_file_prefix` independence — the two helpers must not be
    # accidentally aliased.
    short = ablation.sha256_file_prefix(p)
    assert short is not None
    assert len(short) == 16
    assert full.startswith(short)


def test_p02_pre_flight_check_passes_on_clean_inputs(tmp_path):
    """P02: clean checkpoints + non-null manifest hashes ⇒ ok=True, no errors."""
    ablation = load_ablation_module()
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"model_a_bytes")
    ckpt_b = tmp_path / "b.pt"
    ckpt_b.write_bytes(b"model_b_bytes")
    eligible = [
        {
            "id": "A_s42",
            "condition": "A",
            "seed": 42,
            "model_path": str(ckpt_a),
            "train_contract_hash": "abc",
        },
        {
            "id": "B_s42",
            "condition": "B",
            "seed": 42,
            "model_path": str(ckpt_b),
            "train_contract_hash": "def",
        },
    ]
    args = argparse.Namespace(paired_seed_eval=True, research_grade=False)
    summary = ablation.pre_flight_check(
        args, eligible, {"E1": {}}, {"E1": "manifest_hash_x"}
    )
    assert summary["ok"] is True
    assert summary["errors"] == []
    assert summary["skipped_pairs"] == []
    assert eligible[0]["candidate_hash"] is not None
    assert len(eligible[0]["candidate_hash"]) == 64
    assert eligible[1]["candidate_hash"] is not None


def test_p02_pre_flight_check_blocks_missing_checkpoint(tmp_path):
    """P02: missing checkpoint becomes an error and the run id ends up in
    skipped_pairs so should_compare can drop downstream pairs without
    crashing mid-eval. Under --research-grade ⇒ SystemExit."""
    ablation = load_ablation_module()
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"model_a_bytes")
    eligible = [
        {
            "id": "A_s42",
            "condition": "A",
            "seed": 42,
            "model_path": str(ckpt_a),
            "train_contract_hash": "abc",
        },
        {
            "id": "B_s42",
            "condition": "B",
            "seed": 42,
            "model_path": str(tmp_path / "missing.pt"),
            "train_contract_hash": "def",
        },
    ]
    args = argparse.Namespace(paired_seed_eval=True, research_grade=False)
    summary = ablation.pre_flight_check(
        args, eligible, {"E1": {}}, {"E1": "manifest_hash_x"}
    )
    assert summary["ok"] is False
    assert "B_s42" in summary["skipped_pairs"]
    reasons = {e["reason"] for e in summary["errors"]}
    assert "checkpoint_missing_or_unreadable" in reasons

    # Same inputs under --research-grade fail loudly.
    args_strict = argparse.Namespace(paired_seed_eval=True, research_grade=True)
    with pytest.raises(SystemExit, match="checkpoint_missing_or_unreadable"):
        ablation.pre_flight_check(
            args_strict, eligible, {"E1": {}}, {"E1": "manifest_hash_x"}
        )


def test_p02_pre_flight_check_blocks_null_manifest_hash(tmp_path):
    """P02: every eval condition must produce a non-null search_manifest_hash;
    a None / empty value silently breaks cross-condition comparison and
    must therefore be a hard error before launch."""
    ablation = load_ablation_module()
    ckpt_a = tmp_path / "a.pt"
    ckpt_a.write_bytes(b"model_a_bytes")
    eligible = [
        {
            "id": "A",
            "condition": "A",
            "seed": 1,
            "model_path": str(ckpt_a),
            "train_contract_hash": "abc",
        },
    ]
    args = argparse.Namespace(paired_seed_eval=False, research_grade=False)
    summary = ablation.pre_flight_check(
        args, eligible, {"E1": {}, "E2": {}}, {"E1": None, "E2": "ok"}
    )
    assert summary["ok"] is False
    bad = [
        e
        for e in summary["errors"]
        if e.get("reason") == "search_manifest_hash_missing"
    ]
    assert len(bad) == 1
    assert bad[0]["eval_condition"] == "E1"


def test_p02_pre_flight_check_catches_drifted_label_under_paired_seed(tmp_path):
    """P02: under --paired-seed-eval, two runs with the same (condition,
    seed) but different `train_contract_hash` indicate the label was
    overloaded — the same A_s42 was re-trained with a different config.
    Pairing them with B_s42 silently mixes incompatible models. Hard fail."""
    ablation = load_ablation_module()
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")
    eligible = [
        {
            "id": "A_s42_v1",
            "condition": "A",
            "seed": 42,
            "model_path": str(ckpt),
            "train_contract_hash": "v1_hash",
        },
        {
            "id": "A_s42_v2",
            "condition": "A",
            "seed": 42,
            "model_path": str(ckpt),
            "train_contract_hash": "v2_hash",
        },
    ]
    args = argparse.Namespace(paired_seed_eval=True, research_grade=False)
    summary = ablation.pre_flight_check(args, eligible, {"E1": {}}, {"E1": "ok"})
    assert summary["ok"] is False
    reasons = {e["reason"] for e in summary["errors"]}
    assert "duplicate_label_with_drifted_contract" in reasons


def test_resolve_model_path_prefers_latest_when_best_is_only_bootstrap_seed(tmp_path):
    ablation = load_ablation_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"bootstrap")
    (run_dir / "latest.pt").write_bytes(b"latest")
    (run_dir / "checkpoint_status.json").write_text(
        json.dumps(
            {
                "best_checkpoint_bootstrap_seeded": True,
                "preferred_posttrain_checkpoint": "latest.pt",
            }
        ),
        encoding="utf-8",
    )

    resolved = ablation.resolve_model_path(run_dir)

    assert resolved == run_dir / "latest.pt"


def test_resolve_model_path_prefers_best_when_promotion_recorded(tmp_path):
    """P1: when this run promoted, status advertises best.pt as preferred."""
    ablation = load_ablation_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_bytes(b"promoted")
    (run_dir / "latest.pt").write_bytes(b"latest")
    (run_dir / "checkpoint_status.json").write_text(
        json.dumps(
            {
                "best_checkpoint_bootstrap_seeded": False,
                "saw_promotion": True,
                "preferred_posttrain_checkpoint": "best.pt",
            }
        ),
        encoding="utf-8",
    )

    resolved = ablation.resolve_model_path(run_dir)

    assert resolved == run_dir / "best.pt"


def test_write_checkpoint_status_prefers_latest_when_no_promotion_in_rerun(tmp_path):
    """P1 regression guard (audit_codex_20260425.md W4):

    When a run reuses an output directory whose `best.pt` was a *prior*
    bootstrap (not a real promotion) and the current run also does not
    promote, `_write_checkpoint_status` must mark `latest.pt` as
    preferred — even though `best_checkpoint_bootstrap=False` for *this*
    run because best.pt already existed at start.
    """
    cli_main = importlib.import_module("quartz.cli_main")
    base_dir = tmp_path
    latest_path = base_dir / "latest.pt"
    best_path = base_dir / "best.pt"
    latest_path.write_bytes(b"latest")
    best_path.write_bytes(b"prior_bootstrap")

    payload = cli_main._write_checkpoint_status(
        str(base_dir),
        str(latest_path),
        str(best_path),
        best_checkpoint_bootstrap=False,  # this run did not seed; best.pt was carried over
        saw_promotion=False,  # this run did not promote
    )

    assert payload["preferred_posttrain_checkpoint"] == "latest.pt"
    on_disk = json.loads(
        (base_dir / "checkpoint_status.json").read_text(encoding="utf-8")
    )
    assert on_disk["preferred_posttrain_checkpoint"] == "latest.pt"


def test_write_checkpoint_status_prefers_best_after_promotion(tmp_path):
    """P1: a run that saw a promotion advertises best.pt regardless of bootstrap state."""
    cli_main = importlib.import_module("quartz.cli_main")
    base_dir = tmp_path
    latest_path = base_dir / "latest.pt"
    best_path = base_dir / "best.pt"
    latest_path.write_bytes(b"latest")
    best_path.write_bytes(b"promoted")

    payload = cli_main._write_checkpoint_status(
        str(base_dir),
        str(latest_path),
        str(best_path),
        best_checkpoint_bootstrap=False,
        saw_promotion=True,
    )

    assert payload["preferred_posttrain_checkpoint"] == "best.pt"


def test_select_champion_prefers_eval_leader_and_best_eval_cfg_for_deployment(tmp_path):
    ablation = load_ablation_module()
    base_dir = tmp_path / "results" / "gomoku15"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_runs = [
        {
            "id": "T1_noS_noVL",
            "condition": "T1_noS_noVL",
            "seed": 42,
            "game": "gomoku15",
            "train_cfg": {"search_profile": "baseline", "vl_mode": "disabled"},
            "run_dir": str(base_dir / "models" / "T1_noS_noVL"),
            "model_path": str(base_dir / "models" / "T1_noS_noVL" / "best.pt"),
            "success": True,
            "metrics": {"published_elo": 1500.0, "score_rate": 0.51},
        },
        {
            "id": "T4_S_VL_s42",
            "condition": "T4_S_VL",
            "seed": 42,
            "game": "gomoku15",
            "train_cfg": {"search_profile": "quartz", "vl_mode": "adaptive"},
            "run_dir": str(base_dir / "models" / "T4_S_VL" / "seed_42"),
            "model_path": str(base_dir / "models" / "T4_S_VL" / "seed_42" / "best.pt"),
            "success": True,
            "metrics": {"published_elo": 1675.0, "score_rate": 0.66},
        },
    ]
    matches = [
        {
            "eval_condition": "E2_S_noVL",
            "a_id": "T1_noS_noVL",
            "b_id": "T4_S_VL_s42",
            "games": 20,
            "wins_a": 6,
            "wins_b": 12,
            "draws": 2,
        },
        {
            "eval_condition": "E4_S_VL",
            "a_id": "T1_noS_noVL",
            "b_id": "T4_S_VL_s42",
            "games": 20,
            "wins_a": 4,
            "wins_b": 14,
            "draws": 2,
        },
    ]

    eval_payload = ablation.aggregate_matches(model_runs, matches)
    champion = ablation.select_champion(
        base_dir,
        model_runs,
        eval_payload,
        {
            "E2_S_noVL": {"search_profile": "quartz", "vl_mode": "disabled"},
            "E4_S_VL": {"search_profile": "quartz", "vl_mode": "adaptive"},
        },
    )

    assert champion["model_id"] == "T4_S_VL_s42"
    assert champion["deployment_eval_condition"] == "E4_S_VL"
    assert champion["deployment_search_cfg"]["search_profile"] == "quartz"
    assert champion["deployment_search_cfg"]["vl_mode"] == "adaptive"
    assert champion["deployment_cfg_source"] == "eval_condition:E4_S_VL"
    assert champion["selection_metrics"][
        "deployment_eval_condition_score_rate"
    ] == pytest.approx(0.75)


def test_build_eval_cfg_applies_runtime_overrides():
    ablation = load_ablation_module()
    cfg, _device = ablation.build_eval_cfg(
        "gomoku7",
        {
            "search_profile": "quartz",
            "vl_mode": "adaptive",
            "penalty_mode": "GatedRefreshLegacy",
            "root_only_shaping": False,
        },
        "cpu",
    )

    assert cfg["search_profile"] == "quartz"
    assert cfg["vl_mode"] == "adaptive"
    assert cfg["penalty_mode"] == "GatedRefreshLegacy"
    assert cfg["root_only_shaping"] is False


def test_build_audit_bundle_includes_configs_smoke_and_manifest(tmp_path):
    module = load_audit_bundle_script()
    root = Path(__file__).resolve().parents[1]
    stage_dir = tmp_path / "audit_stage"

    module.build_bundle(root, stage_dir)

    assert (stage_dir / "README.md").exists()
    assert (stage_dir / "configs" / "phase15_systems.default.json").exists()
    assert (stage_dir / "docs" / "RESEARCH_READINESS.md").exists()
    assert (stage_dir / "scripts" / "smoke_e2e.py").exists()
    assert (stage_dir / "scripts" / "evaluator_calibration.py").exists()
    assert (stage_dir / "tests" / "test_training_pipeline_regressions.py").exists()
    assert (stage_dir / "AUDIT_PACKAGE_MANIFEST.txt").exists()
    assert (stage_dir / "FILELIST.txt").exists()


def test_build_eval_engine_defaults_backend_preference_to_auto(monkeypatch):
    ablation = load_ablation_module()

    class FakeEngine:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    captured = {}

    monkeypatch.setattr(
        ablation,
        "build_eval_cfg",
        lambda *args, **kwargs: ({"_name": "gomoku7"}, "cpu"),
    )

    import quartz.runtime_support as support_mod
    import quartz.evaluator_runtime as eval_mod

    monkeypatch.setattr(
        support_mod,
        "load_actor_source_from_checkpoint",
        lambda model_path, engine_cfg, device, backend_preference=None: (
            captured.setdefault("backend_preference", backend_preference) or object()
        ),
    )
    monkeypatch.setattr(eval_mod, "RustNNEvaluatorEngine", FakeEngine)

    args = argparse.Namespace(
        game="gomoku7", device="cpu", rust_binary="./target/release/mcts_demo"
    )
    ablation.build_eval_engine(
        {"id": "m1", "model_path": "best.pt"},
        args,
        {"search_profile": "baseline", "vl_mode": "disabled"},
        "cpu",
    )

    assert captured["backend_preference"] == "auto"


def test_build_runtime_contract_records_binary_provenance(tmp_path):
    ablation = load_ablation_module()
    rust_binary = tmp_path / "mcts_demo"
    rust_binary.write_bytes(b"binary")
    args = argparse.Namespace(
        rust_binary=str(rust_binary),
        backend="torch",
        device="cpu",
        quick=False,
        no_autotune=True,
        resident_session=True,
        runtime_autotune=False,
        paired_seed_eval=True,
        include_strict_reference=True,
    )

    contract = ablation.build_runtime_contract(args)

    assert contract["rust_binary"] == str(rust_binary)
    assert contract["rust_binary_exists"] is True
    assert contract["rust_binary_sha256"] is not None
    assert contract["config_layout"] == "repo_top_level_configs"
    assert contract["paired_seed_eval"] is True
    assert contract["include_strict_reference"] is True


def test_docs_audit_surface_references_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "scripts/smoke_e2e.py",
        "scripts/build_audit_bundle.py",
        "scripts/controller_sweep.py",
        "scripts/controller_optuna.py",
        "scripts/evaluator_calibration.py",
        "scripts/phase15_ablation_study.py",
        "scripts/phase15_online_ablation.py",
        "scripts/phase15_benchmark.py",
        "scripts/phase15_mine_suite.py",
        "configs/phase15_systems.default.json",
        "docs/QUARTZ_THEORY.md",
        "docs/ABLATION_GUIDE.md",
        "docs/RESEARCH_READINESS.md",
        "docs/QUICKSTART.md",
        "README.md",
    ]

    missing = [relative for relative in required if not (root / relative).exists()]
    assert missing == []


def test_pyproject_does_not_claim_missing_quartz_configs_package_data():
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    quartz_data = payload["tool"]["setuptools"]["package-data"]["quartz"]

    assert "configs/*.json" not in quartz_data


def test_run_evaluation_matrix_can_limit_to_paired_seed_comparisons(
    tmp_path, monkeypatch
):
    ablation = load_ablation_module()
    import quartz.evaluator_runtime as eval_mod

    base_dir = tmp_path / "results" / "gomoku7"
    base_dir.mkdir(parents=True, exist_ok=True)
    called_pairs = []

    def fake_build_eval_cfg(game_name, eval_cfg, device_name, model_path=None):
        return (
            {
                "_name": game_name,
                "board": 7,
                "iters": 8,
                "search_profile": eval_cfg["search_profile"],
                "vl_mode": eval_cfg["vl_mode"],
            },
            "cpu",
        )

    class FakeEngine:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def reset(self):
            return None

    class FakeCampaign:
        def __init__(self, engines, num_games):
            self.timings = {"client_start_s": 0.01}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def compare(self, engine_a, engine_b, *args, **kwargs):
            called_pairs.append((engine_a.name(), engine_b.name()))
            return types.SimpleNamespace(
                wins=2,
                losses=2,
                draws=0,
                errors=0,
                voids=0,
                scored=4,
                total=4,
                score_rate=0.5,
            ), {
                "runner_mode": "rust_eval_state_machine",
                "match_elapsed_s": 0.01,
            }

    monkeypatch.setattr(ablation, "build_eval_cfg", fake_build_eval_cfg)
    monkeypatch.setattr(
        ablation,
        "build_eval_engine",
        lambda model_run, args, eval_cfg, device: (FakeEngine(model_run["id"]), {}),
    )
    monkeypatch.setattr(eval_mod, "PersistentRustNNEvalCampaign", FakeCampaign)

    args = type(
        "Args",
        (),
        {
            "game": "gomoku7",
            "device": "cpu",
            "eval_games": 4,
            "force_eval": True,
            "include_strict_reference": False,
            "rust_binary": "./target/release/mcts_demo",
            "paired_seed_eval": True,
        },
    )()

    # P02: pre_flight_check inside run_evaluation_matrix now requires
    # the checkpoint files to exist on disk; create them with distinct
    # bytes per (condition, seed) so the candidate hashes are unique
    # but otherwise opaque to this test (the FakeCampaign doesn't read
    # model bytes — only the pre-flight gate does).
    for fname in ("a41.pt", "b41.pt", "a42.pt", "b42.pt"):
        (base_dir / fname).write_bytes(fname.encode())
    model_runs = [
        {
            "id": "C1_impl_legacy_s41",
            "condition": "C1_impl_legacy",
            "seed": 41,
            "success": True,
            "model_path": str(base_dir / "a41.pt"),
        },
        {
            "id": "C2_theory_doc_s41",
            "condition": "C2_theory_doc",
            "seed": 41,
            "success": True,
            "model_path": str(base_dir / "b41.pt"),
        },
        {
            "id": "C1_impl_legacy_s42",
            "condition": "C1_impl_legacy",
            "seed": 42,
            "success": True,
            "model_path": str(base_dir / "a42.pt"),
        },
        {
            "id": "C2_theory_doc_s42",
            "condition": "C2_theory_doc",
            "seed": 42,
            "success": True,
            "model_path": str(base_dir / "b42.pt"),
        },
    ]

    payload = ablation.run_evaluation_matrix(
        args,
        base_dir,
        model_runs,
        ablation.CONTROLLER_EVAL_CONDITIONS,
    )

    assert len(payload["matches"]) == len(ablation.CONTROLLER_EVAL_CONDITIONS) * 2
    match_ids = {(row["a_id"], row["b_id"]) for row in payload["matches"]}
    assert ("C1_impl_legacy_s41", "C2_theory_doc_s41") in match_ids
    assert ("C1_impl_legacy_s42", "C2_theory_doc_s42") in match_ids
    assert called_pairs == [
        ("C1_impl_legacy_s41", "C2_theory_doc_s41"),
        ("C1_impl_legacy_s42", "C2_theory_doc_s42"),
    ] * len(ablation.CONTROLLER_EVAL_CONDITIONS)


def test_summarize_conditions_groups_training_and_eval_rows():
    ablation = load_ablation_module()
    runs = [
        {
            "id": "F1_legacy_base_s41",
            "condition": "F1_legacy_base",
            "metrics": {"published_elo": 10.0, "score_rate": 0.4, "loss": 4.2},
        },
        {
            "id": "F1_legacy_base_s42",
            "condition": "F1_legacy_base",
            "metrics": {"published_elo": 20.0, "score_rate": 0.6, "loss": 4.0},
        },
        {
            "id": "F4_theory_krefresh_s41",
            "condition": "F4_theory_krefresh",
            "metrics": {"published_elo": 50.0, "score_rate": 0.7, "loss": 3.8},
        },
    ]
    eval_payload = {
        "overall": [
            {
                "id": "F1_legacy_base_s41",
                "condition": "F1_legacy_base",
                "points": 3.0,
                "games": 8,
                "wins": 3,
                "losses": 5,
                "draws": 0,
            },
            {
                "id": "F1_legacy_base_s42",
                "condition": "F1_legacy_base",
                "points": 5.0,
                "games": 8,
                "wins": 5,
                "losses": 3,
                "draws": 0,
            },
            {
                "id": "F4_theory_krefresh_s41",
                "condition": "F4_theory_krefresh",
                "points": 6.0,
                "games": 8,
                "wins": 6,
                "losses": 2,
                "draws": 0,
            },
        ]
    }

    summary = ablation.summarize_conditions(runs, eval_payload)

    assert summary["training"][0]["condition"] == "F4_theory_krefresh"
    legacy_train = next(
        row for row in summary["training"] if row["condition"] == "F1_legacy_base"
    )
    assert legacy_train["mean_elo"] == 15.0
    assert legacy_train["mean_score_rate"] == 0.5
    assert legacy_train["mean_loss"] == 4.1
    legacy_eval = next(
        row for row in summary["evaluation"] if row["condition"] == "F1_legacy_base"
    )
    assert legacy_eval["entries"] == 2
    assert legacy_eval["score_rate"] == 0.5


def test_summarize_selection_trace_contract_groups_eval_conditions():
    ablation = load_ablation_module()
    eval_payload = {
        "matches": [
            {
                "eval_condition": "E1",
                "realized_budget_trace": {
                    "games": 2,
                    "selection_trace_coverage_frac": 1.0,
                    "selection_trace": {
                        "root_selects": 10,
                        "refresh_selected_count": 2,
                        "selected_penalty_abs_sum": 1.0,
                        "selected_effective_prior_l1_sum": 0.5,
                    },
                },
            },
            {
                "eval_condition": "E1",
                "realized_budget_trace": {
                    "games": 2,
                    "selection_trace_coverage_frac": 0.5,
                    "selection_trace": {
                        "root_selects": 30,
                        "refresh_selected_count": 6,
                        "selected_penalty_abs_sum": 3.0,
                        "selected_effective_prior_l1_sum": 1.5,
                    },
                },
            },
        ]
    }

    summary = ablation.summarize_selection_trace_contract(eval_payload)

    row = summary["conditions"][0]
    assert row["eval_condition"] == "E1"
    assert row["games"] == 4
    assert row["root_selects"] == 40
    assert row["selection_trace_coverage_frac"] == pytest.approx(0.75)
    assert row["refresh_selected_frac"] == 0.2
    assert row["mean_penalty_abs_per_root_select"] == 0.1
    assert row["mean_prior_l1_per_root_select"] == 0.05


def test_research_readiness_requires_selection_trace_coverage():
    ablation = load_ablation_module()
    runs = [
        {"condition": "C1", "seed": 41},
        {"condition": "C1", "seed": 42},
        {"condition": "C1", "seed": 43},
    ]
    eval_payload = {
        "matches": [{"eval_condition": "E1", "games": 4, "ci": [0.25, 0.75]}],
        "discarded_matches": [],
        "expected_benchmark_safe": {"E1": True},
        "expected_eval_seeds": {"E1": 17},
        "expected_search_manifests": {"E1": {"eval_seed": 17}},
    }
    champion = {"deployment_cfg_source": "eval_condition:E1"}
    pipeline_summary = {
        "aggregate": {
            "row_count": 3,
            "concurrent_run_count": 0,
            "freshness_coverage_frac": 1.0,
            "throughput_coverage_frac": 1.0,
            "worker_telemetry_coverage_frac": 0.0,
        }
    }
    budget_summary = {
        "budget_trace_coverage_frac": 1.0,
        "root_visit_mean_relative_spread": 0.0,
        "budget_fairness_flag": "ok",
    }
    seed_summary = {
        "condition_count": 1,
        "min_seed_count": 3,
        "common_seed_count": 3,
        "seed_sets_aligned": True,
        "eval_pairs": {"same_seed_pair_frac": None},
        "paired_seed_claim_ready": True,
    }
    hardware_summary = {
        "claim_scope": "runtime_telemetry_only",
        "profiler_artifact_present": False,
        "hardware_performance_claims_allowed": False,
    }
    evaluation_protocol_summary = {
        "protocol_ready": True,
        "match_count": 1,
        "runtime_contract_hash": "abcd1234abcd1234",
        "expected_eval_seed_coverage": True,
        "eval_seed_consistent": True,
        "benchmark_safe_all_expected": True,
        "game_count_consistent": True,
        "one_manifest_per_eval_condition": True,
        "complete_pair_eval_matrix": True,
    }
    evaluator_quality_summary = {
        "stratification_ready": True,
        "match_count": 1,
        "quality_proxy_pair_coverage_frac": 1.0,
        "loss_pair_coverage_frac": 1.0,
        "models_with_quality_proxy": 2,
        "strata_count": 1,
        "missing_model_ids": [],
    }
    heldout_calibration_summary = {
        "artifact_present": True,
        "calibration_ready": True,
        "coverage_frac": 1.0,
        "missing_model_ids": [],
    }

    legacy_summary = {
        "conditions": [
            {
                "eval_condition": "E1",
                "root_selects": 16,
                "selection_trace_coverage_frac": None,
            }
        ]
    }
    legacy_ready = ablation.research_readiness_summary(
        runs,
        eval_payload,
        legacy_summary,
        budget_summary,
        seed_summary,
        pipeline_summary,
        hardware_summary,
        champion,
        evaluation_protocol_summary=evaluation_protocol_summary,
        evaluator_quality_summary=evaluator_quality_summary,
        heldout_calibration_summary=heldout_calibration_summary,
    )
    assert "selection_trace_recorded" in legacy_ready["unmet_criteria"]

    covered_summary = {
        "conditions": [
            {
                "eval_condition": "E1",
                "root_selects": 16,
                "selection_trace_coverage_frac": 1.0,
            }
        ]
    }
    covered_ready = ablation.research_readiness_summary(
        runs,
        eval_payload,
        covered_summary,
        budget_summary,
        seed_summary,
        pipeline_summary,
        hardware_summary,
        champion,
        evaluation_protocol_summary=evaluation_protocol_summary,
        evaluator_quality_summary=evaluator_quality_summary,
        heldout_calibration_summary=heldout_calibration_summary,
    )
    assert "selection_trace_recorded" not in covered_ready["unmet_criteria"]
    assert "hardware_claim_scope_recorded" not in covered_ready["unmet_criteria"]
    assert covered_ready["research_grade_ready"] is True


def test_summarize_evaluator_quality_strata_groups_quality_proxies():
    ablation = load_ablation_module()
    runs = [
        {
            "id": "A_s1",
            "condition": "A",
            "seed": 1,
            "metrics": {
                "loss": 0.8,
                "p_loss": 0.55,
                "v_loss": 0.25,
                "loss_ema": 0.85,
                "published_elo": 1500.0,
                "score_rate": 0.60,
                "games_done": 32,
            },
        },
        {
            "id": "B_s1",
            "condition": "B",
            "seed": 1,
            "metrics": {
                "loss": 1.7,
                "published_elo": 1450.0,
                "score_rate": 0.45,
                "games_done": 32,
            },
        },
    ]
    eval_payload = {
        "matches": [
            {
                "eval_condition": "E1",
                "a_id": "A_s1",
                "b_id": "B_s1",
                "games": 4,
                "scored_games": 4,
                "score_rate_a": 0.75,
            },
            {
                "eval_condition": "E2",
                "a_id": "A_s1",
                "b_id": "B_s1",
                "games": 4,
                "scored_games": 4,
                "score_rate_a": 0.50,
            },
        ]
    }

    summary = ablation.summarize_evaluator_quality_strata(runs, eval_payload)

    assert summary["stratification_ready"] is True
    assert summary["quality_proxy_pair_coverage_frac"] == pytest.approx(1.0)
    assert summary["loss_pair_coverage_frac"] == pytest.approx(1.0)
    assert summary["loss_bucket_counts"] == {"loss_ge_1_5": 1, "loss_lt_1_0": 1}
    assert summary["model_quality"]["A_s1"]["p_loss"] == pytest.approx(0.55)
    assert summary["model_quality"]["A_s1"]["loss_bucket"] == "loss_lt_1_0"
    assert summary["strata"][0]["stratum"] == "loss_ge_1_5__loss_lt_1_0"
    assert summary["strata"][0]["matches"] == 2
    assert summary["strata"][0]["score_rate_a"]["mean"] == pytest.approx(0.625)
    assert sorted(summary["strata"][0]["eval_conditions"]) == ["E1", "E2"]


def test_summarize_evaluator_quality_strata_flags_missing_quality():
    ablation = load_ablation_module()
    runs = [
        {"id": "A_s1", "condition": "A", "seed": 1, "metrics": {}},
    ]
    eval_payload = {
        "matches": [
            {"eval_condition": "E1", "a_id": "A_s1", "b_id": "B_s1", "games": 4},
        ]
    }

    summary = ablation.summarize_evaluator_quality_strata(runs, eval_payload)

    assert summary["stratification_ready"] is False
    assert summary["quality_proxy_pair_coverage_frac"] == pytest.approx(0.0)
    assert summary["missing_model_ids"] == ["B_s1"]
    assert summary["strata"][0]["stratum"] == "quality_unknown"


def test_summarize_heldout_calibration_requires_all_model_metrics(tmp_path):
    ablation = load_ablation_module()
    runs = [
        {"id": "A_s1", "model_path": "/tmp/a.pt"},
        {"id": "B_s1", "model_path": "/tmp/b.pt"},
    ]
    (tmp_path / "evaluator_calibration.json").write_text(
        json.dumps(
            {
                "models": {
                    "A_s1": {
                        "n_positions": 16,
                        "policy_nll": 1.2,
                        "value_mse": 0.3,
                        "top1_acc": 0.5,
                        "brier": 0.2,
                    },
                    "B_s1": {
                        "n_positions": 0,
                        "policy_nll": 1.5,
                        "value_mse": 0.4,
                        "top1_acc": 0.4,
                        "brier": 0.3,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    summary = ablation.summarize_heldout_calibration(tmp_path, runs)

    assert summary["artifact_present"] is True
    assert summary["calibration_ready"] is False
    assert summary["covered_model_count"] == 1
    assert summary["missing_model_ids"] == ["B_s1"]


def test_evaluator_calibration_metrics_scores_policy_and_value():
    calibration = load_evaluator_calibration_module()
    from quartz.replay import ReplayExample, sparse_policy_from_dense

    examples = [
        ReplayExample(
            state=np.zeros((2, 2), dtype=np.float32),
            policy=sparse_policy_from_dense([1.0, 0.0, 0.0]),
            value=0.5,
        ),
        ReplayExample(
            state=np.ones((2, 2), dtype=np.float32),
            policy=sparse_policy_from_dense([0.0, 1.0, 0.0]),
            value=-0.5,
        ),
    ]

    class FakeActor:
        def predict(self, batch):
            probs = []
            values = []
            for state in batch:
                if float(np.mean(state)) < 0.5:
                    probs.append([0.8, 0.1, 0.1])
                    values.append(0.4)
                else:
                    probs.append([0.2, 0.7, 0.1])
                    values.append(-0.25)
            return np.asarray(probs, dtype=np.float32), np.asarray(
                values, dtype=np.float32
            )

    metrics = calibration.calibration_metrics(
        FakeActor(), "cpu", examples, batch_size=2
    )

    assert metrics["n_positions"] == 2
    assert metrics["policy_nll"] == pytest.approx(-0.5 * (np.log(0.8) + np.log(0.7)))
    assert metrics["value_mse"] == pytest.approx((0.01 + 0.0625) / 2.0)
    assert metrics["top1_acc"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.10)


def test_summarize_evaluation_protocol_detects_consistent_protocol():
    ablation = load_ablation_module()
    runs = [
        {"id": "A_s1", "condition": "A", "seed": 1},
        {"id": "B_s1", "condition": "B", "seed": 1},
    ]
    eval_payload = {
        "runtime_contract": {
            "paired_seed_eval": True,
            "backend": "torch",
            "device": "cpu",
        },
        "runtime_contract_hash": "abcd1234abcd1234",
        "expected_eval_seeds": {"E1": 17, "E2": 17},
        "expected_benchmark_safe": {"E1": True, "E2": True},
        "matches": [
            {
                "eval_condition": "E1",
                "a_id": "A_s1",
                "b_id": "B_s1",
                "games": 4,
                "scored_games": 4,
                "search_manifest_hash": "h1",
                "runner_mode": "rust_eval_state_machine",
            },
            {
                "eval_condition": "E2",
                "a_id": "A_s1",
                "b_id": "B_s1",
                "games": 4,
                "scored_games": 4,
                "search_manifest_hash": "h2",
                "runner_mode": "rust_eval_state_machine",
            },
        ],
    }

    summary = ablation.summarize_evaluation_protocol(runs, eval_payload)

    assert summary["protocol_ready"] is True
    assert summary["eval_seed_consistent"] is True
    assert summary["game_count_consistent"] is True
    assert summary["one_manifest_per_eval_condition"] is True
    assert summary["complete_pair_eval_matrix"] is True
    assert summary["pair_id_coverage_frac"] == pytest.approx(1.0)
    assert summary["search_manifest_hash_coverage_frac"] == pytest.approx(1.0)
    assert summary["pair_eval_condition_coverage"] == {"A_s1||B_s1": ["E1", "E2"]}


def test_summarize_evaluation_protocol_flags_protocol_drift():
    ablation = load_ablation_module()
    eval_payload = {
        "runtime_contract": {"backend": "torch", "device": "cpu"},
        "runtime_contract_hash": "abcd1234abcd1234",
        "expected_eval_seeds": {"E1": 17, "E2": 19},
        "expected_benchmark_safe": {"E1": True, "E2": True},
        "matches": [
            {
                "eval_condition": "E1",
                "a_id": "A_s1",
                "b_id": "B_s1",
                "games": 4,
                "search_manifest_hash": "h1",
                "runner_mode": "rust_eval_state_machine",
            },
            {
                "eval_condition": "E1",
                "a_id": "A_s2",
                "b_id": "B_s2",
                "games": 8,
                "search_manifest_hash": "h2",
                "runner_mode": "rust_eval_state_machine",
            },
            {
                "eval_condition": "E2",
                "a_id": "A_s1",
                "b_id": "B_s1",
                "games": 4,
                "search_manifest_hash": "h3",
                "runner_mode": "rust_eval_state_machine",
            },
        ],
    }

    summary = ablation.summarize_evaluation_protocol([], eval_payload)

    assert summary["protocol_ready"] is False
    assert summary["eval_seed_consistent"] is False
    assert summary["game_count_consistent"] is False
    assert summary["one_manifest_per_eval_condition"] is False
    assert summary["complete_pair_eval_matrix"] is False


def test_summarize_budget_fairness_groups_realized_budget_trace():
    ablation = load_ablation_module()
    eval_payload = {
        "matches": [
            {
                "eval_condition": "E1",
                "games": 2,
                "realized_budget_trace": {
                    "games": 2,
                    "moves": 4,
                    "root_visits": {
                        "samples": [8, 10, 12, 10],
                        "mean": 10.0,
                        "max": 12.0,
                    },
                    "halt_reason_hist": {"BudgetExhausted": 4},
                    "benchmark_safe_frac": 1.0,
                },
            },
            {
                "eval_condition": "E2",
                "games": 2,
                "realized_budget_trace": {
                    "games": 2,
                    "moves": 4,
                    "root_visits": {
                        "samples": [20, 20, 20, 20],
                        "mean": 20.0,
                        "max": 20.0,
                    },
                    "halt_reason_hist": {"Converged": 4},
                    "benchmark_safe_frac": 1.0,
                },
            },
        ]
    }

    summary = ablation.summarize_budget_fairness(eval_payload)

    assert summary["budget_trace_coverage_frac"] == pytest.approx(1.0)
    assert summary["budget_fairness_flag"] == "drift"
    assert summary["root_visit_mean_relative_spread"] == pytest.approx(0.5)
    e1 = summary["conditions"][0]
    assert e1["eval_condition"] == "E1"
    assert e1["root_visits"]["mean"] == pytest.approx(10.0)
    assert e1["root_visits"]["sample_count"] == 4
    assert e1["halt_reason_hist"]["BudgetExhausted"] == 4


def test_summarize_seed_protocol_detects_unpaired_seed_sets():
    ablation = load_ablation_module()
    runs = [
        {"id": "A_s1", "condition": "A", "seed": 1},
        {"id": "A_s2", "condition": "A", "seed": 2},
        {"id": "A_s3", "condition": "A", "seed": 3},
        {"id": "B_s2", "condition": "B", "seed": 2},
        {"id": "B_s3", "condition": "B", "seed": 3},
        {"id": "B_s4", "condition": "B", "seed": 4},
    ]
    eval_payload = {
        "runtime_contract": {"paired_seed_eval": True},
        "matches": [
            {"a_id": "A_s2", "b_id": "B_s2"},
            {"a_id": "A_s3", "b_id": "B_s4"},
        ],
    }

    summary = ablation.summarize_seed_protocol(runs, eval_payload)

    assert summary["min_seed_count"] == 3
    assert summary["common_seeds"] == [2, 3]
    assert summary["common_seed_count"] == 2
    assert summary["seed_sets_aligned"] is False
    assert summary["paired_seed_claim_ready"] is False
    assert summary["eval_pairs"]["same_seed_pair_frac"] == pytest.approx(0.5)


def test_summarize_hardware_runtime_downgrades_without_profiler(tmp_path):
    ablation = load_ablation_module()
    run_dir = tmp_path / "models" / "C1"
    run_dir.mkdir(parents=True)
    runs = [
        {
            "id": "C1",
            "run_dir": str(run_dir),
            "train_contract": {
                "runtime_contract": {"backend": "torch", "device": "cuda"},
            },
        }
    ]
    pipeline_summary = {
        "aggregate": {
            "row_count": 1,
            "pos_per_s": {"mean": 12.5},
            "freshness": {"mean": 0.4},
            "worker_rolling_positions_per_s": {"mean": None},
            "inference_eval_items": 0,
            "inference_model_calls": 0,
        }
    }

    summary = ablation.summarize_hardware_runtime(tmp_path, runs, {}, pipeline_summary)

    assert summary["requested_backends"] == ["torch"]
    assert summary["requested_devices"] == ["cuda"]
    assert summary["profiler_artifact_present"] is False
    assert summary["hardware_performance_claims_allowed"] is False
    assert summary["claim_scope"] == "runtime_telemetry_only"


def test_summarize_hardware_runtime_detects_profiler_artifact(tmp_path):
    ablation = load_ablation_module()
    (tmp_path / "throughput_profile.json").write_text("{}", encoding="utf-8")

    summary = ablation.summarize_hardware_runtime(tmp_path, [], {}, {"aggregate": {}})

    assert summary["profiler_artifact_present"] is True
    assert summary["hardware_performance_claims_allowed"] is True
    assert summary["claim_scope"] == "hardware_profiled"
    assert summary["profiler_artifacts"] == ["throughput_profile.json"]


def test_summarize_pipeline_telemetry_reads_train_logs(tmp_path):
    ablation = load_ablation_module()
    run_dir = tmp_path / "models" / "C1" / "s41"
    run_dir.mkdir(parents=True)
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "iter": 1,
                        "replay_freshness": 0.25,
                        "pos_per_s": 12.0,
                        "new_pos": 32,
                        "train_steps": 2,
                        "selfplay_telemetry": {
                            "last_progress_age_s": 0.4,
                            "rolling_positions_per_s": 40.0,
                            "backpressure_waits": 1,
                            "inference": {
                                "eval_items": 96,
                                "eval_messages": 12,
                                "model_calls": 8,
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "iter": 2,
                        "replay_freshness": 0.5,
                        "pos_per_s": 16.0,
                        "new_pos": 48,
                        "train_steps": 3,
                        "selfplay_telemetry": {
                            "last_progress_age_s": 0.2,
                            "rolling_positions_per_s": 44.0,
                            "backpressure_waits": 2,
                            "inference": {
                                "eval_items": 120,
                                "eval_messages": 15,
                                "model_calls": 10,
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runs = [
        {
            "id": "C1_s41",
            "condition": "C1",
            "seed": 41,
            "run_dir": str(run_dir),
            "train_contract": {"concurrent": True},
        }
    ]

    summary = ablation.summarize_pipeline_telemetry(runs)

    agg = summary["aggregate"]
    assert agg["row_count"] == 2
    assert agg["freshness_coverage_frac"] == pytest.approx(1.0)
    assert agg["throughput_coverage_frac"] == pytest.approx(1.0)
    assert agg["worker_telemetry_coverage_frac"] == pytest.approx(1.0)
    assert agg["freshness"]["mean"] == pytest.approx(0.375)
    assert agg["pos_per_s"]["mean"] == pytest.approx(14.0)
    assert agg["selfplay_queue_latency_s"]["max"] == pytest.approx(0.4)
    assert agg["inference_eval_items"] == 216
    assert agg["inference_model_calls"] == 18


def test_build_gomocup_manifest_captures_search_and_selection_metadata():
    gomocup_export = load_gomocup_export_module()
    manifest = gomocup_export.build_gomocup_manifest(
        "gomoku15",
        "models/champion.pt",
        "gomocup_model.onnx",
        metadata={
            "condition": "T4_S_VL",
            "seed": 42,
            "training_metrics": {"published_elo": 1675.0},
            "selection_metrics": {"overall_score_rate": 0.70},
        },
        search_cfg={
            "search_profile": "quartz",
            "vl_mode": "adaptive",
            "tt_enabled": True,
            "budget_ms": 1000,
            "max_visits": 50000,
        },
    )

    assert manifest["game"] == "gomoku15"
    assert manifest["gomocup_rule"] == "freestyle"
    assert manifest["search"]["search_profile"] == "quartz"
    assert manifest["search"]["max_visits"] == 50000
    assert manifest["source"]["condition"] == "T4_S_VL"
    assert manifest["source"]["selection_metrics"]["overall_score_rate"] == 0.70


def test_pin_halt_mode_stamps_fixed_for_controller_axes_preset():
    """P7 (audit_codex_20260425.md W2): attribution presets get
    halt_mode='fixed' on every train + eval condition.
    """
    ablation = load_ablation_module()
    preset = ablation.resolve_study_preset("controller_axes")
    pinned = ablation.pin_halt_mode_for_attribution(preset, "controller_axes")

    for cond_cfg in pinned["train_conditions"].values():
        assert cond_cfg["halt_mode"] == "fixed"
    for cond_cfg in pinned["eval_conditions"].values():
        assert cond_cfg["halt_mode"] == "fixed"


def test_pin_halt_mode_does_not_overwrite_explicit_value():
    """P7: a user-provided halt_mode in the preset cfg is preserved."""
    ablation = load_ablation_module()
    preset = {
        "train_conditions": {
            "T_a": {"penalty_mode": "GatedRefresh", "halt_mode": "voc"},
        },
        "eval_conditions": {
            "E_a": {"penalty_mode": "GatedRefresh"},
        },
    }
    pinned = ablation.pin_halt_mode_for_attribution(preset, "controller_axes")

    assert pinned["train_conditions"]["T_a"]["halt_mode"] == "voc"
    assert pinned["eval_conditions"]["E_a"]["halt_mode"] == "fixed"


def test_pin_halt_mode_is_noop_for_search_vl_preset():
    """P7: non-attribution presets do not get the halt_mode stamp."""
    ablation = load_ablation_module()
    preset = ablation.resolve_study_preset("search_vl")
    pinned = ablation.pin_halt_mode_for_attribution(preset, "search_vl")

    for cond_cfg in pinned["train_conditions"].values():
        assert "halt_mode" not in cond_cfg


def test_q4_halt_attribution_preset_varies_halt_mode():
    """Q4 (audit_codex_20260428.md W'4): halt_attribution rows must differ
    only in halt_mode while penalty_mode and refresh stay constant. This
    is the contract that lets a reader attribute compute savings to the
    halt mode itself rather than to a confound.
    """
    ablation = load_ablation_module()
    preset = ablation.resolve_study_preset("halt_attribution")
    train = preset["train_conditions"]
    # Three rows expected, each pinning a distinct halt_mode.
    halt_modes = sorted(cfg["halt_mode"] for cfg in train.values())
    assert halt_modes == ["fixed", "simple_threshold", "voc"]
    # All other identity fields must be identical across rows.
    invariants = (
        "penalty_mode",
        "root_only_shaping",
        "prior_refresh_rate",
        "vl_mode",
        "search_profile",
    )
    first = next(iter(train.values()))
    for cfg in train.values():
        for key in invariants:
            assert cfg.get(key) == first.get(key), (
                f"halt_attribution preset rows must agree on {key}"
            )


def test_q4_pin_halt_mode_does_not_clobber_halt_attribution_rows():
    """Q4: when halt_attribution is pinned for fairness, the explicit
    per-row halt_mode (the variable being studied) must survive.
    `pin_halt_mode_for_attribution` is a noop for halt_attribution
    today, but if halt_attribution is ever added to
    CONTROLLER_ATTRIBUTION_PRESETS the setdefault discipline still
    preserves the explicit value.
    """
    ablation = load_ablation_module()
    preset = ablation.resolve_study_preset("halt_attribution")
    pinned = ablation.pin_halt_mode_for_attribution(preset, "halt_attribution")
    # Whether the function noops or applies setdefault, every row's
    # explicit halt_mode must still be present and varied.
    halt_modes = sorted(cfg["halt_mode"] for cfg in pinned["train_conditions"].values())
    assert halt_modes == ["fixed", "simple_threshold", "voc"]


def test_q4_resolve_frozen_eval_pins_first_for_halt_attribution():
    """Q4: halt_attribution presets need a single frozen eval condition so
    the comparison varies only halt_mode, not the eval engine itself."""
    ablation = load_ablation_module()
    eval_conditions = {
        "EH2_simple_threshold": {},
        "EH1_voc_default": {},
        "EH3_fixed_full_budget": {},
    }
    args = _make_p8_args("halt_attribution")
    # Sorted-first must be EH1_voc_default.
    assert (
        ablation.resolve_frozen_eval_condition(args, eval_conditions)
        == "EH1_voc_default"
    )


def test_q4_attribution_preset_tag_marks_halt_axis_preset():
    """Q4: study_manifest.attribution metadata must distinguish controller-
    attribution presets (halt pinned) from halt-axis presets (halt varies)."""
    ablation = load_ablation_module()
    halt_tag = ablation.attribution_preset_tag("halt_attribution")
    assert halt_tag["halt_axis_preset"] is True
    assert halt_tag["attribution_preset"] is False
    ctrl_tag = ablation.attribution_preset_tag("controller_axes")
    assert ctrl_tag["halt_axis_preset"] is False
    assert ctrl_tag["attribution_preset"] is True
    none_tag = ablation.attribution_preset_tag("search_vl")
    assert none_tag["halt_axis_preset"] is False
    assert none_tag["attribution_preset"] is False


def test_pin_halt_mode_does_not_mutate_original_preset():
    """P7: deep-copy semantics — module-level constants stay untouched."""
    ablation = load_ablation_module()
    original = ablation.resolve_study_preset("controller_axes")
    before = copy.deepcopy(original)

    ablation.pin_halt_mode_for_attribution(original, "controller_axes")

    # Original preset is unchanged after the call.
    for cond_name, cond_cfg in original["train_conditions"].items():
        assert cond_cfg == before["train_conditions"][cond_name]


def _make_p8_args(study, frozen_eval_condition=None, no_frozen_eval=False):
    return argparse.Namespace(
        study=study,
        frozen_eval_condition=frozen_eval_condition,
        no_frozen_eval=no_frozen_eval,
    )


def test_resolve_frozen_eval_default_picks_first_for_attribution_preset():
    """P8: attribution preset auto-resolves to the alphabetically first eval condition."""
    ablation = load_ablation_module()
    eval_conditions = {"EA2_b": {}, "EA1_a": {}, "EA3_c": {}}
    args = _make_p8_args("controller_axes")
    assert ablation.resolve_frozen_eval_condition(args, eval_conditions) == "EA1_a"


def test_resolve_frozen_eval_returns_none_for_non_attribution_preset():
    """P8: non-attribution presets keep the legacy per-row matrix by default."""
    ablation = load_ablation_module()
    eval_conditions = {"E1_a": {}, "E2_b": {}}
    args = _make_p8_args("search_vl")
    assert ablation.resolve_frozen_eval_condition(args, eval_conditions) is None


def test_resolve_frozen_eval_explicit_name_wins():
    """P8: explicit `--frozen-eval-condition NAME` overrides the auto default."""
    ablation = load_ablation_module()
    eval_conditions = {"EA1_a": {}, "EA2_b": {}, "EA3_c": {}}
    args = _make_p8_args("controller_axes", frozen_eval_condition="EA3_c")
    assert ablation.resolve_frozen_eval_condition(args, eval_conditions) == "EA3_c"


def test_resolve_frozen_eval_no_flag_opts_out():
    """P8: `--no-frozen-eval` opts out even for attribution presets."""
    ablation = load_ablation_module()
    eval_conditions = {"EA1_a": {}, "EA2_b": {}}
    args = _make_p8_args("controller_axes", no_frozen_eval=True)
    assert ablation.resolve_frozen_eval_condition(args, eval_conditions) is None


def test_resolve_frozen_eval_unknown_name_raises():
    """P8: explicit unknown name fails fast."""
    import pytest

    ablation = load_ablation_module()
    eval_conditions = {"EA1_a": {}}
    args = _make_p8_args("controller_axes", frozen_eval_condition="bogus")
    with pytest.raises(SystemExit):
        ablation.resolve_frozen_eval_condition(args, eval_conditions)


def test_controller_identity_hash_is_stable_under_dict_reordering():
    """P5: hash must depend only on values, not Python dict insertion order."""
    ablation = load_ablation_module()
    cfg_a = {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "penalty_mode": "GatedRefresh",
        "halt_mode": "VOC",
        "prior_refresh_temp": 1.0,
    }
    cfg_b = dict(reversed(list(cfg_a.items())))

    assert ablation.controller_identity_hash(
        cfg_a
    ) == ablation.controller_identity_hash(cfg_b)


def test_controller_identity_hash_changes_with_any_identity_field():
    """P5: each controller-identity field individually perturbs the hash."""
    ablation = load_ablation_module()
    base = {key: None for key in ablation.controller_identity_keys()}
    base.update(
        {
            "search_profile": "quartz",
            "penalty_mode": "GatedRefresh",
            "halt_mode": "VOC",
            "prior_refresh_temp": 1.0,
        }
    )
    base_hash = ablation.controller_identity_hash(base)

    for key in ablation.controller_identity_keys():
        perturbed = dict(base)
        perturbed[key] = "PERTURBED-VALUE"
        assert ablation.controller_identity_hash(perturbed) != base_hash, (
            f"hash should change when {key!r} changes"
        )


def test_controller_identity_ignores_unknown_keys():
    """P5: keys outside the identity surface do not perturb the hash."""
    ablation = load_ablation_module()
    cfg = {"penalty_mode": "GatedRefresh", "halt_mode": "VOC"}
    cfg_with_extras = dict(cfg, max_visits=400, eval_games=10, comment="aux")

    assert ablation.controller_identity_hash(cfg) == ablation.controller_identity_hash(
        cfg_with_extras
    )


def test_assert_single_axis_isolation_passes_when_only_axis_varies():
    """P5: hash-modulo-axis is constant across rows of a single-axis preset."""
    ablation = load_ablation_module()
    base = {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
        "halt_mode": "Fixed",
        "prior_refresh_temp": 1.0,
        "hbar_penalty_cap": 0.3,
    }
    surfaces = {
        "A1_legacy": dict(base, penalty_mode="Legacy"),
        "A2_gated": dict(base, penalty_mode="GatedRefresh"),
        "A3_pflip": dict(base, penalty_mode="PFlipMixture"),
    }

    ok, hashes = ablation.assert_single_axis_isolation(surfaces, ("penalty_mode",))

    assert ok, f"single-axis preset should isolate penalty_mode, hashes={hashes}"
    assert len(set(hashes.values())) == 1


def test_assert_single_axis_isolation_fails_when_two_axes_vary():
    """P5: two axes varying → not single-axis isolated."""
    ablation = load_ablation_module()
    base = {
        "search_profile": "quartz",
        "vl_mode": "adaptive",
    }
    surfaces = {
        "row_a": dict(base, penalty_mode="GatedRefresh", halt_mode="Fixed"),
        # halt_mode also drifts — should be detected
        "row_b": dict(base, penalty_mode="GatedRefresh", halt_mode="VOC"),
        "row_c": dict(base, penalty_mode="PFlipMixture", halt_mode="Fixed"),
    }

    ok, _ = ablation.assert_single_axis_isolation(surfaces, ("penalty_mode",))

    assert not ok


def test_smoke_count_sgd_rows_counts_only_loss_present_rows(tmp_path):
    """P3: count_sgd_rows counts non-null `loss` rows only."""
    smoke = load_smoke_module()
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"loss": 0.4, "iter": 1}),
                json.dumps({"loss": None, "iter": 2}),
                json.dumps({"_type": "eval", "verdict": "skip"}),  # no loss key
                json.dumps({"loss": 0.35, "iter": 3}),
                "not-json",  # corrupt row tolerated
            ]
        ),
        encoding="utf-8",
    )

    assert smoke.count_sgd_rows(log_path) == 2


def test_smoke_verify_training_fired_raises_when_zero_sgd(tmp_path):
    """P3: verify_training_fired surfaces zero-SGD smoke runs as fail-fast."""
    import pytest

    smoke = load_smoke_module()
    report_dir = tmp_path / "gomoku7"
    models_dir = report_dir / "models" / "T1_noS_noVL"
    models_dir.mkdir(parents=True)
    (models_dir / "train_log.jsonl").write_text(
        json.dumps({"loss": None, "iter": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        smoke.verify_training_fired(report_dir)
    assert "0 SGD rows" in str(excinfo.value)


def test_smoke_verify_training_fired_passes_with_sgd(tmp_path):
    """P3: verify_training_fired returns counts when SGD fired."""
    smoke = load_smoke_module()
    report_dir = tmp_path / "gomoku7"
    models_dir = report_dir / "models" / "T1_noS_noVL"
    models_dir.mkdir(parents=True)
    (models_dir / "train_log.jsonl").write_text(
        json.dumps({"loss": 0.42, "iter": 1}) + "\n",
        encoding="utf-8",
    )

    total, scanned = smoke.verify_training_fired(report_dir)

    assert total == 1
    assert len(scanned) == 1


def test_smoke_e2e_builds_safe_runtime_ablation_command_by_default(tmp_path):
    smoke = load_smoke_module()
    args = argparse.Namespace(
        study="search_vl",
        conditions="T1_noS_noVL",
        eval_conditions="E1_noS_noVL",
        game="gomoku7",
        iterations=1,
        games_per_iter=4,
        eval_games=2,
        eval_interval=999999,
        seed=11,
        timeout_hours=1,
        no_autotune=True,
        include_strict_reference=False,
        safe_runtime=True,
        resident_session=None,
    )

    command = smoke.build_ablation_command(
        args, tmp_path / "mcts_demo", tmp_path / "out"
    )

    assert "--resident-session" not in command
    assert "--no-autotune" in command


def test_smoke_e2e_summary_includes_log_artifacts(tmp_path):
    smoke = load_smoke_module()
    output_root = tmp_path / "smoke"
    output_root.mkdir(parents=True)
    artifacts = smoke.smoke_artifact_paths(output_root)
    artifacts["events_jsonl"].write_text(
        '{"event":"command_begin"}\n{"event":"command_end"}\n', encoding="utf-8"
    )
    artifacts["python_trace_jsonl"].write_text(
        '{"event":"exchange_begin"}\n', encoding="utf-8"
    )
    artifacts["rust_server_trace_jsonl"].write_text(
        '{"event":"rust_server_ready"}\n', encoding="utf-8"
    )
    args = argparse.Namespace(game="gomoku7", study="search_vl")

    summary = smoke.build_smoke_summary(
        args=args,
        output_root=output_root,
        artifact_paths=artifacts,
        rust_binary=tmp_path / "mcts_demo",
        success=False,
        missing_outputs=["missing.json"],
        error="boom",
    )

    assert summary["success"] is False
    assert summary["artifacts"]["stdout_log"].endswith("stdout.log")
    assert summary["trace_counts"]["events"] == 2
    assert summary["trace_counts"]["python_trace"] == 1
    assert summary["trace_counts"]["rust_server_trace"] == 1
