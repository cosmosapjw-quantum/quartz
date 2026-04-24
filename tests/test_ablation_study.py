import importlib
import importlib.util
import argparse
import json
import sys
import types
import tomllib
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_ablation_module():
    root = Path(__file__).resolve().parents[1]
    return load_module("ablation_study_script", root / "scripts" / "ablation_study.py")


def load_smoke_module():
    root = Path(__file__).resolve().parents[1]
    return load_module("smoke_e2e_script", root / "scripts" / "smoke_e2e.py")


def load_gomocup_export_module():
    return importlib.import_module("quartz.gomocup_export")


def load_audit_bundle_script():
    root = Path(__file__).resolve().parents[1]
    return load_module("build_audit_bundle_script", root / "scripts" / "build_audit_bundle.py")


def write_condition_run(run_dir: Path, condition: str, seed: int | None, elo: float, loss: float) -> None:
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
        {"_type": "eval", "published_elo": elo, "eval_verdict": "promote", "score_rate": 0.61},
    ]
    (run_dir / "train_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_discover_model_runs_handles_flat_and_seeded_layout(tmp_path):
    ablation = load_ablation_module()
    base_dir = tmp_path / "results" / "gomoku15"

    write_condition_run(base_dir / "models" / "T1_noS_noVL", "T1_noS_noVL", None, 1510.0, 1.2)
    write_condition_run(base_dir / "models" / "T4_S_VL" / "seed_41", "T4_S_VL", 41, 1640.0, 0.9)
    write_condition_run(base_dir / "models" / "T4_S_VL" / "seed_42", "T4_S_VL", 42, 1655.0, 0.8)

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
    assert manifest["train_conditions"]["C1_impl_legacy"]["penalty_mode"] == "GatedRefreshLegacy"
    assert manifest["eval_conditions"]["E2_theory_doc"]["root_only_shaping"] is True
    assert manifest["train_condition_surfaces"]["C1_impl_legacy"]["penalty_mode"] == "GatedRefreshLegacy"


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
    assert manifest["eval_conditions_selected"] == ["E1_legacy_base", "E4_theory_krefresh"]
    assert manifest["train_conditions"]["F2_legacy_krefresh"]["prior_refresh_rate"] == 0.5
    assert set(manifest["eval_conditions"]) == {"E1_legacy_base", "E4_theory_krefresh"}
    assert manifest["eval_conditions"]["E4_theory_krefresh"]["prior_refresh_temp"] == 0.0


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
    assert manifest["train_conditions"]["A2_legacy_root_norefresh"]["root_only_shaping"] is True
    assert manifest["train_conditions"]["A3_theory_root_norefresh"]["penalty_mode"] == "GatedRefresh"
    assert manifest["train_conditions"]["A4_theory_root_refresh"]["prior_refresh_rate"] == 0.5
    assert manifest["train_condition_surfaces"]["A4_theory_root_refresh"]["prior_refresh_rate"] == 0.5
    assert manifest["runtime_contract"]["config_layout"] == "repo_top_level_configs"
    assert isinstance(manifest["runtime_contract_hash"], str)
    assert len(manifest["runtime_contract_hash"]) == 16


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


def test_select_champion_prefers_eval_leader_and_train_cfg_for_deployment(tmp_path):
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
    champion = ablation.select_champion(base_dir, model_runs, eval_payload)

    assert champion["model_id"] == "T4_S_VL_s42"
    assert champion["deployment_eval_condition"] is None
    assert champion["deployment_search_cfg"]["search_profile"] == "quartz"
    assert champion["deployment_search_cfg"]["vl_mode"] == "adaptive"
    assert champion["deployment_cfg_source"] == "train_cfg"


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
    assert (stage_dir / "scripts" / "smoke_e2e.py").exists()
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

    monkeypatch.setattr(ablation, "build_eval_cfg", lambda *args, **kwargs: ({"_name": "gomoku7"}, "cpu"))

    import quartz.runtime_support as support_mod
    import quartz.evaluator_runtime as eval_mod

    monkeypatch.setattr(
        support_mod,
        "load_actor_source_from_checkpoint",
        lambda model_path, engine_cfg, device, backend_preference=None: captured.setdefault("backend_preference", backend_preference) or object(),
    )
    monkeypatch.setattr(eval_mod, "RustNNEvaluatorEngine", FakeEngine)

    args = argparse.Namespace(game="gomoku7", device="cpu", rust_binary="./target/release/mcts_demo")
    ablation.build_eval_engine({"id": "m1", "model_path": "best.pt"}, args, {"search_profile": "baseline", "vl_mode": "disabled"}, "cpu")

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
        "scripts/phase15_ablation_study.py",
        "scripts/phase15_online_ablation.py",
        "scripts/phase15_benchmark.py",
        "scripts/phase15_mine_suite.py",
        "configs/phase15_systems.default.json",
        "docs/QUARTZ_THEORY.md",
        "docs/ABLATION_GUIDE.md",
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


def test_run_evaluation_matrix_can_limit_to_paired_seed_comparisons(tmp_path, monkeypatch):
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
            return types.SimpleNamespace(wins=2, losses=2, draws=0, errors=0, voids=0, scored=4, total=4, score_rate=0.5), {
                "runner_mode": "rust_eval_state_machine",
                "match_elapsed_s": 0.01,
            }

    monkeypatch.setattr(ablation, "build_eval_cfg", fake_build_eval_cfg)
    monkeypatch.setattr(ablation, "build_eval_engine", lambda model_run, args, eval_cfg, device: (FakeEngine(model_run["id"]), {}))
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

    model_runs = [
        {"id": "C1_impl_legacy_s41", "condition": "C1_impl_legacy", "seed": 41, "success": True, "model_path": str(base_dir / "a41.pt")},
        {"id": "C2_theory_doc_s41", "condition": "C2_theory_doc", "seed": 41, "success": True, "model_path": str(base_dir / "b41.pt")},
        {"id": "C1_impl_legacy_s42", "condition": "C1_impl_legacy", "seed": 42, "success": True, "model_path": str(base_dir / "a42.pt")},
        {"id": "C2_theory_doc_s42", "condition": "C2_theory_doc", "seed": 42, "success": True, "model_path": str(base_dir / "b42.pt")},
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
    legacy_train = next(row for row in summary["training"] if row["condition"] == "F1_legacy_base")
    assert legacy_train["mean_elo"] == 15.0
    assert legacy_train["mean_score_rate"] == 0.5
    assert legacy_train["mean_loss"] == 4.1
    legacy_eval = next(row for row in summary["evaluation"] if row["condition"] == "F1_legacy_base")
    assert legacy_eval["entries"] == 2
    assert legacy_eval["score_rate"] == 0.5


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

    command = smoke.build_ablation_command(args, tmp_path / "mcts_demo", tmp_path / "out")

    assert "--resident-session" not in command
    assert "--no-autotune" in command


def test_smoke_e2e_summary_includes_log_artifacts(tmp_path):
    smoke = load_smoke_module()
    output_root = tmp_path / "smoke"
    output_root.mkdir(parents=True)
    artifacts = smoke.smoke_artifact_paths(output_root)
    artifacts["events_jsonl"].write_text('{"event":"command_begin"}\n{"event":"command_end"}\n', encoding="utf-8")
    artifacts["python_trace_jsonl"].write_text('{"event":"exchange_begin"}\n', encoding="utf-8")
    artifacts["rust_server_trace_jsonl"].write_text('{"event":"rust_server_ready"}\n', encoding="utf-8")
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
