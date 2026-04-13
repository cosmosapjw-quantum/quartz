import importlib
import importlib.util
import json
import sys
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


def load_gomocup_export_module():
    return importlib.import_module("quartz.gomocup_export")


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


def test_select_champion_prefers_eval_leader_and_best_deployment_profile(tmp_path):
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
    assert champion["deployment_eval_condition"] == "E4_S_VL"
    assert champion["deployment_search_cfg"]["search_profile"] == "quartz"
    assert champion["deployment_search_cfg"]["vl_mode"] == "adaptive"


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


def test_run_evaluation_matrix_can_limit_to_paired_seed_comparisons(tmp_path, monkeypatch):
    ablation = load_ablation_module()
    import quartz.alphazero_train as az

    base_dir = tmp_path / "results" / "gomoku7"
    base_dir.mkdir(parents=True, exist_ok=True)
    called_pairs = []

    def fake_arena(model_a, model_b, cfg, device, n_games, rust_binary, strict):
        called_pairs.append((Path(model_a).name, Path(model_b).name))
        return (2, 2, 0, 0.5, (0.1, 0.9), "inconclusive")

    monkeypatch.setattr(az, "arena_rust_nn", fake_arena)

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
