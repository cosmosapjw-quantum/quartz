import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quartz.prior_revision import (
    PriorRevisionSystem,
    adaptive_challenger_k,
    apply_prior_corruption,
    apply_revision_operator,
    classify_position_buckets,
    load_systems_config,
    make_default_systems,
    normalize_policy,
    policy_argmax,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_prior_revision_runner():
    root = Path(__file__).resolve().parents[1]
    return load_module("prior_revision_runner_script", root / "scripts" / "prior_revision_experiment.py")


def test_make_default_systems_exposes_b0_b1_n1_n2():
    systems = make_default_systems({"prior_refresh_temp": 1.0})
    assert [system.id for system in systems] == ["B0", "B1", "N1", "N2"]


def test_apply_prior_corruption_produces_wrong_top1_with_oracle():
    prior = normalize_policy(np.array([0.70, 0.20, 0.10], dtype=np.float32))
    corrupted = apply_prior_corruption(prior, "swap_top12", oracle_best=0)
    assert policy_argmax(corrupted) != 0

    corrupted = apply_prior_corruption(prior, "inflate_wrong_confidence", oracle_best=0, strength=2.0)
    assert policy_argmax(corrupted) != 0


def test_dual_channel_refresh_moves_toward_search_posterior():
    system = PriorRevisionSystem(
        id="N1",
        label="dual",
        operator="dual_channel",
        params={"gate_epsilon": 0.01, "gate_scale": 1.0, "posterior_mix": 1.0},
    )
    prior = normalize_policy(np.array([0.80, 0.15, 0.05], dtype=np.float32))
    search = normalize_policy(np.array([0.10, 0.75, 0.15], dtype=np.float32))

    effective, meta = apply_revision_operator(system, prior, search, budget=16)

    assert policy_argmax(effective) == 1
    assert meta["dual_gate"] > 0.0


def test_root_snapshot_limits_update_to_candidate_set():
    system = PriorRevisionSystem(
        id="N2",
        label="root",
        operator="root_snapshot",
        params={"challenger_k": 2, "adaptive_k": False, "snapshot_alpha": 0.25},
    )
    prior = normalize_policy(np.array([0.60, 0.25, 0.10, 0.05], dtype=np.float32))
    search = normalize_policy(np.array([0.05, 0.20, 0.70, 0.05], dtype=np.float32))

    effective, meta = apply_revision_operator(system, prior, search, budget=16)

    assert len(meta["root_candidate_set"]) == 2
    assert policy_argmax(effective) in meta["root_candidate_set"]


def test_classify_position_buckets_marks_wrong_and_late_evidence_cases():
    prior = normalize_policy(np.array([0.65, 0.20, 0.15], dtype=np.float32))
    low = normalize_policy(np.array([0.60, 0.25, 0.15], dtype=np.float32))
    oracle = normalize_policy(np.array([0.10, 0.75, 0.15], dtype=np.float32))

    buckets = classify_position_buckets(prior, low, oracle)

    assert "wrong_top1" in buckets
    assert "wrong_confident" in buckets
    assert "late_evidence" in buckets


def test_load_systems_config_accepts_explicit_json(tmp_path: Path):
    path = tmp_path / "systems.json"
    path.write_text(
        json.dumps(
            {
                "systems": [
                    {
                        "id": "B0",
                        "label": "baseline",
                        "operator": "search",
                        "search_overrides": {"prior_refresh_rate": 0.0},
                        "params": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    systems = load_systems_config(str(path), {"prior_refresh_temp": 1.0})

    assert len(systems) == 1
    assert systems[0].id == "B0"


def test_adaptive_challenger_k_is_budget_sensitive():
    assert adaptive_challenger_k(6, 4, adaptive=True) == 2
    assert adaptive_challenger_k(6, 64, adaptive=True) == 6


def test_validate_checkpoint_refs_rejects_truncated_lexical_directory_selection(tmp_path: Path):
    runner = load_prior_revision_runner()
    root = tmp_path / "models"
    for family in ("F1_legacy_base", "F2_legacy_krefresh", "F3_theory_base", "F4_theory_krefresh"):
        for seed in ("seed_41", "seed_42", "seed_43"):
            path = root / family / seed
            path.mkdir(parents=True, exist_ok=True)
            (path / "best.pt").write_bytes(b"x")

    refs = [
        runner.CheckpointRef(id="C01_best", path=str(root / "F1_legacy_base" / "seed_41" / "best.pt")),
        runner.CheckpointRef(id="C02_best", path=str(root / "F1_legacy_base" / "seed_42" / "best.pt")),
        runner.CheckpointRef(id="C03_best", path=str(root / "F1_legacy_base" / "seed_43" / "best.pt")),
    ]
    args = SimpleNamespace(checkpoints=None, checkpoint_dir=str(root))

    try:
        runner.validate_checkpoint_refs(args, refs)
    except ValueError as exc:
        assert "curated checkpoint selection" in str(exc)
    else:
        raise AssertionError("expected lexical truncation to be rejected")


def test_validate_checkpoint_refs_accepts_explicit_curated_checkpoint_files(tmp_path: Path):
    runner = load_prior_revision_runner()
    root = tmp_path / "models"
    refs = []
    for idx, family in enumerate(("F1_legacy_base", "F2_legacy_krefresh", "F3_theory_base"), start=1):
        path = root / family / "seed_41"
        path.mkdir(parents=True, exist_ok=True)
        ckpt = path / "best.pt"
        ckpt.write_bytes(b"x")
        refs.append(runner.CheckpointRef(id=f"C{idx:02d}_best", path=str(ckpt)))

    args = SimpleNamespace(
        checkpoints=",".join(ref.path for ref in refs),
        checkpoint_dir=None,
    )

    runner.validate_checkpoint_refs(args, refs)
