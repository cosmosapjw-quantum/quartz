import importlib.util
import sys
from pathlib import Path

import pytest

optuna = pytest.importorskip("optuna")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_controller_optuna_module():
    root = Path(__file__).resolve().parents[1]
    return load_module("controller_optuna_script", root / "scripts" / "controller_optuna.py")


def test_params_to_candidate_disables_refresh_cleanly():
    mod = load_controller_optuna_module()
    base_cfg = {
        "prior_refresh_temp": 1.0,
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
    }
    params = {
        "controller_family": "legacy",
        "root_only_shaping": False,
        "refresh_enabled": False,
        "prior_refresh_rate": 0.0,
        "prior_refresh_temp": 0.0,
        "hbar_penalty_cap": 0.25,
        "sigma_0": 0.2,
        "min_visits": 12,
        "check_interval": 15,
        "c_puct": 1.75,
    }

    candidate = mod.params_to_candidate(params, base_cfg, trial_number=7)

    assert candidate["id"].startswith("T0007_")
    assert candidate["overrides"]["penalty_mode"] == "GatedRefreshLegacy"
    assert candidate["overrides"]["prior_refresh_rate"] == 0.0
    assert candidate["overrides"]["prior_refresh_temp"] == 1.0


def test_anchor_candidate_to_params_roundtrips_refresh_flag():
    mod = load_controller_optuna_module()
    base_cfg = {
        "prior_refresh_temp": 1.0,
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
    }
    anchor = {
        "id": "A2_legacy_krefresh",
        "label": "legacy+refresh",
        "source": "anchor",
        "overrides": {
            "penalty_mode": "GatedRefreshLegacy",
            "root_only_shaping": False,
            "prior_refresh_rate": 0.5,
            "prior_refresh_temp": 0.0,
            "hbar_penalty_cap": 0.3,
            "sigma_0": 0.3,
            "min_visits": 15,
            "check_interval": 20,
            "c_puct": 2.0,
        },
    }

    params = mod.anchor_candidate_to_params(anchor, base_cfg)

    assert params["controller_family"] == "legacy"
    assert params["refresh_enabled"] is True
    assert params["prior_refresh_rate"] == 0.5
    assert params["prior_refresh_temp"] == 0.0


def test_select_top_trial_candidates_dedupes_equivalent_configs():
    mod = load_controller_optuna_module()
    base_cfg = {
        "prior_refresh_temp": 1.0,
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
    }
    rows = [
        {
            "number": 0,
            "state": "COMPLETE",
            "value": 0.40,
            "params": {
                "controller_family": "legacy",
                "root_only_shaping": False,
                "refresh_enabled": False,
                "prior_refresh_rate": 0.0,
                "prior_refresh_temp": 1.0,
                "hbar_penalty_cap": 0.3,
                "sigma_0": 0.3,
                "min_visits": 15,
                "check_interval": 20,
                "c_puct": 2.0,
            },
        },
        {
            "number": 1,
            "state": "COMPLETE",
            "value": 0.35,
            "params": {
                "controller_family": "legacy",
                "root_only_shaping": False,
                "refresh_enabled": False,
                "prior_refresh_rate": 0.0,
                "prior_refresh_temp": 1.0,
                "hbar_penalty_cap": 0.3,
                "sigma_0": 0.3,
                "min_visits": 15,
                "check_interval": 20,
                "c_puct": 2.0,
            },
        },
        {
            "number": 2,
            "state": "COMPLETE",
            "value": 0.33,
            "params": {
                "controller_family": "theory",
                "root_only_shaping": True,
                "refresh_enabled": True,
                "prior_refresh_rate": 0.5,
                "prior_refresh_temp": 0.0,
                "hbar_penalty_cap": 0.3,
                "sigma_0": 0.3,
                "min_visits": 15,
                "check_interval": 20,
                "c_puct": 2.0,
            },
        },
    ]

    selected = mod.select_top_trial_candidates(rows, base_cfg, topk=3)

    assert len(selected) == 2
    assert selected[0]["overrides"]["penalty_mode"] == "GatedRefreshLegacy"
    assert selected[1]["overrides"]["penalty_mode"] == "GatedRefresh"


def test_summarize_trials_by_segment_counts_refresh_and_family():
    mod = load_controller_optuna_module()
    rows = [
        {
            "number": 0,
            "state": "COMPLETE",
            "value": 0.50,
            "params": {"controller_family": "legacy", "refresh_enabled": False},
        },
        {
            "number": 1,
            "state": "COMPLETE",
            "value": 0.45,
            "params": {"controller_family": "legacy", "refresh_enabled": True},
        },
        {
            "number": 2,
            "state": "COMPLETE",
            "value": 0.55,
            "params": {"controller_family": "theory", "refresh_enabled": True},
        },
        {
            "number": 3,
            "state": "PRUNED",
            "value": None,
            "params": {"controller_family": "theory", "refresh_enabled": False},
        },
    ]

    summary = mod.summarize_trials_by_segment(rows)

    assert summary["refresh"]["off"]["count"] == 1
    assert summary["refresh"]["on"]["count"] == 2
    assert summary["family"]["legacy"]["count"] == 2
    assert summary["family"]["theory"]["count"] == 1


def test_sample_trial_params_respects_requested_family():
    mod = load_controller_optuna_module()
    base_cfg = {
        "prior_refresh_temp": 1.0,
        "hbar_penalty_cap": 0.3,
        "sigma_0": 0.3,
        "min_visits": 15,
        "check_interval": 20,
        "c_puct": 2.0,
    }
    trial = optuna.trial.FixedTrial(
        {
            "controller_family": "legacy",
            "refresh_enabled": False,
            "root_only_shaping": False,
            "hbar_penalty_cap": 0.3,
            "sigma_0": 0.3,
            "min_visits": 15,
            "check_interval": 20,
            "c_puct": 2.0,
        }
    )

    params = mod.sample_trial_params(trial, base_cfg, ["legacy"])

    assert params["controller_family"] == "legacy"
    assert params["refresh_enabled"] is False
    assert params["prior_refresh_rate"] == 0.0
