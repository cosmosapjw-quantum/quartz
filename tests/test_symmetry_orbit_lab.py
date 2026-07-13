"""Tests for symmetry_orbit_lab — the game-agnostic operator audit."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from quartz.experiments import symmetry_orbit as so

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_permute_policy_moves_values_to_new_indices():
    v = [0.1, 0.2, 0.7]
    perm = [2, 0, 1]  # old i -> new perm[i]
    w = so.permute_policy(v, perm)
    # value at old index i must land at new index perm[i]
    assert w[2] == pytest.approx(0.1)
    assert w[0] == pytest.approx(0.2)
    assert w[1] == pytest.approx(0.7)


def test_argmax_equivariant_true_and_negative_case():
    v = [0.1, 0.6, 0.3]
    perm = [1, 2, 0]
    assert so.argmax_equivariant(v, perm) is True
    # a bogus "permutation" that is really the identity output would break
    # equivariance for a non-identity perm on the raw index — sanity via the
    # negative-control operator instead:
    changed = so._mass_on_action_zero(so.permute_policy(v, perm)) != so._mass_on_action_zero(v)
    assert changed is True


def test_dihedral_group_is_eight_bijections_with_rot90_order_four():
    side = 5
    group = dict(so.dihedral_group(side))
    assert len(group) == 8
    n = side * side
    for name, perm in group.items():
        assert sorted(perm) == list(range(n)), f"{name} is not a bijection"
    # rot90 applied four times is the identity
    rot90 = group["rot90"]
    p = list(range(n))
    for _ in range(4):
        p = [rot90[i] for i in p]
    assert p == list(range(n))


def test_permute_trace_bundle_consistent_relabel():
    rng = np.random.default_rng(1)
    bundle = so.random_trace_bundle(rng, 6)
    perm = so.action_permutation(6, rng)
    moved = so.permute_trace_bundle(bundle, perm)
    assert len(moved["trace_policies"]) == len(bundle["trace_policies"])
    for orig, new in zip(bundle["trace_policies"], moved["trace_policies"]):
        assert np.allclose(so.permute_policy(orig, perm), new)


def test_audit_upholds_constraint_and_catches_negative_controls():
    r = so.audit(seed=20260713, n_trials=96)
    assert r["game_agnostic_constraint_upheld"] is True
    assert r["n_operator_violations"] == 0
    assert r["negative_controls_all_caught"] is True
    # every real operator obeys its transform law
    for row in r["operators"] + r["clone_robustness"]:
        assert row["invariant_within_eps"] is True
    # every negative control is flagged
    for row in r["negative_controls"]:
        assert row["flagged_as_violation"] is True


def test_audit_covers_expected_operators_and_channels():
    r = so.audit(seed=7, n_trials=48)
    names = {row["operator"] for row in r["operators"]}
    channels = {row["channel"] for row in r["operators"]}
    assert {"policy_entropy", "k_eff", "top2_margin", "forked_voc.voc_proxy", "voc_tightness"} <= names
    assert {"action_permutation", "dihedral_d4", "trace_bundle_permutation", "move_order_permutation"} <= channels
    # the committed move is audited as equivariant, not invariant
    argmax_rows = [row for row in r["operators"] if row["operator"] == "argmax(committed_move)"]
    assert argmax_rows and all(row["transform_law"] == "equivariant" for row in argmax_rows)


def _load_runner():
    path = REPO_ROOT / "scripts" / "symmetry_orbit_lab.py"
    spec = importlib.util.spec_from_file_location("symmetry_orbit_lab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_loads():
    runner = _load_runner()
    cfg = runner.load_config(runner.DEFAULT_CONFIG)
    assert cfg["experiment_id"] == so.EXPERIMENT_ID
    assert cfg["prohibited_inferences"]


def test_runner_smoke_writes_artifacts(tmp_path):
    runner = _load_runner()
    out = tmp_path / "run"
    rc = runner.main(["--n-trials", "24", "--output-dir", str(out)])
    assert rc == 0
    for name in ("run_manifest.json", "operators.csv", "summary.json"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["audit"]["game_agnostic_constraint_upheld"] is True
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["prohibited_inferences"]
