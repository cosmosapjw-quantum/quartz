#!/usr/bin/env python3
"""symmetry_orbit_lab — empirical evidence for the game-agnostic constraint.

Part of the metacognitive experiment family (see
``docs/METACOGNITIVE_EXPERIMENTS.md``). Unlike the Bernoulli / morphology labs
this is a **diagnostic**, not a screen with a kill-criterion: it audits the
project's own signature operators against the FORBIDDEN "game-agnostic"
constraint asserted by ``quartz/phase15_signatures.py`` ("No game rules, board
topology, or move semantics are used anywhere").

The test is behavioral. A genuinely game-agnostic scalar readout must be
**invariant** under an arbitrary relabeling (permutation) of the action axis;
an index-valued readout (the committed move) must be **equivariant** — its
output must move with the permutation. If an operator secretly depends on which
concrete action a label points at, its output will change under a permutation
that a truly agnostic operator would ignore, and the audit flags it.

Four symmetry channels are exercised:

* **action permutation** on single root policies (``policy_entropy``, ``k_eff``
  invariant; the committed argmax equivariant);
* **dihedral D4** board-cell permutations on a square board (the concrete
  board-symmetry group — the same invariance under a structured subgroup);
* **consistent trace-bundle permutation** (a single relabeling applied to every
  policy in a budget ladder): ``forked_voc`` VOC proxy, argmax-flip count, and
  the O5 revision signatures must be invariant;
* **move-order permutation** for the cross-move O2 dispersion operators
  (``budget_gini`` / ``budget_entropy``) and, jointly, ``voc_tightness``.

**Clone robustness**: appending a zero-mass action (a never-visited clone) must
leave scalar readouts unchanged. **Negative controls**: deliberately
index-dependent probes are included and MUST be flagged — a harness that only
ever passes is vacuous.

Prohibited (claim firewall): reading a clean audit as proof of play strength or
of full game-independence of the *engine* (only the audited Python readouts are
covered); reading a flag as a bug without inspecting the operator's intended
transform law.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from quartz.experiments.forked_voc import label_trace_bundle, top2_margin, voc_proxy
from quartz.phase15_signatures import (
    budget_entropy,
    budget_gini,
    first_revision_step,
    flip_flop_rate,
    k_eff,
    policy_entropy,
    voc_tightness,
)

SYMMETRY_ORBIT_SCHEMA_VERSION = 1
EXPERIMENT_ID = "symmetry_orbit_lab_v1"
EXECUTION_MODE = "synthetic_screening"

_DEFAULT_EPS = 1e-9


# --------------------------------------------------------------------------- #
# permutation / group machinery
# --------------------------------------------------------------------------- #

def action_permutation(n: int, rng: np.random.Generator) -> List[int]:
    return [int(i) for i in rng.permutation(n)]


def permute_policy(vec: Sequence[float], perm: Sequence[int]) -> np.ndarray:
    """Relabel actions by ``perm``: the value at old index ``i`` moves to new
    index ``perm[i]`` (``w[perm[i]] = v[i]``)."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    p = np.asarray(perm, dtype=np.int64)
    if v.size != p.size:
        raise ValueError("policy and permutation length mismatch")
    w = np.empty_like(v)
    w[p] = v
    return w


def argmax_equivariant(vec: Sequence[float], perm: Sequence[int]) -> bool:
    """Does the committed argmax move with the permutation? i.e.
    argmax(permute(v, perm)) == perm[argmax(v)]."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return True
    a = int(np.argmax(v))
    return int(np.argmax(permute_policy(v, perm))) == int(perm[a])


def dihedral_group(side: int) -> List[Tuple[str, List[int]]]:
    """The eight D4 cell permutations of an ``side x side`` row-major board.

    Each entry maps old cell index -> new cell index, usable directly with
    ``permute_policy`` on a policy defined over board cells."""
    def idx(r: int, c: int) -> int:
        return r * side + c

    transforms: Dict[str, Callable[[int, int], Tuple[int, int]]] = {
        "identity": lambda r, c: (r, c),
        "rot90": lambda r, c: (c, side - 1 - r),
        "rot180": lambda r, c: (side - 1 - r, side - 1 - c),
        "rot270": lambda r, c: (side - 1 - c, r),
        "flip_h": lambda r, c: (r, side - 1 - c),
        "flip_v": lambda r, c: (side - 1 - r, c),
        "transpose": lambda r, c: (c, r),
        "anti_transpose": lambda r, c: (side - 1 - c, side - 1 - r),
    }
    group: List[Tuple[str, List[int]]] = []
    for name, fn in transforms.items():
        perm = [0] * (side * side)
        for r in range(side):
            for c in range(side):
                nr, nc = fn(r, c)
                perm[idx(r, c)] = idx(nr, nc)
        group.append((name, perm))
    return group


def permute_trace_bundle(bundle: Dict[str, Any], perm: Sequence[int]) -> Dict[str, Any]:
    """Apply one action relabeling consistently to every policy in a bundle."""
    return {
        "trace_budgets": list(bundle.get("trace_budgets", [])),
        "trace_policies": [permute_policy(p, perm) for p in bundle.get("trace_policies", [])],
    }


# --------------------------------------------------------------------------- #
# synthetic inputs
# --------------------------------------------------------------------------- #

def random_policy(rng: np.random.Generator, n: int, *, zero_frac: float = 0.0) -> np.ndarray:
    p = rng.dirichlet(np.ones(n) * 0.7)
    if zero_frac > 0.0:
        mask = rng.random(n) < zero_frac
        if mask.all():
            mask[int(rng.integers(n))] = False
        p = p * (~mask)
        s = p.sum()
        if s > 0:
            p = p / s
    return p


def random_trace_bundle(rng: np.random.Generator, n_actions: int, ladder=(8, 16, 32, 64)) -> Dict[str, Any]:
    return {
        "trace_budgets": list(ladder),
        "trace_policies": [random_policy(rng, n_actions) for _ in ladder],
    }


# --------------------------------------------------------------------------- #
# operator registry (name, callable, transform law)
# --------------------------------------------------------------------------- #

# scalar readouts over a single policy; must be INVARIANT under action relabeling
POLICY_SCALAR_OPS: Tuple[Tuple[str, Callable[[Any], float]], ...] = (
    ("policy_entropy", policy_entropy),
    ("k_eff", k_eff),
    ("top2_margin", top2_margin),
)

# scalar readouts over a whole budget ladder; INVARIANT under one consistent
# action relabeling of every policy in the ladder
def _voc_proxy_bundle(bundle: Dict[str, Any]) -> float:
    return voc_proxy(bundle["trace_budgets"], bundle["trace_policies"])


def _n_flips_bundle(bundle: Dict[str, Any]) -> float:
    return float(label_trace_bundle(bundle)["n_argmax_flips"])


def _first_revision_bundle(bundle: Dict[str, Any]) -> float:
    step = first_revision_step(bundle["trace_policies"])
    return -1.0 if step is None else float(step)


def _flip_flop_bundle(bundle: Dict[str, Any]) -> float:
    return flip_flop_rate(bundle["trace_policies"])


BUNDLE_SCALAR_OPS: Tuple[Tuple[str, Callable[[Dict[str, Any]], float]], ...] = (
    ("forked_voc.voc_proxy", _voc_proxy_bundle),
    ("forked_voc.n_argmax_flips", _n_flips_bundle),
    ("first_revision_step", _first_revision_bundle),
    ("flip_flop_rate", _flip_flop_bundle),
)

# cross-move dispersion operators over a per-move series; INVARIANT under
# permuting the MOVE order (symmetric functions of the series)
MOVE_SERIES_OPS: Tuple[Tuple[str, Callable[[Sequence[float]], float]], ...] = (
    ("budget_gini", budget_gini),
    ("budget_entropy", budget_entropy),
)


# negative controls: deliberately index-dependent probes that MUST be flagged
def _mass_on_action_zero(pol: Any) -> float:
    p = np.asarray(pol, dtype=np.float64).ravel()
    return float(p[0]) if p.size else 0.0


def _raw_argmax_index(pol: Any) -> float:
    p = np.asarray(pol, dtype=np.float64).ravel()
    return float(np.argmax(p)) if p.size else -1.0


NEGATIVE_CONTROL_OPS: Tuple[Tuple[str, Callable[[Any], float]], ...] = (
    ("neg_control:mass_on_action_zero", _mass_on_action_zero),
    ("neg_control:raw_argmax_index_as_scalar", _raw_argmax_index),
)


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #

def _invariance_verdict(name: str, channel: str, law: str, defects: List[float], eps: float) -> Dict[str, Any]:
    max_defect = float(max(defects)) if defects else 0.0
    return {
        "operator": name,
        "channel": channel,
        "transform_law": law,
        "n_trials": len(defects),
        "max_abs_defect": max_defect,
        "mean_abs_defect": float(np.mean(defects)) if defects else 0.0,
        "invariant_within_eps": bool(max_defect <= eps),
    }


def audit(
    *,
    seed: int = 0,
    n_trials: int = 128,
    n_actions: int = 12,
    board_side: int = 5,
    eps: float = _DEFAULT_EPS,
) -> Dict[str, Any]:
    """Run the full symmetry audit and return per-operator verdicts plus the
    overall game-agnostic-constraint verdict."""
    rng = np.random.default_rng(seed)
    per_operator: List[Dict[str, Any]] = []

    # 1. action permutation on single policies
    for name, op in POLICY_SCALAR_OPS:
        defects = []
        for _ in range(n_trials):
            pol = random_policy(rng, n_actions, zero_frac=0.2)
            perm = action_permutation(n_actions, rng)
            defects.append(abs(float(op(permute_policy(pol, perm))) - float(op(pol))))
        per_operator.append(_invariance_verdict(name, "action_permutation", "invariant", defects, eps))

    # committed argmax: equivariant (index moves with the permutation)
    equi_ok = []
    for _ in range(n_trials):
        pol = random_policy(rng, n_actions)
        perm = action_permutation(n_actions, rng)
        equi_ok.append(argmax_equivariant(pol, perm))
    per_operator.append(
        {
            "operator": "argmax(committed_move)",
            "channel": "action_permutation",
            "transform_law": "equivariant",
            "n_trials": len(equi_ok),
            "equivariance_holds": bool(all(equi_ok)),
            "n_equivariance_failures": int(sum(1 for ok in equi_ok if not ok)),
            "invariant_within_eps": bool(all(equi_ok)),
        }
    )

    # 2. dihedral D4 board-cell permutations (invariant scalar ops + equivariant argmax)
    group = dihedral_group(board_side)
    n_cells = board_side * board_side
    for name, op in POLICY_SCALAR_OPS:
        defects = []
        for _ in range(max(1, n_trials // 4)):
            pol = random_policy(rng, n_cells, zero_frac=0.2)
            for _gname, perm in group:
                defects.append(abs(float(op(permute_policy(pol, perm))) - float(op(pol))))
        per_operator.append(_invariance_verdict(name, "dihedral_d4", "invariant", defects, eps))
    d4_equi_ok = []
    for _ in range(max(1, n_trials // 4)):
        pol = random_policy(rng, n_cells)
        for _gname, perm in group:
            d4_equi_ok.append(argmax_equivariant(pol, perm))
    per_operator.append(
        {
            "operator": "argmax(committed_move)",
            "channel": "dihedral_d4",
            "transform_law": "equivariant",
            "n_trials": len(d4_equi_ok),
            "equivariance_holds": bool(all(d4_equi_ok)),
            "n_equivariance_failures": int(sum(1 for ok in d4_equi_ok if not ok)),
            "invariant_within_eps": bool(all(d4_equi_ok)),
        }
    )

    # 3. consistent trace-bundle relabeling
    for name, op in BUNDLE_SCALAR_OPS:
        defects = []
        for _ in range(n_trials):
            bundle = random_trace_bundle(rng, n_actions)
            perm = action_permutation(n_actions, rng)
            defects.append(abs(float(op(permute_trace_bundle(bundle, perm))) - float(op(bundle))))
        per_operator.append(_invariance_verdict(name, "trace_bundle_permutation", "invariant", defects, eps))

    # 4. move-order permutation for cross-move dispersion ops
    for name, op in MOVE_SERIES_OPS:
        defects = []
        for _ in range(n_trials):
            series = rng.integers(0, 64, size=n_actions).astype(np.float64)
            perm = action_permutation(n_actions, rng)
            reordered = series[perm]
            defects.append(abs(float(op(reordered.tolist())) - float(op(series.tolist()))))
        per_operator.append(_invariance_verdict(name, "move_order_permutation", "invariant", defects, eps))

    # 4b. voc_tightness: invariant under JOINT move-order permutation of the
    # (per_move_budget, voc_proxy) pairs (a correlation is order-free).
    tight_defects = []
    for _ in range(n_trials):
        budgets = rng.integers(1, 64, size=n_actions).astype(np.float64)
        proxies = rng.random(n_actions)
        perm = action_permutation(n_actions, rng)
        base = voc_tightness(budgets.tolist(), proxies.tolist())
        moved = voc_tightness(budgets[perm].tolist(), proxies[perm].tolist())
        if base is None or moved is None:
            continue
        tight_defects.append(abs(float(moved) - float(base)))
    per_operator.append(
        _invariance_verdict("voc_tightness", "joint_move_order_permutation", "invariant", tight_defects, eps)
    )

    # 5. clone robustness: zero-mass clone must not change scalar readouts
    clone_results: List[Dict[str, Any]] = []
    for name, op in POLICY_SCALAR_OPS:
        defects = []
        for _ in range(n_trials):
            pol = random_policy(rng, n_actions, zero_frac=0.1)
            cloned = np.append(np.asarray(pol, dtype=np.float64), 0.0)
            defects.append(abs(float(op(cloned)) - float(op(pol))))
        clone_results.append(_invariance_verdict(name, "zero_mass_clone", "invariant", defects, eps))

    # 6. negative controls: index-dependent probes MUST be flagged
    negative_controls: List[Dict[str, Any]] = []
    for name, op in NEGATIVE_CONTROL_OPS:
        defects = []
        for _ in range(n_trials):
            pol = random_policy(rng, n_actions)
            perm = action_permutation(n_actions, rng)
            defects.append(abs(float(op(permute_policy(pol, perm))) - float(op(pol))))
        v = _invariance_verdict(name, "action_permutation", "invariant(expected_violation)", defects, eps)
        v["flagged_as_violation"] = not v["invariant_within_eps"]
        negative_controls.append(v)

    violations = [
        row for row in per_operator + clone_results if not row["invariant_within_eps"]
    ]
    negative_controls_all_caught = all(row["flagged_as_violation"] for row in negative_controls)
    return {
        "symmetry_orbit_schema_version": SYMMETRY_ORBIT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "execution_mode": EXECUTION_MODE,
        "seed": seed,
        "n_actions": n_actions,
        "board_side": board_side,
        "eps": eps,
        "operators": per_operator,
        "clone_robustness": clone_results,
        "negative_controls": negative_controls,
        "n_operator_violations": len(violations),
        "operator_violations": violations,
        "negative_controls_all_caught": bool(negative_controls_all_caught),
        # The constraint holds iff every real operator obeys its transform law
        # AND the harness demonstrably catches deliberate violations.
        "game_agnostic_constraint_upheld": bool(len(violations) == 0 and negative_controls_all_caught),
    }
