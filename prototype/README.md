# BQ++ Python Prototype

This directory holds a self-contained Python prototype of the BQ++
controller's numerical primitives. It exists **before** the Rust port
(BQ++ Phase 2+) so the math can be locked at the formula level
without a Rust compile-test cycle.

## Why a Python prototype

The external audit (`../report.md`) identified seven mathematical
errors in the original BayesianQuartz design (sign error in VOI,
wrong EB-gap formula, scale inconsistencies, etc.). Catching these
at code-review time was expensive; catching them with `pytest -q`
running against a `numpy` / `scipy.stats` reference is cheap.

Each numerical primitive in `bqpp_prototype/` ships with a test
file that pins its formula to either:

1. A hand-derived expected value (e.g. `σ_a` with N=0, λ₀=4 ⇒
   exactly `σ_root`), or
2. A `scipy.stats` reference (e.g. full `E[max(X, 0)]` for a
   truncated normal matched against `scipy.stats.truncnorm.expect`).

Once a primitive's test pins its expected value, the Rust port
(BQ++ Phase 2+) re-tests against the same expected value. Drift
between Python and Rust is therefore caught at port time.

## Scope

This is a **prototype**. It is not optimized, not concurrency-safe,
not meant to be production code. The tree-search in `controller.py`
runs a synthetic bandit, not the actual MCTS engine. The actual
controller lives in `src/mcts/policy/` (Rust); this directory is
strictly for math validation.

## Layout

```
prototype/
├── README.md              ← this file
├── pyproject.toml         ← numpy, scipy, pytest only
├── bqpp_prototype/
│   ├── __init__.py
│   ├── belief.py          ← Welford + empirical-Bayes variance shrinkage
│   ├── certificate.py     ← Empirical-Bernstein L_b > max U_a certificate
│   ├── kg.py              ← Knowledge Gradient approximation
│   ├── voi.py             ← full E[max(X,0)] expected improvement
│   ├── kl_lucb.py         ← KK13 reference (matches Rust kl_helpers)
│   ├── gumbel_sh.py       ← Gumbel + Sequential Halving root scheduler
│   ├── reservoir.py       ← nested-reservoir live-set maintenance
│   ├── prior_surprise.py  ← χ² statistic as diagnostic (no p-value)
│   ├── controller.py      ← ties modules together; the BQ++ policy
│   └── synthetic.py       ← synthetic bandit + tree fixtures
└── tests/
    └── (one test file per module; 41+ tests total)
```

## Running

```bash
cd prototype
pip install -e .
pytest -q
```

Or directly without install:

```bash
cd prototype
PYTHONPATH=. pytest -q tests/
```

## Test count

41+ numerical tests covering all 9 primitive modules.
See [`../audit_phase1_python_prototype.md`](../audit_phase1_python_prototype.md)
(emitted at end of Phase 1) for the full enumeration with
hand-derived expected values.
