# Metacognitive mechanism experiments

This document describes repository-local experiments that screen computation
allocation mechanisms before any live Rust MCTS integration. These assays are
not training ablations and do not validate QUARTZ play strength.

## Bernoulli root ranking-risk laboratory

The first assay compares three fixed-budget allocation rules on independent,
stationary Bernoulli arms:

- `uniform`: randomized round-robin;
- `raw_sequential_halving`: transparent raw-mean elimination, explicitly not
  Gumbel AlphaZero;
- `kg_rank_risk`: exact one-step Beta-Bernoulli knowledge gradient followed by
  a fixed incumbent/challenger fallback when every myopic KG is zero.

The third allocator has no fitted scalar exploration coefficient, but it is
not hyperparameter-free. It fixes a Beta(1,1) prior, a structural fallback,
and random tie handling. It models ranking risk among visible arms only. It
does not model candidate omission, widening, depth, proof search, replay, or
stopping.

### Quick local run

```bash
python3 scripts/bernoulli_root_lab.py \
  --quick \
  --output-dir results/metacognitive_root/quick_20260713
```

Run selected scenarios:

```bash
python3 scripts/bernoulli_root_lab.py \
  --scenarios equal_null_k8 top2_near_tie_k8 needle_k16 \
  --trials 1000 \
  --permutations 8 \
  --seed 20260713 \
  --output-dir results/metacognitive_root/screen_seed_20260713
```

The checked-in scenario bank is
[`configs/metacognitive_root_scenarios.v1.json`](../configs/metacognitive_root_scenarios.v1.json).
It includes equal-arm and multiple-optimum nulls, easy recognition, near-tie,
three-challenger, gap-grid, and sixteen-arm width-stress cases. Each scenario
is run under seeded arm permutations. Reward tapes are keyed by canonical arm
identity so a canonical arm keeps the same potential outcomes across
permutations and algorithms.

### Outputs

Every run directory contains:

- `run_manifest.json`: resolved scenarios, source and scenario-bank SHA-256,
  Git head/dirty state, Python/platform, run-contract hash, assumptions,
  prohibited inferences, and final artifact hashes;
- `summary.csv` and `summary.json`: per-scenario, permutation, algorithm, and
  budget descriptive metrics;
- `contrasts.csv`: all preregistered pairwise algorithm contrasts, paired by
  trial and budget, including regret deltas, MC standard errors, PCS
  discordances, and exact McNemar/sign-test p-values;
- `trials.jsonl.gz`: complete trial-level rows, always emitted so paired
  summaries can be reconstructed.

Budgets are independent reruns rather than prefixes of one continuing search.
The manifest records this explicitly. MC95 intervals are descriptive normal
Monte Carlo intervals, not correctness guarantees. Pairwise p-values are
labelled exploratory and unadjusted.

> **Permutations are correlated, not independent replicates.** The reward
> tapes and the algorithm RNG are keyed by scenario identity, *not* by
> `permutation_id`; across permutations the canonical reward streams and
> algorithm randomness are byte-identical and only the *presented arm order*
> changes. This is a deliberate index-bias / order-robustness probe. Each
> permutation's summary and contrast rows are kept separate — **do not pool
> or average metrics across permutations as if they were independent MC
> replicates**, or the variance will be understated. Future labs that reuse
> this harness must preserve the per-permutation split.
>
> **Two entry points.** `scripts/bernoulli_root_lab.py` is the preregistered
> harness (manifest, permutations, scenario bank; default seed 20260713).
> `python3 -m quartz.experiments.bernoulli_root` is a simpler direct runner
> (no manifest, no permutations; default seed 20260712). Use the script for
> claim-bearing runs; the module entry point is for quick inspection only.

### Claim firewall

Permitted:

- exact one-step KG identity under independent Beta-Bernoulli arms;
- paired common-random-number estimates for the enumerated scenario bank;
- absence of a fitted scalar exploration coefficient in the tested rule.

Prohibited:

- universal dominance or a regret bound;
- equivalence to Gumbel AlphaZero;
- transfer from IID arms to adaptive shared-tree neural MCTS backups;
- candidate-omission or true dual-risk control;
- CPU/energy efficiency without measured runtime or energy;
- human, brain, metacognitive, or grandmaster mechanism claims;
- fully hyperparameter-free status.

The correct summary is:

> In preregistered IID Bernoulli root-selection scenarios, a fixed structural
> ranking-risk rule changes paired simple regret relative to named baselines at
> specified budgets. This is a mechanism probe, not an MCTS-performance claim.

## Validation

The assay uses standard-library `unittest`, so it can run before the full ML or
Rust toolchains are installed:

```bash
python3 -m unittest discover \
  -s tests \
  -p 'test_bernoulli_root_lab.py' \
  -v

python3 -m compileall -q \
  quartz/experiment_manifest.py \
  quartz/experiments/bernoulli_root.py \
  scripts/bernoulli_root_lab.py \
  tests/test_bernoulli_root_lab.py
```

## Separation from Phase 15 and live MCTS

This experiment must not be registered as another Phase-15 refresh operator.
Most Phase-15 Group-B systems are post-hoc readouts; this laboratory is an
independent synthetic mechanism assay.

Future live allocation experiments should use resident root continuation and
record the actual continuation/fallback mode. A future engine contract needs
explicit meta-actions such as `WIDEN`, `CHALLENGE`, `DEEPEN`, `PROVE`,
`REPLAY`, and `STOP`. The current Rust `SearchPolicy` path substantially wires
observation and halt decisions, but score adjustment and these morphology
actions are not a live selection controller merely because a policy object
exists.

## Candidate morphology laboratory

The second implemented assay (`quartz/experiments/candidate_morphology.py`) is a
*separate model family*: the arm set is no longer fully visible. Each trial
draws a per-arm prior score (true mean + gaussian noise); the top-`n_visible`
arms by prior form the visible pool and the rest are hidden. A priced `WIDEN`
reveals the next-highest-prior hidden arm (progressive widening in prior order)
at a fixed integer price charged against the same budget that funds pulls, so
two regrets separate and add up exactly:

- **omission regret** = global best mean − best *visible* mean (best still
  hidden);
- **ranking regret** = best visible mean − selected mean (wrong pick among the
  revealed);
- `total_regret = omission_regret + ranking_regret`.

Three allocators are compared, all facing the identical CRN trial world:
`no_widen` (baseline — never reveal), `eager_widen` (reveal everything
affordable, then pull), and `priced_widen` (reveal only when the next arm's
prior beats the incumbent posterior mean and the price is affordable with pulls
in reserve, then `STOP` on a normal-approximation commit gate). None uses true
means or a fitted scalar exploration coefficient; the structural constants
(`PULL_RESERVE`, `WIDEN_MARGIN`, `COMMIT_MIN_VISITS`, `COMMIT_THRESHOLD`) are
recorded in the allocator contract — the lab is not hyperparameter-free.

### Quick local run (morphology)

```bash
python3 scripts/candidate_morphology_lab.py \
  --quick \
  --output-dir results/metacognitive_root/candidate_morphology_quick
```

Full preregistered screen (5 scenarios, seed 20260713):

```bash
python3 scripts/candidate_morphology_lab.py \
  --seed 20260713 \
  --output-dir results/metacognitive_root/candidate_morphology_seed_20260713
```

The scenario bank is
[`configs/candidate_morphology_scenarios.v1.json`](../configs/candidate_morphology_scenarios.v1.json):
omission-dominated (best often hidden), ranking-dominated (best almost always
visible, so widening should be wasted), and mixed regimes. Each run also drives
the **H1 discrimination-gate synthetic pre-validation**
(`quartz/experiments/h1_synthetic_gate.py`) unless `--skip-h1-gate` is passed.

### Kill checks

- **Widening lane** (`widening_kill_verdict`): demoted iff NO widen price in NO
  scenario gives a CI-separated reduction in paired omission regret vs
  `no_widen`. The stronger, honest `net_total_improvement_found` flag reports
  whether any config improves *total* regret — omission relief fully repaid in
  ranking regret is a net wash and must not be promoted to a widening claim.
- **H1 gate** (`gate_pass`): False iff the argmax-stability signal is degenerate
  (saturated at ~1.0, std below `trivial_std_eps`) on synthetic ground truth —
  which would kill H1 online wiring before the engine work.

### Claim firewall (morphology)

Same discipline as the Bernoulli assay. Permitted: the exact omission/ranking
decomposition and paired CRN deltas for the enumerated bank; the H1 gate's
non-degeneracy on synthetic ground truth. Prohibited: reading these synthetic
regrets as QUARTZ play-strength, a candidate-omission guarantee, transfer to
neural-MCTS progressive widening, CPU/energy efficiency, or `gate_pass` as
online-halt efficacy.

### Validation (morphology)

```bash
python3 -m pytest tests/test_candidate_morphology_lab.py -q

python3 -m compileall -q \
  quartz/experiments/candidate_morphology.py \
  quartz/experiments/h1_synthetic_gate.py \
  scripts/candidate_morphology_lab.py \
  tests/test_candidate_morphology_lab.py
```

## Symmetry orbit laboratory

The third implemented assay (`quartz/experiments/symmetry_orbit.py`) is a
**diagnostic**, not a screen with a kill-criterion. It supplies empirical
evidence for the game-agnostic FORBIDDEN constraint that
`quartz/phase15_signatures.py` asserts ("No game rules, board topology, or move
semantics are used anywhere") by auditing the project's own signature operators
against the behavior that constraint demands.

A truly game-agnostic scalar readout must be **invariant** under an arbitrary
relabeling (permutation) of the action axis; the committed-move index must be
**equivariant** (move with the permutation). Four symmetry channels are
exercised — action permutation on single policies, the dihedral **D4** group of
a square board (cell-index permutations), one consistent relabeling applied to a
whole trace bundle, and move-order permutation for the cross-move dispersion
operators — plus **zero-mass clone robustness** and **negative controls**
(deliberately index-dependent probes that MUST be flagged, so the harness is not
vacuous).

### Quick local run (symmetry)

```bash
python3 scripts/symmetry_orbit_lab.py \
  --seed 20260713 \
  --output-dir results/metacognitive_root/symmetry_orbit_seed_20260713
```

Config: [`configs/symmetry_orbit_audit.v1.json`](../configs/symmetry_orbit_audit.v1.json).
The runner writes `operators.csv` (per-operator defect / equivariance rows),
`summary.json` (full audit incl. negative controls), and `run_manifest.json`.

### Verdict (symmetry)

`game_agnostic_constraint_upheld` is True iff every real operator obeys its
transform law (max defect within `eps`, argmax equivariant with zero failures)
AND every negative control is flagged. On the checked-in run (seed 20260713,
512 trials, `run_contract_hash 8424eea6…`) all 15 audited real-operator/channel
checks pass (`policy_entropy`, `k_eff`, `top2_margin`, `forked_voc.voc_proxy`,
argmax-flip count, O5 revision signatures, `budget_gini`/`budget_entropy`,
`voc_tightness`; scalar defects ≤ 1e-14, argmax equivariant under both action
permutation and D4), clone robustness is exact, and both negative controls are
caught.

### Claim firewall (symmetry)

Permitted: the audited Python readouts obey the action / D4 / move-order
transform laws on synthetic policies; the harness catches index-dependent
probes. Prohibited: reading a clean audit as play-strength evidence, as proof
the Rust engine is fully game-independent (only the audited readouts are
covered), or as neural-MCTS equivariance without measuring the network itself;
reading a flag as a bug without checking the operator's intended transform law.

### Validation (symmetry)

```bash
python3 -m pytest tests/test_symmetry_orbit_lab.py -q

python3 -m compileall -q \
  quartz/experiments/symmetry_orbit.py \
  scripts/symmetry_orbit_lab.py \
  tests/test_symmetry_orbit_lab.py
```

## Next independent laboratories

These are separate model families, not options silently folded into the
Bernoulli assay (1, 2, and 4 are now implemented — see above):

1. `candidate_morphology_lab` *(implemented)*: visible/hidden pools, priced
   `WIDEN`, separate omission and ranking regret, and `STOP`;
2. `forked_voc_lab` *(implemented)*: labels each possible next computation by
   realized root decision change on frozen traces;
3. `pending_flow_lab`: count-only WU-UCT, fixed/adaptive virtual loss, elastic
   micro-waves, duplication, and measured wall time;
4. `symmetry_orbit_lab` *(implemented)*: board/action permutation equivariance
   and clone robustness;
5. `service_curve_lab`: measured evaluator latency/throughput/energy versus
   batch and global inflight credit.
