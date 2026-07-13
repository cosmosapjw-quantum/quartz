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

## Next independent laboratories

These are separate model families, not options silently folded into the
Bernoulli assay:

1. `candidate_morphology_lab`: visible/hidden pools, priced `WIDEN`, separate
   omission and ranking regret, and `STOP`;
2. `forked_voc_lab`: labels each possible next computation by realized root
   decision change on frozen traces;
3. `pending_flow_lab`: count-only WU-UCT, fixed/adaptive virtual loss, elastic
   micro-waves, duplication, and measured wall time;
4. `symmetry_orbit_lab`: board/action permutation equivariance and clone
   robustness;
5. `service_curve_lab`: measured evaluator latency/throughput/energy versus
   batch and global inflight credit.
