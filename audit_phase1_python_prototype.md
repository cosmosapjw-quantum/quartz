# Audit — BQ++ Phase 1: Python prototype + numerical primitive tests

**Date:** 2026-05-04
**Scope:** Pure-Python prototype of BQ++'s numerical primitives. Math
validation BEFORE the Rust port (BQ++ Phase 2+). Catches sign / scale
errors at `pytest -q` runtime instead of code-review time. Plan
target was "≥41 tests"; **delivered 79** across 9 modules.

## What's in `prototype/`

```
prototype/
├── README.md                ← role of the prototype
├── pyproject.toml           ← numpy, scipy, pytest only
├── bqpp_prototype/
│   ├── belief.py            ← Welford + empirical-Bayes shrinkage
│   ├── certificate.py       ← Empirical-Bernstein L_b > max U_a
│   ├── kg.py                ← Knowledge Gradient approximation
│   ├── voi.py               ← full E[max(X,0)] expected improvement
│   ├── kl_lucb.py           ← KK13 reference matching Rust
│   ├── gumbel_sh.py         ← Gumbel + Sequential Halving
│   ├── reservoir.py         ← nested-reservoir live-set
│   ├── prior_surprise.py    ← χ² statistic, no p-value
│   ├── controller.py        ← end-to-end on synthetic bandit
│   └── synthetic.py         ← clear-lead, tight-gap, hidden-best fixtures
└── tests/                   ← 9 test files, 79 tests total
```

## Test count by module

| Module | Tests | Notes |
|---|---|---|
| belief.py | 10 | Welford ↔ numpy.var; f32 vs f64 drift; empirical-Bayes shrinkage at n∈{0,1,large}; floor; Q↔unit mapping. |
| certificate.py | 9 | EB log-term hand-derived (10.546); R=1 vs R=2 scale; width monotonicity; certificate fires/doesn't-fire; **runner-up bookkeeping correct** (audit §1.3 regression). |
| voi.py | 9 | phi/Phi sanity; EI at Δ=0, ∞, s=0; **scipy.stats.norm.expect cross-check**; **wrong formula overestimates** (audit §1.2 regression); negative-Δ clamping. |
| kg.py | 8 | KG[best]=0 convention; positive elsewhere; monotone in σ_a, in 1/n_a; matches `expected_improvement`; top-m + UC bound; mismatched-length rejection. |
| kl_lucb.py | 7 | Bernoulli KL on diagonal; β = 15.618 sanity; bisection inversion; tight/wide gap; **β grows with t** (audit §1.7 regression). |
| gumbel_sh.py | 9 | Gumbel mean ≈ Euler-Mascheroni; top-m count; concentration on strong prior; uniform-prior distribution; SH bracket arithmetic; halving; **resumable** (anytime). |
| reservoir.py | 7 | Quantile (1, 2, K values); Lambda decomposition; max_size; cooldown / no-thrashing. |
| prior_surprise.py | 7 | χ² statistic value (164 hand-derived); permutation-invariant; zero-N; **does NOT import scipy.stats** (audit §1.6 regression); caller-supplied threshold; mismatched lengths; zero-prior eps clamp. |
| controller.py | 5 | Clear-lead halts; tight-gap ≥50% correct over 10 seeds; canonical halt reasons; pulls recorded; cert history. |

**Total: 79 tests, all passing.**

## Audit-regression tests (the high-value ones)

These tests were added specifically to prevent the math errors the
external review identified:

1. `test_voi_phi_only_overestimates_in_clear_lead` (audit §1.2):
   the `wrong_voi_phi_only` formula MUST be ≥ correct, with a
   non-trivial relative gap, in the clear-lead regime. If a future
   refactor accidentally re-introduces the wrong formula, this test
   fires.
2. `test_certificate_uses_runner_up_not_just_best` (audit §1.3):
   constructs a 3-arm case where a "decoy" arm has a wide upper bound;
   the certificate's `runner_up_pos` field must point at the decoy,
   not at a filler.
3. `test_kl_lucb_gap_decreases_with_t_at_fixed_n` (audit §1.7): β
   grows with t, so a fresh observe at later t can see a smaller
   gap than an earlier one. Pins the audit's correction that "stale
   cache CAN over-eager halt."
4. `test_does_not_emit_p_value` (audit §1.6): inspects `prior_surprise`
   module's runtime globals; ensures no `chi2.ppf` or `chi2.cdf`
   re-introduces the formal-test framing.

## Bugs found during prototype development

Catching these at Python time rather than Rust time is the *whole point*
of having a prototype. The errors below would each have cost
~30-60 min of Rust compile-test cycles plus the cognitive cost of
debugging in a strongly-typed compiled language.

### Bug A: `eb_log_term` hand-derivation arithmetic mistake

My initial test asserted `L = 10.316` for K=4, t=100, δ=0.05. The
correct value is **10.546**. The error: `100^1.1 ≈ 158.49`, not
`125.893` (I mis-applied the exponent). Fixed by re-deriving from
`100^1.1 = exp(1.1 * ln(100))`.

### Bug B: `_pick_next_arm` argmax-tie pollution

When all non-best KG values decay to ~0 (clear-lead bandit, 30+ pulls
per arm), `argmax(kg)` ties at 0 and returns the FIRST 0 — which is
the empirical-best arm itself (since `kg[best_pos] = 0` by convention).
The allocation rule then pulls the best arm forever, the certificate
never fires (non-best arms have only 1 pull), and the controller
hits MaxIters.

Fix: explicitly exclude `best_pos` from the runner-up search. Use
`max(non_best, key=lambda i: (kg[i], -pulls_per_arm[i]))` — the
secondary sort key (fewest pulls) ensures progress when KG ties
genuinely at 0. This is the LUCB "alternate between best and the
arm with maximum upper-CI" allocation, which is the canonical
best-arm-id allocation rule.

### Bug C: scale mixing in the controller

My initial controller called `map_q_to_unit(welford[a].mean)` — but
the synthetic bandit emits values directly on the [0, 1]-ish scale
(true_means in [0.3, 0.8]), so the map produces values in [0.65, 0.9]
while the *variance* (Welford's M2) is computed on the original scale.
This is the audit's §1.4 scale-mixing bug class.

Fix: the controller treats bandit values as already on the canonical
scale; the `map_q_to_unit` utility is documented as "for the real
MCTS engine where backed-up Q is in [-1, 1]" only. The prototype
fixture is on the canonical [0, 1] scale per the audit §6.1
recommendation.

### Bug D: docstring with `scipy.stats` literal trips a string-grep test

My initial `test_does_not_emit_p_value` did
`assert "scipy.stats" not in source_text` — but the module's docstring
intentionally MENTIONS `scipy.stats.chi2` to warn callers away from
re-introducing the formal-test framing. The grep then false-positives
on the docstring.

Fix: inspect runtime globals (`mod.__dict__`) instead of source text.
This catches actual imports and forbidden function names without
flagging educational docstring content.

## Test runtime

`pytest -q` total: ~1.3s on a single CPU thread. The Welford 1e6-sample
test is the slowest individual case (~0.4s); everything else is
near-instant.

## What this unblocks

Phase 2 (Rust port to ArcSwap PolicyCache + edge-local indexing) can
now consume the validated math primitives. Each Rust test will pin
the same hand-derived expected value as the corresponding Python test.
Drift between Python and Rust is therefore caught at port time.

Specifically, the following Rust tests in BQ++ Phase 2+ will use
expected values from this prototype:
- `test_phase2_eb_width_at_n_eq_100_sigma2_eq_0p04` ⇒
  reference width 0.140 (from Python `test_eb_width_*` chain).
- `test_phase2_kl_lucb_beta_at_kk13_inputs` ⇒ 15.618 (already
  shipping in commit `3370f95`).
- `test_phase4_full_ei_matches_python` ⇒ within 1e-6 of the Python
  reference.

## Files touched

- `prototype/` (NEW directory tree; 0 LOC ⇒ ~900 LOC):
  - 9 source modules (~600 LOC total)
  - 9 test files (~750 LOC total)
  - README + pyproject + __init__'s (~50 LOC total)

Net delta: **+~1400 LOC** (overshooting plan estimate of +900 because
each test adds ~5-15 lines of structured assertions; the tests are
carrying explicit hand-derived values per the audit's §11 phase
exit criterion).

## Adversarial review

The Phase 1 risk is "Python prototype is correct but Rust port
silently introduces a different bug." Three mitigations:

1. Every numerical test pins a **hand-derived expected value with
   tolerance** — not "Python and Rust agree on something." The Rust
   port will be tested against the same expected value, so drift is
   visible.
2. The `test_voi_phi_only_overestimates_in_clear_lead` and
   `test_certificate_uses_runner_up_not_just_best` regression tests
   would catch sign / formula errors regardless of language.
3. The prototype lives at `prototype/` (a top-level dir) so it ships
   with the repo. Any Rust review can re-run the Python tests as a
   cross-check.

The Phase 1 risk that is NOT mitigated is "the Python prototype's
choice of allocation rule (LUCB-style) differs from the Rust
production allocation rule (Gumbel SH, Phase 3)." This is by design:
Phase 1 validates the *halt* and *certificate* primitives; the
*allocation* primitive (Gumbel SH) gets its own validation in BQ++
Phase 3 with its own prototype tests.
