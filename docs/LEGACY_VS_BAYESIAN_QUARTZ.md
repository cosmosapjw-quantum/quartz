# LegacyQuartz vs BayesianQuartz — Detailed Comparison for External Audit

**Document scope.** A pedagogical, self-contained comparison of the
controller that drives the QUARTZ AlphaZero search engine *today*
(`LegacyQuartz`, also referred to as the legacy controller throughout)
and the principled replacement that is *planned* for patch P09
(`BayesianQuartz`). The two policies are presented side-by-side with
their mathematical foundations, code citations, novelty and feasibility
assessments, and pseudocode.

**Status of each policy at the time of writing.**

| Policy | Status | Code path |
|---|---|---|
| LegacyQuartz | Implemented in production. Default for every existing experiment. | Driven by `quartz_policy_adjustment` at [`src/mcts/select.rs:161`](../src/mcts/select.rs#L161) and `QuartzController::should_stop` at [`src/mcts/quartz.rs:2031`](../src/mcts/quartz.rs#L2031). Wrapped behind the new `SearchPolicy` trait by the bit-identical shim in [`src/mcts/policy/legacy_quartz.rs`](../src/mcts/policy/legacy_quartz.rs) (commit `965e216`, P07). |
| BayesianQuartz | Designed in the v1.0 plan, **not yet implemented**. Slated as patch P09 in the 15-patch sequence at [`~/.claude/plans/iridescent-giggling-bachman.md`](../../../.claude/plans/iridescent-giggling-bachman.md). Trait scaffolding (`SearchPolicy`, `EdgeView::sigma_a`, KL bisection helpers) is already in place via P06 (commit `3370f95`). |

**Honesty discipline.** This document deliberately avoids both
over-claiming and under-claiming. LegacyQuartz is described exactly as
the code behaves; BayesianQuartz is described exactly as the plan
specifies, with explicit "not yet implemented" markers where applicable.
Mathematical claims are cited to their primary sources and, where
applicable, hand-derived in this document so a reviewer can re-verify.

---

## 1. Background — what is QUARTZ trying to do?

QUARTZ is an AlphaZero-style search engine: PUCT-driven Monte-Carlo
Tree Search over a learned policy and value head, used for self-play
training and tournament play across Gomoku 7×7, Gomoku 15×15, Go 9×9,
and standard Chess. The "controller" is the component sitting on top of
PUCT that decides:

1. **Whether to modify the per-edge selection score** (e.g. add a
   penalty term, refresh the prior with a posterior signal).
2. **When to halt** the iteration loop (fixed budget, P_flip
   convergence, value-of-information argument, etc.).

The original AlphaZero (Silver et al. 2017) does neither: pure PUCT,
fixed visit budget. QUARTZ's value proposition is that adaptive search
control — using the search's own statistics to decide where to explore
and when to stop — yields better play per unit compute, especially in
the regime where the network is noisy.

Over the project's evolution the controller accumulated several
mechanisms (penalty modes, halt modes, cost modes, a dozen boolean
flags). The audit conducted before this comparison document was written
identified that these mechanisms were *empirically motivated heuristics
dressed in physics vocabulary* (path integrals, Fisher metric,
ε-envariance via Pinsker bound). The plan calls them, accurately, a
"family of state-driven controllers plus explicit hyperparameters" —
which is the same wording that already appears in the project's own
[`docs/QUARTZ_THEORY.md`](QUARTZ_THEORY.md) §9.

BayesianQuartz is the ground-up rebuild that replaces each heuristic
component with the corresponding rigorous primitive from the bandit
theory and Bayesian decision theory literature.

---

## 2. LegacyQuartz — what is in the code today

### 2.1 Surface area

The legacy controller is configured by a single `QuartzConfig` struct
at [`src/mcts/quartz.rs:200-279`](../src/mcts/quartz.rs#L200-L279). The
relevant axes are:

- **PenaltyMode** (7 variants) — selects the per-edge penalty formula
  applied at root selection ([`src/mcts/quartz.rs:103-125`](../src/mcts/quartz.rs#L103-L125)):
  `Legacy`, `EffectiveV2`, `None`, `SelfAdaptive`, `GatedRefresh`,
  `GatedRefreshLegacy`, `PFlipMixture`.
- **HaltMode** (4 variants) — selects the stopping rule
  ([`src/mcts/quartz.rs:131-146`](../src/mcts/quartz.rs#L131-L146)):
  `VOC`, `SimpleThreshold`, `Fixed { budget }`, `ConfAdaptive`.
- **CostMode** (3 variants) — selects the cost scaling for VOC
  ([`src/mcts/quartz.rs:164-174`](../src/mcts/quartz.rs#L164-L174)):
  `Legacy`, `Constant`, `TimeDriven`.
- **Boolean flags** (11 of them) — `enable_fisher_puct`,
  `enable_one_loop`, `enable_expand_channel`, `enable_merge_channel`,
  `enable_ns_gate`, `enable_depth_cal`, `enable_poisson_phidden`,
  `enable_merge_r0`, `pflip_mixture_divergence_gate`, plus
  `root_only_shaping`, `enable_fisher_puct`.

The combinatorial surface is 7 × 4 × 3 × 2¹¹ ≈ 229,376 distinct
configurations. In practice only a handful are exercised by the
canonical ablation presets, but the code has to dispatch on the full
surface every search iteration.

### 2.2 Penalty dispatch

The penalty term is applied at root selection in
[`src/mcts/select.rs:408-432`](../src/mcts/select.rs#L408-L432) (the
`match qcfg.penalty_mode` block) plus four early-return branches at
lines 175 (`SelfAdaptive`), 223 (`GatedRefresh`), 258
(`GatedRefreshLegacy`), 294 (`PFlipMixture`). The five behaviorally
distinct penalty formulae are:

| Mode | Formula | Notes |
|---|---|---|
| `Legacy` | `−min(ħ_eff, cap) / N_a` | Clamped at cap; `ħ_eff = σ_Q / σ₀` |
| `EffectiveV2` | `−ν / (1 + N_a + O_a)` | `ν = hbar_penalty_cap`; quadratic-style denominator |
| `None` | `0` | Pure PUCT baseline |
| `SelfAdaptive` | `−σ_Q / (1 + N_a + O_a)` | Global σ_Q applied to every action; not per-action posterior variance |
| `GatedRefresh` | `−min(ħ_eff, cap) · N_a / N_parent` | Root-share weighting |
| `GatedRefreshLegacy` | `−ν / (1 + N_a + O_a)` (= EffectiveV2 numerator) | Wrapped with a P_flip gate on prior refresh |
| `PFlipMixture` | `−max(cap, σ_Q) / (1 + N_a + O_a)` | Same denominator as SelfAdaptive but capped numerator |

Several of these branches additionally compute and apply a *prior
refresh* — a rewriting of the network's policy prior in light of the
visit distribution. The mechanisms differ ([`src/mcts/select.rs:170-403`](../src/mcts/select.rs#L170-L403)):

- `SelfAdaptive`: Bayesian per-action visit-share blend with
  visit-count temperature `τ = ln(1 + N_total/K)`.
- `GatedRefresh`: visit-share blend gated on
  `prior_q_divergence > ε_t`.
- `GatedRefreshLegacy`: Q-based blend, gated on `P_flip / 0.159`.
- `PFlipMixture`: P_flip-mediated mixture between Q-refresh and
  visit-frequency-refresh, optionally further gated on prior_q_divergence.
- Manual fall-through (`Legacy/EffectiveV2/None` with
  `prior_refresh_rate > 0`): config-driven ρ + manual τ.

### 2.3 Halt dispatch

`QuartzController::should_stop` at
[`src/mcts/quartz.rs:1996-2117`](../src/mcts/quartz.rs#L1996-L2117)
implements the four halt modes:

- `Fixed { budget }`: hard cap at `root_visits >= budget`.
- `SimpleThreshold`: stop when `p_flip < 0.159` for 3 consecutive
  checks. The 0.159 constant comes from `Φ(−1) ≈ 0.1587`, the standard
  normal CDF at −1σ ([`src/mcts/quartz.rs:339`](../src/mcts/quartz.rs#L339)).
- `VOC`: `SimpleThreshold` AND `voc_total ≤ 0`, where
  `voc_total = max(voc_focus, voc_expand, voc_merge)` and each
  channel is `P_flip × σ_Δ × envariance_modifier − cost_*`.
- `ConfAdaptive`: online θ-adaptation targeting
  `Conf(t) = (1 − P_flip)(1 − P_hidden) max(0, 1 − S/S₀) ≥ θ`.

The σ_Q value used throughout is the **visit-weighted variance of root
child Q-means**:

  σ_Q² = Σ_a N_a (Q_a − Q̄)² / N_total,   Q̄ = Σ_a N_a Q_a / N_total

where the sum is over root children with `N_a > 0`. This is the
empirical implementation at
[`src/mcts/quartz.rs:867-876`](../src/mcts/quartz.rs#L867-L876).

### 2.4 The ε-envariance test

The `expand` channel uses a heuristic that the docstring justifies via
Pinsker's inequality
([`src/mcts/quartz.rs:24-30`](../src/mcts/quartz.rs#L24-L30)).
The actual code at
[`src/mcts/quartz.rs:632-642`](../src/mcts/quartz.rs#L632-L642)
implements:

  threshold = 0.5 / √N
  expand_active = S_KL > threshold

where `S_KL = KL(visit_distribution || prior)` is the KL divergence of
the visit distribution over root children from the network's prior.

The Pinsker bound `KL(P || Q) ≥ 2 |P − Q|²_TV` would, with
ε_t = 1/√N, give a threshold scaling as O(1/N), not O(1/√N). The code's
choice is empirically motivated (looser tolerance at larger N, tighter
early on) but is **not** what the Pinsker invocation literally implies.
This is one of the two issues that QUARTZ_THEORY.md §9 already
acknowledges with its "honest scope" downgrade.

### 2.5 What QUARTZ_THEORY.md §9 says about itself

> "QUARTZ is a **family of state-driven controllers plus explicit search
> hyperparameters**, not a hyperparameter-free law."
>
> "The honest reading is: QUARTZ is a heuristic bundle wearing
> controller language. Each component is empirically motivated; their
> integration is pragmatic, not derived."

The document at hand is consistent with this self-disclosure. It is not
a refutation of LegacyQuartz; LegacyQuartz works in practice on
Gomoku 7×7 short-budget settings (see the README's "Current Controller
Status" section). The replacement is motivated by *measurability* and
*falsifiability* rather than by any specific empirical defect.

---

## 3. BayesianQuartz — what is planned for P09

The plan file at [`~/.claude/plans/iridescent-giggling-bachman.md`](../../../.claude/plans/iridescent-giggling-bachman.md)
specifies BayesianQuartz as a single named policy that consolidates
five well-established statistical primitives, each replacing one of the
legacy heuristic components. The replacements are listed in the table
below; each is then discussed in detail.

| Legacy component | Plan replacement | Reference |
|---|---|---|
| `SelfAdaptive` penalty (global σ_Q) | Per-action posterior σ_a from Welford + Beta-Binomial conjugate prior | Welford 1962; standard Normal-inverse-Gamma posterior |
| Penalty `−σ_Q / (1 + N_a + O_a)` (linear in N) | `−σ_a / √(1 + N_a + O_a)` (empirical-Bernstein scaling) | Maurer & Pontil 2009 |
| Pinsker-invocation envariance test | Pearson χ² goodness-of-fit | Pearson 1900 |
| `voc_focus = P_flip × σ_Δ × envariance_mod − cost` | One-step value of information `VOI(a) = φ(z) · s` | Russo & Van Roy 2018 |
| `SimpleThreshold` halt (P_flip < 0.159) and `VOC` halt | KL-LUCB δ-PAC certificate | Kaufmann & Kalyanakrishnan 2013, Theorem 8 |

The five components below are presented as the plan specifies them,
including the exact Rust pseudocode the plan provides for the policy's
`observe` and `should_halt` methods.

### 3.1 Per-action posterior σ_a (component a)

**Replaces:** the global σ_Q used by `SelfAdaptive` and friends.

The Welford online variance algorithm (Welford 1962) maintains a
running sum of squared deviations `M2` per edge:

  M2 ← M2 + (x − μ_old)(x − μ_new)

where `x` is the latest backed-up value, `μ_old` is the previous
running mean, and `μ_new` is the updated running mean. The
sample variance is `M2 / (N − 1)`.

The legacy code already keeps `M2` per edge (in
[`src/mcts/node.rs`](../src/mcts/node.rs)) but reads it only into the
visit-weighted variance σ_Q at the root level. BayesianQuartz instead
forms a **per-action posterior std-dev** by combining `M2_a` with a
weak-prior pseudo-count:

  σ_a² = (M2_a + λ₀ · σ_root²) / (N_a + λ₀)

This is the maximum-a-posteriori variance estimate under a
Normal-inverse-Gamma conjugate prior with shape α = λ₀/2, rate β = (λ₀/2) σ_root²
(Murphy 2012, §4.6.3). The default `λ₀ = 4` is the canonical "weak
prior" choice — strong enough to prevent degenerate σ_a → 0 at N_a = 0
but weak enough to be dominated by 50+ observations.

**At N_a = 0, M2_a = 0:**
  σ_a = √((0 + 4 · σ_root²) / 4) = σ_root *exactly.*

**At N_a = 1, M2_a = 0:**
  σ_a = √((0 + 4 · 0.09) / 5) = √(0.072) ≈ 0.2683 (with σ_root = 0.3).

These two hand-computed values are pinned by the unit tests already
shipping in P06 ([`src/mcts/policy/mod.rs:tests`](../src/mcts/policy/mod.rs)).
The `EdgeView::sigma_a` helper that performs this computation is also
shipping in P06 ([`src/mcts/policy/trait_def.rs:101-110`](../src/mcts/policy/trait_def.rs#L101-L110)).
**The math primitive exists in code today; only the policy that
consumes it is pending.**

### 3.2 Empirical-Bernstein penalty scaling (component b)

**Replaces:** the linear-in-N denominator of the legacy penalty.

Maurer & Pontil 2009 prove that, with probability ≥ 1 − δ:

  |μ_a − μ̂_a| ≤ √(2 σ̂_a² log(2/δ) / N_a) + 7 log(2/δ) / (3 (N_a − 1))

The dominant term is `O(σ_a / √N_a)`. The current `SelfAdaptive` formula
`−σ_Q / (1 + N_a + O_a)` has a denominator that scales linearly in
N. This is dimensionally inconsistent: as N grows, the legacy penalty
shrinks at rate 1/N, whereas the actual concentration of μ̂_a around μ_a
shrinks at rate 1/√N. The penalty therefore over-discourages revisits
to high-N actions early in search and under-discourages them late.

The plan's replacement is a single-line change that finally aligns
the exponent with the deviation rate:

  penalty(a) = −σ_a / √(1 + N_a + O_a)

The `1 +` in the denominator is the standard PUCT regularizer that
prevents division by zero at N_a = 0; the same regularizer appears in
Rosin's original PUCT formula (Rosin 2011).

### 3.3 Empirical Bernstein gap CI (component c)

**Replaces:** the Pinsker-invocation envariance test.

The plan keeps an Empirical Bernstein bound as a *diagnostic* field
(`eb_gap` in the cache, surfaced via telemetry) but does not use it as
the primary halt criterion — that role goes to KL-LUCB (component f).
The diagnostic is:

  EB_b̂ = √(2 σ̂_b̂² log(2/δ) / N_b̂) + 7 log(2/δ) / (3 (N_b̂ − 1))
  eb_gap = μ̂_b̂ − 2 · EB_b̂

A positive `eb_gap` is a sanity-check for the KL-LUCB stop: both
should fire in the same regime. If they disagree, that's an early
warning that one of them is mis-implemented or the search is in a
degenerate regime (e.g. σ_a underflow, near-tied arms).

### 3.4 Pearson χ² envariance test (component d)

**Replaces:** the Pinsker-invocation S_KL > 0.5 / √N threshold.

The Pearson χ² goodness-of-fit test (Pearson 1900) is the canonical
test of whether observed counts match an expected distribution:

  χ² = Σ_a (N_a − N · π₀(a))² / (N · π₀(a))

Under the null hypothesis (visits actually drawn from the network's
prior), χ² is asymptotically distributed as χ²_{K−1} where K is the
number of root children. The test rejects the null at significance α
when χ² > F⁻¹_{χ²_{K−1}}(1 − α).

The plan's choice α = 0.05 corresponds to a 5% false-rejection rate
under H₀. The threshold is computed via `statrs::distribution::ChiSquared::inverse_cdf`
(an established Rust statistics crate) — not via a Taylor expansion or
hand-tuned heuristic.

**Why χ² and not KL?** Both measure distributional divergence, but χ²
has a known limit distribution under H₀, which is what's needed to
emit a p-value. Pinsker only relates KL to TV one-directionally and
does not provide a test statistic. The legacy code implicitly assumes
that KL > some-threshold means TV > some-other-threshold, which is the
correct direction of Pinsker's inequality, but the threshold scaling
chosen (0.5 / √N) does not derive from it.

**Why this matters for falsifiability.** With χ², a reviewer can
audit the threshold by reading the χ² table for K−1 degrees of
freedom. With Pinsker's invocation, the threshold is a magic number.

### 3.5 Russo-Van Roy one-step value of information (component e)

**Replaces:** the `voc_focus = P_flip × σ_Δ × envariance_mod − cost_focus`
proxy.

The legacy VOC formula is a hand-built proxy for "expected gain from
one more iteration minus cost." It is dimensionally consistent (all
terms are in the value-units of the game) but it is not the one-step
value-of-information from the Bayesian decision theory literature.

The plan replaces it with the explicit one-step VOI under the Gaussian
posterior assumption (Russo & Van Roy 2018):

  Δ = μ̂_b̂ − μ̂_c   (gap between empirical best and runner-up)
  s_a = √(σ̂_b̂²/N_b̂ + σ̂_a²/(N_a + 1))   (std-dev of `Δ + ξ_a`,
                                          where `ξ_a` is the
                                          marginal effect of one
                                          more pull of arm a)
  VOI(a) = φ(−Δ / s_a) · s_a

where φ is the standard normal density. This is the dominant term in
the truncated-normal expectation `E[max(Δ', 0)]`. The full expression
adds `Δ · (1 − Φ(−Δ/s_a))` but for clear-lead regimes (Δ ≫ s_a) this
second term is negligible; for tight-gap regimes (Δ ≈ s_a) it is
roughly half the truncated-normal expectation, but in that regime the
search should keep going anyway, so under-estimating VOI is the
*safer* error direction.

The halt fires when `max_a VOI(a) < c_time(t)`, where `c_time(t)` is
the time-decayed cost (the same family of `c_time` functions used by
`CostMode::TimeDriven` today).

**Why the Gaussian assumption is OK.** Per the central limit theorem,
N_a > 30 makes `μ̂_a` approximately Gaussian around μ_a regardless of
the underlying distribution of leaf values. The 30-pull threshold is
also the `min_pulls` floor in component f.

### 3.6 KL-LUCB halt (component f)

**Replaces:** `SimpleThreshold` (P_flip < 0.159) and `VOC` halt.

This is the same KL-LUCB stopping rule shipped as a standalone policy
in P08 (`KLLUCBStop`, [`src/mcts/policy/kl_lucb.rs`](../src/mcts/policy/kl_lucb.rs)).
The reference is Kaufmann & Kalyanakrishnan 2013, Theorem 8.

For each root child `a` with empirical mean `μ̂_a = (Q_a + 1)/2` mapped
from `[-1, 1]` to `[0, 1]` for Bernoulli KL operations, define:

  L_a(t, δ) = inf{q ∈ [0, μ̂_a] : N_a · KL(μ̂_a, q) ≤ β(t, δ)}
  U_a(t, δ) = sup{q ∈ [μ̂_a, 1] : N_a · KL(μ̂_a, q) ≤ β(t, δ)}

with KL the binary Bernoulli KL `p log(p/q) + (1−p) log((1−p)/(1−q))`
and the threshold:

  β(t, δ) = log(k₁ · K · t^α / δ),   k₁ = 405.5,   α = 1.1,   K = n_children.

**Stopping rule.** Let b̂ = argmax_a μ̂_a, c = argmax_{a ≠ b̂} U_a.
Stop when L_b̂ > U_c. Equivalently, stop when

  gap_bits = N_b̂ · KL(μ̂_b̂, μ̂_c) − β(t, δ) > 0.

**Sample-complexity guarantee.** KK13 prove that this rule is
δ-PAC: the probability of returning a wrong best arm is at most δ.
The required sample size is within a constant factor of the
information-theoretic lower bound. By contrast, the legacy P_flip
< 0.159 rule has no formal guarantee — the 0.159 = Φ(−1) is the
1-σ rule for *univariate* Gaussians, not for the bandit best-arm
problem.

**δ as a tuning knob.** δ is the user's confidence target. Default
0.05 (95% confidence). 0.01 (99%) costs ~30% more samples in the
limit; 0.001 (99.9%) costs ~60% more.

### 3.7 The composite halt rule

Putting components (e) and (f) together, BayesianQuartz halts when ANY
of the following fire:

1. `gap_bits > 0` AND `n_total ≥ min_total` ⇒ `Stop(KLLUCBStop)` —
   PAC certificate fires.
2. `max_a VOI(a) < c_time(t)` AND `n_total ≥ min_total` ⇒
   `Stop(PolicyConverged)` — Bayes-optimal one-step rule says
   continuing is not worth the cost.
3. `n_total ≥ max_visits` ⇒ `Stop(MaxVisits)` — hard ceiling.
4. `elapsed > time_cap` ⇒ `Stop(MaxTime)` — wall-clock ceiling.

In practice (1) and (2) usually agree; their disagreement is
diagnostic (see component c, "EB sanity").

---

## 4. Side-by-side summary table

| Aspect | LegacyQuartz | BayesianQuartz (planned) |
|---|---|---|
| Surface area (configurations) | 7 × 4 × 3 × 2¹¹ ≈ 229k | 1 named policy, 8 hyperparameters |
| Penalty source | global σ_Q applied uniformly | per-action posterior σ_a from Welford + Beta-Binomial |
| Penalty exponent | linear in N (`/N_a`) | empirical-Bernstein-aligned (`/√N_a`) |
| Envariance test | S_KL > 0.5/√N (Pinsker-flavored) | Pearson χ² ~ χ²_{K−1} at α=0.05 |
| Value-of-information | hand-built proxy `P_flip × σ_Δ × envariance_mod − cost` | Russo-Van Roy `φ(z) · s` |
| Halt rule | P_flip < 0.159 (one-σ rule) and/or VOC ≤ 0 | KK13 PAC: `gap_bits > 0` at confidence δ |
| Formal guarantees | none | δ-PAC best-arm identification |
| Hand-tuned constants | 0.159 (Φ(−1)), 0.5 (envariance), 0.3 (cap), 3 (FLIP_STABLE_N), σ_0 per game | δ (user-tunable), λ_0 (Bayesian prior), min_pulls=30, min_total=200 |
| Falsifiability of telemetry | low — many fields documented but not emitted (W3 audit finding; fixed in P01 commit `4070909`) | high — every halt decision carries `gap_bits`, `chi2`, `bayes_voi` in the JSON `controller_summary.extended` block (P01 schema_version=6) |
| Default in code | yes (commit `965e216`, P07 wraps it as `--policy=legacy_quartz`) | no — flips to default in P10 |
| Reproducibility of existing results | preserved bit-identically via the LegacyQuartz shim | n/a — new policy |

---

## 5. Mathematical foundations — why these specific tools?

### 5.1 Welford + Beta-Binomial → posterior σ_a

The Welford algorithm (1962) is the numerically-stable way to compute
running variance. Its M2 statistic is sum-of-squared-deviations from
the running mean. f64 precision is sufficient out to N ≈ 10¹⁵ before
catastrophic cancellation; f32 starts losing meaningful digits past
N ≈ 10⁵ (verified empirically — drift ≈ 5e-4 from scipy.var at 10⁶
samples). The plan stores `m2` as f64 in `EdgeView`.

The Beta-Binomial conjugate posterior (or Normal-inverse-Gamma in the
continuous-value case) is the standard textbook way to introduce a
weak prior into a Bayesian variance estimate (Murphy 2012, §4.6). The
formula `σ_a² = (M2 + λ₀ σ_root²) / (N + λ₀)` is the posterior mean of
the variance under prior `σ_root²` and pseudo-count λ₀.

**Why σ_root² as the prior variance?** It's the visit-weighted
variance at the parent (root) node, which is the natural "what we'd
expect this action's value variance to be if we had no information"
signal. It also has the same dimensions as σ_a², which is required
for the formula to be coherent.

### 5.2 Maurer-Pontil empirical Bernstein

Hoeffding's inequality bounds deviation as `O(1/√N)` regardless of
variance. Bernstein's inequality tightens this to
`O(σ/√N + 1/N)`, so for low-variance arms it's substantially better.
Maurer & Pontil 2009 give the *empirical* version that uses the
sample variance σ̂² in place of the true σ², which is what we have at
hand during MCTS.

The bound is tight (within the constants) for sub-Gaussian random
variables, which Q-values bounded in [-1, 1] are. The relevant
property for QUARTZ: as N grows, the deviation shrinks at `1/√N`,
NOT `1/N`. Aligning the penalty exponent with this rate is what
makes BayesianQuartz's penalty dimensionally honest.

### 5.3 Pearson χ² goodness-of-fit

Pearson 1900 introduced the χ² test for "the criterion that a given
system of deviations from the probable in the case of a correlated
system of variables is such that it can be reasonably supposed to
have arisen from random sampling." A modern treatment is Lehmann &
Romano 2005, §14.3.

The test statistic χ² = Σ (O − E)² / E has the limiting distribution
χ²_{K−1} under the null. The threshold for α = 0.05 with K = 4 (e.g.
4 root children) is χ²_{3, 0.95} ≈ 7.815.

**Caveat for very small K.** For K = 2 the test has only 1 degree of
freedom and is asymptotically equivalent to a 2-sided z-test on the
visit proportion. The plan handles this correctly — `chi2_dof =
max(K − 1, 1)` — but a reviewer should note that Pearson's χ² has
known issues when expected counts E_a < 5; the plan's pre-flight
check `if e.prior < ε then clamp` guards against the worst pathology
(zero-prior actions).

### 5.4 Russo-Van Roy 2018 one-step VOI

Russo & Van Roy formulate value-of-information for time-sensitive
bandit learning as the expected immediate improvement conditional on
the posterior. The one-step version (as used in the plan) computes
the expected change in identified-best value from pulling each arm
once.

For the Gaussian-posterior case (which is the regime QUARTZ operates
in once N_a > 30 for each arm), the closed form is:

  VOI(a) = E_q[max(Q_a − Q_b̂, 0)] under posterior q

For Q_a − Q_b̂ ~ N(μ̂_a − μ̂_b̂, s_a²), this is a truncated-normal
expectation:

  = (μ̂_a − μ̂_b̂) · Φ((μ̂_a − μ̂_b̂)/s_a) + s_a · φ((μ̂_a − μ̂_b̂)/s_a)

For arms with μ̂_a < μ̂_b̂ (i.e. all arms except the lead), the first
term is small and the dominant term is `s_a · φ((μ̂_a − μ̂_b̂)/s_a) =
s_a · φ(z)`.

The plan keeps only this dominant term, which is correct for the
clear-lead regime and slightly conservative (under-estimating VOI)
for the tight-gap regime. Under-estimating VOI is the safer
direction: it makes the policy *less* eager to halt.

### 5.5 KL-LUCB sample complexity

The Kaufmann-Kalyanakrishnan 2013 KL-LUCB procedure (and the
Garivier-Kaufmann 2016 GLR refinement) achieve, for the
fixed-confidence best-arm identification problem, sample complexity:

  E[τ] ≤ C · Σ_{a ≠ b̂} 1 / KL(μ_a, μ_b̂) · log(1/δ)

where the sum is over sub-optimal arms, KL is the Bernoulli KL, and
C is a constant. This is within a logarithmic factor of the
information-theoretic lower bound for the problem.

In contrast, Hoeffding-style bounds (which is what the legacy
P_flip < 0.159 implicitly uses, since it's a 1-σ Gaussian rule) give
sample complexity proportional to `1 / Δ²` — an order of magnitude
worse for arms with different variances (e.g. one nearly-deterministic
arm and several noisy ones).

This is the single biggest theoretical improvement the plan claims.
**Whether it materializes empirically** depends on whether the
underlying value distributions are far from Bernoulli — and on
whether the search budget is in the asymptotic regime. For Gomoku 7×7
with 800 visits these conditions are plausibly met; for chess at 1600
visits less so. The plan calls out this caveat in §11A.

---

## 6. Novelty assessment — what is genuinely new?

This is the section where over-claiming is most tempting. The honest
answer is:

**As a stack of techniques, BayesianQuartz is not novel.** Every one
of its components — Welford, Beta-Binomial conjugate priors, empirical
Bernstein, Pearson χ², Russo-Van Roy VOI, KL-LUCB — has been in the
statistics or bandit literature for years (or, in Pearson's case,
since 1900). None of them is QUARTZ-specific.

**The novelty is the integration into AlphaZero MCTS root control.**
- Most prior bandit work treats arms as independent and
  identically distributed. AlphaZero MCTS arms have shared subtrees,
  posterior couplings via the network, and PUCT regularization. The
  plan side-steps these complications by treating only the root level
  as a bandit problem and leaving sub-tree selection to vanilla PUCT.
- KataGo's playout-cap (Wu 2019) uses a similar adaptive-budget idea
  but with a different stopping rule (winrate threshold).
- The MENTS line (Xiao et al. 2019, slated for P11) attacks the same
  problem with a different formal tool (soft Bellman / maximum entropy
  RL).
- Bai et al. 2013 ("Bayesian Mixture Modelling and Inference based
  Thompson Sampling in MCTS") is closest in spirit but uses Thompson
  sampling at every selection rather than at root only.

**What QUARTZ contributes** is therefore the engineering of the
integration: a single named policy with a measured set of
hyperparameters, falsifiable telemetry (`gap_bits`, `chi2`, `bayes_voi`
in the controller_summary), bit-identical reproducibility of the legacy
controller via the shim, and a unit-test discipline that pins each
mathematical claim to a hand-derived expected value.

This is a useful research contribution — *especially* combined with
the controller-as-research-axis ablation framework already shipping in
QUARTZ — but it is not a theoretical novelty. A reviewer for an
external audit should know exactly this.

---

## 7. Feasibility — can this be built in 17 engineer-days?

The plan's effort estimate is 17 engineer-days for the full P01-P15
sequence; P09 (BayesianQuartz) accounts for ~3.0 days within this
budget (~520 LOC including tests).

### 7.1 Computational cost analysis

The performance-critical path is `score_adjustment`, which fires once
per selection step. In the worst case (parallel search with 16 threads
at 50,000 iterations per second of search), this is ~16 × 50,000 =
800,000 calls per second across all workers. Each call is a single
read of a `parking_lot::Mutex<Cache>` — typical cost ~50 ns
uncontested — plus a few arithmetic ops on the cached arrays. Total
overhead estimated at ~0.2% of search throughput.

The non-hot-path `observe` runs at most every `check_interval` (=100)
iterations. It performs:
- A single pass over `n_children` (≤ 361 for Go; typically ≤ 50 for
  Gomoku) to compute σ_a values: O(K).
- A single pass for χ² with K terms: O(K).
- A single pass to compute VOI per arm: O(K).
- KL-LUCB `kl_upper`/`kl_lower` bisection at 32 iterations each, for
  K-1 arms: O(K · 32).

Total: O(K · 32) per observe call. For K = 50, this is ~1,600 fp ops,
which is microseconds. Acceptable.

### 7.2 Concurrency model

Multiple workers share a single `Arc<dyn SearchPolicy>`. The trait
methods take `&self`; internal mutability is via `parking_lot::Mutex`.
Workers compute their own selection scores via `score_adjustment`
(which only reads the cache); only the worker hitting the periodic
boundary writes the cache via `observe`.

Under contention (e.g. N workers all trying to write `observe`
simultaneously), the last writer wins. This is safe because the cache
is monotonic: as visits grow, gap_bits can only increase for a stable
best arm. A stale read produces a slightly older PAC bound, which
gives an *under-eager* halt (more samples, never fewer than necessary).

### 7.3 Test methodology

The plan specifies 12 hand-computed tests. Three of them already ship
in P06 (the σ_a smoothing tests). Four ship in P08 (KL-LUCB tests with
hand-derived β = 15.618 and gap_bits sanity values). The remaining
five — empirical Bernstein, χ², VOI, Welford f32-vs-f64, MENTS soft-
policy — are slated for P09 / P11 fixtures.

Each test pins a numerical claim to a hand-derived expected value
within an explicit tolerance (typically 1e-3 to 1e-9). This discipline
prevents the kind of "hand-computation error" that produced the
β ≈ 18.4 mistake during P06 development (corrected to 15.618 after
re-derivation from the KK13 formula).

### 7.4 Migration risk

The plan mitigates migration risk via:

- **LegacyQuartz shim** (P07, commit `965e216`): preserves bit-identical
  behavior of every published number under `--policy=legacy_quartz`.
  Already shipped.
- **Standalone policies** (P07, P08): Both LegacyAlphaZero and
  KLLUCBStop are unit-tested standalone — the engine doesn't call them
  yet. P10 lands the engine integration after the policies have been
  exercised.
- **Default flip is one commit** (P10): a single line change with
  rollback by reverting the commit.
- **CLI deprecation** (P10): old `--penalty-mode`, `--halt-mode`,
  `--cost-mode` flags continue to parse and emit deprecation WARNs
  for ≥1 release.

The riskiest part of the migration is P15 (deprecation cleanup) which
deletes ~1500 LOC of dead heuristic branches. This patch lands only
after a full release cycle of the new defaults.

### 7.5 What could derail P09

- **Hidden assumptions in the legacy controller.** The σ_Q calculation,
  RTT covariance corrections, NS gate, depth calibration — these are
  all currently load-bearing for the published Gomoku 7×7 numbers.
  BayesianQuartz drops most of them. If empirical regression on the
  same Gomoku 7×7 fixtures shows degradation, P09 will need to
  re-introduce specific corrections (the plan's adversarial-review
  section calls this out).
- **Numerical precision at extreme N.** f32 sums lose precision past
  N ≈ 10⁵; the plan stores `m2` as f64 in `EdgeView` to address this.
- **Single-game-per-process server lifetime.** σ_0 is configured per
  process via `QUARTZ_CALIBRATION_DIR` (P05, commit `4901854`).
  Multi-game runs require separate server invocations.

---

## 8. Pseudocode — both policies, hot path through halt

### 8.1 LegacyQuartz (current code, simplified)

```rust
// from src/mcts/select.rs:170-403 + src/mcts/policy/legacy_quartz.rs
struct LegacyQuartz { cfg: QuartzConfig, ctrl: Arc<QuartzController> }

impl SearchPolicy for LegacyQuartz {
    fn observe(&self, _snap, _edges) {
        // No-op. The legacy controller updates QuartzStats inside the
        // engine's own backup path. Duplicating here would either
        // no-op or double-count.
    }

    fn score_adjustment(&self, edge: EdgeView<'_>) -> ScoreAdjustment {
        let stats = self.ctrl.last_stats();        // Mutex<QuartzCtrlInner> read
        let sqrt_parent = (*edge.root_total_n as f32).sqrt().max(1e-3);
        let q_eff = edge.q;
        let prior = edge.prior;

        // Dispatch on penalty_mode (7 branches; full code at
        // src/mcts/select.rs:170-432). Sample: SelfAdaptive branch
        // (lines 175-213).
        let penalty = match self.cfg.penalty_mode {
            PenaltyMode::SelfAdaptive => {
                if edge.n > 0 || edge.o_a > 0 {
                    let m_a = 1.0 + (edge.n + edge.o_a) as f32;
                    -stats.sigma_q.max(0.001) / m_a
                } else { 0.0 }
            }
            PenaltyMode::None => 0.0,
            PenaltyMode::Legacy => {
                if self.cfg.enable_one_loop {
                    one_loop_visit_penalty(edge.n, stats.hbar_eff, 1.0,
                                           self.cfg.hbar_penalty_cap)
                } else { 0.0 }
            }
            PenaltyMode::EffectiveV2 => effective_penalty_v2(
                edge.n, edge.o_a, self.cfg.hbar_penalty_cap),
            PenaltyMode::GatedRefresh => root_share_penalty(
                edge.n, sqrt_parent, stats.hbar_eff,
                self.cfg.hbar_penalty_cap),
            PenaltyMode::GatedRefreshLegacy =>
                effective_penalty_v2(edge.n, edge.o_a,
                                     self.cfg.hbar_penalty_cap),
            PenaltyMode::PFlipMixture => {
                let nu = self.cfg.hbar_penalty_cap.max(stats.sigma_q);
                if edge.n > 0 || edge.o_a > 0 {
                    let m_a = 1.0 + (edge.n + edge.o_a) as f32;
                    -nu / m_a
                } else { 0.0 }
            }
        };

        // Optional prior refresh (5 different formulas depending on
        // mode + cfg.prior_refresh_rate; full code at lines 175-403).
        let effective_prior = compute_effective_prior(
            &self.cfg, &stats, edge.n, edge.q, edge.prior);

        let bonus = if self.cfg.enable_one_loop {
            eft_action_bonus(&stats)
        } else { 0.0 };

        ScoreAdjustment {
            effective_prior,
            penalty: penalty + bonus,
            fisher_alpha: if self.cfg.enable_fisher_puct { 0.5 } else { 0.0 },
            q_override: None,
        }
    }

    fn should_halt(&self, snap: &SearchSnapshot, _edges) -> HaltDecision {
        if self.ctrl.should_stop(snap.root_visits, snap.elapsed_ms) {
            // Map StopReason → HaltReason (see legacy_quartz.rs:97-115)
            let reason = match self.ctrl.last_stop_reason() {
                StopReason::BudgetExhausted { .. } if matches!(
                    self.cfg.halt_mode, HaltMode::Fixed { .. }
                ) => HaltReason::FixedBudget,
                StopReason::BudgetExhausted { .. } => HaltReason::MaxVisits,
                StopReason::TimeCapHit { .. } => HaltReason::MaxTime,
                StopReason::VocNonPositive { .. } => HaltReason::VOCNonPositive,
                StopReason::Converged { .. } => HaltReason::PFlipConverged,
                StopReason::MaxNodesHit { .. } => HaltReason::MaxVisits,
                StopReason::Unknown => HaltReason::PFlipConverged,
            };
            HaltDecision::Stop(reason)
        } else {
            HaltDecision::Continue
        }
    }
}

// Inside QuartzController::should_stop (src/mcts/quartz.rs:1996-2117),
// the actual halt logic for HaltMode::VOC:
fn should_stop_voc(stats: &QuartzStats, cfg: &QuartzConfig) -> bool {
    let p_flip_ok = stats.p_flip < FLIP_THRESH                   // 0.159
                    && stats.flip_stable >= FLIP_STABLE_N;       // 3
    let voc_total = max(stats.voc_focus, stats.voc_expand, stats.voc_merge);
    let n_total_ok = stats.root_visits >= cfg.min_visits;
    p_flip_ok && voc_total <= 0.0 && n_total_ok
}
```

### 8.2 BayesianQuartz (planned for P09)

```rust
// to be created at src/mcts/policy/bayesian_quartz.rs
struct BayesianQuartz {
    delta: f32,                   // 0.05 (95% PAC)
    lambda0: f32,                 // 4.0 (Beta-Binomial pseudo-count)
    min_pulls: u32,               // 30
    min_total: u32,               // 200
    max_visits: u32,
    time_cap_ms: u64,
    chi2_alpha: f32,              // 0.05
    voi_cost_floor: f32,          // 1e-3
    eval_uncertainty_kappa: f32,  // 0.5; 0 if no σ_eval available
    cached: parking_lot::Mutex<BqCache>,
}
struct BqCache {
    sigma_a: SmallVec<[f32; 32]>,
    voi_a:   SmallVec<[f32; 32]>,
    chi2: f32, dof: u32,
    gap_bits: f32, eb_gap: f32, best: u16, second: u16,
    expand_active: bool,
}

impl SearchPolicy for BayesianQuartz {
    fn observe(&self, snap: &SearchSnapshot, edges: &[EdgeView<'_>]) {
        if snap.root_visits < self.min_total { return; }
        let n_total = snap.root_visits.max(1) as f32;
        let prior_var = snap.sigma_q_root.powi(2).max(1e-8);

        // (a) per-action posterior σ via Welford + Beta-Binomial
        let mut sigma_a = SmallVec::<[f32; 32]>::new();
        for e in edges {
            let n = e.n as f32 + self.lambda0;
            let m2 = e.m2 as f32 + self.lambda0 * prior_var;
            sigma_a.push((m2 / n).sqrt().max(1e-3));
        }

        // (d) Pearson χ² envariance test — clamp prior ≥ ε to avoid
        // degenerate divisions when network outputs near-zero priors.
        let eps = 1.0 / (snap.n_children as f32 * n_total).max(1.0);
        let mut chi2 = 0.0_f32;
        for e in edges {
            let pi0 = e.prior.max(eps);
            let expected = n_total * pi0;
            chi2 += (e.n as f32 - expected).powi(2) / expected.max(1.0);
        }
        let dof = (snap.n_children as u32).saturating_sub(1).max(1);
        let chi2_thresh = statrs_chi2_inverse_cdf(1.0 - self.chi2_alpha, dof);
        let expand_active = chi2 > chi2_thresh;

        // (e) Russo-Van Roy one-step VOI
        let mut best = 0_u16; let mut best_mu = f32::NEG_INFINITY;
        for e in edges {
            if e.n >= self.min_pulls && e.q > best_mu {
                best_mu = e.q; best = e.idx;
            }
        }
        let n_b = edges.iter().find(|e| e.idx == best)
            .map(|e| e.n.max(1)).unwrap_or(1) as f32;
        let sigma_b = sigma_a[best as usize];
        let kappa = if snap.sigma_eval.is_some() {
            self.eval_uncertainty_kappa
        } else { 0.0 };
        let extra_eval_var = kappa * snap.sigma_eval.unwrap_or(0.0).powi(2);
        let s_b2 = sigma_b.powi(2) / n_b + extra_eval_var;

        let mut voi_a = SmallVec::<[f32; 32]>::new();
        let mut max_voi = 0.0_f32;
        for (i, e) in edges.iter().enumerate() {
            if e.idx == best { voi_a.push(0.0); continue; }
            let n_a = (e.n + 1) as f32;
            let s_a2 = sigma_a[i].powi(2) / n_a + extra_eval_var;
            let s = (s_b2 + s_a2).sqrt().max(1e-6);
            let delta_q = (best_mu - e.q).max(0.0);
            let z = -delta_q / s;
            let voi = (1.0 / (2.0_f32 * std::f32::consts::PI).sqrt())
                    * (-0.5 * z * z).exp() * s;
            voi_a.push(voi);
            if voi > max_voi { max_voi = voi; }
        }

        // (f) KL-LUCB halt — same as KLLUCBStop policy (P08)
        let beta = kl_lucb_beta(snap.iteration as f32,
                                snap.n_children as f32, self.delta);
        let (gap_bits, second) = kl_lucb_gap_internal(
            edges, best, beta, self.min_pulls);

        // (c) empirical Bernstein gap — sanity diagnostic only
        let log_term = (2.0 / self.delta).ln();
        let mu_b = 0.5 * (best_mu + 1.0);
        let n_b_u = edges.iter().find(|e| e.idx == best)
            .map(|e| e.n).unwrap_or(0).max(2);
        let eb_b = (2.0 * sigma_b.powi(2) * log_term / n_b_u as f32).sqrt()
                 + 7.0 * log_term / (3.0 * (n_b_u as f32 - 1.0));
        let eb_gap = best_mu - 2.0 * eb_b;

        *self.cached.lock() = BqCache {
            sigma_a, voi_a, chi2, dof, gap_bits, eb_gap,
            best, second, expand_active,
        };
    }

    fn score_adjustment(&self, e: EdgeView<'_>) -> ScoreAdjustment {
        let cache = self.cached.lock();
        let sigma_a = cache.sigma_a.get(e.idx as usize).copied()
                                   .unwrap_or(1e-3);
        // (b) principled penalty — empirical-Bernstein-aligned
        let penalty = -sigma_a / ((1.0 + (e.n + e.o_a) as f32).sqrt());
        ScoreAdjustment {
            effective_prior: e.prior,         // no refresh
            penalty,
            fisher_alpha: 0.0,
            q_override: None,
        }
    }

    fn should_halt(&self, snap: &SearchSnapshot, _edges) -> HaltDecision {
        if snap.root_visits >= self.max_visits {
            return HaltDecision::Stop(HaltReason::MaxVisits);
        }
        if self.time_cap_ms > 0 && snap.elapsed_ms > self.time_cap_ms {
            return HaltDecision::Stop(HaltReason::MaxTime);
        }
        if snap.root_visits < self.min_total {
            return HaltDecision::Continue;
        }
        let c = self.cached.lock();
        if c.gap_bits > 0.0 {
            HaltDecision::Stop(HaltReason::KLLUCBStop)
        } else if c.voi_a.iter().cloned().fold(0.0, f32::max)
                  < self.voi_cost_floor {
            HaltDecision::Stop(HaltReason::PolicyConverged)
        } else {
            HaltDecision::Continue
        }
    }
}
```

---

## 9. Implementation status — what is done, what is pending

### 9.1 Already implemented and committed

The following are in the codebase as of commit `10d05a7`:

- **P01 (commit `4070909`):** Telemetry counters + JSON
  schema_version 6. The `controller_summary.extended` block now
  carries `controller_penalty_mode_counts`, `mean_prior_refresh_rate`,
  `halt_reason_count` — fields the README claimed but Rust never
  emitted before P01. `HaltReason` enum at
  [`src/mcts/quartz.rs:108-141`](../src/mcts/quartz.rs#L108-L141) has
  reserved variants `KLLUCBStop`, `GLRCertified`, `EmpBernsteinSep`,
  `PolicyConverged` for the P08/P09/P11 policies.

- **P02 (commit `e83bbe7`):** Pre-flight hash gate (sha256_checkpoint
  + paired-seed contract drift).

- **P03 (commit `e205f69`):** Replay freshness exponential-decay
  (`exp(-mean_age / half_life_gen)`).

- **P04 (commit `17d5e5a`):** Multi-seed enforcement under
  `--research-grade`; concurrent-mode zero-SGD warning.

- **P05 (commit `4901854`):** σ_0 calibration auto-load via
  `QuartzConfig::with_calibration` and `QUARTZ_CALIBRATION_DIR` env var.

- **P06 (commit `3370f95`):** `SearchPolicy` trait, `SearchSnapshot`,
  `EdgeView`, `EdgeView::sigma_a` (the Welford+Beta-Binomial helper
  BayesianQuartz needs in component a), `kl_helpers::{bernoulli_kl,
  kl_upper, kl_lower, kl_lucb_beta}` (the KL bisection BayesianQuartz
  needs in component f), `DefaultBridgePolicy` no-op fallback.

- **P07 (commit `965e216`):** `LegacyAlphaZero` (pure PUCT, fixed
  budget) and `LegacyQuartz` (bit-identical shim around the existing
  controller, exposed via `--policy=legacy_quartz` once P10 wires the
  CLI).

- **P08 (commit `10d05a7`):** `KLLUCBStop` standalone policy. Uses
  components (e) and (f) — KL-LUCB halt rule with the KK13 β formula.
  Hand-derived test fixtures: tight-gap (Q=[0.6,0.5,0.4] at
  N=[100,50,1] → gap ≈ −0.084, Continue) and wide-gap (Q=[0.9,0.0,−0.5]
  at N=[10000,500,1] → gap ≈ +0.296, Stop).

### 9.2 Still pending

The following remain in the plan:

- **P09** — BayesianQuartz policy (the focus of this document).
  Estimated ~520 LOC, ~3.0 engineer-days. 12 hand-computed unit tests
  (5 of which are already shipping in P06/P08; 7 new for P09).

- **P10** — Engine wiring (`MctsConfig.search_policy:
  Option<Arc<dyn SearchPolicy>>` field, hot-path consumption inside
  the select loop), CLI translator (`--policy {legacy_az,
  legacy_quartz, kl_lucb_stop, bayesian_quartz, ments}` with
  deprecation aliases), default flip from `legacy_quartz` to
  `bayesian_quartz`.

- **P11** — MENTS opt-in policy (Xiao et al. 2019 soft Bellman /
  maximum entropy MCTS). Default OFF.

- **P12** — `mcts_server.rs` decomposition into 9 modules, byte-
  identical JSON output.

- **P13** — JAX rename + WARN per option (c) of the JAX-asymmetry
  resolution.

- **P14** — Pipeline contract dataclasses (`SelfPlayBatch`,
  `LearnerStep`, `ArenaResult`, `ReplayState`) + Evaluator
  uncertainty hook (Bootstrap-DQN-style ensemble σ_eval, opt-in via
  `QUARTZ_ENABLE_ENSEMBLE_UNCERTAINTY=1`).

- **P15** — Deprecation cleanup: remove the unused legacy PenaltyMode
  /HaltMode/CostMode branches once the new defaults have shipped for
  ≥1 release. Estimated −1500 LOC.

### 9.3 What this document is NOT

This document does *not* describe behavior that doesn't yet exist in
the codebase. Where it describes BayesianQuartz, the relevant phrase
is "is planned to" — not "does." A reviewer auditing the codebase
will find LegacyQuartz behavior on every search invocation today; the
machinery the plan calls "BayesianQuartz" exists only as
SearchPolicy scaffolding and standalone policies that the engine does
not yet consult.

This document also does *not* claim BayesianQuartz outperforms
LegacyQuartz on any specific task. The plan's criterion for the
default-flip in P10 is "BayesianQuartz must reproduce LegacyQuartz's
top-1 action ≥ 90% of the time on a 100-position Gomoku 7×7 fixture"
— not "BayesianQuartz must win." The empirical question of whether
BayesianQuartz is *better* is left for the experimental phase that
follows P15.

---

## 10. References

1. Bai, A., Wu, F., Chen, X. 2013. "Bayesian Mixture Modelling and
   Inference based Thompson Sampling in Monte-Carlo Tree Search."
   NeurIPS 2013.
2. Garivier, A., Kaufmann, E. 2016. "Optimal Best Arm Identification
   with Fixed Confidence." COLT 2016.
3. Kaufmann, E., Cappé, O., Garivier, A. 2016. "On the Complexity of
   Best-Arm Identification in Multi-Armed Bandit Models." JMLR 17(1).
4. Kaufmann, E., Kalyanakrishnan, S. 2013. "Information Complexity in
   Bandit Subset Selection." COLT 2013, Theorem 8.
5. Lakshminarayanan, B., Pritzel, A., Blundell, C. 2017. "Simple and
   Scalable Predictive Uncertainty Estimation using Deep Ensembles."
   NeurIPS 2017.
6. Lehmann, E.L., Romano, J.P. 2005. *Testing Statistical Hypotheses*,
   3rd ed. Springer. (Pearson χ², §14.3.)
7. Maddox, W.J., Garipov, T., Izmailov, P., Vetrov, D., Wilson, A.G.
   2019. "A Simple Baseline for Bayesian Uncertainty in Deep Learning"
   (SWAG). NeurIPS 2019.
8. Maurer, A., Pontil, M. 2009. "Empirical Bernstein Bounds and
   Sample Variance Penalization." COLT 2009.
9. Murphy, K.P. 2012. *Machine Learning: A Probabilistic Perspective*.
   MIT Press. (Beta-Binomial conjugate, §4.6.3.)
10. Osband, I., Blundell, C., Pritzel, A., Van Roy, B. 2016. "Deep
    Exploration via Bootstrapped DQN." NeurIPS 2016.
11. Pearson, K. 1900. "On the criterion that a given system of
    deviations from the probable in the case of a correlated system
    of variables is such that it can be reasonably supposed to have
    arisen from random sampling." Philosophical Magazine, 50:157-175.
12. Rosin, C.D. 2011. "Multi-armed Bandits with Episode Context"
    (PUCT). ISAIM 2011.
13. Russo, D., Van Roy, B. 2018. "Satisficing in Time-Sensitive
    Bandit Learning." Mathematics of Operations Research.
14. Silver, D., Hubert, T., Schrittwieser, J., et al. 2017. "Mastering
    Chess and Shogi by Self-Play with a General Reinforcement
    Learning Algorithm" (AlphaZero).
15. Wald, A. 1945. "Sequential Tests of Statistical Hypotheses"
    (SPRT). Annals of Mathematical Statistics 16(2):117-186.
16. Welford, B.P. 1962. "Note on a Method for Calculating Corrected
    Sums of Squares and Products." Technometrics 4(3):419-420.
17. Wu, D.J. 2019. "Accelerating Self-Play Learning in Go" (KataGo).
    arXiv:1902.10565.
18. Xiao, C., Mei, J., Müller, M., Schuurmans, D. 2019. "Maximum
    Entropy Monte-Carlo Planning" (MENTS). NeurIPS 2019.

### Internal references

- [`docs/QUARTZ_THEORY.md`](QUARTZ_THEORY.md) — the project's own
  controller theory document; §9 contains the "honest scope"
  disclosure that LegacyQuartz is a heuristic family.
- [`docs/RESEARCH_READINESS.md`](RESEARCH_READINESS.md) — the
  readiness checklist that `--research-grade` enforces.
- [`docs/ABLATION_GUIDE.md`](ABLATION_GUIDE.md) — the canonical
  ablation protocol, including the `controller_axes` preset that
  isolates penalty / refresh / shaping factors.
- [`~/.claude/plans/iridescent-giggling-bachman.md`](../../../.claude/plans/iridescent-giggling-bachman.md) —
  the v1.0 plan covering all 15 patches; contains the full
  pseudocode for BayesianQuartz that the §3 description here
  paraphrases.
- Per-patch audit notes: `audit_p01_telemetry.md`,
  `audit_p02_preflight.md`, `audit_p03_freshness.md`,
  `audit_p04_seed_enforcement.md`, `audit_p05_calibration.md`,
  `audit_p06_searchpolicy_scaffolding.md`,
  `audit_p07_legacy_policies.md`, `audit_p08_kl_lucb.md` (each
  patch in the 15-patch sequence has its own audit note per the
  per-step semantic audit protocol).

---

## 11. One-line summaries for quick scan

- **LegacyQuartz today:** a 229k-combination heuristic surface that
  empirically works for Gomoku 7×7 short-budget self-play, with no
  formal guarantees and a number of telemetry/calibration gaps that
  P01-P05 just closed.
- **BayesianQuartz tomorrow (P09):** a single named policy with
  Welford σ_a + empirical-Bernstein penalty + Pearson χ² envariance
  + Russo-Van Roy one-step VOI + Kaufmann-Kalyanakrishnan PAC stopping;
  every component has a primary citation; not novel as math, novel
  as integrated MCTS controller.
- **What will exist immediately after P10:** `--policy=legacy_quartz`
  reproduces every published number bit-for-bit;
  `--policy=bayesian_quartz` is the new default and emits all of
  `gap_bits`, `chi2`, `bayes_voi`, `eval_sigma` in
  `controller_summary.extended` so the headline claims become
  falsifiable from artifact JSON.

End of document.
