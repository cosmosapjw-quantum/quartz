# Audit Response — External Review of LEGACY_VS_BAYESIAN_QUARTZ.md

**Date:** 2026-05-04
**Trigger:** External technical audit (`report.md` at repo root) of the
v1.0 plan + the `docs/LEGACY_VS_BAYESIAN_QUARTZ.md` external-audit
document (commit `3366c00`).
**Outcome:** Mathematical errors acknowledged. The BayesianQuartz design
in P09 is replaced by **BQ++** (Bounded-rational Bayesian Free-Energy
Search Controller). Shipped work P01-P08 mostly preserved; trait
re-designed in the new plan.

This document is the per-step audit entry for the architectural pivot
itself, following the per-step semantic audit protocol. The new
implementation plan lives at
[`~/.claude/plans/bq_plus_plus_plan.md`](../../../.claude/plans/bq_plus_plus_plan.md).

---

## 1. Mathematical errors I made

All eight findings of the external review (`report.md` §2) are correct.
I list them here verbatim with the relevant code/doc location, so the
audit trail can be re-traced.

### 1.1 Configuration surface arithmetic — wrong

I wrote `7 × 4 × 3 × 2¹¹ ≈ 229,376` in three places:

- [`docs/LEGACY_VS_BAYESIAN_QUARTZ.md:54-55`](docs/LEGACY_VS_BAYESIAN_QUARTZ.md#L54-L55)
- [`~/.claude/plans/iridescent-giggling-bachman.md` §P06](../../../.claude/plans/iridescent-giggling-bachman.md)
- `audit_p06_searchpolicy_scaffolding.md` (motivating section)

The actual product is **172,032**. `229,376` would only be
reached at `7 × 4 × 4 × 2¹¹`. Additionally the boolean-flag list
contains `enable_fisher_puct` twice, so the count of *unique* flags
is 10 not 11; the surface is therefore `7 × 4 × 3 × 2¹⁰ = **86,016`**
under the more accurate count. The qualitative point (the surface
is too large to ablate cleanly) remains correct.

### 1.2 VOI sign / conservative-direction error

In my P09 plan and in §3.5 of the external-audit doc I wrote:

  `VOI(a) = φ(−Δ/s_a) · s_a`

and described dropping the second truncated-normal term as a "safer
underestimate." This is wrong on two counts. First, the proper
expected improvement under `X = Q_a − Q_b ~ N(−Δ, s²)` is:

  `E[max(X, 0)] = s · φ(Δ/s) − Δ · Φ(−Δ/s)`

The second term is *subtracted*, not added. Dropping it therefore
**overestimates** VOI in the clear-lead regime (where the two terms
nearly cancel). Second, an overestimated VOI delays halt — search
runs longer than necessary — which is the opposite of "safer."

The correct prescription is the full expected-improvement / Knowledge
Gradient approximation:

  `EI_a = s_a · φ(Δ_a / s_a) − Δ_a · Φ(−Δ_a / s_a)`

with `Δ_a = μ̂_b − μ̂_a ≥ 0`, `s_a² = σ̂_b²/(n_b + λ₀) + σ̂_a²/(n_a + λ₀)`.
For "value of one more *computation* on arm a" (the actual control
quantity), the Knowledge Gradient approximation:

  `KG_a ≈ E[max_j μ_j⁺] − max_j μ̂_j`

is what should be used; CPU-friendly implementation evaluates KG only
on the top-m candidate set and bounds the rest by `U_a − L_b`.

### 1.3 EB gap formula — wrong certificate

In §3.3 of the external-audit doc and the P09 pseudocode I defined:

  `eb_gap = best_mu − 2 · EB_b`

This uses only the best arm's bound; it is not a best-vs-runner-up
gap. The correct empirical-Bernstein certificate is:

  `g_EB = L_b − max_{a ≠ b} U_a = (μ̂_b − w_b) − max_{a ≠ b} (μ̂_a + w_a)`

where `w_a` is the per-arm Bernstein width. Stopping when `g_EB > 0`
guarantees, with probability ≥ 1 − δ, that arm b is δ-PAC best.
My version had no such property.

### 1.4 Scale inconsistency

The pseudocode mixed `[-1, 1]` and `[0, 1]`:
- `best_mu` was used as Q (in `[-1, 1]`)
- KL-LUCB internally mapped Q to μ̂ via `(Q+1)/2` (in `[0, 1]`)
- The empirical-Bernstein gap then used Q-scale `best_mu` again

Mixed scales make the Bernstein width inconsistent between policies:
the EB width formula has a range parameter `R` (range of the random
variable) which is `1` for `[0, 1]` and `2` for `[-1, 1]`. Failing
to be explicit about which scale you're in produces silent factor-of-2
errors.

The fix per the external audit's §6.1: pick `[0, 1]` as the canonical
scale (`x = (Q + 1) / 2`), do all backup mean / variance / certificate
math on `[0, 1]`, and convert back to `[-1, 1]` only when emitting
the final selection score.

### 1.5 KL-LUCB δ-PAC claim — overclaimed

The Kaufmann-Kalyanakrishnan 2013 PAC guarantee assumes iid Bernoulli
samples per arm. AlphaZero MCTS arms violate this in three ways:
- shared subtree backups (transposition, virtual loss) couple arms
- network value bias is not iid noise; it's a systematic prior
- adaptive sampling (PUCT-driven, not uniform) breaks the
  "fixed-confidence" sample-complexity argument

The external audit's correction is right: BAI-MCTS (Kaufmann-Koolen
2017) handles the depth-one abstraction carefully but still requires
"calibrated bounded/iid backup" assumptions. The PAC claim should be
downgraded to "δ-PAC under an idealized root-bandit abstraction of
calibrated bounded backups." For NN-driven value backups with bias,
the EB certificate is the safer primary halt rule; KL-LUCB is the
strong certificate for terminal Bernoulli backups (win/loss).

### 1.6 Pearson χ² formal-test claim — overclaimed

Pearson 1900 is exact under iid multinomial counts. MCTS visits are
adaptive (PUCT, virtual loss, value feedback drive the count
dynamics), so the null distribution is not χ²_{K-1} in general.
Calling it a "formal hypothesis test" with α = 0.05 is overclaim.

The fix: keep the χ² *statistic* as a prior-surprise diagnostic (a
scalar that is zero when visits exactly match the prior, larger when
they diverge), but do not advertise a p-value or 5% false-rejection
rate. The threshold for triggering the EXPAND channel becomes a
calibrated empirical decision (e.g. learn the threshold from
self-play held-out positions), not a textbook χ² inverse-CDF.

### 1.7 Concurrency monotonicity claim — wrong

I wrote that `gap_bits` is monotonic-non-decreasing for a stable best
arm, so a stale cache can only produce under-eager halt. This is
false. Three counter-cases:
- The empirical best arm `b̂` can change between observe calls, in
  which case `gap_bits` discontinuously resets.
- Even with a stable `b̂`, μ̂_b̂ can decrease as new rollouts come in
  (especially with NN value noise), shrinking the gap.
- `β(t, δ)` grows as `O(log t^α)`, which **subtracts** from
  `gap_bits` the larger `t` is. A stale cache from earlier `t` can
  show a positive `gap_bits` that is no longer valid at the current
  `t`.

The correct concurrency contract: stop decisions must be made on a
fresh snapshot, not from cache; or, the cache must include a
freshness key (`root_visits_at_observe`, `edge_version_hash`) and
the halt path must reject stale certificates.

### 1.8 Index bug risk in pseudocode

In my P09 pseudocode I used `sigma_a[best as usize]` where `best` was
typed as `u16` and conceptually was meant to be an *edge-local
position* but was sometimes treated as an *action_id*. For
Chess/Gomoku/Go these are different — chess has 4672 action slots,
Go has up to 361 action slots, but typical edge counts are much
smaller. Indexing into a per-edge `sigma_a` array with an action_id
will read out-of-bounds.

The fix: in the new design every cache array is indexed by
`edge_pos: u32` (a dense position 0..n_children), and `action_id`
is stored separately when the policy needs to communicate back to
the engine. This is enforced in the BQ++ `PolicyCache` struct.

---

## 2. Algorithmic deficiencies

The external audit's §2 also identified two systemic problems beyond
math:

### 2.1 No allocation controller

My BayesianQuartz design only addressed *when to stop*, not *what to
compute next*. For the user's stated goal — "reduce GPU NN evals per
move while maintaining play quality" — the allocation question is
the dominant one.

The BQ++ single principle directly addresses this:

> **Choose the next computation `c_t = argmax_c E[ΔR(B_t) | c] / cost(c)`,
> where `ΔR` is expected reduction in posterior simple regret.
> Stop when `max_c E[ΔR | c] ≤ cost(c)` or a `L_b > max_{a≠b} U_a`
> certificate fires.**

KL-LUCB / EB / VOI / Gumbel SH / nested-reservoir / MENTS are all
approximations of this principle, applied to specific allocation
sub-problems.

### 2.2 Hidden-win recall

A pure-PAC stopping rule certifies the best among *visited* arms.
Low-prior actions that never enter the candidate set are invisible
to the certificate. For tactical games (forced wins on Gomoku,
checkmate-in-N on Chess), this is the dominant failure mode.

The BQ++ candidate reservoir + tactical sentinel + nested-reservoir
escape channels (Phase 5 + Phase 6 of the new plan) are designed to
address this directly.

---

## 3. Architectural pivot — BayesianQuartz → BQ++

The external audit proposes BQ++ (Bounded-rational Bayesian
Free-Energy Search Controller) with a single principle (§2.1 above)
and five modules:

1. **Calibrated Belief Module** — Welford on `[0, 1]` scale,
   empirical-Bayes variance shrinkage with `λ₀` tied to value
   network calibration ECE.
2. **Confidence/Certificate Module** — empirical-Bernstein primary
   certificate `L_b > max_{a≠b} U_a`; KL-LUCB only for calibrated
   Bernoulli backups.
3. **Computation-Value Module** — Knowledge Gradient (KG)
   approximation on top-m candidates, not raw `s · φ(z)`.
4. **Candidate Reservoir / Anti-local-minimum Module** — Gumbel
   without-replacement sampling + upper-confidence challenger
   injection + tactical sentinel + nested-reservoir live set.
5. **CPU-friendly Cache Module** — `ArcSwap<PolicyCache>` immutable
   publish, edge-local indexing, no mutex on hot path.

The free-energy formalism is explicitly downgraded:
- **Path measure**, not "path integral."
- **Log-sum-exp / soft Bellman**, not "saddle-point one-loop expansion."
- **No Keldysh, Jarzynski, IIT** in the hot path; these belong in
  motivation discussion only, with explicit "analogical, not derived"
  framing.

This is consistent with how `docs/QUARTZ_THEORY.md` §9 already
characterizes the controller honestly.

---

## 4. Status of shipped work after pivot

### 4.1 Preserved unchanged

- **P01 (commit `4070909`):** telemetry counters + schema_version 6.
  No math errors here; the `controller_summary.extended` block
  remains the canonical telemetry surface and is reused by BQ++.
- **P02 (commit `e83bbe7`):** pre-flight hash gate. Independent of
  the controller redesign.
- **P03 (commit `e205f69`):** replay freshness exp-decay. Independent.
- **P04 (commit `17d5e5a`):** multi-seed enforcement. Independent.
- **P05 (commit `4901854`):** σ₀ calibration auto-load. Independent.

### 4.2 Reused with re-design

- **P06 (commit `3370f95`):** SearchPolicy trait + helpers.
  - Trait surface (`observe`, `score_adjustment`, `should_halt`,
    `telemetry`) survives, BUT internal mutability moves from
    `parking_lot::Mutex<Cache>` to `arc-swap::ArcSwap<PolicyCache>`
    so the hot path is genuinely lock-free.
  - `EdgeView::sigma_a` (Welford + smoothing) survives unchanged —
    the Beta-Binomial vs Normal-inverse-Gamma terminology will be
    clarified to "empirical-Bayes variance shrinkage" per the audit's
    §6.1. Math is correct.
  - `kl_helpers::{bernoulli_kl, kl_upper, kl_lower, kl_lucb_beta}`
    survives unchanged. Math is correct.
  - The `ScoreAdjustment` shape gains a `q_override` and the cache
    becomes the canonical place where per-edge quantities live.
    The shape is otherwise unchanged.

- **P07 (commit `965e216`):** LegacyAlphaZero + LegacyQuartz shim.
  - LegacyAlphaZero unchanged (pure PUCT identity).
  - LegacyQuartz shim: re-ported in BQ++ phase 2 to use the immutable
    cache. Behavior unchanged (still a bit-identical bridge to the
    legacy `quartz_policy_adjustment`).

- **P08 (commit `10d05a7`):** KLLUCBStop policy.
  - Re-ported in BQ++ phase 2 to use the immutable cache.
  - The δ-PAC claim is downgraded in code comments and telemetry
    field documentation: `policy_name = "kl_lucb_stop"` carries
    "best for terminal Bernoulli backups" caveat.
  - Math (the `gap_bits = N · KL − β` formula) survives unchanged.

### 4.3 Cancelled

- **P09 (BayesianQuartz):** the design as previously specified is
  cancelled. The conceptual successor is the BQ++ phases (the new
  plan covers this in 9 phases instead of 1 patch).

### 4.4 Renumbered

- P10-P15 effectively replaced by BQ++ phases 0-8 in the new plan.
  Some content (CLI translator, JAX rename, deprecation cleanup)
  carries over with minor adjustments.

### 4.5 Documents amended

- `docs/LEGACY_VS_BAYESIAN_QUARTZ.md` (commit `3366c00`): a footer
  amendment will be added pointing to this audit response and
  flagging §3.5 (VOI) and §3.3 (EB gap) and §2.1 (configuration
  count) as superseded. The doc is preserved for audit-trail integrity;
  it is not silently rewritten.
- `~/.claude/plans/iridescent-giggling-bachman.md`: the original P09
  pseudocode is preserved (audit trail). The successor plan at
  `~/.claude/plans/bq_plus_plus_plan.md` supersedes it from this
  point forward.

---

## 5. What stays from the original plan

The hygiene layer (P01-P05) was never the controversial part of the
plan. It addresses real audit weaknesses (W3 telemetry phantom claims,
W5/W6 same-NN/same-evaluator drift, W8 passive seed protocol, W10
calibration silent skip) and the math is straightforward
(SHA256 hashing, exponential decay, env-var plumbing).

The trait scaffolding (P06) was also not the controversial part.
The audit's correction is on hot-path concurrency design (mutex →
ArcSwap), not on the trait surface itself. Re-porting is mechanical.

The `LegacyQuartz` shim (P07) preserves bit-identical reproducibility
of every existing experiment under `--policy=legacy_quartz` once the
engine wiring lands in BQ++ Phase 2. This is unchanged.

`KLLUCBStop` (P08) is mathematically correct *as a stopping rule for
calibrated Bernoulli backups*. The claim's scope was overstated; the
math wasn't. P08 stays.

---

## 6. What this response is NOT

- **Not a dismissal of LegacyQuartz.** LegacyQuartz works on the
  Gomoku 7×7 short-budget regime per the README's "Current Controller
  Status" section. Its empirical performance is not in dispute; only
  its theoretical framing was.
- **Not a claim that BQ++ is novel.** Every primitive in BQ++ (Welford,
  empirical Bernstein, Gumbel SH, KG, MENTS, nested sampling) is
  established in the literature. The contribution is the
  CPU-friendly integration into AlphaZero MCTS root control with
  falsifiable telemetry. The novelty discipline established in the
  external-audit doc §6 carries over unchanged.
- **Not a guarantee that BQ++ will outperform LegacyQuartz.** Both
  the original P09 plan and the new BQ++ plan target *non-inferiority
  on play quality* with *strict reduction in NN evals per move* as
  the primary goal. Win-rate dominance is a *post-hoc* empirical
  question for the experimental phase.

---

## 7. Pointers

- New implementation plan:
  [`~/.claude/plans/bq_plus_plus_plan.md`](../../../.claude/plans/bq_plus_plus_plan.md)
- Original v1.0 plan (preserved):
  [`~/.claude/plans/iridescent-giggling-bachman.md`](../../../.claude/plans/iridescent-giggling-bachman.md)
- External audit input: `report.md` (repo root)
- Original external-audit doc:
  [`docs/LEGACY_VS_BAYESIAN_QUARTZ.md`](docs/LEGACY_VS_BAYESIAN_QUARTZ.md)
  (will be amended with a footer pointing to this response in BQ++
  Phase 0).
- Per-step audit trail for shipped work: `audit_p01_*.md` through
  `audit_p08_*.md`.

End of audit response.
