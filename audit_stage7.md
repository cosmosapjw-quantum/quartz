# Audit — Stage 7: Live-Engine Conditional Work

**Date:** 2026-07-13
**Scope:** wire the four green-lit metacognitive-lab lanes into the live
engine / online phase15 stack, run their pre-registered experiments, and close
the Stage 7 conditional gate table. Every claim is anchored to the Stage 1-6
lab verdicts (CLAIM_LEDGER; `docs/RESEARCH_PLAN_PARTB.md` Metacognitive-lab
campaign section).

This file is the **pre-registration record**. Kill/success criteria for every
lane are fixed here *before* any Stage 7 experiment runs; CLAIM_LEDGER rows
reference this file. Per-commit audit sections are appended below as work lands.

## Gate table (resolved from Stage 1-6)

| Lane | Gate input | Status |
|---|---|---|
| KG-stop `SearchPolicy` wrapper (Rust) | Stage 1 `kg_rank_risk` CI-separated regret win | GREEN — build authorized; play claim must be re-earned on real MCTS |
| H1 online halt + flip-calibration vs P_flip (Python) | Stage 3 discrimination gate `gate_pass=True` | GREEN |
| H3 backflow burst + O6 precision via forked_voc labels | Stage 2 VOC labels non-degenerate | GREEN (impl was SPECIFIED) |
| B13 research-grade verdict | unconditional residual item | OPEN |
| Danihelka ranking restoration | Stage 3 narrowing net value | NOT triggered → closed (below) |
| Constraint | Stage 5 H5 dup lane KILLED | No Stage 7 claim may cite adaptive-VL duplication reduction |

## Danihelka closure (cancel reason)

Stage 3 `candidate_morphology_lab` produced **zero** CI-separated *total*-regret
improvements from widening/narrowing (omission relief always repaid in ranking
regret under a shared budget; run_contract_hash `c02c718b…`). The condition for
restoring the guarantee-preserving Gumbel-SH ranking (`g(a)+logits(a)+σ(q̂(a))`)
— "narrowing shows evidenced value" — is therefore not met. The Danihelka
guarantee-restoration row is closed **DEPRECATED (Stage 7 gate: NOT triggered)**.
Reopen only if a future lane shows a CI-separated net-total gain from
widening/narrowing. The bracket's honest budget accounting (a separate,
IMPLEMENTED row) is untouched.

## Pre-registered kill / success criteria

### KG-stop engine smoke (E3 / C4)

Paired per position: `QUARTZ_SEARCH_POLICY=kg_stop` vs env-unset fixed halt, on
`seed_101/gen_20.pt`, positions × budgets {64,128,256} × kg_threshold
{1e-4,1e-3,1e-2}, `check_interval = max(4, budget//8)`.

- **Success (SMOKE ceiling):** some grid point achieves **mean budget saved ≥ 20%
  with top-1 argmax agreement ≥ 0.95** vs the fixed-halt decision.
- **Kill:** zero halts anywhere in the grid ⇒ "KG scale does not transfer to
  real adaptive shared-tree backups" — the wrapper stays IMPLEMENTED, the lane
  closes (echoes the KL-LUCB A1-a low-budget-unreachability history).
- **Demote:** halts fire but top-1 agreement < 0.80 at every saving level ⇒
  anti-conservative; diagnostic only, no efficacy claim.
- **Tier ceiling:** SMOKE-VALIDATED (engine result). NOT a play-strength or P2
  nn_evals claim — that must be re-earned under the Ablation Start Conditions.

### H1 flip-calibration lane (E2 / C8)

Predictors at each chunk boundary of the same trace: `s_H1 =
argmax_stability(counts(π_b, b))`, `s_Pflip = 1 − p_flip_b` (engine's own
incumbent p_flip). Outcome `y = 1[argmax(π_b) == argmax(π_holdout)]`, holdout =
ladder max (64), secondary = oracle-256. Confirmatory statistic = paired
Δagreement (H1 − P_flip) at **matched realized budget** (P_flip threshold tuned
to match H1's mean realized budget within ±5%), `paired_bootstrap_ci` (2000
resamples, seed 0). θ\* = 0.9 is the pre-registered confirmatory operating
point; {0.85, 0.95} are descriptive calibration only.

- **Kill (H1 dies):** matched-budget Δagreement CI excludes zero **in P_flip's
  favor**.
- **Survive:** CI straddles zero (match) or excludes zero in H1's favor.
- Reliability diagram (10 bins), ECE, Brier are descriptive.
- Restart-per-chunk and root-continuation rows are stratified, never pooled;
  the headline uses the majority mode.

### H3 / O6 burst precision (E2 / C9)

Burst event = B15 row with `budget_burst_triggered == 1`. Difficulty label
`hard := forked_voc.final_overturns_shallow` on the shared A4 trace bundle for
the same `(checkpoint_id, position_id)` (label uses the full ladder incl.
budgets above the decision point ⇒ different source than the entropy trigger ⇒
non-circular). Statistic = lift `P(hard|burst)/P(hard)`, position-level
bootstrap CI (2000, seed 0).

- **Kill:** lift CI includes 1 ("burst fires at the base difficulty rate").
- **Degeneracy demotion (diagnostic only, no O6 claim):** burst rate > 0.9 or
  < 0.02, or fewer than 30 pooled burst events.

### B13 research-grade verdict (E1 / C11d)

6 checkpoints (3 seed families × weak/mid/strong) × {A4, B5, B13@1.0,
B13@0.25} × budgets {8,16,32,64}, 96 positions, oracle 256, under the ported
`--research-grade` gate. Bonferroni over 2 curvatures × 3 metrics.

- **VALIDATED (improvement):** for some curvature, `delta_kl_to_oracle` CI
  excludes 0 favorably AND neither `delta_accuracy_to_oracle` nor
  `delta_topk_recall` CI excludes 0 unfavorably.
- **Kill (no efficacy):** all delta CIs straddle 0 ⇒ verdict "NON-HARMFUL, no
  efficacy"; B13 stays SMOKE-VALIDATED, the H2 lane closes.
- **Harmful:** any accuracy/topk CI excludes 0 unfavorably ⇒ demote that
  curvature.
- **Interpretation gate (not a quality verdict):** median `one_loop_top1_delta`
  at budget 64 must be below budget 8's, else "finite-N curvature" is demoted to
  "diagnostic reweighting" (honors the corrected non-monotone claim).

### voc_tightness (P3 bonus — measurement, not a claim)

H1 continuation early-stop gives positions different realized budgets ⇒ first
real `forked_voc.measure_tightness(bundles, realized_budgets)`. Report Spearman
ρ; if realized budgets collapse to one value it returns None and P3 stays
unmeasured — recorded honestly, no claim either way.

## Constraint (Stage 5)

No Stage 7 claim may cite adaptive VL duplication reduction as a rationale
(Stage 5 killed it: neither the synthetic screen nor the real engine supports
"adaptive VL lowers dup_rate"; adaptive VL's engine-default basis is the
measured ~6× virtual-loss pessimism reduction at preserved agreement).

---

## Per-commit audit log

### C0 — pre-registration + Danihelka closure

- Wrote this file (pre-registration record).
- CLAIM_LEDGER: Danihelka guarantee-restoration row → DEPRECATED with the
  Stage-3 cancel reason above; added Stage 7 SPECIFIED rows referencing this
  file's kill/success criteria; added the adaptive-VL-dup constraint row.
- No code touched; no regression run required (docs only).
