# QUARTZ Research Thesis (frozen)

> **Status:** FROZEN framing document (Part B / B0). This file fixes the
> research propositions so that later ablation lanes cannot silently drift
> the goalposts. It is deliberately short. It states what QUARTZ claims,
> what would falsify each claim, and what is explicitly *not* a claim.
>
> Governing discipline: every sentence here is a claim under the
> [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md) status vocabulary. Nothing in this
> document is `VALIDATED`. The four propositions are the *targets* the
> Part B lanes are designed to test, not achievements.
>
> This document was produced through the adversarial-audit trail recorded
> in `~/.claude/plans/parallel-popping-owl.md` §0 (self-discover →
> step-back → metacognitive self-ask → CoVe → adversarial self-ask →
> CCoT → PDR). The reframes below (P3 signature-as-emergent-prediction,
> the VOC-tightness non-circularity guard) are outputs of that audit, not
> fresh assertions.

---

## The one sentence

> QUARTZ is a single-principle search controller: **allocate and stop
> computation to maximize expected decision-loss reduction per unit
> compute cost, `E[ΔR(B_t) | c] / cost(c)`** — and the human-grandmaster
> *statistical* signatures we care about are predicted to **emerge** from
> that principle rather than being coded as targets.

Everything below decomposes that sentence into falsifiable propositions.

---

## P1 — Single governing principle (not a heuristic surface)

**Claim.** The controller's allocation and stopping behavior is derived
from one objective, `E[ΔR(B_t) | c] / cost(c)`, where `R(B_t) =
E_θ[max_a θ_a − θ_{b̂_t}]` is the posterior simple regret of the currently
identified best root arm. Every primitive (belief calibration, stopping
certificate, computation-value estimate, candidate reservoir) is an
*approximation of this one objective* applied to a sub-problem — not an
independent tunable knob.

**Why this is a claim and not decoration.** The predecessor design carried
a ~229k mode-combination dispatch surface (7 penalty × 4 halt × 3 cost ×
2¹¹ booleans). P1 asserts that surface is replaced by one named policy
with one hyperparameter set. See `docs/BQ_PLUS_PLUS_DESIGN.md` §1.

**Falsifier.** If a lane can only match its target metric by adding a
control that is *not* an approximation of `E[ΔR]/cost` — i.e. a
signature-tuned knob that directly optimizes an O-metric (see P3) — then
P1 is violated for that lane and the result is reported as heuristic
tuning, not principle-derived behavior. The B3 double-dissociation
experiment (a signature-tuned control with no governing principle) is the
dedicated test.

**Current status:** `SPECIFIED`. No composed single-principle policy is
wired into the engine yet (CLAIM_LEDGER: "BQ++ composed policy … exists as
code" = `SPECIFIED`). P1 is a design proposition awaiting integration.

---

## P2 — Efficiency claim (the only performance number)

**Claim (verbatim from the repo objective).** ≥30% reduction in
`nn_evals_per_move` at **non-inferior play quality** on Gomoku 7×7
short-budget self-play, versus the safest existing hand-written anchor
(`A1_legacy_base` / `legacy_quartz`). Source: README "Current Controller
Status" and objective (README:235-236); `docs/BQ_PLUS_PLUS_DESIGN.md`
§4-§5.

**Definitions fixed here (so the number cannot be gamed).**
- *Efficiency* = `nn_evals_per_move`, not wall-clock (wall-clock is a
  CPU-friendliness engineering gate, P4, not the research number).
- *Non-inferior play quality* = paired-seed evaluation-matrix score-rate
  whose confidence interval's lower bound does not fall below the
  baseline's point estimate by more than a pre-registered non-inferiority
  margin. Uses the existing `docs/RESEARCH_READINESS.md` `--research-grade`
  gate (multi-seed, paired protocol, CIs, benchmark-safe path).
- *Budget fairness* = comparisons are effective-budget-normalized; any
  extra budget (e.g. H3 burst) is reported via `budget_burst_triggered` /
  `extra_budget_used`, never hidden.

**Falsifier.** Any of: (a) the 30% reduction is achieved but the
non-inferiority CI fails; (b) non-inferiority holds but the reduction is
<30%; (c) the reduction depends on a comparison that is not
effective-budget-normalized. Each is a documented failure row, not a
silent omission.

**Current status:** `SPECIFIED` target (CLAIM_LEDGER row: "BQ++ ≥30%
nn_evals_per_move reduction … is a target"). No paired ablation exists.

---

## P3 — Human-GM statistical signatures are *emergent predictions*, not targets

This is the proposition most at risk of "metareasoning rebranding" and
"de Groot refutation" attacks. The audit (§0.2 SB3, §0.5) forced the
reframe below.

**Claim.** If P1 holds, then a specific *pre-registered battery* of
game-agnostic search statistics (the O1–O6 signatures + the VOC-tightness
discriminator, specified in Part B / B1) will move in **pre-registered
directions as a byproduct** of principled allocation — without those
statistics ever being optimization targets.

**The two-pattern requirement (the actual falsifiable content).** Drawing
the distinction the cognitive-science literature forces:
- **Macro-structure metrics** (candidate-set size, search depth,
  node-count analogues; here O1 `K_eff`, O4 depth-selectivity) are
  *skill-invariant* — de Groot's classical result. QUARTZ predicts these
  do **not** cleanly separate strong from weak checkpoints. Matching them
  is a **necessary but not sufficient** sanity check, never a claim.
- **VOC-tightness** — the correlation between per-move compute and the
  *value of computation* on that move (Russek 2025-style) — is
  *skill-discriminating*. QUARTZ predicts tightness **increases** with
  checkpoint strength.

The claim is the **conjunction**: macro metrics skill-invariant AND
VOC-tightness skill-increasing (a *double pattern* / dissociation). A
single-metric match is explicitly insufficient and will not be reported as
signature evidence.

**Non-circularity guard (audit §0.5, mandatory).** VOC-tightness is an
**offline evaluation metric computed by the analyst with a high-budget
reference oracle**. It is **not** a runtime control input. The engine never
computes or consumes VOC during search; if it did, tightness would be
tautological. This guard is what separates P3 from "just optimize the
signature."

**Falsifier.** Any of: (a) macro metrics turn out skill-discriminating
(then they were doing hidden work — the story is wrong); (b)
VOC-tightness is flat across skill (no discriminating signature — motto
fails); (c) tightness only appears when the engine is fed VOC at runtime
(circular — disqualified by the guard).

**Current status:** `PROPOSED`. The battery is a measurement-infra spec
(B1); no dissociation has been run.

---

## P4 — CPU-friendliness is an engineering gate, not a research claim

**Claim.** The controller's hot path (`observe` / `should_halt` /
`score_adjustment`) stays within a CPU compute budget compatible with
short-budget (8–64 visit) self-play: edge-local indexing, bounded
allocation, no per-iteration global lock on the wired path.

**Explicitly demoted.** "CPU-friendly" is a **pass/fail engineering
gate**, not a scientific result and not part of the efficiency claim P2.
It gates whether a lane is *admissible*, nothing more. (Audit §0.2 SB2:
the three motto legs are one principle's consequence, and this leg is the
engineering-admissibility filter.)

**Reality check baked in (CLAIM_LEDGER A4-b).** The advertised ArcSwap
"no-mutex" `PolicyCache` is `SPECIFIED` and **not wired**; the actually
wired policy (`KLLUCBStop`) uses a `parking_lot::Mutex` for its small
cached state. P4 is measured against the *wired* path's real cost, not the
aspirational cache design.

**Falsifier.** A lane whose hot path exceeds the per-visit CPU budget at
the target visit counts is inadmissible until optimized — regardless of
its play quality.

**Current status:** engineering gate, tracked in CLAIM_LEDGER per-module
rows.

---

## What is explicitly NOT claimed (FORBIDDEN — carried from CLAIM_LEDGER)

- **No game-specific strategy / rule / pattern injection.** Every
  candidate is a function of root/search statistics only. Game knowledge
  enters solely through the NN prior and the `GameState` trait's
  `is_winning_move` / terminal semantics (which every game already
  defines).
- **No literal quantum / thermodynamic superiority claims.** Legacy
  physics motifs (`docs/legacy/`) are *idea sources* for statistical
  observables only. Any surviving physical term is a name with a
  schema-versioned alias, not a mechanism claim.
- **No node-count mimicry as the goal.** Per de Groot, matching human
  node counts / depth is skill-invariant and therefore not evidence of
  grandmaster-like search (P3).
- **No claim above its evidence tier.** This document's four propositions
  are `SPECIFIED` / `PROPOSED` targets. Promotion requires the
  CLAIM_LEDGER "Ablation Start Conditions".

---

## Pointers

- Governing principle detail: [`BQ_PLUS_PLUS_DESIGN.md`](BQ_PLUS_PLUS_DESIGN.md) §1-§3
- Claim firewall: [`CLAIM_LEDGER.md`](CLAIM_LEDGER.md)
- Readiness gate: [`RESEARCH_READINESS.md`](RESEARCH_READINESS.md)
- Signature battery + algorithm lanes (Part B): `RESEARCH_PLAN_PARTB.md`
- Adversarial audit trail that produced this framing:
  `~/.claude/plans/parallel-popping-owl.md` §0
