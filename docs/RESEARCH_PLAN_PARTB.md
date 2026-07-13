# QUARTZ Part B — Refined Research Plan

> **Status:** living research-design document (Part B of
> `~/.claude/plans/parallel-popping-owl.md`). It refines the CPU-friendly /
> human-GM-signature / low-MCTS-budget research direction into measurable
> lanes with pre-registered kill-criteria. It is a *plan*, not evidence:
> nothing here is `VALIDATED`.
>
> Read [`THESIS.md`](THESIS.md) first — this document operationalizes its
> four propositions. Every lane below carries its adversarial-audit
> provenance (`~/.claude/plans/parallel-popping-owl.md` §0.5 adversarial
> self-ask, §0.6 CCoT discriminating experiment, §0.7 PDR revision).

Central narrative (THESIS.md P1/P3): **implement the single principle
`E[ΔR]/cost` as game-agnostic statistics; the human-GM signatures are
measured as pre-registered *emergent predictions*, never as optimization
targets.** The discriminating signature (P3, SB3) is *compute-VOC
allocation tightness*, not node count.

Section order = increasing verification cost. Where a lane can be
partially exercised without the (currently unavailable) GPU/engine loop,
the offline-computable substrate is implemented and unit-tested now; the
online efficacy claim stays `SPECIFIED` / `PROPOSED` until a paired
ablation runs.

---

## B1 — Signature battery (measurement infrastructure)

**Deliverable (this session):** `quartz/phase15_signatures.py` +
`tests/test_phase15_signatures.py`. Game-agnostic, trace-only signature
functions with hand-derived regression pins.

The battery is split by the P3 two-pattern requirement. **Macro** metrics
are predicted skill-*invariant* (matching them is a sanity check, never a
claim). The **discriminating** metric is predicted skill-*increasing*.

| ID | Signature | Source data | Predicted skill relation | Status |
|---|---|---|---|---|
| O1 | `K_eff = exp(H(policy))` candidate concentration + its `d K_eff / d log-budget` trajectory | `trace_policies`, `trace_budgets` (present) | **invariant** (de Groot) | implemented (`k_eff`, `concentration_vs_budget`) |
| O2 | per-move budget Gini / entropy | per-move realized budgets (present) | descriptive | implemented (`budget_gini`, `budget_entropy`) |
| O3 | commit-time vs **external** difficulty | oracle top-2 margin / cross-seed disagreement | discriminating | **needs oracle** (analyst offline) |
| O4 | depth-selectivity = top-2 subtree visit share × mean max depth | small Rust telemetry addition | **invariant** (de Groot) | **needs Rust telemetry** |
| O5 | revision dynamics: `first_revision_step`, `flip_flop_rate`, `final_sparsity` | `trace_policies` (present) | descriptive | implemented |
| O6 | surprise-conditional burst precision `P(hard\|burst)/P(hard)` | burst events (H3) + external oracle | discriminating | **needs H3 + oracle** |
| **VOC** | `voc_tightness = corr(per_move_budget, voc_proxy)` | budgets (present) + **offline oracle proxy** | **increasing** (Russek 2025) | implemented (`voc_tightness`) |

### Non-circularity guard (THESIS.md P3, mandatory)

`voc_proxy` = shallow-vs-deep argmax disagreement × Q-gap, computed
**offline by the analyst from a high-budget reference oracle**. The
function `voc_tightness(per_move_budgets, voc_proxy_values, ...)` takes the
proxy as an explicit argument on purpose: the engine never computes or
consumes VOC during search. If it did, tightness would be tautological.
This guard is what makes P3 falsifiable rather than "optimize the metric".

### What each signature can and cannot decide

- O1/O4 (macro) turning out skill-*discriminating* would falsify the "node
  counts are skill-invariant" premise — the whole story would be wrong,
  not merely incomplete. That is a *good* falsifier to have.
- VOC-tightness flat across checkpoint strength falsifies the motto (P3
  falsifier b).
- The reportable signature claim is the **conjunction** (O1/O4 invariant
  AND VOC-tightness increasing) — a single-metric match is explicitly
  insufficient (THESIS.md P3).

### Wiring plan (deferred, needs engine/oracle)

1. O4 telemetry: add top-2 subtree visit share + mean max depth to the
   Rust root telemetry that already feeds `trace_policies`; surface in the
   phase15 trace artifact (schema bump).
2. O3/O6/VOC proxy: an offline analyzer pass that replays each committed
   position at a high reference budget to produce the oracle margin and
   the `voc_proxy`, then joins on `(checkpoint, position)`.
3. Study-level rollup: `budget_gini` / `voc_tightness` are computed across
   many moves at the study level, not per trace; `trace_signature_summary`
   only bundles the per-position O1/O5 signatures.

**Kill/keep for B1 itself:** B1 is infrastructure, not a hypothesis — it
cannot be "killed". Its job is to make P3 measurable. It is done when the
double-pattern (§B3 dissociation) can be computed from real traces.

---

## B2 — Algorithm lanes (my intuitions, translated to statistics)

Each lane = a hypothesis + reuse of existing experiment infrastructure +
a **pre-registered kill-criterion** + a **CCoT discriminating experiment**
(the observation that separates the correct mechanism from the
plausible-but-wrong one). Ordered by increasing verification cost. Every
lane's origin is one of my own legacy notes (`docs/legacy/`) or prior math,
not a transplanted paper; the literature enters only as baseline/compare.

| Lane | Hypothesis | Origin | Cheapest substrate | Status |
|---|---|---|---|---|
| H2 | finite-N curvature readout ("B13") | v5.0 one-loop | phase15 readout (`quartz/phase15_one_loop.py`) | **substrate implemented** |
| H1 | bootstrap argmax-stability stop | Darwinism redundancy | offline resample gate (`quartz/phase15_argmax_stability.py`) | (next) |
| H3 | backflow-triggered burst | legacy backflow | phase15 online burst (A2-a substrate) | SPECIFIED |
| H4 | concentration-scheduled batching | — | self-play throughput lane | SPECIFIED |
| H5 | LSH path-interference | — | CPU dedup lane | SPECIFIED (exploratory) |

### H2 — One-Loop finite-N curvature readout ("B13") [readout lane]

**Implemented this session:** `quartz/phase15_one_loop.py` +
`tests/test_phase15_one_loop.py` (10 passed). Wired as posthoc operator
`one_loop_finite_n`, dispatchable through `apply_system_readout` (the
`test_readout_reachable_through_apply_system_readout_dispatch` test asserts
the wiring is live — the A0-a lesson: an unreachable readout is a lie).

**Reframe (PDR item 2, §0.5).** The correction
`log π_eff(a) = log π̄(a) + curvature / max(N_a, N_floor)` is **not** an
extra Q term (that double-counts, since π̄ is already PUCT+Q-shaped and the
term diverges at `N_a → 0`). It is a **finite-N discretization curvature
correction**: the realized visit distribution π̄ is a biased,
over-concentrated estimate of the continuous policy-improvement target
(Grill et al. 2020 worst-case gap `(|A|-1)/(|A|+N)`, largest at small N).
It re-inflates under-visited arms and, by construction, `→ 0` as N grows.
Per-arm counts are reconstructed as `N_a = π̄(a)·N_total`; unvisited arms
stay at 0 (no completed-Q is available in the trace, so none is invented).

**Discriminating experiment (CCoT §0.6):** stratify the readout's effect
`one_loop_effect_kl = KL(π̄ ‖ π_eff)` by root visit count. The kill test is
now a *unit test* (`test_kill_test_effect_vanishes_as_n_grows`): the effect
falls monotonically and is `< 1e-4` by 65k visits.

**Kill-criterion.** (a) paired CI vs A4 / π̄ baseline crosses zero → no
efficacy; (b) the effect does **not** vanish at large N on real traces →
double-counting → demote to diagnostic. The `ℏ_eff` / `3-4` constant is
deliberately a single `curvature` knob until a real-trace calibration
fixes it (no blind first-principles value; §0.7 mirrors the A3-c
deferral).

**Real-trace validation (B13, GPU — DONE).** B13 is now a registered
`Phase15System` (operator `one_loop_finite_n`, same `search_overrides` as
A4 so it reuses A4's trace — trace-cache showed 4 hits / 2 misses across
{A4, B5, B13} × 2 positions, confirming same-trace pairing). Ran through
`phase15_ablation_study.py` on real RTX 3080 Ti traces at budgets
{8,16,32,64}.

**Finding (real data corrected a synthetic assumption).** The synthetic
unit-test kill test held the policy shape fixed and scaled N; on **real**
traces from a *random-init* checkpoint the policy is near-uniform and
*spreads* with budget (support 7→45, K_eff 6.6→42.5), so per-arm `N_a`
stays ≈1-2 and the full-support `one_loop_effect_kl` does **not** vanish —
it is **tail-dominated**. That makes full-support KL the **wrong**
kill-test metric in the diffuse regime.

**Adversarial-verify correction (do not overclaim).** A 6-skeptic
verification workflow (`partb-gpu-adversarial-verify`, 5/6 survived)
**refuted** the first draft of this finding: I had written that
`one_loop_top1_delta` shrinks *monotonically* with budget. That is false —
on real bundle `bed6feee` `|top1_delta|` **grows +43%** from budget 8
(0.0165) to 16 (0.0237) before netting down, because the policy *shape*
changes with budget (support grows), not just N. The **defensible**
statement is: (a) the **argmax is preserved at every budget** on every real
bundle (confirmed 8/8), and (b) the top-1 shift **net-decreases
endpoint-to-endpoint** (trace1: −0.104 @8 → −0.016 @64) and is negligible
at high budget — but it is **not** per-step monotone on diffuse policies.
Both the monotone-only-for-fixed-shape property and the real non-monotone
behaviour are now pinned by tests.

Consequences, encoded in code + tests:
1. the kill-test metric is switched to `one_loop_top1_delta` /
   `one_loop_argmax_preserved` (added to the readout metadata), not
   full-support KL. **Scope caveat (skeptic C1b):** this relabel is a
   *methodological choice* justified by argmax-preservation + full-KL
   being tail-dominated/uninformative, **not** by monotonicity; it is
   provisional until confirmed on a trained checkpoint.
2. `curvature=1.0` is too aggressive in the diffuse regime (adds ≈0.7 to
   ~45 log-terms); `0.25` keeps top-1 shift <3%. The constant needs
   calibration on a **trained** checkpoint (random-init gives
   pathologically diffuse policies — the clean kill test, and any efficacy
   verdict, must run there).

**Trained-checkpoint smoke (DONE — `docs/PHASE15_B13_TRAINED_SMOKE_20260710.md`).**
Trained a gomoku7 net (5 self-play generations, architecture-compatible)
and re-ran A4/B5/B13. Outcome: B13 flips from *harmful* (random-init,
KL +0.47) to *non-harmful, marginally KL-favorable but statistically
indistinguishable* (delta_kl −0.0008, CI includes 0; accuracy/topk
identical); the A2-b guard still refuses to crown it; argmax preserved at
every budget. `top1_delta` remains non-monotone even on the trained net
because gomoku7 root search *spreads* with budget (K_eff 4→20), so the
finite-N premise (`N_a` grows) is only partially met. **Status:
SMOKE-VALIDATED plumbing + non-negative direction, not a quality claim.**
Deferred for a research-grade verdict: longer training + multi-seed + more
positions under `--research-grade`; calibrate `curvature` (0.25 vs 1.0).

### H1 — Bootstrap Argmax-Stability Stop [stop lane]

**Implemented this session:** `quartz/phase15_argmax_stability.py` +
`tests/test_phase15_argmax_stability.py` (11 passed).

**Redesign (PDR item 1, §0.5).** The original H1 — "split the visit stream
into k independent fragments and check agreement" — was **killed** in the
audit: a shared MCTS tree + virtual loss makes the fragments
non-independent, so their agreement is trivially ~1 and meaningless. The
surviving design places a Bayesian bootstrap / Dirichlet posterior on the
visit allocation, `θ ~ Dir(n + α)`, and defines

    argmax_stability = P(argmax(θ) == argmax(n))

as a nonparametric flip-risk with **no iid-across-time or stationarity
assumption** (a posterior over the exchangeable multinomial parameter). It
is the honest translation of the legacy "Darwinism redundancy" intuition
(many resampled votes agreeing = redundant confidence). It **replaces
KL-LUCB as the low-budget primary stop** — A1-a made that certificate
correct-but-near-never-firing at 8-64 visits (CLAIM_LEDGER Module 2 row);
KL-LUCB is demoted to a high-budget / terminal-Bernoulli backup (§B4).

**Discriminating experiment (CCoT §0.6) — run BEFORE the paired lane.** The
kill test is *not* "does it stop" but "does the signal *discriminate*".
`stability_discrimination_gate` measures the spread of stability across
positions; a signal stuck ~1.0 (zero variance) is no better than the
trivial point-argmax and **fails the gate → H1 killed** before the
expensive experiment. Unit-tested both ways
(`test_discrimination_gate_passes_on_varied_positions` /
`..._fails_on_saturated_signal`), plus the core property that stability
rises with N at fixed gap (`test_stability_increases_with_n_at_fixed_gap`).

**Kill-criterion.** (a) fails the discrimination gate on real traces; (b)
in the paired lane (A4 vs P_flip vs bootstrap-stop at matched realized
budget), loses on flip-calibration — a reliability diagram of predicted
stability vs realized argmax agreement at held-out higher budget — to
P_flip.

**Deferred (needs engine/online integration):** wire the stop into the
online search-halt path (the A2-a `run_online_readout` substrate is the
natural host) so it can gate real budget, then run the calibration lane.
The offline signal + gate are complete and testable now.

### H3 — Backflow-Triggered Burst [allocation lane]

**Status:** `SPECIFIED` (needs the online engine loop; A2-a is the
substrate). Not implemented this session — it requires real search
iteration, not a trace readout.

**Strengthening (PDR item 3, §0.5).** The naive trigger "root entropy went
up → think more" is dominated by integer-visit quantization noise at 8-64
visits and is circular if difficulty is defined from the same entropy
signal. The surviving design fires a burst only under a **2-signal gate**
measured on a **smoothed posterior with a minimum-visit floor**:

    burst  ⇐  (ΔH_root > 0  on smoothed posterior)  AND  (argmax margin shrinking)

Built on the A2-a `budget_routing_signal` substrate (which already
computes sub-target instability without pre-paying for extra budget).

**Discriminating experiment (CCoT §0.6).** Correlate burst events with an
**external** difficulty proxy (oracle top-2 margin / cross-seed
disagreement — a *different source* from the entropy trigger, closing the
circularity). Burst precision `P(hard | burst)/P(hard)` (signature O6) must
exceed 1; ≈1 means the burst is firing on noise. **Kill** if precision ≈ 1
or if bursts do not track external difficulty.

### H4 — Concentration-Scheduled Batching [self-play throughput lane]

**Status:** `SPECIFIED` (needs the self-play NN-eval batching path).

**Rescope (PDR item 4, §0.5).** This is **not** a low-budget *quality*
lane — at 8-64 visits there is little batching headroom to improve
decisions. It is a **self-play throughput** lane: schedule NN-eval batch
width `W(t) = W_max · f(K_eff / K_legal)` so that a concentrated (already
decided) root does not waste NN evals. The physical `τ_D` claim is
**deleted**; only the empirical throughput description remains. Reuses the
adaptive virtual-loss infrastructure. `K_eff` is exactly the B1 O1
signature (`phase15_signatures.k_eff`), so the scheduler input is already
implemented.

**Discriminating experiment (CCoT §0.6).** Compare against a fixed
optimal batch width. Honest outcome = **quality identical, throughput
only ↑**. Claiming a *quality* improvement from batch width is the
plausible-but-wrong story. **Kill** if throughput does not improve.

### H5 — LSH Path-Interference [CPU lane, exploratory]

**Status:** `SPECIFIED`, lowest priority. Explicitly exploratory.

In-flight path MinHash → LSH bucketing → add a small `vvalue` penalty to
near-duplicate in-flight paths that edge-local virtual loss cannot see
(edge-VL only decorrelates at the branching edge, not along whole paths).

**Discriminating experiment (CCoT §0.6).** Stratify path `dup_rate` by
thread count. Only a **high-thread-count** improvement is real; at low
thread counts the sketch cost outweighs the benefit. **Kill** if
`dup_rate` does not improve over adaptive VL at any thread count.

---

## B3 — Judgment experiments (novelty defense)

These are the experiments that separate "principled" from "tuned to look
principled". They are the highest-value, highest-cost lanes and run last.

### B3.1 Double dissociation (P1 + P3)

Three arms at **matched playing strength**:
1. **single-principle** controller (`E[ΔR]/cost`),
2. a **strength-matched heuristic** with no governing principle,
3. a **signature-tuned control** that directly optimizes the O1-O6
   metrics *without* the principle.

Prediction: the GM signatures (esp. VOC-tightness) appear in arm 1 and can
be *forced* in arm 3, but arm 3 pays a generalization cost that arm 1 does
not (the signatures in arm 1 **emerge**; in arm 3 they are **imposed**).
This is the concrete test of THESIS.md P1's falsifier and P3's
"emergent, not target" claim.

### B3.2 Zero-retune transfer (P2 generality)

Run gomoku7 → gomoku15 → go9 with **every constant derived, none
retuned**: `δ` from ply count, `λ₀` from measured ECE, cost floor from
measured latency. Zero manual retune across games. A controller that needs
per-game retuning is a per-game heuristic in disguise (violates the
game-agnostic FORBIDDEN constraint). The transfer holding is the strongest
available evidence for P1's single-principle claim.

---

## B4 — Existing-axis verdicts (audit synthesis)

The audit's disposition of the pre-existing controller/candidate axes.
These are recorded as CLAIM_LEDGER rows (this session) so the verdicts are
enforceable, not just prose.

| Axis | Verdict | Rationale |
|---|---|---|
| root-only shaping | **KEEP** (substrate) | the shared substrate every lane builds on |
| P_flip stop | **KEEP** (incumbent) | H1 must *beat* it on flip-calibration to win the stop lane |
| adaptive split VL | **KEEP** (strongest evidence) | most original, best-supported axis |
| KL-LUCB / EB certificate | **DEMOTE** | A1-a made it near-never-fire at 8-64 visits; low-budget stop → H1 / P_flip; keep only as high-budget / terminal-Bernoulli backup |
| prior-refresh pre-family | **KILL** (mainline) | Gomoku7 no-refresh basin confirmed |
| halt-VOC | **MERGE** → KG-stop lane | duplicate mechanism |
| B5 (A4 alias) | **KEEP-DIAGNOSTIC** | sanity anchor; never a champion |
| B9 (argmax/tie-guard) | **KEEP-DIAGNOSTIC** | difficulty instrument |
| B12 (entropy-gated stabilizer) | **KEEP-DIAGNOSTIC** | narrow stabilizer; use when retargeting the training-target smoother |
| B4 / B6 / B7 / B8 / B11 | **KILL** | redundant or null in aligned rehearsals |
| B1 (dual-channel commit) | **MERGE** | recover `commit_confidence` only → feed the stop signal |
| B2 (root challenger) | **MERGE** → Gumbel-SH narrowing | candidate-reservoir duplicate |
| B10 (trace stabilizer) | **MERGE** → B12 | subsumed by the narrower gate |

---

## Session status (Part B substrate)

Implemented + tested offline this session (no GPU / engine loop needed):

- B0 `docs/THESIS.md` — frozen propositions.
- B1 `quartz/phase15_signatures.py` — O1/O2/O5 + VOC-tightness (15 tests).
- H2 `quartz/phase15_one_loop.py` — finite-N curvature readout, wired as
  operator `one_loop_finite_n` (10 tests).
- H1 `quartz/phase15_argmax_stability.py` — Dirichlet argmax-stability stop
  + discrimination gate (11 tests).

Every lane's *online efficacy* claim stays `SPECIFIED` / `PROPOSED` until a
paired ablation runs on real traces under `--research-grade`. H3/H4/H5 and
B3 need the engine/self-play loop and are design-only here.

### Metacognitive-lab campaign (external contribution + follow-ups)

The pulled `metacognitive` experiment family (`docs/METACOGNITIVE_EXPERIMENTS.md`)
supplies synthetic mechanism assays that gate the engine work here:

- **Bernoulli root lab → KG-stop.** The IID screen (Stage 1) is the
  pre-registered gate for the B4 `KG-stop`/`kg_rank_risk` disposition:
  `kg_rank_risk` lowered paired regret in every non-null scenario and was
  never worse, so the `kg_stop` `SearchPolicy` wrapper is **authorized to be
  built and re-tested on real MCTS** (CLAIM_LEDGER). Synthetic → not itself a
  play claim.
- **`forked_voc_lab` → B1 O3/VOC.** This lab is the offline VOC oracle B1
  flagged as missing: it labels each frozen-trace computation by realized
  root-decision change and feeds `voc_tightness`. On real random-init traces
  the label is non-degenerate. The P3 double-pattern (skill discrimination)
  still needs a trained checkpoint, and `voc_tightness` itself needs an
  online/adaptive run (fixed budget ladder → constant budget → undefined).

---

## GPU validation session (RTX 3080 Ti; torch 2.11.0+cu128)

The environment's torch was a ROCm/AMD build on an NVIDIA-only machine
(unusable GPU); swapped to `torch 2.11.0+cu128`. Then validated the full
phase15 pipeline end-to-end on real GPU traces (all runs `device: cuda`):

1. **Toy posthoc ablation** (A4,B1..B12): 52 real rows; trace-cache
   24 hits / 2 misses — real-data confirmation of the A0-b same-trace fix
   (2 positions → 2 traces, all 13 systems reuse them).
2. **H2 / B13 real-trace validation** (see B2/H2 above): registered B13,
   ran A4,B5,B13; trace-cache 4/2 (same-trace pairing holds for B13). The
   real-trace kill test corrected the metric from full-support KL to the
   decision-relevant `one_loop_top1_delta` (which vanishes with budget;
   argmax preserved).
3. **Benchmark path** (`--run benchmark`): produced continuation-vs-restart
   timing + a 3-check gate (`bundle_speedup`, `tie_aware_match`,
   `policy_kl`). Gate correctly reported `passed: false` on the random-init
   toy (speedup 1.09<1.8, KL 0.78>0.25) — the research-grade guard working,
   not a pipeline failure (`--enforce-gate` off → exit 0).
4. **A2-b analyzer** on real B13 deltas: `screening_multiplicity` applied
   Bonferroni (corrected_alpha 0.025 = 0.05/2 comparisons) + paired
   bootstrap CIs. Correctly **refused to crown B13**: accuracy Δ=0.0 with
   CI [0,0] (indistinguishable from A4/B5) and KL Δ=+0.47 with CI
   [0.040,0.912] (distinguishably *worse*). No false champion — consistent
   with the H2 finding that `curvature=1.0` inflates a diffuse random-init
   policy without helping the decision.

**Caveat carried forward:** all of the above use a *random-init* checkpoint
(smoke). A real quality verdict for B13 (and a proper H2 kill-test
stratification) needs a **trained** checkpoint where the policy
concentrates. That run is the remaining deferred item.
