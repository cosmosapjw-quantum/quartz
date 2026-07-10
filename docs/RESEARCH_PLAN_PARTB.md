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

**Deferred (needs real traces + GPU):** register a "B13" `Phase15System`
config row and run the paired posthoc ablation vs A4/B5 with `π̄` added as
an explicit baseline. The operator is already selectable; only the config
row and the run remain.

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
