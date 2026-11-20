# QUARTZ COG-02 COST-MATCHED-VOC-08 | actual CPU incremental cost and stop/continue experiment

**Date**: 2026-11-21 Asia/Seoul. Baseline immutable commit `4a4d1381635393715c29d672124cfe32c08d20d5`. Branch `research/quartz-cog02-cost-matched-voc-20261121`. Native agents unavailable in this client: `native_subagent=NOT_RUN`; serial actual execution with independent readback.

## Goal and estimand

Run actual QUARTZ `MctsEngine<Connect4>` on **standard 7×6 gravity Connect Four** with 3 fixed evaluator modes (uniform, new_MLP, ConvNet). The same root is continued through cumulative 8→16→32→64 visits; the previous `COG-01B-GENERALIZATION-07` trained 640/192/192 exact-oracle-labeled positions in **40/12/12 distinct five-ply opening families**.

Observed action utility at a root, computed **only after search**:
```text
U(s,a) = sign(S_PascalPons(s,a)), S != -1000  (external exact WDL oracle)
G_b(s) = U(s,a_2b) - U(s,a_b) ∈ {-2,-1,0,1,2}
Regret_b(s) = max_legal U(s,a) - U(s,a_b)
C_b(s) = measured actual CPU search wall microseconds from current checkpoint b to 2b
```

The proposed predictor is **a supervised surrogate for conditional signed realized continuation gain** `E[G_b | X_b]`, not a Bayes preposterior KG/true Bayesian VOC. True KG nonnegativity cannot be assumed for these signed action changes; both negative and zero outcomes must be preserved. No oracle actions, labels, or future observations may enter online controller features.

## Frozen observables, learner and train/validation isolation

At each checkpoint, observable `X_b` has a fixed feature vector drawn only from current root MCTS state:
- intercept 1, stage b/64; normalized entropy of 7-column visit policy; largest visit probability, top1-top2 gap; fraction of legal actions, current evaluator calls /(b+1), ply/42, side-to-move color ±1. **No exact game-theoretic scores or future search results**.
- Search `visit_policy` comes from actual `engine.pi_target(1.0)`. The root policy may contain zero-visit actions and tie patterns; finite checks and normalization mandatory.
- 3 fixed baseline model types separately; no hidden engine mode guessed at inference time.
- 640 train roots per evaluator and 192 validation roots, all **same frozen source families** from COG-07. 192 test roots and historic 12 are not used for learner fit, threshold selection, or cost estimators.

For each mode and each transition `8→16`, `16→32`, `32→64` fit **Ridge(α=1.0)** on 640 train roots, intercept unregularized; inputs normalized to bounded fixed ranges. The target is observed exact signed `G_b` from completed resident search. Forecast next computation cost `\hat C_b` is training-family mean additional search wall time (milliseconds) for this mode and stage; distinguish forecast from realized cost. The proposed policy continues if `\hat G_b(X_b) > λ \hat C_b`; stops otherwise. Fixed candidate λ ∈ {0.0, 0.01, 0.03, 0.05, 0.1} [WDL/ms]. Choose λ **only on validation families** by maximum `mean U(stop)-0.04 mean(search cost in ms)`, ties prefer lower cost and larger λ. No test-data-based retuning. This is a bounded myopic surrogate, not a theory-optimal VOC policy.

## CPU and fairness

Use 1 Rust search thread, `MctsConfig::evaluation(2.0)`, seed17, no Dirichlet, fixed 8/16/32/64 visit checkpoints. New `cache_model=true` endpoint reuses frozen model weights **within the same Rust server process** keyed by frozen identity `(model type, canonical path, caller-supplied declared SHA256)`; scientific runner verifies model bytes SHA256 before and after. Cache misses and hits must be auditable. Cache retains one model per mode; root-level model loading cost is no longer paid for every root. Always report both *search wall-time* and full request cost (root setup plus search), treating model-load once per process separately. Terminal/TT can reduce actual evaluator invocations; a visit is not an NN call.

- Train/val full-trajectory runs: 832 roots × 3 evaluators = **2496 actual resident MCTS requests**.
- Test full-trajectory run: 192 roots × 3 modes = 576 actual requests, used only as **frozen retrospective counterfactual** data, not to choose policy; extra 12 original historic cases if resources permit.
- **Live adaptive test**: on same 192 test roots, execute a genuinely online stop/continue policy within Rust at checkpoints. No stopping decision may depend on a future checkpoint, external oracle, or labels. The source diff, actual returned stop budget, observed evaluator calls and CPU wall-time must be recorded. Test full trajectories can be used for *posthoc verification only*.
- Fair primary time comparisons: per-mode adaptive vs fixed b8,b16,b32,b64 at **actual measured mean per-root search CPU wall-time**. We do not call matched visits matched physical cost. If no fixed policy matches within ±10% mean wall-time, classify `COST_MATCHED_NOT_ESTABLISHED` and report the observed Pareto instead of fabricating comparable fairness. Stage-averaged time interpolation is only a descriptive upper/lower baseline, not proof of a realized randomized-time policy.
- Costs from Rust debug build may be noisy. Record binary/source identity and calibrate repeated runs. Strong or universal ability claims require release-build and out-of-distribution replicated tests, not this bounded stage.

## Validation and errors

- Test-first red/green for (i) online cache reuse and byte identity, (ii) features matching a Python reference, (iii) online policy cannot read future/oracle values and stops at correct checkpoints, (iv) signed negative `G`, illegal moves, cumulative evaluator calls.
- Reject full game, malformed history, invalid budgets, malformed model/coefficient vectors, nonfinite numbers, wrong sha. No fallback silently.
- Test before coding claim and after all changes: Rust full build/tests and Python research suites. Fresh source/test receipts, input/output SHA256, exact external oracle requery + mirror check at 12 fixed test-family members.
- Interpreting the result: if learned surrogate underperforms equal-time fixed budgets or degenerates to a trivial always-stop, preserve the exact negative result and do not expand into meta-audits or claim cognitive novelty.

## Bounded phases, backups, posting

A. Freeze contract, inputs, exact parent SHA and next measurable deliverables. Commit → tar.gz under `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups/` → SHA256 → push → `ls-remote` match.
B. Build and test Rust model residency/cache and online stop endpoint, Python feature/fit runner; publish as an isolated commit and backup.
C. Generate 2496 train/val complete trajectories; train/validate λ-gated surrogate from train/val only; publish costs/coefficients/evidence and backup.
D. Run live 192-root×3-mode test, frozen retrospective parity and heldout exact WDL; independently read back, then publish.
E. Conclude with scoped scientific judgment and next-stage ID; final backup and publication.

**Other worktrees/dev branches are out of scope.** Local Dropbox backup archive existence is not a verified cloud-sync acknowledgment or restoration. External AGPL source/binary/book/weights are never vendored. Related primary literature: Hay et al. (2012), https://proceedings.mlr.press/r10/hay12a.html ; Sezener & Dayan (2020), https://proceedings.mlr.press/v124/sezener20a.html ; Tolpin & Shimony (2012), https://ojs.aaai.org/index.php/AAAI/article/view/8126. These are precedents, not independent confirmation of this implementation.
