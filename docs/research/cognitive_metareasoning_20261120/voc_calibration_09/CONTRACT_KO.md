# QUARTZ COG-02 VOC-CALIBRATION-09 | sealed cross-cohort CPU replication contract

Date 2026-11-21 Asia/Seoul. Parent source `569a3ff106a1f0b84a3cd94c189a48a8b05bcdb1`, new branch `research/quartz-cog02-voc-calibration-20261121`. Only this bounded unit; no next loop autoexecuted.

## Central question, exact costs, and caution

Does the OLD, FROZEN signed realized next-batch utility surrogate (COG02 VOC08, original 640 train-family roots; validation12 families for λ) retain its benefit over **independently live** fixed16 and validation-clock baselines on **new disjoint Connect4 opening families** using a repeatable Rust release build, without training on any new labels or adjusting coefficients, λ, MCTS policy, or model weights?

Define for a root s and legal action a: `u(s,a)=sign(S_PascalPons(s,a)) ∈ {-1,0,1}`, with -1000 invalid-column sentinel. `regret(s,a)=max_legal u(s,a)-u(s,a)`. Actual signed gain of a future resident search step is `G_b(s)=u(s,a_2b)-u(s,a_b)`; `E[G_b|X_b]` approximated by the old Ridge, not Bayes preposterior KG. `C_b` is actual **wall-clock** CPU search ms. Model coefficients and cost forecasts are **frozen** from the original debug-build training; possible forecast/domain shift when moving to release is itself part of test. Cost-aware decision still uses old validation λ.

## Cohort identity and strict isolation

Freeze *before teacher labels or new model evaluation*: `COHORT_384_FROZEN.json` generation RNG 20261123 and 20261124, two separate draws `replica_A` and `replica_B`, each 12 mirror-canonical five-ply opening families ×16 roots =192. Combined 24 families, 384 roots, each cohort 96 black and 96 white next-to-move. All older G07 64-family train/validation/TEST, COG06 192, old12 prefixes/mirror and exact canonical board states are excluded. These are **two disjoint new fixed-family cohorts**, not PRNG-based stochastic search replication or iid roots. Same source generation method may introduce systematic bias. Seed17→231 diagnostic on original fixed 24 roots found **zero changes** to action/visit policy, so alternate integer seed alone is NOT an independent search replicate and is NOT used to inflate n.

Teacher: external exact Pascal Pons `c4solver -a -b 7x6_small.book`, deadline 5s/root, global max120s; all 384 status records retained, timeout/invalid explicitly. No oracle label is permitted inside Rust. Values only joined after ALL search responses complete.

## Explicit pre-registered execution design

1. Stage A: freeze contract + 384 roots + original source SHA and checksum. Backup, commit, push, remote readback.
2. Stage B: exact standalone Connect4 solver 384 new roots; independent mirror-check 12 each cohort (24 mirror queries) with actual hashes. Publish raw result. Implement minimal new `force_stop_visits=16` Rust research command only; test-first RED/GREEN and preserve prior stop and uniform protocols. Same-root resident prefix and action/eval-call identity must pass. Publish code and tests.
3. Stage C: build RELEASE Rust `mcts_demo` with cargo offline locked, distinct target and exact binary SHA; compare debug vs release source identity. Validate old fixed supervised policies on PRIOR 192 validation roots; derive comparator clock caps using ONLY release validation actual timing (not external WDL) on grid 0.01..10ms step0.01, choosing cap minimizing absolute mean CPU search time difference from frozen adaptive. **Do not retrain or update λ, gain coefficients, or gain cost forecasts**. Publish calibrated clock caps and release validation CPU data and SHA.
4. Stage D: execute exactly 384 new roots × 3 evaluator modes (Uniform/new MLP/Conv) × (full64 for ex-post calibration, online-adaptive, true live fixed16, validation-clock stop) = **4608 actual new Rust root requests** in one release-server process. Model weights resident, cache seen after the first request. Full search responses contain future actions but are *not fed to the adaptive controller*. All exact oracle WDL values are joined only after completing all root searches. Both separately frozen cohorts and combined 24-family sample report regret, actual wall CPU cost, completed evaluator calls, end-to-end setup/load time, cost-match verdict, family-cluster conditional bootstrap, plus side and true win/draw/loss stratification. No post-test threshold adjustment.
5. Stage D extended: calibration of the FROZEN gain predictor at stages 8/16/32 on *new* full trajectories, per evaluator. Report `mean(predicted G)`, `mean(observed signed G)`, MSE/MAE, negative-gain rate, false-stop (predicted stop despite positive G), waste (continue despite nonpositive G), and **predicted next-batch ms versus release actual**. Also report 6 pre-registered gain prediction bins `(-inf,-.2],(-.2,-.05],(-.05,0],(0,.05],(.05,.2],(.2,inf)`; no test-dependent bin choice. Stratify by family and avoid treating 384 as 384 IID samples.
6. Stage E: one fresh Rust full test suite and Python research suites, independent exact external solver requery+mirror readback, source/weight/executable/output hashes; final verdict, publish, archive and next research ID.

## Allowed claims and hard gates

- The criterion `mean search time difference <=10%` was pre-set in COG02 VOC08 for a **matched-cost claim**. Apply independently to each cohort and each evaluator when comparing adaptive vs validation-clock. If it fails label `COST_MATCHED_NOT_ESTABLISHED`, even when WDL looks better.
- Live fixed16 uses real stop after 16 root visits and no continuation of the live controller. Historical fixed16 prefix from previous source is only secondary crosscheck. Do not claim release speedup until its actual measured baseline exists.
- If the release build cannot compile or 5s/120s exact teacher cap blocks sources, preserve the exact blocker and complete only feasible bounded work. Do not convert debug results into release evidence.
- Strong cost-aware benefit requires the sign of mean regret difference, time and fair baseline alignment separately per new cohort; a wide bootstrap or failed replicate remains unresolved/negative.
- Since original λ/cost forecasts were debug-calibrated, unchanged coefficients under release are a domain-shift test, not a cost-optimal deployment baseline.
- `native_subagent: NOT_RUN` unless actual host delegation exposed. No third-party AGPL source/solver/book copies published, only source hashes, outcomes and model weights originally generated in QUARTZ.
- Reusable original 15x15 Gomoku/Rapfi and attention-reopening is later stage; never silently start now.

## Primary reading

Hay, Russell, Tolpin, Shimony (2012), https://proceedings.mlr.press/r10/hay12a.html; Sezener & Dayan (2020) https://proceedings.mlr.press/v124/sezener20a.html. Both concern formal computation valuation. Our bounded signed realized gain regressor is an empirical surrogate, not the theorems' exact KG.

Backups to local `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups` per-stage `git archive` + sha256sum + GitHub push and `ls-remote` exact commit identity. Local Dropbox path is not a cloud sync ACK/restore test. Preserve `agent/local-experiment-foundry` source tree, other worktrees, and existing untracked context file.
