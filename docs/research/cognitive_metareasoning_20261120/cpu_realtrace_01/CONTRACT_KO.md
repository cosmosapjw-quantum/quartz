# COG-01B-CPU-REALTRACE-01 | 입력 계약과 수용 기준

Date: 2026-11-20. Base: `cc3aa1771460f5a0f608375d985fd02998861364`.
Git branch: `research/quartz-cog01b-cpu-realtrace-20261120`.
Scope: frozen **real trained NN** checkpoint CPU inference + **retrospective** independent-budget policy observations. No GPU, no new self-play, no live MCTS implementation, no publication-grade game strength claim.

## Immutable inputs and actual authority

`INPUT_LOCK.json` records 296 input files, byte sizes, SHA256: 288 phase15 trace-cache records, one 6912-line assay, one 48-position suite and six trained `.pt` checkpoints (3 seeds × gen_5/latest). Input data live in the *other* QUARTZ checkout at observed `65e7d3e9596ca259ebfe523b0c74058692b6a755`; that commit does not establish the historic generator of the results. Freeze the external input bytes. For a durable reproducible public packet, publish an extracted minimal data table and sufficient source/checkpoint hash metadata in this branch; do not pretend input weight bytes themselves were uploaded.

The six checkpoint labels `C01_gen_5,C02_latest,C03_gen_5,C04_latest,C05_gen_5,C06_latest` share 48 deliberately selected Gomoku7 positions. Traces have budgets `{8,16,32,64}` and `trace_source=fresh`. A4 assayed actions match the `argmax(trace_policies[b])` for all 1152 pairs; there are substantial **exact top-probability ties** and they must be counted. `trace_p_flips` is **excluded** as a possible lookahead-derived value. The 6912 assay rows repeat each A4 search under six posthoc systems; select **only** A4 to avoid pseudo-replication.

The position suite is *constructed using strong-reference buckets*, not an iid sample. `reference_best` and `oracle_best` are stronger-search **proxies**, not game-theoretic truth. C06_latest is the reference checkpoint itself: reference self-dependence is a limitation. Budget levels are recorded as `search_continuation=independent_restarts`: **no pair represents continuation after the previous budget**.

## Executable CPU model check

For each of the 6 trained checkpoints load `model_state_dict` and `cfg` with `torch.load(..., map_location="cpu", weights_only=True)` in an existing CPU-only PyTorch environment. Encode the 48 suite positions by `quartz.runtime_support.encode_board`, exactly as phase15. Model must have (C,H,W)=(17,7,7). Record policy logits, `softmax` and scalar NN value, and check unmasked `argmax` against the A4 assay `argmax_prior`. Do not replace unmasked prior with legality-masked prior; report this semantic difference. Record every checkpoint SHA, source code SHA, backend, batch, thread cap, output dimensions and disagreement count. Only checkpoint's actual present source pixels are an input; do not synthesize history.

## Retrospective scientific objective

Primary observational signed reference-alignment change, for a fixed checkpoint/position:
`Y_b = 1[argmax(pi_{2b}) == reference_best] - 1[argmax(pi_b) == reference_best]`, at b=8,16,32.
This is a **difference between independent reruns** and not causal marginal VOC, ground-truth regret, nor actual benefit of a resident-tree computation. It may be negative. Report totals of {-1,0,+1} per budget and how ties affect interpretation. No outcome label or future policy enters feature construction.

Compare frozen deterministic cross-fitted predictors:
- M0: per-budget train-fold mean of Y, no position knowledge.
- M1: search-only features from pi_b: top probability, top-two gap, entropy and budget indicators.
- M2: M1 + current NN initial prior *only* (probability top, gap, entropy, NN scalar value, agreement of NN argmax with pi_b argmax).
M2 does **not** model real observation-channel reliability and may perform worse. Label it "recognition feature enrichment" not a proved calibrated VOC.
- 6 fixed folds by **position ID**, all six checkpoint rows at that position held out together. Test rows and positions must not leak through source-derived features, classification labels, fitted scaler or model selection. Frozen ridge penalty and feature dimensions must be declared in the implementation; no hyperparameter search. No torch/sci-kit dependence needed to fit the small predictor.
- Report held-out MSE of signed Y, rank correlation or other fixed metric only when defined, and *retrospective* top-quartile action selection value at a fixed budget transition (8→16). Compare to M0 and M1. Its calls are nominal budget units; not real marginal NN calls or wall time. Do not claim causal improvement from such reweighting.
- Group uncertainty: 48-position cluster bootstrap with fixed out-of-fold predictions and fixed deterministic seed, conditional on the fitted cross-validation. Cluster dependency across six checkpoints and three budget transitions is preserved, no independent-row CIs.
- Inspect sensitivity excluding C06_latest, reference-linked checkpoint, and ambiguity ties. Retain adverse results.

## Non-goals and quality gates

The CPU inference mapping and exact 288 model-prior replicas are allowed to be positive evidence of checkpoint usability. Good predictive performance on mined positions is **not** evidence of universal search quality, correct prior posterior, or computational metacognition. Report negative predictor comparisons and all leaks as failures. Preserve study invariants; do not rerun old GPU campaigns or change user source branches.

Stages and backup policy: (A) freeze inputs+contract, checkpoint tar+push; (B) implement test-first CPU/outcome reader+CV analysis, checkpoint tar+push; (C) clean-code execute pinned input files, record exact model/data/run identity and outcomes, checkpoint tar+push; (D) finish adverse interpretation and next experiment handoff, checkpoint tar+push. For every stage verify remote ref with `git ls-remote` and archive SHA256. Backup directory: `/home/cosmosapjw/Dropbox/personal_projects/quartz_cog01_backups/`, local Dropbox-backed folder; no server-side sync guarantee.

Next real research after this bounded offline screen: construct true **paired resident-root** computation forks on CPU if an explicit incremental Rust/Python search API is available, or create one in a separately scoped coding stage. Performance gates require genuine additional NN calls and a grounded independent action-value reference, not budget-ladder proxy differences.
