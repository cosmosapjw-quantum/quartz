# NN-DECISION-02: identifiable signed gain and native allocation

## Decision
**Research cycle complete; learned controller NOT ADMITTED.** Direct base 87c15dc828d6eea519c219fd8a412fa21dfc464d. No main merge. Existing sibling NN-DECISION-01, CHET-01 and historical COG branches are evidence, not replacement authority. This cycle adds an explicit research-only command with existing two-thread MCTS and the previously trained COG07 Conv3x3 evaluator; it neither retrains the network nor changes standard Quartz control. Full R01–R10 and C01–C10, plus pre-label correction C07A and blind-before-label checkpoint C08A. Every commit is published and backed up. R07 2,227 rational checks; C04/C05 red/green Python fixture, C06 release build and native smoke, C09 focused Rust test (test-only leading-dot literal syntax corrected, release behavior unchanged). No full Rust test suite claim.

## Source and exposure
Attempt cap 1800: 1768 complete, 32 interrupted unrecorded charged; 1184/292/292 completed TRAIN/STRESS/NATURAL. The C07A allocation correction used only observed source yield before exact labels and training traces. Preserve both logs and all 18511 generation events; do not claim pure natural game distribution. Distinct five-ply opening families modulo reflection and exact boards excluded versus 208 prior openings and 3148 prior canonical boards, checked independently. Targets 144/48/48; actual 104/15/44, both sides represented. All 163 exact Pons WDL vectors obtained, no imputation; nine fresh reflection queries independently matched. TRAIN and STRESS have no current-player immediate win; NATURAL has 22 immediate-win roots (unfiltered for this property, still exploratory trajectories and ply-selected). 1757 completed trajectories terminated and 11 reached depth cap.

104 TRAIN roots x two seed traces = 208 observations, cumulative 4/8/16/32 visits. 4→16: gain -1:8, -0.5:2, 0:182, +0.5:2, +1:14. Thus 26/208 nonzero, mean 6/208=0.02884615 utility, 13 distinct source roots. 4→32: 30/208 nonzero, 16 distinct roots with nonzero 16 OR 32 (required >=20; FAIL). This breaks the preceding cycle's all-zero pathology but does not meet its exposure requirement. Seven of sixteen runtime cells occupied. Exact empirical 8-observation shrinkage recomputed; empty cells use global fallback; no learned ranking in those cells independent of the prior. Costs measured in microseconds, not neural evaluations substituted for time. Frozen trained controller uses g/c, a heuristic not a proven optimal adaptive knapsack.

## Prospective results
2 seeds 17/43, 20 native requests, 590 root-arm traces. Raw blind checkpoint 23d6ddb7df47b9a940489722671534adfd19ab64 posted before heldout labels. Exact action utility U=(sign(score)+1)/2; lower regret better. Two seeds averaged per family, not treated as independent source families.

| Stratum | Arm | Final regret | Within-arm utility gain | Mean whole ms | Mean optional selections |
|---|---|---:|---:|---:|---:|
| STRESS | PREFIX | .166667 | .000000 | 47.257 | 0 |
| STRESS | FIXED8 | .116667 | .050000 | 53.085 | 12.5 |
| STRESS | RANDOM16 | .200000 | -.033333 | 51.8765 | 7 |
| STRESS | ENTROPY16 | .100000 | .066667 | 48.614 | 7 |
| STRESS | SIGNED16 | .166667 | .000000 | 49.5185 | 7.5 |
| NATURAL | PREFIX | .125000 | .000000 | 39.104 | 0 |
| NATURAL | FIXED8 | .068182 | .056818 | 71.985 | 44 |
| NATURAL | RANDOM16 | .073864 | .051136 | 139.786 | 29 |
| NATURAL | ENTROPY16 | .068182 | .056818 | 85.007 | 24 |
| NATURAL | SIGNED16 | .079545 | .045455 | 77.195 | 30.5 |

SIGNED minus RANDOM advantage defined as regret_random-regret_signed: stress +.033333 (15 families, descriptive paired bootstrap [-.133333,.200000]); natural -.005682 (44, [-.0625,.045455]). Both cross zero; no superiority. ENTROPY16 is descriptively better than SIGNED16 in both. Stress whole elapsed mismatch 4.55% passes primary <=10%; natural 44.78% fails even though SIGNED is faster. Optional allowance 5.734/16.819 ms, time accounting includes pool setup and drop; observed overhead is large and variable. Maximum reported overrun and exact times in raw/summary. Do not infer strict budget, causal speedup, or equal-time result. Stronger baseline needed. Gate exposure FAIL, natural time FAIL, decision evidence INCONCLUSIVE; no promotion.

## Actual 15x15 reuse
Four new legal exploratory prefixes at 7,8,9,10 plies, both colors; use previous actual Torch checkpoint SHA 57b93ba867ff8f9b74f91adef6b38f02f94de0bd7dc177785d668d32e1dcc3ae and prior verified IPC binary SHA 590ef926224d5488e4c6c136833446faff5bc983f6fa2e310fb385650452fc04, not new Connect4 binary. Each resident root continued 8→24 with unchanged root hash, legal action, exact incremental native evaluator accounting and actual 9+16 NN rows; total 100 rows in four final-root traces (additional generation calls separate). Integration only; no Gomoku utility, Elo, learned allocation transfer or training claim.

## Theory and origin watchdog
Headroom/downside bounds and signed probability decomposition are derived for exact root utility; the fixed-option knapsack holds for exogenous known discrete costs and additive expected gains, not arbitrary dependent adaptive searches or random time. Original Gibbs prior and finite-temperature search motivation, KL and VOC, same mean CPU time and stronger play retained as objective. No claim of new thermodynamic law. Key scientific result is an empirically observable but sparse intervention mechanism, including negative gain, not predictive confidence as a substitute for decision benefit. Original goal remains unmet; TT geometry, full QUARTZ FOCUS/EXPAND/MERGE and broad gameplay remain open. Native subagent unavailable; independent bitboard and rational fit/score audit used instead.

## Next: NN-DECISION-03 (intervention support and stable timing)
1. Keep this cohort and all previous exact boards/families immutable. New opening-family reserve and split budgets before generation; use a predeclared mixture of exploratory neural selfplay plus legal nonterminal tactical construction, with strata separated. No heldout label filtering. On TRAIN only, screen exact headroom H and downside L and both 4→16,4→32 transition signs; log attempted denominator and avoid claiming population calibration from enriched sample. Aim for >=40 nonzero distinct TRAIN families (>=20 is minimum, not guarantee), with a fixed cap; hold out untouched independent families.
2. Investigate pool allocation overhead directly: profile engine initialization/arena drop, worker scheduling, cache, and transport separately, keep resident session workers live for sequential batches. Implement no-op/prefix matched overhead measurements and an independent timing calibration split. Freeze quantized time budgets and repeat in interleaved order; compare whole elapsed <=10% in each stratum, plus optional elapsed and evaluation counts.
3. Retain PREFIX/FIXED8/ENTROPY16/RANDOM16; signed/cost must beat best eligible baseline at matched wall cost rather than just random. Assess heldout calibration of nonzero probabilities and signs; verify with exact labels, avoid sparse-cell pseudo-replication. If data support and stable cost exist, compare finite-budget multi-option DP against g/c; otherwise no controller promotion. Do not change core production policy until evidence.
4. Transfer only after Connect4 gate passes. For 15x15 reuse same actual neural IPC/parallel engine, add independent paired seed gameplay with predefined opponent and equal time, report exact integration separately from empirical win-rate. No exact WDL claim for Gomoku15. Keep old weights immutable unless a new training experiment is explicitly scoped.
