# KINETIC-17 closeout
Parent: 2d59bd4fd2419bd4bfda948f95b298f1423e578e. Research branch only.
Verdict: theoretical decomposition derived and exact-checked; native research intervention implementation-verified; fixed expansion policy NOT promoted. No P1/P2/P3 validation, no production merge, no new training, no Gomoku strength claim.

## Experimental evidence
The broad cohort contains 96 accepted new reflection families (32 each TRAIN/CAL/TEST); 93 tactical. Its TEST has zero regret for every evaluated recommendation and zero pair omission. TRAIN had one case of complete omission recovered by a third candidate. This ceiling effect motivated a separate exploratory follow-up (quiet), without changing the fixed allocation/readout policy.
Quiet: 192 frozen attempts, 178 accepted (59/59/60), with no immediate win available to either side at each generated continuation step. This is deliberately constructed, not a random natural-game sample or confirmatory test of an unchanged research plan. Both sets disjoint in their attempt-family and accepted-board identities; historical exclusion locks preserved in COHORT files.
Total 274 exact WDL roots, 1409 legal actions, 888537 independent negamax nodes, 1644 native searches, 78912 native visits and 39498 NN calls (excluding replay/parity calls), plus 548 actual parent baseline parity requests. All exact WDL signs matched Pons and the independent row-major reference.
Quiet TEST: 9/60 pair omission, third recovered all lost access in 4/9. Mean omission 0.133333 -> 0.066667 (WDL normalized to [0,1]).
Quiet TEST regret:
| Readout | cap32 | cap64 |
|---|---:|---:|
| K16N native | 0.141667 | 0.116667 |
| K16BAL pulse pair | 0.200000 | 0.150000 |
| K16BAL pulse triple | 0.150000 | 0.125000 |
| K17EXP pulse pair | 0.183333 | 0.183333 |
| K17EXP pulse triple | 0.175000 | 0.208333 |
| K17EXP final triple | 0.175000 | 0.216667 |
At cap64, on identical EXP snapshots, reduction in omission by 0.066667 is offset by increased ranking loss 0.091667 (0.050000 -> 0.141667): net regret increases 0.025. Against ordinary native baseline, expansion pulse gain at cap32 = -0.033333 (2 improved, 4 worsened); cap64 = -0.091667 (0 improved, 6 worsened). Descriptive paired family bootstrap only: [-0.116667,0.05] and [-0.175,-0.033333]. No general-population significance or matched-resource claim. CPU costs at cap32 N/BAL/EXP 1.9352/2.0292/1.9846 ms; cap64 2.6508/2.6549/2.5991 ms. Visit caps matched, not cost budgets. All candidates already have visible edges; third visited at prefix on 182/274 roots and 48/60 quiet TEST. No literal widening demonstrated.

## Validation and reproducibility
Exact rational finite theorem checks 4374 cases; Wolfram symbolic/numeric crosscheck. Native actual endpoint red-green. Independent grid replay of all 288 attempted histories, all raw readouts/visit increments/moments, WDL signs, 48 aggregate rows, four rejected malformed requests. Full locked offline Rust tests: 646 passed, 0 failed, 89 ignored. No independent native subagent facility; serial implementation, separate arithmetic/grid/oracle verification, no full internal search trace reconstruction. C07 label-import failure repaired using explicit reference-module path; original failure log preserved. R02 and C09 initial publication whitespace checks repaired before commits.
Native experiment depends on parent K16 worktree and executable, fixed model, external exact Pons executable/book, and compatible release Cargo dependencies. Preserve old unique backups/assets; current incremental bundle assumes parent base. Dropbox local filesystem preservation is not proof of remote cloud sync. Final restore receipt is maintained outside Git to avoid self-referential commit identity.

## Next research: KINETIC-18
Ranking-aware reservoir selection before additional allocation policy tuning. Separate action access, posterior calibration and decision rule; retain ordinary native action as an eligible incumbent. Compare fixed native allocation vs balanced vs expansion on same candidate sets, separate observed prefix-visited and unvisited candidates, explicitly distinguish candidate recommendation-set growth from genuine new tree edges. Build conditional joint-law estimate of WDL utility gaps using independent TRAIN data only; pre-register features, proper scores and uncertainty; require exposure in both positive and negative ranking-gap strata, freeze on CAL and assess on independent TEST. Start from exact conditional Bayes decision risk and interval-proof stop rule, not a raw-Q threshold. Include true widening only when a not-yet-visible edge exists (e.g. progressive widening game); Gomoku must use independent continuation labels, not claim exact full-board optimality. Compare actual NN/time tradeoff with equal-resource design and preserve negative results. No controller or main merge until ranking/access benefit and cost are evidenced.
