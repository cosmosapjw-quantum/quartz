# K62: two-action allocation and regret decomposition

## Scientific verdict
Prespecified main TARGET-H4 performance gate False, signed utility lower > 0 True, combined benefit False; separate TARGET-H8 direction signed utility gate True. Do not retune on heldout. Parent K61 c675203fe0ce8a3c6354275b90ffd114a7f1730f; clean/remote equal and parent actual-restore receipt checked.

## Restricted action-level continuation study
All 1073 eligible H4 root occurrences in K61, five cells [160, 134, 279, 322, 178]; outcome-blind SRS 80 per cell, 400 roots, four real branch games each (H4, FORCE_B with actual H4 prefix and external rival action, H8, TARGET). Same subsequent H4 vs same-model B32; no Q-v-Q. Actual {'games': 1600, 'logical': 132226, 'moves': 15541, 'physical': 132208, 'roots': 400, 'switches': 233, 'targets': 400}. The restricted oracle is retrospective, never played as an oracle policy.

|quantity|HT population mean|finite-population design SE|raw sample sum|
|---|---:|---:|---:|
|h8_better_oracle|0.000000000|0.000000000|0.0|
|h8_extra|4.000000000|0.000000000|1600.0|
|h8_gain|0.004764678|0.008381160|1.0|
|h8_outside|0.310531221|0.019252498|128.0|
|h8_residual|0.059412861|0.008990205|27.0|
|h8_utility|-0.003312675|0.008430582|-2.27|
|harmful_switch|0.023229264|0.006647609|9.0|
|helpful_switch|0.036626281|0.008118321|14.0|
|missed_positive|0.027551258|0.005778619|14.0|
|oracle_positive|0.064177540|0.009846143|28.0|
|regret|0.064177540|0.009846143|28.0|
|residual|0.050780522|0.008702035|23.0|
|rival_negative|0.038979497|0.008556655|15.0|
|switch|0.608515843|0.020344134|233.0|
|target_extra|3.953273532|0.008913164|1582.0|
|target_gain|0.013397018|0.010654710|5.0|
|target_utility|0.005211580|0.010780711|1.796|

Exact HT regret identity verified independently: TARGET gain = restricted H4 regret minus target residual. H8 outside-pair signed residual is not regret. Candidate generation and evaluator discrimination separated. No stationarity or Bayesian guarantee.

## MAIN held-out panel
Actual {'episodes': 896, 'games': 2688, 'logical': 195420, 'moves': 21531, 'physical': 195358, 'switches': 854, 'targets': 1319}. 64 disjoint fresh families, 7 frozen models; both colors, each Quartz arm vs same-model B32.

|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|H4|896|710/0/186|0.792410714|1.675749319|15.889964789|0.894540401|0.886081085|0.778683036|
|H8|896|704/0/192|0.785714286|3.092172055|16.003154574|0.806777343|0.796504605|0.759879464|
|TARGET|896|743/0/153|0.829241071|3.141527002|16.128464203|0.805218466|0.794015336|0.802881696|
H8-H4: point score -0.006696429, saving -0.087763058, signed utility -0.018803571; lower95 score -0.015625000, saving -0.091076320, utility -0.028566964; two-sided utility CI [-0.030383928571428687, -0.007522321428571455]; gates {'absolute_saving': False, 'cpu_positive': True, 'relative_saving': False, 'score_NI': True, 'signed_utility': False}. {'diverged': 272, 'improved': 14, 'worsened': 20}
TARGET-H4: point score 0.036830357, saving -0.089321935, signed utility 0.024198661; lower95 score 0.024553571, saving -0.092873761, utility 0.012627232; two-sided utility CI [0.01028125000000002, 0.038508928571428624]; gates {'absolute_saving': False, 'cpu_positive': True, 'relative_saving': False, 'score_NI': True, 'signed_utility': True}. {'diverged': 445, 'improved': 46, 'worsened': 13}
TARGET-H8: point score 0.043526786, saving -0.001558877, signed utility 0.043002232; lower95 score 0.030133929, saving -0.006182840, utility 0.029377232; two-sided utility CI [0.027033482142857168, 0.05949330357142857]; gates {'absolute_saving': False, 'cpu_positive': True, 'relative_saving': True, 'score_NI': True, 'signed_utility': True}.
Separate exact family bootstrap lower bounds: {'TARGET-H4:saving': -0.09293065626195487, 'TARGET-H4:score': 0.024553571428571428, 'TARGET-H4:utility': 0.012390625, 'TARGET-H8:saving': -0.00619822793477295, 'TARGET-H8:score': 0.030133928571428572, 'TARGET-H8:utility': 0.02925892857142857}.
Common-prefix accounting (not causal attribution): {'H8': {'common_physical_delta': 3872, 'common_q_physical_delta': 3872, 'common_roots': 4659, 'delta_both_physical': 7074, 'delta_score2': -12, 'episodes': 896, 'equal_history': 624, 'first_divergence': 272}, 'TARGET': {'common_physical_delta': 2969, 'common_q_physical_delta': 2969, 'common_roots': 3065, 'common_target_events': 753, 'common_target_extra': 2969, 'delta_both_physical': 7375, 'delta_score2': 66, 'episodes': 896, 'equal_history': 451, 'first_divergence': 445, 'target_events': 1319, 'target_extra_physical': 5214, 'target_switch': 854}}

## ARCHIVE held-out panel
Actual {'episodes': 96, 'games': 288, 'logical': 18048, 'moves': 2073, 'physical': 18046, 'switches': 70, 'targets': 117}. 16 additional disjoint families, three distinct pretrained phase8c models, exploratory separate inference; both colors, each Quartz arm vs same-model B32.

|arm|games|W/D/L|score|Q NN/move|B NN/move|saving|CPU saving|signed utility|
|---|---:|---|---:|---:|---:|---:|---:|---:|
|H4|96|69/0/27|0.718750000|1.535087719|15.317757009|0.899783779|0.897057228|0.707812500|
|H8|96|69/0/27|0.718750000|2.939828080|15.695121951|0.812691606|0.791496638|0.697375000|
|TARGET|96|73/0/23|0.760416667|2.773087071|15.194915254|0.817499010|0.805698209|0.738520833|
H8-H4: point score 0.000000000, saving -0.087092173, signed utility -0.010437500; lower95 score 0.000000000, saving -0.093105148, utility -0.013104167; two-sided utility CI [-0.013562499999999922, -0.007020833333333365]; gates {'absolute_saving': False, 'cpu_positive': True, 'relative_saving': False, 'score_NI': True, 'signed_utility': False}. {'diverged': 25, 'improved': 0, 'worsened': 0}
TARGET-H4: point score 0.041666667, saving -0.082284769, signed utility 0.030708333; lower95 score 0.010416667, saving -0.093755192, utility 0.003395833; two-sided utility CI [0.00012500000000004174, 0.06362499999999993]; gates {'absolute_saving': False, 'cpu_positive': True, 'relative_saving': False, 'score_NI': True, 'signed_utility': True}. {'diverged': 41, 'improved': 4, 'worsened': 0}
TARGET-H8: point score 0.041666667, saving 0.004807404, signed utility 0.041145833; lower95 score 0.010416667, saving -0.006219841, utility 0.011937500; two-sided utility CI [0.01027083333333334, 0.07333333333333347]; gates {'absolute_saving': False, 'cpu_positive': True, 'relative_saving': True, 'score_NI': True, 'signed_utility': True}.
Separate exact family bootstrap lower bounds: {'TARGET-H4:saving': -0.09391637702002953, 'TARGET-H4:score': 0.020833333333333332, 'TARGET-H4:utility': 0.005375, 'TARGET-H8:saving': -0.006606653038944516, 'TARGET-H8:score': 0.020833333333333332, 'TARGET-H8:utility': 0.017895833333333333}.
Common-prefix accounting (not causal attribution): {'H8': {'common_physical_delta': 332, 'common_q_physical_delta': 332, 'common_roots': 452, 'delta_both_physical': 732, 'delta_score2': 0, 'episodes': 96, 'equal_history': 71, 'first_divergence': 25}, 'TARGET': {'common_physical_delta': 318, 'common_q_physical_delta': 318, 'common_roots': 373, 'common_target_events': 80, 'common_target_extra': 318, 'delta_both_physical': 988, 'delta_score2': 8, 'episodes': 96, 'equal_history': 55, 'first_divergence': 41, 'target_events': 117, 'target_extra_physical': 466, 'target_switch': 70}}

## Rapfi separate diagnostic
16 compact families, both colors; Q96/Q104 each vs same Rapfi B128. Actual {'Q104_events': 0, 'Q96_events': 0, 'episodes': 32, 'games': 64, 'moves': 1415, 'nn': 98112}; {'Q104': {'cpu_saving': 0.2395399862997215, 'saving': 0.3604311473758436, 'score': 0.5}, 'Q96': {'cpu_saving': 0.24324402035204395, 'saving': 0.37279115873881585, 'score': 0.46875}}; Q104-Q96 score 0.03125, CI [0.0, 0.09375]; saving -0.012360011362972267, CI [-0.04256697156885503, 0.026118035948264007]. Exploratory fixed cap diagnostic, not direct NN target transfer.

## Verification
Total main empirical games 4640, played moves 40560, physical evaluator calls 443724; NN/Rapfi distinct units, not a speed comparison. 10 separate smoke games and parity calls excluded. Rust release build, 667 passed, 0 failed, 89 ignored; existing warnings. Genuine red-green pooled test. 64 old/new parity, 16 H4 target prefix checks, 2 malformed rejected; three archive checkpoint strict load and inference, 10 distinct tensor-state hashes. Exact rational 899 cases and Wolfram 3 True; Wolfram context no specific result. NN proof/cache/target audit {'DEV': {'proof_queries': 798137, 'proof_receipts': 4812, 'early_snapshots': 3554, 'votes': 6586, 'edges': 266370, 'cache_requests': 15541, 'cache_logical': 132226, 'cache_physical': 132208, 'cache_hits': 18, 'cache_root_hits': 0, 'cache_target_hits': 18, 'common': 7297, 'target62_children': 800, 'target62_events': 400, 'target62_switches': 233, 'target62_logical': 1600}, 'MAIN': {'common': 10738, 'cache_requests': 21531, 'cache_logical': 195420, 'cache_physical': 195358, 'cache_hits': 62, 'cache_root_hits': 0, 'cache_target_hits': 62, 'proof_queries': 834657, 'proof_receipts': 5768, 'early_snapshots': 5121, 'votes': 8860, 'edges': 373966, 'target62_children': 2638, 'target62_events': 1319, 'target62_switches': 854, 'target62_logical': 5276}, 'ARCHIVE': {'votes': 804, 'edges': 34429, 'cache_requests': 2073, 'cache_logical': 18048, 'cache_physical': 18046, 'cache_hits': 2, 'cache_root_hits': 0, 'cache_target_hits': 2, 'proof_queries': 66534, 'proof_receipts': 538, 'early_snapshots': 450, 'common': 1067, 'target62_children': 234, 'target62_events': 117, 'target62_switches': 70, 'target62_logical': 468}}; independent natural exact solver record {'binary_sha256': 'f14e6906e738fbc232580cb05a7481fe2d792514975bf8e03a714dba8cc4ad74', 'independent_complete': 0, 'modes': ['DEV', 'MAIN', 'ARCHIVE'], 'receipt_counts': {}, 'roots': 0}. Rapfi audit {'common': 303, 'edges': 211197, 'proof_queries': 539914, 'proof_receipts': 580, 'votes': 1009, 'wave_receipts': 1009, 'waves': 11709}. Not all NN outputs independently re-inferred. CPU-only, 4 dev/6 main workers, no native subagents, no main merge.

## Drift watchdog and next
Original 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS reread. P1 posterior expected regret per unit computation remains the target; K62 tests fixed-suffix realized restricted action regret and pooled estimator, not posterior Bayes optimality. P2 original A1_legacy_base not substituted by B32. P3 emergent signatures open; no signature tuning. Fixed two-arm candidate and 2+2 attempts experimental. Preserve H4 if TARGET benefit gate fails. K63: separate candidate coverage from action-value inference, use development restricted oracle and H8 outside-pair evidence to decide whether to expand candidate set or improve discriminator. Do not tune heldout outcomes; validate new families and preserve the archived panel separately. Investigate zero-extra-cost H4 candidate choice against targeted computation and actual terminal utility, and use targeted proof/counterfactual continuation with same evaluator. Rapfi separate 15x15 risk/cost experiment.

Implementation issue: the first MAIN statistics run reached JSON serialization and failed on NumPy bool; corrected Python bool conversion without changing arithmetic, thresholds, sample or labels. Failure log retained. Native release test red was intentional and repaired. K62 target direction signed utility gate passes; absolute and H4-relative savings gates fail, so preserve H4 cost-first reference and TARGET as research candidate, not production promotion. No main merge.
