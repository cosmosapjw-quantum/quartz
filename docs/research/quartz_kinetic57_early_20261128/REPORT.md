# K57 early root information and decision value

## Authority and theory
Parent 76acccfeccbf587a0e588f0e50f47cf2772b3ce2 clean, remote equal,145-file restore verified. Reread 15f3a198 README/QUARTZ_THEORY and 7241ad0c THESIS. Bayesian simple regret decrease equals conditional E[theta_new-theta_old], by tower, factorizes flip probability and signed changed-action benefit. Jensen nonnegative for posterior-mean maximizing decision only; arbitrary PUCT not guaranteed. Actual finite opponent paired gain is not minimax regret. Finite checks63 factor,625 Jensen,396 boundary enumeration,20 cache accounting; Wolfram4 True. R10 4479e56b04fe47b88658f168788a032a76f31dbe. R01/R03 whitespace failures repaired.
## Implementation
Native source 4356929238d77d5c0fe3a9fa6856a9b2e48096c2, SHA 2d50fb465ffee39b2ad347c71d6cdcedfe817c9413ab069ecf70e9dd307a3f17, rustc 1.99.0 (b940084d7 2026-09-28). First noncert Q turn only, assigned h=4/8/12/16, common Q20 prefix. Pre-lock snapshots at reached horizons, no extra evaluator calls; if locked at h or before or no nondominated rival, all arms continue unchanged Q20. Otherwise actual STOP at h, COLD fresh child5, REUSE retained child5, Q20 reference continues. Exact proof intervals and physical cache preserved, cap h intentional stop not original lock. Native red-green and full regression663/0/89. Actual56 old/new Q20 parity,52 eligible; counters {'reached': 52, 'roots': 56, 'old_nn': 1076, 'control_nn': 1076, 'early_nn': 2723}. EpochMemo57 separate fallible synchronous cache seam, test verifies failure not cached, capacity and generation invalidation; no integration with real async or mutable weights. GPU/native subagent unavailable, no training/main merge. parallel6 NN/4 Rapfi; available 16889516 KiB.
## Fresh NN
64 families,32 TRAIN/16 CAL/16 TEST, each horizon balanced within split, seven fixed checkpoints and paired colors, four actual Q-v-B32 policies=3584 games. Symmetric exact cache. 896 paired episodes including all no-event. Family bootstrap8000. Early policy effect not same as late K52.
|arm|W/D/L|score|NN saving|NN lower95|CPU saving|gate|
|---|---|---:|---:|---:|---:|---|
|cold|726/0/170|0.810267857|0.553891816|0.539056233|0.556930664|{'CPU': True, 'NN': True, 'score': True}|
|control|723/0/173|0.806919643|0.535751253|0.522673141|0.531531106|{'CPU': True, 'NN': True, 'score': True}|
|reuse|729/0/167|0.813616071|0.540129487|0.524641275|0.536959354|{'CPU': True, 'NN': True, 'score': True}|
|stop|725/0/171|0.809151786|0.623295333|0.606869465|0.620532900|{'CPU': True, 'NN': True, 'score': True}|

|contrast|reached|changed|gain points|local physical delta|target extra|base delta|net Q|net total|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|cold|586|328|1|5197|5197|0|4100|2912|
|cold_vs20|586|370|3|-396|5197|-5593|-1574|-2498|
|reuse|586|317|4|6021|6021|0|4941|3588|
|reuse_vs20|586|359|6|428|6021|-5593|-733|-1822|
|stop|586|140|2|-5593|0|-5593|-5674|-5410|
cold: paired score 0.0011160714285714285, CI [-0.013392857142857142, 0.014508928571428572], factor {'m': 0.003048780487804878, 'mu': 0.0017064846416382253, 'p_flip': 0.5597269624573379}; local gain/call {'CI': [-0.0022920009168003667, 0.0025848698948819577], 'nonpositive_resamples': 0, 'point': 0.00019241870309794111}.
cold_vs20: paired score 0.0033482142857142855, CI [-0.011160714285714286, 0.017857142857142856], factor {'m': 0.008108108108108109, 'mu': 0.005119453924914676, 'p_flip': 0.6313993174061433}; local gain/call {'CI': None, 'nonpositive_resamples': 7232, 'point': None}.
reuse: paired score 0.004464285714285714, CI [-0.008928571428571428, 0.017857142857142856], factor {'m': 0.012618296529968454, 'mu': 0.006825938566552901, 'p_flip': 0.5409556313993175}; local gain/call {'CI': [-0.0014124293785310734, 0.0026371308016877636], 'nonpositive_resamples': 0, 'point': 0.0006643414715163594}.
reuse_vs20: paired score 0.006696428571428571, CI [-0.0078125, 0.021205357142857144], factor {'m': 0.016713091922005572, 'mu': 0.010238907849829351, 'p_flip': 0.6126279863481229}; local gain/call {'CI': None, 'nonpositive_resamples': 702, 'point': 0.014018691588785047}.
stop: paired score 0.002232142857142857, CI [-0.005580357142857143, 0.011160714285714286], factor {'m': 0.014285714285714285, 'mu': 0.0034129692832764505, 'p_flip': 0.23890784982935154}; local gain/call {'CI': None, 'nonpositive_resamples': 8000, 'point': None}.

|assigned h|reached|STOP vs Q20 gain|COLD vs STOP gain|REUSE vs STOP gain|COLD vs Q20 gain|REUSE vs Q20 gain|
|---|---:|---:|---:|---:|---:|---:|
|4|125|0|1|2|1|2|
|8|202|3|0|0|3|3|
|12|131|-1|-4|-4|-5|-5|
|16|128|0|4|6|4|6|
Actual first unresolved root reasons {'eligible': 586, 'lock_before_horizon': 23, 'no_noncert': 287}; snapshot count {'4': 609, '8': 609, '12': 559, '16': 559}; support {'12': {'distance': {'max': 5, 'min': 1, 'unique': 5}, 'leader_value': {'max': 0.15932761132717133, 'min': -0.4474954307079315, 'unique': 509}, 'prior_gap': {'max': 0.020066593837340687, 'min': 0.0, 'unique': 369}}, '16': {'distance': {'max': 4, 'min': 1, 'unique': 4}, 'leader_value': {'max': 0.18729740381240845, 'min': -0.4474954307079315, 'unique': 521}, 'prior_gap': {'max': 0.020066593837340687, 'min': 0.0, 'unique': 369}}, '4': {'distance': {'max': 4, 'min': 1, 'unique': 3}, 'leader_value': {'max': 0.3851686082780361, 'min': -0.44588813185691833, 'unique': 512}, 'prior_gap': {'max': 0.020066593837340687, 'min': 0.0, 'unique': 252}}, '8': {'distance': {'max': 8, 'min': 1, 'unique': 7}, 'leader_value': {'max': 0.41478120908141136, 'min': -0.4474954307079315, 'unique': 536}, 'prior_gap': {'max': 0.020066593837340687, 'min': 0.0, 'unique': 364}}}. Assigned h changes base computation and survivor population, not simply one common cohort followed to every h. No-event included; local early-vs-Q20 cost may be negative, ratio only positive denominator.

## Factorized conditional predictor
TRAIN-only model {'arms': {'cold': {'cells': {'12:0': {'cost': 561, 'flip': 37, 'gain2': -4, 'm': -0.04878048780487805, 'mu': -0.02851782363977486, 'n': 63, 'p': 0.5846153846153846}, '12:1': {'cost': 17, 'flip': 2, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 5, 'p': 0.42857142857142855}, '16:0': {'cost': 480, 'flip': 30, 'gain2': 8, 'm': 0.11764705882352941, 'mu': 0.06881243063263041, 'n': 51, 'p': 0.5849056603773585}, '16:1': {'cost': 41, 'flip': 6, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 9, 'p': 0.6363636363636364}, '4:0': {'cost': 551, 'flip': 35, 'gain2': 2, 'm': 0.02564102564102564, 'mu': 0.016194331983805668, 'n': 55, 'p': 0.631578947368421}, '4:1': {'cost': 200, 'flip': 11, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 21, 'p': 0.5217391304347826}, '8:0': {'cost': 669, 'flip': 41, 'gain2': -4, 'm': -0.044444444444444446, 'mu': -0.025925925925925925, 'n': 70, 'p': 0.5833333333333334}, '8:1': {'cost': 129, 'flip': 6, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 21, 'p': 0.30434782608695654}}, 'constant': {'mu': 0.003389830508474576, 'p': 0.5694915254237288}, 'total': {'cost': 2648, 'flip': 168, 'gain2': 2, 'n': 295}}, 'reuse': {'cells': {'12:0': {'cost': 643, 'flip': 34, 'gain2': -4, 'm': -0.05263157894736842, 'mu': -0.02834008097165992, 'n': 63, 'p': 0.5384615384615384}, '12:1': {'cost': 52, 'flip': 3, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 5, 'p': 0.5714285714285714}, '16:0': {'cost': 522, 'flip': 28, 'gain2': 8, 'm': 0.125, 'mu': 0.06839622641509434, 'n': 51, 'p': 0.5471698113207547}, '16:1': {'cost': 95, 'flip': 7, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 9, 'p': 0.7272727272727273}, '4:0': {'cost': 568, 'flip': 33, 'gain2': 4, 'm': 0.05405405405405406, 'mu': 0.03224276908487435, 'n': 55, 'p': 0.5964912280701754}, '4:1': {'cost': 239, 'flip': 9, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 21, 'p': 0.43478260869565216}, '8:0': {'cost': 704, 'flip': 39, 'gain2': -4, 'm': -0.046511627906976744, 'mu': -0.025839793281653745, 'n': 70, 'p': 0.5555555555555556}, '8:1': {'cost': 222, 'flip': 6, 'gain2': 0, 'm': 0.0, 'mu': 0.0, 'n': 21, 'p': 0.30434782608695654}}, 'constant': {'mu': 0.006779661016949152, 'p': 0.5389830508474577}, 'total': {'cost': 3045, 'flip': 159, 'gain2': 4, 'n': 295}}}, 'episodes': 448, 'lock': '0a296bd464bd0f85f284a8e6f3a5eea257941ce738932f08ff776be1d8ac3ec9', 'reached': 295, 'scope': 'TRAIN only. Beta(1,1) flip; signed changed-score mean with4 zero pseudo-observations; replay choose highest predicted positive mu else STOP, tie lower op; no live deployment or CAL/TEST tuning'}, freeze before CAL/TEST analysis, P(flip) Beta(1,1), conditional changed signed mean zero pseudo-count4; exact fractions independently verified. Constant train mean baseline. Offline replay chooses largest strictly positive predicted gain, no physical budget claim and no live new policy.
|split|arm|reached|Brier|constant Brier|gain MSE|constant gain MSE|
|---|---|---:|---:|---:|---:|---:|
|CAL|cold|133|0.257360174|0.246991682|0.053651173|0.052694045|
|CAL|reuse|133|0.257686652|0.246536882|0.044538306|0.044954846|
|TEST|cold|158|0.260613047|0.248671595|0.058277706|0.056930607|
|TEST|reuse|158|0.254320373|0.249545853|0.052028275|0.050678875|
|TRAIN|cold|295|0.237566042|0.245170928|0.062893357|0.064395289|
|TRAIN|reuse|295|0.240576482|0.248480322|0.059322832|0.060970985|
CAL replay {'cold': 30, 'episodes': 224, 'gain2': 0, 'local_nn': 423, 'pred_gain': 2.5480144552520274, 'reuse': 15, 'stop': 179}.
TEST replay {'cold': 32, 'episodes': 224, 'gain2': 0, 'local_nn': 488, 'pred_gain': 2.8146103928567867, 'reuse': 19, 'stop': 173}.
TRAIN replay {'cold': 51, 'episodes': 448, 'gain2': 12, 'local_nn': 1048, 'pred_gain': 5.282786261932236, 'reuse': 55, 'stop': 342}.

## Rapfi
72 archived qualified K56 root offline reference, actual counts {'roots': 72, 'nn96': 6901, 'nnT': 7765, 'nn128': 9129, 'target_changed': 27, 'global_changed': 18, 'target_agrees128': 47, 'base_agrees128': 54, 'both_changed': 10}; Q128 is a finite budget comparator, not oracle. TARGET action agreement with Q128 is not decision quality. Additional fresh48 families24 compact24 outer, Q96 and TARGET each separately vs same B128=192 games.
|stratum|arm|W/D/L|score|score lower95|NN saving|NN lower95|CPU saving|gate|
|---|---|---|---:|---:|---:|---:|---:|---|
|all|96|46/2/48|0.489583333|0.468750000|0.375834387|0.353945650|0.286569622|{'CPU': True, 'NN': True, 'score': True}|
|all|T|46/3/47|0.494791667|0.473958333|0.363427134|0.344352551|0.268093901|{'CPU': True, 'NN': True, 'score': True}|
|compact|96|22/2/24|0.479166667|0.437500000|0.357248233|0.331138422|0.277010533|{'CPU': True, 'NN': True, 'score': False}|
|compact|T|22/2/24|0.479166667|0.437500000|0.356126320|0.330054595|0.267204716|{'CPU': True, 'NN': True, 'score': False}|
|outer|96|24/0/24|0.500000000|0.500000000|0.395912774|0.380671435|0.300146291|{'CPU': True, 'NN': True, 'score': True}|
|outer|T|24/1/23|0.510416667|0.500000000|0.370476593|0.351425192|0.268990087|{'CPU': True, 'NN': True, 'score': True}|
all: paired score 0.005208333333333333 CI [0.0, 0.015625], counts {'base_interventions': 0, 'changed': 17, 'delta2': 1, 'delta_B': 9030, 'delta_Q': 7481, 'delta_total': 16511, 'downstream_B': 9030, 'downstream_Q': 6821, 'improved': 2, 'interventions': 55, 'local_nn': 660, 'n': 96, 'reached': 55, 'same_history': 79, 'worsened': 1}. Small stratum diagnostic, not superiority confirmation.
compact: paired score 0.0 CI [0.0, 0.0], counts {'base_interventions': 0, 'changed': 5, 'delta2': 0, 'delta_B': 504, 'delta_Q': 406, 'delta_total': 910, 'downstream_B': 504, 'downstream_Q': 118, 'improved': 0, 'interventions': 24, 'local_nn': 288, 'n': 48, 'reached': 24, 'same_history': 43, 'worsened': 0}. Small stratum diagnostic, not superiority confirmation.
outer: paired score 0.010416666666666666 CI [0.0, 0.03125], counts {'base_interventions': 0, 'changed': 12, 'delta2': 1, 'delta_B': 8526, 'delta_Q': 7075, 'delta_total': 15601, 'downstream_B': 8526, 'downstream_Q': 6703, 'improved': 2, 'interventions': 31, 'local_nn': 372, 'n': 48, 'reached': 31, 'same_history': 36, 'worsened': 1}. Small stratum diagnostic, not superiority confirmation.

## Audit and next
Actual 3776 games, 36705 moves, 830581 physical and 832463 logical game calls; 28670 separate parity/reference calls, total 859251. Game proof receipts 11411, bounds 822724, queries 3815021, early snapshots 6776, cold child receipts 1172, reuse child receipts 1172, Rapfi waves 54213. Independent stdlib family8000 bootstrap and rational model verification. Exact late roots {'complete': 0, 'roots': 0}.
K58: exploit early snapshot variation using actual matched score and action changes; choose future estimator only from frozen TRAIN and new cohort, not holdout retuning. Compare early targeted vs continue with physical cost and account for censoring and prior/backup gaps; use separate new actual matched games for selected rule. Do not interpret Q128 agreement as truth. Next integrate fallible epoch seam with strict real history-plane adapter including model transition and failure-recovery tests; generic async still open. Original posterior P1, A1 P2 and emergent P3 remain open. Dropbox sync NOT_VERIFIED.
