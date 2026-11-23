# C08 — held-out empirical outcome and failure diagnosis

## Frozen execution

The TEST execution completed with command exit code 0 in 11.0270 s. It used 150 source parents, 1,800 timing requests (97,200 measured policy runs) and 1,800 trace requests (10,800 policy trace runs). One untimed warmup precedes nine timing repetitions; traces are separate. Four policy seeds are averaged only after per-seed timing medians, and each parent receives equal weight. The model SHA is 0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126. No parameter, policy, source split or primary endpoint changed after TRAIN freeze.

Raw requests/responses, timings, traces, parent aggregates, calibration rows, analysis and manifests are in coding/results/TEST. The independently reproducible post-hoc script coding/c26_describe_results.py consumes these frozen files; it does not fit models or rerun the campaign.

## Primary scientific decision

The predeclared full-cap PANDORA minus INITIAL_STOP endpoint is

    mean(delta utility - 2^-10 delta charged nodes) = 0.005690104166666667
    95% paired stratified source-parent bootstrap CI = [-0.013693033854166667, 0.029039388020833317]

The positive-net-gain gate requires a strictly positive lower confidence bound. **It is not met.** This is an empirical admission failure, not an implementation failure or a proof that metareasoning cannot help.

CAL was diagnostic: mean 0.06865234375, CI [0.027977294921875, 0.11551188151041666], 60 parents. It cannot replace TEST. All intervals condition on the fitted TRAIN model and chosen finite source frame. Roots may share descendants; no independent-world or universal-game generalization is asserted.

## Full-cap held-out observations

| Policy | Mean exact utility | Mean charged nodes | Additional nodes | Mean native microseconds | Mean node-priced utility |
|---|---:|---:|---:|---:|---:|
| INITIAL_STOP | 0.696667 | 58.3133 | 0 | 2.724110 | 0.639720 |
| ALL_ACTIONS | 0.726667 | 124.7200 | 66.4067 | 4.889702 | 0.604870 |
| RANDOM_ORDER | 0.726667 | 85.2600 | 26.9467 | 3.643057 | 0.643405 |
| GREEDY_VOC | 0.726667 | 83.4733 | 25.1600 | 3.601163 | 0.645150 |
| PANDORA | 0.726667 | 83.2067 | 24.8933 | 3.587080 | 0.645410 |
| GIBBS_VOC | 0.726667 | 81.1567 | 22.8433 | 3.581522 | 0.647412 |

All policies charge the common cue cost (46.6133 nodes) and initial solve (11.7 nodes). Native timing includes cue, exact solver and controller work; transport and model loading are recorded separately in the run manifests and overhead summaries. Microsecond timings are machine-specific.

At cap 9, every nonbaseline policy matched ALL_ACTIONS exact decision utility for each observed source/seed cell, not just in the mean. Finite observed optimality is distinct from the implementation invariant: exact-known incumbent retention guarantees no loss below INITIAL_STOP under correct solving, even when the predictive prior is wrong. It does not guarantee globally optimal final utility or net gain on a new state.

PANDORA gained 0.03 utility over INITIAL_STOP [0.01, 0.0533333], spending 24.8933 additional nodes. The node charge is 0.0243099, consuming 81.03% of that gross gain. Its secondary wall-priced gain at the independently fixed 1e-6 utility/microsecond is 0.02999913703 [0.00999928405, 0.05333238155]. This secondary choice of price does not rescue the failed primary node-priced admission.

Against ALL_ACTIONS, PANDORA saved 41.5133 nodes [34.6928, 48.2743] and 1.30262 microseconds [1.07526, 1.52536]. ALL_ACTIONS deliberately enumerates every permitted action even after a known win; RANDOM_ORDER also benefits from the common known-win certificate. Therefore these savings do not isolate the learned prior or Gibbs mechanism, and are not evidence for the original Gomoku NN-evaluation P2 threshold.

PANDORA versus RANDOM_ORDER saved 2.05333 nodes [0.505, 4.00029]; GIBBS_VOC versus PANDORA saved 2.05 nodes [0.33317, 3.85342] with observed equal utility. These are secondary, unadjusted multiple contrasts. Gibbs wall-time difference versus Pandora is not resolved: -0.0055583 microseconds [-0.0669783, 0.0536003]. No general Gibbs superiority is admitted.

## Post-hoc diagnosis — not retuning

TEST had only 8/150 initially improvable parents; the other 142 were already decision-optimal. PANDORA spent additional work on 52 of those 142 already-optimal parents: 3,208 of its total 3,734 additional nodes across equal-weight parents (85.91%). The eight improvable parents contributed 526 nodes and a summed utility gain of 4.5. CAL instead had 9/60 improvable parents and 5.0 summed gain. Available mean headroom was 0.0833333 in CAL and 0.03 in TEST.

Across the 600 full-cap TEST source/seed cells, the 692 selected PANDORA queries had summed predicted one-step gains 82.0845 and realized incumbent increments 18.0. Predicted query nodes totaled 13,770.8257 versus 14,936 actual nodes. This selected-path discrepancy is descriptive; it neither proves a causal correlation mechanism nor licenses a new TEST fit. It points to conditional gain prediction after incumbent selection as a stronger next target than merely increasing search volume.

TEST nonterminal-candidate calibration: pooled Brier 0.327273; equal-parent Brier 0.306133; actual/predicted mean solve nodes 22.7495/21.6289; cost MAE 4.69127 and RMSE 7.68832. Within-root residual correlation was 0.176731 across 139 eligible parents (descriptive symmetrized pair moments). CAL correlation was 0.278499. Product closure remains an approximation; the independent-box theorem does not prove optimal ordering for this game distribution.

Finite-temperature Gibbs was actually sampled: 1,390 sampled decisions, minimum positive action probability 2.55855e-10, no numerical underflow. Negative net scores can be sampled at positive temperature by design. The objective is regularized decision utility, not an unregularized optimality theorem.

## Evidence-publication correction

Parent delivery inspection discovered that inherited Git ignore rules excluded coding/results. Thus the eight CAL result files were present and scientifically valid locally but omitted from the C07 commit and cumulative archive. This is a provenance/publication defect. C07 history is retained truthfully.

Before C08 publication a separate CAL supplemental archive was created and all eight files were read back byte-for-byte. Both CAL and TEST directories (16 files total) were explicitly staged in C08. The publisher now rejects any future C26 result file absent from the Git index; the final delivery audit additionally requires all 16 files in the final archive. See evidence/C08_PACKAGING_CORRECTION.json for hashes, timestamps and the exact omission/correction boundary. Historical C07 archive receipts are not retroactively described as containing these bytes.

The scientific figure figures/C26_primary_and_costs.png/.svg was rendered with Matplotlib and visually checked. The environment emitted a nonblocking Axes3D import warning; this figure uses only 2D axes. No source/runtime dependency was changed.

## Claim status

- Derived and numerically checked: R01–R10 model-scoped results, exact finite examples, Gibbs variational identity, informative-signal fluctuation correction, detailed-balance/CPTP construction.
- Implementation-verified: exact query game mapping, root frame, incumbent retention, genuine Gibbs sampling, trace/cost accounting, TRAIN/CAL/TEST separation.
- Empirically observed: finite TEST quality recovery and paid-work comparisons above.
- Primary admission not met: positive node-priced PANDORA gain over INITIAL_STOP.
- Unresolved: correlated posterior controller, transferable NN-evaluation savings, original human dissociation P3, physical Wick/RG/Lindblad interpretation beyond the tested finite mechanism.

Next: C09 independent review closure and actual four-archive restoration; then C10 publish the final scientific decision and a bounded C27 plan.
