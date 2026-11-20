# QUARTZ COG-01: Recognition calibration and value of computation

2026-11-20, Asia/Seoul. Status: **EXACT FINITE-SURROGATE VALIDATED; REAL NN/MCTS UNTESTED**.

## Executed identity and meaning

- Research base: `6d279096c8b125c8af8d222784fac31a2d36b187`.
- Contract commit: `c183b7210a451ba3445c65b86672609d8807f954`.
- Exact interpreter and 9 tests commit: `8a624e69978a6df18bbd07dfd6b4b0465fc4c371`.
- Clean executed runner commit: `5fc71264bbc44c6b1174c32e9a21cfb4988d1687`.
- Exact result: `evidence/exact_20261120/exact_results.json` SHA256 `7593205a8cc55e84cb3f18579fac2fb5a89902ee678647428e959d8d6b1efc2b`.
- `execution_manifest.json` records committed code hashes, exact SHA, command, environment, output hash. A second direct-four-atom test enumerator confirms the source implementation. Wolfram independent rational formulas also agree with three scenarios.
- Source tests: 9/9 new + 27/27 earlier W1 passed. The runner rejected an attempted second write into the existing evidence path, preserving original artifacts. Initial test-first red was one import-error collection failure before `cognitive_exact.py` existed, not a scientific counterexample.
- Every source commit was pushed and checked by `git ls-remote`; local Dropbox-folder tarball backups were made for each publication. These are **local folder** backups, not provider-side Dropbox cloud synchronization acknowledgments.

## Experimental estimand

Fixed A utility `u=3/5`, binary B utility `X∈{0,1}`. The recognizer emits binary cue `S`. Train on prior `P(X=1)=1/5`, sensitivity `P(+|X=1)=17/20`, and false-positive rate `P(+|X=0)=1/5`; its frozen posteriors are `q(+)=17/33`, `q(-)=3/67`. Perfect inspection reveals X at cost `c=1/10` utility units. The trained controller inspects iff `g(q)=min(q(1-u),u(1-q))>c`. Therefore it inspects only positive cues.

Same-population `stop_only` is the baseline. `random_matched_rate` matches the *actual deployment* expected inspection count, so it is only a **descriptive matched-cost counterfactual with oracle rate**, not a prospectively implementable control. `oracle_recalibrated` is the conditional Bayes policy given each deployment population, not a fair learned competitor.

All outputs are exact rational enumerations of `(X,S)` probability masses; the displayed decimals are rounded. No random seed, sampling interval, trained-network or real engine result exists in this screen.

| Deployment | Stop | Selective | Random rate matched | Recalibrated oracle | Inspection rate | Quality ECE | Gain ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| in_distribution | .6000 | .6350 | .5934 | .6350 | .3300 | 0 | 0 |
| uninformative_cue | .6000 | .5900 | .5900 | .6000 | .5000 | .2352 | .0941 |
| inverted_cue | .6000 | .5440 | .5856 | .6360 | .7200 | .4784 | .1914 |
| prevalence_shift | .6000 | .6900 | .6276 | .6900 | .4600 | .1389 | .0555 |
| perfect_cue | .6000 | .6600 | .5960 | .6800 | .2000 | .1328 | .0531 |

In distribution: exact source budget-matched gain over random is `127/200-2967/5000=26/625=0.0416` at `33/100` average inspections. Inverted cue: exact negative gain relative to stop `68/125-3/5=-7/125=-0.056`. The good candidate frequency-shift row retains decision optimality despite nonzero probability calibration error, demonstrating decision boundaries matter.

## Important theory correction identified during this run

The setup above uses **perfect verification** and both source `q(S)<u`. For a deployment-true posterior `r(S)=P(X=1|S)`, source decision A and its predicted gross inspection gain satisfy

```text
g_hat(S)=(1-u)q(S)
g_true(S)=(1-u)r(S)
|g_hat-g_true|=(1-u)|q-r|.
```

Thus across *every one of the five scenarios* `gain_ECE=(2/5) quality_ECE`. Consequently this finite-perfect-observation experiment **does not independently establish** recognition calibration and VOC calibration as distinct statistical capacities. They are arithmetically coupled here. Suppressing this caveat would invalidate a claimed cognitive dissociation.

### Separate exact observation-channel counterexample

Take `u=3/5`, `P(X=1|T)=1/2` in two equally likely contexts `T=H,L`, so the recognition probability `q=1/2` is **perfectly calibrated in both**. Verification still costs `1/10`, but its observation channel differs:
- H reveals `Y=X`, and gross value of computation is `1/5`.
- L reports independent noise, hence gross value of computation is exactly `0`.

A recognizer-only controller wrongly assuming all verifications are perfect predicts gross VOC `1/5` in both contexts and verifies both. Its total expected net utility is `(7/10+1/2)/2=3/5`. A correctly channel-aware controller verifies H but not L and obtains `(7/10+3/5)/2=13/20`. Without any verification the utility is `3/5`.

**Conclusion:** calibrated option-value beliefs alone do **not** calibrate the usefulness of a computation with an unspecified observation channel. The channel and expected innovation must be modeled. This example is exact Bayesian value of information, not a new neuroscience theorem or an MCTS guarantee. A separate Wolfram check confirmed the rational values.

## Claims and gates

| Question | Outcome |
|---|---|
| Finite Gaussian-free recognition and perfect-inspection model | Exact, derived, independently enumerated |
| Small recognizer can improve budget allocation in the frozen source world | Exact restricted-model existence example (+26/625) |
| Frozen recognition is automatically robust to new cue meaning | **Refuted** in inversion (+ computational spending, lower utility) |
| Probability calibration forces correct VOC for arbitrary computation channels | **Refuted** by H/L observation-model construction |
| Separate gain calibration empirically demonstrated in the five-screen perfect-observation study | **Not demonstrated**; gain ECE = 0.4 quality ECE identically |
| Actual trained NN, high-budget MCTS or Elo improvement | **Unresolved / NOT RUN** |

The manuscript does not revise the historical BQ++ 30% saving target, resurrect old quarantined factorial evidence, or promote an Idea Foundry axis to ACTIVE.

## Next executable project step

**ID: `QUARTZ-COG-01B-REALTRACE-20261120`.** Ground the channel-aware recognizer and computation-value predictions in *actual* QUARTZ frozen checkpoint/position traces, without learning from the high-budget reference answers.

1. Inventory and hash the **existing** trained network checkpoint, exact inference configuration, game rules and opening-family corpus. Establish the root/evaluator/target/representation epoch before any reuse.
2. Fix held-out opening families and checkpoint families. Determine the binary/exact action-quality target or explicitly state that a high-budget stronger reference is only a proxy; avoid falsely claiming minimax truth.
3. Create paired resident-root computation forks: same prefix/search RNG then a declared computation `c` versus no `c`; measure **signed** outcome improvement and completed NN calls, CPU/GPU/wall time, pending costs. A policy-TV change is not an actual VOC label.
4. Compare (a) calibrated candidate-quality head alone, (b) same head with observation-channel reliability/correlation/fidelity and estimated conditional gain, (c) a strong fixed-schedule or Gumbel/PUCT allocation reference. Prohibit held-out oracle labels from controller inputs.
5. Report separate held-out quality calibration, gain calibration, wrong-inspection events and regret/time curves; family/seed dependence requires paired clustered uncertainty. Negative findings remain negative. Stop if gain prediction has no incremental benefit after overhead.

After COG-01B, **COG-02 Einstellung / candidate reopening** targets appealing but suboptimal high-prior candidates versus decisive low-prior refutations. Its goal is causal regret reduction at matched calls/time, not fitting human gaze signatures.

## Reproduction

```bash
# Fresh clean checkout at 5fc71264bbc44c6b1174c32e9a21cfb4988d1687
python -m unittest discover -s docs/research/cognitive_metareasoning_20261120/tests_research -p 'test_*.py' -v
python docs/research/cognitive_metareasoning_20261120/run_cog01_exact.py --output-dir /tmp/cog01_new_unique_path
```

Never supply the committed evidence directory as output. The script deliberately refuses overwrites. Source and evidence identity are linked by manifest hashes; run timestamp differences do not change exact rational output claims.

Literature context: Callaway et al. (2018) `https://arxiv.org/abs/1711.06892`; Sezener–Dayan (2020) `https://proceedings.mlr.press/v124/sezener20a.html`; Russek et al. (2025) `https://pmc.ncbi.nlm.nih.gov/articles/PMC12553403/`. They motivate distinct models of mental computation, not an already demonstrated QUARTZ cognitive mechanism.
