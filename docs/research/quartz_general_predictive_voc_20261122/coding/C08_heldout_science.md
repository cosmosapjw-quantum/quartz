# C08 | Actual prospective heldout exact EVSI versus coarse predictor and CPU-overhead accounting

Full C20 experiment source C04 SHA256 d6cf96c26d9ac913963f702d9b0418f0002038979f6a470da3865999368e4ae7, executable code frozen/pushed C07 Git620df8133d416dff9bfaaf94761e358172ddf082 before labels. C07A TRAIN/CAL raw and source-only clock model SHA256 530ca204c96b5fda4f3b060e8ebd87b52e4c4ec7e97be42dac56da93ca81cb04 frozen/pushed Git39f820d7105e104f93e4a235d889bdc3df7f3512 before TEST. C08A TEST raw independently sealed/pushed Git a331f8bfb3a94fa9cb331d6fb604eeec4d05e706, SHA256 5ab524aff074d8f441f09679309807079c694035eded182d86c5517d82cf5c1f, before any scoring. C08 source Git at test scoring is C08A. The original C19 trained coarse comparator SHA256 5fdd172a13d51c4d17dc8e7829fb7c5518421028046ebac6e418a1158adf6735 is fixed historical evidence, not refit to C20 new TEST.

Actually generated 576 TRAIN/CAL +144 TEST = **720 genuine stochastic additional source computations** with independently frozen new source parents, 240 unique original latent worlds across two task families; no latent world crosses split. TEST n72 Beta arms (24 parents), n72 finite categorical sensors (24 parents). The posterior predictive model only sees initially observed successes or sensor outcomes, correct public priors and likelihoods, and fixed proposed sample budget. The hidden actual theta/true arm means, extra observation draws, final action, reward changes and measured future CPU task wall were excluded by typed initial_only input; only offline source/scorer uses true latent utility.

| New heldout family | Frozen historical C19 coarse signed-Z MSE | C20 exact posterior EVSI MSE | Parent bootstrap500 signed-MSE difference 95% |
|---|---:|---:|---|
| Beta arms n72 | 0.01411918 | **0.01381147** | [-0.00195766, +0.00105184] |
| Sensor n72 | 0.01119080 | **0.01068388** | [-0.00285884, +0.00146938] |

Both point estimates improve, but each group-parent interval includes zero. Exact model mean conditional EVSI Beta 0.04716533 vs finite group mean realized signed gain0.03883327; sensor EVSI0.02775210 vs mean realized0.00304986. Bayesian exact conditional gain is not a guarantee for finite ex-post source-world sample, and systematic model calibration would need additional independent source groups. In TEST exact evaluator actual mean wall Beta **4621.98 μs** against actual added task **8.722 μs**, sensor evaluator **214.98 μs** versus added task **16.958 μs**; C19 coarse evaluator roughly 47.61 μs Beta and37.31 μs sensor. Train medians by resource bucket were determined before TEST and not replaced by TEST timing.

Under prespecified primary price **lambda=0.004 terminal utility/microsecond**, actual mean paired priced counterfactual net vs STOP:

| Policy | Beta n72 | Sensor n72 |
|---|---:|---:|
| NEVER | 0 | 0 |
| ALWAYS underlying computation, no evaluator | +0.003944 | -0.064783 |
| C19 coarse shadow, ignore metaevaluator overhead | +0.009159 | -0.019075 |
| C20 exact shadow, ignore metaevaluator overhead | +0.009159 | -0.006020 |
| C19 coarse shadow, **charge** actual evaluator overhead | -0.190434 | -0.149241 |
| C20 exact shadow, **charge** actual evaluator overhead | **-18.487910** | **-0.859928** |

The operationally charged C20 proposal selects **0 of 72 computations in BOTH** domains using precommitted TRAIN median evaluator and task costs, yet spends the actual predictor evaluation CPU in all cases to reach STOP. Its operational loss is therefore approximately -lambda times evaluator wall. C20 charged minus ALWAYS parent-bootstrap500 descriptive 95% CI Beta[-28.423,-9.736], sensor[-0.818,-0.762], both below0. At predeclared price sensitivities 0.001 and0.010, charged C20 also selects zero and has strongly negative net against no metadata computation. The exact EVSI is mathematically useful as a reference but NOT operationally economic as implemented in interpreted Python.

All future calculations physically executed irrespective of shadow recommendation for source utility label availability. None of these results demonstrates measured hardware CPU saving, improved actual game search or effective live STOP. The optimiser's CPU E was physically measured; the Ctask saving is only counterfactual. Test source label n22 revisions Beta, n7 sensor, and 144 actual sample paths; causal identification of an intervened policy would require prospective activation.

For 72 TEST finite-sensor initial-only public prior/likelihood/utility requests, actual compiled Rust C20 independent model matched Python posterior EVSI/normalization/before/after with maximum abs difference6.661338147750939e-16. Rust does NOT implement Beta full count law, so no Rust Beta parity claim. Original frozen THESIS, C16–C19 and live SearchPolicy unchanged; main and PR untouched. Full numeric artifacts are evidence/C08_TEST/C08_HELDOUT_EXACT_EVSI.json, FROZEN_SHADOW_FORECASTS.jsonl and RUST_SHADOW_NO_LABEL_REQUEST.json.