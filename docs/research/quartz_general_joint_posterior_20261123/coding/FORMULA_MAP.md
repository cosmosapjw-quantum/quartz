# C02 — equation, data and code mapping

| Mathematical object | Data | New implementation obligation |
|---|---|---|
| Root-player U=(v+1)/2 | Public replay + explicitly paid exact result | Reuse unchanged GameState/TicTacToe/exact sign path; allknown deployment |
| Base p_a and d_a | Exact C26 model.json, public remaining/counts | Same feature->remaining->global fallback; no new marginal fit at deployment |
| f_a=(u-mu)/s | Categorical base probabilities | Center and scale on positive support; degenerate f=0 |
| log_odds posterior | Initial and later distinct exact observations | Add log1p factors, predictive law via tanh(log_odds/2) |
| Finite K1/K2 | Current b, unknown set, remaining cap, frozen query prices | H2 thenH1, STOP, stable ties; hypothetical branch is local |
| Post-initial product control | Joint posterior after common initialization | Snapshot probabilities thenfreeze; no later crossupdates |
| rho CV logR | Old90 complete TRAIN vectors + inside-fold refit | Stable mixture likelihood, root folds, no oldCAL/TEST |
| Time-to-utility price |1480 newly instrumented TRAIN nonterminal queries | elapsed_ns/nodes aggregate, positive finite, freezebeforeCAL |
| Primary net contrast | New public144split; offline labels separate; actual native timings | rootmedian repeats, seedaverage,120rootmean, fixedlambda_time_ns |
| Conditional prediction diagnostics | Actual recorded histories + offline audit labels | complete-world oracle comparison; no truth in runrequests |
| Stage artifacts | Exact source/input/binary/return code identities | append receipt, archive cumulativeoverlay, pushandreadback |

Rust file: src/bin/quartz_c27_joint_posterior.rs. Python harness: coding/c27_campaign.py. Python tests: tests/test_c27_joint_posterior.py. Theory oracle: theory/exact_joint_oracle.py. C26 substrate files and THESIS/CLAIM_LEDGER stay byte-identical. Implementation names/line locations will be resolved against actual source duringC05/C06, not invented now.

The physics transport is an actual finite partition/posterior coordinate; cognition transport is actual observation-conditioned computation value. Neither substitutes entropy reduction for realized utility. Source-derived posterior likelihood and selected-gain diagnostics are separate.
