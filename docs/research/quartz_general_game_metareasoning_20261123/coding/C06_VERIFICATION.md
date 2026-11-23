# C06 — actual implementation and regression verification

| Layer | Observed result | Scope |
|---|---|---|
| Native targeted tests |18passed, exit0; release build exit0 | New native feature plus imported module tests |
| Parent current Python suite |21tests, exit0,1.269s | Independent signs, full-state oracle, costs, cues, RNG, trace, validation and aggregation |
| Complete reachable-state audit |4520roots,16167children agree | Every nonterminal reachable TicTacToe root; exact child values/node multiplicities/cues |
| Parent full release Rust suite |754passed invocations,89ignored,12result blocks, exit0,54.39s | Full default Cargo targets; shared module tests may recur |
| Cross-environment mathematical replay |9495cases, scientific fields exactly equal | LOCAL Python3.12.14 original versus remote3.12.3 |
| New Python formatting |7files pass Ruff0.17.6 | New C26 runnable Python plus publisher/test |
| Full Python regression |exit2 from pytest; driver exit1; collection blocked | tests/test_a18_evaluator_ablation.py requires absent torch;4skipped,1collection error |
| Hosted CI at parent and docs checkpoint |Rust and real-loop smoke succeed; Python job stops at formatting | Unchanged C25 read-only check has272files to reformat; no full Python PASS |

Parent inspected native model validation, random generation, signs, complete recursion, terminal cue handling, common initialization, score and STOP branches, sampled Gibbs and timer scope. It inspected source generation, TRAIN statistics/fallbacks, request filtering, output verification, source/model hashes, paired aggregation and calibration paths. Independent C09 review remains open.

The first cross-environment result comparison returned a driver assertion because it compared software.python as well as scientific results. Recursive inspection showed exactly one difference:3.12.14 versus3.12.3. Both calculations themselves exited0. The adjudication preserves distinct provenance and verifies every remaining field exactly. It is not a mathematical discrepancy or byte identity; evidence/C06_PARENT_CHECKS.json and C06_REPLAY_DIFFERENCES.json retain the failed check and correction.

Full Python collection's missing Torch is an environment failure of a preexisting test dependency. The standalone C26 native/Python paths do not import Torch and their relevant tests executed. Installing a new neural stack is outside this finite research experiment and was not needed to run it. No claim of a green whole-repository Python suite or green global CI is made.

All hard C26 implementation invariants are verified on the stated finite graph and fixtures. Empirical prediction quality and net gain remain to be measured. Proceed to TRAIN-only fitting, immutable model audit and CAL without tuning, then fresh TEST.
