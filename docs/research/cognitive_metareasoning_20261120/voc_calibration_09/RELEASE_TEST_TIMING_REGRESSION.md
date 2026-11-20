# Release NPS timer quantization: bounded regression record

Source event 2026-11-21. Observed unexpected failure during final **fresh** release whole-suite test:

- Invocation: `cargo test --offline --locked --release -- --test-threads=4`
- Failure: `mcts::tests::test_run_par_quartz_four_threads` at original `src/mcts/mod.rs:2192`.
- Output: `592 passed; 1 failed; 89 ignored`, offending assertion `stats.nps > 0.0`.
- An earlier same-release full-suite execution had returned `593 passed; 0 failed; 89 ignored`.
- The test already confirmed `stats.root_visits >= 400`, so the observed symptom was **not** a failure to visit root states.
- Direct code readback of production `run_par_quartz`: `let ms = start.elapsed().as_millis() as u64`; `SearchStats.nps = rv / ms * 1000` when `ms>0`, and **exactly 0** when `ms=0`. An optimized operation completed faster than 1 millisecond is therefore indistinguishable from zero elapsed time in this field. The original unconditional `nps>0` test assumption was invalid on optimized fast runs.
- Classification: **test-contract / timer-resolution regression**, not theoretical Connect4 error, not original MCTS search-action semantics, not external solver issue. This discovery is material to release wall-time reliability and should remain visible.

**Smallest scoped repair:** test-only change in `src/mcts/mod.rs`. Require that if `stats.elapsed_ms==0` then the expected documented rounded `stats.nps==0.0`; otherwise require positive NPS. Do not alter game physics, search execution, QUARTZ controller, or any of the prior 4,608 experimentally recorded action/cost outcomes. This change is in the **post-experiment** source branch, not retroactively claimed to be in the compiled binary used for the frozen experiment.

Verification commands:
```sh
cargo test --offline --locked --release test_run_par_quartz_four_threads -- --nocapture
cargo test --offline --locked --release -- --test-threads=4
cargo test --offline --locked -- --test-threads=4
```
The independent original release experimental binary identity remains `e3c44f8630c67d0c00929f10b429f35d62805b779e025d73ce16f142100c7f78` and original compilation source commit remains `1c2b00c1bad2585380fee97bde65e3862dc1f25d`.

No scientific result or chosen policy was recomputed because the repair touched only this unit-test assertion. Following test results must be recorded from actual execution, not inferred from the patch.
