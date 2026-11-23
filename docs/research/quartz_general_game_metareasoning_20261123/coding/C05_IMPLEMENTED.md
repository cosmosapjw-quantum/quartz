# C05 — complete executable feature

Implemented src/bin/quartz_c26_game_metareasoning.rs with unchanged GameState/TicTacToe modules: legal history validation, integer root-frame exact minimax, paid generic cues and common verification, six distinct policies, TRAIN probability/cost lookup, guarded reservation index, actual Gibbs sampling, explicit exact-win mask, complete trace and measured costs. Native function/output mapping is in NATIVE_IMPLEMENTATION.md.

Implemented coding/c26_campaign.py with independent board/minimax and SplitMix code, D4-grouped public sources separated from offline oracles, TRAIN-only fit/freeze, persistent native execution, every-trace replay, actual-cost aggregation, parent-level paired bootstrap and calibration/dependence diagnostics. DATA_IMPLEMENTATION.md gives commands and exact field meanings.

Both constructors preserved meaningful failed behavioral scaffolds before green implementations. Native final18tests and release build exited0. Data final21tests and scoped formatter checks exited0. Independent complete-state audit covered4520nonterminal roots and16167children. Parent additionally read the actual code paths and observed full release Rust repository test exit0:754passed test invocations across12targets,89ignored. Shared imported tests can recur across targets.

The source/harness/test/binary hashes in evidence/IMPLEMENTATION_FREEZE.json precede all TRAIN/CAL/TEST outcomes. The executable feature is complete; empirical gain is not claimed here. The C09 restore and final delivery scripts are concrete, as-yet-unexecuted delivery checks.

## Formatter/provenance correction

GitHub CI failed Python formatting at both the unchanged C25 parent and C26 documentation checkpoints, while Rust and real-loop smoke succeeded. A read-only check of C25 with Ruff0.17.6 found272preexisting files requiring formatting. An isolated formatter was installed under /tmp solely to verify new C26 files, changing no project/runtime dependency or lockfile. New C26 Python source now passes. Original executed mathematical source bytes were compressed into evidence/PREFORMAT_PYTHON_SOURCES.json.gz before formatting; Python ASTs match exactly. Execution manifests continue to bind their original bytes. No old project file was reformatted.

Next / upcoming task: C06 parent implementation/regression verification, then actual TRAIN/CAL and fresh TEST.
