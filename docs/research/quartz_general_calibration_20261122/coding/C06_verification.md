# C6. Native Rust read-only shadow controller and tests

Actual RED: standalone src/bin/quartz_c18_shadow.rs returned an explicit missing-implementation Err. cargo test --offline --locked --bin quartz_c18_shadow exited101 with correctly failing zero-coefficient sigmoid test. Actual GREEN: 3/3 Rust focused tests passed after implementing a generic shadow forecast on two initial-only fields, probability clipping, invalid input and oracle/future/domain input rejection.

Fresh full regression: cargo test --offline --locked DEBUG 617 existing mcts_demo PASS +3 new shadow CLI PASS +7 existing C17 CLI PASS, zero failed (main 89 ignored); --release same 617+3+7 PASS. cargo build --offline --locked --release --bin quartz_c18_shadow succeeded. Compiled Rust shadow executable SHA256 **77b2000f90fc6f39188ec85ead2c24cf004f50f31e88aa0a90144a6335f32919**. Python focused C5 tests 5 PASS.

Independent actual compiled Rust binary vs NumPy on **96 deterministic initial-only generic rows**: maximum absolute probability discrepancy **1.1102230246251565e-16**, within frozen 1e-10. Output evidence/C06_RUST_PYTHON_PARITY.json. Native Rust module takes theta, CAL offset, normalized initial gap and n_actions, NEVER reads label, domain, true utility, final MCTS response or mutates runtime SearchPolicy. It is only a shadow inference CLI, **not a deployed controller**.

C7 source and prospective split code to be frozen/published before actual test outcomes; true value of computation in Gomoku remains NOT_ESTIMABLE.