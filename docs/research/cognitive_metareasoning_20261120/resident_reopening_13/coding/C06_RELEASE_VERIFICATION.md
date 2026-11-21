# C6. Full Rust regression, compiled release CPU executable and live original-model preflight

**Exact source:** code commit `8c63a60973ac11b9e1fa4cdf93d387083c65ee57` compiled via `CARGO_TARGET_DIR=/home/cosmosapjw/quantum/research/quartz_connect4_20261120_target CARGO_BUILD_JOBS=4 cargo build --offline --locked --release --bin mcts_demo`. **Binary SHA256 `5dd61a01cc87dd715183f3d4145367b471182e0676a9a6bde7c38c9c225eede0`**, 3,794,144 bytes. Source Git SHA and compiled binary SHA are separately locked in `C06_PREFLIGHT_RECEIPT.json`. The binary is an external compiled artifact not committed as a source file.

**Fresh full debug Rust:** `cargo test --offline --locked -- --test-threads=4` → 604 PASS, 0 FAIL, 89 ignored. **Fresh full release Rust:** `cargo test --offline --locked --release -- --test-threads=4` → 604 PASS, 0 FAIL, 89 ignored. These are full Rust *crate test executable* suites, not a blanket claim about every system component.

**Actual on-host Rust release preflight:** In one warmed JSON-line `mcts_demo --server` process, ran 3 real evaluator regimes × 4 fixed research arms = 12 actual root searches on valid source `history="121213"` (MCTS seed17; root constraint top3; first8, then total24 same engine). Arms evidence, closed sham, all-legal, all-open after8. Every result:
- exact root identity pre/post and original root game hash constant;
- original full legal edge count7, first edge visits sum8, final sum24, exactly16 additional;
- no changes in any edge stats `(n,p,w,m2)` or model evaluation calls at the mask-store checkpoint;
- initial selection in active subset; final selection in final active subset; always valid root action;
- every mode uses actual trained Rust CPU weights already locked in C1, not dummy/reference labels.

Representative known MLP `history=121213` evidence reopens omitted vertical winning column0, original mask56→57, final action0. Uniform did not reactivate, so no false claim of general triggering. Initial tiny wall times varied from ~213us to several ms; a single preflight call is not a stable timing estimator. **Scientific CPU cost-match remains UNCHECKED until C8**.

**Model source identity:** see `C06_PREFLIGHT_RECEIPT.json`; MLP/Conv actual JSON weights were hash-verified before passing them to Rust. No external exact game-theoretic oracle read by server. Original research home worktree kept untouched.

**Failure classification:** Rust code-path test+release integration: implementation-verified. Real source-neural preflight completed. Physical CPU wall-time efficacy and the heldout root utility: unresolved pending C8. Date: 2026-11-21 Asia/Seoul.
