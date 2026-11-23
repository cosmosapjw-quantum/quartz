# C05 native implementation handoff

## Summary and ownership

Implemented the complete isolated executable `src/bin/quartz_c27_joint_posterior.rs`, including embedded Rust tests, against C04 HEAD `cb6266eed040e2a9c4637cf0722b52763ee84937`. Authority remains user SSOT `87c15dc828d6eea519c219fd8a412fa21dfc464d` and the frozen R09 CODE_CONTRACT / EXPERIMENT_CONTRACT. This lane edited only that new source and new `evidence/NATIVE_*` artifacts. It did not edit C26, GameState, TicTacToe, THESIS or CLAIM_LEDGER, and made no commit or push.

The five policies are INITIAL_STOP, RANDOM_TWO, PRODUCT_H2, INITIAL_PROJECTED_H2 and JOINT_H2. The common initialization is unchanged, terminal deployment remains exact-known only, and the additional budget is at most two: H2 then H1 then STOP. The one-parameter joint law uses log-odds updates, rejects unsupported/repeated observations, and does not mutate actual history while scoring hypothetical branches. Initial projection freezes the post-initial marginals before later decisions.

The old replay, root-frame utility, exact solver, rollout, candidate construction and initialization function blocks are byte-identical in the new source to C26. Their individual hashes are recorded in NATIVE_IDENTITIES.json. GameState and TicTacToe source files were also unchanged before/after formatting and build.

## Executed proof

All commands used the shared target cache:
`CARGO_TARGET_DIR=/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123/target`.
No Cargo clean, full rebuild, dependency installation or full historical campaign was run.

| Check | Actual command/result |
|---|---|
| Semantic RED | `cargo test --locked --offline --bin quartz_c27_joint_posterior -j2 information_ -- --nocapture`; exit 101; both tests reached assertions: product posterior 0.5 vs required 0.82, product H2 value 0.5 vs required 0.545 |
| First GREEN | `cargo test --locked --offline --bin quartz_c27_joint_posterior -j2 -- --nocapture`; exit 0; 17 passed |
| Scoped formatting | `rustfmt --edition 2021 --config skip_children=true src/bin/quartz_c27_joint_posterior.rs`; exit 0; protected source hashes unchanged |
| Release build | `cargo build --release --locked --offline --bin quartz_c27_joint_posterior -j2`; exit 0; 4.893530977 seconds |
| Final-source GREEN | Same targeted test command; exit 0; 17 passed, 0 failed; 0.619263887 seconds |
| Release JSONL smoke | `python3 docs/research/quartz_general_joint_posterior_20261123/evidence/NATIVE_SMOKE_DRIVER.py`; exit 0; one old TRAIN root, 25 measured policy records, one survey containing 25 instrumented exact calls, invalid-input recovery |
| Protected binary | Original C26 release SHA unchanged: 8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c |

Of the 17 native tests, 13 exercise C27 behavior and 4 are the reused TicTacToe module tests. New tests cover the literal .82/.545 mechanism, prior/conditional probabilities, degeneracy/unsupported/repeated observations, observation order, rho-zero product limit, initial projection, H1 versus H2, zero threshold and known-win stopping, query uniqueness/cap, moving-incumbent sign counterexample, literal root-frame/exact-node/cue outputs, five-policy rotation and repeated semantics, timing partition, terminal-win initialization, survey timing cardinality, strict public/model fields and JSON-lines recovery. No mocked solver is used.

The first RED source and exact failure log are preserved as NATIVE_RED_SOURCE.rs / NATIVE_RED.log / NATIVE_RED.json. RED failed for missing conditional behavior, not a compiler failure. It was the explicit product-only starting behavior, not the final implementation.

The release smoke uses only the first preexisting TRAIN source `C26-p4-0059adf99c0886cf`, history [5,1,6,4], with fixed software-fixture rho=.8 and policy_seed=3. Five measured repeats and one discarded warmup yield 25 run records. Trace semantics repeat exactly, execution slots rotate, all query indices are unique and at most two, and measured phase times sum to elapsed time. It additionally sends malformed JSON and an unknown offline-label field before valid requests, confirming recovery in the same process. Survey timing records are not used to estimate or freeze a price in this lane.

The Python implementation lane independently reported successful full-world-oracle validation of 90 native records (three fixed old TRAIN roots, rho 0/.8/.9, five policies, two repeats), including exact labels/nodes and conditional scores. That evidence belongs to its HARNESS artifacts, and is not duplicated or claimed as a fresh C09 review here.

## Final identities and environment

- Source SHA-256: `31de740dd4c37a1ddb6324930572e60146b93d1e6e14b46eb3ba97f137ff3632`.
- Release binary: `/home/cosmosapjw/quantum/research/quartz_general_meta26_20261123/target/release/quartz_c27_joint_posterior`.
- Release binary SHA-256: `755c93c55711ebc9ecaa6c05907ee9d8dff381ae1942df95ee33e8b1dce2f989`.
- Test binary: `.../target/debug/deps/quartz_c27_joint_posterior-70bada63bc9c3453`.
- Test binary SHA-256: `fd18e4bdebb7bb43c64aa935d02528ebbc6578ee7dcda6aa58ac19cf9c4ee999`.
- rustc 1.99.0 (b940084d7 2026-09-28); cargo 1.99.0 (5f94df478 2026-08-27); smoke Python 3.12.3.
- Free space at final identity capture: 1,551,253,504 bytes. No old tracked file appeared dirty in the captured Git status; other new C27 files belong to parent/Python lanes.

## Python request / response interface

Transport is one JSON object per input line and one output line per input line. Unknown request or model fields are errors. Errors produce an error object and the process continues to the next line.

Run request:
```python
request = dict(
    id=source["id"], history=source["history"],
    cue_seed=source["cue_seed"], policy_seed=3, mode="run",
    additional=2, repeats=5, warmup=1, trace=True,
    model=original_c26_model, rho=0.8,
)
```

Survey request:
```python
request = dict(
    id=source["id"], history=source["history"],
    cue_seed=source["cue_seed"], policy_seed=0, mode="survey",
    additional=2, repeats=5, warmup=0, trace=False, model=None, rho=0,
)
```

Survey is intended for the parent-authorized offline protocol. This executable cannot determine a root's split from the intentionally public input, so the harness enforces TRAIN-only price inputs and new-CAL/TEST label isolation.

Top-level schema is `c27-joint-v1`; mode and id are echoed. Run responses contain `runs`, `warmup_elapsed_ns`, `execution_elapsed_ns`. Each run retains C26 initial_move, initial_solve_move, initial_u, selected_move, final_u, cue_nodes, initial_nodes, additional_nodes, charged_nodes, additional_queries, query_moves and predicted_additional_nodes. New policy/repeat/execution_slot fields show actual rotated order; serialized records use canonical policy order within each repeat.

All time fields are integer nanoseconds. Individual `elapsed_ns` includes cues, initialization, posterior/planning, exact queries and trace construction, plus temporary controller state destruction. `cue_ns`, `initial_ns`, `query_ns` time those phases; `controller_ns` is their remainder in elapsed_ns. JSON decoding/serialization, IPC and process launch lie outside the individual native interval.

`decode_validate_ns` is measured separately at the top level. `serialize_payload_ns` measures the single serde serialization of the response payload. The tiny final formatting of that timing field and output I/O happen afterward and are not in that serialization measurement. The Python harness can report process roundtrip and its remaining transport cost without claiming it is individual-controller latency.

Trace fields:

- `predictions[]`: move, remaining, counts, terminal_known, base probabilities[3], predicted_nodes, observed_u. Base probabilities include terminal delta masses; only terminal values appear in observed_u before initialization.
- `known_observations[]`: move, u, kind=terminal or initial. Later exact values appear only after their paid query in decisions.
- `initial_log_odds`: field immediately after common initialization for joint/projected policies; product controls use zero.
- `projection`: null except INITIAL_PROJECTED_H2, where it lists remaining move/probabilities snapshots when trace=true.
- `final_log_odds`: the policy's final field. Projected law is represented after projection as an independent law with field zero; its original conditioning field remains initial_log_odds.
- `decisions[]`: b_before, b_after, remaining moves, horizon, log_odds_before/after, candidates, chosen_move, choice=FOCUS/STOP, queried_u, queried_nodes, reason.
- Each scored candidate: move, probabilities[3], predicted_nodes, g, k1, k2, q. At horizon=2, q=k2. At horizon=1, k2=null and q=k1.
- A forced STOP emits an empty candidates list and does not evaluate Bellman scores. Nonforced RANDOM_TWO constructs product-law diagnostic scores when trace=true but selects only with the C26 SplitMix64 policy stream; diagnostic construction is charged in elapsed_ns.

Survey candidates retain move, terminal_known, remaining, counts, rollout_mean, u, solve_nodes and cue_nodes, and add `timings:[{repeat,solve_nodes,elapsed_ns}]`. Nonterminal candidates have exactly five timing entries; publicly terminal candidates have an empty timings array and contribute no price samples. The parent must aggregate the timing entries, not count terminal solve_nodes=1 as a timed query.

## Limits and checkpoint

No rho fitting, TRAIN-wide 1480-query price measurement, price freeze, new CAL/TEST labeling, campaign, or empirical admission claim occurred in this lane. The restricted positive-common-mode closure, omitted predicted planner cost and finite two-query horizon remain scientific limitations from R09. Full C04 cargo-suite evidence is parent-owned and predates this source. The parent reported the preimplementation full Python CI as skipped after an existing Ruff-format failure, and local full pytest blocked by missing Torch; this lane neither relabels those as fresh passes nor installs the environment.

Keystone checkpoint: implementation -> parent integration/C06 targeted independent validation, then the single planned C09 change-review. Isolation, meaningful RED, GREEN and native proof gates have concrete recorded evidence. Action: hand off to parent; no shipping claim and no commit/push by this lane. Next check: parent verifies source/interface hashes and merges the native/Python proof packets before C05 publication and the authorized C06 TRAIN-only freeze.
