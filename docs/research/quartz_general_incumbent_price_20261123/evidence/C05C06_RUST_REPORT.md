# C25 native C05/C06 implementation handoff

## Result and scope

C25 native implementation is runnable. The focused RED command observed 12 genuine missing-behavior assertion failures before implementation; the identical command passed all 13 C25 tests after implementation. One of those 13 is an explicitly reused C24 control/initial-state invariant, not a new RED claim. Full debug and release test runs each passed 40 tests: 27 unchanged inherited C24 tests and the 13 C25 tests. Debug and release application builds succeeded. No broad scientific campaign or timing benchmark was run by this worker.

All paths in this report are on remote device cosmosapjw, device ID edd1b263-56a0-4391-8a2a-677dd00eabf5, under /home/cosmosapjw/quantum/research/quartz_general_meta25_20261123. They are not paths on the shared ChatGPT container.

Owned production change: src/bin/quartz_c25_incumbent_price.rs only. Owned evidence: this directory's C05C06_RUST* files only. No Cargo.toml, source contracts, old code, data scripts, branch, commits, or publication refs were modified by this worker.

The original src/bin/quartz_c24_cheap_certificate.rs remains byte identical:
SHA256 5a7cc595c267d5337f268045d1df8bf5909093ed7920ee50293da76da94cd381.

The new file contains the full inherited source and tests exactly, except old main is renamed c24_main with an allow(dead_code) attribute. The appended C25 section alone was formatted. The complete inherited prefix identity was actually checked after formatting.

Current new source SHA256:
84c3937e42836431145fbde74999a57f98540e6b65760f2d71c6a4f78271a8d8.

## Implemented behavior

The six cyclically interleaved policies are PLAIN_STOP, PLAIN_ALWAYS, FAST_BLOCK, RETAIN_BLOCK, RETAIN_WORK, RETAIN_TIME. The first three directly call the inherited run_policy function. An outer timing window includes C25 metadata construction in each control's total_us; numerical state, arithmetic, decisions, counters and original policy branches are unchanged. Controls retain continuation_us=null because the inherited function does not expose a separate exact continuation timer.

RETAIN initializes from the same four-CG-step ordinary score and smallest-index tie break. It evaluates inherited FAST enclosures, intersects their history and makes at most one safe maximum-lower-endpoint switch per checkpoint. The order is safe free switch, strict unique/weak zero-regret optimum, budget/no-progress exhaustion, price gate, admitted continuation. Budget returns the retained action. Final ordinary selection is separately recorded without an extra evaluator call.

Each admitted continuation uses min(8,remaining budget) CG attempts followed unconditionally by FAST evaluation of all K actions, including zero-CG-progress continuations. The total declared charge is n*(solver_matvecs+fast_evaluations+ordinary_evaluations), n=side^2. It is a cell-stencil accounting convention, not a CPU operation count. Successful retained continuation blocks therefore incur at least K*n additional charge.

Gain is exactly zero when max_other_upper<=incumbent_lower; otherwise subtraction is rounded upward. Work threshold is next_down(2^-30*K*n), clamped nonnegative. Time threshold is next_down(next_down(binary64(1e-6))*predicted_block_us), clamped nonnegative. The latter uses a TRAIN-only point prediction and is marked empirical, not a universal cost certificate. Prediction zero is accepted and disables positive-gap price stopping. A positive-gap PRICE_WORK sets price_stop_certified=true; PRICE_TIME never does.

Retained continuation_us covers the interval after initial solver/readout/ordinary initial selection through subsequent controller work, all continuation CG and fresh FAST evaluations, final output preparation and workspace destruction. It includes initial checkpoint control after the initial readout. The entire native total_us includes validation, initial solver/readout, continuation/controller work and workspace destruction, excluding JSON serialization/transport and offline oracle work.

## Interface

Input uses C24 task/additional/repeats/warmup/trace plus predicted_block_us, numeric finite nonnegative with default zero. Task and top-level unknown fields, including oracle fields, are rejected. Existing finite arithmetic, solver and request caps remain unchanged.

Each result flattens the C24 Outcome and adds c25:

- executed_blocks: retained admitted-block count; controls count completed post-initial readout batches (PLAIN_ALWAYS has one full-budget readout).
- continuation_us: number for retained policies, null for controls.
- charged_cell_stencils: actual total declared charge.
- final_gain_upper and weakly_optimal: retained final diagnostics, null for controls.
- final_ordinary_selected: final fresh ordinary-score argmax.
- price_stop_certified: true only for PRICE_WORK.
- decisions: present only for retained trace=true runs, one entry per trace checkpoint.

Decision fields are checkpoint, incumbent_before, incumbent_after, gain_upper, price_threshold, cost_kind, reason, switch_proof, executed_blocks, charged_cell_stencils. switch_proof contains from, to, challenger_lower, incumbent_upper. Cost kinds are NONE, CELL_STENCIL, TRAIN_PREDICTED_US. Reasons are CONTINUE, INITIAL_CERTIFICATE, BLOCK_CERTIFICATE, ZERO_REGRET_OPTIMUM, BUDGET, NO_PROGRESS, PRICE_WORK, PRICE_TIME.

Strict uniqueness remains in base.certified. Weak optimality is c25.weakly_optimal. A positive-gap price stop is not an optimal-action certificate.

## Actual RED and GREEN evidence

Focused command, unchanged between RED and GREEN:

    cargo test --bin quartz_c25_incumbent_price c25_behavior_tests -- --nocapture

C05C06_RUST_RED_CORRECTED.log: exit 101; 12 failed, 1 passed, 27 filtered out.
C05C06_RUST_GREEN.log: exit 0; 13 passed, 27 filtered out.

The original first RED run is retained in C05C06_RUST_RED.log. Its test adapter counted only FAST/ordinary reads despite delegating the old six-policy suite, so two checks hit an OUTWARD-only count subtraction underflow. That was a test-scaffold implementation error, not a theory or numerical failure. It was corrected before new behavior implementation and the RED command was rerun; all 12 new tests then failed by the intended absent-behavior assertions. The corrected preimplementation source is preserved in C05C06_RUST_RED_SOURCE.rs and its identity is in C05C06_RUST_RED_CORRECTED.json.

The 12 actual new RED-to-GREEN tests:

1. retained_budget_rejects_a_worse_ordinary_reselection
2. strict_max_lower_switch_precedes_price_and_is_auditable
3. endpoint_equality_does_not_switch_and_lower_ties_use_first_index
4. zero_gap_certifies_weak_optimality_without_a_unique_winner
5. work_gate_uses_outward_gap_and_inward_exact_dyadic_threshold
6. time_gate_is_separate_and_zero_prediction_disables_positive_stop
7. budget_and_zero_progress_precede_positive_price_gate
8. zero_progress_block_still_reads_every_action_once
9. zero_budget_keeps_initial_action_and_exposes_one_checkpoint
10. public_suite_runs_exact_six_policies_with_cyclic_repetitions
11. public_input_rejects_negative_nonfinite_prediction_and_oracle_fields
12. trace_off_omits_vectors_and_preserves_compact_decision_result

Reused C25 integration invariant: inherited_controls_and_initial_scores_are_identical.
The 27 inherited C24 tests are unchanged, reused regression evidence.

Full verification commands (each actual exit 0, complete logs retained):

    cargo test --bin quartz_c25_incumbent_price
    cargo test --release --bin quartz_c25_incumbent_price
    cargo build --bin quartz_c25_incumbent_price
    cargo build --release --bin quartz_c25_incumbent_price

Observed environment: rustc 1.99.0 (b940084d7 2026-09-28), cargo 1.99.0 (5f94df478 2026-08-27). OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1. No fast-math or numerical-core change.

Debug binary SHA256 1209b18ba37cdf8ab3a5e17c8a2c812923c77fcfaa2337c3746d6089581ba7fd.
Release binary SHA256 2ddccac45f1fa9c171509bcc329a37f2f9b575fe3c17797e806a74ee4d8febf5.

## Native stream and error verification

Both actual binaries processed 14 line-delimited input cases, returned native exit 0, rejected all 12 invalid cases with explicit error JSON, and continued to two valid requests. Invalid cases: malformed JSON, top-level oracle, task oracle, negative prediction, nonnumeric prediction, excessive budget, zero repeats, excessive warmup, unsupported operator, nonfinite RHS, unsupported tiny RHS, wrong RHS shape.

Valid cases cover zero prediction with trace and positive prediction without trace. Exact debug/release semantic outputs agree after removing only timing fields. Raw request and output lines, stderr and case-level observed error messages are preserved. The inherited stream protocol deliberately reports per-line errors as JSON while retaining process exit 0; this is distinguished from native process failure.

## Skills and frozen contracts used

Actually read PhysMath Research Loop 1.3.0 SKILL.md, C05/C06 definitions in references/CODING_LOOP.md, references/POLICY_INTEGRATION.md, references/SUBAGENT_MODE.md; test-driven-development SKILL.md and writing-good-tests.md; implementation SKILL.md and isolation/red/proof/checkpoint gate files. using-superpowers was read and explicitly excludes a dispatched subagent from its routing.

Read remote CODE_CONTRACT.json, SOURCE_SPECS.json (full JSON parsed and structure checked), C04_RUN_CONTRACT.json, theory/DERIVATION.md, and the inherited native source. Mutation began only after parent CODING_RELEASE at HEAD b87a559d541aaa3952bc879822ddedd915c45b4d. Isolation was checked: designated C25 worktree and branch, native path absent, no preexisting dirty path overlapped owned files.

No child agents were spawned. Parent owns acceptance, commits, stage backup/publication, C07/C08 cohorts and C09 review.

## Checkpoint and remaining obligation

C05 implementation and worker-owned C06 unit/build/native-schema verification are implementation-verified. Actual c24_test_34_1 trajectory regression and independent exact public-seam verification are assigned to /root/c25_data and are not claimed by this report. Fresh TRAIN/CAL/TEST campaigns, model fitting, scientific performance and P2 claims are not established here.

Checkpoint: implementation -> parent C06 integration / selected C09 review.
Completed gates: isolated scope, genuine RED, GREEN, focused inherited regressions, debug/release builds and native error-path proof.
Next check: parent reads the new-source diff and /root/c25_data exact UNIT/adverse-regression evidence, then records C05/C06 stage boundaries before any benchmark.
Action: handoff to parent; no build or benchmark process remains running from this worker.
