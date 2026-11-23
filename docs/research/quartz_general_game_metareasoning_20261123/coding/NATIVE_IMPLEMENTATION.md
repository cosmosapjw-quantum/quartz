# Native C26 implementation and verification

## Status and identity

The native exact-query bridge is implemented in `src/bin/quartz_c26_game_metareasoning.rs`. It uses unchanged path modules `../game.rs` and `../games/tictactoe.rs`; no library, dependency, Cargo, prior policy, prior campaign, or existing game code changed.

- Source SHA256: `145b586fea7ff7526eb30d15d683580678f1c9dd3a433967f536697957327a01`.
- Release binary SHA256: `8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c`.
- Binary: `target/release/quartz_c26_game_metareasoning`, 618160 bytes.
- Frozen contract SHA256: `1821e46989ab21f9259ee9c61e4935824a362da5139d08db13d10fb86c1d0985`.
- Frozen source-specification SHA256: `07826834be13bdb7f779459d0ca7e964ace402af61af286bda1a108503b3a466`.
- Toolchain: rustc 1.99.0 (b940084d7 2026-09-28), cargo 1.99.0 (5f94df478 2026-08-27).
- Stage head observed after C04: `0b1cee307c67b37b804264407407daaa731b637a`.
- Identity, exact commands, and all ten BASELINE protected-file checks are retained in `evidence/NATIVE_IDENTITY.json`; all protected hashes match.

This is scoped implementation and unit-fixture verification. The independent Python audit, frozen TRAIN/CAL/TEST campaign, broader repository suite, independent C09 review, stage publication, and four-archive restore belong to their assigned subsequent steps. No empirical priced gain is claimed here.

## Equation-to-function mapping

| Contract object | Production functions | Semantics |
|---|---|---|
| Legal public root | `Request::validate`, `replay_history` | Replay from `TicTacToe::initial()`; reject illegal moves before unchecked `apply_move`. No `from_board` use. |
| Exact root value | `root_value`, `exact`, `utility` | Terminal sign is root_player times current_player times existing negamax outcome; integer minimax in {-1,0,1}, then map to {0,.5,1}. |
| Exact query cost | `exact` | One count per entered state including terminal; visit every legal subtree, no transposition cache, alpha-beta, or value-triggered pruning. |
| Paid four-path cues | `SplitMix64`, `rollout`, `candidates` | Every starting/terminal state counted; terminal child evaluated once, represented by four equal outcome counts. |
| TRAIN model | `Model::validate`, `Model::predict` | Feature bin then remaining then global probability lookup; remaining then global cost lookup. Native consumes frozen model; fitting belongs to Python. |
| Common initialization | `initialize` | Terminal win skips exact initial query; otherwise largest raw cue mean, lowest move on tie, then best known incumbent across all terminal and solved children. |
| Positive excess | `positive_excess` | Exact three-atom expected positive part. Zero atom is also included for mathematical negative-index boundary checks. |
| Pandora | `reservation_index`, `choose` | Guarded three branches, no zero denominator; strict index greater than incumbent; fixed law and cost after reveals. |
| Gibbs | `softmax`, `sample_probabilities`, `choose` | Uniform-reference normalized weights, max-subtracted logits, an actual SplitMix uniform draw and inverse CDF selection. Nonpositive FOCUS remains eligible at b<1. |
| Runtime decision | `run_one` | Exactly known incumbent only; equal later reveals retain incumbent; no offline unqueried child labels. |
| Public transport | `handle_line`, `serve` | One JSON response per input line; validation failures become error objects and the stream continues. |

All six policies use the same regenerated cue/initialization path and pay it. ALL_ACTIONS opens remaining moves in ascending order even at known b=1. RANDOM_ORDER and rational policies stop at b=1. Additional budget counts only additional nonterminal exact queries.

Model probabilities are strictly positive finite triples, checked against sum-one tolerance 1e-12 and normalized before arithmetic. Node predictions are finite in [1,1000000]. All frozen price/temperature fields are checked exactly. Serde rejects unknown request/model fields. Feature keys and remaining keys require canonical nonnegative integer syntax and the prescribed ranges/count sums.

## Output semantics and independent-replay schema

Survey top level is `schema,mode,id,candidates,cue_nodes,measurement_elapsed_us`. Candidate fields are `move,terminal_known,remaining,counts,rollout_mean,u,solve_nodes,cue_nodes`. A terminal candidate has remaining=0, solve_nodes=1 and cue_nodes=1. Its known terminal observation is reused; solve_nodes describes the size of its trivial exact traversal, not a second paid traversal. Survey is explicit offline measurement for fitting/audit and is not a runtime model input.

Run top level is `schema,mode,id,runs,warmup_elapsed_us,execution_elapsed_us`. Runs are serialized in repeat/canonical-policy order after the rotated execution order has completed. Repeat is zero based. `initial_move` means the best known action after initialization; `initial_solve_move` is nullable and identifies the actual nonterminal initial exact query. These can differ if a known terminal child dominates the initial solved child.

`query_moves` contains additional queries only. `initial_nodes` counts the nonterminal initial exact traversal or zero. `additional_nodes` is the sum of actual additional exact traversals. `charged_nodes=cue_nodes+initial_nodes+additional_nodes`. `predicted_additional_nodes` sums frozen predicted node means for those same additional queries; it does not replace measured costs. `final_u` and `selected_move` are exact-known observations.

When trace=true, each run adds:

- `predictions`: records `move,remaining,counts,terminal_known,probabilities,predicted_nodes,observed_u`. Probability/cost are null for terminal candidates. `observed_u` is non-null only for already known terminal candidates. It is null for every nonterminal candidate, including the subsequently initialized candidate.
- `decisions`: records `b_before,remaining,candidates,stop_probability,chosen_move,choice,uniform_draw,queried_u,queried_nodes,b_after,reason`.
- Each decision candidate is `move,g,q,z,probabilities,predicted_nodes,probability`. The plural probabilities is the three-class model law; singular probability is that action's actual choice mass for this decision.
- `remaining` and decision candidates are in ascending move order. STOP is the first mass in internal inverse-CDF selection. `chosen_move=null` and `choice="STOP"` encode stopping; a numeric move and `choice="FOCUS"` encode an actual paid query.
- `uniform_draw` is non-null only when Gibbs actually samples. RANDOM_ORDER uses the specified rejection-sampled uniform index and can be replayed from the public generator seed.
- `queried_u=null,queried_nodes=0` for STOP. Every run ends in an explicit STOP decision. Forced stops give STOP probability1 and all move probabilities0. ALL_ACTIONS and greedy/index decisions are one-hot; RANDOM_ORDER is uniform over remaining moves; Gibbs uses its calculated categorical distribution.

Final reasons are `initial_stop,cap_reached,exhausted,known_win,nonpositive_voc,reservation_below_incumbent,sampled_stop`. Forced-stop precedence is INITIAL_STOP, then cap, then exhaustion, then known-win. Query reasons are `all_actions,random_order,positive_voc,reservation_above_incumbent,gibbs_sample`.

## RNG and timing boundary

SplitMix64 uses explicit wrapping u64 constants from the contract. Index draws reject r below 2^64 mod n, then use r mod n; f64 uniform uses the upper53 bits. Child cue seed is cue_seed XOR wrapping((move+1)*GAMMA). Policy seed is cue_seed XOR 0xd1b54a32d192ed03 XOR policy_seed, reset for every policy and every repetition. Repetitions therefore change timing only.

Root history validation and model normalization happen before decision timing. Each `run_one` timer covers candidate allocation, cue generation, feature lookup, initialization, controller operations, queries and temporary-state destruction. Only output metadata escapes the timed scope. Per-phase meanings:

- `cue_us`: candidate allocation, all paid cues, model feature/cost lookup, and cue-count summation.
- `initial_us`: initial candidate selection, initial exact traversal if needed, and known-incumbent choice.
- `query_us`: elapsed sum around actual additional exact solver calls.
- `controller_us`: total elapsed minus the preceding three phases, including RNG/control bookkeeping, remaining arrays, trace metadata if requested, and temporary work-state cleanup.
- `elapsed_us`: entire decision timer after temporary state is dropped.
- `warmup_elapsed_us`: sum of native elapsed_us across discarded warmup runs.
- `execution_elapsed_us`: run dispatcher wall time through warmups, recorded repetitions and output ordering, before conversion of the run records to a JSON value. This includes scheduling/aggregation overhead outside individual decision timers.
- `measurement_elapsed_us`: survey cue generation, offline exact measurements and candidate-result construction after root validation.

Parsing, serialization, process startup and file/pipe I/O are outside elapsed_us. Python roundtrip residuals and total experiment wall time must retain those costs separately. Trace runs perform additional diagnostic arithmetic and allocations and are excluded from campaign timing estimates. Floating exponential underflow may produce zero choice masses at extreme valid cost parameters; no positive-Q mask is introduced to avoid it.

## Executed red/green proof

Before full implementation the executable RED scaffold deliberately omitted the required root-frame transformation, positive-excess law, reservation law and Gibbs weighting. It did not assert a bug in the unchanged GameState semantics. Four real numeric/behavior assertions failed after successful compilation; actual Cargo exit code101 is retained in `NATIVE_RED.json` and `NATIVE_RED.log`, with exact scaffold bytes in `NATIVE_RED_SCAFFOLD.rs`.

The implemented focused suite then passed all18 tests (14 C26 +4 imported game tests), exit0. A behavior-preserving refactor replaced scratch use of a probability field with an explicit incumbent argument to the chooser; final focused tests again passed all18, exit0.

Final commands:

```text
cargo test --offline --bin quartz_c26_game_metareasoning -j 2 -- --nocapture
cargo build --release --offline --bin quartz_c26_game_metareasoning -j 2
rustfmt --edition 2021 --config skip_children=true --check src/bin/quartz_c26_game_metareasoning.rs
```

Actual wall times recorded by the subprocess wrapper: targeted suite0.514s and release build11.848s. Both Cargo commands exited0; formatter exited0. No package installation or campaign run occurred. The initial green and final green/build logs and exit records are retained separately.

The exact tests cover root-frame signs for X/O, full-tree node multiplicities, independently calculated cue/count/value literals, terminal single-charge initialization, ALL_ACTIONS continuing after a known win, shared initialization, cap safety, nondecreasing known utility, model fallback and normalization, invalid JSON/history/model rejection with stream recovery, repeat determinism, non-leakage of unqueried oracle values, three-point index boundaries including degenerate distributions, stable Gibbs probabilities, and common-shift/positive-scale invariance.

The public fixture history [0,4,1,8], cue_seed123 has total cue_nodes61. Its candidate tuples (move,counts,cue_nodes,u,solve_nodes) are independently enumerated literals:
(2,[0,0,4],1,1,1), (3,[0,0,4],14,1,33), (5,[0,2,2],16,.5,47), (6,[0,1,3],16,1,37), (7,[2,0,2],14,0,43). INITIAL_STOP charges61; ALL_ACTIONS additionally traverses160 nodes and charges221.

For Q=(0,.2,-.1), tau=.1, expected Gibbs probabilities are (0.11419519938459448,0.8437947344813395,0.042010066134066035). The prospective test uses N=100000, seed260023, absolute frequency tolerance0.006 per category; actual counts were (11268,84391,4341), within tolerance. This is a mechanism sampling test, not a game-performance estimate.

## Checkpoint and next event

Source skill: Keystone Implementation, with PhysMath scientific coding and Superpowers meaningful RED/green. Isolation, numeric RED and scoped executable proof are complete. Known parent/data-worker files were not modified. Native protected-file audit passes all ten identities.

Target: parent-managed C05 integration checkpoint, then C06/C07 independent Python production-output audit and C09 change review in the frozen stage sequence. First next check: inspect this exact source against CODE_CONTRACT and replay its output with the independently implemented Python board/minimax/RNG/trace oracle. Parent owns the single broader Cargo attempt and all commits, pushes, archive publication and restores. No new user permission is required within the already authorized task.

Action: continue now through the parent handoff. Remaining proof scope: independent full-state audit, TRAIN/CAL/TEST execution/calibration/inference, repository regression outcome, independent source review and restore/publication receipts.
