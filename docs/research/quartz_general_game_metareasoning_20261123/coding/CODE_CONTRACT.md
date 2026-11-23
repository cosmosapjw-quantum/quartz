# C26 frozen implementation contract (R09)

Frozen before production implementation or policy outcomes. Operative initialization is R03, not the earlier constructor packet. Changes needed to fix a demonstrated implementation defect must preserve old evidence, explain the defect and be published; TEST-driven parameter changes are forbidden.

## Scope and source identity

Root /home/cosmosapjw/quantum/research/quartz_general_meta26_20261123; C25 parent d6019d674ae76524b1895016f49b82d4f7b001b2. Reuse unchanged src/game.rs and src/games/tictactoe.rs through path modules in one new binary. No lib.rs exists. Do not change Cargo.toml/lock, existing policies, games, NN behavior, C25 files or other worktrees. Allowed production source: src/bin/quartz_c26_game_metareasoning.rs. Tests: tests/test_c26_game_metareasoning.py. Harness/data/figures live in this C26 doc subtree.

## Mathematical and numerical contract

Exact utility is root-frame minimax in {-1,0,1}, mapped to {0,.5,1}. The recursive solver uses only GameState legal_moves/apply_move/current_player/is_terminal/outcome and no TT or alpha-beta, counting every entered state (including terminals) once. Max at root-player turns, min at opponent turns. All legal children remain available; an exact query reveals a child's full value once.

Cue: four independent uniform random playout paths per nonterminal root child. Count every visited state in each path, including starting child and terminal. Terminal children are evaluated once, marked known, and encoded as four identical outcome counts; there are not four paid traversals. Feature remaining is the child's legal-move count (terminal semantic legal list may be empty; terminal candidates do not need a predicted query).

Initialization: if a known terminal win exists, retain lowest-index winning child, no additional initial solve. Otherwise solve the nonterminal child with highest raw cue sample mean, ties lowest action index. Retain the best known value across this and terminal children, initial ties lowest move. If all children terminal, select best known. Later equal-valued reveals retain incumbent. All policies regenerate and pay identical initialization; baseline INITIAL_STOP stops then. Additional query budget does not count initial solve or terminal cue observations.

Costs: lambda_nodes=2^-10 utility per visited state; tau=2^-5 utility. Compute Q_a=g_a(b)-lambda_nodes*predicted_nodes and Q_STOP=0. g is exact expected positive excess from the fitted three-point law. Index uses R05's guarded three-branch equation. No divisions by zero. Full-horizon primary cap=9; cap1 and cap2 are labeled constrained comparisons. No price or temperature fitting after outcomes. Report actual total charged nodes, additional nodes and measured wall time. Secondary wall price=10^-6 utility/microsecond. Training/offline oracle/JSON/I/O costs are reported separately from runtime decision cost, not hidden in it.

GREEDY chooses largest Q if strictly >0, ties move index, else STOP. PANDORA chooses largest z if strictly >b, ties move index, else STOP. Model laws/predicted costs stay fixed after reveals; only b and remaining set change. Both are approximations to the actual correlated game population.

GIBBS samples proportional exp(Q/tau) with a uniform reference over STOP (first) and remaining moves in ascending order, using max-subtracted log-sum-exp. At known b=1 all FOCUS are masked and STOP is returned. At b<1 do not mask nonpositive Q. Finite floating underflow may yield zero for probabilities below representable scale; finite actual cohort parameters must be reported. This is a regularized local policy, not a global net optimizer.

Policies, in canonical order: INITIAL_STOP, ALL_ACTIONS, RANDOM_ORDER, GREEDY_VOC, PANDORA, GIBBS_VOC. ALL_ACTIONS opens remaining moves ascending until cap or exhaustion even if b=1. RANDOM_ORDER samples remaining moves uniformly, stopping at cap/exhaustion or b=1. Rational policies also obey cap and exact b=1 certificate. All retain exact-known incumbent.

## Deterministic random generator

Use explicit SplitMix64, wrapping u64: state += 0x9e3779b97f4a7c15; z=(z^(z>>30))*0xbf58476d1ce4e5b9; z=(z^(z>>27))*0x94d049bb133111eb; return z^(z>>31).
Uniform index n: threshold=2^64 mod n; reject values below threshold, then r mod n. Uniform f64: (r>>11)*2^-53 in [0,1).
Cue child seed=cue_seed XOR ((move+1)*0x9e3779b97f4a7c15 modulo2^64); one child generator supplies its four paths. Policy generator seed=cue_seed XOR 0xd1b54a32d192ed03 XOR policy_seed, reset for each policy and repetition. Thus stochastic choice is deterministic for a public request and repetitions differ only in timing. policy_seed is one of 0,1,2,3 in the campaign.

## TRAIN-only fitted model

Feature key is 'remaining:loss:draw:win', last three counts sum to4. For observed TRAIN feature bins use (n_class+1)/(n_bin+3). For unseen bins use the same +1-smoothed TRAIN distribution by remaining, else smoothed global distribution. No minimum-bin threshold. Fit only nonterminal candidates; terminal values are known exactly.

Predict solve_nodes by arithmetic mean of actual TRAIN nonterminal exact traversals grouped by remaining. Fall back to global mean; require finite cost>=1. Store raw training counts, sums, source IDs and hashes separately so every fitted parameter is reproducible. Model is frozen before CAL and then unchanged through TEST; CAL diagnoses without tuning.

Model JSON schema:
{"schema":"c26-categorical-v1","feature_probs":{"3:0:2:2":[p0,ph,p1]},"remaining_probs":{"3":[p0,ph,p1]},"global_probs":[p0,ph,p1],"remaining_costs":{"3":17.0},"global_cost":17.0,"lambda_nodes":0.0009765625,"tau":0.03125}.
Validate known schema, finite strictly positive probabilities summing to1 within1e-12 (normalize by that sum before arithmetic), valid feature counts, remaining range0..8, finite costs in[1,1000000], exact frozen lambda/tau. Reject unknown model fields. Probability and node prediction helpers are deterministic.

## Native JSON-lines interface

The binary reads one request per stdin line, emits one response per line and does not log diagnostics to stdout. A malformed/invalid request emits {"error": "..."} and processing can continue. Validate before invoking unchecked GameState.apply_move.

Request (all fields required except model nullable):
{"id":"C26-p4-...","history":[0,4,1,8],"cue_seed":123,"policy_seed":0,"mode":"run","additional":9,"repeats":9,"warmup":1,"trace":false,"model":{...}}.
mode is run or survey. Reject unknown fields, terminal root, terminal continuation, illegal/duplicate/out-of-range moves, history length>8, id empty or>128bytes, cap outside1..9, repeats outside1..15, warmup outside0..2, model missing for run. Model-free survey convention: additional9,repeats1,warmup0,tracefalse,policy_seed0. No offline utility/value/oracle fields are allowed. Root is built by replaying an actual legal history from TicTacToe::initial() or the existing trait's equivalent constructor, never unchecked from_board.

Survey returns schema='c26-game-v1', mode,id, and 'candidates': records with move, terminal_known, remaining, counts[3], rollout_mean, u, solve_nodes. It computes all child values as explicit TRAIN measurement, not as free run inputs. Record cue_nodes as well.

Run returns schema,mode,id and 'runs': records with policy,repeat,initial_move,initial_u,selected_move,final_u,cue_nodes,initial_nodes,additional_nodes,charged_nodes,additional_queries,query_moves,predicted_additional_nodes,elapsed_us,cue_us,initial_us,query_us,controller_us,stop_reason. query_moves contains only additional paid queries. Prices are applied in the analysis using actual counts/time, never substituted predicted costs. charged_nodes=cue_nodes+initial_nodes+additional_nodes. Record initial_solve_move nullable to disambiguate zero-query initialization.

For trace=true, each run additionally returns 'predictions': records move,remaining,counts,terminal_known,probabilities,predicted_nodes, and 'decisions': records b_before,remaining moves, candidate Q/z/g/probabilities, chosen move or STOP, actual queried u/nodes, b_after, and reason. Record the uniform Gibbs draw for independent replay. Output naming can add unambiguous fields; changing required field meanings needs a defect note. All candidates' offline u must not appear as run predictions. Revealed terminal/queried u can appear as observations.

## Timing boundary

Construct/validate the public root and deserialize the frozen model outside the decision timer; these are request transport. For each policy/repetition regenerate cues, fit-feature lookup, initial verification, controller choices and exact queries inside the timer. Include temporary candidate allocation/cleanup and all policy arithmetic. Read elapsed only after temporary work state is released; retain output metadata for serialization after the timer. Count this boundary explicitly; JSON parsing/serialization, offline oracle, process startup, training and output-file I/O are excluded and reported as experiment overhead.

Use one warmup per policy and nine recorded timing repetitions. Rotate policy execution order by (repeat+policy_seed)%6. Reset RNG every repetition. A separate trace=true one-repeat pass verifies logical signatures but does not contribute timing estimates. Per fixed source/cap/seed/policy require identical initial/final utility, selected move, query order and charged-node components across repetitions and trace pass. Keep raw timings and median per seed. Never treat repetitions or policy seeds as independent source parents.

## Acceptance

- Meaningful red signal before implementation; independent literal/oracle expectations.
- Native release build and targeted Rust tests; invalid JSON/history/model boundary tests.
- Independent Python board/minimax oracle checks actual production results, candidate node counts, all trace choices and safety; separate exact Bellman/index theorem oracle remains available.
- Reproduce public RNG/cues; test both players and all reachable nonterminal root outcomes where feasible. Check no oracle fields accepted.
- Statistical mechanism tests: three-point boundaries, constant/common shifts, cost/utility scales, stable softmax, actual sampling frequencies within prospectively fixed tolerance.
- TRAIN parameter audit, fixed CAL/TEST identity, parent-level paired inference, predicted-vs-actual cost calibration and probability Brier diagnostics.
- Focused regressions plus one bounded broader repository test attempt, with preexisting/environment failures reported separately.
- C09 independent code/science review and actual four-archive restore; C10 all stage receipts and final remote HEAD/archive verification.
- Stop after C26; propose C27 with the observed evidence, without starting it.
