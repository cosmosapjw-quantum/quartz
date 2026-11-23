# C28 native substrate mapping — read-only R2/C2 preparation

This is a source mapping and proposed executable contract, not a code implementation. Inputs actually inspected:

- `reference_quartz_c26_game_metareasoning.rs`, SHA256 `fd456a45b4a14115c8fc7693e6f8718d01fcadc9cef9337f14829426a2141cfc`.
- `reference_quartz_c27_joint_posterior.rs`, SHA256 `ec757281c871656fc26dedb8e60e564c3de0c5103400889254f35add4eb03105`.

## Smallest independent binary

Add one new `src/bin/quartz_c28_lifted_h2.rs` with the existing relative modules `#[path = "../game.rs"] mod game;` and `#[path = "../games/tictactoe.rs"] mod tictactoe;`. Existing imports are `game::GameState`, `tictactoe::TicTacToe`, `serde::{Deserialize, Serialize}`, `serde_json::{json,Value}`, `BTreeMap`, `std::io::{self,BufRead,Write}`, and `std::time::Instant`. For microbenchmarks also use `std::hint::black_box`. Existing Cargo dependencies already cover these imports. The C26/C27 bins use private helper functions; copying the small frozen substrate with exact provenance is simpler than modifying their visibility or sharing their `main` implementations. Do not change the original bins.

## Reusable C26 source units

| Unit | Source function/type | Semantics to preserve |
|---|---|---|
| Model parser | `Model`, `remaining_key`, `normalize_probs`, `valid_cost`, `Model::{validate,predict}` | Schema `c26-categorical-v1`; three probabilities ordered loss/draw/win; strict positivity; sums within `1e-12`; feature key `remaining:loss:draw:win`; fallback feature → remaining → global. Costs in solver-node units, `lambda_nodes=1/1024`; `tau=1/32` is retained for source compatibility but unused by deterministic H2 |
| Legal root construction | `replay_history` | Histories ≤8 moves; reject illegal/duplicate/out-of-range moves and histories continuing past terminal; root nonterminal |
| Cue RNG | `SplitMix64::{new,next,index}`, `GAMMA` | Unbiased rejection indexing; four random playouts per nonterminal candidate; per-action seed `cue_seed ^ (mv+1)*GAMMA` |
| Root frame / solve | `root_value`, `utility`, `exact`, `rollout` | Terminal integer value `root_player * current_player * outcome`; minimax maximizes iff current player equals root; utility `(v+1)/2`; count every entered node including terminal |
| Candidate preparation | `Candidate`, `rollout_mean`, `candidates` | Legal moves sorted ascending; terminal children known directly; no minimax of unqueried candidates; costs and probabilities from frozen model |
| Common initial verification | `Initial`, `initialize` | If any terminal win exists, do not query another child. Otherwise exactly solve the unknown action with highest rollout mean; strict `>` means smallest move breaks equal cue means. Among all exact-known values, highest value with smallest-move tie is initial incumbent |

In the sequential run retain incumbent on equal revealed value (`v > current_value` required for replacement). Observe terminal children and the paid initial solve in the latent belief before either planner. With deterministic terminal laws, their likelihood tilt should be zero. Forced `known_win`, exhausted candidates or zero remaining query budget must stop before evaluating H2. Cap is two additional paid queries; after the first reveal use H1.

## What C27 actually computes

`AtomLaw::new` creates `f_i(u)=(u-mu_i)/max_supported|u-mu_i|`; deterministic laws use zero. `Belief` has a common nonnegative scalar `rho`, log odds, and observed flags. Its prediction is `p_i(u)[1+rho*tanh(log_odds/2)*f_i(u)]`. `after` uses log-likelihood increments. `score_actions` loops over first action × outcome × second action, calling `predict_at` (including tanh) and a three-term `positive_excess` each time. It does **not** cache the one-step affine coefficients.

Therefore C27 is useful as a correctness reference, but timing a lifted implementation against that repeated-transcendental implementation does not isolate the benefit of lifting. C28's primary comparator should be an optimized normalized-Bayes H2 using the same cached affines as lifted H2.

## Shared fair computational form

Let the binary latent mean be `m`, signed per-action tilt `t_i(u)`, and base marginal `p_i(u)`, with `sum p_i*t_i=0`, `|t_i(u)|<1`, and `m∈[-1,1]`. Then conditional predictive probability is `p_i(u)(1+m*t_i(u))`. Cache once for every action and every incumbent value actually needed:

`A_j(b)=sum_u p_j(u)*max(u-b,0)-d_j`;

`C_j(b)=sum_u p_j(u)*t_j(u)*max(u-b,0)`.

Actual-game incumbents are in `{0,.5,1}`. A synthetic initial `b` may add another cache row; branch thresholds are `max(b,u)`.

For each first query `a` and outcome `u`, set `s=1+m*t_a(u)` and `r=m+t_a(u)`.

- Direct normalized comparator: compute `m'=r/s` once for that outcome, evaluate `max(0,max_{j!=a}[A_j(b')+C_j(b')*m'])`, then multiply by `p_a(u)*s`.
- Lifted: evaluate `p_a(u)*max(0,max_{j!=a}[A_j(b')*s+C_j(b')*r])`.
- Both add the same first-query net term `A_a(b)+C_a(b)*m`.

Both implementations cache and charge the same arrays. Direct's single division is amortized over every second action; lifted adds an extra multiplication per second action. Consequently lifting is not guaranteed faster. Report measured cost, including null or negative speed results. If an action-gap certificate is added, compare it as a separate third implementation so the source of savings remains identifiable.

C27's choice rule is: STOP if maximum incremental score ≤`1e-12`; otherwise choose the smallest move with score ≥maximum−`1e-12`. Reuse exactly for both paths. Record maximum absolute score discrepancy and histories near this decision tolerance. Do not widen the tolerance to make decisions agree.

## Proposed JSONL CLI

One request per line, one result or explicit error per line. Use a tagged request union and reject unknown fields.

**Synthetic request:** `mode="synthetic"`, `id`, unique sorted `moves`, `probabilities` (three-point laws), `tilts` (same shape), `costs` (already utility units), `m`, `b`, `horizon=2`, `repeats`, `warmup`, `trace`. Returns scores and choice for direct/lifted, maximum numerical disagreement, operation counts and paired timings. No GameState or oracle inputs are involved.

**Game request:** `mode="game"`, `id`, legal `history`, `cue_seed`, frozen C26 `model`, `loading_rule`, `rho`, `additional=2`, `repeats`, `warmup`, `trace`. The signed loading rule must be frozen and depend only on permitted runtime cues or action indexing; output the actual loadings. For example, a fixed cue contrast can be a mechanism fixture, but cannot be called a trained or empirically calibrated model. Run direct and lifted independently from the same immutable root/model/seeds, alternately ordering them each repeat. Return initial/final moves and utility, exact query sequence, query/charged nodes, initial/final latent state, forced-stop reason, and timing channels. A controller must never accept `oracle_values`, `optimal_move`, `regret`, or TEST labels as runtime fields. External evaluation can join such labels after the run.

## Timing boundary

Primary real-game elapsed time begins before cue generation/model prediction and ends after the final decision. Include belief construction, shared coefficient caching, conditioning, planning and allocation; include initial and additional exact solves. Report separately cue ns, initial solve ns, additional solve ns, controller residual ns, and cache ns. Both comparators pay their own cache costs. Parse/validate/history reconstruction and JSON serialization may be measured separately as in C27, with the exclusion explicitly named. Use `trace=false` for timing and a separate traced pass for policy equality; traced allocations are not representative controller costs. Warmups are separate, order alternates, same CPU/thread conditions apply. Microbenchmarks call both cached planners on identical states through `black_box`; they support a kernel-cost claim only, not end-to-end efficiency or playing strength.

No code was implemented, compiled or tested in this mapping subtask. Skill context: PhysMath Research Loop 1.3.0 plus selected C2 definition from CODING_LOOP, source-only R2/C2 mapping and the existing integrated evidence rules.
