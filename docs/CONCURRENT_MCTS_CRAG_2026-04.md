# Concurrent Parallel MCTS CRAG Baseline — 2026-04

## Purpose

This document establishes a research-backed baseline for evaluating QUARTZ's
current concurrent MCTS design on `gomoku7`.

It serves two purposes:

1. Provide a durable external reference for what standard high-throughput,
   neural-network-guided concurrent MCTS designs usually look like.
2. Compare QUARTZ's present implementation against that reference so later
   optimization work can distinguish:
   - algorithmic defects
   - orchestration defects
   - measurement contamination caused by local bugs

This is a CRAG baseline, not a final design spec.

## Scope

This baseline focuses on:

- concurrent / tree-parallel MCTS
- neural network batching
- evaluation orchestration
- CPU/GPU utilization implications
- search scheduling design

It does **not** justify changing search semantics by default. QUARTZ exists for
ablation work, so systems changes should preserve result identity for a fixed
algorithm/config pair whenever possible.

## Audit Method Note: Prompt-Engineering Frame Calibration

The requested audit loop mixes established prompting/reasoning methods with
custom procedural labels. Before using that loop as an investigation scaffold,
the labels should be normalized:

### Methods with clear literature

- **SELF-DISCOVER**
  - Reference: https://deepmind.google/research/publications/64816/
  - Meaning used here:
    - first identify candidate reasoning structures before committing to one
      path

- **Step-Back Prompting**
  - Reference: https://huggingface.co/papers/2310.06117
  - Meaning used here:
    - abstract away from local implementation details and ask what problem class
      we are really solving

- **CRAG (Corrective Retrieval-Augmented Generation)**
  - Reference: https://huggingface.co/papers/2401.15884
  - Meaning used here:
    - evaluate retrieval quality, correct weak retrieval with web search, and
      filter evidence before using it in conclusions

- **CoVe (Chain-of-Verification)**
  - Reference: https://huggingface.co/papers/2309.11495
  - Meaning used here:
    - draft a provisional conclusion, then explicitly generate and answer
      verification questions before finalizing the conclusion

- **CCoT (Contrastive Chain-of-Thought)**
  - Reference: https://huggingface.co/papers/2311.09277
  - Meaning used here:
    - contrast plausible-good and plausible-bad explanations rather than only
      refining a single story

- **Metacognitive prompting**
  - Reference: https://huggingface.co/papers/2308.05342
  - Meaning used here:
    - structured self-monitoring: what do we know, what are we assuming, what
      evidence is missing

### Methods that are usable but less canonical in the exact requested wording

- **self-ask / metacognitive self-ask**
  - No single canonical research label exactly matches the requested phrase.
  - Operational meaning used here:
    - recursively generate the next uncertainty-reducing question before
      committing to a diagnosis

- **adversarial self-ask**
  - Also not a single standard paper label.
  - Operational meaning used here:
    - force a red-team pass that asks how the current favorite explanation could
      be wrong

- **PDR (Parallel-Distill-Refine)**
  - Emerging structured-inference label rather than a universally standard one.
  - Practical meaning used here:
    - consider multiple architectural explanations in parallel
    - distill to the few that survive evidence
    - refine only the survivors

### Consequence for this audit

The audit below therefore uses:

- literature-backed methods where available
- explicit operational definitions where the requested label is non-standard

This is important because the audit needs to be reproducible rather than
dependent on ad-hoc interpretation of method names.

## Local Observations That Must Be Explained

From current local artifacts:

- `models/alphazero_gomoku7/autotune_profile.json`
- `models/alphazero_gomoku7/train_log.jsonl`
- `models/alphazero_gomoku7/eval_matches.jsonl`
- `artifacts/runtime_monitor/gomoku7_eval_batch_phase1_retry/summary.json`

the current system shows:

1. Self-play improved after batching work, but callback density is still high.
2. `io_time_s` still dominates `codec_time_s`, so serialization is no longer the
   first-order problem.
3. Evaluation is still too expensive relative to the game size.
4. Current batched evaluation results are contaminated by a correctness bug:
   the newest `eval_matches.jsonl` tail contains repeated void games with
   `error: "'gi'"`, and `train_log.jsonl` records later eval rows as
   `games: 0`.

That last point matters: any architectural conclusion that relies on the latest
batched evaluation phase must be treated as provisional until the bug is fixed.

## External Baseline: What Standard Concurrent Neural MCTS Looks Like

### 1. AlphaGo Zero / AlphaZero baseline

AlphaGo Zero introduced the modern "policy/value network + MCTS self-play"
stack, and AlphaZero generalized it. The important systems-level takeaway is
not just that search is parallelized, but that search and neural evaluation are
already treated as tightly coupled components of one execution loop rather than
as loosely coordinated independent workers.

Relevant references:

- DeepMind / Nature AlphaGo Zero paper:
  https://pubmed.ncbi.nlm.nih.gov/29052630/
- AlphaZero paper:
  https://pubmed.ncbi.nlm.nih.gov/30523106/

The AlphaZero paper explicitly describes MCTS as the move-selection mechanism
and reports fixed hardware for each search during evaluation. That is not
directly a batching paper, but it establishes the canonical baseline:

- the NN is part of the search loop, not an external convenience process
- MCTS is the expensive control structure
- self-play/evaluation are measured in terms of search throughput, not just
  learner throughput

### 2. WU-UCT: parallelism is hard because rollouts depend on fresh statistics

WU-UCT is a strong baseline for the claim that parallel MCTS is inherently hard
to scale because each rollout depends on statistics updated by prior rollouts.

Reference:

- OpenReview:
  https://openreview.net/forum?id=BJlQtJSKDB

Key result:

- Standard parallelization loses quality because rollouts depend on stale tree
  statistics.
- WU-UCT introduces "unobserved samples" to track in-flight simulations rather
  than bluntly perturbing node values.

Why this matters for QUARTZ:

- If throughput is poor, the first question is not "should we add more
  threads?" but "what synchronization and in-flight bookkeeping policy are we
  using?"
- If the system is dominated by callback waiting, poor scaling can be a systems
  issue even when the search equations themselves are reasonable.

### 3. KataGo: analysis/search throughput comes from cross-position batching

KataGo's analysis engine is one of the clearest practical references for
high-throughput neural MCTS.

Reference:

- KataGo analysis engine doc mirror quoting upstream text:
  https://gitee.com/LZY2006/KataGo/blob/master/docs/Analysis_Engine.md

Important design points from that document:

- KataGo supports analyzing multiple positions/games in parallel.
- It distinguishes:
  - `numSearchThreadsPerAnalysisThread`
  - `numAnalysisThreads`
- The explicit reason is to exploit **cross-position batching** on modern GPUs.

This is directly relevant to QUARTZ:

- QUARTZ currently still behaves too much like multiple search clients calling
  back independently.
- Standard high-throughput design instead treats many positions as one shared
  inference opportunity.

### 4. Leela Chess Zero: batching is the main scaling bottleneck

Lc0's design notes are unusually explicit about why neural MCTS stops scaling.

References:

- Lc0 overview:
  https://draft.lczero.org/dev/lc2/overview/
- Lc0 batching note:
  https://lczero.org/dev/old/lc2/batching/

Key takeaways from those notes:

- CPU cache locality is a major limiter in tree traversal.
- The search algorithm may fail to scale past a small number of GPUs/threads.
- Dependence on single-eval latency is itself a scalability problem.
- Improving batch gathering is described as the main obstacle to scaling.
- Visit gathering "one by one" is explicitly called out as the wrong shape for
  large batches.
- Message/event passing is suggested as a more scalable architecture.

This is highly aligned with QUARTZ's local symptoms:

- CPU and GPU both underfill.
- Batch sizes remain smaller than desired.
- The control plane still looks too request/response-heavy.

## CRAG Comparison: Standard Design vs QUARTZ Current State

### Where QUARTZ matches the literature

1. QUARTZ already uses neural-guided MCTS rather than separating policy/value
   from search conceptually.
2. QUARTZ has already moved away from pure JSON payload traffic toward binary
   and SHM payloads.
3. QUARTZ now recognizes that `n_threads=1` plus multi-game search should still
   batch rather than flood single eval requests.

### Where QUARTZ still diverges from the standard high-throughput pattern

1. **Python still owns too much control-plane responsibility.**
   Standard practice in KataGo/Lc0-like systems is much closer to an engine-side
   coordinator or broker.

2. **Inference is not yet governed by a true global broker.**
   QUARTZ improved request merging, but the architecture still behaves like many
   searches negotiating with Python rather than one central batch scheduler.

3. **Evaluation orchestration is not yet first-class.**
   The latest attempt at batched Rust evaluation introduced a correctness bug.
   That suggests the design direction may be right, but the implementation is
   not yet trustworthy enough to measure.

4. **Search throughput is still too sensitive to callback density.**
   Lc0's batching notes suggest that visit-by-visit gathering is precisely the
   sort of structure that stops scaling, and QUARTZ still looks closer to that
   model than to a wave/broker model.

## Local Evidence Against "MCTS Math Is Wrong"

The current evidence does **not** strongly suggest that QUARTZ's MCTS equations
or algorithm family are the main problem.

Reasons:

1. Self-play throughput changed materially when orchestration changed.
   That means systems topology matters a lot.
2. `io_time_s` dominates `codec_time_s`.
   That points to wait/cadence/orchestration, not policy/value arithmetic.
3. Multi-game batching improvements changed throughput without changing search
   semantics.
4. The latest evaluation path is currently bugged, so "evaluation is slow
   because the algorithm is bad" is not supported by reliable evidence yet.

In other words:

- there is no strong evidence yet that the **MCTS design itself is invalid**
- there is strong evidence that the **execution topology is suboptimal**

## Local Evidence Against "Just Keep Tuning Heuristics"

The current evidence also argues against further Python-side heuristic accretion:

1. Hardcoded worker caps were only a partial problem.
2. SHM and binary payload work helped but did not remove the main wait.
3. Adaptive tweaks improved local cases but did not eliminate the core
   under-filled pipeline.
4. The latest evaluation restructuring created a correctness regression,
   showing that piecemeal orchestration changes are now in a risky zone.

Conclusion:

- more ad-hoc timeout tuning is unlikely to produce the next major gain
- the next real gain will probably require moving orchestration deeper into Rust

## Adversarial Audit

### self-discover

Current symptoms:

- GPU can be busy while CPU stays relatively low.
- Evaluation appears thread-heavy yet not meaningfully productive.
- Self-play still oscillates between short bursts and long wait-heavy periods.
- The latest batched evaluation path is functionally unreliable.

### step-back

The real goal is not to maximize CPU utilization. The real goal is:

- reduce wall-clock iteration time
- reduce eval wall-clock per scored game
- reduce callback density
- improve batch fill
- preserve algorithm semantics for ablation

### metacognitive self-ask

Questions that matter:

1. Is the system compute-bound or orchestration-bound?
2. Are we seeing an MCTS math problem or a scheduling problem?
3. Are current evaluation measurements even valid?
4. Is pushing more work into Rust a principled systems move, or just a language
   preference?

### web CRAG

The external literature and engine docs converge on three practical lessons:

1. Parallel MCTS is fundamentally synchronization-sensitive.
2. Throughput depends on cross-position batching, not just more threads.
3. Brokered / engine-side scheduling is more scalable than request-by-request
   host-side mediation.

### CoVe

Evidence that pushing more into Rust is justified:

- Lc0 explicitly identifies batching and latency dependence as scaling limits.
- KataGo explicitly organizes search to exploit cross-position batching.
- QUARTZ local traces show waiting still dominates payload handling.

Evidence that caution is still necessary:

- current batched evaluation path is not yet correct
- therefore the latest eval-phase timing is contaminated

### adversarial self-ask

Counterargument:

- maybe QUARTZ's MCTS controller design is itself flawed, and moving it to Rust
  will only preserve a bad algorithm faster

Response:

- possible, but not currently well-supported
- the stronger evidence is that orchestration is limiting observed throughput
- a fair baseline path without QUARTZ controllers should therefore be preserved
  on the same infrastructure for direct comparison

### CCoT

Most likely explanation:

- QUARTZ is paying a structural penalty for having the hottest scheduling logic
  outside the engine.
- The search algorithm may still have controller-level ablation questions, but
  those are separable from the systems bottleneck.

### PDR

Parallel:

- build Rust-native evaluation runner
- build Rust global inference broker
- preserve `search_profile=quartz|baseline`

Distil:

- keep only changes that materially reduce:
  - eval wall-clock
  - callback count
  - idle gaps

Refine:

- only after architecture stabilizes, revisit SIMD, warning cleanup, and deeper
  CPU-local optimizations

## Verdict

### What is supported

It is supported to conclude that:

1. **Rust-side orchestration is the right next systems direction.**
2. **A true brokered batching model is more aligned with standard concurrent
   neural MCTS design than the current Python-mediated control plane.**
3. **Evaluation should move toward a Rust-native runner before further tuning.**
4. **QUARTZ should keep a baseline controller-free MCTS path on the same
   infrastructure to protect ablation integrity.**

### What is not yet supported

It is **not** supported to conclude that:

1. QUARTZ's MCTS math is the root cause of the observed throughput problem.
2. More Python-side heuristics are likely to unlock the next major gain.
3. The latest batched evaluation timings are trustworthy enough to justify large
   architecture decisions without first fixing correctness.

## Immediate Implication

Before drawing any performance conclusions from evaluation:

1. fix the current batched evaluation correctness bug (`'gi'`)
2. re-establish valid evaluation measurements
3. then judge the Rust-native evaluation runner / broker migration on clean data

Until then, the correct high-level conclusion is:

- **the evidence favors moving orchestration deeper into Rust**
- **but the current evaluation path is too buggy to use as final proof**

## Code-Reading Audit: Concrete Critique Points In The Current Implementation

This section records criticisms derived from reading the current code, not just
from runtime symptoms.

### 1. Batched evaluation is structurally misnamed on the Rust side

In `src/mcts_server.rs`, `handle_search_nn_multi()` eventually routes through
`run_multi_generic!` and `run_multi_with_eval(...)`.

Current behavior:

- one incoming multi-search batch becomes:
  - one scoped OS thread per state / job
  - and each such job may itself run `n_threads` internal MCTS threads

This creates a multiplicative thread topology:

- `jobs × n_threads`

rather than a bounded global worker model.

Why this is a problem:

- it explains the "thread count spikes high during evaluation" symptom
- it is not how engines like KataGo describe their high-throughput analysis
  architecture
- it means evaluation batching is still mostly a packaging trick around many
  independent searches, not a true global coordinator

This is the strongest code-level argument that QUARTZ still lacks a proper Rust
search/inference broker.

### 2. Current evaluation batching does not preserve valid scoring outputs

`quartz/evaluation.py` now prefers `play_match_tally_batched(...)` when both
engines have `select_moves_batch(...)`.

However, current local evidence shows:

- `models/alphazero_gomoku7/eval_matches.jsonl` tail contains many void games
  with `error: "'gi'"`
- `models/alphazero_gomoku7/train_log.jsonl` records later eval rows as
  `games: 0`

Therefore:

- the new batched evaluation path is currently not benchmark-safe
- any timing conclusions drawn from those eval rows are contaminated

### 3. Evaluation worker autotune is currently bypassed in the exact path that now fails

When batched Rust evaluation is active, the training loop records:

- `"mode": "batched_rust", "workers": 1, "benchmarks": []`

This means the previous evaluation worker benchmark logic is skipped entirely.
That is reasonable if batched Rust evaluation is correct and truly centralizes
work, but problematic when the new path is still buggy.

Effect:

- the system loses a previously measured fallback path
- but the replacement path is not yet trustworthy

### 4. Monitor summary fields are partially out of sync with sample schema

`artifacts/runtime_monitor/gomoku7_eval_batch_phase1_retry/summary.json`
contains useful aggregate information, but the per-phase sample reduction is
currently inconsistent because sample rows use fields such as:

- `cpu_percent_total`
- `proc_tree.total_threads`

while some downstream summarization logic still expects different field names.

This does not create the MCTS bottleneck, but it does reduce confidence in
secondary utilization summaries and can mislead diagnosis if not noticed.

### 5. The Python-side evaluation loop still owns too much policy around correctness and tallying

Even with batched move selection, `quartz/evaluation.py` still owns:

- session creation
- per-game color assignment
- move application
- tallying
- promotion gating handoff

This means:

- Rust is not yet a true evaluation runner
- evaluation overhead is reduced only partially
- there are more cross-layer invariants that can break, as the current `'gi'`
  failure demonstrates

### 6. Current "batched" move timing is partly synthetic

In `play_match_tally_batched(...)`, per-batch wall time is split evenly across
all sessions via `share_ms` when move metadata does not report time.

That is acceptable for coarse telemetry, but it means:

- evaluation timing is not yet a faithful per-game/per-move measurement
- instrumentation is not precise enough to explain where evaluation time goes

So the path is not just buggy, it is also under-instrumented for final
throughput claims.

## Audit Conclusion

The strongest current evidence is:

1. QUARTZ's main remaining problem is still execution topology, not proven MCTS
   equation failure.
2. The current Rust multi-search path is **not yet architected as a bounded
   brokered concurrent MCTS system**. It still fans work out into many
   independent searches too eagerly.
3. Evaluation is currently the least trustworthy part of the system because the
   newest fast path is correctness-broken.

Therefore the most defensible next conclusion is:

- **Rust-native orchestration is still the right direction**
- **but not in the current per-job thread-fanout form**
- **and evaluation correctness must be restored before its timing can be used as
  optimization evidence**

## Second-Pass Adversarial Audit: Additional Concrete Failures

This section records a deeper code-reading pass performed after the first CRAG
baseline was written. These findings are stronger because they connect runtime
symptoms to explicit local contract failures and coverage gaps.

### 1. Batched evaluation has a concrete Python-side contract bug

In `quartz/alphazero_train.py`:

- `_run_batched_eval_groups(...)` assumes every incoming group has a `gi` key.
- the self-play batched path does provide `gi`
- but `NNSearchClient._exchange_search_request().parse_eval_group(...)` builds
  batched groups for `search_nn_multi` **without** `gi`

This mismatch is sufficient to explain the current evaluation corruption:

- `models/alphazero_gomoku7/eval_matches.jsonl` tail contains repeated
  `error: "'gi'"`
- `models/alphazero_gomoku7/train_log.jsonl` eval rows record `games: 0`

This is not an abstract concern. It is a concrete cross-layer protocol bug.

### 2. The current tests missed the real failing path

`tests/test_training_pipeline_regressions.py` checks
`_run_batched_eval_groups(...)`, but only with synthetic groups that already
contain `gi`.

`tests/test_evaluation_pipeline_regressions.py` checks
`play_match_tally_batched(...)` with toy batched engines, but it does **not**
exercise:

- `RustNNEvaluatorEngine.select_moves_batch(...)`
- `NNSearchClient.search_moves_multi(...)`
- `NNSearchClient._exchange_search_request(...)`
- `_run_batched_eval_groups(...)`

as one integrated path.

So the current evaluation fast-path regression was exactly the sort of
cross-layer failure that the test suite was not designed to catch.

### 3. The current `baseline` path is only partially controller-free

`src/mcts_server.rs` currently implements:

- `search_profile=baseline` by setting `cfg.quartz = None`
- and `cfg.vl_mode = Disabled`

That is good, but it does **not** imply a truly minimal MCTS baseline. The same
engine config still carries other engine-level behavior such as:

- `root_forced_win`
- `fpu_reduction`
- exact terminal handling
- the same transport / orchestration / batching stack

This means the current `baseline` label should be read as:

- "same systems substrate, QUARTZ/VL removed"

not:

- "pure minimal MCTS with all higher-level search heuristics removed"

That distinction matters for later ablation claims.

### 4. Evaluation timing is currently only partly trustworthy even when it runs

`quartz/evaluation.py` uses `share_ms = batch_elapsed_ms / len(batch)` when
per-result timing metadata is not available.

This is acceptable for coarse telemetry but not for fine-grained performance
claims. It means:

- timing is partly synthetic
- slow batch members and fast batch members are flattened together
- evaluation instrumentation is still too coarse to localize latency inside one
  batch

So even after the correctness bug is fixed, the current evaluation timing path
will still need more precise instrumentation before it can be treated as the
final evidence base.

### 5. Monitor schema drift is contaminating some secondary summaries

`scripts/profile_training_monitor.py` writes live samples using keys like:

- `cpu_percent_total`
- `proc_tree.total_threads`

Some downstream analyses and earlier summaries assumed older names like:

- `system_cpu_percent`
- `process_tree.thread_count`

This does not create the MCTS bottleneck, but it does create diagnosis noise:

- some phase summaries become `None`
- thread/cpu summaries can silently degrade

This should be treated as measurement contamination, not as a primary search
defect.

### 6. `jobs × n_threads` remains the core topology defect

The earlier conclusion still holds after the second pass.

`run_multi_with_eval(...)` in `src/mcts_server.rs`:

- spawns one scoped host thread per active job
- then each engine may itself run `n_threads`

That topology is the most plausible explanation for:

- thread spikes during evaluation
- low CPU efficiency despite high GPU activity
- weak batch fill relative to total callback count

This is still the strongest systems critique in the codebase.

### 7. QUARTZ itself still contains many fixed theory-derived constants

`src/mcts/quartz.rs` still hardcodes several constants and clamps, including:

- `FLIP_THRESH`
- `FLIP_STABLE_N`
- `ENVAR_CONST`
- `CTM_SCALE_FRAC`
- entropy clamps inside `ParallelismController`

This does **not** prove that QUARTZ math is the current throughput root cause.
But it does mean:

- the controller is not a parameter-free object in practice
- later ablations must distinguish systems changes from controller-constant
  effects carefully

So the current evidence still points first to orchestration, but the controller
layer itself is not "free of fixed choices" and should not be treated as such.

### 8. Batched evaluation is still split by engine and process, not globally brokered

`RustNNEvaluatorEngine` lazily creates its own `NNSearchClient`, and each
`NNSearchClient` owns its own Rust server process.

Inside `MatchRunner.play_match_tally_batched(...)`, active sessions are then
grouped by mover engine:

- one batch for candidate
- one batch for champion

So even the new batched evaluation path still has these limits:

- candidate and champion requests never share one global inference queue
- each side can maintain its own Rust subprocess and thread fanout
- the system is still far from a single global broker for all active eval work

This weakens the performance value of the current batched-eval design even
before the correctness bug is considered.

### 9. Zero-game evaluations are still allowed to mutate rating/publication state

`quartz/evaluation.py` currently does the following in
`TrainingEvaluator.evaluate_checkpoint(...)`:

- `promo = self.gate.evaluate(tally)`
- `self.ladder.advance_period()`
- `self.ladder.record_match(...)`
- `self.calibrator.calibrate(...)`

This happens even when `tally.scored == 0`.

Since `advance_period()` and calibration still run, the system can produce:

- `games: 0`
- but changed `published_elo`
- and changed `elo_gap`

as seen in `models/alphazero_gomoku7/train_log.jsonl`.

That means current evaluation failures do not just waste wall-clock. They can
also pollute the ladder/publication state after a broken evaluation round.

## Revised Practical Verdict

After the second-pass audit, the most defensible ordering is:

1. fix the concrete batched-evaluation protocol bug
2. restore trustworthy evaluation measurements
3. replace the current per-job thread fanout with a bounded Rust broker/pool
4. only then evaluate whether controller-level QUARTZ assumptions are the next
   real bottleneck

This strengthens, rather than weakens, the earlier conclusion:

- **moving more orchestration into Rust is still the right direction**
- **but the next Rust design must be pooled/brokered, not more fanout**

## Third-Pass Audit Addendum

### 10. Evaluation correctness was only one bug; the surrounding contract is still fragile

The missing-`gi` bug has now been fixed, but the surrounding interface remains
fragile for two reasons:

- batched-eval group shape is still assembled across multiple call sites
- engine-local batching and process-local batching are conflated in the same
  code paths

This means the next refactor should define one canonical "eval request group"
contract and make both self-play and evaluation use it, rather than letting
multiple parsers synthesize equivalent-but-not-identical payloads.

### 11. The autotune cache signature is too weak for systems ablations

`load_autotune_profile(...)` validates only:

- profile version
- hardware signature

It does **not** include search semantics or topology-sensitive config such as:

- `search_profile`
- `iters`
- penalty mode or QUARTZ toggles
- thread topology assumptions

So after a systems refactor, old autotune decisions can still be reused even
when the execution substrate changed materially. That makes performance
comparisons less trustworthy unless `--retune` is forced or the signature is
expanded.

### 12. Eval autotune signature also omits search semantics

`eval_autotune_signature(...)` includes hardware, game, eval game count,
iters, `n_threads`, batch size, and backend. It still omits:

- `search_profile`
- penalty mode / QUARTZ-related semantics
- whether batched Rust evaluation is active

This is another way stale measurements can leak across experiments.

### 13. Stop-reason metadata is not always semantically exact

`FixedIterations.stop_reason()` always reports `BudgetExhausted { iterations:
limit }`, and `TimeManager.stop_reason()` reports `elapsed_ms: budget_ms`
rather than the observed stop time.

These are small metadata issues, but they matter because later audit or
ablation analysis can over-trust stop-reason logs as if they were precise
observations rather than controller-declared categories.

### 14. "Baseline" still inherits engine-level search behavior beyond QUARTZ/VL removal

Even with `search_profile=baseline`, the shared `MctsConfig` substrate still
retains defaults such as:

- `root_forced_win = true`
- shared expansion/evaluation substrate
- other engine-level defaults unrelated to QUARTZ/VL

So any result labeled "baseline MCTS" must still be described precisely as:

- same engine substrate, QUARTZ/VL removed

not as:

- pure textbook minimal MCTS

### 15. The monitor can over-report evaluation residence time

The live monitor flips phase to `evaluation` when it sees `Evaluating gen_...`
and only flips back on the next iteration line. So if evaluation ends and the
next training iteration has not yet printed, sampling can stay tagged as
`evaluation` longer than the real evaluation window.

That does not create the performance problem, but it can exaggerate how long
"evaluation phase" appears in the monitor UI.

### 16. Shared dual-model batching is now possible, but only inside one evaluation client

The current fix-first pass added a `model_tag` to QIPC eval requests and a
shared collector pair for `BatchStdioEval`, which means candidate/champion
evaluation can now share a single Rust subprocess and a single batch collector.

This is an important correction because it removes one artificial split between
the two models during batched evaluation.

But it is still **not** the same thing as a global broker:

- self-play and evaluation still do not share one broker
- evaluation scheduling still partly lives in Python
- batch assembly remains collector-local, not run-global

So this change should be read as:

- a necessary intermediate substrate improvement

not as:

- the final concurrent MCTS topology

### 17. High GPU usage with low CPU/RAM/VRAM is only partly a hardware/model-size effect

The current monitor evidence does **not** support the claim that the observed
"high GPU / low CPU" pattern is purely intrinsic to the hardware or to the
small model size.

What is normal:

- VRAM can stay low because the current model is small relative to a 16GB GPU.
- CPU utilization does not need to match GPU utilization in batched
  NN-guided MCTS; some asymmetry is expected.

What is **not** explained by hardware alone:

- Rust QIPC summaries remain dominated by `io_time_s` rather than codec time.
- Process thread counts can become very high while system CPU utilization stays
  modest.
- Evaluation throughput can remain poor even when the GPU appears busy.

This pattern is more consistent with:

- broker-less orchestration
- callback-heavy scheduling
- worker threads waiting on inference responses

than with:

- a fundamentally GPU-bound steady-state design

So the correct interpretation is:

- part of the utilization pattern is normal for a small NN-guided search stack
- the stronger symptom is still topology-driven underfilling / waiting

This means future optimization decisions should continue to prioritize:

- execution topology
- global batching/brokering
- evaluation orchestration

before treating low CPU or low VRAM by themselves as optimization targets.

### 18. Clean async-core run confirms bottleneck class but narrows root cause

The latest clean run (`gomoku7_async_core_full`) with runtime tuner disabled
and eval/self-play isolation enabled confirms:

- no evaluation correctness collapse
- no replay starvation in this run shape
- substantial async runner activity on both self-play and eval

But it also confirms the dominant bottleneck class remains unchanged:

- result-side wait dominates queue wait by a large factor
- effective CPU thread-equivalent usage remains low
- null-result rate in async batch completion is still non-trivial

So the CRAG diagnosis is now stricter:

1. The problem is no longer "measurement contamination first."
2. The problem is now "async substrate still has blocking completion behavior
   and null-result churn."
3. Further gains should target pending-eval/apply separation and completion
   mechanics, not additional heuristic tuning.

## Sources

- AlphaGo Zero (Nature / PubMed): https://pubmed.ncbi.nlm.nih.gov/29052630/
- AlphaZero (Science / PubMed): https://pubmed.ncbi.nlm.nih.gov/30523106/
- AlphaZero paper text mirror: https://docslib.org/doc/2353397/mastering-chess-and-shogi-by-self-play-with-a-general-reinforcement-learning-algorithm
- WU-UCT (OpenReview): https://openreview.net/forum?id=BJlQtJSKDB
- WU-UCT PDF landing result: https://openreview.net/pdf/a348732b5ca8e27b8fe0b211c92ec4d1b6defbae.pdf
- KataGo analysis engine doc mirror: https://gitee.com/LZY2006/KataGo/blob/master/docs/Analysis_Engine.md
- Lc0 overview: https://draft.lczero.org/dev/lc2/overview/
- Lc0 batching note: https://lczero.org/dev/old/lc2/batching/
