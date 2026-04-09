# Async Concurrent MCTS Research Notes (2026-04)

## Goal

Use external references as a CRAG baseline for the current QUARTZ symptoms:

- low effective CPU worker usage
- high wait/IO time relative to codec time
- batch fill is already reasonably high
- evaluation and self-play still spend most wall-clock outside learner compute

The question is not "how to tune one more timeout", but "what execution topology do mature concurrent NN-guided MCTS systems use when synchronous search workers stop scaling?"

## Executive Summary

The outside references are surprisingly consistent.

1. The usual next step after "one worker calls `evaluate()` and blocks" is not bigger locks or more threads.
   The next step is a staged or streamed pipeline:
   - gather/select workers
   - evaluation workers / broker
   - backprop/update workers

2. Successful systems separate:
   - number of positions/games searched concurrently
   - number of search threads per position
   rather than multiplying them blindly.

3. They treat in-flight NN queries as first-class state.
   Common patterns:
   - virtual loss or related "in-flight" accounting
   - explicit collision handling
   - event queues / streaming requests
   - async result delivery out of order

4. Cross-position batching is the main scaling lever on GPU systems.
   If batch fill is already decent but wall-clock is still bad, the remaining problem is usually worker blocking / orchestration, not codec or raw NN throughput.

5. The literature and engine docs do not recommend "make the whole tree lock-free first" as the first remedy.
   They recommend reducing blocking boundaries and moving to event-queue / brokered execution first.

## Source Findings

### 1. KataGo analysis engine: async protocol + explicit split between positions and per-position search threads

Source:
- https://gitee.com/mamh-mixed/KataGo/blob/master/docs/Analysis_Engine.md

Relevant findings:
- KataGo's analysis engine is designed specifically to analyze many positions in parallel and benefit from cross-position batching.
- It explicitly distinguishes:
  - `numSearchThreadsPerAnalysisThread`
  - `numAnalysisThreads`
- Its stdin/stdout protocol is explicitly asynchronous: new requests can arrive at any time, and results can come back later and out of order.

Implication for QUARTZ:
- QUARTZ should stop treating `parallel * n_threads` as one fused knob internally.
- Evaluation and self-play should be routed through an async query/result protocol keyed by job/session ID.
- The system should optimize "how many games/positions are active concurrently" separately from "how many search threads one game gets".

Why this matters for current QUARTZ symptoms:
- QUARTZ already fills batches well enough.
- The remaining symptom is blocked workers.
- KataGo's design says the right unit of concurrency is "positions in flight", not just more tree threads.

### 2. WU-UCT: incomplete queries must be represented explicitly

Source:
- https://openreview.net/forum?id=BJlQtJSKDB

Relevant findings:
- Parallelizing MCTS is hard because each rollout depends on statistics updated by earlier ones.
- WU-UCT's core idea is to track unfinished simulations ("unobserved samples") and adjust selection using those counts.
- The claimed benefit is near-linear speedup with limited quality loss.

Implication for QUARTZ:
- If QUARTZ moves to a genuinely async broker/state-machine design, it should also make "requests in flight" explicit in search state.
- This can be done without changing high-level search semantics:
  - add in-flight/pending counters
  - treat pending leaves as a distinct state
  - keep result application separate from request issuance

Why this matters:
- Right now QUARTZ workers appear to block waiting for eval.
- In a real async design, workers should not simply sleep; they should leave behind a pending marker and continue gathering other work where valid.

### 3. Lc0 classic docs: gathering larger batches is hard; local hacks often fail

Source:
- https://lczero.org/dev/old/lc2/batching/
- https://draft.lczero.org/dev/lc0/search/alphazero/

Relevant findings:
- Lc0 explicitly documents that improving batch gathering was the main scaling bottleneck.
- They note that one-by-one visit gathering and collisions limit batch size and scaling.
- They also document that virtual loss helps, but still leaves collisions and batching inefficiencies.
- Their notes explicitly say many "hacky" fixes failed.

Implication for QUARTZ:
- QUARTZ should not keep piling on timeout and local batching heuristics.
- If worker blocking persists after reasonable batch fill is achieved, bigger architectural change is justified.
- "Make batch timeout adaptive" was worth trying, but the docs support moving beyond that now.

Why this matters:
- QUARTZ already reached reasonably large effective batches.
- Yet wall-clock is still dominated by waiting.
- This is exactly the point where Lc0's own docs say more local hacks stop paying off.

### 4. Lc0 `lc3`: event queues and streaming instead of batch-centric classic search

Source:
- https://lczero.org/dev/lc0/search/lc3/overview/

Relevant findings:
- `lc3` explicitly moves away from the classic batch-based search model to a streaming worker model.
- It uses workers connected by event queues.
- The core workers are:
  - GatherWorker
  - EvalWorker
  - BackpropWorker
- It stores "N in flight" on edges and avoids needing to lock more than one node at a time.
- It treats nodes to compute as a stream, not as a synchronous visit that blocks the whole worker.

Implication for QUARTZ:
- This is the closest documented match to the direction QUARTZ now needs.
- A real fix likely needs:
  - gather/select stage
  - eval broker stage
  - backprop/apply stage
  - per-job/session event IDs
- QUARTZ should likely transition toward a streamed node-event pipeline, not just "one collector thread per evaluator".

Why this matters:
- QUARTZ's current symptom is that workers are mostly not computing.
- `lc3` attacks exactly that symptom by restructuring search as work queues and partial updates.

### 5. Lc0 engine overview: GPU backends are designed around batching many positions

Source:
- https://lczero.org/dev/overview/

Relevant findings:
- Lc0's NN backend batches multiple positions together for GPU efficiency.
- The engine architecture explicitly separates search algorithms from neural network backends.

Implication for QUARTZ:
- QUARTZ should do the same architecturally:
  - search side produces eval work items
  - broker/backends consume eval work items
  - the search loop should not own the batching boundary directly

### 6. MCTS survey literature: no one-size-fits-all, but parallelization is a distinct design axis

Source:
- https://link.springer.com/article/10.1007/s10462-022-10228-y

Relevant findings:
- The survey emphasizes that MCTS modifications cluster by search structure, action reduction, simulation changes, parallelization, and ML coupling.
- There is no universally best extension, but parallelization is its own first-class design problem, not an afterthought.

Implication for QUARTZ:
- It is reasonable to treat execution topology as a separate ablation axis from QUARTZ search semantics.
- This supports keeping:
  - `search_profile=quartz`
  - `search_profile=baseline`
  on the same async substrate.

### 7. AlphaGo Zero / AlphaZero: policy+value search loop is tightly integrated

Source:
- https://www.nature.com/articles/nature24270
- https://pubmed.ncbi.nlm.nih.gov/29052630/

Relevant findings:
- The AlphaGo Zero structure is a tight loop of:
  - tree visit
  - leaf evaluation
  - backprop
- The papers are not implementation guides for async broker design, but they reinforce that search quality depends on handling the evaluation boundary carefully.

Implication for QUARTZ:
- Systems refactors should preserve the leaf-eval-backprop semantics.
- The safe architectural move is to change execution ownership, not search mathematics.

## What This Means For QUARTZ

### Problems the research does NOT support as the primary next fix

1. More timeout heuristics
2. More codec optimization
3. Immediate full-tree lock-free rewrite
4. Simply raising `n_threads`
5. Treating CPU utilization itself as the objective

### Problems the research DOES support as the next fix

1. Replace synchronous `evaluate()` blocking with staged async execution
2. Track in-flight / pending eval work explicitly
3. Separate "games in flight" from "threads per game"
4. Introduce a broker that owns queueing and flush policy
5. Prefer streamed/event-queue orchestration over collector-lifetime locks and ad hoc local batching

## Recommended Architecture Direction

### Phase A: async substrate, not lock-free tree

First change:
- async node/session event pipeline
- broker-owned eval queue
- explicit pending eval state

Do NOT start with:
- lock-free whole-tree data structure
- lock-free memory reclamation overhaul

Reason:
- The outside references consistently attack the blocking boundary first.

### Phase B: coordinator split

Introduce distinct roles:
- gather/select workers
- eval broker / eval workers
- backprop/apply workers

This matches Lc0 lc3 most closely and fits QUARTZ's current symptoms.

### Phase C: same substrate for baseline and quartz

The system substrate should be shared by:
- baseline_shared_substrate
- quartz

Only controller/search policy differences should remain above it.

### Phase D: only then revisit tree-level lock-free ideas

Once the async substrate is working, profile again.
If tree contention remains dominant, then consider:
- edge-local in-flight counts
- lighter per-node locking
- sharded repositories
- eventually lock-free node repository patterns

## Practical Takeaways For The Next Implementation Plan

1. The correct next step is a true async broker/state-machine transition.
2. The minimal useful unit is not "more threads", but "more independent search work items in flight".
3. Evaluation should become a streamed service, not a synchronous function call from each worker.
4. Search should be able to leave pending work behind and continue elsewhere.
5. QUARTZ should move closer to:
   - KataGo's async multi-query engine model
   - WU-UCT's explicit in-flight accounting
   - Lc0 lc3's gather/eval/backprop worker decomposition

## Field Validation Against Current QUARTZ Run (2026-04-09)

Reference run:

- `artifacts/runtime_monitor/gomoku7_async_core_full/summary.json`
- runtime tuner off
- eval/self-play isolation on

What matched the research predictions:

1. Batch formation itself is not the first bottleneck.
   - weighted batch is high enough (`~15.6` for target `18`)
2. Codec/serialization is not the first bottleneck.
   - `codec_time_s` remains tiny compared to wait/IO
3. Event-driven staging helps correctness and control.
   - eval completes correctly (`errors=0`, `voids=0`)
   - runner progress signals are now visible

What remains unresolved:

1. Result-side wait still dominates queue wait by a large margin.
2. Effective CPU thread-equivalent usage is still low.
3. Async completion path still shows null-result churn.

Interpretation:

- The research direction is still correct.
- But current implementation is still in an intermediate async stage.
- The next required transition is from "async request submission" to
  "non-blocking pending completion/apply with reduced null-result churn."

Practical update to priorities:

1. Keep runtime tuner off for benchmark correctness.
2. Prioritize completion-path fixes over new heuristic knobs.
3. Continue strict baseline comparison on the same substrate.
4. Add direct instrumentation for pending-eval lifecycle and null-result causes.

## Sources

- KataGo analysis engine:
  - https://gitee.com/mamh-mixed/KataGo/blob/master/docs/Analysis_Engine.md
- WU-UCT:
  - https://openreview.net/forum?id=BJlQtJSKDB
- Lc0 overview:
  - https://lczero.org/dev/overview/
- Lc0 AlphaZero primer:
  - https://draft.lczero.org/dev/lc0/search/alphazero/
- Lc0 gathering larger batches:
  - https://lczero.org/dev/old/lc2/batching/
- Lc0 lc3 overview:
  - https://lczero.org/dev/lc0/search/lc3/overview/
- MCTS review:
  - https://link.springer.com/article/10.1007/s10462-022-10228-y
- AlphaGo Zero:
  - https://www.nature.com/articles/nature24270
  - https://pubmed.ncbi.nlm.nih.gov/29052630/
