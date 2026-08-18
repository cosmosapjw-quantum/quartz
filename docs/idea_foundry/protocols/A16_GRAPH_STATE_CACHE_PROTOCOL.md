# A16: Graph / State Sharing Cache Protocol & Semantic Decomposition

## 1. Executive Summary & Gating Invariant

- **Axis ID**: `A16` (`A16.graph_state_cache_v1`)
- **Mechanism Description**: Eval-cache sharing vs SearchNode tree-transposition sharing under DAG graph structures.
- **Previous Synthetic Result**: $+14.27$ calls avoided in synthetic toy graphs (`A16.toy_graph_cache_gate`).
- **Audit Diagnosis**: Production QUARTZ already incorporates a lockless transposition table (`src/mcts/tt.rs`) where transpositions share `MctsNode`. A16 proposed evaluation-cache sharing across disconnected tree paths without sharing parent edge statistics ($N, W, Q$). In the current Rust foundry codebase, `propose()` is a stub (`MetaAction::Noop`).
- **Production Gating Contract**: **MERGE IS BLOCKED** until the two cache layers are formally decomposed and live engine profiling confirms NN evaluation reduction without memory bloat or lock contention.

---

## 2. Architectural Decomposition: Two-Tier Cache Model

```
[MCTS Tree Search]
        |
        v
+-----------------------------------------------------------+
| Tier 1: SearchNode Transposition Table (src/mcts/tt.rs)   |
| - Shared MctsNode pointers across transposition edges     |
| - Shares visit counts N(s,a), values W(s,a), priors P(s,a)|
| - In-tree backpropagation updates all incoming paths      |
+-----------------------------------------------------------+
        |  (Miss in active tree)
        v
+-----------------------------------------------------------+
| Tier 2: Pure State/Evaluation Cache (A16 Proposal)        |
| - Global LRU / Lockless Key-Value Cache: Zobrist -> (P, V)|
| - Does NOT share tree statistics or parent edge visits    |
| - Bypasses neural network inference for known positions   |
+-----------------------------------------------------------+
        |  (Miss in Eval Cache)
        v
+-----------------------------------------------------------+
| Tier 3: Neural Network Evaluator (Batched CUDA TensorRT)  |
+-----------------------------------------------------------+
```

---

## 3. Experimental Design & Estimands

### Test Suite
1. **Transposition-Rich Domains**:
   - `chess` (Pawn structures, piece maneuvers, opening books)
   - `gomoku15` (Independent quadrant move transpositions)
2. **Cache Capacity & Replacement Policies**:
   - LRU vs Clock vs Direct-Mapped 2-way associative.
   - Memory budgets: $C_{\rm mem} \in \{64\,\text{MB}, 256\,\text{MB}, 1024\,\text{MB}\}$.

### Primary Estimands
1. **Evaluator Call Reduction ($\Delta E_{\rm eval}$)**:
   $$\Delta E_{\rm eval} = \frac{E_{\rm A16}}{E_{\rm baseline}} - 1 < -0.10 \quad (\ge 10\% \text{ reduction in NN forward passes})$$
2. **Search Speedup Factor ($\Delta S_{\rm nps}$)**:
   $$\Delta S_{\rm nps} = \frac{\text{Nodes/sec}_{\rm A16}}{\text{Nodes/sec}_{\rm baseline}} - 1 > +0.05 \quad (\ge 5\% \text{ search speedup})$$
3. **Memory Bounding & Contention Invariant**:
   - Zero lock contention spikes ($< 50\,\text{ns}$ average lookup overhead).
   - Bounded resident memory strictly within configured cache budget.

---

## 4. Promotion Criteria

A16 may only be promoted to production search if:
1. `src/mcts/tt.rs` and `src/mcts/foundry/` implement the Tier-2 pure state evaluation cache with zero lock contention.
2. Differential testing confirms $\ge 10\%$ evaluation reduction on standard benchmark suites.
3. Win rate against the frozen baseline does not degrade ($\text{Score Rate} \ge 0.500$).
