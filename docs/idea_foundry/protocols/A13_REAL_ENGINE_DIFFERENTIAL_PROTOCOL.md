# A13: WU-UCT / Pending-Flow Real-Engine Differential Ablation Protocol

## 1. Executive Summary & Gating Invariant

- **Axis ID**: `A13` (`A13.real_engine_pending_flow_v1`)
- **Mechanism Description**: WU-UCT (Waiting-Update UCT) / Pending-Flow tracker modifying tree policy selection under asynchronous parallel MCTS batch evaluation by tracking in-flight visits across batch dispatch cycles.
- **Previous Synthetic Result**: $+38.07\%$ duplicate dispatch reduction in synthetic toy trees (`A13.toy_wu_uct_gate`).
- **Audit Diagnosis**: Toy tree setup has artificially synchronized search depths where standard UCT suffers exaggerated duplicate selection. In real parallel MCTS with lockless transposition tables (`src/mcts/tt.rs`) and dynamic batching, past experiments demonstrated that naive virtual loss / pending-flow adjustments can induce search trajectory distortion or thread contention without improving net throughput.
- **Production Gating Contract**: **MERGE IS BLOCKED** until the differential protocol defined below proves a statistically significant throughput gain ($\Delta Q_{\rm throughput} > 0$) at matched decision quality ($\Delta Q_{\rm score} \ge 0$) on real board game engines.

---

## 2. Experimental Design

### Evaluation Matrix
1. **Games**:
   - `gomoku15` (Standard Gomoku, high branching $b \approx 200$)
   - `chess` (Standard Chess, tactical complexity $b \approx 35$)
   - `go9` (9x9 Go, deep positional convergence)
2. **Concurrency Regimes**:
   - 1 thread (Sequential baseline / sanity check)
   - 4 threads (Low concurrency)
   - 8 threads (Medium concurrency)
   - 16 threads (High GPU saturation concurrency)
3. **Batching Modes**:
   - Batch sizes: $B \in \{4, 8, 16, 32\}$
   - Timeout: $\tau_{\rm timeout} \in \{0\,\mu\text{s}, 200\,\mu\text{s}, 1000\,\mu\text{s}\}$

### Compared Arms
- **Arm 0 (Incumbent Baseline)**: Lockless Transposition Table with Fixed Virtual Loss ($VL = 1.0$).
- **Arm 1 (A13 WU-UCT Pending-Flow)**: Active in-flight pending visit tracker adjusting prior / Q-value penalty proportional to in-flight queue depth.
- **Arm 2 (Adaptive VL Reference)**: Depth-attenuated virtual loss ($VL(d) = VL_0 \cdot \gamma^d$).

---

## 3. Estimands & Success Criteria

1. **Duplicate Dispatch Rate ($\Delta D$)**:
   $$\Delta D = D_{\rm A13} - D_{\rm baseline} < -0.15 \quad (\text{at least } 15\% \text{ reduction in redundant node evaluations})$$
2. **Net Search Throughput ($\Delta Q_{\rm throughput}$)**:
   $$\Delta Q_{\rm throughput} = \frac{\text{Visits/sec}_{\rm A13}}{\text{Visits/sec}_{\rm baseline}} - 1 > +0.05 \quad (\ge 5\% \text{ realized throughput improvement})$$
3. **Decision Quality Invariance ($\Delta Q_{\rm score}$)**:
   $$\text{Score Rate}(\text{A13 vs Baseline}) \ge 0.500 \quad (95\% \text{ CI lower bound } > 0.485)$$

---

## 4. Implementation & Audit Requirements

1. **Rust Engine Adapter**:
   - Must implement `propose()` and `apply()` in `src/mcts/foundry/policy.rs` directly interacting with `SearchContext` / `TranspositionTable`.
   - `MetaAction::Noop` is prohibited in real-engine execution.
2. **Thread Safety & Contention Profiling**:
   - Must measure `TtContentionSnapshot` (read/write lock wait nanoseconds) to ensure pending-flow tracking does not introduce synchronization bottlenecks.
