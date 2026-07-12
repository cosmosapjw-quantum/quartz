# Feature Specification: MCTS Throughput Optimization (Single Machine)

**Feature Branch**: `005-mcts-throughput-optimization`
**Created**: 2025-10-20
**Status**: Draft
**Input**: User description: "Maximize MCTS simulations/second on Ryzen 5900X + RTX 3060 Ti through elimination of state cloning bottleneck (86.6% of execution time), coordinator serialization, broken OpenMP, and excessive tensor copying. Maintain AlphaZero PUCT semantics for Gomoku, Chess, Go (9x9). Target: 7,000-9,000 sims/sec (58-75× current baseline of 120 sims/sec)."

## Problem Statement

The current MCTS implementation achieves only 120.4 simulations/second (mean, 560 trials), which is 66× slower than the 8,000 sims/sec target achievable on the target hardware (Ryzen 5900X 12-core + RTX 3060 Ti). Profiling has identified four primary bottlenecks:

1. **State Cloning** (86.6% of execution time): Each simulation clones the game state at ~418μs per clone due to 223 allocations, negating zero-copy tree traversal benefits
2. **Coordinator Serialization** (99.6% blocking time): Single coordinator thread becomes system-wide bottleneck during batch formation and tensor preparation
3. **Broken OpenMP** (0% success rate): Missing or incorrectly linked OpenMP causes serial feature extraction instead of parallel processing
4. **Tensor Copy Overhead** (37ms per batch): 4-6 memory copies per batch (C++ → Python → GPU) with non-pinned memory causing GPU stalls

These bottlenecks prevent the system from achieving the target throughput of 200-300 self-play games per hour, which is critical for training superhuman-level models within 48 hours.

## Business Goal

Maximize MCTS simulations per second on consumer hardware (single machine) while maintaining model iteration flexibility (Python PyTorch) and AlphaZero PUCT policy semantics. The system must be robust across multiple board games (Gomoku 15×15, Chess 8×8, Go 9×9) without modifying game logic or search algorithm correctness.

**Success Target**: Achieve 7,000-9,000 simulations/second (58-75× improvement) to enable 200-300 self-play games per hour and 48-hour training cycles for superhuman performance.

## Clarifications

### Session 2025-10-20

- Q: Go board size (9×9 vs 19×19) - spec mentions 9×9 but CLAUDE.md shows 19×19 in legacy context. Should we support both or pick one? → A: Support both dynamically (9×9 and 19×19 Go boards). Buffer sizing must accommodate maximum dimensions (19×19 = 361 action space).
- Q: Maximum batch size for pinned buffer pre-allocation? → A: max_batch=64 (optimal for RTX 3060 Ti, 3.3MB CPU + 3.3MB GPU pinned buffers).
- Q: Virtual-loss restart corner cases - safe to restart at any point during expansion? → A: Restart from root immediately on any virtual-loss failure. If node becomes expanded during retry, selection naturally picks different path (no special handling needed).
- Q: Should legacy state pool path be kept behind a flag for experiments? → A: Remove state pool code entirely. Profiling shows 56% regression due to copyFrom() overhead. No fallback path needed (violates Principle I: Zero-Copy First).
- Q: Phase 3A target coordinator count and CUDA stream count for RTX 3060 Ti? → A: 3 coordinators with 3 dedicated CUDA streams (optimal balance for mid-tier GPU, avoids excessive contention while saturating GPU).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Eliminate State Cloning Bottleneck (Priority: P1) 🎯 **MVP**

**Scenario**: As a self-play training pipeline, I need MCTS simulations to complete without cloning game state on every iteration so that simulation throughput increases from 120 sims/sec to 1,500-3,000 sims/sec.

**Why this priority**: State cloning is the PRIMARY bottleneck (86.6% of execution time). Eliminating it alone provides 10-25× performance gain and is the critical path to meeting the target. This is the minimum viable optimization that delivers measurable value.

**Independent Test**: Run benchmark with 800 simulations and verify profiling shows <1% time attributed to state copying. Measure throughput increase to 1,500-3,000 sims/sec range.

**Acceptance Scenarios**:

1. **Given** MCTS simulation reaches a leaf node, **When** feature extraction occurs, **Then** features are extracted in-place from the current game state without creating a state copy
2. **Given** features are extracted, **When** submitting to inference queue, **Then** features are moved (rvalue semantics) with zero additional copies
3. **Given** coordinator receives inference requests, **When** forming batches, **Then** coordinator aggregates pre-extracted features without performing feature extraction or state cloning
4. **Given** profiling campaign with 100+ trials is run, **When** analyzing results, **Then** state cloning contributes <1% of total execution time (down from 86.6%)
5. **Given** benchmark completes, **When** measuring throughput, **Then** system achieves 1,500-3,000 simulations/second

---

### User Story 2 - Fix Tensor Pipeline and OpenMP Parallelization (Priority: P2) ✅ **TARGET**

**Scenario**: As a training system, I need efficient tensor transfer and parallel feature extraction so that MCTS throughput reaches the 7,000-9,000 sims/sec target required for 48-hour training cycles.

**Why this priority**: This is the PRIMARY TARGET that unlocks production-ready performance (58-75× improvement). Fixes two critical bottlenecks: broken OpenMP (0% success rate) and excessive tensor copying (37ms per batch). Achieves business goal.

**Independent Test**: Run benchmark and verify (1) OpenMP thread count >1 in logs, (2) tensor preparation time ≤2ms per batch, (3) GPU utilization ≥80%, (4) throughput ≥7,000 sims/sec.

**Acceptance Scenarios**:

1. **Given** OpenMP build configuration, **When** compiling C++ extensions, **Then** OpenMP library is correctly linked (verified via `ldd` showing `libomp.so`)
2. **Given** feature extraction executes, **When** processing batch items, **Then** OpenMP parallelizes work across available cores (thread count >1 in execution logs)
3. **Given** coordinator prepares tensors, **When** batching features, **Then** pre-allocated pinned memory buffers are reused (zero allocation per batch)
4. **Given** tensors transfer to GPU, **When** initiating transfer, **Then** non-blocking transfers with CUDA streams minimize GIL holding time
5. **Given** profiling shows results, **When** analyzing batch latency, **Then** tensor creation completes in ≤2ms per batch (down from 37ms)
6. **Given** GPU profiling runs, **When** measuring utilization, **Then** GPU utilization reaches ≥80% during search (up from ~68%)
7. **Given** benchmark completes, **When** measuring throughput, **Then** system achieves 7,000-9,000 simulations/second

---

### User Story 3 - Parallel Coordinators for High Throughput (Priority: P3) 🎯 **STRETCH**

**Scenario**: As a high-throughput training system, I need multiple coordinator threads to eliminate serialization bottlenecks so that MCTS can exceed 12,000 sims/sec for research-grade performance.

**Why this priority**: STRETCH GOAL for research applications requiring >12,000 sims/sec. Only needed if Phase 2 doesn't meet target or if stretch goals (100-166× improvement) are desired. Adds architectural complexity.

**Independent Test**: Run benchmark with 3 coordinator threads and 3 CUDA streams for multi-stream GPU inference. Verify coordinator blocking <10% and throughput ≥12,000 sims/sec.

**Acceptance Scenarios**:

1. **Given** system initializes, **When** starting coordinators, **Then** 3 coordinator threads run concurrently with shared queue (lock-free MPMC)
2. **Given** multiple coordinators operate, **When** batching requests, **Then** each coordinator uses dedicated CUDA stream for GPU inference
3. **Given** inference completes, **When** results are ready, **Then** condition variables notify waiting threads immediately (no polling)
4. **Given** profiling runs, **When** analyzing coordinator metrics, **Then** coordinator blocking time is <10% of iteration time (down from 99.6%)
5. **Given** benchmark completes, **When** measuring throughput, **Then** system achieves 12,000-20,000 simulations/second

---

### User Story 4 - Multi-Process Architecture (Priority: P4) 🚀 **OPTIONAL**

**Scenario**: As an advanced research system, I need multi-process architecture to bypass GIL limitations when Python callback overhead exceeds 5ms per batch, enabling 20,000-35,000 sims/sec for extreme-scale training.

**Why this priority**: OPTIONAL - only if Phase 3A insufficient AND profiling proves GIL is >50% bottleneck AND target >25,000 sims/sec. Requires significant architectural changes (6+ weeks). High complexity, high risk.

**Independent Test**: Run multi-process benchmark with shared-memory tensor handoff. Verify Python callback overhead <5ms and throughput ≥20,000 sims/sec.

**Acceptance Scenarios**:

1. **Given** profiling shows GIL bottleneck >50%, **When** analyzing Phase 3A results, **Then** multi-process architecture is justified
2. **Given** multi-process architecture runs, **When** transferring tensors, **Then** shared memory handoff eliminates serialization through Python
3. **Given** benchmark completes, **When** measuring throughput, **Then** system achieves 20,000-35,000 simulations/second

---

### Edge Cases

- **What happens when feature extraction buffer overflows?** System must detect buffer overflow and either allocate additional capacity or queue request for next batch
- **What happens when GPU inference fails or times out?** System must retry with exponential backoff and fallback to CPU inference if GPU unavailable
- **What happens when coordinator queue fills to capacity?** Backpressure mechanism must signal simulation threads to pause submission until queue drains
- **What happens when OpenMP is unavailable at runtime?** System must gracefully fall back to serial feature extraction with warning log
- **What happens during thread contention on tree nodes?** On virtual-loss failure (node busy), thread immediately restarts selection from root without sleep. If node becomes expanded during retry, selection naturally picks different child via PUCT. Busy-edge masking (PUCT = -∞) prevents re-selection of expanding nodes.
- **What happens when game state becomes invalid?** Validation layer must detect illegal states and abort simulation with detailed error reporting
- **What happens when profiling campaign detects regression?** Automated rollback procedure must revert changes and notify maintainers with profiling comparison

## Requirements *(mandatory)*

### Functional Requirements

#### Phase 1: State Cloning Elimination (MVP)

- **FR-001**: System MUST extract features in-place from game state at leaf node during selection phase (no state cloning)
- **FR-002**: System MUST submit features to inference queue using move semantics (rvalue references) to avoid copies
- **FR-003**: Inference queue MUST accept moved feature vectors plus metadata (action space size, plane count, board dimensions, node index, path)
- **FR-004**: Coordinator MUST aggregate pre-extracted features without performing feature extraction or state cloning
- **FR-005**: System MUST support optional on-demand state reconstruction (store path only; reconstruct via make/unmake at expansion) as validation/debugging fallback only (NOTE: distinct from deprecated state pool which is removed entirely)
- **FR-006**: Code review MUST verify zero state cloning in simulation hot path (automated via grep for `clone()`, `copy()`, `new State()`)

#### Phase 2: Tensor Pipeline & OpenMP (TARGET)

- **FR-007**: Build system MUST link OpenMP library correctly (verified via `ldd` output showing `libomp.so`)
- **FR-008**: Feature extraction MUST parallelize work across available cores using OpenMP (thread count >1 in execution logs)
- **FR-009**: Coordinator MUST pre-allocate pinned memory tensor buffers at initialization with max_batch=64 capacity (64 × 36 × 19 × 19 = 3.3MB CPU + 3.3MB GPU, reused across batches, zero allocation per iteration)
- **FR-010**: GPU tensor transfers MUST use non-blocking mode with CUDA streams to minimize GIL holding time
- **FR-011**: Tensor creation pipeline MUST complete in ≤2ms per batch (down from 37ms baseline)
- **FR-012**: System MUST minimize Python-C++ boundary crossings (simulation loop executes entirely in C++ except inference callbacks)

#### Phase 3A: Parallel Coordinators (STRETCH)

- **FR-013**: System MUST support 3 coordinator threads operating concurrently with shared lock-free MPMC queue (4096 entries)
- **FR-014**: Each coordinator MUST use dedicated CUDA stream for multi-stream GPU inference
- **FR-015**: Result notification MUST use condition variables instead of polling loops
- **FR-016**: Coordinator blocking time MUST be <10% of iteration time (down from 99.6%)

#### Phase 3B: Multi-Process Architecture (OPTIONAL)

- **FR-017**: Multi-process implementation MUST be justified by profiling showing Python callback >5ms/batch AND GIL >50% bottleneck
- **FR-018**: Multi-process architecture MUST use shared-memory tensor handoff to eliminate serialization overhead
- **FR-019**: System MUST remain single-process unless Phase 3A fails to meet target AND profiling proves GIL-bound

#### Cross-Cutting Requirements

- **FR-020**: System MUST maintain AlphaZero PUCT policy semantics (no changes to search algorithm correctness)
- **FR-021**: System MUST support Gomoku (15×15), Chess (8×8), and Go (9×9 or 19×19, detected at runtime) without modifying game logic
- **FR-022**: System MUST use Python PyTorch for neural network inference (C++ LibTorch is prohibited per constitution)
- **FR-023**: System MUST operate on single machine with 12 cores/24 threads, 64GB RAM, RTX 3060 Ti (no distributed orchestration)
- **FR-024**: Code changes MUST focus on `continuous_simulation_runner.cpp/hpp` and coordinator/queue components (legacy files are reference-only)
- **FR-025**: Every optimization phase MUST include profiling campaign (100+ trials, <5% variance) with automated rollback if targets missed

### Key Entities

- **InferenceRequest**: Represents a single neural network inference request containing pre-extracted feature tensor, metadata (action space size, plane count, board dimensions), node reference, and path history
- **InferenceResult**: Represents neural network output containing policy vector (action probabilities) and value scalar (position evaluation)
- **SimulationState**: Represents complete MCTS simulation state including game state (board position, move history), tree state (node graph, visit counts, Q-values), and search metadata
- **ProfilingMetrics**: Represents performance measurement data including execution timings (state cloning, feature extraction, tensor creation, inference), throughput (sims/sec), bottleneck analysis (percentage contributions), and GPU utilization
- **CoordinatorQueue**: Represents batching and synchronization structure containing bounded queue (4096 entries), condition variables (result readiness), and batch formation logic
- **FeatureBuffer**: Represents pre-allocated memory region for game state features with fixed capacity (max: 36 planes × 19×19 board = 12,996 floats for Go 19×19 worst case), pinned memory allocation, and reuse tracking

## Success Criteria *(mandatory)*

### Measurable Outcomes

#### Phase 1 Acceptance (MVP)

- **SC-001**: System achieves 1,500-3,000 simulations/second in benchmark (10-25× improvement from 120 baseline)
- **SC-002**: Profiling shows state cloning contributes <1% of total execution time (down from 86.6%)
- **SC-003**: Code review confirms zero `clone()`, `copy()`, or `new State()` calls in simulation hot path
- **SC-004**: Memory allocations in hot path reduced to zero (measured via allocation profiler)

#### Phase 2 Acceptance (TARGET) ✅

- **SC-005**: System achieves 7,000-9,000 simulations/second in benchmark (58-75× improvement, PRIMARY GOAL)
- **SC-006**: OpenMP thread count >1 in execution logs (confirms successful OpenMP parallelization)
- **SC-007**: Tensor creation completes in ≤2ms per batch (down from 37ms, 18× improvement)
- **SC-008**: GPU utilization reaches ≥80% during search operations (up from ~68%)
- **SC-009**: Feature buffer allocations reduced to zero per iteration (pre-allocated pools only)

#### Phase 3A Acceptance (STRETCH)

- **SC-010**: System achieves 12,000-20,000 simulations/second in benchmark (100-166× improvement)
- **SC-011**: Coordinator blocking time reduced to <10% of iteration time (down from 99.6%)
- **SC-012**: Multi-stream GPU inference shows near-linear scaling (3 coordinators → 3.2-3.6× throughput vs 1 coordinator)

#### Phase 3B Acceptance (OPTIONAL)

- **SC-013**: System achieves 20,000-35,000 simulations/second in benchmark (166-291× improvement)
- **SC-014**: Python callback overhead reduced to <5ms per batch
- **SC-015**: GIL-related bottlenecks eliminated (measured via `py-spy` profiling)

#### Cross-Cutting Acceptance

- **SC-016**: Self-play pipeline generates 200-300 games per hour at target throughput
- **SC-017**: Training cycles complete within 48 hours for superhuman-level performance
- **SC-018**: All game configurations (Gomoku 15×15, Chess 8×8, Go 9×9, Go 19×19) achieve target throughput with <10% variance
- **SC-019**: Regression detection system flags performance drops >5% automatically
- **SC-020**: Rollback procedure documented and tested (restores previous performance in <1 hour)
- **SC-021**: End-to-end batch latency remains consistent across all phases (<5% variance)
- **SC-022**: Search algorithm produces identical results to baseline (PUCT semantics preserved)

### Assumptions

- Target hardware available: Ryzen 5900X (12c/24t), 64GB RAM, RTX 3060 Ti with 8GB VRAM
- Python 3.10-3.12 (tested) with PyTorch 2.0+ supports pinned memory and non-blocking GPU transfers
- OpenMP library available for linking during build process
- Profiling infrastructure already exists and provides accurate timing data
- Game implementations support make/unmake move semantics for in-place state manipulation
- Neural network model size fits within GPU memory constraints (8GB VRAM)
- Training pipeline can consume 200-300 games/hour output rate
- Constitution principles (zero-copy, coordinator efficiency, threading saturation) are enforceable via code review

### Out of Scope (Non-Goals)

- C++ LibTorch adoption (Python PyTorch only per constitution)
- Changes to game logic or PUCT policy semantics
- Distributed/multi-machine orchestration
- Support for WU-UCT, RAVE, or progressive widening (legacy/deprecated)
- Modifications to legacy files (`mcts_guide.md`, `simulation_runner.cpp/hpp`)
- State pool implementation (removed entirely - profiling shows 56% regression)
- Training algorithm changes or hyperparameter tuning
- Model architecture modifications
- Support for games beyond Gomoku, Chess, Go (9×9/19×19)

### Dependencies & Constraints

**Hardware Dependencies**:
- Single machine: Ryzen 5900X (12 cores / 24 threads, dual-CCD architecture)
- Memory: 64GB RAM minimum
- GPU: NVIDIA RTX 3060 Ti (8GB VRAM, Ampere architecture) for GPU inference only

**Software Dependencies**:
- Python 3.10-3.12 (tested) with PyTorch 2.0+ (GPU-enabled)
- OpenMP library for parallel feature extraction
- CUDA 11.8+ runtime for GPU operations
- pybind11 for Python-C++ bindings
- Profiling tools: custom C++ instrumentation, `py-spy`, `nsys`

**Architectural Constraints**:
- Focus on `continuous_simulation_runner.cpp/hpp` (current implementation)
- Legacy files (`mcts_guide.md`, `simulation_runner.cpp/hpp`) are reference-only
- Single-process architecture unless Phase 3B justified by profiling
- Python PyTorch only (no C++ LibTorch per constitution)

**Performance Constraints**:
- Tree memory budget: <1GB for 10M nodes (already achieved: 270MB)
- Node footprint: <64 bytes per node (already achieved: 27 bytes)
- Thread efficiency: ≥70% at 8 threads (currently 45% at 4 threads)
- Lock contention: <1% of execution time (use thread-local arenas)

**Constitution Compliance**:
- Principle I (Zero-Copy First): NO state cloning in hot paths
- Principle II (Coordinator Efficiency): Condition variables, pre-allocated buffers
- Principle III (Python-C++ Boundary): Minimal crossings, pinned memory
- Principle IV (Threading Saturation): 8-12 threads, OpenMP verified
- Principle V (Legacy Code Discipline): Focus on current implementation only
- Principle VI (Evidence-Based Gates): 100+ trial profiling for every phase

## Validation & Rollback

### Profiling Requirements

Each phase MUST include:
- 100+ trial profiling campaign with <5% variance
- 100% execution time capture (no "unknown" categories >1%)
- Dominant metric analysis (identify >50% time contributors)
- Comparison to baseline and phase targets
- Results committed to `docs/performance/phase_X_results.md`

### Regression Detection

Automated system MUST:
- Run benchmark after each optimization phase
- Compare throughput to phase targets (SC-001 to SC-015)
- Flag regressions >5% from target
- Trigger rollback procedure if phase target missed
- Notify maintainers with profiling comparison report

### Rollback Procedure

If phase target missed:
1. Git revert to previous stable commit
2. Run verification benchmark (confirm baseline restored)
3. Document findings in `docs/performance/rollback_report_YYYY-MM-DD.md`
4. Investigate root cause before retry
5. Update implementation plan based on findings

## Documentation & Artifacts

### Specification Documents

- `specs/005-mcts-throughput-optimization/spec.md` (this file)
- `specs/005-mcts-throughput-optimization/plan.md` (generated by `/speckit.plan`)
- `specs/005-mcts-throughput-optimization/tasks.md` (generated by `/speckit.tasks`)
- Links to optimization master plan: `MCTS_OPTIMIZATION_MASTER_PLAN.md`
- Links to profiling analysis: `COMPREHENSIVE_PROFILING_ANALYSIS_20251018.md`
- Links to architecture tradeoffs: `ARCHITECTURE_TRADEOFFS.md`
- Links to documentation index: `OPTIMIZATION_DOCUMENTATION_INDEX.md`

### Profiling & Benchmarks

- Profiling harness: `scripts/profiling/run_campaign.py`
- Analysis tools: `scripts/profiling/analyze_campaign.py`
- Benchmark scripts: `scripts/benchmark_phase*.py`
- Telemetry dashboards: `scripts/profiling/generate_dashboard.py`
- Historical data: `docs/performance/profiling_suite_*` directories

### Code Review Artifacts

- Constitution compliance checklist (6 principles)
- State cloning audit script (`scripts/audit_state_cloning.sh`)
- OpenMP verification script (`scripts/verify_openmp.sh`)
- Thread safety analysis (TSan reports)
- Memory leak detection (1-hour soak test results)
