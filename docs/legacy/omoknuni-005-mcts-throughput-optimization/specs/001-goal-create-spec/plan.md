# Implementation Plan: High-Performance AlphaZero Engine

**Branch**: `001-goal-create-spec` | **Date**: 2025-09-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-goal-create-spec/spec.md`

## Summary

Ship a production-ready AlphaZero-style reinforcement learning engine that achieves superhuman performance across multiple board games (Gomoku, Chess, Go) while operating efficiently on consumer hardware. The system uses hybrid CPU/GPU architecture with shared-tree MCTS on CPU and asynchronous micro-batched neural network inference on GPU, targeting 30-40k simulations/second with 80-92% GPU utilization.

## Technical Context

**Language/Version**: Python 3.12 orchestration, C++17 with pybind11, Cython with nogil blocks
**Primary Dependencies**: PyTorch 2.x, pybind11, Cython, scikit-build-core, OpenMP, CUDA 12.x
**Storage**: Memory-mapped files for experience replay, Parquet format, RAM cache
**Testing**: pytest for Python, Google Test for C++, thread safety verification tools
**Target Platform**: Linux with CUDA support, AMD Ryzen 5900X, NVIDIA RTX 3060 Ti 8GB
**Project Type**: single (high-performance compute library with Python orchestration)
**Performance Goals**: 30-40k simulations/sec including NN, 80-92% GPU utilization, 32-64 avg batch size
**Constraints**: <1GB tree memory, single machine, 8GB VRAM limit, 1-hour memory stability
**Scale/Scope**: 3 games (Gomoku/Chess/Go), 200-300 games/hour self-play, superhuman play targets

## Constitution Check

Since no specific constitution was provided, applying general software development principles:

**Library-First**: ✅ Core MCTS engine built as reusable library with C++ extension
**Testing-First**: ✅ Unit tests for tree operations, backup correctness, thread safety required
**Observability**: ✅ Comprehensive telemetry for GPU utilization, memory usage, performance metrics
**Simplicity**: ⚠️ Complex architecture justified by performance requirements and hardware constraints

## Project Structure

### Documentation (this feature)
```
specs/001-goal-create-spec/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── core/                # C++/Cython MCTS engine
├── games/               # Game implementations (Gomoku, Chess, Go)
├── neural/              # Neural network and inference
├── training/            # Self-play and training pipeline
├── telemetry/           # Monitoring and profiling
└── utils/               # Shared utilities

tests/
├── contract/            # API contract tests
├── integration/         # End-to-end game tests
├── unit/                # Unit tests for all components
└── performance/         # Benchmark and regression tests

cpp_extensions/          # C++/pybind11 extensions
├── mcts/                # Core MCTS tree operations
├── games/               # Game rule implementations
└── utils/               # Vectorized operations, memory management
```

**Structure Decision**: Option 1 (single project) - This is a high-performance compute library with Python orchestration layer

## Phase 0: Outline & Research

**Unknowns from Technical Context**: All technical requirements are specified in the comprehensive mcts_guide.md source material.

**Research Areas Completed**:
- **Architecture Decision**: Shared-tree MCTS with CPU-GPU hybrid approach chosen over wave-based alternatives
- **Rationale**: Traditional tree parallelism avoids "stale frontier problem" while maximizing GPU utilization through dynamic batching
- **Memory Layout**: Structure-of-Arrays (SoA) design for cache efficiency, 32-64 bytes per node
- **Threading Strategy**: 8-10 worker threads with virtual loss coordination, atomic operations for thread safety
- **GPU Optimization**: Dynamic micro-batching (≥32 positions OR ≤3ms timeout), mixed precision fp16, pinned memory buffers

**Key Technical Decisions**:
1. **MCTS Engine**: C++17 with pybind11 for GIL release during hot loops
2. **Concurrency**: Shared tree with virtual loss (+1.0 default), atomic updates
3. **GPU Inference**: Central worker with dynamic batching, CPU fallback capability
4. **Memory Management**: Pre-allocated node pools, SoA layout, indices instead of pointers
5. **Build System**: scikit-build-core with optimization flags `-O3 -march=znver3 -fopenmp`

**Output**: research.md with all architectural decisions documented

## Phase 1: Design & Contracts

**Data Model Extraction**:
- **MCTSNode**: visit_count, total_value, prior_prob, virtual_loss, parent_index, first_child_index, num_children, flags
- **GameState**: board representation, current_player, legal_moves_mask, terminal_status, terminal_value
- **InferenceBatch**: positions, policies, values, batch_metadata
- **ExperienceBuffer**: states, policies, outcomes, metadata

**API Contracts**:
- **MCTS Search API**: `search(state, simulations, cpuct=1.25, n_threads=8) -> np.ndarray[visits]`
- **Game Interface**: `apply_move_inplace()`, `get_legal_moves()`, `is_terminal()`, `extract_features()`
- **Inference Worker**: `warmup(input_shape)`, `inference_loop(in_queue, out_queues)`
- **Training Pipeline**: `generate_self_play()`, `train_model()`, `evaluate_model()`

**Contract Tests**: Failing tests for all API endpoints to be generated

**Quickstart Validation**: Build → Self-play → Training → Evaluation pipeline verification

**Agent Context Update**: Update Claude Code context with new technical stack and project structure

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each API contract → contract test task [P]
- Each data model entity → implementation task [P]
- Each game → game implementation task
- Integration tests for full pipeline
- Performance benchmarking and optimization tasks

**Ordering Strategy**:
- TDD order: Contract tests → Data models → Core MCTS → Games → Neural network → Training pipeline
- Dependency order: C++ extensions → Python bindings → Game abstractions → MCTS engine → Inference → Training
- Mark [P] for parallel execution (independent components)

**Performance Milestones**:
1. Basic MCTS: 10k nodes/sec single-threaded
2. Multi-threaded: 30k simulations/sec with CPU-only evaluation
3. GPU integration: 80%+ GPU utilization with micro-batching
4. Full pipeline: 200-300 games/hour self-play generation
5. Superhuman performance: Gomoku within 48 hours training

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md covering full implementation

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

**Critical Validation Steps**:
- Unit tests verify value sign flipping and illegal move masking
- Performance benchmarks achieve 30k+ simulations/second targets
- Memory profiling confirms <1GB footprint and no leaks over 1-hour operation
- GPU utilization consistently shows 80-92% during search operations
- Deterministic mode produces identical results with fixed random seed

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multi-language architecture | Performance requirements demand GIL bypass | Pure Python cannot achieve 30k+ sims/sec targets |
| Complex memory management | Cache efficiency critical for tree traversal | Standard allocators cause performance degradation |
| Asynchronous GPU batching | Hardware utilization targets require overlapping compute | Synchronous inference limits GPU utilization to <50% |

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS (complexity justified by performance requirements)
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented

---
*Based on Constitution template - See `/memory/constitution.md`*