# Feature Specification: High-Performance AlphaZero Engine

**Feature Branch**: `001-goal-create-spec`
**Created**: 2025-09-16
**Status**: Alpha Release (v1.0.0-alpha)
**Release Date**: 2025-10-01
**Input**: User description: Create spec.md  the authoritative specification for a high-performance, game-agnostic AlphaZero engine (Gomoku/Chess/Go) on a single machine (Ryzen 5900X, 64GB RAM, RTX 3060 Ti 8GB).

## Problem & Goals

Ship a production-ready AlphaZero-style reinforcement learning engine that achieves superhuman performance across multiple board games (Gomoku, Chess, Go) while operating efficiently on consumer hardware. The system must deliver intelligent search quality over raw throughput metrics, targeting realistic performance benchmarks that account for actual GPU utilization and system constraints.

Primary objectives:
- Achieve superhuman play in Gomoku within 48 hours of training
- Reach strong amateur level in Chess (960-capable) within 1 week of training
- Demonstrate competitive performance on Go (9�9 to 19�19 board sizes)
- Maintain 30-40k simulations per second including neural network inference time
- Sustain 80-92% GPU utilization with 32-64 average batch sizes
- Operate as game-agnostic engine requiring no code changes between games

## Non-Goals

- Full device-resident, wave-locked MCTS implementations as primary architecture
- Distributed multi-GPU training systems
- Support for games beyond Gomoku, Chess, and Go in initial release
- Integration with existing game engines or external APIs
- Real-time human interaction interfaces

## System Overview

The engine employs a hybrid CPU/GPU architecture where Python orchestrates the system while performance-critical operations execute in compiled languages. The core MCTS tree search runs on CPU using shared-tree parallelism with multiple threads, while neural network inference batches requests asynchronously on GPU. This separation allows continuous tree exploration without blocking on GPU operations, maximizing both CPU and GPU utilization.

Key architectural principles:
- Shared-tree MCTS with virtual loss for thread coordination
- Asynchronous, dynamically micro-batched neural network inference
- Structure-of-Arrays memory layout for cache efficiency
- Game-agnostic abstraction layer for universal compatibility
- Experience replay buffer with memory-mapped file storage

## Users & Scenarios

### Primary Users
- AI researchers training and evaluating reinforcement learning models
- Game engine developers integrating strong AI opponents
- Performance engineers optimizing MCTS implementations
- Competitive programming teams benchmarking algorithmic approaches

### Usage Scenarios
- **Self-Play Training**: Generate large volumes of high-quality training data through automated game playing
- **Model Evaluation**: Test neural network performance against baseline models and human players
- **Engine-Only Play**: Provide move recommendations for external game interfaces
- **Performance Profiling**: Analyze and optimize search algorithms and GPU utilization patterns

## Functional Requirements

### Core Game Engine Requirements
- **FR-001**: System MUST support Gomoku (15�15 board, 5-in-a-row victory condition)
- **FR-002**: System MUST support Chess including Chess 960 position generation
- **FR-003**: System MUST support Go on variable board sizes from 9�9 to 19�19
- **FR-004**: System MUST enforce legal move validation before policy normalization
- **FR-005**: System MUST detect terminal game states and provide terminal values
- **FR-006**: System MUST apply moves in-place without creating board state copies during search

### MCTS Search Requirements
- **FR-007**: System MUST implement shared-tree search with multiple threads accessing single tree
- **FR-008**: System MUST apply virtual loss during tree traversal to prevent duplicate selection
- **FR-009**: System MUST perform value backup with sign flipping at each ply level
- **FR-010**: System MUST use PUCT selection formula with configurable exploration parameter
- **FR-011**: System MUST maintain per-search evaluation cache to avoid redundant neural network calls
- **FR-012**: System MUST support optional transposition table for position reuse across searches

### Neural Network Requirements
- **FR-013**: System MUST implement ResNet architecture with Squeeze-Excitation blocks (20 blocks, 256 channels)
- **FR-014**: System MUST support mixed precision (fp16) inference with CPU fallback capability
- **FR-015**: System MUST batch inference requests dynamically (minimum 32 positions OR 3ms timeout)
- **FR-016**: System MUST use pinned memory buffers to optimize GPU data transfer
- **FR-017**: System MUST extract game-specific feature planes for neural network input (see `data-model.md` for detailed tensor specifications: Gomoku 36 planes, Chess 30 planes, Go 25 planes)

### Performance Requirements
- **FR-018**: System MUST achieve 30-40k simulations per second including neural network inference
- **FR-019**: System MUST sustain 80-92% GPU utilization during search operations
- **FR-020**: System MUST maintain 32-64 average inference batch size
- **FR-021**: System MUST limit tree memory usage to under 1GB for typical search depths
- **FR-022**: System MUST generate 200-300 self-play games per hour

### Training Pipeline Requirements
- **FR-023**: System MUST implement experience replay buffer using memory-mapped files
- **FR-024**: System MUST apply data augmentation through game symmetries (rotations, reflections)
- **FR-025**: System MUST support temperature-based move selection during self-play
- **FR-026**: System MUST add Dirichlet noise at root node for exploration during training
- **FR-027**: System MUST maintain training stability with monotonic loss curves and no NaN values

### Monitoring & Telemetry Requirements
- **FR-028**: System MUST track GPU utilization percentage and batch fill rates
- **FR-029**: System MUST monitor simulation rates (leaves processed per second)
- **FR-030**: System MUST measure selection versus backup timing distributions
- **FR-031**: System MUST log policy entropy and Q-value calibration metrics
- **FR-032**: System MUST detect memory leaks during extended operation (1-hour soak tests)

### Quality Requirements
- **FR-033**: System MUST include comprehensive unit tests for tree operations and backup correctness
- **FR-034**: System MUST verify thread safety with race condition detection
- **FR-035**: System MUST provide performance profiling for both CPU and GPU components
- **FR-036**: System MUST implement automated performance regression testing
- **FR-037**: System MUST maintain documentation of all hyperparameters and design decisions

### Hardware Optimization Requirements
- **FR-038**: System MUST exploit Ryzen 5900X dual-CCD architecture with thread affinity
- **FR-039**: System MUST optimize memory access patterns for 32MB L3 cache per chiplet
- **FR-040**: System MUST manage RTX 3060 Ti memory allocation within 8GB VRAM constraint
- **FR-041**: System MUST utilize AVX2 instructions for vectorized operations where applicable
- **FR-042**: System MUST align memory structures for SIMD operation efficiency

## Non-Functional Requirements (Acceptance Targets)

### Performance Targets
- Simulations per second (including NN): 30-40k sustained rate
- CPU utilization: 85-95% during search operations
- GPU utilization: 80-92% during inference operations
- Average batch size: 32-64 positions per inference call
- Tree memory footprint: <1GB for typical search configurations
- Self-play generation rate: 200-300 complete games per hour
- Memory stability: No leaks detectable in 1-hour continuous operation
- GIL bypass: No Python Global Interpreter Lock contention in performance-critical loops

### Training Stability Targets
- Loss convergence: Monotonic decrease in policy and value losses
- Numerical stability: Zero NaN values in gradients or model parameters
- Value calibration: Q-value predictions align with actual game outcomes within statistical bounds
- Policy sharpening: Entropy decreases appropriately as training progresses

## Constraints

### Hardware Constraints
- Single machine deployment (no distributed computation)
- AMD Ryzen 5900X: Unified Memory Architecture with dual CCD L3 cache topology
- NVIDIA RTX 3060 Ti: 8GB VRAM limit constraining batch sizes and model dimensions
- System RAM: 64GB available for tree storage and experience replay buffers

### Software Constraints
- PyTorch 2.x framework for neural network operations
- Python 3.12 orchestration layer with compiled extensions for performance
- Single-GPU inference (no multi-GPU parallelization)
- Linux operating system with CUDA support

### Architectural Constraints
- Shared-tree MCTS (not wave-based or device-resident approaches)
- CPU-based tree search with GPU-based inference separation
- Asynchronous communication pattern between search threads and inference worker

## Risks & Mitigations

### Atomic Operations Contention
**Risk**: Multiple threads updating shared tree nodes cause performance degradation
**Mitigation**: Use virtual loss with minimal atomic updates, benchmark contention levels, implement lock-free data structures where possible

### Inference Queue Back-Pressure
**Risk**: GPU inference cannot keep pace with CPU search thread requests
**Mitigation**: Implement dynamic batching with timeout fallback, monitor queue depths, add CPU fallback inference path

### Mixed Precision Numerical Stability
**Risk**: fp16 operations introduce training instability or gradient underflow
**Mitigation**: Use gradient scaling, monitor for NaN values, implement CPU fp32 fallback for problematic operations

### Illegal Move Masking Errors
**Risk**: Neural network selects invalid moves causing game state corruption
**Mitigation**: Strict pre-normalization masking, comprehensive unit tests for all game rule implementations, assertion checks in debug builds

## Resolved Design Decisions

**Note**: These questions have been resolved during implementation (see `tasks.md` lines 268-275 for resolution status).

### Game-Specific Configuration (RESOLVED)
- ✅ Feature planes: Gomoku 36 planes (enhanced with tactical analysis), Chess 30 planes (with move history), Go 25 planes (with proper history separation) - See `data-model.md`
- ✅ CPUCT exploration: Configurable via config system, game-specific defaults in training guide
- ✅ Dirichlet noise alpha: Game-specific values documented in `docs/training_guide.md`

### Performance Tuning Parameters (RESOLVED)
- ✅ Transposition table: Memory-mapped buffer design with LRU cache (T029)
- ✅ Virtual loss magnitude: Default 1.0, tuning script implemented (`scripts/tune_virtual_loss.py`)
- ✅ Batch timeout: Default ≤3ms, tuning script implemented (`scripts/tune_timeout.py`)

### Training Hyperparameters (RESOLVED)
- ✅ Learning rate schedules: Documented in training guide with game-specific recommendations
- ✅ Experience buffer: 1M capacity with rotation policy, balanced sampling across game types
- ✅ Model capacity: ResNet-256 (20 blocks) sufficient for target performance

## Related Specifications

### specs/002-cpp-simulation-runner

**Status**: Ready for implementation
**Purpose**: Address performance bottleneck (246 sims/sec → 35,000 sims/sec)

The C++ MCTS Simulation Runner specification resolves critical GIL contention in the Python orchestration layer. This enhancement is required to achieve the FR-018 performance target (30-40k simulations/second).

**Key Documents**:
- `specs/002-cpp-simulation-runner/spec.md` - Complete specification
- `specs/002-cpp-simulation-runner/plan.md` - Implementation roadmap
- `specs/002-cpp-simulation-runner/tasks.md` - Detailed task breakdown
- `specs/002-cpp-simulation-runner/research.md` - Investigation findings supporting the design

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (specific performance targets provided)
- [x] Scope is clearly bounded (three games, single machine, specific hardware)
- [x] Dependencies and assumptions identified (hardware constraints, software versions)

### Objective Verification Items
- [ ] Unit tests verify backup value sign flipping at each tree level
- [ ] Unit tests confirm illegal move masking prevents invalid selections
- [ ] Unit tests validate terminal state detection across all supported games
- [ ] Performance benchmarks achieve 30k+ simulations/second targets
- [ ] GPU utilization metrics consistently show 80-92% usage during search
- [ ] Memory profiling confirms <1GB tree footprint and no leaks over 1-hour operation
- [ ] Deterministic test mode produces identical results with fixed random seed
- [ ] Documentation coverage includes all hyperparameters and their tuning rationales

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed