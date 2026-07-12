# Research: High-Performance AlphaZero Engine Architecture

**Context**: AlphaZero-style reinforcement learning engine for board games (Gomoku, Chess, Go) targeting consumer hardware (Ryzen 5900X, RTX 3060 Ti 8GB)

## Architecture Decisions

### Decision: Shared-Tree MCTS with CPU-GPU Hybrid Architecture
**Rationale**:
- Traditional shared-tree MCTS avoids the "stale frontier problem" where hundreds of trees explore independently without learning from each other
- CPU excels at branchy tree traversal operations while GPU optimizes dense neural network inference
- Asynchronous pattern allows continuous CPU exploration while GPU processes batches, maximizing utilization of both resources

**Alternatives Considered**:
- **Wave-based MCTS** (MCTX-style): Rejected due to lockstep exploration limiting tactical depth and requiring complex device-resident state management
- **Fully CPU-based**: Cannot achieve target GPU utilization or neural network throughput
- **Fully GPU-based**: GPU memory constraints limit tree size and branching factor handling

### Decision: Structure-of-Arrays (SoA) Memory Layout
**Rationale**:
- Cache-efficient access patterns for tree traversal (visiting related node attributes together)
- SIMD-friendly memory alignment enables vectorized operations on node arrays
- Reduces memory footprint from ~200-300 bytes per node (AoS with pointers) to 32-64 bytes (SoA with indices)

**Alternatives Considered**:
- **Array-of-Structures (AoS)**: Standard object-oriented approach rejected due to pointer chasing and cache misses
- **Hybrid layouts**: Complexity not justified by marginal performance gains

### Decision: Dynamic Micro-Batching (≥32 positions OR ≤3ms timeout)
**Rationale**:
- Balances GPU utilization (prefers larger batches) with search latency (timeout prevents starvation)
- 32-position minimum ensures efficient GPU occupancy on RTX 3060 Ti
- 3ms timeout prevents CPU threads from waiting indefinitely during low-demand periods

**Alternatives Considered**:
- **Fixed batch sizes**: Creates artificial bottlenecks during variable search intensity
- **Pure timeout-based**: Can result in inefficient small batches under high load
- **Adaptive batch sizes**: Added complexity not justified by performance gains

### Decision: C++17 with pybind11 for Performance-Critical Paths
**Rationale**:
- Enables GIL release during tree operations, eliminating Python threading bottlenecks
- Native atomic operations for thread-safe tree updates
- Direct memory management and SIMD instruction usage
- Maintains Python interface for orchestration and flexibility

**Alternatives Considered**:
- **Pure Python with NumPy**: Cannot achieve GIL release for tree operations, limited to ~5-10k sims/sec
- **Cython with nogil**: Viable alternative but pybind11 provides better C++ ecosystem integration
- **JAX/XLA**: Requires complete algorithm restructuring for device-resident execution

### Decision: Virtual Loss Coordination (+1.0 default)
**Rationale**:
- Prevents multiple threads from selecting identical paths simultaneously
- Lightweight coordination mechanism using atomic operations
- 1.0 magnitude balances exploration prevention with Q-value disruption

**Alternatives Considered**:
- **Mutex-based locking**: Too coarse-grained, creates bottlenecks
- **Lock-free algorithms**: Complex implementation with marginal benefits
- **Higher virtual loss values**: Distorts Q-values excessively

## Performance Optimizations

### Decision: Hardware-Specific Compilation (-march=znver3)
**Rationale**:
- Enables AVX2 instructions and Zen 3 optimizations
- Optimizes for dual-CCD L3 cache topology of Ryzen 5900X
- Compiler can optimize for specific instruction latencies and throughput

**Implementation Notes**:
- Build flags: `-O3 -march=znver3 -fopenmp`
- Thread affinity considerations for CCD placement
- Memory alignment for SIMD operations

### Decision: Mixed Precision fp16 Inference with Scaling
**Rationale**:
- Doubles effective memory bandwidth on RTX 3060 Ti
- Enables larger batch sizes within 8GB VRAM constraint
- Gradient scaling prevents underflow during training

**Risk Mitigations**:
- CPU fp32 fallback for numerical instability
- Gradient scaling and NaN monitoring
- Policy entropy tracking to detect convergence issues

### Decision: Memory-Mapped Experience Replay with RAM Cache
**Rationale**:
- Scales beyond RAM capacity for long training runs
- Random access performance adequate for sampling patterns
- RAM cache for recent experiences improves locality

**Implementation Details**:
- Parquet format for compression and schema evolution
- LRU cache for frequently accessed data
- Async writing to prevent training pipeline stalls

## Game-Specific Considerations

### Feature Extraction Strategy
**Gomoku**: 7 planes (current player, opponent, last move, 4 move history)
**Chess**: 12 planes (piece types × 2 colors, castling rights, en passant, move history)
**Go**: 17 planes (current stones, opponent stones, capture patterns, ko detection, move history)

**Rationale**: Balances information richness with neural network capacity and inference speed

### Symmetry Augmentation
**Gomoku/Go**: 8-fold symmetry (4 rotations × 2 reflections)
**Chess**: Limited symmetry (horizontal reflection only for certain positions)

**Impact**: 8× data augmentation for Gomoku/Go significantly improves sample efficiency

## Training Pipeline Optimizations

### Decision: Temperature Scheduling (1.0 → 0.1 transition)
**Rationale**:
- Early exploration (temp=1.0) for first 30 moves generates diverse positions
- Late exploitation (temp=0.1) ensures decisive endgame play
- Gradual transition prevents abrupt strategy changes

### Decision: Dirichlet Noise at Root (α varies by game)
**Rationale**:
- Ensures exploration of underexplored moves during self-play
- Game-specific alpha values account for typical branching factors
- Applied only during training, not evaluation

**Tuning Guidelines**:
- Gomoku: α=0.3 (moderate branching)
- Chess: α=0.2 (high branching complexity)
- Go: α=0.03 (extremely high branching)

## Risk Mitigations

### Atomic Contention Management
**Approach**: Minimize atomic operations to visit count and value updates only
**Monitoring**: Track contention ratios and thread efficiency metrics
**Fallback**: Reduce thread count if contention exceeds 10% overhead

### Memory Leak Prevention
**Approach**: Pre-allocated node pools, RAII principles, smart pointers
**Validation**: 1-hour soak tests with memory profiling
**Monitoring**: RSS memory tracking and growth rate alerting

### GPU Memory Management
**Approach**: Pre-allocated buffers, memory fraction limits (85% of VRAM)
**Monitoring**: Peak usage tracking, OOM detection and graceful fallback
**Optimization**: Buffer reuse, memory pool management

---

*Research conducted based on mcts_guide.md analysis and performance engineering principles for consumer hardware constraints.*