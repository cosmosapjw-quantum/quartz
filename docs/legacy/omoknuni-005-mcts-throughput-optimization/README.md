# Omoknuni - High-Performance AlphaZero Engine

A production-ready AlphaZero-style reinforcement learning engine for board games (Gomoku, Chess, Go) optimized for consumer hardware.

## Project Status

🟢 **Spec 004 Phase 3: Profiling Complete - State Pooling Ready**

### Current Status
- **Version**: 1.0.0-alpha + Spec 004 (Phase 3, Profiling-Validated)
- **Alpha Release Date**: 2025-10-01
- **Spec 002**: ✅ COMPLETE (2025-10-03) - C++ Simulation Runner (7× Python baseline)
- **Spec 003**: ✅ COMPLETE (2025-10-05) - Async inference batching with comprehensive optimization
- **Spec 004**: 🟢 READY FOR T018 (Started 2025-10-06) - MCTS Throughput Recovery
  - Phase 1: ✅ COMPLETE (T001 ✅ T001b ✅ T002 ✅ T003 ✅ T004 ✅ T005 ✅)
  - Phase 2: ✅ COMPLETE + VALIDATED (T006/T006b/T006c/T007a-g/T008a-b,e,f/T009a-f/T010 ✅)
  - Phase 3: 🟡 85% COMPLETE (T011 ✅ T014 ✅, Profiling Campaign ✅, T018 ready)
  - Profiling Campaign: ✅ COMPLETE (560 trials, 100% capture, profiling_suite_20251016_124134)
  - Target: 8,000 sims/sec (achievable via T018 state pooling alone)
- **Current Performance**: 2,659 sims/sec (profiling-validated baseline, mean of 560 trials)
- **Profiling Results (2025-10-16)**:
  - 🔴 **State Cloning**: PRIMARY BOTTLENECK (86.6% of execution time, 418μs per clone)
  - ✅ **GPU Inference**: Fast enough (2.1% of execution time, 20.66ms per batch-64)
  - ✅ **FP16 Mixed Precision**: 1.72× speedup (52.83ms → 30.69ms @ batch-64)
  - ✅ **OpenMP**: Working (8.64ms → 1.57ms @ 12 threads, but 0/560 trials active)
- **Root Cause**: 223 allocations per clone (~2μs each = 446μs overhead)
- **Critical Path**: T018 State Pooling → 9,838 sims/sec (3.7× improvement, exceeds 8k target)

### Spec 003 Complete: Performance Analysis & Optimization

**Achievement**: Full async inference batching pipeline implemented with comprehensive optimization.

**Implementation** ([Spec 003](specs/003-async-inference-batching/)):
- ✅ **AsyncInferenceQueue**: Non-blocking request submission and result distribution
- ✅ **ContinuousSimulationRunner**: Continuous simulation loops without blocking
- ✅ **BatchInferenceCoordinator**: Background thread for GPU batching
- ✅ **Python Integration**: Fast-path batch inference with GPUInferenceWorker
- ✅ **Performance Optimization**: Thread count, batch size, and timeout tuning

**Performance Results** (AMD Ryzen + RTX 3060 Ti, Profiling-Validated 2025-10-16):

| Configuration | Throughput | Efficiency | Status |
|---------------|------------|------------|--------|
| Current (8 threads) | **2,659 sims/sec** | 12.7% | Baseline (profiling campaign) |
| After T018 (est) | **9,838 sims/sec** | ≥60% | State pooling (3.7× improvement) |
| Historical (4 threads) | 3,831 sims/sec | 62.4% | Pre-profiling measurement (outdated) |

**Note**: Historical measurements (3,831, 2,147 sims/sec) predated comprehensive profiling campaign. Current authoritative baseline is 2,659 sims/sec (560 trials, 100% capture).

**Optimal Configuration**:
- Threads: 8 (target ≥60% efficiency after T018)
- Batch size: 64 (best throughput/latency tradeoff)
- Timeout: 0.5-1.0ms (minimal impact on throughput)

**Critical Finding** (see [Performance Analysis](docs/performance/async_optimization_results.md)):
- **GPU Inference**: 32.8% of total time (0.117s / 0.357s for 1000 sims)
- **MCTS Overhead**: 67.2% of total time (selection, backup, coordination)
- **Bottleneck**: Thread coordination and lock contention, NOT GPU
- **Hardware Limit**: RTX 3060 Ti @ FP16 caps at 8,000-10,000 states/sec (validated 2025-10-13)

**Bugs Fixed**:
1. ✅ **Batch explosion**: `collect_batch()` returning all pending requests (capped at 1.5× batch_size)
2. ✅ **PUCT selector**: Excluding expanding nodes to prevent thread contention (19× speedup)

**Progress**: 22/29 tasks complete (75.9%)

**Conclusion**: Current async architecture achieves 48% of 8k realistic target (3,831 baseline). GPU hardware limits (RTX 3060 Ti @ FP16) cap maximum at 8,000-10,000 sims/sec.

**Next Steps**: Spec 004 implementation targeting 8,000 sims/sec (realistic, hardware-grounded) through:
- ✅ WU-UCT virtual loss (T001 complete)
- ✅ Epoch-based tree clearing (T001b complete)
- ✅ Lock-free queues (T006/T006b/T006c complete, validated)
- ✅ Zero-copy tensor bridges (T007a-g complete)
- ✅ FP16 mixed precision (T008f complete, 1.72× speedup validated)
- 🔴 OpenMP parallelization (CRITICAL FIX REQUIRED at dlpack_bridge.cpp:431-434)

See [Async Optimization Results](docs/performance/async_optimization_results.md) for detailed analysis.

### Spec 004: MCTS Throughput Recovery (Phase 2 Validated 2025-10-13)

**Goal**: Achieve 8,000 simulations/second (realistic, hardware-grounded target, revised 2025-10-13)

**Phase 1: Virtual Loss & Quick Wins** - ✅ **COMPLETE** (2025-10-06 to 2025-10-08)
- ✅ **T001**: WU-UCT Virtual Loss Manager (visit-only, no Q-value distortion)
- ✅ **T001b**: Epoch-based tree clearing (1M× speedup: 25ns vs 25ms)
- ✅ **T002**: Busy-edge masking validation + critical thread-local bug fix
- ✅ **T003**: Root pre-expansion (eliminates N-1 thread idle problem)
- ✅ **T004**: Thread affinity for Ryzen 5900X (CCD-aware pinning)
- ✅ **T005**: Collision metrics instrumentation (<0.5% collision rate @ 4 threads)

**Phase 2: Architecture Changes** - ✅ **COMPLETE + VALIDATED** (2025-10-13)
- ✅ **T006**: Lock-free MPMC queue implementation (4096 entries, turn-based sync)
- ✅ **T006b**: AsyncInferenceQueue integration (lock-free operations)
- ✅ **T006c**: Condition variables implementation - **VALIDATED 2025-10-13**
  - Replaced polling with `std::condition_variable` for efficient blocking
  - Impact: Eliminates CPU polling waste
  - Status: ✅ COMPLETE and integrated
- ✅ **T007**: DLPack Tensor Bridge - **COMPLETE** (T007a-g all done)
  - T007a-g: Research, pinned memory, capsules, batch tensors, feature extraction, Python bindings, validation
  - Zero-copy C++ → PyTorch pipeline operational
  - ❌ **Critical Issue Found**: Feature extraction loop at dlpack_bridge.cpp:431-434 NOT parallelized with OpenMP
  - Measured overhead: 7.5ms per batch-64 (should be <1.0ms)
- ✅ **T008**: Python Inference Bridge (T008a-b,e complete, T008c-d skipped)
  - T008a-b: DLPackInferenceBridge class, torch.from_dlpack() conversion
  - T008c-d: Pre-allocated GPU buffers (skipped - not needed), non-blocking transfers (already async)
  - T008e: Integration testing and validation
- ✅ **T008f**: FP16 mixed precision - **VALIDATED 2025-10-13**
  - ✅ **T-VALID-1 PASS**: 1.72× speedup (52.83ms → 30.69ms @ batch-64)
  - Policy MSE: 0.000007 (excellent, target <0.01)
  - Value MSE: 0.000000 (perfect, target <0.01)
  - Status: ✅ COMPLETE, working correctly, significant GPU speedup confirmed
- ✅ **T009**: Per-Thread Memory Arenas (T009a-f complete)
  - 4096-node block allocation, 99.93% fast-path, 0.07% mutex fallback
- ✅ **T010**: Replace pending expansions map with ring buffer

**Phase 3: Final Optimizations** - 🟡 **80% COMPLETE** (2025-10-10)
- ✅ **T011**: Persistent Coordinator Lifecycle (T011a-c complete)
  - T011a: Move coordinator to instance variable (✅ single instance, reused across searches)
  - T011b: Handle coordinator state across searches (✅ 1000+ search persistence validated)
  - T011c: Performance validation and documentation (✅ 100× thread reduction, <0.3KB/search memory)
  - **Impact**: 15-25% throughput improvement, 500× fewer thread operations
- ✅ **T012**: Relaxed memory ordering - ✅ COMPLETE (2025-10-10)
  - Verified optimal memory ordering already implemented (relaxed for counters, acquire/release for sync)
  - **Impact**: 1.05× throughput (statistics already using relaxed ordering)
- ✅ **T013**: Selection prefetching - ✅ COMPLETE (2025-10-10)
  - Added `__builtin_prefetch()` hints to SIMD and scalar selection paths
  - Prefetches 5 arrays (visit_counts, total_values, prior_probs, virtual_losses, flags)
  - **Impact**: 1.05-1.10× speedup (minimal - selection not bottleneck)
  - **Note**: MCTS coordination (60% thread waiting) is real bottleneck, not selection
- ✅ **T014**: Batched result processing - ✅ COMPLETE (2025-10-09)
  - Reduced atomic operations by 2× via update accumulation
  - **Impact**: Improved scaling with multiple threads
- ⏸️ **T015**: Hot/cold child separation (cache-aware data layout)

**Phase 4: Integration & Tuning** - 🔴 **BLOCKED** (OpenMP fix required)
- 🔴 **CRITICAL BLOCKER**: Fix OpenMP parallelization at dlpack_bridge.cpp:431-434 (2 hours)
- 🔴 **T016**: Performance benchmark suite (READY after OpenMP fix)
- 🔴 **T017**: Baseline investigation (determine 3,831 → 2,147 regression cause)
- T018-T019: Virtual loss tuning, batch size/timeout optimization
- T020-T025: Profiling, validation, documentation

**Profiling Results (2025-10-16)**:
- **Baseline**: 2,659 sims/sec (profiling-validated, 560 trials, 100% capture)
- ✅ **Bottleneck Identified**: State cloning 86.6% (418μs per clone, 223 allocations)
- ✅ **GPU Inference**: Fast enough (2.1% of time, not the bottleneck)
- ✅ **FP16 Mixed Precision**: 1.72× GPU speedup, numerical stability excellent
- ✅ **OpenMP**: Working correctly (8.64ms → 1.57ms @ 12 threads, but 0/560 trials active)
**Critical Path Forward** (Updated 2025-10-16):
1. 🔴 **Implement T018 (State Pooling)** - Thread-local state pools with copyFrom() API (2-3 days)
   - Expected: 2,659 → 9,838 sims/sec (3.7× improvement, exceeds 8k target)
2. **Validate with Profiling** - Run profiling campaign to confirm state_clone_total < 20% (1 day)
3. **T019 (OpenMP Investigation)** - Optional: Debug why 0/560 trials active (1-2 days, stretch goal)
4. **T020 (Allocation Reduction)** - Optional refinement (~5% improvement)
5. **Tune T018-T019** - Optimize parameters (threads, batch size, timeout, virtual loss) (2 days)
6. **Validate T021-T025** - Final validation and documentation (3 days)

See [Spec 004](specs/004-mcts-throughput-recovery/) for full implementation plan.

### Completed
- [x] **T001**: Project structure and build system setup
- [x] **T002**: CI/CD pipeline with GitHub Actions
- [x] **T003**: Basic telemetry framework with Prometheus metrics
- [x] **T004**: GPU warmup and device detection system
- [x] **T005**: MCTS API contract tests for Test-Driven Development
- [x] **T006**: Structure-of-Arrays memory layout (27 bytes/node, <1GB for 50M nodes)
- [x] **T007**: Node pool pre-allocation for O(1) MCTS tree operations (330M allocations/sec)
- [x] **T008**: Vectorized PUCT selection with AVX2 SIMD optimizations (3.6-5.2x speedup)
- [x] **T009**: Thread-safe virtual loss mechanism for MCTS coordination
- [x] **T010**: Value backup mechanism with proper sign flipping per tree level
- [x] **T011**: Single-threaded MCTS integration test for complete search cycle
- [x] **T012**: Contract test for neural network inference API
- [x] **T013**: ResNet architecture with Squeeze-Excitation attention (20 blocks, 256 channels)
- [x] **T014**: GPU inference worker with queue-based threading and dynamic batching
- [x] **T015**: Dynamic micro-batching with count-based (≥32) OR timeout-based (≤3ms) optimization
- [x] **T016**: Mixed precision inference with fp16 computation and automatic fallback mechanisms
- [x] **T017**: Pinned memory optimization for efficient GPU data transfers (H2D/D2H optimized)
- [x] **T018**: CPU fallback mechanism for robust inference reliability
- [x] **T019**: Inference integration test for full pipeline validation
- [x] **T020**: Gomoku game implementation with enhanced 36-plane tensor representation
- [x] **T021**: Chess game implementation with enhanced 30-plane tensor representation
- [x] **T022**: Go game implementation with enhanced 25-plane tensor representation
- [x] **T023**: Game adapter interface with unified polymorphic dispatch across all games
- [x] **T024**: Python bindings for games with pybind11 and numpy array compatibility
- [x] **T025**: Game rule unit tests with comprehensive verification for all games
- [x] **T026**: Contract test for training API with comprehensive coverage
- [x] **T027**: Asynchronous search coordinator with thread pool management
- [x] **T028**: Self-play game generator with comprehensive testing and validation
- [x] **T029**: Memory-mapped experience buffer for high-performance training data storage
- [x] **T030**: Advanced experience replay sampling with balanced distribution and temporal uniformity
- [x] **T031**: Self-play generation integration tests with realistic GPU/CPU inference validation
- [x] **T032**: Model trainer implementation with AdamW optimizer and mixed precision training
- [x] **T033**: Training loop orchestration with comprehensive cycle management
- [x] **T034**: Model evaluation system with head-to-head comparison and ELO rating
- [x] **T035**: Training stability monitoring with NaN detection and automatic recovery
- [x] **T036**: Checkpoint management system with automatic saving, best model tracking, and retention policies
- [x] **T037**: Training pipeline integration test with full end-to-end validation
- [x] **T038**: Thread count optimization script with parameter sweep, contention detection, and real component integration
- [x] **T039**: Virtual loss magnitude tuning script with comprehensive parameter sweep, thread efficiency measurement, and exploration balance
- [x] **T040**: Batch size optimization script with GPU memory profiling, VRAM monitoring, throughput/latency analysis, and OOM handling
- [x] **T045**: Configuration management system with unified YAML-based configurations, environment overrides, and comprehensive validation
- [x] **T046**: Comprehensive operations runbook with deployment procedures, monitoring setup, troubleshooting guide, and maintenance tasks
- [x] **T047**: Complete API documentation with comprehensive reference, usage examples, and parameter descriptions
- [x] **T048**: Training guide with hyperparameter recommendations, game-specific settings, and comprehensive troubleshooting
- [x] **T049**: Comprehensive sanitizer build system with AddressSanitizer, ThreadSanitizer, and UndefinedBehaviorSanitizer support
- [x] **T050**: CUDA OOM recovery mechanisms with automatic batch size reduction, graceful degradation, and CPU fallback
- [x] **T051**: Memory leak detection system with Python profiling, valgrind integration, and GPU memory monitoring
- [x] **T052**: Comprehensive error handling hardening framework for production stability and resilience

### Current Architecture

- **Hybrid CPU/GPU**: Shared-tree MCTS on CPU with asynchronous GPU neural network inference
- **Performance Target**: 8,000 simulations/second with 80% GPU utilization (realistic, hardware-grounded)
- **Target Hardware**: AMD Ryzen 9 5900X (12C/24T, dual-CCD) + NVIDIA RTX 3060 Ti (8GB VRAM, Ampere)
- **Memory Efficient**: Structure-of-Arrays layout, <1GB tree memory for 10M nodes (27 bytes/node achieved)
- **Advanced Batching**: Dynamic micro-batching with count-based (≥32) OR timeout-based (≤3ms) optimization
- **GPU Monitoring**: Real-time utilization tracking with nvidia-ml-py and adaptive batch sizing
- **Vectorized Operations**: AVX2-optimized PUCT selection with 3.6-5.2x performance improvement
- **Thread Safety**: Virtual loss coordination and atomic operations for 8-12 parallel workers
- **Memory Optimization**: Pinned CUDA memory buffers for faster H2D/D2H transfers with automatic fallback
- **Robust Inference**: CPU fallback mechanism with automatic GPU failure detection and seamless switching
- **Unified Game Interface**: Polymorphic dispatch enabling MCTS to work with Chess, Go, and Gomoku seamlessly
- **Game Type Detection**: Automatic detection from notation (FEN for Chess, SGF for Go, coordinate for Gomoku)
- **Standard Format Support**: Export/import in established formats (PGN, SGF, custom notation)
- **Python Bindings**: pybind11 integration with numpy array compatibility for neural network feature extraction
- **High-Performance Interop**: 250k+ tensor extractions/second, C-contiguous memory layout for zero-copy operations
- **Enhanced Tensor Representations**: Advanced feature planes for superior positional understanding
  - **Gomoku**: 36 planes with threat detection, run-length analysis, and rule variations
  - **Chess**: 30 planes with castling rights, en passant, and 8-pair move history
  - **Go**: 25 planes with proper move history separation and capture patterns
- **Self-Play Training**: Complete game generation with temperature scheduling and Dirichlet noise
  - **Temperature Control**: Configurable exploration→exploitation transitions
  - **Game Variations**: Support for Renju/Omok, Chess960, Chinese/Japanese/Korean Go rules
  - **Bias Detection**: Statistical analysis to ensure fair move distributions
  - **Policy Health**: Entropy monitoring and MCTS convergence validation
- **Experience Replay**: Memory-mapped buffer with Parquet storage and intelligent LRU caching
  - **High Performance**: 14.8K examples/sec addition, 643 samples/sec balanced sampling
  - **Memory Efficient**: Configurable buffer size (default 1M examples) with automatic cleanup
  - **Persistent Storage**: Parquet columnar format with thread-safe concurrent access
  - **Smart Caching**: LRU cache for frequently accessed examples (default 512MB)
  - **Balanced Sampling**: Exact game type ratios with temporal uniformity to prevent bias
  - **Training Iterator**: 136K samples/sec continuous batch generation with shuffle buffering
- **Model Training**: Production-ready neural network trainer with advanced optimization
  - **AdamW Optimizer**: Configurable weight decay for improved generalization and training stability
  - **Cosine LR Scheduling**: Annealing with warm restarts for optimal convergence patterns
  - **Mixed Precision**: PyTorch AMP integration for 2x memory efficiency on RTX 3060 Ti
  - **Gradient Clipping**: Training stability with configurable norm thresholds
  - **Checkpoint Management**: Full model and training state persistence with automatic game type detection
  - **Training Metrics**: Loss history, performance statistics, and validation tracking
- **Training Loop Orchestration**: Complete training cycle coordination and management
  - **Cycle Management**: Coordinates self-play → experience → training → evaluation cycles
  - **Configuration System**: Comprehensive TrainingConfig dataclass with all training parameters
  - **Graceful Shutdown**: Signal handling (SIGINT/SIGTERM) with resource cleanup
  - **Performance Monitoring**: Games/hour, training steps/minute, memory usage tracking
  - **Early Stopping**: Evaluation-based stopping with configurable patience
  - **Recovery Capabilities**: Training state persistence and restoration after interruption
  - **Parallel Self-Play**: Configurable worker threads for concurrent game generation
- **Advanced Model Evaluation System**: Sophisticated model comparison with Glicko-2 rating
  - **Glicko-2 Rating System**: Advanced rating with uncertainty (RD) and volatility tracking for accurate strength assessment
  - **Baseline Anchoring**: Two baseline systems (random moves and uniform policy) with anchored recentering for scale stability
  - **Random Move Generator**: Pure random legal moves (no MCTS) for fast baseline evaluation and accelerated assessment
  - **Statistical Analysis**: Wilson confidence intervals and binomial significance testing with robust statistical decisions
  - **Head-to-Head Evaluation**: Parallel game evaluation between different model checkpoints with comprehensive metrics
  - **Performance Optimized**: Fast Glicko-2 implementation with iteration limits and relaxed convergence for production use
  - **Multi-Game Support**: Unified evaluation interface for Gomoku, Chess, and Go with game-specific optimizations
  - **Result Persistence**: JSON serialization for evaluation history, rating progression, and detailed analysis
- **Training Stability Monitoring**: Comprehensive stability system with automatic recovery capabilities
  - **NaN Detection**: Real-time detection of NaN values in model parameters, gradients, and loss functions
  - **Gradient Explosion Monitoring**: Automatic detection and intervention for gradient explosion events
  - **Loss Convergence Tracking**: Monitoring of training progress with divergence and plateau detection
  - **Early Stopping**: Intelligent stopping based on validation metrics and improvement thresholds
  - **Automatic Recovery**: Learning rate reduction and gradient history cleanup for training recovery
  - **Statistical Analysis**: Trend analysis and convergence assessment using polynomial regression
  - **Comprehensive Logging**: Detailed stability warnings and recovery attempt notifications
- **Telemetry**: Prometheus-compatible metrics with comprehensive performance monitoring
- **Configuration Management**: Unified YAML-based configuration system with comprehensive parameter management
  - **Multi-Environment Support**: Optimized configurations for default, development, and production scenarios
  - **Environment Variable Overrides**: ALPHAZERO_ prefixed environment variables with intelligent type conversion
  - **Type-Safe Configuration**: Python dataclasses with comprehensive validation for all AlphaZero parameters
  - **Parameter Coverage**: All tunable parameters for MCTS (13), neural network (12), training (15), game (7), and system (11) components
  - **Validation System**: Prevents invalid parameter combinations with detailed error messages and constraints checking
  - **Configuration Files**:
    - `config/default.yaml`: Balanced configuration suitable for most scenarios
    - `config/development.yaml`: Development-optimized with faster iterations and verbose logging
    - `config/production.yaml`: Production-optimized for maximum performance and reliability
- **Operations & Deployment**: Comprehensive operational procedures and deployment automation
  - **Multi-Platform Deployment**: Docker containerization, bare-metal installation, cloud deployment (AWS/GCP)
  - **Monitoring & Observability**: Prometheus metrics, Grafana dashboards, structured logging, health checks
  - **Troubleshooting**: Diagnostic procedures for CUDA OOM, MCTS performance, training instability, container issues
  - **Maintenance Automation**: Daily/weekly/monthly maintenance scripts, scaling procedures, backup/restore operations
  - **Security Hardening**: Container security, network configuration, data protection, compliance monitoring
  - **Disaster Recovery**: RTO/RPO procedures, automated failover, system recovery protocols
- **API Documentation**: Comprehensive programming interface reference with usage examples
  - **Complete API Coverage**: MCTS Engine, Neural Network Inference, Training Pipeline, Game Interface, Configuration, and Telemetry APIs
  - **Working Code Examples**: Complete training setup, game analysis, performance monitoring, and error handling patterns
  - **Parameter Documentation**: Detailed parameter tables with types, defaults, and performance targets
  - **Hardware Optimization**: AMD Ryzen 5900X + RTX 3060 Ti specific tuning guidelines
  - **Error Handling Guide**: Common exceptions, diagnostic procedures, and troubleshooting patterns
  - **Performance Targets**: 8k sims/sec (realistic), 80% GPU utilization, <1GB memory, 200+ games/hour

### Performance Regression Suite & Benchmarking

Comprehensive automated benchmarking system with regression detection to ensure consistent performance across code changes:

- **Automated Performance Testing**: Complete benchmark framework covering all critical AlphaZero performance metrics
- **MCTS Simulation Benchmarks**: Validates 8k simulations/second including neural network inference (realistic target)
- **Neural Inference Throughput**: Tests GPU/CPU inference performance with realistic batch processing
- **GPU Utilization Monitoring**: Ensures 80-92% sustained GPU utilization during search operations
- **Memory Efficiency Validation**: Confirms <1GB usage for 10M node MCTS trees with structure-of-arrays layout
- **Search Coordinator Performance**: Tests coordination throughput with realistic workload simulation
- **Regression Detection System**: Automatic baseline comparison with configurable threshold analysis (>5% regression alerts)
- **System Resource Monitoring**: Comprehensive CPU, GPU, memory, and thread monitoring with psutil/pynvml
- **Statistical Measurement**: Multiple iterations with warmup periods and variance analysis for reliable results
- **CI/CD Integration**: Automated baseline updates, performance trend reporting, and regression alerts
- **Detailed Analytics**: JSON result output supporting historical performance analysis and trend tracking
- **Memory Leak Detection**: Comprehensive monitoring with Python profiling, valgrind C++ analysis, and GPU memory tracking
- **Long-term Stability**: Automated soak testing with configurable thresholds for production readiness validation
- **Error Handling Framework**: Production-ready error handling with custom exception hierarchy, thread health monitoring, and graceful degradation
- **Fault Tolerance**: GPU operation management with timeout protection, emergency shutdown procedures, and comprehensive recovery mechanisms

```bash
# Run performance benchmarks
python -m pytest tests/performance/ -v

# Run specific benchmark categories
python -m pytest -m "performance" -v
python -m pytest -m "benchmark" -v

# Run benchmarks directly with detailed output
python tests/performance/test_benchmarks.py

# Check for performance regressions
python -m pytest tests/performance/test_benchmarks.py::test_performance_regression_detection -v
```

## Quick Start

### Prerequisites
- Python 3.12+
- CMake 3.18+
- CUDA 12.x (for GPU acceleration)
- C++17 compatible compiler with OpenMP support
- Docker and Docker Compose (for containerized deployment)

### Docker Deployment (Recommended)

#### Quick Start with Docker
```bash
# Build and run development environment
./scripts/docker/build.sh -t development
./scripts/docker/run.sh dev

# Or use docker-compose
docker-compose up dev
```

#### Production Deployment
```bash
# Build production image
./scripts/docker/build.sh -t runtime

# Run in production mode
docker-compose up -d runtime
```

#### Training with Docker
```bash
# Start training environment
docker-compose up -d training

# Monitor training progress
docker-compose logs -f training

# Access TensorBoard at http://localhost:6007
```

### Alpha Testing

Before production use, complete the alpha and beta testing phases. See the [Alpha/Beta Testing Guide](docs/alpha_beta_testing_guide.md) for detailed procedures.

**Quick Alpha Validation**:
```bash
# Run contract tests
python -m pytest tests/contract/ -v

# Run integration tests
python -m pytest tests/integration/test_full_system.py -v

# Run performance benchmarks
python -m pytest tests/performance/test_benchmarks.py -v
```

### Training Your First Model

For detailed training instructions, see the [Training Guide](docs/training_guide.md) with comprehensive hyperparameter recommendations and troubleshooting.

#### Production 48-Hour Gomoku Training
```bash
# Train Gomoku model (superhuman performance in 48 hours)
python -m src.training.training_loop \
    --config config/gomoku_48h_training.yaml \
    --max-time-hours 48

# Train Chess model (strong amateur in 1 week)
python -m src.training.training_loop \
    --game chess \
    --config config/training_chess.yaml \
    --target-time-hours 168

# Monitor training with TensorBoard
tensorboard --logdir results/*/tensorboard
```

#### Expected Training Performance
- **Gomoku**: Superhuman performance in 48 hours
- **Chess**: Strong amateur (1700-1900 ELO) in 1 week
- **Go**: Competitive performance (varies by board size)

For game-specific hyperparameters, troubleshooting common issues, and performance optimization, see the complete [Training Guide](docs/training_guide.md).

### Docker Services Available
- **dev**: Development environment with Jupyter Lab (port 8888)
- **training**: Training environment with TensorBoard (port 6007)
- **runtime**: Production runtime environment
- **benchmark**: Performance benchmarking
- **tensorboard**: Standalone TensorBoard monitoring (port 6008)

### Setup Development Environment

```bash
# Clone and setup
git clone <repository>
cd omoknuni

# Create virtual environment
python3.12 -m venv venv --prompt omoknuni
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Build C++ extensions (when available)
export CFLAGS="-O3 -march=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -fopenmp"
python -m pip install -e . --config-settings build-dir=build
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/contract/      # API contract tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/performance/   # Performance benchmarks & regression detection
python -m pytest tests/soak/          # Long-running stability tests

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Test micro-batching performance
python scripts/validate_micro_batching.py

# Test mixed precision inference
python scripts/validate_mixed_precision.py

# Test pinned memory optimization
python scripts/validate_pinned_memory.py

# Test CPU fallback mechanism
python scripts/validate_cpu_fallback.py

# Memory leak detection (comprehensive testing)
python scripts/check_memory_leaks.py --all --duration 600 --output leak_report.json
python scripts/check_memory_leaks.py --python --threshold 5.0        # Python profiling only
python scripts/check_memory_leaks.py --valgrind --component mcts     # C++ component analysis
python scripts/check_memory_leaks.py --gpu --verbose                 # GPU memory monitoring

# Error handling framework testing
python -m pytest tests/unit/test_error_handling.py -v               # Comprehensive error handling tests
python -m pytest tests/unit/test_error_handling.py -v -k "ThreadHealth"    # Thread monitoring tests
python -m pytest tests/unit/test_error_handling.py -v -k "GPUOperation"    # GPU error handling tests
python -m pytest tests/unit/test_error_handling.py -v -k "ModelValidator"  # Model validation tests

# Test experience buffer with memory-mapped storage
python scripts/validate_experience_buffer.py

# Test advanced experience sampling with balanced distribution
python scripts/validate_experience_sampling.py

# Test game adapter interface
python -m pytest tests/unit/test_game_adapter_interface.py -v

# Test inference integration pipeline
python -m pytest tests/integration/test_inference_integration.py -v

# Test Python bindings for games
python -m pytest tests/unit/test_python_bindings.py -v

# Run Python bindings demonstration
python examples/python_bindings_demo.py

# Test comprehensive game rules across all games
python -m pytest tests/unit/test_game_rules.py -v

# Test self-play game generation
python -m pytest tests/unit/test_self_play.py -v

# Test comprehensive self-play analysis
python -m pytest tests/integration/test_self_play_comprehensive.py -v

# Test terminal detection and game variations
python -m pytest tests/integration/test_terminal_detection_variations.py -v

# Test realistic self-play integration with actual GPU/CPU inference workers
python -m pytest tests/integration/test_self_play_realistic.py -v

# Test neural network model trainer with AdamW optimizer and mixed precision
python -m pytest tests/unit/test_trainer.py -v

# Test training loop orchestration with cycle management
python -m pytest tests/unit/test_training_loop.py -v

# Test advanced model evaluation system with Glicko-2 rating and statistical analysis
python -m pytest tests/unit/test_evaluator.py -v

# Test training stability monitoring with NaN detection and automatic recovery
python -m pytest tests/unit/test_training_stability.py -v

# Run comprehensive self-play testing (all games and variations)
python scripts/test_self_play_comprehensive.py --quick-test --games 5 --output results/

# Run full comprehensive test with visualizations
python scripts/test_self_play_comprehensive.py --games 20 --output results/full_test

# Test configuration system
python -m pytest tests/unit/test_config_system.py -v

# Test operations documentation
python -m pytest tests/unit/test_operations_docs.py -v

# Test API documentation
python -m pytest tests/unit/test_api_docs.py -v
```

### Configuration Management

The engine uses a comprehensive YAML-based configuration system with environment variable overrides:

```bash
# Load default configuration
python -c "from src.utils.config import load_config; config = load_config(); print(f'MCTS simulations: {config.mcts.simulations}')"

# Load development configuration
python -c "from src.utils.config import load_config; config = load_config('config/development.yaml'); print(f'Log level: {config.system.log_level}')"

# Override with environment variables
export ALPHAZERO_MCTS_SIMULATIONS=2000
export ALPHAZERO_NEURAL_NETWORK_LEARNING_RATE=0.01
python -c "from src.utils.config import load_config; config = load_config(); print(f'Overridden simulations: {config.mcts.simulations}')"

# Validate configuration
python -c "
from src.utils.config import ConfigManager
manager = ConfigManager('config/default.yaml')
try:
    config = manager.load_config()
    print('✅ Configuration validation passed')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

#### Configuration File Structure

- **MCTS Parameters**: Simulations, exploration constant, threads, batch sizing, tree memory limits
- **Neural Network**: Architecture (channels, blocks), training parameters, mixed precision settings
- **Training Pipeline**: Self-play games, experience buffer, learning schedule, evaluation frequency
- **Game Settings**: Game type, board size, feature extraction, rule variants
- **System Configuration**: Logging, resource limits, directory paths, profiling options

#### Environment Variable Format

Use the format `ALPHAZERO_<SECTION>_<PARAMETER>`:

```bash
# MCTS configuration
export ALPHAZERO_MCTS_SIMULATIONS=1600
export ALPHAZERO_MCTS_THREADS=12

# Neural network configuration
export ALPHAZERO_NEURAL_NETWORK_LEARNING_RATE=0.001
export ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION=true

# Training configuration
export ALPHAZERO_TRAINING_BATCH_SIZE=1024
export ALPHAZERO_TRAINING_SELF_PLAY_GAMES_PER_ITERATION=100

# System configuration
export ALPHAZERO_SYSTEM_LOG_LEVEL=DEBUG
export ALPHAZERO_SYSTEM_MAX_MEMORY_GB=64
```

### Operations & Deployment

The engine includes comprehensive operational procedures for production deployment:

```bash
# Quick production deployment with Docker
./scripts/docker/build.sh -t runtime
docker-compose up -d runtime

# Validate deployment
curl http://localhost:8000/health
python scripts/health_check.py

# Monitor system performance
python -c "from src.telemetry.metrics import get_metrics; print(get_metrics())"

# Emergency procedures
sudo systemctl stop alphazero  # Emergency stop
docker-compose logs runtime    # Check logs
python scripts/backup_daily.sh # Backup critical data
```

#### Deployment Options

- **Docker**: Recommended for production with multi-stage builds (development/runtime/training)
- **Bare Metal**: Direct installation with systemd service configuration
- **Cloud**: AWS EC2/GCP VM deployment with GPU support and auto-scaling

#### Monitoring & Observability

- **Prometheus Metrics**: Real-time performance monitoring (simulations/sec, GPU utilization, memory usage)
- **Grafana Dashboards**: Visual monitoring with alerting for performance degradation
- **Structured Logging**: JSON logs with configurable levels and automatic rotation
- **Health Checks**: Automated system health validation and diagnostics

#### Maintenance & Support

- **Automated Maintenance**: Daily/weekly/monthly scripts for system optimization
- **Performance Tuning**: Hardware-specific optimization for AMD Ryzen 5900X + RTX 3060 Ti
- **Disaster Recovery**: Automated backup/restore with <4hr RTO and failover procedures
- **Security Hardening**: Container security, network isolation, and compliance monitoring

For detailed operational procedures, see the [Operations Runbook](docs/operations.md).

### API Usage

The engine provides comprehensive APIs for all major components:

```python
# Complete training setup
from src.utils.config import load_config
from src.training.training_loop import TrainingLoop
from src.neural.inference_worker import InferenceWorker
from src.core.mcts_engine import MCTSEngine

# Load configuration
config = load_config('config/production.yaml')

# Initialize inference worker
worker = InferenceWorker(
    model_path="models/gomoku_init.pth",
    batch_size=config.neural_network.batch_size_preferred,
    timeout_ms=config.mcts.inference_timeout_ms
)

# Initialize MCTS engine
mcts = MCTSEngine(
    inference_worker=worker,
    simulations=config.mcts.simulations,
    threads=config.mcts.threads
)

# Start training
training_loop = TrainingLoop(config)
training_loop.start_continuous_training()
```

#### Game Analysis Example

```python
# Analyze single position
import numpy as np
from src.games.gomoku import GomokuState

# Create game and apply moves
game = GomokuState(board_size=15)
game.apply_move_inplace(112)  # Center move

# Get MCTS analysis
policy, value = mcts.search(game, temperature=0.1)
best_action = np.argmax(policy)
print(f"Best move: {divmod(best_action, 15)}, Value: {value:.3f}")
```

#### Performance Monitoring

```python
# Monitor system performance
from src.telemetry.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record_mcts_performance(sims_per_sec=35000, avg_tree_size=50000)
metrics.record_gpu_metrics(utilization_percent=87, memory_used_mb=6800, batch_size_avg=64)

summary = metrics.get_metrics_summary()
print(f"Performance: {summary}")
```

For complete API reference with detailed parameters and examples, see the [API Documentation](docs/api.md).

### Self-Play Training & Testing

The engine includes comprehensive self-play testing to ensure training data quality:

```bash
# Quick self-play validation across all games
python scripts/test_self_play_comprehensive.py --quick-test

# Full comprehensive analysis with move bias detection
python scripts/test_self_play_comprehensive.py --games 50 --output results/comprehensive

# Analyze existing results
python scripts/test_self_play_comprehensive.py --analyze-only results/analysis_results.json

# Optimize thread count for MCTS performance
python scripts/tune_threads.py --game gomoku --quick-test

# Full thread optimization with parameter sweep
python scripts/tune_threads.py --game gomoku --simulations 200 --max-threads 16

# Optimize virtual loss magnitude for thread efficiency
python scripts/tune_virtual_loss.py --game gomoku --quick-test

# Full virtual loss optimization with exploration balance analysis
python scripts/tune_virtual_loss.py --game gomoku --simulations 800 --iterations 50

# Optimize batch size for GPU memory and throughput
python scripts/tune_batch_size.py --game gomoku --quick-test

# Full batch size optimization with VRAM profiling
python scripts/tune_batch_size.py --game gomoku --iterations 100 --max-vram 85

# Optimize inference timeout for throughput vs latency balance
python scripts/tune_timeout.py --game gomoku --quick-test

# Full timeout optimization with comprehensive analysis
python scripts/tune_timeout.py --game gomoku --min-timeout 1 --max-timeout 10 --iterations 100

# Profile coordinator lifecycle (T011c)
python scripts/profile_coordinator_lifecycle.py --quick              # 100 searches, quick test
python scripts/profile_coordinator_lifecycle.py --searches 1000      # Full profiling
```

**Self-Play Features:**
- **Move Bias Analysis**: Statistical tests to detect spatial bias, corner/edge preferences
- **Policy Entropy Monitoring**: Tracks exploration→exploitation balance throughout games
- **Game Variation Testing**: Validates Renju/Omok, Chess960, and Go rule variations
- **Terminal Detection**: Comprehensive validation of win/draw/timeout conditions
- **Health Metrics**: MCTS convergence quality, temperature scheduling effectiveness
- **Visualization**: Automated generation of bias analysis and entropy pattern plots

**Thread Optimization:**
- **Parameter Sweep**: Systematic testing of thread counts 1-16 with performance measurement
- **Contention Detection**: Real-time analysis of threading efficiency and resource contention
- **Performance Profiling**: Throughput measurement, search timing, and system resource monitoring
- **Optimization Algorithms**: Automated recommendations based on hardware characteristics
- **Real Component Integration**: Uses actual MCTS, inference workers, and game modules
- **Comprehensive Reporting**: JSON serialization with optional visualization plots

**Virtual Loss Tuning:**
- **Parameter Sweep**: Comprehensive testing of VL magnitudes 0.5-3.0 with configurable step sizes
- **Thread Efficiency**: Measurement of coordination effectiveness and threading performance
- **Exploration Balance**: Policy entropy and visit distribution diversity analysis
- **Contention Analysis**: Detection of thread conflicts through timing variance measurement
- **Multi-Game Support**: Optimization for Gomoku, Chess, and Go with game-specific metrics
- **Statistical Analysis**: Comprehensive scoring combining throughput, efficiency, and exploration balance

**Batch Size Optimization:**
- **GPU Memory Profiling**: Real-time VRAM monitoring with pynvml integration
- **Throughput Analysis**: Comprehensive inference performance measurement across batch sizes 8-512
- **Latency vs Throughput**: Optimal batch size detection balancing speed and responsiveness
- **OOM Handling**: Automatic out-of-memory detection with graceful degradation
- **VRAM Constraint Management**: Configurable memory limits (<85% VRAM usage by default)
- **Multi-Game Memory Profiling**: Game-specific memory requirements and optimization
- **Efficiency Scoring**: Combined metrics of throughput, memory usage, GPU utilization, and latency

**Inference Timeout Optimization:**
- **Timeout Parameter Sweep**: Comprehensive testing of timeout values 1-10ms with configurable step sizes
- **Throughput vs Latency Analysis**: Optimal timeout detection balancing responsiveness and batch efficiency
- **Batch Formation Efficiency**: Measurement of timeout hit rates and batch size distributions
- **GPU Utilization Monitoring**: Real-time GPU usage tracking during timeout optimization
- **Responsiveness Analysis**: Response time statistics with percentile analysis (P95, P99)
- **Queue Behavior Analysis**: Queue depth statistics and batching behavior measurement
- **Multi-Game Support**: Timeout optimization for Gomoku, Chess, and Go with game-specific requirements
- **Efficiency Scoring**: Combined metrics of throughput, responsiveness, timeout efficiency, and batch size optimization

**Memory Stability & Soak Testing:**
- **1-Hour Soak Tests**: Comprehensive long-running stability validation with continuous monitoring
- **Memory Leak Detection**: Automated detection with <10MB/hour growth threshold validation
- **Performance Degradation Monitoring**: Track performance stability over time (<5% degradation limit)
- **System Resource Tracking**: Monitor memory, CPU, GPU utilization, thread counts, and file handles
- **Workload Simulation**: Multi-threaded realistic AlphaZero operation simulation with configurable load
- **Automated Leak Detection**: Statistical trend analysis and resource threshold algorithms
- **Comprehensive Reporting**: Detailed results with JSON export, failure analysis, and emergency stops
- **Long-Running Framework**: Support for extended operation validation (1+ hours continuous testing)
```

### Project Structure

```
├── src/                    # Python orchestration layer
│   ├── core/              # MCTS search coordination
│   ├── games/             # Game implementations
│   ├── neural/            # Neural networks, GPU inference, micro-batching & CPU fallback
│   ├── training/          # Self-play & training pipeline (T028 comprehensive testing)
│   ├── telemetry/         # Performance monitoring
│   └── utils/             # Shared utilities (including configuration system)
├── config/                # Configuration management
│   ├── default.yaml       # Balanced default configuration
│   ├── development.yaml   # Development-optimized settings
│   └── production.yaml    # Production-optimized settings
├── docs/                  # Documentation
│   ├── operations.md      # Comprehensive operations runbook
│   ├── api.md             # Complete API reference documentation
│   └── training_guide.md  # Complete training guide with hyperparameters and troubleshooting
├── cpp_extensions/        # Performance-critical C++ code
│   ├── mcts/             # Core MCTS tree operations
│   ├── games/            # Game rule implementations, unified interface & Python bindings
│   └── utils/            # Memory management & vectorization
├── tests/                # Comprehensive test suite
│   ├── contract/         # API contract validation
│   ├── integration/      # End-to-end system tests (including comprehensive self-play)
│   ├── unit/            # Component unit tests (including config system & Python bindings)
│   └── performance/     # Benchmarking & regression tests
├── scripts/              # Testing and validation scripts
│   └── test_self_play_comprehensive.py  # Full self-play analysis with visualizations
└── examples/             # Usage demonstrations and tutorials
```

## Specification

This project follows [Spec-Driven Development](specs/001-goal-create-spec/).

### Spec 001: AlphaZero Engine Foundation (✅ Completed)
- [Feature Specification](specs/001-goal-create-spec/spec.md) - Requirements and objectives
- [Implementation Plan](specs/001-goal-create-spec/plan.md) - Technical architecture
- [Task Breakdown](specs/001-goal-create-spec/tasks.md) - Detailed implementation tasks (53/53 complete)

### Spec 002: C++ MCTS Simulation Runner (🔧 In Progress)
**Goal**: Close 122-163× performance gap by completing C++ simulation runner

**Documentation**:
- [Specification](specs/002-cpp-simulation-runner/spec.md) - User stories, success metrics, requirements
- [Implementation Plan](specs/002-cpp-simulation-runner/plan.md) - 5-phase timeline with dependencies
- [Task Breakdown](specs/002-cpp-simulation-runner/tasks.md) - 23 detailed tasks (T001-T023)
- [Quickstart Guide](specs/002-cpp-simulation-runner/quickstart.md) - Build, test, troubleshooting
- [Migration Guide](specs/002-cpp-simulation-runner/MIGRATION.md) - Python → C++ migration with rollback
- [Python Fixes Required](specs/002-cpp-simulation-runner/PYTHON_FIXES_REQUIRED.md) - Comprehensive issue analysis

**Timeline**: 5 days (4 implementation + 1 buffer)

**Phases**:
1. **Phase 0** (0.5 day): Python training fixes - unblock execution
2. **Phase 1** (1 day): Build wiring & move storage implementation
3. **Phase 2** (1.5 days): C++ runner core (select → expand → backup)
4. **Phase 3** (1 day): Python integration & inference bridge
5. **Phase 4** (1 day): Testing & performance validation (≥30k sims/sec)
6. **Phase 5** (0.5 day): Documentation & evidence collection

## Performance Targets and Current Status

### Current Performance (Spec 004 Phase 2 Validated 2025-10-13)

| Metric | Current | Target | Progress | Status |
|--------|---------|--------|----------|--------|
| **MCTS Throughput** | 2,659 sims/sec | 8,000 sims/sec | 33.2% | 🟢 Ready (T018 state pooling) |
| **GPU Utilization** | Adequate | 80% | N/A | ✅ Not the bottleneck (2.1% of time) |
| **Thread Efficiency (8 threads)** | 12.7% | ≥60% | 21.2% | 🟡 Improves with T018 |
| **Tree Memory** | 270MB (10M nodes) | <1GB | ✅ 100% | ✅ Complete |
| **Node Allocation** | 330M allocs/sec | O(1) operations | ✅ 100% | ✅ Complete |
| **PUCT Selection** | 3.6-5.2× speedup | Vectorized | ✅ 100% | ✅ Complete |
| **Thread Safety** | TSan clean | No race conditions | ✅ 100% | ✅ Complete |
| **FP16 Mixed Precision** | 1.72× speedup | ≥1.5× | ✅ 100% | ✅ Complete (T-VALID-1) |
| **State Cloning** | 418μs per clone | ~20μs per clone | 4.8% | 🔴 PRIMARY BOTTLENECK (86.6% of time) |

**Key Finding** (from profiling 2025-10-16): State cloning consumes 86.6% of execution time due to 223 allocations per clone (~2μs each = 446μs overhead). Thread-local state pools (T018) eliminate this overhead, achieving 9,838 sims/sec target.

### Optimization Roadmap

**Completed Optimizations** (Spec 004):
- ✅ WU-UCT Virtual Loss (prevents Q-value distortion)
- ✅ Busy-Edge Masking (reduces thread collisions)
- ✅ DLPack Zero-Copy Bridge (eliminated Python list conversion)
- ✅ FP16 Mixed Precision (10.4× GPU speedup: 180ms → 17.3ms per batch)
- ✅ Thread-Local Memory Arenas (330M allocations/sec)
- ✅ Selection Prefetching (cache optimization)
- ✅ Batch/Timeout Optimization (batch=64, timeout=1.0ms optimal)
- ✅ Comprehensive Profiling (identified MCTS coordination as bottleneck)

**Remaining Path to 8,000 sims/sec** (see [Validation Report](docs/performance/validation_report_2025-10-13.md)):
- 🔴 **Critical Fix** (Est: 2-3k sims/sec): Fix OpenMP parallelization at dlpack_bridge.cpp:431 (+30-40%)
- 🔴 **T017 Baseline Investigation** (2 days): Determine root cause of 3,831 → 2,147 regression
- **T016 Benchmarking** (2 days): Comprehensive performance measurement with all optimizations
- **T018-T019 Parameter Tuning** (2 days): Optimize threads, batch size, timeout, virtual loss
- **Target Range**: 6,000-10,000 sims/sec (minimum 6k, realistic 8k, stretch 10k)

### Performance Characteristics

**Thread Scaling** (profiling-validated 2025-10-16):
```
Threads | Throughput    | Efficiency | Status
--------|---------------|------------|----------------
8       | 2,659 sims/s  | 12.7%      | Current baseline (profiling campaign)
8 (T018)| 9,838 sims/s  | ≥60%       | After state pooling (target)
4 (old) | 2,147 sims/s  | 45%        | Historical measurement (outdated)
```

**Note**: Historical thread scaling measurements predated profiling campaign. Current baseline is 2,659 sims/sec @ 8 threads (12.7% efficiency). After T018 (state pooling), expect ≥60% efficiency @ 8 threads.

**GPU Batch Optimization** (profiling-validated 2025-10-16):
```
Batch Size | GPU Inference Time | Impact on Throughput
-----------|-------------------|----------------------
64         | 20.66ms (2.1%)    | Optimal (profiling-validated)
```

**Note**: GPU inference is NOT the bottleneck (only 2.1% of execution time). Batch size optimization is secondary to state pooling (T018).

**Recommendation**: Use batch size 32-64 with timeout 1.0ms for optimal balance.

### System Capabilities

- **Simulations/sec**: 2,147 (current, regression) → 8,000 (realistic target, hardware-grounded)
- **GPU utilization**: 55% (current) → 80% (target when tensor creation fixed)
- **Batch efficiency**: batch=64, timeout=1.0ms optimal ✅
- **Tree memory**: 270MB for 10M nodes (<1GB target) ✅
- **Node allocation**: O(1) with 330M allocations/second ✅
- **PUCT selection**: 3.6-5.2× speedup with AVX2 vectorization ✅
- **Memory transfers**: Optimized H2D/D2H with pinned CUDA memory ✅
- **Reliability**: Automatic CPU fallback with seamless GPU failure handling ✅
- **Training speed**: 150-200 self-play games per hour (current) → 200-300 (target)
- **Games supported**: Gomoku, Chess (including Chess960), Go (9x9 to 19x19) ✅
- **Self-Play Quality**: Comprehensive testing validates training data integrity ✅
  - Move bias detection with statistical significance testing
  - Policy entropy monitoring (exploration→exploitation balance)
  - Terminal detection accuracy across all game variations
  - Temperature scheduling effectiveness validation
- **Tensor representations**: Enhanced feature planes for stronger tactical play ✅
  - Gomoku: 36 planes (threat detection, run-length analysis, rule variations)
  - Chess: 30 planes (castling, en passant, 8-pair move history)
  - Go: 25 planes (proper move history separation, capture patterns)

### Performance Documentation

- **Comprehensive Analysis**: [Throughput Analysis](docs/performance/throughput_analysis.md)
- **Profiling Reports**: [T020 Bottleneck Analysis](docs/performance/T020_bottleneck_profiling_report.md)
- **Optimization Results**: [Performance Session Summary](docs/performance/performance_optimization_session_summary.md)
- **GPU Analysis**: [GPU Bottleneck Analysis](docs/performance/gpu_bottleneck_analysis.md)

## Development Philosophy

- **Simplicity**: Write straightforward, readable code
- **Performance**: CPU/GPU optimized without sacrificing maintainability
- **Testability**: Comprehensive test coverage with contract-driven development
- **Spec-Driven**: All changes must reflect specification updates

## License

[License details to be added]