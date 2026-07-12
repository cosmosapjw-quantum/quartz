# Quickstart: C++ MCTS Simulation Runner
**Spec ID**: 002-cpp-simulation-runner
**Status**: Implementation Guide (2025-10-02)
**Target**: Close 122-163× performance gap (246 → 30,000-40,000 sims/sec)

This quickstart walks through building, testing, and validating the C++ MCTS simulation runner implementation. It assumes you are working inside the repository root (`/home/cosmosapjw/omoknuni`).

**Prerequisites**: Review `PYTHON_FIXES_REQUIRED.md` for comprehensive analysis of all 18 critical/major issues across 11 files.

---

## 1. Environment Setup

```bash
# 1.1 Activate virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 1.2 Install dependencies (dev extras required for formatting/tests)
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .[dev] --config-settings build-dir=build

# 1.3 Confirm extensions load
python -c "import mcts_py, alphazero_py; print('✓ extensions ready')"
```

> **Tip**: export `CFLAGS`/`CXXFLAGS` (e.g. `-O3 -march=znver3 -fopenmp`) before running `pip install -e .` to match target hardware.

---

## 2. Baseline Performance Metrics

**Current State** (Python simulation loop):
- **Throughput**: 246 sims/sec (target: 30,000-40,000 sims/sec)
- **GIL hold time**: 800µs/sim = 80% wall time (target: <10%)
- **Thread efficiency**: 3% scaling 1→8 threads (target: ≥75%)
- **GPU batch size**: 1-2 positions (target: 32-64)
- **GPU utilization**: <5% (target: 80-92%)
- **Move storage memory**: 1000MB for 10M nodes (target: 20MB)
- **Thread count**: 32-256 threads (oversubscribed, target: 8-12)

Run these commands to measure your baseline:

```bash
# Throughput baseline
python -m pytest tests/performance/test_benchmarks.py::test_mcts_throughput -v

# Memory baseline
python scripts/check_memory_footprint.py --nodes 10000000

# GPU utilization (if available)
nvidia-smi dmon -c 10 &  # monitor in background
python scripts/test_mcts.py --simulations 1000
```

---

## 3. Initial Verification (Pre-Implementation)

Before changing code, run the current suites to establish test status.

```bash
# Contract + unit + integration (legacy Python runner still active)
python -m pytest tests/contract -k "simulation_runner" -vv || true  # currently failing stubs
python -m pytest tests/unit -vv
python -m pytest tests/integration -vv

# C++ tests (gtest)
cmake -S . -B build && cmake --build build --target run_unit_tests || true  # will fail until runner implemented
```

At this stage, the new contract tests you add (see `tasks.md`) should fail, signalling that the implementation work is outstanding.

---

## 4. Phase 0: Python Training Fixes (CRITICAL - 0.5 day)

**Purpose**: Unblock training pipeline execution (currently crashes on first batch)

```bash
# T001: Fix policy loss function
# Edit: src/training/trainer.py:601
# Replace F.cross_entropy → F.kl_div

# T002: Add TrainingConfig fields
# Edit: src/training/training_loop.py:47-94
# Add: mcts_threads, batch_size_min/max, inference_timeout_ms

# T003: Fix config factory function
# Edit: src/training/training_loop.py:789-840

# T004: Guard signal handlers
# Edit: src/training/training_loop.py:162-164

# T005: Smoke test
python -m pytest tests/integration/test_training_pipeline.py::test_training_initialization -v
```

**Validation**: Training loop executes first epoch without crashes.

---

## 5. Phase 1-3: C++ Runner Implementation (3.5 days)

See `tasks.md` for detailed steps. High-level sequence:

1. **Build & Move Storage** (T006-T008):
   - Add `simulation_runner.cpp` to CMake, enable sanitizers
   - Implement `MCTSTree::moves_` array with get/set accessors
   - Add contract tests (failing until implementation)

2. **C++ Runner Core** (T009-T012):
   - Implement `select_leaf()`, `expand_node()`, `backup_value()`
   - Connect pipeline in `run_simulation()`
   - Write gtests with deterministic fixtures

3. **Python Integration** (T013-T016):
   - Create `PyInferenceCallback` bridge (C++→Python inference)
   - Refactor `AlphaZeroMCTS` with `use_cpp_runner=True` default
   - Fix `SearchCoordinator` duplicate `stop()`, connect GPU worker
   - Create `CppInferenceBridge` for batching

Work in small commits; after each task run targeted tests from `tasks.md`.

---

## 6. Phase 4: Testing & Performance Validation (1 day)

Once the runner path is wired in, execute the full test matrix:

```bash
# Python suites
python -m pytest tests/contract -vv
python -m pytest tests/unit -vv
python -m pytest tests/integration -vv
python -m pytest tests/performance -vv
python -m pytest tests/soak -k "quick" -vv   # short variant for local runs

# C++ unit tests
cmake --build build --target run_unit_tests

# Sanitizers (MANDATORY - will be enforced in CI)
python scripts/build_with_sanitizers.py --sanitizer asan --clean && cmake --build build
ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure

python scripts/build_with_sanitizers.py --sanitizer tsan --clean && cmake --build build
TSAN_OPTIONS="suppressions=.tsan-suppressions" ctest --output-on-failure
```

**Performance targets** (from `tasks.md` T017):
- ≥30k sims/sec on 8 CPU threads
- ≥75% thread efficiency (1→8 threads)
- GPU batch size 32-64, utilization 80-92%
- <10MB leak after 1-hour soak

Address any regressions before proceeding.

---

## 7. Docker Testing (Optional but Recommended)

Mirror CI environment locally with Docker:

```bash
# Build development image
./scripts/docker/build.sh -t development

# Run test suite in container
docker-compose run --rm dev python -m pytest tests/contract tests/unit tests/integration -vv

# Run performance benchmarks in container
docker-compose run --rm benchmark python -m pytest tests/performance -vv

# Soak test (1-hour variant)
docker-compose run --rm dev python -m pytest tests/soak -vv
```

**Why Docker?**
- Eliminates "works on my machine" issues
- Matches CI build environment exactly
- Validates sanitizer configs in clean environment

---

## 8. Profiling & Evidence Collection

Use these scripts to capture throughput and GIL statistics required by the spec:

```bash
# Measure sims/sec and batching
python tests/performance/test_simulation_runner_performance.py -k "throughput" -vv

# Profile GIL contention (requires python -X gil)
python -X perf -m pytest tests/performance/test_simulation_runner_performance.py -k "throughput"

# Optionally capture flamegraph for C++ runner
tools/perf_record_runner.sh  # custom helper; generates flamegraph in docs/performance/runner
```

Archive the resulting JSON/plots under `docs/performance/runner/` and reference them in the implementation PR.

---

## 9. Phase 5: Documentation & Cleanup (0.5 day)

After tests pass and profiling targets are met (T021-T023):

1. Update `mcts_guide.md`, `docs/performance/*`, and `CLAUDE.md` with new architecture
2. Refresh `AGENTS.md` and verify spec/plan/tasks in sync with code
3. Mark PYTHON_FIXES_REQUIRED.md issues as complete
4. Run formatters and linters

```bash
black src tests scripts
isort src tests scripts
flake8 src tests scripts
mypy src
```

Finally, regenerate stubs or caches (`rm -rf build && pip install -e .` for clean rebuild) and push your branch once CI passes.

---

## 10. Troubleshooting

### Issue: Training crashes with `RuntimeError: expected scalar type Long but got Float`
**Cause**: Policy loss uses `F.cross_entropy` with float targets (T001)
**Fix**: Change to `F.kl_div(F.log_softmax(policy_pred, dim=1), policy_target, reduction='batchmean')`
**File**: `src/training/trainer.py:601`

### Issue: `AttributeError: 'TrainingConfig' object has no attribute 'mcts_threads'`
**Cause**: Missing config fields (T002)
**Fix**: Add `mcts_threads`, `batch_size_min/max`, `inference_timeout_ms` to `TrainingConfig` dataclass
**File**: `src/training/training_loop.py:47-94`

### Issue: `TypeError` when creating `TrainingConfig` from YAML
**Cause**: Config factory passes unknown kwargs (T003)
**Fix**: Split dict flattening into sections, pass separately
**File**: `src/training/training_loop.py:789-840`

### Issue: `ValueError: signal only works in main thread` in tests
**Cause**: Signal handlers registered unconditionally (T004)
**Fix**: Guard with `if threading.current_thread() is threading.main_thread():`
**File**: `src/training/training_loop.py:162-164`

### Issue: C++ runner not found / `NotImplementedError`
**Cause**: Build wiring incomplete (T006) or implementation not done (T009-T012)
**Fix**: Add `simulation_runner.cpp` to CMakeLists, rebuild with `pip install -e . --force-reinstall`

### Issue: Move storage memory still 1000MB for 10M nodes
**Cause**: Python dict `_move_mapping` still in use (T014)
**Fix**: Delete `_move_mapping` (lines 136,169,518,565-566 in `mcts.py`), use `tree.get_move()`

### Issue: SearchCoordinator threads never terminate
**Cause**: Duplicate `stop()` method shadows first definition (T015)
**Fix**: Delete second `stop()` at line 549, consolidate shutdown in first definition

### Issue: GPU utilization still <5%
**Cause**: Dummy inference active (T015) or batching not working (T016)
**Fix**: Connect `GPUInferenceWorker` in `SearchCoordinator`, verify `CppInferenceBridge` batching

### Issue: Performance test fails with <30k sims/sec
**Possible causes**:
1. **Python loop still active**: Check `use_cpp_runner=True` in config
2. **GIL not released**: Verify `py::call_guard<py::gil_scoped_release>` in bindings (T013)
3. **Thread oversubscription**: Ensure no nested `ThreadPoolExecutor` (T014, T015)
4. **Inference blocking**: Check `CppInferenceBridge` releases GIL while waiting (T016)

**Debug commands**:
```bash
# Profile GIL contention
python -X perf -m pytest tests/performance/test_simulation_runner_performance.py -v

# Check thread count
ps -eLf | grep python | wc -l  # should be ≤12 during search, not 32-256

# Verify C++ runner active
python -c "from src.core.mcts import AlphaZeroMCTS; m = AlphaZeroMCTS(...); print(m.use_cpp_runner)"  # True
```

### Issue: ASan/TSan failures
**Common causes**:
1. **Move storage leak**: Verify allocation/deallocation in `tree.cpp` (T008)
2. **Virtual loss not removed**: Check `VirtualLossGuard` RAII destructor (T011)
3. **Data race in selection**: Run under TSan, check atomic operations (T019)

**Fix**: Address specific sanitizer reports, add gtests for edge cases

### Issue: Docker build fails
**Cause**: Missing build context or outdated base image
**Fix**:
```bash
# Rebuild from scratch
docker-compose build --no-cache dev
docker-compose build --no-cache benchmark
```

---

## 11. Reference Commands

### Build & Test
- Rebuild after C++ changes: `pip install -e . --force-reinstall --no-deps`
- Gtest runner: `ctest --output-on-failure --parallel 8`
- Run specific test: `python -m pytest tests/unit/test_mcts.py::test_search_cpp_mode -vv`

### Docker
- Development shell: `docker-compose run --rm dev bash`
- Performance benchmarks: `docker-compose run --rm benchmark python -m pytest tests/performance -vv`
- Clean rebuild: `docker-compose down -v && docker-compose build --no-cache`

### Performance
- Quick throughput check: `python scripts/test_mcts.py --simulations 800 --threads 8 --game gomoku`
- Detailed profiling: `python -m cProfile -o profile.out scripts/test_mcts.py && snakeviz profile.out`
- GIL analysis: `python -X gil_interval=0.001 scripts/test_mcts.py`

Follow the phases sequentially (0→1→2→3→4→5) and keep spec/plan/tasks in sync as you land each milestone.
