# Tasks: MCTS Throughput Optimization

**Input**: Design documents from `/home/cosmosapjw/omoknuni/specs/005-mcts-throughput-optimization/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Included as part of each phase's Definition of Done (DoD)

**Organization**: Tasks grouped by user story (US1-US4) to enable independent implementation and validation.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase Numbering Mapping

**tasks.md Organization** (by user story for independent delivery):
- **Phase 1 (Setup)**: Environment initialization (T001-T005)
- **Phase 2 (Foundational)**: Shared infrastructure (T006-T010)
- **Phase 3 (US1 - MVP)**: State cloning elimination → 1,500-3,000 sims/sec (T011-T035)
- **Phase 4 (US2 - TARGET)**: Tensor + OpenMP optimization → 7,000-9,000 sims/sec (T036-T056) ✅
- **Phase 5 (US3 - STRETCH)**: Multi-coordinator → 12,000-20,000 sims/sec (T057-T066)
- **Phase 6 (US4 - OPTIONAL)**: Multi-process → 20,000-35,000 sims/sec (T067-T069)

**plan.md / spec.md Organization** (by optimization type):
- **Phase 1**: Zero-copy via in-place extraction (maps to tasks.md Phase 3 = US1)
- **Phase 2**: Tensor pipeline + OpenMP (maps to tasks.md Phase 4 = US2)
- **Phase 3A**: Multi-coordinator (maps to tasks.md Phase 5 = US3)
- **Phase 3B**: Multi-process (maps to tasks.md Phase 6 = US4)

**Why Different?**
- tasks.md groups by user story for parallel implementation and independent testing
- plan.md groups by optimization technique for technical clarity and sequential reasoning
- Both are valid organizational schemes; use this mapping to translate between them

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, profiling infrastructure, and baseline validation

- [X] T001 Verify development environment (Python 3.10-3.12 (tested), PyTorch 2.0+, CUDA 11.8+, g++ 9.0+, OpenMP available)
- [X] T002 Run baseline profiling campaign (100+ trials) to establish 120 sims/sec baseline per scripts/profiling/run_campaign.py
- [X] T003 [P] Create audit script scripts/audit_state_cloning.sh to grep for clone()/copy()/new State() in hot paths
- [X] T004 [P] Create verification script scripts/verify_openmp.sh to check ldd output for libomp.so linkage
- [X] T005 [P] Create validation script scripts/validate_all_phases.sh with automated rollback procedures

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure for profiling and contract validation

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 [P] Extend ProfilingMetrics struct in cpp_extensions/mcts/instrumentation.hpp with Phase 1-2 metrics (state_cloning_us, tensor_creation_ms, openmp_thread_count, etc.)
- [X] T007 [P] Implement profiling recorder methods in cpp_extensions/mcts/instrumentation.cpp (record_state_cloning, record_feature_extraction, record_tensor_creation)
- [X] T008 [P] Create profiling analysis script scripts/profiling/analyze_campaign.py with phase-specific metric extraction (analyze_phase1, analyze_phase2)
- [X] T009 Create benchmark harness scripts/benchmark_phase1.py for Phase 1 validation (1.5k-3k sims/sec target)
- [X] T010 Create benchmark harness scripts/benchmark_phase2.py for Phase 2 validation (7k-9k sims/sec target)

**Checkpoint**: ✅ Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Eliminate State Cloning Bottleneck (Priority: P1) 🎯 MVP

**Goal**: Achieve 1,500-3,000 sims/sec (10-25× baseline) by eliminating state cloning (86.6% bottleneck) via in-place feature extraction and move semantics

**Independent Test**: Run scripts/benchmark_phase1.py --trials 100 and verify profiling shows <1% time in state cloning, throughput 1,500-3,000 sims/sec

**Success Criteria** (SC-001 to SC-004):
- Throughput: 1,500-3,000 sims/sec
- State cloning overhead: <1% of total execution time
- Zero `clone()` calls in simulation hot path
- Zero memory allocations in hot path

### Phase 1A: Thread-Local Feature Buffer

- [X] T011 [US1] Add feature_buffer field to ThreadLocalState struct in cpp_extensions/mcts/continuous_simulation_runner.hpp (std::vector<float> feature_buffer, bool feature_buffer_initialized)
- [X] T012 [US1] Implement initialize_feature_buffer() method in cpp_extensions/mcts/continuous_simulation_runner.cpp to allocate 36×19×19 floats (52KB) once per thread
- [X] T013 [US1] Replace state cloning with in-place extraction in continuous_simulation_runner.cpp (call game->extract_features_to_buffer(current_state, tls.feature_buffer.data()) at leaf node)
- [X] T014 [US1] Build InferenceRequest with std::move(tls.feature_buffer) in continuous_simulation_runner.cpp to transfer ownership via move semantics

**DoD**: Unit test in tests/unit/test_feature_extraction.py verifies in-place extraction produces identical features to copy-based; micro-benchmark shows extract_features_to_buffer() cost <50μs per call

### Phase 1B: AsyncInferenceQueue API Shift

- [X] T015 [US1] Modify InferenceRequest struct in cpp_extensions/mcts/async_inference_queue.hpp to move-only semantics (delete copy constructor/assignment, default move constructor/assignment)
- [X] T016 [US1] Add fields to InferenceRequest in async_inference_queue.hpp (std::vector<float> features, int32_t node_index, int32_t action_space_size, int16_t board_size, int16_t planes, std::vector<int16_t> path, uint64_t request_id)
- [X] T017 [US1] Change submit_request() signature in async_inference_queue.hpp to accept InferenceRequest&& (rvalue reference)
- [X] T018 [US1] Implement submit_request() in async_inference_queue.cpp to move request into queue (requests_.push_back(std::move(request)))
- [X] T019 [US1] Add std::condition_variable cv_request_ready_ to AsyncInferenceQueue in async_inference_queue.hpp for coordinator wake-up
- [X] T020 [US1] Call cv_request_ready_.notify_one() in submit_request() after enqueue in async_inference_queue.cpp

**DoD**: Compile succeeds; InferenceRequest contains features only; zero state cloning in hot path verified by instrumentation asserts; contract test in tests/contract/test_async_queue_api.py verifies move semantics via:
1. Assert InferenceRequest has **deleted copy constructor**: `InferenceRequest(const InferenceRequest&) = delete;`
2. Assert **move constructor enabled**: `InferenceRequest(InferenceRequest&&) = default;`
3. Verify **move leaves source empty**: after `req2 = std::move(req1);`, assert `req1.features.empty() == true;`
4. Performance test: moving 10,000 InferenceRequest objects completes in <1ms (vs ~500ms if copying)

### Phase 1C: Coordinator Simplification

- [X] T021 [US1] Remove feature extraction logic from BatchInferenceCoordinator::form_batch() in cpp_extensions/mcts/batch_inference_coordinator.cpp
- [X] T022 [US1] Remove state cloning logic from BatchInferenceCoordinator in batch_inference_coordinator.cpp
- [X] T023 [US1] Collect and move features only in form_batch() in batch_inference_coordinator.cpp (aggregate pre-extracted features from requests)
- [X] T024 [US1] Pre-reserve batch vectors in BatchInferenceCoordinator::form_batch() (use reserve() not resize() to avoid creating empty elements)
- [X] T025 [US1] Condition variables already implemented in AsyncInferenceQueue (notify_all() on submit, wait in collect_batch)

**DoD**: Allocation counters stable (zero allocations in hot path); batch assembly completes in <0.3ms measured via profiling; contract test in tests/contract/test_coordinator_api.py verifies zero allocation in batching

### Phase 1D: Optional Fallback (Path Reconstruction)

- [ ] T026 [P] [US1] Implement reconstruct_state_from_path() function in cpp_extensions/mcts/continuous_simulation_runner.cpp for validation/debugging (apply make_move sequence)
- [ ] T027 [P] [US1] Add validation test in tests/unit/test_feature_extraction.py comparing direct extraction vs reconstruction (features should be identical)

**DoD**: Test verifies direct extraction and reconstruction produce identical features

### Phase 1 Enhancements: Condition Variables & Virtual-Loss Restart

- [ ] T028 [P] [US1] Add std::condition_variable cv_results_ready_ to AsyncInferenceQueue in async_inference_queue.hpp for result notification
- [ ] T029 [P] [US1] Implement notify_results_ready() and wait_for_result() methods in async_inference_queue.cpp
- [ ] T030 [P] [US1] Modify select_leaf() in cpp_extensions/mcts/selection.cpp to restart immediately on virtual-loss contention (remove sleep, use goto restart pattern)

**DoD**: Contention test in tests/integration/test_phase1_integration.py shows no sleeps; conflicts reselect immediately; CPU usage during idle <10% (down from 100%)

### Phase 1 Validation & Profiling

- [ ] T031 [US1] Run scripts/audit_state_cloning.sh and verify zero clone()/copy() calls in continuous_simulation_runner.cpp, async_inference_queue.cpp, batch_inference_coordinator.cpp
- [ ] T032 [US1] Run profiling campaign via python scripts/benchmark_phase1.py --trials 100 --output profiling_phase1_$(date +%Y%m%d)
- [ ] T033 [US1] Analyze results via python scripts/profiling/analyze_campaign.py profiling_phase1_*/ --compare-to-baseline and verify SC-001 to SC-004 acceptance criteria
- [ ] T034 [US1] Commit profiling results to docs/performance/phase_1_results.md with throughput metrics and bottleneck analysis
- [ ] T035 [US1] If targets missed (throughput <1,500 sims/sec OR state_cloning >1%), execute rollback via git revert HEAD~10..HEAD and investigate root cause

**Checkpoint**: User Story 1 should deliver 1,500-3,000 sims/sec with <1% state cloning overhead

---

## Phase 4: User Story 2 - Fix Tensor Pipeline and OpenMP Parallelization (Priority: P2) ✅ TARGET

**Goal**: Achieve 7,000-9,000 sims/sec (58-75× baseline, PRIMARY TARGET) by fixing broken OpenMP (0% → >95% success) and tensor copy overhead (37ms → ≤2ms per batch)

**Independent Test**: Run scripts/benchmark_phase2.py --trials 100 and verify (1) OpenMP thread count >1 in logs, (2) tensor preparation time ≤2ms per batch, (3) GPU utilization ≥80%, (4) throughput ≥7,000 sims/sec

**Success Criteria** (SC-005 to SC-009):
- Throughput: 7,000-9,000 sims/sec (PRIMARY GOAL ✅)
- OpenMP thread count >1 in execution logs
- **Python-side overhead** (tensor build + H2D) ≤2.0ms per batch at p95 (down from 37ms baseline); **GPU kernel time** model-dependent, gated against baseline + 20%
- GPU utilization ≥80% during search
- Feature buffer allocations: zero per iteration (pre-allocated pools only)

### Phase 2A: OpenMP Linking Fix

- [X] T036 [US2] OpenMP already linked in CMakeLists.txt (OpenMP::OpenMP_CXX in target_link_libraries at line 167)
- [X] T037 [US2] Rebuild C++ extensions via pip install -e . --force-reinstall --no-deps --config-settings build-dir=build
- [X] T038 [US2] Verify OpenMP linked via ldd (libgomp.so.1 confirmed present)
- [X] T039 [US2] Add runtime OpenMP thread count reporting in python_bindings.cpp (get_openmp_threads() and get_openmp_enabled())
- [X] T040 [US2] Create CI check scripts/verify_openmp.sh (24 threads confirmed available)

**DoD**: Runtime gauge shows omp_threads>1 in coordinator logs; OpenMP parallel region verified in feature extraction; contract test in tests/contract/test_openmp.py verifies thread count >1

### Phase 2B: Pinned Memory Tensor Pipeline

- [X] T041 [US2] DLPackInferenceBridge class already exists in src/core/dlpack_inference_bridge.py
- [X] T042 [US2] Pre-allocate pinned CPU buffer with lazy init (64×36×19×19, pin_memory=True, fallback for overflow)
- [X] T043 [US2] Pre-allocate GPU buffer with lazy init (64×36×19×19, device='cuda')
- [X] T044 [US2] CUDA stream pool already exists (stream_pool with 2 streams)
- [X] T045 [US2] Implemented batch tensor creation in batch_inference_features() (copies to pinned buffer)
- [X] T046 [US2] Non-blocking GPU transfer implemented (copy_ with non_blocking=True in stream context)
- [X] T047 [US2] Buffer reuse assertion added (assert pinned_buffer.is_pinned())

**DoD**: Batch CPU prep completes in ≤2ms (measured via profiling); no repeated allocations (same buffer address across batches); GPU stream semantics validated; unit test in tests/unit/test_pinned_buffers.py verifies pinned buffer never reallocates

### Phase 2C: Coordinator Integration & Throughput Optimization

- [ ] T048 [US2] Integrate DLPackInferenceBridge into BatchInferenceCoordinator in batch_inference_coordinator.cpp (call Python bridge via callback)
- [ ] T049 [US2] Add micro-batch timeout configuration to form_batch() in batch_inference_coordinator.cpp (0.5-1.0ms timeout tuned via profiling)
- [ ] T050 [US2] Add CUDA stream ID storage to BatchInferenceCoordinator in batch_inference_coordinator.cpp for Phase 3A multi-stream preparation
- [ ] T051 [US2] Tune batch_size parameter **POST-Phase 2** via scripts/tune_batch_size.py --game gomoku --range 32,64,128 --iterations 100 (baseline profiling favored 128 with old 37ms overhead; new <2ms overhead changes tradeoff, expect 64 optimal; target GPU utilization ≥80%)

**DoD**: End-to-end batch latency <1ms per batch; GPU utilization ≥80% measured via nvidia-smi; micro-benchmark shows form_batch() completes in <0.3ms

### Phase 2 Validation & Profiling

- [ ] T052 [US2] Run scripts/verify_openmp.sh and verify OpenMP linked successfully (exit code 0)
- [ ] T053 [US2] Run profiling campaign via python scripts/benchmark_phase2.py --trials 100 --output profiling_phase2_$(date +%Y%m%d)
- [ ] T054 [US2] Analyze results via python scripts/profiling/analyze_campaign.py profiling_phase2_*/ --compare-to-baseline --compare-to-phase1 and verify SC-005 to SC-009 acceptance criteria
- [ ] T055 [US2] Commit profiling results to docs/performance/phase_2_results.md with throughput metrics and GPU utilization graphs
- [ ] T056 [US2] If targets missed (throughput <7,000 sims/sec OR tensor_creation_ms >2.0 OR openmp_thread_count ≤1), execute rollback via git revert HEAD~15..HEAD and investigate root cause

**Checkpoint**: User Story 2 should deliver 7,000-9,000 sims/sec with ≥80% GPU utilization (PRIMARY TARGET ✅)

---

## Phase 5: User Story 3 - Parallel Coordinators for High Throughput (Priority: P3) 🎯 STRETCH

**Goal**: Achieve 12,000-20,000 sims/sec (100-166× baseline, STRETCH GOAL) by eliminating coordinator serialization (99.6% → <10%) via K parallel coordinators (**default K=3** on RTX 3060 Ti, **auto-tuned** at startup from {1,2,3,4}, CLI override `--coordinators K`) with K-stream GPU inference (one stream per coordinator)

**Independent Test**: Run benchmark with auto-tuned K coordinators and multi-stream GPU inference; verify coordinator blocking <10% and throughput ≥12,000 sims/sec

**Success Criteria** (SC-010 to SC-012):
- Throughput: 12,000-20,000 sims/sec
- Coordinator blocking time <10% of iteration time (down from 99.6%)
- Linear-ish scaling: K coordinators → (K × 0.8 to K × 0.95)× throughput vs 1 coordinator (accounts for GIL contention when PyTorch re-acquires GIL for Python callbacks, queue synchronization overhead; actual scaling validated via auto-tuner benchmark, not assumed)

**⚠️ ONLY IMPLEMENT IF**: Phase 2 meets 7k-9k target but stretch goal of 12k+ is desired

### Phase 3A Implementation

- [ ] T057 [US3] Create MultiCoordinatorManager class in src/core/search_coordinator.py
- [ ] T058 [US3] Implement multi-coordinator initialization in MultiCoordinatorManager.__init__() (spawn K coordinator threads where K=auto-tuned value or CLI override; **default K=3** on RTX 3060 Ti; each coordinator gets dedicated CUDA stream via torch.cuda.Stream(); load K from `~/.mcts_autotune.json` if exists, else use default)
- [ ] T059 [US3] Implement _run_coordinator() method in search_coordinator.py to run coordinator loop with stream isolation
- [ ] T059a [US3] Implement coordinator count auto-tuner: (1) Create `scripts/bench_autotune_coordinators.py` that runs 3-5s micro-benchmark (100 simulations × K coordinators for K∈{1,2,3,4}), (2) Measure p95 sims/sec for each K, (3) Select K with best throughput, (4) Persist to `~/.mcts_autotune.json` as {"gpu_model": "GA104", "optimal_coordinators": K, "measured_throughput": X, "timestamp": "..."}, (5) Validate stability: run tuner twice, assert selected K matches or differs by ≤1 (prevents thrashing), (6) Add CI check: tuner completes in <10s and selects valid K∈{1,2,3,4}
- [ ] T060 [US3] Add backpressure mechanism to AsyncInferenceQueue::submit_request_with_backpressure() in async_inference_queue.cpp (block when queue full, wake on dequeue)
- [ ] T061 [US3] Add cv_space_available_ condition variable to AsyncInferenceQueue in async_inference_queue.hpp for backpressure signaling
- [ ] T062 [US3] Implement notify_dequeued() method in async_inference_queue.cpp to wake waiting simulation threads after batch dequeue

**DoD**: Linear-ish gain up to concurrency limit (4 coordinators → 3.5× throughput); telemetry shows fair scheduling; integration test in tests/integration/test_phase3a_integration.py verifies multi-coordinator scaling

### Phase 3A Validation

- [ ] T063 [US3] Run profiling campaign via python scripts/benchmark_phase3a.py --coordinators auto --trials 100 --output profiling_phase3a_$(date +%Y%m%d) (uses auto-tuned K value; also run with --coordinators 1,2,3,4 to validate auto-tuner choice)
- [ ] T064 [US3] Analyze results and verify SC-010 to SC-012 acceptance criteria (throughput ≥12,000 sims/sec, coordinator blocking <10%)
- [ ] T065 [US3] Commit profiling results to docs/performance/phase_3a_results.md with coordinator utilization metrics
- [ ] T066 [US3] If targets met (throughput ≥12,000 sims/sec), mark Phase 3B as NOT NEEDED; otherwise proceed to decision gate

**Checkpoint**: User Story 3 should deliver 12,000-20,000 sims/sec (OPTIONAL STRETCH GOAL)

---

## Phase 6: User Story 4 - Multi-Process Architecture (Priority: P4) 🚀 OPTIONAL

**Goal**: Achieve 20,000-35,000 sims/sec (166-291× baseline) by bypassing GIL via multi-process architecture with shared-memory tensor handoff

**Independent Test**: Run multi-process benchmark with shared-memory tensor handoff; verify Python callback overhead <5ms and throughput ≥20,000 sims/sec

**Success Criteria** (SC-013 to SC-015):
- Throughput: 20,000-35,000 sims/sec
- Python callback overhead <5ms per batch
- GIL-related bottlenecks eliminated (measured via py-spy profiling)

**⚠️ ONLY IMPLEMENT IF**: Phase 3A < 12k sims/sec AND Python callback >5ms/batch AND target >25k sims/sec

**Decision Gate**: Before implementing Phase 3B, verify:
1. Phase 3A throughput <12,000 sims/sec
2. Profiling shows Python callback >5ms/batch
3. Profiling shows GIL >50% bottleneck
4. Business requirement for >25,000 sims/sec justified

### Phase 3B Implementation (Deferred - Document Only)

- [ ] T067 [US4] Document multi-process architecture design in specs/005-mcts-throughput-optimization/phase3b_design.md (shared memory layout, semaphore synchronization, process health monitoring)
- [ ] T068 [US4] Document rollback procedure for multi-process in specs/005-mcts-throughput-optimization/phase3b_rollback.md
- [ ] T069 [US4] Add feature flag ENABLE_MULTI_PROCESS=0 in cpp_extensions/mcts/continuous_simulation_runner.cpp for quick rollback

**DoD**: Effective speedup ~3.7-4.0× vs single process; only enabled behind feature flag; health-check and teardown procedures documented

**Checkpoint**: User Story 4 delivers 20,000-35,000 sims/sec (ONLY IF REQUIRED)

---

## Phase 7: Cross-Cutting Concerns & Instrumentation

**Purpose**: Profiling infrastructure, rollback mechanisms, and documentation

### Instrumentation

- [ ] T070 [P] Add state_cloning counter to ProfilingMetrics in cpp_extensions/mcts/instrumentation.hpp (uint64_t state_clone_count)
- [ ] T071 [P] Add allocation counter to ProfilingMetrics in instrumentation.hpp (uint64_t allocations_in_hot_path)
- [ ] T072 [P] Add OpenMP thread count gauge to ProfilingMetrics in instrumentation.hpp (int32_t openmp_thread_count, bool openmp_enabled)
- [ ] T073 [P] Add batch latency histogram to ProfilingMetrics in instrumentation.hpp (double batch_latency_ms, double python_callback_ms)
- [ ] T074 [P] Create profiling dashboard script scripts/profiling/generate_dashboard.py with phase-by-phase comparison charts
- [ ] T075 [P] Add telemetry export to JSON in scripts/profiling/analyze_campaign.py (write metrics to docs/performance/metrics_$(date +%Y%m%d).json)

**DoD**: Counters for clones, allocations, OpenMP threads, batch latency, and Python callback time all exposed; dashboards generated per scripts/profiling/generate_dashboard.py

### Rollback Switches & Validation

- [ ] T076 [P] Add feature flag ENABLE_INPLACE_EXTRACTION in cpp_extensions/mcts/continuous_simulation_runner.cpp (default: 1 for Phase 1+)
- [ ] T077 [P] Add feature flag ENABLE_PINNED_BUFFERS in src/core/dlpack_inference_bridge.py (default: 1 for Phase 2+)
- [ ] T078 [P] Add feature flag ENABLE_MULTI_COORDINATOR in src/core/search_coordinator.py (default: 0 until Phase 3A validated)
- [ ] T079 Implement validate_all_phases.sh pipeline in scripts/validate_all_phases.sh with automated phase validation and rollback
- [ ] T080 Add regression detection to validate_all_phases.sh (compare throughput to phase targets, flag >5% regressions, trigger rollback)
- [ ] T081 Document rollback procedure for each phase in specs/005-mcts-throughput-optimization/rollback_procedures.md

**DoD**: Per-phase feature flags enable quick rollback; validate_all_phases.sh executes baseline → Phase 1 → Phase 2 validation with automated rollback on regression

### Documentation & Polish

- [ ] T082 [P] Update quickstart.md with Phase 1 validation procedures (build, benchmark, profiling campaign)
- [ ] T083 [P] Update quickstart.md with Phase 2 validation procedures (OpenMP verification, tensor pipeline validation)
- [ ] T084 [P] Update CLAUDE.md with new optimization approaches (thread-local buffers, pinned memory, condition variables)
- [ ] T085 [P] Create performance comparison table in docs/performance/optimization_summary.md (baseline vs Phase 1 vs Phase 2)
- [ ] T086 [P] Add troubleshooting section to quickstart.md (low throughput, OpenMP not linked, tensor creation slow, GPU utilization low)
- [ ] T087 Run quickstart.md validation end-to-end to verify all build/benchmark/validation procedures work

**DoD**: Documentation complete; quickstart.md validated; troubleshooting guide tested

### Success Criteria Validation (SC-020, SC-021, SC-022)

- [ ] T100 [CONST] Test Phase 1 rollback procedure: (1) **Dry-run revert first**: `git revert --no-commit <Phase1-commits>`, (2) Run smoke test: `pytest tests/integration/test_phase1_integration.py -v`, (3) If smoke test passes, finalize revert; else abort and fix, (4) pip install -e . --force-reinstall, (5) Run baseline benchmark, verify throughput restores to 120 ± 6 sims/sec (5% variance), (6) Document procedure in plan.md "Rollback Procedures" section, (7) Create automated rollback script `scripts/rollback_phase1.sh` with dry-run mode
- [ ] T101 [CONST] Measure end-to-end batch latency variance: (1) Run 100 trials with fixed config (batch_size=64, threads=8, simulations=2000), (2) Extract `coordinator_python_callback` p50/p95/p99 from profiling JSONs, (3) Calculate coefficient of variation (CV = stddev/mean), (4) Assert CV < 5% for stable inference, (5) Plot CDF of latencies to identify outliers, (6) Commit variance report to docs/performance/latency_variance_analysis.md with CDF plot
- [ ] T102 [CONST] Validate PUCT semantics preservation: (1) **Baseline trace generation**: Run 1000 sims with seed=42, log all (node_id, move, Q, N, P) selections to baseline_trace.json before Phase 1 changes, (2) **Optimized trace**: Run same config (seed=42, 1000 sims) after Phase 1-3 optimizations, log to optimized_trace.json, (3) **Exact match on visit order**: Assert 100% match on (node_id, move_selected, visit_order), (4) **Floating-point tolerance on Q/N**: Allow |Q_opt - Q_base| < 1e-6, |N_opt - N_base| < 1 (integer match), (5) **Adversarial seed set**: Repeat with 10 additional seeds (43-52), assert policy-head **argmax equivalence** at root distribution (top-3 moves match within 1e-6 probability), (6) Implement test in tests/integration/test_puct_semantics_preserved.py with parametrized seeds
- [ ] T103 [CONST] Verify state pool removal per clarification Q4: (1) `grep -rn "state_pool" cpp_extensions/ src/` → assert 0 matches (excluding comments/docs), (2) Verify files deleted: assert `! -f cpp_extensions/mcts/state_pool.cpp` and `! -f cpp_extensions/mcts/state_pool.hpp`, (3) Verify no imports: `grep -rn "#include.*state_pool" cpp_extensions/` → assert 0 matches, (4) Add CI check: `scripts/audit_legacy_code.sh` fails if state_pool detected, runs on every PR, (5) Update constitution.md to document removal with amendment note

**DoD**: SC-020 validated (rollback script + dry-run); SC-021 validated (latency variance <5% with CDF plot); SC-022 validated (PUCT semantics preserved on 11 seeds with argmax equivalence); state pool removal confirmed (clarification Q4)

---

## Constitution Compliance & Validation

**Purpose**: Ensure implementation adheres to `.specify/memory/constitution.md`

### Code Review Checklist

- [ ] T088 Verify no state cloning in hot paths via scripts/audit_state_cloning.sh (Principle I - Zero-Copy First)
- [ ] T089 Verify feature buffers pre-allocated in ThreadLocalState initialization (Principle II - Coordinator Efficiency)
- [ ] T090 Verify OpenMP linkage via scripts/verify_openmp.sh and runtime thread count >1 (Principle IV - Threading Saturation)
- [ ] T091 Verify no modifications to legacy files (mcts_guide.md, simulation_runner.cpp/hpp) per git diff (Principle V - Legacy Code Discipline)
- [ ] T092 Verify thread safety via ThreadSanitizer build (atomic operations or thread-local storage only, zero data races)
- [ ] T093 Verify pinned memory reuse via assertion in DLPackInferenceBridge (Principle II - Coordinator Efficiency)

**DoD**: All constitution principles validated; zero violations detected

### Profiling & Performance Gates

- [ ] T094 Run Phase 1 profiling campaign (100+ trials) via scripts/benchmark_phase1.py and verify SC-001 to SC-004 acceptance (Principle VI - Evidence-Based Gates)
- [ ] T095 Run Phase 2 profiling campaign (100+ trials) via scripts/benchmark_phase2.py and verify SC-005 to SC-009 acceptance (Principle VI - Evidence-Based Gates)
- [ ] T096 Analyze Phase 1 results via scripts/profiling/analyze_campaign.py and verify throughput 1,500-3,000 sims/sec, state_cloning <1%
- [ ] T097 Analyze Phase 2 results via scripts/profiling/analyze_campaign.py and verify throughput 7,000-9,000 sims/sec, **python_side_overhead_ms** ≤2.0 (p95, tensor build + H2D), **gpu_kernel_ms** ≤ baseline + 20% (model-dependent), openmp_thread_count >1
- [ ] T098 Commit profiling results to docs/performance/phase_1_results.md and docs/performance/phase_2_results.md
- [ ] T099 Document rollback procedure in specs/005-mcts-throughput-optimization/rollback_procedures.md (restore previous performance in <1 hour)

**DoD**: Profiling campaigns complete; phase targets met or rollback executed; results committed to repository

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 / Phase 3 (P1 - MVP)**: Depends on Foundational phase completion
- **User Story 2 / Phase 4 (P2 - TARGET)**: Depends on User Story 1 completion (builds on zero-copy foundation)
- **User Story 3 / Phase 5 (P3 - STRETCH)**: Depends on User Story 2 completion, ONLY IF throughput target ≥12k sims/sec desired
- **User Story 4 / Phase 6 (P4 - OPTIONAL)**: Depends on User Story 3 completion AND decision gate conditions met (Phase 3A <12k AND Python callback >5ms AND GIL >50%)
- **Cross-Cutting (Phase 7)**: Can run in parallel with user stories (instrumentation), finalized after all phases complete
- **Constitution Compliance**: Runs after each user story phase for validation

### User Story Dependencies

- **User Story 1 (P1 - MVP)**: No dependencies on other stories (foundational zero-copy optimization)
- **User Story 2 (P2 - TARGET)**: Builds on User Story 1 (requires zero-copy foundation to see OpenMP/tensor benefits)
- **User Story 3 (P3 - STRETCH)**: Builds on User Story 2 (requires fast tensor pipeline before multi-coordinator makes sense)
- **User Story 4 (P4 - OPTIONAL)**: Only if User Story 3 insufficient (conditional dependency based on profiling results)

### Within Each User Story

**User Story 1 (Phase 3)**:
1. Phase 1A (Thread-Local Buffer): T011-T014 → Sequential (modify same file)
2. Phase 1B (Queue API): T015-T020 → Sequential (modify same file)
3. Phase 1C (Coordinator): T021-T025 → Sequential (modify same file)
4. Phase 1D (Fallback): T026-T027 → Parallel with 1A-1C (different files)
5. Phase 1 Enhancements: T028-T030 → Parallel (different files)
6. Phase 1 Validation: T031-T035 → Sequential after all implementation complete

**User Story 2 (Phase 4)**:
1. Phase 2A (OpenMP): T036-T040 → Sequential (CMake changes)
2. Phase 2B (Pinned Memory): T041-T047 → Sequential (same Python file)
3. Phase 2C (Integration): T048-T051 → Sequential after 2B (depends on DLPackInferenceBridge)
4. Phase 2 Validation: T052-T056 → Sequential after all implementation complete

**User Story 3 (Phase 5)**:
1. Phase 3A Implementation: T057-T062 → Mostly sequential (coordinator architecture)
2. Phase 3A Validation: T063-T066 → Sequential after implementation

### Parallel Opportunities

**Within Setup Phase (Phase 1)**:
- T003, T004, T005 can run in parallel (different scripts)

**Within Foundational Phase (Phase 2)**:
- T006, T007 can run in parallel (same file, but different methods)
- T008, T009, T010 can run in parallel (different scripts)

**Within User Story 1 (Phase 3)**:
- T026-T027 (Phase 1D Fallback) can run parallel with T011-T030 (core implementation)
- T028-T030 (Enhancements) can run parallel with T011-T027 (different files)

**Within Cross-Cutting Phase (Phase 7)**:
- T070-T075 (Instrumentation) can all run in parallel (different metrics)
- T076-T078 (Feature Flags) can all run in parallel (different files)
- T082-T086 (Documentation) can all run in parallel (different docs)

**Across User Stories** (if team capacity allows):
- Once User Story 1 is complete and validated, User Story 2 can begin
- User Story 3 can only begin after User Story 2 completes (requires fast tensor pipeline)
- Cross-cutting instrumentation (Phase 7) can run in parallel with any user story phase

---

## Parallel Example: User Story 1 (Phase 1A-1D)

```bash
# Core implementation (sequential within each sub-phase):
T011-T014: Thread-Local Buffer (sequential - same file)
T015-T020: Queue API Shift (sequential - same file)
T021-T025: Coordinator Simplification (sequential - same file)

# Parallel with core implementation (different files):
T026-T027: Optional Fallback (can run parallel)
T028-T030: Enhancements (can run parallel)

# After all implementation complete:
T031-T035: Validation & Profiling (sequential)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only - Phase 1)

1. Complete **Phase 1: Setup** (T001-T005)
2. Complete **Phase 2: Foundational** (T006-T010) - CRITICAL, blocks all stories
3. Complete **Phase 3: User Story 1** (T011-T035)
4. **STOP and VALIDATE**: Run profiling campaign, verify 1,500-3,000 sims/sec
5. **Checkpoint Decision**:
   - If targets met → Proceed to User Story 2 (Phase 2 TARGET)
   - If targets missed → Rollback and investigate

### Incremental Delivery (Phase 1 → Phase 2 TARGET)

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 (Phase 1) → Validate independently (1.5k-3k sims/sec) → **MVP Delivered!**
3. Add User Story 2 (Phase 2) → Validate independently (7k-9k sims/sec) → **PRIMARY TARGET ACHIEVED! ✅**
4. **Optional**: Add User Story 3 (Phase 3A) → Validate independently (12k-20k sims/sec) → **STRETCH GOAL**
5. **Optional**: Add User Story 4 (Phase 3B) → Only if decision gate conditions met → **ADVANCED OPTIONAL**

### Expected Timeline

- **Phase 1 Setup + Foundational**: 1-2 days
- **User Story 1 (Phase 1)**: 3-5 days (MVP - 1,500-3,000 sims/sec)
- **User Story 2 (Phase 2)**: 4-6 days (PRIMARY TARGET - 7,000-9,000 sims/sec) ✅
- **Total to TARGET**: 8-13 days (1.5-2.5 weeks)
- **User Story 3 (Phase 3A)**: +4-6 days (optional, 12k-20k sims/sec)
- **User Story 4 (Phase 3B)**: +4-6 weeks (only if required, 20k-35k sims/sec)

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **[Story] label**: Maps task to specific user story for traceability (US1, US2, US3, US4)
- **DoD (Definition of Done)**: Each sub-phase includes acceptance criteria
- **Profiling-Driven**: Each phase validated with 100+ trial campaign (Principle VI)
- **Rollback-Ready**: Automated rollback on regression detection
- **Constitution-Compliant**: All 6 principles enforced via code review and profiling gates
- **Sequential Phases**: Phase 2 (US2) MUST follow Phase 1 (US1) due to dependency on zero-copy foundation
- **Optional Phases**: Phase 3A (US3) and Phase 3B (US4) only if stretch goals desired and decision gates met
- **Commit Strategy**: Commit after each sub-phase (1A, 1B, 1C, etc.) or logical group, NOT per task
- **Stop at Checkpoints**: Validate each user story independently before proceeding to next

---

## Summary

**Total Tasks**: 99 tasks across 7 phases
- **Phase 1 (Setup)**: 5 tasks
- **Phase 2 (Foundational)**: 5 tasks
- **Phase 3 (US1 - MVP)**: 25 tasks (T011-T035) → **1,500-3,000 sims/sec**
- **Phase 4 (US2 - TARGET)**: 21 tasks (T036-T056) → **7,000-9,000 sims/sec** ✅
- **Phase 5 (US3 - STRETCH)**: 10 tasks (T057-T066) → **12,000-20,000 sims/sec** (optional)
- **Phase 6 (US4 - OPTIONAL)**: 3 tasks (T067-T069) → **20,000-35,000 sims/sec** (deferred)
- **Phase 7 (Cross-Cutting)**: 18 tasks (T070-T087)
- **Constitution Compliance**: 12 tasks (T088-T099)

**Parallel Opportunities**: 25+ tasks marked [P] can run in parallel within their phases

**Independent Test Criteria**:
- **US1**: Verify <1% state cloning, throughput 1,500-3,000 sims/sec via scripts/benchmark_phase1.py
- **US2**: Verify OpenMP >1 thread, tensor ≤2ms, GPU ≥80%, throughput 7,000-9,000 sims/sec via scripts/benchmark_phase2.py
- **US3**: Verify coordinator blocking <10%, throughput 12,000-20,000 sims/sec (optional)
- **US4**: Verify Python callback <5ms, throughput 20,000-35,000 sims/sec (deferred)

**Suggested MVP Scope**: Phase 1-2-3 only (Setup + Foundational + US1 = 35 tasks total, delivers 1.5k-3k sims/sec in 1 week)

**Suggested TARGET Scope**: Phase 1-2-3-4 (Setup + Foundational + US1 + US2 = 56 tasks total, delivers 7k-9k sims/sec ✅ in 2 weeks)
