# Alpha and Beta Testing Guide

**Version**: 1.0.0-alpha
**Last Updated**: 2025-10-01
**Target Audience**: Developers, QA Engineers, Early Adopters

This guide outlines the testing procedures required before production deployment of the AlphaZero engine. Follow these procedures sequentially to validate system stability, performance, and correctness.

---

## Table of Contents

1. [Alpha Testing Phase](#alpha-testing-phase)
2. [Beta Testing Phase](#beta-testing-phase)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Known Issues and Limitations](#known-issues-and-limitations)
5. [Reporting Issues](#reporting-issues)

---

## Alpha Testing Phase

**Objective**: Validate core functionality, identify critical bugs, and verify performance targets on reference hardware.

**Duration**: 1-2 weeks
**Environment**: Development/Staging with reference hardware (Ryzen 5900X, RTX 3060 Ti)

### A1. Build and Environment Validation

**Goal**: Ensure the system builds cleanly and dependencies are correctly configured.

#### Steps:

1. **Clean build from source**:
   ```bash
   # Remove any previous builds
   rm -rf build/ dist/ *.egg-info
   pip uninstall -y alphazero-engine

   # Fresh build with optimizations
   export CFLAGS="-O3 -march=znver3 -fopenmp"
   export CXXFLAGS="-O3 -march=znver3 -fopenmp"
   pip install -e . --force-reinstall --no-deps
   ```

2. **Verify build output**:
   ```bash
   # Check that C++ extensions compiled successfully
   python -c "import alphazero_cpp; print('C++ extensions loaded')"

   # Verify CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **Run basic smoke tests**:
   ```bash
   # Contract tests (must all pass)
   python -m pytest tests/contract/ -v --tb=short

   # Core unit tests
   python -m pytest tests/unit/ -v -k "not slow" --tb=short
   ```

**Expected Outcome**: All builds succeed, CUDA detected, contract tests pass.

**Failure Criteria**: Build errors, missing dependencies, contract test failures.

---

### A2. Component Integration Testing

**Goal**: Verify that all major components integrate correctly.

#### Steps:

1. **MCTS engine integration**:
   ```bash
   python -m pytest tests/integration/test_mcts_integration.py -v
   ```

2. **Inference pipeline integration**:
   ```bash
   python -m pytest tests/integration/test_inference_integration.py -v
   ```

3. **Full system integration**:
   ```bash
   python -m pytest tests/integration/test_full_system.py -v --tb=short
   ```

**Expected Outcome**: All integration tests pass within 2 minutes total runtime.

**Failure Criteria**: Any integration test failures, hangs, or crashes.

---

### A3. Performance Baseline Validation

**Goal**: Establish performance baselines and verify hardware utilization targets.

#### Steps:

1. **Run performance benchmarks**:
   ```bash
   python -m pytest tests/performance/test_benchmarks.py -v
   ```

2. **Validate MCTS throughput**:
   ```bash
   # Should achieve >10k simulations/sec single-threaded
   python scripts/tune_threads.py --game gomoku --quick-test
   ```

3. **Validate GPU utilization**:
   ```bash
   # Should achieve 80%+ GPU utilization
   python scripts/tune_batch_size.py --game gomoku --quick-test
   ```

4. **Memory footprint validation**:
   ```bash
   # Tree memory should be <1GB for 10M nodes
   python -m pytest tests/performance/test_benchmarks.py::test_memory_efficiency_performance -v
   ```

**Expected Outcome**:
- MCTS throughput: >10k sims/sec single-threaded, >30k sims/sec with 8-10 threads
- GPU utilization: 80-92% during search operations
- Memory footprint: <1GB tree memory

**Failure Criteria**: Performance below 50% of targets, OOM errors, GPU utilization <60%.

---

### A4. Short-Duration Training Run

**Goal**: Validate the complete training pipeline with a short run (2-4 hours).

#### Steps:

1. **Prepare training environment**:
   ```bash
   mkdir -p logs/alpha_test checkpoints/alpha_test training_data/alpha_test
   ```

2. **Launch short training run** (use reduced parameters):
   ```bash
   # Create test config (copy gomoku_48h_training.yaml and reduce parameters)
   # Modify: self_play_games_per_iteration: 20, training_steps_per_iteration: 200

   python -m src.training.training_loop \
     --config config/gomoku_alpha_test.yaml \
     --max-iterations 3
   ```

3. **Monitor training progress**:
   ```bash
   # Watch logs
   tail -f logs/alpha_test/training.log

   # Monitor GPU usage
   watch -n 1 nvidia-smi
   ```

4. **Validate training outputs**:
   - Experience buffer populated (check `training_data/alpha_test/`)
   - Checkpoints created (check `checkpoints/alpha_test/`)
   - TensorBoard logs (check `logs/tensorboard/alpha_test/`)
   - No memory leaks or crashes

**Expected Outcome**:
- 3 training iterations complete successfully
- ~60 self-play games generated
- Checkpoints saved every 2 iterations
- No OOM errors or crashes

**Failure Criteria**: Training crashes, memory leaks, corrupt checkpoints, loss becomes NaN.

---

### A5. Memory Stability Testing

**Goal**: Verify no memory leaks during extended operation.

#### Steps:

1. **Run 1-hour soak test**:
   ```bash
   python -m pytest tests/soak/test_memory_stability.py -v --tb=short
   ```

2. **Analyze results**:
   ```bash
   # Check for memory leaks (growth should be <50MB over 1 hour)
   python -c "
   import json
   with open('soak_test_results.json') as f:
       data = json.load(f)
       print(f'Memory growth: {data[\"memory_growth_mb\"]:.2f} MB')
       print(f'Leak detected: {data[\"resource_leaks_detected\"]}')
       print(f'Test passed: {data[\"passed\"]}')
   "
   ```

**Expected Outcome**: Memory growth <50MB/hour, no resource leaks detected.

**Failure Criteria**: Memory growth >100MB/hour, resource leaks detected, test fails.

---

### A6. Error Handling and Recovery

**Goal**: Verify graceful degradation and error recovery mechanisms.

#### Steps:

1. **Test CUDA OOM recovery**:
   ```bash
   python -m pytest tests/unit/test_oom_recovery.py -v
   ```

2. **Test CPU fallback**:
   ```bash
   python -m pytest tests/unit/test_cpu_fallback.py -v
   ```

3. **Test error handling framework**:
   ```bash
   python -m pytest tests/unit/test_error_handling.py -v
   ```

**Expected Outcome**: All error handling tests pass, graceful degradation works.

**Failure Criteria**: Unhandled exceptions, crashes, hanging on errors.

---

### Alpha Phase Exit Criteria

✅ All contract tests passing (99/99)
✅ All integration tests passing (5/5)
✅ Performance baselines met (>30k sims/sec, 80%+ GPU util)
✅ Short training run completes successfully (3 iterations)
✅ Memory stability validated (<50MB/hour growth)
✅ Error handling validated (OOM recovery, CPU fallback)

**Sign-off Required**: Lead Developer, QA Engineer

---

## Beta Testing Phase

**Objective**: Validate production readiness with extended training runs, multi-game support, and real-world scenarios.

**Duration**: 2-4 weeks
**Environment**: Production-like environment with monitoring

### B1. Extended Training Validation

**Goal**: Complete a full 48-hour Gomoku training run to validate superhuman performance.

#### Steps:

1. **Launch 48-hour training**:
   ```bash
   # Ensure monitoring is enabled
   python -m src.training.training_loop \
     --config config/gomoku_48h_training.yaml \
     --max-time-hours 48
   ```

2. **Monitor training progress**:
   - Check TensorBoard: `tensorboard --logdir logs/tensorboard/gomoku_48h`
   - Monitor system resources: `htop`, `nvidia-smi`
   - Validate checkpoints every 2 hours

3. **Evaluate final model**:
   ```bash
   # Test against random baseline
   python -m src.training.evaluator \
     --model checkpoints/gomoku_48h/best_model.pth \
     --baseline random \
     --games 100
   ```

**Expected Outcome**:
- Training completes 20-30 iterations
- 4,000-6,000 self-play games generated
- Win rate >95% vs random baseline
- Win rate >70% vs strong amateur baseline
- No crashes or memory leaks

**Failure Criteria**: Training crashes, performance plateaus, memory exhaustion, win rate <60% vs random.

---

### B2. Multi-Game Validation

**Goal**: Verify engine works correctly across all supported games.

#### Steps:

1. **Test Chess training** (short run, 6 hours):
   ```bash
   # Modify config for chess
   python -m src.training.training_loop \
     --config config/chess_test.yaml \
     --max-iterations 5
   ```

2. **Test Go training** (short run, 6 hours):
   ```bash
   # Modify config for Go (9x9 board)
   python -m src.training.training_loop \
     --config config/go_test.yaml \
     --max-iterations 5
   ```

3. **Validate game rules**:
   ```bash
   python -m pytest tests/unit/test_game_rules.py -v
   ```

**Expected Outcome**: All games train successfully, rules correctly enforced.

**Failure Criteria**: Game-specific crashes, illegal moves executed, rule violations.

---

### B3. Stress Testing and Edge Cases

**Goal**: Test system under extreme conditions and edge cases.

#### Steps:

1. **High-load stress test**:
   ```bash
   # Run with maximum parallelism
   python scripts/tune_threads.py --game gomoku --threads 16 --simulations 2000
   ```

2. **Memory pressure test**:
   ```bash
   # Run with limited GPU memory
   CUDA_VISIBLE_DEVICES=0 python -m pytest tests/unit/test_oom_recovery.py -v
   ```

3. **Long-game scenarios**:
   ```bash
   # Test games that exceed typical length
   python -m pytest tests/integration/test_terminal_detection_variations.py -v
   ```

**Expected Outcome**: System handles stress gracefully, no crashes or data corruption.

**Failure Criteria**: Crashes under high load, memory corruption, deadlocks.

---

### B4. Production Configuration Validation

**Goal**: Validate production configurations and deployment procedures.

#### Steps:

1. **Docker deployment test**:
   ```bash
   # Build and run Docker container
   ./scripts/docker/build.sh -t runtime
   ./scripts/docker/run.sh runtime

   # Verify container health
   docker ps
   docker logs alphazero-runtime
   ```

2. **Configuration validation**:
   ```bash
   # Test all config files load correctly
   python -c "from src.utils.config import ConfigManager; \
              cfg = ConfigManager('config/gomoku_48h_training.yaml'); \
              print('Config valid:', cfg.validate())"
   ```

3. **Monitoring setup**:
   - Deploy Prometheus + Grafana (see `docs/operations.md`)
   - Verify metrics collection
   - Test alert thresholds

**Expected Outcome**: Docker deployment successful, configs valid, monitoring operational.

**Failure Criteria**: Docker build failures, config validation errors, monitoring gaps.

---

### B5. User Acceptance Testing

**Goal**: Validate user-facing features and documentation.

#### Steps:

1. **Documentation review**:
   - Verify `README.md` is accurate
   - Test all code examples in `docs/api.md`
   - Validate training guide procedures in `docs/training_guide.md`

2. **End-to-end user workflow**:
   ```bash
   # New user setup (follow README)
   git clone <repo>
   cd alphazero-engine
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .

   # Launch quick training test
   python -m src.training.training_loop --config config/default.yaml --max-iterations 1
   ```

3. **API usability test**:
   - Test MCTS API (see `specs/001-goal-create-spec/contracts/mcts_api.py`)
   - Test Inference API (see `specs/001-goal-create-spec/contracts/inference_api.py`)
   - Test Training API (see `specs/001-goal-create-spec/contracts/training_api.py`)

**Expected Outcome**: Documentation accurate, workflows intuitive, APIs easy to use.

**Failure Criteria**: Documentation errors, broken examples, confusing APIs.

---

### Beta Phase Exit Criteria

✅ 48-hour Gomoku training completes successfully
✅ Superhuman Gomoku performance achieved (>70% win rate vs strong amateur)
✅ Multi-game support validated (Chess, Go)
✅ Stress testing passed (high load, memory pressure)
✅ Docker deployment validated
✅ Production configurations validated
✅ User acceptance testing passed

**Sign-off Required**: Lead Developer, QA Engineer, Product Owner

---

## Pre-Deployment Checklist

Before deploying to production, verify the following:

### Code Quality
- [ ] All tests passing (1129+ tests)
- [ ] No compiler warnings in C++ code
- [ ] Python code passes linting (flake8, mypy)
- [ ] No TODO comments in production code paths

### Performance
- [ ] Performance targets met (30k+ sims/sec, 80-92% GPU util)
- [ ] Memory footprint within limits (<1GB tree memory)
- [ ] No memory leaks detected (soak tests pass)

### Documentation
- [ ] README.md updated with current version
- [ ] CHANGELOG.md includes all changes
- [ ] API documentation complete and accurate
- [ ] Operations runbook validated

### Deployment
- [ ] Docker images built and tested
- [ ] Configuration files validated
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested

### Security
- [ ] No credentials in code or configs
- [ ] Docker containers run as non-root
- [ ] File permissions properly set
- [ ] Network ports properly secured

---

## Known Issues and Limitations

### Alpha Version Limitations

1. **Training Time**: 48-hour training required for superhuman Gomoku performance
   - **Impact**: Cannot achieve superhuman play immediately
   - **Mitigation**: Use pre-trained models for demos

2. **Single GPU Support**: Only single GPU training/inference supported
   - **Impact**: Limited scalability for large models
   - **Mitigation**: Future multi-GPU support planned for beta

3. **Game Support**: Currently supports Gomoku, Chess, Go
   - **Impact**: Other board games not supported
   - **Mitigation**: Game adapter interface allows future extensions

4. **Platform**: Linux-only, CUDA required for production performance
   - **Impact**: Cannot run on Windows/macOS for production
   - **Mitigation**: CPU fallback available for development

### Known Bugs (Alpha)

- **None critical** - All known issues resolved in T001-T053

---

## Reporting Issues

### Issue Severity Levels

- **Critical**: Crashes, data loss, security vulnerabilities
- **High**: Performance degradation >50%, memory leaks, incorrect results
- **Medium**: Performance degradation <50%, usability issues
- **Low**: Documentation errors, minor UI issues

### How to Report

1. **Check existing issues**: Search GitHub issues first
2. **Gather information**:
   - System configuration (CPU, GPU, RAM, OS)
   - Software versions (Python, PyTorch, CUDA)
   - Minimal reproduction steps
   - Logs and error messages
   - Expected vs actual behavior

3. **Submit issue** on GitHub with template:
   ```markdown
   **Environment**:
   - OS: Ubuntu 22.04
   - Python: 3.12.3
   - PyTorch: 2.1.0
   - CUDA: 12.1
   - Hardware: Ryzen 5900X, RTX 3060 Ti

   **Issue Description**:
   [Clear description]

   **Steps to Reproduce**:
   1. [Step 1]
   2. [Step 2]

   **Expected Behavior**:
   [What should happen]

   **Actual Behavior**:
   [What actually happens]

   **Logs**:
   ```
   [Relevant log output]
   ```
   ```

---

## Testing Timeline

| Phase | Duration | Key Activities | Exit Criteria |
|-------|----------|----------------|---------------|
| **Alpha** | 1-2 weeks | Component validation, short training runs, soak tests | All tests pass, baselines met |
| **Beta** | 2-4 weeks | 48h training, multi-game validation, stress testing | Superhuman play, production ready |
| **Release Candidate** | 1 week | Final validation, documentation review | Sign-off from all stakeholders |
| **Production** | Ongoing | Monitoring, incremental improvements | Stable operation |

---

## Success Criteria Summary

### Alpha Success
- ✅ Core functionality validated
- ✅ Performance baselines met
- ✅ No critical bugs
- ✅ Memory stability confirmed

### Beta Success
- ✅ Superhuman Gomoku performance
- ✅ Multi-game support validated
- ✅ Production deployment tested
- ✅ User acceptance passed

### Production Ready
- ✅ All alpha/beta criteria met
- ✅ Documentation complete
- ✅ Monitoring configured
- ✅ Stakeholder sign-off obtained

---

**Document Version**: 1.0.0-alpha
**Next Review**: After Alpha Phase Completion
**Maintained By**: AlphaZero Engine Team
