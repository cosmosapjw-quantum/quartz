#!/bin/bash
# Comprehensive validation script for all optimization phases
# Purpose: Validate each phase independently with automated rollback on failure
# Usage:
#   ./validate_all_phases.sh [--verify-baseline|--verify-phase1|--verify-phase2|--verify-phase3a]
#   No args: Run full validation (baseline → Phase 1 → Phase 2)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:---full}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MCTS Throughput Optimization - Phase Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found"
    echo "   Run: python3 -m venv venv && source venv/bin/activate && pip install -e ."
    exit 1
fi

source venv/bin/activate

# Helper: Check if benchmarking scripts exist
check_benchmark_script() {
    local script="$1"
    if [ ! -f "$script" ]; then
        echo "⚠️  Benchmark script not found: $script"
        echo "   This script will be created during Phase 2 implementation"
        return 1
    fi
    return 0
}

# Helper: Extract throughput from profiling results
get_throughput() {
    local profiling_dir="$1"
    if [ ! -d "$profiling_dir" ]; then
        echo "0"
        return
    fi

    # Try to find throughput in results JSON or analysis output
    # This is a placeholder - actual implementation depends on profiling output format
    local throughput=$(find "$profiling_dir" -name "*.json" -exec grep -h "simulations_per_second\|throughput" {} \; 2>/dev/null | head -1 | grep -oP '\d+(\.\d+)?' || echo "0")
    echo "$throughput"
}

# ============================================================================
# Phase 0: Baseline Validation
# ============================================================================

if [[ "$MODE" == "--verify-baseline" ]] || [[ "$MODE" == "--full" ]]; then
    echo "=== Phase 0: Baseline Validation ==="
    echo ""

    # Check if baseline profiling data exists
    BASELINE_DIR=$(find . -maxdepth 1 -type d -name "profiling_suite_*" | head -1)

    if [ -z "$BASELINE_DIR" ]; then
        echo "⚠️  No baseline profiling data found"
        echo "   Expected: profiling_suite_YYYYMMDD_HHMMSS directory"
        echo "   Run baseline profiling first (this may take 10-30 minutes):"
        echo ""
        echo "   python scripts/profiling/run_campaign.py --baseline --trials 100"
        echo ""
    else
        echo "✅ Baseline profiling data found: $BASELINE_DIR"

        # Check for documented baseline (120 sims/sec from CLAUDE.md)
        echo "   Documented baseline: 120.4 sims/sec (560 trials)"
        echo ""
    fi
fi

# ============================================================================
# Phase 1: State Cloning Elimination
# ============================================================================

if [[ "$MODE" == "--verify-phase1" ]] || [[ "$MODE" == "--full" ]]; then
    echo "=== Phase 1: State Cloning Elimination ==="
    echo ""

    # Run state cloning audit
    if [ -f "scripts/audit_state_cloning.sh" ]; then
        echo "Running state cloning audit..."
        if bash scripts/audit_state_cloning.sh; then
            echo ""
        else
            echo ""
            echo "❌ Phase 1 FAILED: State cloning detected"
            echo ""
            echo "Rollback procedure:"
            echo "  git revert HEAD~10..HEAD"
            echo "  pip install -e . --force-reinstall --no-deps"
            echo "  ./scripts/validate_all_phases.sh --verify-baseline"
            exit 1
        fi
    else
        echo "⚠️  State cloning audit script not found"
    fi

    # Run Phase 1 benchmarks (if available)
    if check_benchmark_script "scripts/benchmark_phase1.py"; then
        echo "Running Phase 1 profiling campaign (this may take 10-20 minutes)..."
        python scripts/benchmark_phase1.py --trials 100 --output profiling_results/phase1

        P1_THROUGHPUT=$(get_throughput "profiling_results/phase1")
        echo ""
        echo "Phase 1 throughput: $P1_THROUGHPUT sims/sec"

        # Validate against target (1,500-3,000 sims/sec)
        if (( $(echo "$P1_THROUGHPUT < 1500" | bc -l) )); then
            echo "❌ Phase 1 FAILED: Throughput $P1_THROUGHPUT < 1,500 sims/sec"
            echo ""
            echo "Rollback procedure:"
            echo "  git revert HEAD~10..HEAD"
            echo "  pip install -e . --force-reinstall --no-deps"
            exit 1
        fi

        echo "✅ Phase 1 PASSED: Throughput $P1_THROUGHPUT sims/sec (target: 1,500-3,000)"
    else
        echo "⚠️  Phase 1 benchmarking skipped (script not yet created)"
    fi

    echo ""
fi

# ============================================================================
# Phase 2: OpenMP + Tensor Pipeline
# ============================================================================

if [[ "$MODE" == "--verify-phase2" ]] || [[ "$MODE" == "--full" ]]; then
    echo "=== Phase 2: OpenMP + Tensor Pipeline ==="
    echo ""

    # Run OpenMP verification
    if [ -f "scripts/verify_openmp.sh" ]; then
        echo "Running OpenMP verification..."
        if bash scripts/verify_openmp.sh; then
            echo ""
        else
            echo ""
            echo "❌ Phase 2 FAILED: OpenMP not linked"
            echo ""
            echo "Fix required in CMakeLists.txt:"
            echo "  target_link_libraries(mcts_py PRIVATE OpenMP::OpenMP_CXX)"
            exit 1
        fi
    else
        echo "⚠️  OpenMP verification script not found"
    fi

    # Run Phase 2 benchmarks (if available)
    if check_benchmark_script "scripts/benchmark_phase2.py"; then
        echo "Running Phase 2 profiling campaign (this may take 10-20 minutes)..."
        python scripts/benchmark_phase2.py --trials 100 --output profiling_results/phase2

        P2_THROUGHPUT=$(get_throughput "profiling_results/phase2")
        echo ""
        echo "Phase 2 throughput: $P2_THROUGHPUT sims/sec"

        # Validate against target (7,000-9,000 sims/sec)
        if (( $(echo "$P2_THROUGHPUT < 7000" | bc -l) )); then
            echo "❌ Phase 2 FAILED: Throughput $P2_THROUGHPUT < 7,000 sims/sec"
            echo ""
            echo "Rollback procedure:"
            echo "  git revert HEAD~15..HEAD  # Revert Phase 2 commits"
            echo "  pip install -e . --force-reinstall --no-deps"
            echo "  ./scripts/validate_all_phases.sh --verify-phase1"
            exit 1
        fi

        echo "✅ Phase 2 PASSED: Throughput $P2_THROUGHPUT sims/sec (target: 7,000-9,000) 🎯"
    else
        echo "⚠️  Phase 2 benchmarking skipped (script not yet created)"
    fi

    echo ""
fi

# ============================================================================
# Phase 3A: Multi-Coordinator (Optional)
# ============================================================================

if [[ "$MODE" == "--verify-phase3a" ]]; then
    echo "=== Phase 3A: Multi-Coordinator (Stretch) ==="
    echo ""

    if check_benchmark_script "scripts/benchmark_phase3a.py"; then
        echo "Running Phase 3A profiling campaign (this may take 10-20 minutes)..."
        python scripts/benchmark_phase3a.py --coordinators 3 --trials 100 --output profiling_results/phase3a

        P3A_THROUGHPUT=$(get_throughput "profiling_results/phase3a")
        echo ""
        echo "Phase 3A throughput: $P3A_THROUGHPUT sims/sec"

        # Validate against target (12,000-20,000 sims/sec)
        if (( $(echo "$P3A_THROUGHPUT < 12000" | bc -l) )); then
            echo "❌ Phase 3A FAILED: Throughput $P3A_THROUGHPUT < 12,000 sims/sec"
            echo "   Consider Phase 3B (multi-process) if needed"
        else
            echo "✅ Phase 3A PASSED: Throughput $P3A_THROUGHPUT sims/sec (target: 12,000-20,000) 🚀"
        fi
    else
        echo "⚠️  Phase 3A not yet implemented"
    fi

    echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All requested validations completed"
echo ""
echo "Next steps:"
echo "  - If all phases passed: Commit and document results"
echo "  - If any phase failed: Follow rollback procedure above"
echo "  - Review profiling results in profiling_results/"
echo ""
