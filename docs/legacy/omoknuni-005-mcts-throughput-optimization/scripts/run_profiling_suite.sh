#!/bin/bash
################################################################################
# Profiling Suite Runner - Sequential Execution with Validation
#
# UPDATED 2025-10-15: Added profiling validation checks to ensure < 10% unaccounted time
#
# Runs all profiling tools sequentially with validation:
#   0. Profiling setup validation (ensures PROFILE_LEVEL_VALUE=3)
#   1. Wall-clock validation (ground-truth baseline)
#   2. Profiling campaign (comprehensive metrics with parameter sweep)
#   3. Results analysis (analyze and identify bottlenecks)
#
# Usage:
#   # Validate profiling setup only
#   ./scripts/run_profiling_suite.sh --validate-only
#
#   # Quick test (fast, few trials)
#   ./scripts/run_profiling_suite.sh --quick
#
#   # Full suite (comprehensive, many trials)
#   ./scripts/run_profiling_suite.sh --full
#
#   # Custom configuration
#   ./scripts/run_profiling_suite.sh \
#       --simulations 800,1600 \
#       --threads 4,8 \
#       --output results/my_suite
#
# Author: MCTS Performance Team
# Date: 2025-10-15
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
MODE="full"  # Changed from quick to full as default
OUTPUT_DIR="profiling_suite_$(date +%Y%m%d_%H%M%S)"
SIMULATIONS=""
THREADS=""
BATCH_SIZES=""
VALIDATION_RUNS="3"
VALIDATION_SIMS="100"
VALIDATE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --production)
            MODE="production"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --simulations)
            SIMULATIONS="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --validation-runs)
            VALIDATION_RUNS="$2"
            shift 2
            ;;
        --validation-sims)
            VALIDATION_SIMS="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --validate-only        Only validate profiling setup (no benchmarks)"
            echo "  --quick                Quick mode (default: 100 sims, 1,4 threads)"
            echo "  --full                 Full mode (default: 100-1600 sims, 1-8 threads)"
            echo "  --output DIR           Output directory (default: timestamped)"
            echo "  --simulations LIST     Comma-separated sim counts (e.g., 100,400,800)"
            echo "  --threads LIST         Comma-separated thread counts (e.g., 1,2,4,8)"
            echo "  --batch-sizes LIST     Comma-separated batch sizes (e.g., 32,64)"
            echo "  --validation-runs N    Number of validation runs (default: 3)"
            echo "  --validation-sims N    Simulations per validation run (default: 100)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick"
            echo "  $0 --full --output results/experiment_001"
            echo "  $0 --simulations 800,1600 --threads 4,8 --output results/custom"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set mode-specific defaults
if [[ "$MODE" == "quick" ]]; then
    # Quick mode: ~1 minute smoke test
    SIMULATIONS="${SIMULATIONS:-100,200}"
    THREADS="${THREADS:-1,4}"
    BATCH_SIZES="${BATCH_SIZES:-64}"
    VALIDATION_RUNS="${VALIDATION_RUNS:-3}"
    VALIDATION_SIMS="${VALIDATION_SIMS:-100}"
    REPETITIONS=1
elif [[ "$MODE" == "production" ]]; then
    # Production mode: ~40 minutes comprehensive with statistical rigor
    SIMULATIONS="${SIMULATIONS:-2000,4000,8000,16000}"
    THREADS="${THREADS:-1,2,4,6,8,10,12}"
    BATCH_SIZES="${BATCH_SIZES:-16,32,64,128}"
    VALIDATION_RUNS="${VALIDATION_RUNS:-20}"
    VALIDATION_SIMS="${VALIDATION_SIMS:-8000}"
    REPETITIONS=5
else
    # Full mode: ~15 minutes thorough exploration
    SIMULATIONS="${SIMULATIONS:-2000,4000,8000}"
    THREADS="${THREADS:-1,2,4,6,8}"
    BATCH_SIZES="${BATCH_SIZES:-32,64,128}"
    VALIDATION_RUNS="${VALIDATION_RUNS:-10}"
    VALIDATION_SIMS="${VALIDATION_SIMS:-4000}"
    REPETITIONS=3
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Logging
LOG_FILE="$OUTPUT_DIR/suite.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}PROFILING SUITE - SEQUENTIAL EXECUTION${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "Mode:             ${GREEN}$MODE${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Simulations:      ${GREEN}$SIMULATIONS${NC}"
echo -e "Threads:          ${GREEN}$THREADS${NC}"
echo -e "Batch sizes:      ${GREEN}$BATCH_SIZES${NC}"
echo ""
echo -e "Suite log: ${LOG_FILE}"
echo ""

SUITE_START=$(date +%s.%N)

################################################################################
# Step 0: Profiling Setup Validation (NEW - 2025-10-15)
################################################################################

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 0/4: PROFILING SETUP VALIDATION${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${YELLOW}Purpose: Verify C++ profiling is enabled and Python integration works${NC}"
echo -e "${YELLOW}Target:  < 10% unaccounted time (vs 77-97% before fixes)${NC}"
echo ""

STEP0_START=$(date +%s.%N)

# Run profiling validation
if python scripts/validate_profiling_setup.py; then
    echo ""
    echo -e "${GREEN}✅ Profiling setup validated successfully${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Profiling validation FAILED!${NC}"
    echo ""
    echo -e "${RED}CRITICAL: Profiling is not properly enabled!${NC}"
    echo ""
    echo -e "${YELLOW}This means:${NC}"
    echo -e "  1. C++ extensions were not compiled with PROFILE_LEVEL_VALUE=3"
    echo -e "  2. You will see 77-97% unaccounted time (profiling gaps)"
    echo -e "  3. Results will be INCOMPLETE and MISLEADING"
    echo ""
    echo -e "${YELLOW}To fix:${NC}"
    echo -e "  1. Read: REBUILD_WITH_PROFILING.md"
    echo -e "  2. Run:  export CXXFLAGS=\"-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3\""
    echo -e "  3. Run:  rm -rf build/ *.so"
    echo -e "  4. Run:  pip install -e . --force-reinstall --no-deps"
    echo -e "  5. Retry: ./scripts/run_profiling_suite.sh --validate-only"
    echo ""
    exit 1
fi

STEP0_END=$(date +%s.%N)
STEP0_DURATION=$(echo "$STEP0_END - $STEP0_START" | bc | xargs printf "%.1f")

echo -e "${GREEN}✅ Step 0 complete in ${STEP0_DURATION}s${NC}"
echo ""

# If validate-only, exit here
if [[ "$VALIDATE_ONLY" == "true" ]]; then
    echo -e "${GREEN}✅ Validation complete - profiling is correctly configured!${NC}"
    echo ""
    echo -e "${YELLOW}Next step: Run full profiling suite${NC}"
    echo -e "  ./scripts/run_profiling_suite.sh --quick    # Quick test"
    echo -e "  ./scripts/run_profiling_suite.sh --full     # Full campaign"
    echo ""
    exit 0
fi

################################################################################
# Step 1: Wall-Clock Validation
################################################################################

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 1/4: WALL-CLOCK VALIDATION${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${YELLOW}Purpose: Establish ground-truth performance baseline (no profiling overhead)${NC}"
echo ""

STEP1_START=$(date +%s.%N)

python scripts/wall_clock_validation.py \
    --simulations "$VALIDATION_SIMS" \
    --runs "$VALIDATION_RUNS" \
    --no-warmup

STEP1_END=$(date +%s.%N)
STEP1_DURATION=$(echo "$STEP1_END - $STEP1_START" | bc | xargs printf "%.1f")

echo ""
echo -e "${GREEN}✅ Step 1 complete in ${STEP1_DURATION}s${NC}"
echo ""

################################################################################
# Step 2: Profiling Campaign
################################################################################

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 2/4: PROFILING CAMPAIGN${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${YELLOW}Purpose: Comprehensive profiling with EnhancedProfiler (295 C++ + Python metrics)${NC}"
echo -e "${YELLOW}Target:  < 10% unaccounted time (fully instrumented)${NC}"
echo -e "${YELLOW}Output:  JSON, Chrome Trace, Markdown per trial${NC}"
echo ""

STEP2_START=$(date +%s.%N)

CAMPAIGN_DIR="$OUTPUT_DIR/campaign"

python scripts/profiling_campaign.py \
    --simulations "$SIMULATIONS" \
    --threads "$THREADS" \
    --batch-sizes "$BATCH_SIZES" \
    --repetitions "$REPETITIONS" \
    --output "$CAMPAIGN_DIR" \
    --yes

STEP2_END=$(date +%s.%N)
STEP2_DURATION=$(echo "$STEP2_END - $STEP2_START" | bc | xargs printf "%.1f")

echo ""
echo -e "${GREEN}✅ Step 2 complete in ${STEP2_DURATION}s${NC}"
echo ""

################################################################################
# Step 3: Results Analysis
################################################################################

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 3/4: RESULTS ANALYSIS${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${YELLOW}Purpose: Analyze campaign results and identify bottlenecks${NC}"
echo ""

STEP3_START=$(date +%s.%N)

python scripts/analyze_profiling_results.py \
    --detailed \
    "$CAMPAIGN_DIR/campaign_summary.json"

STEP3_END=$(date +%s.%N)
STEP3_DURATION=$(echo "$STEP3_END - $STEP3_START" | bc | xargs printf "%.1f")

echo ""
echo -e "${GREEN}✅ Step 3 complete in ${STEP3_DURATION}s${NC}"
echo ""

################################################################################
# Step 4: Validation Check (NEW - 2025-10-15)
################################################################################

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}STEP 4/4: PROFILING COMPLETENESS CHECK${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${YELLOW}Purpose: Verify < 10% unaccounted time across all trials${NC}"
echo ""

STEP4_START=$(date +%s.%N)

# Check unaccounted time from campaign results
echo "Checking profiling completeness from campaign results..."
echo ""

# Parse campaign summary for unaccounted time
if [[ -f "$CAMPAIGN_DIR/campaign_summary.json" ]]; then
    # Extract average unaccounted percentage
    AVG_UNACCOUNTED=$(python3 -c "
import json
import sys
with open('$CAMPAIGN_DIR/campaign_summary.json', 'r') as f:
    data = json.load(f)
    # Calculate average unaccounted percentage
    unaccounted_values = []
    for trial in data.get('trials', []):
        if 'validation' in trial and 'unaccounted_percentage' in trial['validation']:
            unaccounted_values.append(abs(trial['validation']['unaccounted_percentage']))
    if unaccounted_values:
        avg = sum(unaccounted_values) / len(unaccounted_values)
        print(f'{avg:.1f}')
    else:
        print('N/A')
" 2>/dev/null || echo "N/A")

    if [[ "$AVG_UNACCOUNTED" != "N/A" ]]; then
        echo -e "Average Unaccounted Time: ${GREEN}${AVG_UNACCOUNTED}%${NC}"
        echo ""

        # Check if within acceptable range
        if (( $(echo "$AVG_UNACCOUNTED < 10.0" | bc -l) )); then
            echo -e "${GREEN}✅ PASS: Profiling completeness validated${NC}"
            echo -e "   Unaccounted time: ${AVG_UNACCOUNTED}% (< 10% threshold)"
            echo ""
        else
            echo -e "${YELLOW}⚠️  WARNING: High unaccounted time detected${NC}"
            echo -e "   Unaccounted time: ${AVG_UNACCOUNTED}% (>= 10% threshold)"
            echo ""
            echo -e "${YELLOW}This suggests:${NC}"
            echo -e "  1. Some code paths are not instrumented"
            echo -e "  2. Profiling may not be fully enabled"
            echo -e "  3. Results may be incomplete"
            echo ""
            echo -e "${YELLOW}Recommendations:${NC}"
            echo -e "  1. Check PROFILE_LEVEL_VALUE=3 in build"
            echo -e "  2. Review INSTRUMENTATION_CHECKLIST.md"
            echo -e "  3. Add more PROFILE_SCOPE macros to hot paths"
            echo ""
        fi
    else
        echo -e "${YELLOW}⚠️  Could not determine unaccounted time percentage${NC}"
        echo ""
    fi
else
    echo -e "${YELLOW}⚠️  Campaign summary not found, skipping validation${NC}"
    echo ""
fi

STEP4_END=$(date +%s.%N)
STEP4_DURATION=$(echo "$STEP4_END - $STEP4_START" | bc | xargs printf "%.1f")

echo -e "${GREEN}✅ Step 4 complete in ${STEP4_DURATION}s${NC}"
echo ""

################################################################################
# Suite Summary
################################################################################

SUITE_END=$(date +%s.%N)
SUITE_DURATION=$(echo "$SUITE_END - $SUITE_START" | bc | xargs printf "%.1f")

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}✅ PROFILING SUITE COMPLETE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${GREEN}Timing Summary:${NC}"
echo -e "  Step 0 (Validation):     ${STEP0_DURATION}s"
echo -e "  Step 1 (Wall-Clock):     ${STEP1_DURATION}s"
echo -e "  Step 2 (Campaign):       ${STEP2_DURATION}s"
echo -e "  Step 3 (Analysis):       ${STEP3_DURATION}s"
echo -e "  Step 4 (Completeness):   ${STEP4_DURATION}s"
echo -e "  ${GREEN}Total:                   ${SUITE_DURATION}s${NC}"
echo ""
echo -e "${GREEN}Results Location:${NC}"
echo -e "  Suite directory:         $OUTPUT_DIR"
echo -e "  Wall-clock results:      wall_clock_validation_*.json"
echo -e "  Campaign results:        $CAMPAIGN_DIR/"
echo -e "  Campaign summary:        $CAMPAIGN_DIR/campaign_summary.json"
echo -e "  Campaign CSV:            $CAMPAIGN_DIR/results.csv"
echo -e "  Suite log:               $LOG_FILE"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo -e "  1. Review campaign summary: cat $CAMPAIGN_DIR/campaign_summary.json | jq"
echo -e "  2. Open Chrome Trace:       Open $CAMPAIGN_DIR/trial_001/cpp_trace.json in chrome://tracing"
echo -e "  3. Check bottlenecks:       See analysis output above"
echo -e "  4. Implement fixes based on priority bottlenecks"
echo ""
echo -e "${BLUE}================================================================================${NC}"
