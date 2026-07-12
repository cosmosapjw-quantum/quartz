#!/bin/bash
################################################################################
# CRITICAL FIX: Rebuild with Correct Buffer Size
################################################################################
#
# ISSUE: Previous rebuild didn't fix buffer overflow because there were
#        TWO buffer implementations:
#        1. thread_local_metrics.hpp (fixed first time)
#        2. thread_metrics.hpp (NOW FIXED - this is the one actually used!)
#
# CHANGES MADE:
#   - thread_metrics.hpp line 43: Capacity = 4096 → 524288
#   - thread_metrics.hpp line 212: TimingRingBuffer<4096> → <524288>
#   - thread_metrics.hpp line 262: TimingRingBuffer<4096> → <524288>
#
################################################################################

set -e  # Exit on error

echo "================================================================================================"
echo "REBUILDING C++ EXTENSIONS WITH CORRECT BUFFER SIZE"
echo "================================================================================================"
echo ""

# Step 1: Verify changes are in place
echo "Step 1: Verifying buffer size changes..."
echo ""

EXPECTED="524288"
FOUND=$(grep "template<size_t Capacity =" cpp_extensions/mcts/profiling/thread_metrics.hpp | grep -o "[0-9]*")

if [ "$FOUND" = "$EXPECTED" ]; then
    echo "✅ thread_metrics.hpp template default: $FOUND"
else
    echo "❌ ERROR: thread_metrics.hpp template default is $FOUND, expected $EXPECTED"
    exit 1
fi

FOUND=$(grep "TimingRingBuffer<" cpp_extensions/mcts/profiling/thread_metrics.hpp | grep "get_timing_buffer" | grep -o "<[0-9]*>" | grep -o "[0-9]*")
if [ "$FOUND" = "$EXPECTED" ]; then
    echo "✅ thread_metrics.hpp get_timing_buffer(): $FOUND"
else
    echo "❌ ERROR: thread_metrics.hpp get_timing_buffer() is $FOUND, expected $EXPECTED"
    exit 1
fi

FOUND=$(grep "alignas.*TimingRingBuffer<" cpp_extensions/mcts/profiling/thread_metrics.hpp | grep -o "<[0-9]*>" | grep -o "[0-9]*")
if [ "$FOUND" = "$EXPECTED" ]; then
    echo "✅ thread_metrics.hpp timing_buffer_ member: $FOUND"
else
    echo "❌ ERROR: thread_metrics.hpp timing_buffer_ member is $FOUND, expected $EXPECTED"
    exit 1
fi

echo ""
echo "✅ All buffer size changes verified!"
echo ""

# Step 2: Clean build artifacts
echo "Step 2: Cleaning old build artifacts..."
echo ""

rm -rf build/ *.so
echo "✅ Cleaned"
echo ""

# Step 3: Rebuild
echo "Step 3: Rebuilding C++ extensions..."
echo ""

export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
pip install -e . --force-reinstall --no-deps

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful"
else
    echo ""
    echo "❌ Build failed"
    exit 1
fi

echo ""

# Step 4: Validate profiling
echo "Step 4: Validating profiling setup..."
echo ""

./scripts/run_profiling_suite.sh --validate-only

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Profiling validation passed"
else
    echo ""
    echo "❌ Profiling validation failed"
    exit 1
fi

echo ""

# Step 5: Quick test with 100 simulations
echo "Step 5: Running quick test (100 simulations)..."
echo ""

python scripts/unified_profiler.py --simulations 100 --threads 1 --output test_buffer_fix

if [ $? -ne 0 ]; then
    echo "❌ Test failed"
    exit 1
fi

echo ""

# Step 6: Verify capture rate
echo "Step 6: Verifying 100% capture rate..."
echo ""

python << 'EOF'
import json
import sys

try:
    with open('test_buffer_fix/cpp_profiling.json', 'r') as f:
        data = json.load(f)

    counters = data['counters']
    timing = data['timing_stats']

    # Check key metrics
    checks = [
        ('state_clone_total', 'state_clone_count'),
        ('expansion_total', 'python_callback_entry'),
        ('selection_total', 'python_callback_entry'),
    ]

    all_good = True
    for timing_metric, counter_metric in checks:
        timing_count = timing.get(timing_metric, {}).get('count', 0)
        counter_value = counters.get(counter_metric, 0)

        if counter_value == 0:
            continue

        capture_rate = 100 * timing_count / counter_value

        status = "✅" if capture_rate >= 99.0 else "❌"
        print(f"{status} {timing_metric:<25} {timing_count:>4} / {counter_value:>4} = {capture_rate:>6.1f}%")

        if capture_rate < 99.0:
            all_good = False

    print()

    if all_good:
        print("✅ BUFFER FIX SUCCESSFUL - 100% capture rate!")
        sys.exit(0)
    else:
        print("❌ BUFFER STILL BROKEN - Capture rate < 99%")
        sys.exit(1)

except Exception as e:
    print(f"❌ Error reading profiling data: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================================"
    echo "✅ SUCCESS: Buffer fix verified!"
    echo "================================================================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run full profiling campaign:"
    echo "     ./scripts/run_profiling_suite.sh --production"
    echo ""
    echo "  2. This will take ~40 minutes and generate complete profiling data"
    echo ""
    echo "  3. After completion, re-run analysis to get accurate bottleneck data"
    echo ""
else
    echo ""
    echo "================================================================================================"
    echo "❌ FAILURE: Buffer fix did not work"
    echo "================================================================================================"
    echo ""
    echo "Debug steps:"
    echo "  1. Check test_buffer_fix/cpp_profiling.json manually"
    echo "  2. Verify build actually recompiled (check timestamps)"
    echo "  3. Check for CMake cache issues"
    echo ""
    exit 1
fi
