#!/bin/bash
# T040: OpenMP verification script for Phase 2
# Verifies that OpenMP is properly linked and functional

set -e  # Exit on error

echo "============================================================"
echo "OPENMP VERIFICATION"
echo "============================================================"
echo ""

# Check if mcts_py library exists
MCTS_LIB=$(find build -name "mcts_py*.so" 2>/dev/null | head -1)
if [ -z "$MCTS_LIB" ]; then
    echo "❌ ERROR: mcts_py libContinue with Phase 2 (OpenMP + Tensor Pipeline)rary not found in build/"
    echo "   Run: python -m pip install -e . --force-reinstall --no-deps"
    exit 1
fi

echo "✅ Found mcts_py library: $MCTS_LIB"
echo ""

# Check if OpenMP is linked
echo "Checking OpenMP linkage..."
if ldd "$MCTS_LIB" | grep -qi "gomp\|omp"; then
    OMP_LIB=$(ldd "$MCTS_LIB" | grep -i "gomp\|omp")
    echo "✅ OpenMP linked: $OMP_LIB"
else
    echo "❌ ERROR: OpenMP not linked to mcts_py"
    echo "   Check CMakeLists.txt for OpenMP::OpenMP_CXX"
    exit 1
fi
echo ""

# Test Python import and OpenMP runtime
echo "Testing OpenMP runtime from Python..."
python3 << 'PYTHON'
import sys
try:
    import mcts_py
    
    # Check if OpenMP is enabled
    if hasattr(mcts_py, 'get_openmp_enabled'):
        enabled = mcts_py.get_openmp_enabled()
        if enabled:
            print("✅ OpenMP support compiled in")
        else:
            print("❌ OpenMP support NOT compiled in")
            sys.exit(1)
    else:
        print("⚠️  Warning: get_openmp_enabled() not available (old build?)")
    
    # Check thread count
    if hasattr(mcts_py, 'get_openmp_threads'):
        threads = mcts_py.get_openmp_threads()
        print(f"✅ OpenMP max threads: {threads}")
        if threads < 2:
            print(f"⚠️  Warning: Only {threads} OpenMP thread(s) available")
            print("   Set OMP_NUM_THREADS environment variable for more threads")
    else:
        print("⚠️  Warning: get_openmp_threads() not available (old build?)")
    
except ImportError as e:
    print(f"❌ ERROR: Failed to import mcts_py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ OPENMP VERIFICATION PASSED"
    echo "============================================================"
    exit 0
else
    echo ""
    echo "============================================================"
    echo "❌ OPENMP VERIFICATION FAILED"
    echo "============================================================"
    exit 1
fi
