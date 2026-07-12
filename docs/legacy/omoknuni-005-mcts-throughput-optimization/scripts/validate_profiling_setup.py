#!/usr/bin/env python3
"""
Simple Profiling Setup Validator
=================================

Validates that C++ profiling is compiled and Python profiling is available.
Does NOT require MCTS or neural network imports.

Author: MCTS Performance Team
Date: 2025-10-15
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def validate_cpp_profiling():
    """Verify C++ profiling is compiled in"""
    print("\n" + "="*60)
    print("VALIDATING C++ PROFILING")
    print("="*60)
    print()

    # Check if mcts_py module exists
    try:
        import mcts_py
    except ImportError:
        print("❌ ERROR: mcts_py module not found!")
        print()
        print("Build the C++ extensions:")
        print('  export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"')
        print("  pip install -e . --force-reinstall --no-deps")
        return False

    # Check if EnhancedProfiler is available
    if not hasattr(mcts_py, 'EnhancedProfiler'):
        print("❌ ERROR: EnhancedProfiler not found in mcts_py!")
        print()
        print("This means PROFILE_LEVEL_VALUE was NOT set during build.")
        print()
        print("Rebuild with profiling enabled:")
        print('  export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"')
        print('  rm -rf build/ *.so')
        print("  pip install -e . --force-reinstall --no-deps")
        return False

    # Try to instantiate profiler
    try:
        profiler = mcts_py.EnhancedProfiler.instance()
        profiler.set_enabled(True)
        print("✅ C++ EnhancedProfiler available and functional")
    except Exception as e:
        print(f"❌ ERROR: Could not instantiate EnhancedProfiler: {e}")
        return False

    return True

def validate_python_profiling():
    """Verify Python profiling modules are available"""
    print("\n" + "="*60)
    print("VALIDATING PYTHON PROFILING")
    print("="*60)
    print()

    # Check profiling decorators
    try:
        from src.profiling.decorators import (
            set_profiling_enabled,
            get_profiling_summary,
            profile_function
        )
        print("✅ Profiling decorators available")
    except ImportError as e:
        print(f"❌ ERROR: Could not import profiling decorators: {e}")
        return False

    # Check GIL profiler
    try:
        from src.profiling.gil_profiler import GILProfiler
        print("✅ GIL profiler available")
    except ImportError as e:
        print(f"❌ ERROR: Could not import GIL profiler: {e}")
        return False

    # Check unified context
    try:
        from src.profiling.unified_context import UnifiedProfilingContext
        print("✅ Unified profiling context available")
    except ImportError as e:
        print(f"⚠️  WARNING: Could not import unified context: {e}")
        print("   (Optional - not critical)")

    return True

def main():
    """Run all validation checks"""
    print("\n" + "="*60)
    print("PROFILING SETUP VALIDATION")
    print("="*60)
    print()
    print("Checking if profiling is properly configured...")
    print()

    cpp_ok = validate_cpp_profiling()
    python_ok = validate_python_profiling()

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print()

    if cpp_ok and python_ok:
        print("✅ PASS: Profiling is correctly configured!")
        print()
        print("Next steps:")
        print("  1. Run quick test:")
        print("     ./scripts/run_profiling_suite.sh --quick")
        print()
        print("  2. Run full profiling:")
        print("     ./scripts/run_profiling_suite.sh --full")
        print()
        return 0
    else:
        print("❌ FAIL: Profiling setup is incomplete")
        print()
        if not cpp_ok:
            print("CRITICAL: C++ profiling not enabled")
            print()
            print("Fix:")
            print("  1. Set profiling flags:")
            print('     export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"')
            print()
            print("  2. Clean and rebuild:")
            print("     rm -rf build/ *.so")
            print("     pip install -e . --force-reinstall --no-deps")
            print()
            print("  3. Validate again:")
            print("     python scripts/validate_profiling_setup.py")
            print()

        if not python_ok:
            print("ERROR: Python profiling modules not available")
            print()
            print("This is unusual - check your installation:")
            print("  pip install -e .")
            print()

        return 1

if __name__ == "__main__":
    sys.exit(main())
