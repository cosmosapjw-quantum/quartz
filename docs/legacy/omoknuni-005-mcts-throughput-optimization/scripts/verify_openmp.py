#!/usr/bin/env python3
"""
OpenMP Verification Script for MCTS Throughput Recovery

This script verifies that OpenMP is properly compiled, linked, and configured
for the MCTS C++ extensions. It performs comprehensive checks as specified in
tasks.md T002.

Usage:
    python scripts/verify_openmp.py
    python scripts/verify_openmp.py --verbose

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(message):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{message}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_check(name, passed, details=""):
    """Print check result."""
    status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{name:50} {status}")
    if details:
        print(f"  {Colors.YELLOW}{details}{Colors.END}")


def check_openmp_symbols():
    """
    Check 1: Verify OpenMP symbols in compiled extension.

    Looks for GOMP_* symbols in the mcts_py.so extension to confirm
    OpenMP was linked.
    """
    print_header("CHECK 1: OpenMP Symbols in Compiled Extension")

    # Find the .so file
    build_dir = Path("build")
    so_files = list(build_dir.glob("**/mcts_py*.so"))

    if not so_files:
        print_check("Find mcts_py*.so", False, "Extension not found. Run 'pip install -e .' first.")
        return False

    so_file = so_files[0]
    print(f"Extension: {so_file}")

    # Check for OpenMP symbols using nm
    try:
        result = subprocess.run(
            ["nm", str(so_file)],
            capture_output=True,
            text=True,
            check=True
        )

        # Look for GOMP symbols
        gomp_symbols = [line for line in result.stdout.split('\n') if 'GOMP' in line or 'omp' in line.lower()]

        if gomp_symbols:
            print_check("OpenMP symbols present", True, f"Found {len(gomp_symbols)} OpenMP symbols")
            if len(sys.argv) > 1 and sys.argv[1] == '--verbose':
                for sym in gomp_symbols[:5]:
                    print(f"    {sym}")
                if len(gomp_symbols) > 5:
                    print(f"    ... and {len(gomp_symbols) - 5} more")
            return True
        else:
            print_check("OpenMP symbols present", False, "No GOMP symbols found. OpenMP not linked.")
            return False

    except subprocess.CalledProcessError as e:
        print_check("OpenMP symbols present", False, f"nm command failed: {e}")
        return False
    except FileNotFoundError:
        print_check("OpenMP symbols present", False, "nm command not found. Install binutils.")
        return False


def check_openmp_runtime():
    """
    Check 2: Verify OpenMP runtime is accessible from Python.

    Tests that the C++ extension can report OpenMP thread count.
    """
    print_header("CHECK 2: OpenMP Runtime Accessibility")

    try:
        # Try to import the extension
        import mcts_py

        # Check if get_omp_max_threads exists
        if not hasattr(mcts_py, 'get_omp_max_threads'):
            print_check("get_omp_max_threads() available", False,
                       "Function not exposed. Check python_bindings.cpp")
            return False

        # Call the function
        max_threads = mcts_py.get_omp_max_threads()

        if max_threads > 1:
            print_check("OpenMP max threads", True, f"Max threads: {max_threads}")
            return True
        else:
            print_check("OpenMP max threads", False,
                       f"Max threads={max_threads}. OpenMP not active or OMP_NUM_THREADS=1")
            return False

    except ImportError as e:
        print_check("Import mcts_py", False, f"Cannot import extension: {e}")
        return False
    except Exception as e:
        print_check("OpenMP runtime check", False, f"Error: {e}")
        return False


def check_environment_variables():
    """
    Check 3: Verify OpenMP environment variables are set correctly.
    """
    print_header("CHECK 3: OpenMP Environment Variables")

    all_passed = True

    # OMP_NUM_THREADS
    omp_num_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_num_threads:
        try:
            num_threads = int(omp_num_threads)
            expected = 12  # Ryzen 5900X physical cores
            if num_threads == expected:
                print_check(f"OMP_NUM_THREADS={omp_num_threads}", True)
            else:
                print_check(f"OMP_NUM_THREADS={omp_num_threads}", True,
                           f"Warning: Expected {expected} for Ryzen 5900X")
        except ValueError:
            print_check(f"OMP_NUM_THREADS={omp_num_threads}", False, "Invalid value")
            all_passed = False
    else:
        print_check("OMP_NUM_THREADS", False, "Not set. Should be set to 12 for Ryzen 5900X")
        all_passed = False

    # OMP_PROC_BIND (recommended)
    omp_proc_bind = os.environ.get('OMP_PROC_BIND')
    if omp_proc_bind:
        if omp_proc_bind.lower() in ['close', 'spread', 'master']:
            print_check(f"OMP_PROC_BIND={omp_proc_bind}", True)
        else:
            print_check(f"OMP_PROC_BIND={omp_proc_bind}", True,
                       "Warning: Unusual value")
    else:
        print_check("OMP_PROC_BIND", True, "Not set (optional, but recommended='close')")

    # OMP_PLACES (recommended)
    omp_places = os.environ.get('OMP_PLACES')
    if omp_places:
        print_check(f"OMP_PLACES={omp_places}", True)
    else:
        print_check("OMP_PLACES", True, "Not set (optional, but recommended='cores')")

    # OMP_NESTED (should be false)
    omp_nested = os.environ.get('OMP_NESTED', '').upper()
    if omp_nested in ['FALSE', '0', '']:
        print_check("OMP_NESTED", True, "Correctly disabled (prevents MCTS + OpenMP conflict)")
    else:
        print_check("OMP_NESTED", False,
                   "Should be FALSE to prevent nested parallelism conflicts")
        all_passed = False

    return all_passed


def check_compiler_flags():
    """
    Check 4: Verify CMakeLists.txt has OpenMP flags.
    """
    print_header("CHECK 4: CMakeLists.txt OpenMP Configuration")

    cmake_file = Path("CMakeLists.txt")
    if not cmake_file.exists():
        print_check("CMakeLists.txt exists", False, "File not found")
        return False

    with open(cmake_file, 'r') as f:
        content = f.read()

    all_passed = True

    # Check for find_package(OpenMP)
    if 'find_package(OpenMP' in content:
        print_check("find_package(OpenMP)", True)
    else:
        print_check("find_package(OpenMP)", False, "Not found in CMakeLists.txt")
        all_passed = False

    # Check for OpenMP flags
    if '-fopenmp' in content or 'OpenMP::OpenMP' in content or '${OpenMP_CXX_FLAGS}' in content:
        print_check("OpenMP flags present", True)
    else:
        print_check("OpenMP flags present", False, "No -fopenmp or OpenMP:: found")
        all_passed = False

    return all_passed


def check_feature_extraction_performance():
    """
    Check 5: Benchmark feature extraction to verify OpenMP is active.

    This is a quick performance test that should show ~7× speedup
    if OpenMP is working (7.5ms → <1ms for batch-64).
    """
    print_header("CHECK 5: Feature Extraction Performance Test")

    try:
        import time
        import numpy as np

        # Import after checking it exists
        import mcts_py

        # Check if we have the test function
        if not hasattr(mcts_py, 'benchmark_feature_extraction'):
            print_check("benchmark_feature_extraction()", False,
                       "Function not exposed. Cannot test performance.")
            return False

        # Run benchmark (batch size 64, 10 iterations)
        print("Running benchmark...")
        times_ms = mcts_py.benchmark_feature_extraction(batch_size=64, iterations=10)

        mean_time = np.mean(times_ms)
        std_time = np.std(times_ms)

        print(f"  Mean time: {mean_time:.2f} ± {std_time:.2f} ms per batch-64")

        # Check against target
        target_ms = 1.0  # Target from spec
        if mean_time < target_ms:
            print_check("Feature extraction <1ms", True,
                       f"{mean_time:.2f}ms < {target_ms}ms (OpenMP working)")
            return True
        else:
            print_check("Feature extraction <1ms", False,
                       f"{mean_time:.2f}ms ≥ {target_ms}ms (OpenMP likely NOT active)")
            print(f"  {Colors.YELLOW}Expected: <1ms with OpenMP, ~7.5ms without{Colors.END}")
            return False

    except ImportError:
        print_check("Feature extraction test", False, "Cannot import mcts_py")
        return False
    except AttributeError as e:
        print_check("Feature extraction test", False, f"Function not available: {e}")
        return False
    except Exception as e:
        print_check("Feature extraction test", False, f"Error: {e}")
        return False


def print_recommendations(failed_checks):
    """Print recommendations based on failed checks."""
    if not failed_checks:
        return

    print_header("RECOMMENDATIONS")

    if 'symbols' in failed_checks:
        print(f"{Colors.YELLOW}OpenMP not linked. Try:{Colors.END}")
        print("  1. Install OpenMP: sudo apt-get install libomp-dev")
        print("  2. Rebuild with flags: export CXXFLAGS='-fopenmp' && pip install -e . --force-reinstall")
        print()

    if 'runtime' in failed_checks:
        print(f"{Colors.YELLOW}OpenMP runtime not accessible. Try:{Colors.END}")
        print("  1. Set OMP_NUM_THREADS: export OMP_NUM_THREADS=12")
        print("  2. Verify extension: python -c 'import mcts_py; print(mcts_py.get_omp_max_threads())'")
        print()

    if 'environment' in failed_checks:
        print(f"{Colors.YELLOW}Environment variables not configured. Try:{Colors.END}")
        print("  export OMP_NUM_THREADS=12")
        print("  export OMP_PROC_BIND=close")
        print("  export OMP_PLACES=cores")
        print("  export OMP_NESTED=FALSE")
        print()

    if 'cmake' in failed_checks:
        print(f"{Colors.YELLOW}CMakeLists.txt missing OpenMP config. Add:{Colors.END}")
        print("  find_package(OpenMP REQUIRED)")
        print("  if(OpenMP_CXX_FOUND)")
        print("    target_link_libraries(mcts_py PUBLIC OpenMP::OpenMP_CXX)")
        print("  endif()")
        print()

    if 'performance' in failed_checks:
        print(f"{Colors.YELLOW}Feature extraction too slow. This indicates OpenMP is NOT active.{Colors.END}")
        print("  Check all previous items and rebuild.")
        print()


def main():
    """Run all OpenMP verification checks."""
    print_header("OpenMP Verification for MCTS Throughput Recovery")

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")

    failed_checks = []

    # Run checks
    if not check_compiler_flags():
        failed_checks.append('cmake')

    if not check_openmp_symbols():
        failed_checks.append('symbols')

    if not check_openmp_runtime():
        failed_checks.append('runtime')

    if not check_environment_variables():
        failed_checks.append('environment')

    if not check_feature_extraction_performance():
        failed_checks.append('performance')

    # Summary
    print_header("VERIFICATION SUMMARY")

    if not failed_checks:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CHECKS PASSED{Colors.END}")
        print(f"{Colors.GREEN}OpenMP is properly configured and active.{Colors.END}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ {len(failed_checks)} CHECK(S) FAILED{Colors.END}")
        print(f"{Colors.RED}OpenMP is NOT properly configured.{Colors.END}")
        print_recommendations(failed_checks)
        return 1


if __name__ == "__main__":
    sys.exit(main())
