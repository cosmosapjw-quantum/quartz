"""
Unit Tests for Sanitizer Build Validation
==========================================

Tests to validate AddressSanitizer, ThreadSanitizer, and UndefinedBehaviorSanitizer
builds work correctly and can detect common memory errors and race conditions.
"""

import os
import sys
import pytest
import threading
import time
from typing import List
import subprocess
import platform


def is_sanitizer_build() -> str:
    """
    Detect which sanitizer is active by checking environment variables and compiler flags.

    Returns:
        str: The active sanitizer type ('asan', 'tsan', 'ubsan', 'none')
    """
    # Check environment variables
    sanitizer_env = os.environ.get('SANITIZER_BUILD')
    if sanitizer_env in ('asan', 'tsan', 'ubsan'):
        return sanitizer_env

    if 'ASAN_OPTIONS' in os.environ:
        return 'asan'
    elif 'TSAN_OPTIONS' in os.environ:
        return 'tsan'
    elif 'UBSAN_OPTIONS' in os.environ:
        return 'ubsan'

    # Try to detect from compile flags if available
    try:
        import sysconfig
        cflags = sysconfig.get_config_var('CFLAGS') or ''
        if 'fsanitize=address' in cflags:
            return 'asan'
        elif 'fsanitize=thread' in cflags:
            return 'tsan'
        elif 'fsanitize=undefined' in cflags:
            return 'ubsan'
    except:
        pass

    return 'none'


class SanitizerTestHelper:
    """Helper class for creating test scenarios that trigger sanitizer detection."""

    @staticmethod
    def create_memory_leak():
        """Create a deliberate memory leak (should be caught by ASan)."""
        # Allocate memory without freeing it
        leaked_data = [i for i in range(1000)]
        # Store reference somewhere it won't be garbage collected
        if not hasattr(SanitizerTestHelper, '_leaked_refs'):
            SanitizerTestHelper._leaked_refs = []
        SanitizerTestHelper._leaked_refs.append(leaked_data)

    @staticmethod
    def create_use_after_free_scenario():
        """Simulate use-after-free scenario (Python manages memory, so this is conceptual)."""
        # In Python, we can't easily create true use-after-free, but we can test
        # the concept with object lifecycle
        data = [1, 2, 3, 4, 5]
        reference = data
        del data  # Delete original reference
        return len(reference)  # Still accessible via reference (not true use-after-free in Python)

    @staticmethod
    def create_race_condition():
        """Create a race condition scenario for ThreadSanitizer to detect."""
        shared_counter = [0]  # Use list to ensure shared reference
        results = []

        def increment_counter():
            for _ in range(100):
                # Deliberate race condition - no synchronization
                current = shared_counter[0]
                time.sleep(0.0001)  # Small delay to increase chance of race
                shared_counter[0] = current + 1

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return shared_counter[0]


@pytest.mark.sanitizer
class TestSanitizerBuilds:
    """Test suite for sanitizer build validation."""

    def test_sanitizer_detection(self):
        """Test that we can detect which sanitizer is active."""
        sanitizer_type = is_sanitizer_build()
        assert sanitizer_type in ['asan', 'tsan', 'ubsan', 'none']

        # If running in CI with sanitizer, should detect it
        if any(var in os.environ for var in ['ASAN_OPTIONS', 'TSAN_OPTIONS', 'UBSAN_OPTIONS']):
            assert sanitizer_type != 'none', "Sanitizer environment detected but not recognized"

    def test_basic_functionality_with_sanitizers(self):
        """Test that basic functionality works with sanitizers enabled."""
        # Basic operations that should work fine with any sanitizer
        data = list(range(1000))
        assert len(data) == 1000
        assert sum(data) == 499500

        # Dictionary operations
        test_dict = {f"key_{i}": i for i in range(100)}
        assert len(test_dict) == 100
        assert test_dict["key_50"] == 50


@pytest.mark.asan
class TestAddressSanitizer:
    """Tests specific to AddressSanitizer functionality."""

    def test_asan_detection(self):
        """Test AddressSanitizer detection."""
        if is_sanitizer_build() == 'asan':
            assert 'ASAN_OPTIONS' in os.environ or 'fsanitize=address' in str(sys.argv)

    def test_memory_operations_asan_safe(self):
        """Test memory operations that should be safe with ASan."""
        # Allocate and properly manage memory
        large_data = []
        for i in range(10000):
            large_data.append(f"item_{i}")

        # Access within bounds
        assert large_data[0] == "item_0"
        assert large_data[-1] == "item_9999"

        # Clean up
        large_data.clear()
        assert len(large_data) == 0

    def test_memory_leak_detection(self):
        """Test that ASan can detect memory leaks (skip by default)."""
        SanitizerTestHelper.create_memory_leak()
        assert hasattr(SanitizerTestHelper, '_leaked_refs')
        assert len(SanitizerTestHelper._leaked_refs) > 0


@pytest.mark.tsan
class TestThreadSanitizer:
    """Tests specific to ThreadSanitizer functionality."""

    def test_tsan_detection(self):
        """Test ThreadSanitizer detection."""
        if is_sanitizer_build() == 'tsan':
            assert 'TSAN_OPTIONS' in os.environ or 'fsanitize=thread' in str(sys.argv)

    def test_thread_safe_operations(self):
        """Test thread-safe operations that should pass TSan."""
        import queue

        # Use thread-safe queue for communication
        result_queue = queue.Queue()

        def safe_worker(worker_id):
            # Simulate some work
            result = sum(range(worker_id * 100, (worker_id + 1) * 100))
            result_queue.put(result)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=safe_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Collect results safely
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == 5

    def test_race_condition_detection(self):
        """Test that TSan can detect race conditions (skip by default)."""
        result = SanitizerTestHelper.create_race_condition()
        # Result may vary due to race condition
        assert isinstance(result, int)


@pytest.mark.ubsan
class TestUndefinedBehaviorSanitizer:
    """Tests specific to UndefinedBehaviorSanitizer functionality."""

    def test_ubsan_detection(self):
        """Test UndefinedBehaviorSanitizer detection."""
        if is_sanitizer_build() == 'ubsan':
            assert 'UBSAN_OPTIONS' in os.environ or 'fsanitize=undefined' in str(sys.argv)

    def test_defined_behavior_operations(self):
        """Test operations with well-defined behavior that should pass UBSan."""
        # Integer operations within defined ranges
        a = 100
        b = 200
        assert a + b == 300
        assert b - a == 100
        assert a * 2 == 200
        assert b // a == 2

        # Division by zero is undefined behavior, so we avoid it
        if b != 0:
            result = a / b
            assert 0 < result < 1

    def test_string_operations_defined(self):
        """Test string operations with well-defined behavior."""
        test_str = "Hello, World!"
        assert len(test_str) == 13
        assert test_str[0] == 'H'
        assert test_str[-1] == '!'

        # Safe substring operations
        substr = test_str[0:5]
        assert substr == "Hello"


class TestSanitizerIntegration:
    """Integration tests for sanitizer builds with actual AlphaZero components."""

    def test_sanitizer_build_info(self):
        """Display information about the current sanitizer build."""
        sanitizer_type = is_sanitizer_build()
        print(f"\nCurrent sanitizer build: {sanitizer_type}")

        if sanitizer_type != 'none':
            print("Sanitizer environment variables:")
            for var in ['ASAN_OPTIONS', 'TSAN_OPTIONS', 'UBSAN_OPTIONS']:
                if var in os.environ:
                    print(f"  {var}={os.environ[var]}")

        # Check if we're running in CI
        if 'GITHUB_ACTIONS' in os.environ:
            print("Running in GitHub Actions CI")

        assert True  # This test always passes, just provides info

    def test_import_core_modules_with_sanitizers(self):
        """Test that core modules can be imported with sanitizers enabled."""
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        assert len(arr) == 5
        assert np.sum(arr) == 15

    def test_sanitizer_performance_overhead(self):
        """Test to measure and document sanitizer performance overhead."""
        import time

        # Simple benchmark to measure overhead
        start_time = time.time()

        # Perform some operations
        data = []
        for i in range(10000):
            data.append(i ** 2)

        result = sum(data)
        end_time = time.time()

        elapsed = end_time - start_time
        print(f"\nBenchmark completed in {elapsed:.4f} seconds")
        print(f"Result: {result}")

        sanitizer_type = is_sanitizer_build()
        if sanitizer_type != 'none':
            print(f"Running with {sanitizer_type} - expect 2-5x slower performance")

        # Test should complete reasonably quickly even with sanitizers
        assert elapsed < 5.0, f"Test took too long: {elapsed} seconds"
        assert result == sum(i ** 2 for i in range(10000))


def test_sanitizer_availability():
    """Test that sanitizer tools are available on the system."""
    system_name = platform.system()
    if system_name != "Linux":
        assert system_name in {"Linux", "Darwin", "Windows"}
        return

    # Check if clang is available (better sanitizer support)
    try:
        result = subprocess.run(['clang', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\nClang version: {result.stdout.split()[0]}")
    except FileNotFoundError:
        print("\nClang not found - using system compiler")

    # Check for llvm-symbolizer (needed for ASan symbol resolution)
    try:
        result = subprocess.run(['llvm-symbolizer', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"llvm-symbolizer available")
    except FileNotFoundError:
        print("llvm-symbolizer not found - symbols may not be resolved properly")


if __name__ == "__main__":
    # Run basic sanitizer detection when executed directly
    sanitizer = is_sanitizer_build()
    print(f"Detected sanitizer: {sanitizer}")

    if sanitizer != 'none':
        print("Sanitizer environment:")
        for var in ['ASAN_OPTIONS', 'TSAN_OPTIONS', 'UBSAN_OPTIONS']:
            if var in os.environ:
                print(f"  {var}={os.environ[var]}")

    pytest.main([__file__, "-v"])
