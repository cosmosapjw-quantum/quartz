"""
Unit Tests for Memory Leak Detection System (T051)
==================================================

Tests for the comprehensive memory leak detection system including Python profiling,
valgrind integration, GPU memory monitoring, and automated leak analysis.
"""

import json
import pytest
import tempfile
import threading
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the memory leak detection components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.check_memory_leaks import (
    MemorySnapshot,
    LeakDetectionResult,
    PythonMemoryProfiler,
    ValgrindIntegration,
    GPUMemoryProfiler,
    MemoryLeakDetector
)


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass functionality."""

    def test_memory_snapshot_creation(self):
        """Test MemorySnapshot creation and attributes."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            python_rss_mb=100.5,
            python_vms_mb=200.0,
            gpu_allocated_mb=50.0,
            process_count=3,
            thread_count=12,
            fd_count=25
        )

        assert snapshot.python_rss_mb == 100.5
        assert snapshot.python_vms_mb == 200.0
        assert snapshot.gpu_allocated_mb == 50.0
        assert snapshot.process_count == 3
        assert snapshot.thread_count == 12
        assert snapshot.fd_count == 25

    def test_memory_snapshot_defaults(self):
        """Test MemorySnapshot with default values."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            python_rss_mb=100.0,
            python_vms_mb=150.0
        )

        assert snapshot.gpu_allocated_mb == 0.0
        assert snapshot.process_count == 0
        assert snapshot.thread_count == 0
        assert snapshot.fd_count == 0


class TestPythonMemoryProfiler:
    """Test Python memory profiling functionality."""

    @pytest.fixture
    def profiler(self):
        """Create a Python memory profiler instance."""
        return PythonMemoryProfiler()

    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.snapshots == []
        assert profiler.process is not None
        assert profiler.tracemalloc_enabled == False

    @patch('tracemalloc.start')
    def test_start_profiling_success(self, mock_tracemalloc_start, profiler):
        """Test successful profiling start."""
        mock_tracemalloc_start.return_value = None

        profiler.start_profiling()

        assert profiler.tracemalloc_enabled == True
        mock_tracemalloc_start.assert_called_once()

    @patch('tracemalloc.start')
    def test_start_profiling_failure(self, mock_tracemalloc_start, profiler):
        """Test profiling start failure handling."""
        mock_tracemalloc_start.side_effect = RuntimeError("Tracemalloc error")

        profiler.start_profiling()

        assert profiler.tracemalloc_enabled == False

    @patch('psutil.Process')
    def test_take_snapshot(self, mock_process_class, profiler):
        """Test taking memory snapshots."""
        # Mock process memory info
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024**2  # 100 MB
        mock_memory_info.vms = 200 * 1024**2  # 200 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.children.return_value = []
        mock_process.num_threads.return_value = 5
        mock_process.num_fds.return_value = 10
        mock_process_class.return_value = mock_process

        profiler.process = mock_process

        snapshot = profiler.take_snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.python_rss_mb == 100.0
        assert snapshot.python_vms_mb == 200.0
        assert snapshot.process_count == 1  # Self + children
        assert snapshot.thread_count == 5
        assert snapshot.fd_count == 10
        assert len(profiler.snapshots) == 1

    def test_analyze_growth_no_data(self, profiler):
        """Test growth analysis with no data."""
        leak_detected, growth_rate = profiler.analyze_growth()

        assert leak_detected == False
        assert growth_rate == 0.0

    def test_analyze_growth_with_growth(self, profiler):
        """Test growth analysis detecting memory growth."""
        # Add snapshots with increasing memory usage
        base_time = time.time()
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + i * 60,  # 1 minute intervals
                python_rss_mb=100.0 + i * 20,  # 20 MB increase per minute
                python_vms_mb=200.0
            )
            profiler.snapshots.append(snapshot)

        leak_detected, growth_rate = profiler.analyze_growth(threshold_mb_per_hour=50.0)

        assert leak_detected == True  # 20 MB/min = 1200 MB/hour > 50 MB/hour threshold
        assert abs(growth_rate - 1200.0) < 100.0  # Allow some tolerance

    def test_analyze_growth_no_leak(self, profiler):
        """Test growth analysis with stable memory usage."""
        # Add snapshots with stable memory usage
        base_time = time.time()
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=base_time + i * 60,
                python_rss_mb=100.0 + (i % 2),  # Slight oscillation
                python_vms_mb=200.0
            )
            profiler.snapshots.append(snapshot)

        leak_detected, growth_rate = profiler.analyze_growth(threshold_mb_per_hour=10.0)

        assert leak_detected == False
        assert abs(growth_rate) < 10.0


class TestValgrindIntegration:
    """Test valgrind integration functionality."""

    @pytest.fixture
    def valgrind_integration(self):
        """Create a valgrind integration instance."""
        return ValgrindIntegration()

    @patch('subprocess.run')
    def test_valgrind_available_check_success(self, mock_run):
        """Test successful valgrind availability check."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "valgrind-3.18.1"
        mock_run.return_value = mock_result

        integration = ValgrindIntegration()

        assert integration.valgrind_available == True
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_valgrind_not_available(self, mock_run):
        """Test valgrind not available handling."""
        mock_run.side_effect = FileNotFoundError()

        integration = ValgrindIntegration()

        assert integration.valgrind_available == False

    def test_run_valgrind_check_not_available(self, valgrind_integration):
        """Test valgrind check when valgrind is not available."""
        valgrind_integration.valgrind_available = False

        result = valgrind_integration.run_valgrind_check('mcts')

        assert 'error' in result
        assert result['error'] == 'Valgrind not available'

    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    def test_run_valgrind_check_success(self, mock_temp_file, mock_run, valgrind_integration):
        """Test successful valgrind check execution."""
        valgrind_integration.valgrind_available = True

        # Mock temporary file
        mock_file = Mock()
        mock_file.name = '/tmp/test.valgrind'
        mock_temp_file.return_value.__enter__.return_value = mock_file

        # Mock subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock file reading
        with patch('builtins.open', mock_open_read_data="==123== HEAP SUMMARY:\n==123== ERROR SUMMARY: 0 errors from 0 contexts"):
            result = valgrind_integration.run_valgrind_check('python_bindings')

        assert result['component'] == 'python_bindings'
        assert result['return_code'] == 0
        assert 'analysis' in result

    def test_parse_valgrind_output(self, valgrind_integration):
        """Test valgrind output parsing."""
        sample_output = """
==12345== HEAP SUMMARY:
==12345==     in use at exit: 1,024 bytes in 8 blocks
==12345== LEAK SUMMARY:
==12345==    definitely lost: 512 bytes in 4 blocks
==12345==    indirectly lost: 256 bytes in 2 blocks
==12345==    possibly lost: 128 bytes in 1 blocks
==12345== ERROR SUMMARY: 3 errors from 3 contexts
"""

        analysis = valgrind_integration._parse_valgrind_output(sample_output)

        assert analysis['error_count'] == 3
        assert analysis['heap_summary']['bytes_in_use'] == 1024
        assert analysis['heap_summary']['blocks_in_use'] == 8
        assert analysis['leaks_definitely'] == 512
        assert analysis['leaks_indirectly'] == 256
        assert analysis['leaks_possible'] == 128


def mock_open_read_data(data):
    """Helper to mock file reading."""
    from unittest.mock import mock_open
    return mock_open(read_data=data)


class TestGPUMemoryProfiler:
    """Test GPU memory profiling functionality."""

    @pytest.fixture
    def gpu_profiler(self):
        """Create a GPU memory profiler instance."""
        return GPUMemoryProfiler()

    def test_profiler_initialization_no_cuda(self, gpu_profiler):
        """Test profiler initialization without CUDA."""
        # Assuming CUDA is not available in test environment
        assert gpu_profiler.gpu_snapshots == []

    @patch('scripts.check_memory_leaks.TORCH_AVAILABLE', True)
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_start_monitoring_success(self, mock_empty_cache, mock_cuda_available, gpu_profiler):
        """Test successful GPU monitoring start."""
        mock_cuda_available.return_value = True
        gpu_profiler.cuda_available = True

        result = gpu_profiler.start_monitoring()

        assert result == True
        mock_empty_cache.assert_called_once()

    @patch('scripts.check_memory_leaks.TORCH_AVAILABLE', True)
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.max_memory_allocated')
    def test_take_gpu_snapshot(self, mock_max_alloc, mock_reserved, mock_allocated, mock_cuda_available, gpu_profiler):
        """Test taking GPU memory snapshots."""
        mock_cuda_available.return_value = True
        mock_allocated.return_value = 100 * 1024**2  # 100 MB
        mock_reserved.return_value = 200 * 1024**2   # 200 MB
        mock_max_alloc.return_value = 150 * 1024**2  # 150 MB
        gpu_profiler.cuda_available = True

        snapshot = gpu_profiler.take_gpu_snapshot()

        assert snapshot['allocated_mb'] == 100.0
        assert snapshot['cached_mb'] == 200.0
        assert snapshot['max_allocated_mb'] == 150.0
        assert len(gpu_profiler.gpu_snapshots) == 1

    def test_analyze_gpu_leaks_no_data(self, gpu_profiler):
        """Test GPU leak analysis with no data."""
        leaks = gpu_profiler.analyze_gpu_leaks()

        assert leaks == []

    def test_analyze_gpu_leaks_with_growth(self, gpu_profiler):
        """Test GPU leak detection with memory growth."""
        # Add snapshots showing consistent memory growth
        base_time = time.time()
        for i in range(10):
            snapshot = {
                'timestamp': base_time + i * 10,
                'allocated_mb': 100.0 + i * 20,  # 20 MB increase per snapshot
                'cached_mb': 200.0,
                'max_allocated_mb': 150.0 + i * 20
            }
            gpu_profiler.gpu_snapshots.append(snapshot)

        leaks = gpu_profiler.analyze_gpu_leaks()

        assert len(leaks) >= 1
        leak = leaks[0]
        assert leak['type'] == 'gpu_memory_leak'
        assert leak['total_growth_mb'] > 100.0  # Significant growth
        assert leak['severity'] in ['medium', 'high']


class TestMemoryLeakDetector:
    """Test main memory leak detector functionality."""

    @pytest.fixture
    def detector(self):
        """Create a memory leak detector instance."""
        return MemoryLeakDetector()

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.python_profiler is not None
        assert detector.valgrind is not None
        assert detector.gpu_profiler is not None

    def test_default_workload(self, detector):
        """Test default workload execution."""
        # Run workload for a short duration
        start_time = time.time()
        detector._run_default_workload(1.0)  # 1 second
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time >= 1.0
        assert end_time - start_time < 2.0  # Allow some overhead

    def test_generate_recommendations_no_leaks(self, detector):
        """Test recommendation generation with no leaks."""
        recommendations = detector._generate_recommendations(
            leak_detected=False,
            growth_rate=2.0,
            valgrind_reports=[],
            gpu_leaks=[]
        )

        assert len(recommendations) >= 2
        assert "No significant memory leaks detected" in recommendations[0]

    def test_generate_recommendations_with_leaks(self, detector):
        """Test recommendation generation with detected leaks."""
        valgrind_reports = [{
            'component': 'mcts',
            'analysis': {'leaks_definitely': 1024}
        }]
        gpu_leaks = [{
            'type': 'gpu_memory_leak',
            'total_growth_mb': 50.0
        }]

        recommendations = detector._generate_recommendations(
            leak_detected=True,
            growth_rate=25.0,
            valgrind_reports=valgrind_reports,
            gpu_leaks=gpu_leaks
        )

        assert len(recommendations) >= 3
        assert any("Python memory leak detected" in rec for rec in recommendations)
        assert any("C++ memory leak" in rec for rec in recommendations)
        assert any("GPU memory leak" in rec for rec in recommendations)

    @patch.object(PythonMemoryProfiler, 'start_profiling')
    @patch.object(PythonMemoryProfiler, 'stop_profiling')
    @patch.object(PythonMemoryProfiler, 'analyze_growth')
    @patch.object(GPUMemoryProfiler, 'start_monitoring')
    @patch.object(GPUMemoryProfiler, 'analyze_gpu_leaks')
    @patch.object(ValgrindIntegration, 'run_valgrind_check')
    def test_comprehensive_check(self, mock_valgrind, mock_gpu_leaks, mock_gpu_start,
                                mock_analyze_growth, mock_stop_profiling, mock_start_profiling, detector):
        """Test comprehensive memory leak check."""
        # Mock return values
        mock_gpu_start.return_value = True
        mock_analyze_growth.return_value = (False, 5.0)
        mock_stop_profiling.return_value = {'total_snapshots': 10}
        mock_gpu_leaks.return_value = []
        mock_valgrind.return_value = {'component': 'test', 'analysis': {}}

        # Add some mock snapshots
        detector.python_profiler.snapshots = [
            MemorySnapshot(timestamp=time.time(), python_rss_mb=100.0, python_vms_mb=150.0),
            MemorySnapshot(timestamp=time.time() + 10, python_rss_mb=105.0, python_vms_mb=155.0)
        ]

        result = detector.run_comprehensive_check(
            duration_seconds=1.0,
            sample_interval=0.5,
            components=['test_component']
        )

        assert isinstance(result, LeakDetectionResult)
        assert result.duration_seconds == 1.0
        assert result.growth_rate_mb_per_hour == 5.0
        assert result.leak_detected == False
        assert len(result.recommendations) > 0


class TestIntegration:
    """Integration tests for the memory leak detection system."""

    def test_script_import(self):
        """Test that the script can be imported without errors."""
        import scripts.check_memory_leaks as leak_detector
        assert hasattr(leak_detector, 'MemoryLeakDetector')
        assert hasattr(leak_detector, 'main')

    @patch('sys.argv', ['check_memory_leaks.py', '--python', '--duration', '0.5'])
    @patch('scripts.check_memory_leaks.MemoryLeakDetector')
    def test_main_function_python_mode(self, mock_detector_class):
        """Test main function in Python profiling mode."""
        # Mock detector and its methods
        mock_detector = Mock()
        mock_result = Mock()
        mock_result.leak_detected = False
        mock_result.duration_seconds = 0.5
        mock_result.total_samples = 5
        mock_result.memory_growth_mb = 2.0
        mock_result.growth_rate_mb_per_hour = 5.0
        mock_result.components_tested = ['skip_valgrind']
        mock_result.gpu_leaks = []
        mock_result.recommendations = ["No leaks detected"]

        mock_detector.run_comprehensive_check.return_value = mock_result
        mock_detector_class.return_value = mock_detector

        # Import and run main (should not raise exception)
        from scripts.check_memory_leaks import main
        try:
            main()
        except SystemExit as e:
            # Should exit with success code (0) when no leaks detected
            assert e.code == 0

    def test_memory_snapshot_serialization(self):
        """Test that memory snapshots can be serialized to JSON."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            python_rss_mb=100.0,
            python_vms_mb=150.0,
            gpu_allocated_mb=50.0,
            process_count=3,
            thread_count=10,
            fd_count=20
        )

        # Convert to dict and serialize
        from dataclasses import asdict
        snapshot_dict = asdict(snapshot)
        json_str = json.dumps(snapshot_dict)

        # Should not raise exception
        assert json_str is not None
        assert len(json_str) > 0

        # Deserialize and check
        restored = json.loads(json_str)
        assert restored['python_rss_mb'] == 100.0
        assert restored['gpu_allocated_mb'] == 50.0


# Additional helper functions for testing
def create_test_memory_leak():
    """Create a controlled memory leak for testing purposes."""
    leaked_data = []
    for _ in range(100):
        leaked_data.append(np.random.randn(1000))
    return leaked_data


def simulate_gpu_usage():
    """Simulate GPU usage for testing (if CUDA available)."""
    try:
        import torch
        if torch.cuda.is_available():
            tensors = [torch.randn(100, 100).cuda() for _ in range(10)]
            result = sum(t.sum() for t in tensors)
            return result.item()
    except ImportError:
        pass
    return 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])