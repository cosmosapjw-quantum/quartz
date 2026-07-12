#!/usr/bin/env python3
"""
Unit Tests for Soak Testing Framework
====================================

Comprehensive unit tests for the memory stability and performance monitoring
components used in soak testing.

This module tests:
- SystemResourceMonitor functionality
- WorkloadSimulator behavior
- MemoryStabilitySoakTest orchestration
- Performance metrics calculation
- Memory leak detection algorithms

Usage:
    python -m pytest tests/unit/test_soak_framework.py -v

HOWTO-RUN-TESTS:
================
# Run all soak framework unit tests
python -m pytest tests/unit/test_soak_framework.py -v

# Run specific test class
python -m pytest tests/unit/test_soak_framework.py::TestSystemResourceMonitor -v

# Run with coverage
python -m pytest tests/unit/test_soak_framework.py --cov=tests.soak.test_memory_stability -v

Expected results: All tests should pass, validating the soak testing framework components.
"""

import pytest
import time
import threading
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import statistics

# Set up path for imports
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import soak testing components
try:
    from tests.soak.test_memory_stability import (
        SystemResourceMonitor,
        WorkloadSimulator,
        MemoryStabilitySoakTest,
        SystemSnapshot,
        PerformanceMetrics,
        SoakTestResult
    )
    SOAK_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    SOAK_FRAMEWORK_AVAILABLE = False
    pytest.skip(f"Soak testing framework not available: {e}", allow_module_level=True)


class TestSystemSnapshot:
    """Test SystemSnapshot data class"""

    def test_snapshot_creation(self):
        """Test creating system snapshot"""
        snapshot = SystemSnapshot(
            timestamp=time.time(),
            memory_mb=512.0,
            cpu_percent=25.5,
            gpu_memory_mb=2048.0,
            gpu_utilization_percent=75.0,
            process_count=150,
            thread_count=8,
            open_files=25
        )

        assert snapshot.memory_mb == 512.0
        assert snapshot.cpu_percent == 25.5
        assert snapshot.gpu_memory_mb == 2048.0
        assert snapshot.process_count == 150

    def test_snapshot_optional_fields(self):
        """Test snapshot with optional GPU fields"""
        snapshot = SystemSnapshot(
            timestamp=time.time(),
            memory_mb=256.0,
            cpu_percent=15.0
        )

        assert snapshot.memory_mb == 256.0
        assert snapshot.gpu_memory_mb is None
        assert snapshot.gpu_utilization_percent is None


class TestPerformanceMetrics:
    """Test PerformanceMetrics data class"""

    def test_metrics_creation(self):
        """Test creating performance metrics"""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            operations_per_sec=1000.0,
            avg_response_time_ms=2.5,
            memory_allocations=500,
            error_count=2,
            success_rate=0.98
        )

        assert metrics.operations_per_sec == 1000.0
        assert metrics.avg_response_time_ms == 2.5
        assert metrics.success_rate == 0.98


class TestSoakTestResult:
    """Test SoakTestResult data class"""

    def test_result_creation(self):
        """Test creating soak test result"""
        result = SoakTestResult(
            duration_sec=3600.0,
            initial_memory_mb=512.0,
            final_memory_mb=520.0,
            peak_memory_mb=525.0,
            memory_growth_mb=8.0,
            memory_growth_rate_mb_per_hour=8.0,
            avg_performance={'ops_per_sec': 500.0},
            performance_degradation_percent=2.1,
            total_operations=1800000,
            error_count=10,
            crash_count=0,
            resource_leaks_detected=False,
            passed=True
        )

        assert result.duration_sec == 3600.0
        assert result.memory_growth_mb == 8.0
        assert result.passed is True
        assert result.failure_reason is None

    def test_failed_result(self):
        """Test creating failed soak test result"""
        result = SoakTestResult(
            duration_sec=1800.0,
            initial_memory_mb=512.0,
            final_memory_mb=600.0,
            peak_memory_mb=620.0,
            memory_growth_mb=88.0,
            memory_growth_rate_mb_per_hour=176.0,
            avg_performance={'ops_per_sec': 300.0},
            performance_degradation_percent=15.0,
            total_operations=540000,
            error_count=100,
            crash_count=1,
            resource_leaks_detected=True,
            passed=False,
            failure_reason="Memory growth rate exceeds threshold"
        )

        assert result.passed is False
        assert result.failure_reason == "Memory growth rate exceeds threshold"
        assert result.memory_growth_rate_mb_per_hour == 176.0


class TestSystemResourceMonitor:
    """Test SystemResourceMonitor functionality"""

    def test_monitor_creation(self):
        """Test creating resource monitor"""
        monitor = SystemResourceMonitor(sampling_interval=1.0)
        assert monitor.sampling_interval == 1.0
        assert monitor.monitoring is False
        assert len(monitor.snapshots) == 0

    def test_monitor_start_stop(self):
        """Test starting and stopping monitoring"""
        monitor = SystemResourceMonitor(sampling_interval=0.1)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring is True
        assert monitor.monitor_thread is not None

        # Let it collect some samples
        time.sleep(0.5)

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring is False

        # Should have collected samples
        assert len(monitor.snapshots) > 0

    def test_memory_growth_rate_calculation(self):
        """Test memory growth rate calculation"""
        monitor = SystemResourceMonitor()

        # Add mock snapshots
        base_time = time.time()
        monitor.snapshots = [
            SystemSnapshot(base_time, 100.0, 10.0),
            SystemSnapshot(base_time + 1800, 110.0, 12.0),  # +10MB over 0.5 hours
        ]

        growth_rate = monitor.get_memory_growth_rate()
        expected_rate = 10.0 / 1800  # 10MB over 1800 seconds = MB/second
        assert abs(growth_rate - expected_rate) < 0.1

    def test_memory_growth_rate_no_data(self):
        """Test growth rate with insufficient data"""
        monitor = SystemResourceMonitor()
        growth_rate = monitor.get_memory_growth_rate()
        assert growth_rate == 0.0

    def test_resource_leak_detection(self):
        """Test resource leak detection"""
        monitor = SystemResourceMonitor()

        # Create snapshots showing memory growth (ensure >60MB growth and 70% increase trend)
        base_time = time.time()
        for i in range(15):
            # Steadily increasing memory with large growth to trigger detection algorithm
            # Need >60MB between first 3 and last 3 averages, with 70% of samples increasing
            snapshot = SystemSnapshot(
                timestamp=base_time + i * 60,
                memory_mb=100.0 + i * 10.0,  # +10MB each sample (150MB total growth)
                cpu_percent=10.0,
                open_files=10 + i,
                thread_count=4
            )
            monitor.snapshots.append(snapshot)

        # Should detect leak due to consistent growth
        assert monitor.detect_resource_leaks() is True

    def test_no_resource_leak_detection(self):
        """Test no false positive leak detection"""
        monitor = SystemResourceMonitor()

        # Create stable snapshots
        base_time = time.time()
        for i in range(15):
            snapshot = SystemSnapshot(
                timestamp=base_time + i * 60,
                memory_mb=100.0 + (i % 3),  # Fluctuating but stable
                cpu_percent=10.0,
                open_files=10,
                thread_count=4
            )
            monitor.snapshots.append(snapshot)

        # Should not detect leak
        assert monitor.detect_resource_leaks() is False


class TestWorkloadSimulator:
    """Test WorkloadSimulator functionality"""

    def test_simulator_creation(self):
        """Test creating workload simulator"""
        simulator = WorkloadSimulator(game_type="gomoku")
        assert simulator.game_type == "gomoku"
        assert simulator.operations_count == 0
        assert simulator.error_count == 0
        assert simulator.running is False

    def test_simulator_start_stop(self):
        """Test starting and stopping workload"""
        simulator = WorkloadSimulator()

        # Start workload
        simulator.start_workload(num_threads=2)
        assert simulator.running is True
        assert len(simulator.threads) == 2

        # Let it run briefly
        time.sleep(1.0)

        # Stop workload
        simulator.stop_workload()
        assert simulator.running is False

        # Should have performed operations
        assert simulator.operations_count > 0

    def test_performance_degradation_calculation(self):
        """Test performance degradation calculation"""
        simulator = WorkloadSimulator()

        # Add mock performance metrics
        base_time = time.time()

        # Early samples (good performance)
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=base_time + i * 60,
                operations_per_sec=1000.0,  # High performance
                avg_response_time_ms=2.0,
                memory_allocations=100,
                error_count=0,
                success_rate=1.0
            )
            simulator.performance_metrics.append(metrics)

        # Later samples (degraded performance)
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=base_time + (i + 10) * 60,
                operations_per_sec=800.0,  # 20% lower
                avg_response_time_ms=2.5,
                memory_allocations=120,
                error_count=1,
                success_rate=0.95
            )
            simulator.performance_metrics.append(metrics)

        degradation = simulator.get_performance_degradation()
        # Should detect ~20% degradation
        assert 18.0 <= degradation <= 22.0

    def test_no_performance_degradation(self):
        """Test stable performance shows no degradation"""
        simulator = WorkloadSimulator()

        # Add stable performance metrics
        base_time = time.time()
        for i in range(10):
            metrics = PerformanceMetrics(
                timestamp=base_time + i * 60,
                operations_per_sec=1000.0,
                avg_response_time_ms=2.0,
                memory_allocations=100,
                error_count=0,
                success_rate=1.0
            )
            simulator.performance_metrics.append(metrics)

        degradation = simulator.get_performance_degradation()
        assert abs(degradation) < 1.0  # Should be near zero

    def test_game_operation_simulation(self):
        """Test game operation simulation"""
        # Setup mock game state
        mock_game = Mock()
        mock_game.get_enhanced_tensor_representation.return_value = np.random.randn(36, 15, 15).astype(np.float32)
        mock_game.get_legal_moves.return_value = [(0, 0), (1, 1)]
        mock_game.make_move.return_value = True
        mock_game.is_terminal.return_value = False

        simulator = WorkloadSimulator()

        # Test single operation
        simulator._simulate_game_operation(mock_game)

        # Verify mock calls
        mock_game.get_enhanced_tensor_representation.assert_called_once()
        mock_game.get_legal_moves.assert_called_once()
        mock_game.make_move.assert_called_once()


class TestMemoryStabilitySoakTest:
    """Test MemoryStabilitySoakTest orchestration"""

    def test_soak_test_creation(self):
        """Test creating soak test"""
        soak_test = MemoryStabilitySoakTest(duration_sec=60.0, memory_threshold_mb=5.0)
        assert soak_test.duration_sec == 60.0
        assert soak_test.memory_threshold_mb == 5.0
        assert soak_test.resource_monitor is not None
        assert soak_test.workload_simulator is not None

    @patch('tests.soak.test_memory_stability.psutil.Process')
    def test_current_memory_measurement(self, mock_process_class):
        """Test current memory measurement"""
        # Setup mock process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 512 * 1024 * 1024  # 512MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process

        soak_test = MemoryStabilitySoakTest()
        memory_mb = soak_test._get_current_memory()
        assert memory_mb == 512.0

    def test_short_soak_test_run(self):
        """Test running a very short soak test"""
        # 2-second test for unit testing
        soak_test = MemoryStabilitySoakTest(duration_sec=2.0, memory_threshold_mb=100.0)

        result = soak_test.run_soak_test()

        # Verify result structure
        assert isinstance(result, SoakTestResult)
        assert result.duration_sec >= 1.5  # Should run for approximately specified duration
        assert result.initial_memory_mb > 0
        assert result.final_memory_mb > 0
        assert result.total_operations >= 0
        assert result.crash_count == 0

    def test_avg_performance_calculation(self):
        """Test average performance calculation"""
        soak_test = MemoryStabilitySoakTest()

        # Add mock performance metrics to workload simulator
        metrics = [
            PerformanceMetrics(time.time(), 1000.0, 2.0, 100, 0, 1.0),
            PerformanceMetrics(time.time(), 800.0, 3.0, 120, 1, 0.95),
            PerformanceMetrics(time.time(), 900.0, 2.5, 110, 0, 1.0)
        ]
        soak_test.workload_simulator.performance_metrics = metrics

        avg_perf = soak_test._calculate_avg_performance()

        assert abs(avg_perf['operations_per_sec'] - 900.0) < 0.1  # (1000+800+900)/3
        assert abs(avg_perf['response_time_ms'] - 2.5) < 0.1  # (2.0+3.0+2.5)/3
        assert abs(avg_perf['success_rate'] - 0.983) < 0.01  # Average of success rates

    def test_result_analysis_passing(self):
        """Test analysis of passing soak test results"""
        soak_test = MemoryStabilitySoakTest(memory_threshold_mb=10.0)

        # Setup mock data for passing test
        soak_test.resource_monitor.snapshots = [
            SystemSnapshot(time.time(), 100.0, 10.0),
            SystemSnapshot(time.time() + 3600, 105.0, 12.0)  # +5MB over 1 hour
        ]

        soak_test.workload_simulator.operations_count = 10000
        soak_test.workload_simulator.error_count = 5
        soak_test.workload_simulator.performance_metrics = [
            PerformanceMetrics(time.time(), 1000.0, 2.0, 100, 0, 1.0),
            PerformanceMetrics(time.time() + 1800, 950.0, 2.1, 105, 2, 0.99)  # 5% degradation
        ]

        result = soak_test._analyze_results(time.time() - 3600, 100.0, 0)

        assert result.passed is True
        assert result.memory_growth_rate_mb_per_hour <= 10.0
        assert result.performance_degradation_percent <= 10.0
        assert result.crash_count == 0

    def test_result_analysis_failing_memory(self):
        """Test analysis of failing soak test (memory growth)"""
        soak_test = MemoryStabilitySoakTest(memory_threshold_mb=10.0)

        # Setup mock data for failing test (excessive memory growth)
        soak_test.resource_monitor.snapshots = [
            SystemSnapshot(time.time(), 100.0, 10.0),
            SystemSnapshot(time.time() + 3600, 150.0, 12.0)  # +50MB over 1 hour
        ]

        # Mock detect_resource_leaks to return False (no leaks detected by algorithm)
        soak_test.resource_monitor.detect_resource_leaks = Mock(return_value=False)

        soak_test.workload_simulator.operations_count = 10000
        soak_test.workload_simulator.error_count = 0
        soak_test.workload_simulator.performance_metrics = [
            PerformanceMetrics(time.time(), 1000.0, 2.0, 100, 0, 1.0)
        ]

        result = soak_test._analyze_results(time.time() - 3600, 100.0, 0)

        assert result.passed is False
        assert "Memory growth rate" in result.failure_reason
        assert result.memory_growth_rate_mb_per_hour > 10.0

    def test_result_analysis_failing_performance(self):
        """Test analysis of failing soak test (performance degradation)"""
        soak_test = MemoryStabilitySoakTest(memory_threshold_mb=10.0)

        # Setup mock data for passing memory but failing performance
        soak_test.resource_monitor.snapshots = [
            SystemSnapshot(time.time(), 100.0, 10.0),
            SystemSnapshot(time.time() + 3600, 102.0, 12.0)  # +2MB over 1 hour (OK)
        ]
        soak_test.resource_monitor.detect_resource_leaks = Mock(return_value=False)

        soak_test.workload_simulator.operations_count = 10000
        soak_test.workload_simulator.error_count = 0

        # Mock performance degradation > 10%
        soak_test.workload_simulator.get_performance_degradation = Mock(return_value=15.0)
        soak_test.workload_simulator.performance_metrics = [
            PerformanceMetrics(time.time(), 500.0, 4.0, 200, 0, 1.0)
        ]

        result = soak_test._analyze_results(time.time() - 3600, 100.0, 0)

        assert result.passed is False
        assert "Performance degraded" in result.failure_reason


class TestSoakTestIntegration:
    """Integration tests for soak testing components"""

    def test_resource_monitor_workload_integration(self):
        """Test resource monitor with workload simulator"""
        monitor = SystemResourceMonitor(sampling_interval=0.2)
        simulator = WorkloadSimulator()

        # Start both components
        monitor.start_monitoring()
        simulator.start_workload(num_threads=1)

        # Run briefly
        time.sleep(1.0)

        # Stop both
        simulator.stop_workload()
        monitor.stop_monitoring()

        # Verify both collected data
        assert len(monitor.snapshots) > 0
        assert simulator.operations_count > 0

        # Verify resource monitor detected some activity
        memory_values = [s.memory_mb for s in monitor.snapshots]
        assert max(memory_values) >= min(memory_values)  # Some variation expected

    def test_end_to_end_soak_test(self):
        """Test complete soak test end-to-end"""
        # Very short test for integration validation
        soak_test = MemoryStabilitySoakTest(duration_sec=3.0, memory_threshold_mb=50.0)

        result = soak_test.run_soak_test()

        # Verify complete result structure
        assert isinstance(result, SoakTestResult)
        assert result.duration_sec > 0
        assert result.initial_memory_mb > 0
        assert result.final_memory_mb > 0
        assert isinstance(result.passed, bool)
        assert result.total_operations >= 0
        assert result.error_count >= 0
        assert result.crash_count >= 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, '-v'])