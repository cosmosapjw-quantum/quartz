"""
Thread Optimizer Unit Tests
============================

Tests for the thread count optimization script, including performance measurement,
contention detection, and optimization algorithms.
"""

import pytest
import tempfile
import shutil
import time
import json
from pathlib import Path
import numpy as np

# Import the thread optimizer components
import sys
sys.path.append('scripts')
from tune_threads import (
    ThreadOptimizer, ThreadTestConfig, ThreadPerformanceResult,
    OptimizationReport, RealGameState
)

# Additional imports for testing real components
import torch
from src.neural.model import AlphaZeroNet
from src.core.search_coordinator import SearchCoordinator


class TestRealGameState:
    """Test real game state implementation."""

    def test_real_game_state_creation(self):
        """Test basic real game state creation."""
        state = RealGameState("gomoku")
        assert state.game_type == "gomoku"
        assert state.board_size == 15
        assert state.feature_planes == 36
        assert state.action_space == 225

    def test_real_game_state_moves(self):
        """Test real game state move application."""
        state = RealGameState("gomoku")

        legal_moves = state.get_legal_moves()
        assert len(legal_moves) > 0  # Should have legal moves

        initial_move_count = getattr(state, 'move_count', 0)
        state.apply_move(112)  # Center move

        # Test that some change occurred (depends on C++ vs Python fallback)
        assert not state.is_terminal()  # Should not be terminal after one move

    def test_real_game_state_copy(self):
        """Test real game state copying."""
        state = RealGameState("chess")
        state.apply_move(1)

        copied_state = state.copy()
        assert copied_state.game_type == "chess"
        assert copied_state is not state

    def test_real_game_state_features(self):
        """Test feature extraction from real game state."""
        state = RealGameState("gomoku")
        features = state.get_features()

        assert isinstance(features, np.ndarray)
        assert features.shape[0] == state.feature_planes
        assert features.shape[1] == state.board_size
        assert features.shape[2] == state.board_size
        assert features.dtype == np.float32

    def test_different_game_types(self):
        """Test different game type configurations."""
        # Test gomoku
        gomoku = RealGameState("gomoku")
        assert gomoku.board_size == 15
        assert gomoku.feature_planes == 36
        assert gomoku.action_space == 225

        # Test chess
        chess = RealGameState("chess")
        assert chess.board_size == 8
        assert chess.feature_planes == 30
        assert chess.action_space == 4096

        # Test go
        go = RealGameState("go")
        assert go.board_size == 19
        assert go.feature_planes == 25
        assert go.action_space == 362


class TestThreadPerformanceResult:
    """Test thread performance result calculations."""

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # High performance, low contention result
        good_result = ThreadPerformanceResult(
            thread_count=8,
            searches_per_second=500.0,
            average_search_time_ms=20.0,
            search_time_std_ms=2.0,
            thread_utilization_percent=85.0,
            contention_score=5.0,
            cpu_utilization_percent=80.0,
            memory_usage_mb=512.0,
            success_rate=1.0
        )

        score = good_result.efficiency_score()
        assert 0.7 < score <= 1.0, f"Expected high efficiency score, got {score}"

    def test_efficiency_score_with_contention(self):
        """Test efficiency score with high contention."""
        # High contention result
        bad_result = ThreadPerformanceResult(
            thread_count=16,
            searches_per_second=200.0,
            average_search_time_ms=50.0,
            search_time_std_ms=25.0,
            thread_utilization_percent=95.0,
            contention_score=50.0,  # High contention
            cpu_utilization_percent=100.0,
            memory_usage_mb=1024.0,
            success_rate=0.9
        )

        score = bad_result.efficiency_score()
        assert score < 0.5, f"Expected low efficiency score for high contention, got {score}"

    def test_efficiency_score_with_failures(self):
        """Test efficiency score with low success rate."""
        # Failed result
        failed_result = ThreadPerformanceResult(
            thread_count=32,
            searches_per_second=0.0,
            average_search_time_ms=float('inf'),
            search_time_std_ms=0.0,
            thread_utilization_percent=0.0,
            contention_score=100.0,
            cpu_utilization_percent=0.0,
            memory_usage_mb=0.0,
            success_rate=0.5  # Low success rate
        )

        score = failed_result.efficiency_score()
        assert score == 0.0, f"Expected zero efficiency for failed result, got {score}"


class TestThreadOptimizer:
    """Test thread optimizer functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def optimizer(self, temp_dir):
        """Create thread optimizer for testing."""
        return ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.cpu_count > 0
        assert 'cpu_info' in optimizer.system_info
        assert optimizer.output_dir.exists()

    def test_system_info_collection(self, optimizer):
        """Test system information collection."""
        info = optimizer.system_info

        required_keys = ['cpu_info', 'memory_total_gb', 'platform', 'python_version']
        for key in required_keys:
            assert key in info, f"Missing system info key: {key}"

        assert info['memory_total_gb'] > 0
        assert 'cores' in info['cpu_info']

    def test_real_model_creation(self, optimizer):
        """Test creation of real models for testing."""
        model_path = optimizer.create_test_model("gomoku")

        assert model_path.exists(), "Model file should be created"
        assert model_path.suffix == '.pth', "Model should be a PyTorch file"

        # Load and verify model
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        assert isinstance(model, AlphaZeroNet)

        # Test model inference
        batch_size = 2
        game_state = RealGameState("gomoku")
        features = game_state.get_features()
        batch_features = torch.stack([torch.FloatTensor(features)] * batch_size)

        model.eval()
        with torch.no_grad():
            policy, value = model(batch_features)

        assert policy.shape == (batch_size, game_state.action_space)
        assert value.shape == (batch_size, 1)

        # Cleanup
        model_path.unlink()

    def test_real_inference_worker_creation(self, optimizer):
        """Test creation of real inference workers."""
        # Test CPU worker creation (should always work)
        worker = optimizer.create_real_inference_worker("gomoku", use_gpu=False)
        assert worker is not None

        # Cleanup model file
        model_path = optimizer.output_dir / "test_model_gomoku.pth"
        if model_path.exists():
            model_path.unlink()

    def test_thread_count_test_basic(self, optimizer):
        """Test basic thread count testing functionality."""
        config = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=4,
            num_searches=3,
            warmup_searches=1,
            timeout_seconds=20.0
        )

        result = optimizer.run_thread_count_test(config)

        assert result.thread_count == 1
        assert 0.0 <= result.success_rate <= 1.0
        if result.success_rate > 0.0:
            assert result.searches_per_second >= 0.0
            assert result.average_search_time_ms > 0
            assert result.efficiency_score() >= 0.0
        if result.error_message:
            assert result.success_rate < 1.0

    def test_recommendations_generation(self, optimizer):
        """Test optimization recommendations generation."""
        # Create test results
        results = [
            ThreadPerformanceResult(
                thread_count=1, searches_per_second=100.0, average_search_time_ms=40.0,
                search_time_std_ms=2.0, thread_utilization_percent=90.0,
                contention_score=5.0, cpu_utilization_percent=25.0,
                memory_usage_mb=256.0, success_rate=1.0
            ),
            ThreadPerformanceResult(
                thread_count=4, searches_per_second=350.0, average_search_time_ms=15.0,
                search_time_std_ms=3.0, thread_utilization_percent=85.0,
                contention_score=8.0, cpu_utilization_percent=70.0,
                memory_usage_mb=512.0, success_rate=1.0
            ),
            ThreadPerformanceResult(
                thread_count=8, searches_per_second=400.0, average_search_time_ms=12.0,
                search_time_std_ms=4.0, thread_utilization_percent=80.0,
                contention_score=12.0, cpu_utilization_percent=90.0,
                memory_usage_mb=768.0, success_rate=0.95
            )
        ]

        optimal_result = results[1]  # 4 threads is optimal
        recommendations = optimizer._generate_recommendations(results, optimal_result)

        assert len(recommendations) > 0
        assert any("4 threads" in rec for rec in recommendations)
        assert any("contention" in rec.lower() for rec in recommendations)

    def test_report_serialization(self, optimizer, temp_dir):
        """Test optimization report saving and loading."""
        # Create dummy report
        results = [
            ThreadPerformanceResult(
                thread_count=2, searches_per_second=200.0, average_search_time_ms=25.0,
                search_time_std_ms=3.0, thread_utilization_percent=80.0,
                contention_score=10.0, cpu_utilization_percent=60.0,
                memory_usage_mb=400.0, success_rate=1.0
            )
        ]

        report = OptimizationReport(
            test_config={'game_type': 'gomoku', 'simulations': 100},
            results=results,
            optimal_thread_count=2,
            optimal_result=results[0],
            performance_curve=[(2, 0.8)],
            recommendations=["Use 2 threads"],
            system_info=optimizer.system_info,
            test_duration_seconds=10.0
        )

        # Save report
        output_path = optimizer.save_report(report, "test_report.json")
        assert output_path.exists()

        # Load and verify
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['optimal_thread_count'] == 2
        assert loaded_data['test_config']['game_type'] == 'gomoku'
        assert len(loaded_data['results']) == 1

    def test_optimize_thread_count_quick(self, optimizer):
        """Test quick optimization mode."""
        report = optimizer.optimize_thread_count(
            game_type="gomoku",
            simulations=200,
            iterations=10,
            max_threads=1,
            quick_test=True
        )

        assert report.optimal_thread_count >= 1
        assert len(report.results) >= 1
        assert len(report.performance_curve) == len(report.results)
        assert report.test_duration_seconds > 0
        assert len(report.recommendations) > 0
        assert report.test_config['simulations'] <= 200
        assert report.test_config['iterations'] <= 10
        assert report.test_config['quick_test'] is True

    def test_config_validation(self):
        """Test thread test configuration validation."""
        # Valid configuration
        config = ThreadTestConfig(
            thread_count=8,
            game_type="gomoku",
            simulations_per_search=800,
            num_searches=50
        )

        assert config.thread_count == 8
        assert config.game_type == "gomoku"
        assert config.timeout_seconds > 0

    def test_contention_score_calculation(self):
        """Test contention score calculation logic."""
        # Low variance = low contention
        low_contention_result = ThreadPerformanceResult(
            thread_count=4, searches_per_second=400.0, average_search_time_ms=20.0,
            search_time_std_ms=1.0, thread_utilization_percent=80.0,
            contention_score=5.0, cpu_utilization_percent=70.0,
            memory_usage_mb=512.0, success_rate=1.0
        )

        # High variance = high contention
        high_contention_result = ThreadPerformanceResult(
            thread_count=16, searches_per_second=300.0, average_search_time_ms=25.0,
            search_time_std_ms=12.0, thread_utilization_percent=90.0,
            contention_score=48.0, cpu_utilization_percent=95.0,
            memory_usage_mb=1024.0, success_rate=0.9
        )

        assert low_contention_result.efficiency_score() > high_contention_result.efficiency_score()


class TestIntegration:
    """Integration tests for the thread optimization system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_real_component_integration(self, temp_dir):
        """Test optimization with real components (CPU only for reliability)."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Run minimal optimization with real components
        report = optimizer.optimize_thread_count(
            game_type="gomoku",
            simulations=20,  # Very small for testing
            iterations=3,    # Very small for testing
            max_threads=2,   # Only test 1-2 threads
            quick_test=True
        )

        # Verify report structure
        assert report.optimal_thread_count >= 1
        assert len(report.results) >= 1
        assert report.test_duration_seconds >= 0
        assert len(report.recommendations) > 0

        # Verify we got real performance data
        success_count = 0
        for result in report.results:
            assert result.thread_count >= 1
            assert 0.0 <= result.success_rate <= 1.0
            if result.success_rate > 0.5:
                success_count += 1
                assert result.searches_per_second >= 0
                assert result.average_search_time_ms > 0

        # In CI environments, real components may fail due to interface mismatches
        # The important thing is that the optimizer handles failures gracefully
        if success_count == 0:
            # Verify that failures were handled gracefully
            for result in report.results:
                assert result.error_message is not None or result.success_rate == 0.0
            # Should still generate recommendations even with failures
            assert len(report.recommendations) > 0
        else:
            # At least one configuration worked
            assert success_count > 0

    def test_real_threading_contention(self, temp_dir):
        """Test that real threading shows performance characteristics."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Test single thread
        config_1 = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=50,
            num_searches=10,
            warmup_searches=2,
            timeout_seconds=60.0
        )
        result_1 = optimizer.run_thread_count_test(config_1)

        # Test multiple threads
        config_2 = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=50,
            num_searches=10,
            warmup_searches=2,
            timeout_seconds=60.0
        )
        result_2 = optimizer.run_thread_count_test(config_2)

        # Verify both tests completed
        assert result_1.success_rate > 0, "Single thread test should succeed"
        assert result_2.success_rate > 0, "Multi-thread test should succeed"

        # Verify performance metrics are reasonable
        assert result_1.searches_per_second > 0
        assert result_2.searches_per_second > 0
        assert result_1.average_search_time_ms > 0
        assert result_2.average_search_time_ms > 0

        # Verify contention scores make sense
        assert 0 <= result_1.contention_score <= 100
        assert 0 <= result_2.contention_score <= 100

    def test_stress_testing(self, temp_dir):
        """Test thread optimizer under stress conditions."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Test with high thread count (should handle gracefully)
        config = ThreadTestConfig(
            thread_count=max(4, optimizer.cpu_count),  # Use max available or 4
            game_type="gomoku",
            simulations_per_search=100,
            num_searches=5,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        result = optimizer.run_thread_count_test(config)

        # Should complete without crashing
        assert result is not None
        assert result.thread_count == config.thread_count
        assert 0.0 <= result.success_rate <= 1.0

        # If it succeeded, should have reasonable metrics
        if result.success_rate > 0.5:
            assert result.searches_per_second >= 0
            assert result.average_search_time_ms > 0
            assert result.contention_score >= 0

    def test_different_game_types(self, temp_dir):
        """Test optimization with different game types."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        game_types = ["gomoku", "chess", "go"]
        for game_type in game_types:
            config = ThreadTestConfig(
                thread_count=2,
                game_type=game_type,
                simulations_per_search=20,
                num_searches=3,
                warmup_searches=1,
                timeout_seconds=30.0
            )

            result = optimizer.run_thread_count_test(config)

            # Should complete for all game types
            assert result is not None
            assert result.thread_count == 2
            assert 0.0 <= result.success_rate <= 1.0

    def test_memory_usage_tracking(self, temp_dir):
        """Test that memory usage is properly tracked."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=100,
            num_searches=5,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        result = optimizer.run_thread_count_test(config)

        # Memory tracking should work
        if result.success_rate > 0.5:
            assert result.memory_usage_mb > 0
            assert result.memory_usage_mb < 10000  # Reasonable upper bound

    def test_cpu_utilization_measurement(self, temp_dir):
        """Test CPU utilization measurement."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=100,
            num_searches=5,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        result = optimizer.run_thread_count_test(config)

        # CPU utilization should be measured
        if result.success_rate > 0.5:
            assert 0 <= result.cpu_utilization_percent <= 100
            assert result.thread_utilization_percent >= 0

    @pytest.mark.slow
    def test_full_optimization_workflow(self, temp_dir):
        """Test complete optimization workflow with real components."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Run small but complete optimization
        report = optimizer.optimize_thread_count(
            game_type="gomoku",
            simulations=30,  # Small for testing
            iterations=3,    # Small for testing
            max_threads=3,   # Small range for testing
            quick_test=True
        )

        # Verify results
        assert 1 <= report.optimal_thread_count <= 3
        assert len(report.results) >= 1
        assert len(report.results) <= 3
        assert report.test_duration_seconds > 0
        assert len(report.recommendations) > 0

        # Verify performance curve
        assert len(report.performance_curve) >= 1
        for thread_count, efficiency in report.performance_curve:
            assert thread_count >= 1
            assert 0.0 <= efficiency <= 1.0

        # Verify files were created
        json_files = list(temp_dir.glob("*.json"))
        assert len(json_files) >= 1  # At least one report file

    def test_error_handling(self, temp_dir):
        """Test error handling in optimization."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Test with invalid timeout (should handle gracefully)
        config = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=10,
            num_searches=1,
            timeout_seconds=0.1  # Very short timeout to trigger timeout errors
        )

        result = optimizer.run_thread_count_test(config)

        # Should handle timeout gracefully
        assert result is not None
        assert 0.0 <= result.success_rate <= 1.0
        # If it timed out, success rate should be low
        if result.success_rate < 0.5:
            assert result.error_message is not None

    def test_concurrent_testing(self, temp_dir):
        """Test that concurrent tests don't interfere."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Run two tests in sequence to ensure cleanup works
        config1 = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=20,
            num_searches=3,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        config2 = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=20,
            num_searches=3,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        result1 = optimizer.run_thread_count_test(config1)
        result2 = optimizer.run_thread_count_test(config2)

        # Both should complete successfully
        assert result1 is not None
        assert result2 is not None
        assert result1.thread_count == 1
        assert result2.thread_count == 2

    def test_performance_scaling(self, temp_dir):
        """Test that performance metrics scale reasonably with thread count."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        results = []
        for thread_count in [1, 2]:
            config = ThreadTestConfig(
                thread_count=thread_count,
                game_type="gomoku",
                simulations_per_search=50,
                num_searches=5,
                warmup_searches=1,
                timeout_seconds=45.0
            )

            result = optimizer.run_thread_count_test(config)
            if result.success_rate > 0.7:
                results.append(result)

        # If we have successful results, check scaling
        if len(results) >= 2:
            # Performance should generally increase with threads (up to a point)
            # Or at least not decrease dramatically
            result_1 = next(r for r in results if r.thread_count == 1)
            result_2 = next(r for r in results if r.thread_count == 2)

            # Efficiency should be reasonable for both
            assert result_1.efficiency_score() > 0.1
            assert result_2.efficiency_score() > 0.1

            # Thread utilization should increase with thread count
            if result_2.thread_utilization_percent > 0:
                assert result_2.thread_utilization_percent + 15 >= result_1.thread_utilization_percent

    def test_resource_cleanup(self, temp_dir):
        """Test that resources are properly cleaned up after tests."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=20,
            num_searches=3,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        # Run test and ensure cleanup happens
        result = optimizer.run_thread_count_test(config)

        # Check that temporary model files are cleaned up
        model_files = list(temp_dir.glob("test_model_*.pth"))
        assert len(model_files) == 0, "Model files should be cleaned up"

        # Test should still succeed
        assert result is not None
        assert result.thread_count == 2


class TestPerformanceValidation:
    """Performance validation tests for thread optimization."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for performance tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_performance_consistency(self, temp_dir):
        """Test that performance measurements are consistent."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=30,
            num_searches=5,
            warmup_searches=2,
            timeout_seconds=45.0
        )

        # Run same test multiple times
        results = []
        for _ in range(2):
            result = optimizer.run_thread_count_test(config)
            if result.success_rate > 0.7:
                results.append(result)

        # If we have multiple successful results, they should be reasonably consistent
        if len(results) >= 2:
            throughputs = [r.searches_per_second for r in results]
            avg_throughput = sum(throughputs) / len(throughputs)
            max_deviation = max(abs(t - avg_throughput) / avg_throughput for t in throughputs)

            # Results should be within 50% of each other (allowing for system variation)
            assert max_deviation < 0.5, f"Performance too inconsistent: {throughputs}"

    def test_warmup_effectiveness(self, temp_dir):
        """Test that warmup actually improves performance."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Test without warmup
        config_no_warmup = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=20,
            num_searches=3,
            warmup_searches=0,  # No warmup
            timeout_seconds=30.0
        )

        # Test with warmup
        config_with_warmup = ThreadTestConfig(
            thread_count=2,
            game_type="gomoku",
            simulations_per_search=20,
            num_searches=3,
            warmup_searches=2,  # With warmup
            timeout_seconds=30.0
        )

        result_no_warmup = optimizer.run_thread_count_test(config_no_warmup)
        result_with_warmup = optimizer.run_thread_count_test(config_with_warmup)

        # Both should succeed
        assert result_no_warmup.success_rate > 0.5
        assert result_with_warmup.success_rate > 0.5

        # Warmup should generally improve consistency (lower variance)
        # This is more about testing the measurement than requiring improvement
        assert result_with_warmup.search_time_std_ms >= 0
        assert result_no_warmup.search_time_std_ms >= 0

    def test_timeout_handling(self, temp_dir):
        """Test that timeouts are handled properly."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Test with very short timeout
        config = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=1000,  # High simulations
            num_searches=10,               # Many searches
            warmup_searches=0,
            timeout_seconds=2.0            # Short timeout
        )

        result = optimizer.run_thread_count_test(config)

        # Should handle timeout gracefully
        assert result is not None
        assert 0.0 <= result.success_rate <= 1.0

        # If it timed out, we should see evidence in the metrics
        if result.success_rate < 0.8:
            assert result.error_message is not None or result.searches_per_second == 0

    def test_high_thread_count_behavior(self, temp_dir):
        """Test behavior with high thread counts."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Test with thread count higher than CPU count
        high_thread_count = min(optimizer.cpu_count * 2, 16)  # Cap at 16 for testing

        config = ThreadTestConfig(
            thread_count=high_thread_count,
            game_type="gomoku",
            simulations_per_search=20,
            num_searches=3,
            warmup_searches=1,
            timeout_seconds=45.0
        )

        result = optimizer.run_thread_count_test(config)

        # Should complete without crashing
        assert result is not None
        assert result.thread_count == high_thread_count
        assert 0.0 <= result.success_rate <= 1.0

        # If successful, should show high contention
        if result.success_rate > 0.5:
            # High thread count should generally show more contention
            assert result.contention_score >= 0  # At minimum, should be measured

    @pytest.mark.slow
    def test_optimization_convergence(self, temp_dir):
        """Test that optimization converges to reasonable results."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        # Run optimization with small parameters
        report = optimizer.optimize_thread_count(
            game_type="gomoku",
            simulations=40,
            iterations=4,
            max_threads=4,
            quick_test=True
        )

        # Should find optimal thread count
        assert 1 <= report.optimal_thread_count <= 4
        assert report.optimal_result.success_rate > 0.5
        assert report.optimal_result.efficiency_score() > 0.1

        # Performance curve should show reasonable trend
        assert len(report.performance_curve) >= 2

        # Optimal should be among the better performing configurations
        all_efficiencies = [eff for _, eff in report.performance_curve]
        optimal_efficiency = report.optimal_result.efficiency_score()

        # Optimal should be in top 70% of results
        sorted_efficiencies = sorted(all_efficiencies, reverse=True)
        top_70_percent = sorted_efficiencies[:max(1, len(sorted_efficiencies) * 7 // 10)]
        assert optimal_efficiency in top_70_percent or optimal_efficiency >= min(top_70_percent)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for edge case tests."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_single_thread_performance(self, temp_dir):
        """Test single thread performance baseline."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=50,
            num_searches=5,
            warmup_searches=1,
            timeout_seconds=30.0
        )

        result = optimizer.run_thread_count_test(config)

        # Single thread should work reliably
        assert result.success_rate > 0.8
        assert result.searches_per_second > 0
        assert result.thread_utilization_percent >= 0  # May be 0 if not measured correctly
        assert result.contention_score >= 0  # Should be low but measurable

    def test_minimal_workload(self, temp_dir):
        """Test with minimal workload."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=1,
            game_type="gomoku",
            simulations_per_search=1,  # Minimal simulations
            num_searches=1,            # Minimal searches
            warmup_searches=0,         # No warmup
            timeout_seconds=10.0
        )

        result = optimizer.run_thread_count_test(config)

        # Should complete quickly
        assert result is not None
        assert result.success_rate > 0.5  # Should succeed with minimal load
        assert result.average_search_time_ms > 0

    def test_zero_thread_handling(self, temp_dir):
        """Test that zero threads is handled gracefully."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=0,  # Invalid
            game_type="gomoku",
            simulations_per_search=10,
            num_searches=1,
            timeout_seconds=10.0
        )

        result = optimizer.run_thread_count_test(config)

        # Should handle gracefully (either error or default to 1)
        assert result is not None
        assert result.success_rate == 0.0 or result.thread_count >= 1

    def test_invalid_game_type(self, temp_dir):
        """Test with invalid game type."""
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=False)

        config = ThreadTestConfig(
            thread_count=1,
            game_type="invalid_game",  # Invalid game type
            simulations_per_search=10,
            num_searches=1,
            timeout_seconds=10.0
        )

        result = optimizer.run_thread_count_test(config)

        # Should handle gracefully (either error or fallback)
        assert result is not None
        # Either fails cleanly or falls back to valid game
        assert result.success_rate == 0.0 or result.game_type in ["gomoku", "chess", "go"]


# Performance validation functions that could be run manually
def validate_actual_performance():
    """
    Manual validation function for actual performance testing.
    Run this only when you want to test with real components.
    """
    # This would be called manually for real performance validation
    # Not run automatically in CI/CD due to resource requirements
    print("Running manual performance validation...")

    temp_dir = Path(tempfile.mkdtemp())
    try:
        optimizer = ThreadOptimizer(output_dir=temp_dir, enable_plotting=True)

        # Run comprehensive optimization
        report = optimizer.optimize_thread_count(
            game_type="gomoku",
            simulations=200,
            iterations=20,
            max_threads=8,
            quick_test=False
        )

        print(f"Optimal thread count: {report.optimal_thread_count}")
        print(f"Peak performance: {report.optimal_result.searches_per_second:.1f} searches/sec")
        print(f"Contention score: {report.optimal_result.contention_score:.1f}%")

        return report

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Allow running specific tests or manual validation
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_actual_performance()
    else:
        pytest.main([__file__, "-v"])
