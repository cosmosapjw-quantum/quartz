#!/usr/bin/env python3
"""
Unit tests for Batch Size Optimizer

Tests the batch size optimization script functionality including:
- GPU memory profiling and VRAM monitoring
- Batch size parameter sweep logic
- Throughput and latency measurement
- Optimization algorithm and efficiency scoring
- OOM handling and GPU constraint validation

Run with:
    python -m pytest tests/unit/test_batch_size_optimizer.py -v
"""

import pytest
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.tune_batch_size import (
    BatchSizeOptimizer,
    BatchSizeTestConfig,
    BatchSizePerformanceResult,
    BatchSizeOptimizationReport,
    GPUMonitor,
    RealGameState
)


class TestBatchSizeTestConfig:
    """Test BatchSizeTestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BatchSizeTestConfig(batch_size=32)

        assert config.batch_size == 32
        assert config.game_type == "gomoku"
        assert config.iterations == 100
        assert config.warmup_iterations == 20
        assert config.timeout_seconds == 60.0
        assert config.max_vram_percent == 85.0
        assert config.measure_latency == True
        assert config.monitor_gpu == True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BatchSizeTestConfig(
            batch_size=64,
            game_type="chess",
            iterations=200,
            warmup_iterations=40,
            timeout_seconds=120.0,
            max_vram_percent=80.0,
            measure_latency=False,
            monitor_gpu=False
        )

        assert config.batch_size == 64
        assert config.game_type == "chess"
        assert config.iterations == 200
        assert config.warmup_iterations == 40
        assert config.timeout_seconds == 120.0
        assert config.max_vram_percent == 80.0
        assert config.measure_latency == False
        assert config.monitor_gpu == False


class TestBatchSizePerformanceResult:
    """Test BatchSizePerformanceResult dataclass and methods."""

    def test_default_result(self):
        """Test default performance result values."""
        result = BatchSizePerformanceResult(
            batch_size=32,
            throughput_inferences_per_sec=1000.0,
            average_latency_ms=32.0,
            latency_std_ms=5.0,
            gpu_utilization_percent=85.0,
            vram_usage_mb=4096.0,
            vram_usage_percent=50.0,
            memory_efficiency=0.244  # 1000/4096
        )

        assert result.batch_size == 32
        assert result.success_rate == 1.0  # Default
        assert result.oom_occurred == False  # Default
        assert result.error_message is None  # Default

    def test_efficiency_score_optimal(self):
        """Test efficiency score calculation for optimal case."""
        result = BatchSizePerformanceResult(
            batch_size=64,
            throughput_inferences_per_sec=8000.0,  # High throughput
            average_latency_ms=8.0,  # Low latency
            latency_std_ms=1.0,
            gpu_utilization_percent=90.0,  # High GPU utilization
            vram_usage_mb=6000.0,
            vram_usage_percent=75.0,  # Near target (75%)
            memory_efficiency=1.33,  # 8000/6000
            success_rate=1.0
        )

        score = result.efficiency_score()
        assert score > 0.7  # Should be high for optimal case
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_efficiency_score_oom(self):
        """Test efficiency score with OOM condition."""
        result = BatchSizePerformanceResult(
            batch_size=512,
            throughput_inferences_per_sec=0.0,
            average_latency_ms=float('inf'),
            latency_std_ms=0.0,
            gpu_utilization_percent=0.0,
            vram_usage_mb=0.0,
            vram_usage_percent=100.0,
            memory_efficiency=0.0,
            oom_occurred=True,  # OOM condition
            success_rate=0.0
        )

        score = result.efficiency_score()
        assert score == 0.0  # Should be zero for OOM

    def test_efficiency_score_low_success(self):
        """Test efficiency score with low success rate."""
        result = BatchSizePerformanceResult(
            batch_size=32,
            throughput_inferences_per_sec=5000.0,
            average_latency_ms=10.0,
            latency_std_ms=2.0,
            gpu_utilization_percent=80.0,
            vram_usage_mb=4000.0,
            vram_usage_percent=50.0,
            memory_efficiency=1.25,
            success_rate=0.5  # Low success rate
        )

        score = result.efficiency_score()
        assert score == 0.0  # Should be zero for low success rate

    def test_efficiency_score_components(self):
        """Test efficiency score responds correctly to different components."""
        base_result = BatchSizePerformanceResult(
            batch_size=32,
            throughput_inferences_per_sec=4000.0,
            average_latency_ms=15.0,
            latency_std_ms=3.0,
            gpu_utilization_percent=75.0,
            vram_usage_mb=4000.0,
            vram_usage_percent=60.0,
            memory_efficiency=1.0
        )

        base_score = base_result.efficiency_score()

        # Higher throughput should improve score
        high_throughput = BatchSizePerformanceResult(
            batch_size=64,
            throughput_inferences_per_sec=8000.0,  # Higher
            average_latency_ms=15.0,
            latency_std_ms=3.0,
            gpu_utilization_percent=75.0,
            vram_usage_mb=4000.0,
            vram_usage_percent=60.0,
            memory_efficiency=2.0  # Also higher due to higher throughput
        )

        assert high_throughput.efficiency_score() > base_score

        # Better VRAM usage (closer to 75% target) should improve score
        optimal_vram = BatchSizePerformanceResult(
            batch_size=32,
            throughput_inferences_per_sec=4000.0,
            average_latency_ms=15.0,
            latency_std_ms=3.0,
            gpu_utilization_percent=75.0,
            vram_usage_mb=6000.0,
            vram_usage_percent=75.0,  # Optimal VRAM usage
            memory_efficiency=0.67  # 4000/6000
        )

        assert optimal_vram.efficiency_score() > base_score


class TestGPUMonitor:
    """Test GPUMonitor class functionality."""

    def test_gpu_monitor_initialization(self):
        """Test GPU monitor initialization."""
        monitor = GPUMonitor()

        assert hasattr(monitor, 'nvml_available')
        assert hasattr(monitor, 'device_count')
        assert monitor.device_count >= 0

    @patch('scripts.tune_batch_size.torch.cuda.is_available', return_value=True)
    @patch('scripts.tune_batch_size.torch.cuda.device_count', return_value=1)
    @patch('scripts.tune_batch_size.torch.cuda.current_device', return_value=0)
    @patch('scripts.tune_batch_size.torch.cuda.get_device_properties')
    def test_get_gpu_info_cuda_available(self, mock_props, mock_current, mock_count, mock_available):
        """Test GPU info collection when CUDA is available."""
        # Mock device properties
        mock_device_props = Mock()
        mock_device_props.name = "Test GPU"
        mock_device_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_device_props.major = 7
        mock_device_props.minor = 5
        mock_device_props.multi_processor_count = 68
        mock_props.return_value = mock_device_props

        with patch('scripts.tune_batch_size.torch.cuda.memory_allocated', return_value=1024*1024*1024):  # 1GB
            with patch('scripts.tune_batch_size.torch.cuda.memory_reserved', return_value=2*1024*1024*1024):  # 2GB
                monitor = GPUMonitor()
                gpu_info = monitor.get_gpu_info(0)

                assert gpu_info['cuda_available'] == True
                assert gpu_info['device_count'] == 1
                assert gpu_info['current_device'] == 0
                assert gpu_info['name'] == "Test GPU"
                assert gpu_info['total_memory_mb'] == 8192
                assert gpu_info['compute_capability'] == "7.5"
                assert gpu_info['multiprocessor_count'] == 68
                assert gpu_info['memory_allocated_mb'] == 1024
                assert gpu_info['memory_reserved_mb'] == 2048
                assert gpu_info['memory_free_mb'] == 6144  # 8192 - 2048

    @patch('scripts.tune_batch_size.torch.cuda.is_available', return_value=False)
    def test_get_gpu_info_cuda_unavailable(self, mock_available):
        """Test GPU info collection when CUDA is unavailable."""
        monitor = GPUMonitor()
        gpu_info = monitor.get_gpu_info(0)

        assert gpu_info['cuda_available'] == False
        assert gpu_info['device_count'] == 0
        assert gpu_info['current_device'] is None

    @patch('scripts.tune_batch_size.torch.cuda.is_available', return_value=True)
    @patch('scripts.tune_batch_size.torch.cuda.device_count', return_value=1)
    @patch('scripts.tune_batch_size.torch.cuda.get_device_properties')
    def test_get_memory_info(self, mock_props, mock_count, mock_available):
        """Test memory info collection."""
        # Mock device properties
        mock_device_props = Mock()
        mock_device_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_props.return_value = mock_device_props

        with patch('scripts.tune_batch_size.torch.cuda.memory_allocated', return_value=1024*1024*1024):  # 1GB
            with patch('scripts.tune_batch_size.torch.cuda.memory_reserved', return_value=2*1024*1024*1024):  # 2GB
                monitor = GPUMonitor()
                memory_info = monitor.get_memory_info(0)

                assert memory_info['total_mb'] == 8192
                assert memory_info['allocated_mb'] == 1024
                assert memory_info['reserved_mb'] == 2048
                assert memory_info['free_mb'] == 6144
                assert abs(memory_info['usage_percent'] - 25.0) < 0.1  # 2048/8192 * 100


class TestRealGameState:
    """Test RealGameState fallback implementation."""

    def test_gomoku_game_state(self):
        """Test Gomoku game state initialization."""
        game_state = RealGameState("gomoku")

        assert game_state.game_type == "gomoku"
        assert game_state.board_size == 15
        assert game_state.feature_planes == 36
        assert game_state.action_space == 225

    def test_chess_game_state(self):
        """Test Chess game state initialization."""
        game_state = RealGameState("chess")

        assert game_state.game_type == "chess"
        assert game_state.board_size == 8
        assert game_state.feature_planes == 30
        assert game_state.action_space == 4096

    def test_go_game_state(self):
        """Test Go game state initialization."""
        game_state = RealGameState("go")

        assert game_state.game_type == "go"
        assert game_state.board_size == 19
        assert game_state.feature_planes == 25
        assert game_state.action_space == 362

    def test_get_features_fallback(self):
        """Test feature extraction in fallback mode."""
        game_state = RealGameState("gomoku")

        features = game_state.get_features()

        assert features.shape == (36, 15, 15)  # Gomoku features
        assert features.dtype == np.float32

        # Initially, only current player plane should be set
        assert np.all(features[0] == 0)  # No player 1 pieces
        assert np.all(features[1] == 0)  # No player 2 pieces
        assert np.all(features[2] == 1)  # Current player is 1


class TestBatchSizeOptimizer:
    """Test BatchSizeOptimizer class functionality."""

    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.optimizer = BatchSizeOptimizer(
            output_dir=self.temp_dir,
            enable_plotting=False  # Disable plotting for tests
        )

    def teardown_method(self):
        """Cleanup after each test method."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.output_dir == self.temp_dir
        assert self.optimizer.output_dir.exists()
        assert not self.optimizer.enable_plotting
        assert isinstance(self.optimizer.system_info, dict)
        assert isinstance(self.optimizer.gpu_info, dict)
        assert isinstance(self.optimizer.gpu_monitor, GPUMonitor)

    def test_system_info_collection(self):
        """Test system information collection."""
        system_info = self.optimizer._get_system_info()

        assert 'cpu_info' in system_info
        assert 'memory_total_gb' in system_info
        assert 'platform' in system_info
        assert 'python_version' in system_info
        assert isinstance(system_info['memory_total_gb'], float)

    @patch('scripts.tune_batch_size.AlphaZeroNet')
    @patch('scripts.tune_batch_size.torch.save')
    def test_create_test_model(self, mock_save, mock_model_class):
        """Test test model creation."""
        # Mock model
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_model_class.return_value = mock_model

        # Mock torch.save to actually create the file
        def mock_save_side_effect(model, path):
            Path(path).touch()
        mock_save.side_effect = mock_save_side_effect

        model_path, model = self.optimizer.create_test_model("gomoku")

        assert model_path.exists()
        assert model_path.suffix == '.pth'
        assert 'gomoku' in model_path.name
        assert model == mock_model
        mock_model_class.assert_called_once()
        mock_save.assert_called_once()

    @patch('scripts.tune_batch_size.torch.cuda.is_available', return_value=False)
    def test_run_batch_size_test_no_cuda(self, mock_cuda):
        """Test batch size test when CUDA is not available."""
        config = BatchSizeTestConfig(batch_size=32)

        result = self.optimizer.run_batch_size_test(config)

        assert result.batch_size == 32
        assert result.success_rate == 0.0
        assert result.throughput_inferences_per_sec == 0.0
        assert result.error_message == "CUDA not available for batch size testing"

    @patch('scripts.tune_batch_size.torch.cuda.is_available', return_value=True)
    @patch('scripts.tune_batch_size.torch.cuda.empty_cache')
    def test_run_batch_size_test_oom(self, mock_cache, mock_cuda):
        """Test batch size test with OOM condition."""
        config = BatchSizeTestConfig(batch_size=1024, warmup_iterations=1, iterations=1)

        # Mock the model creation to succeed
        with patch.object(self.optimizer, 'create_test_model') as mock_create:
            mock_model = Mock()
            mock_model_path = self.temp_dir / "test_model.pth"
            mock_model_path.touch()
            mock_create.return_value = (mock_model_path, mock_model)

            # Mock torch.device and model.to
            with patch('scripts.tune_batch_size.torch.device'):
                mock_model.to.return_value = mock_model

                # Mock GPU memory monitoring
                with patch.object(self.optimizer.gpu_monitor, 'get_memory_info') as mock_memory:
                    mock_memory.return_value = {
                        'total_mb': 8192,
                        'allocated_mb': 1024,
                        'reserved_mb': 2048,
                        'free_mb': 6144,
                        'usage_percent': 25.0
                    }

                    # Mock warmup to raise OOM
                    with patch.object(self.optimizer, '_run_warmup_batch'):
                        # Mock torch.randn to raise OOM
                        with patch('scripts.tune_batch_size.torch.randn', side_effect=torch.cuda.OutOfMemoryError()):
                            result = self.optimizer.run_batch_size_test(config)

                            assert result.batch_size == 1024
                            assert result.oom_occurred == True
                            assert result.success_rate == 0.0
                            assert result.error_message == "Out of Memory"

    def test_generate_recommendations_optimal(self):
        """Test recommendation generation for optimal results."""
        results = [
            BatchSizePerformanceResult(
                batch_size=64,
                throughput_inferences_per_sec=6000.0,
                average_latency_ms=12.0,
                latency_std_ms=2.0,
                gpu_utilization_percent=85.0,  # Good GPU utilization
                vram_usage_mb=6000.0,
                vram_usage_percent=75.0,  # Optimal VRAM usage
                memory_efficiency=1.0,  # 6000/6000
                success_rate=0.95
            )
        ]

        optimal_result = results[0]
        max_vram_percent = 85.0
        recommendations = self.optimizer._generate_recommendations(results, optimal_result, max_vram_percent)

        assert len(recommendations) > 0
        assert any("64 for optimal performance" in rec for rec in recommendations)
        assert any("Good VRAM efficiency" in rec for rec in recommendations)
        assert any("Excellent GPU utilization" in rec for rec in recommendations)

    def test_generate_recommendations_suboptimal(self):
        """Test recommendation generation for suboptimal results."""
        results = [
            BatchSizePerformanceResult(
                batch_size=16,
                throughput_inferences_per_sec=1000.0,
                average_latency_ms=80.0,  # High latency
                latency_std_ms=20.0,
                gpu_utilization_percent=45.0,  # Low GPU utilization
                vram_usage_mb=1000.0,
                vram_usage_percent=12.5,  # Low VRAM usage
                memory_efficiency=1.0,
                success_rate=0.85
            )
        ]

        optimal_result = results[0]
        max_vram_percent = 85.0
        recommendations = self.optimizer._generate_recommendations(results, optimal_result, max_vram_percent)

        assert len(recommendations) > 0
        assert any("Low VRAM usage" in rec for rec in recommendations)
        assert any("Low GPU utilization" in rec for rec in recommendations)
        assert any("High latency" in rec for rec in recommendations)

    def test_generate_recommendations_with_oom(self):
        """Test recommendation generation when OOM occurs."""
        results = [
            BatchSizePerformanceResult(
                batch_size=32,
                throughput_inferences_per_sec=4000.0,
                average_latency_ms=10.0,
                latency_std_ms=2.0,
                gpu_utilization_percent=80.0,
                vram_usage_mb=4000.0,
                vram_usage_percent=50.0,
                memory_efficiency=1.0,
                success_rate=1.0
            ),
            BatchSizePerformanceResult(
                batch_size=128,
                throughput_inferences_per_sec=0.0,
                average_latency_ms=float('inf'),
                latency_std_ms=0.0,
                gpu_utilization_percent=0.0,
                vram_usage_mb=0.0,
                vram_usage_percent=100.0,
                memory_efficiency=0.0,
                oom_occurred=True,
                success_rate=0.0
            )
        ]

        optimal_result = results[0]
        max_vram_percent = 85.0
        recommendations = self.optimizer._generate_recommendations(results, optimal_result, max_vram_percent)

        assert len(recommendations) > 0
        assert any("Out of memory occurs at batch size 128" in rec for rec in recommendations)

    def test_save_report(self):
        """Test report saving functionality."""
        # Create a minimal report
        report = BatchSizeOptimizationReport(
            test_config={'game_type': 'gomoku'},
            results=[],
            optimal_batch_size=32,
            optimal_result=BatchSizePerformanceResult(
                batch_size=32,
                throughput_inferences_per_sec=4000.0,
                average_latency_ms=10.0,
                latency_std_ms=2.0,
                gpu_utilization_percent=80.0,
                vram_usage_mb=4000.0,
                vram_usage_percent=50.0,
                memory_efficiency=1.0
            ),
            performance_curve=[],
            recommendations=[],
            system_info={},
            gpu_info={},
            test_duration_seconds=60.0
        )

        # Save report
        saved_path = self.optimizer.save_report(report, "test_report.json")

        assert saved_path.exists()
        assert saved_path.suffix == '.json'

        # Load and verify content
        with open(saved_path, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data['optimal_batch_size'] == 32
        assert loaded_data['test_config']['game_type'] == 'gomoku'

    def test_print_summary(self, capsys):
        """Test summary printing functionality."""
        # Create a minimal report
        report = BatchSizeOptimizationReport(
            test_config={
                'game_type': 'gomoku',
                'iterations': 100,
                'min_batch_size': 8,
                'max_batch_size': 256,
                'max_vram_percent': 85.0
            },
            results=[
                BatchSizePerformanceResult(
                    batch_size=32,
                    throughput_inferences_per_sec=4000.0,
                    average_latency_ms=10.0,
                    latency_std_ms=2.0,
                    gpu_utilization_percent=80.0,
                    vram_usage_mb=4000.0,
                    vram_usage_percent=50.0,
                    memory_efficiency=1.0
                )
            ],
            optimal_batch_size=32,
            optimal_result=BatchSizePerformanceResult(
                batch_size=32,
                throughput_inferences_per_sec=4000.0,
                average_latency_ms=10.0,
                latency_std_ms=2.0,
                gpu_utilization_percent=80.0,
                vram_usage_mb=4000.0,
                vram_usage_percent=50.0,
                memory_efficiency=1.0
            ),
            performance_curve=[(32, 0.8)],
            recommendations=["Use batch size 32 for optimal performance"],
            system_info={'cpu_info': 'Test CPU', 'memory_total_gb': 16.0},
            gpu_info={'name': 'Test GPU', 'total_memory_mb': 8192},
            test_duration_seconds=120.0
        )

        # Print summary
        self.optimizer.print_summary(report)

        # Capture output
        captured = capsys.readouterr()

        assert "BATCH SIZE OPTIMIZATION SUMMARY" in captured.out
        assert "Batch Size: 32" in captured.out
        assert "Throughput: 4000 inferences/sec" in captured.out
        assert "gomoku" in captured.out


class TestBatchSizeOptimizationReport:
    """Test BatchSizeOptimizationReport dataclass."""

    def test_report_creation(self):
        """Test optimization report creation."""
        optimal_result = BatchSizePerformanceResult(
            batch_size=64,
            throughput_inferences_per_sec=5000.0,
            average_latency_ms=15.0,
            latency_std_ms=3.0,
            gpu_utilization_percent=85.0,
            vram_usage_mb=5000.0,
            vram_usage_percent=62.5,
            memory_efficiency=1.0
        )

        report = BatchSizeOptimizationReport(
            test_config={'game_type': 'gomoku'},
            results=[optimal_result],
            optimal_batch_size=64,
            optimal_result=optimal_result,
            performance_curve=[(64, 0.85)],
            recommendations=["Test recommendation"],
            system_info={'cpu_info': 'Test CPU'},
            gpu_info={'name': 'Test GPU'},
            test_duration_seconds=180.0
        )

        assert report.test_config['game_type'] == 'gomoku'
        assert len(report.results) == 1
        assert report.optimal_batch_size == 64
        assert report.optimal_result == optimal_result
        assert len(report.performance_curve) == 1
        assert len(report.recommendations) == 1
        assert report.test_duration_seconds == 180.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])