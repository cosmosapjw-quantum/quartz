"""
Unit tests for GPU profiling system.

Tests the GPU profiler's ability to:
- Capture hardware metrics via NVML
- Profile batch inference timing
- Track memory usage
- Generate comprehensive reports
- Export to TensorBoard
"""

import pytest
import torch
import torch.nn as nn
import time
import tempfile
from pathlib import Path

from src.telemetry.gpu_profiler import (
    GPUProfiler,
    CUDAMetrics,
    InferenceBatchMetrics,
    ProfilingSession
)
from src.neural.model import create_model_for_game


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for profiling outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 10)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUProfiler:
    """Test GPU profiler functionality."""

    def test_profiler_initialization(self, temp_log_dir):
        """Test profiler initializes correctly."""
        profiler = GPUProfiler(
            device='cuda:0',
            log_dir=temp_log_dir,
            enable_nvml=True
        )

        assert profiler.device.type == 'cuda'
        assert profiler.device_id == 0
        assert Path(profiler.log_dir).exists()
        assert profiler._profiling is False

    def test_start_stop_profiling(self, temp_log_dir):
        """Test start/stop profiling workflow."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)

        profiler.start_profiling()
        assert profiler._profiling is True
        assert profiler.session_start_time is not None

        # Allow monitoring thread to collect some samples
        time.sleep(0.5)

        profiler.stop_profiling()
        assert profiler._profiling is False
        assert profiler.session_end_time is not None

    def test_profile_batch_context(self, temp_log_dir, simple_model):
        """Test batch profiling context manager."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        batch_size = 32
        input_tensor = torch.randn(batch_size, 3, 64, 64).cuda()

        with profiler.profile_batch(batch_size=batch_size):
            with torch.no_grad():
                output = model(input_tensor)

        profiler.stop_profiling()

        # Check batch metrics were recorded
        assert len(profiler._batch_metrics) == 1
        batch_metrics = profiler._batch_metrics[0]

        assert batch_metrics.batch_size == batch_size
        assert batch_metrics.total_time_ms > 0
        assert batch_metrics.samples_per_second > 0

    def test_multiple_batches(self, temp_log_dir, simple_model):
        """Test profiling multiple batches."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        num_batches = 10
        batch_size = 32

        with torch.no_grad():
            for i in range(num_batches):
                input_tensor = torch.randn(batch_size, 3, 64, 64).cuda()

                with profiler.profile_batch(batch_size=batch_size):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Check all batches were recorded
        assert len(profiler._batch_metrics) == num_batches

        # Check metrics are reasonable
        for batch_metrics in profiler._batch_metrics:
            assert batch_metrics.total_time_ms > 0
            assert batch_metrics.total_time_ms < 1000  # Should be < 1 second
            assert batch_metrics.samples_per_second > 0

    def test_transfer_profiling(self, temp_log_dir, simple_model):
        """Test H2D and D2H transfer profiling."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        batch_size = 64
        input_tensor = torch.randn(batch_size, 3, 64, 64)

        with profiler.profile_batch(batch_size=batch_size):
            # Profile H2D transfer
            with profiler.profile_transfer('h2d'):
                input_gpu = input_tensor.cuda()

            # Profile inference
            with torch.no_grad():
                output_gpu = model(input_gpu)

            # Profile D2H transfer
            with profiler.profile_transfer('d2h'):
                output_cpu = output_gpu.cpu()

        profiler.stop_profiling()

        # Check transfer times were recorded
        batch_metrics = profiler._batch_metrics[0]
        assert batch_metrics.h2d_transfer_ms >= 0
        assert batch_metrics.d2h_transfer_ms >= 0

    def test_cuda_metrics_collection(self, temp_log_dir):
        """Test CUDA metrics collection via NVML."""
        profiler = GPUProfiler(
            device='cuda:0',
            log_dir=temp_log_dir,
            enable_nvml=True,
            sampling_interval_ms=100
        )

        if not profiler.enable_nvml:
            pytest.skip("NVML not available")

        profiler.start_profiling()

        # Wait for some samples
        time.sleep(0.5)

        profiler.stop_profiling()

        # Check CUDA metrics were collected
        assert len(profiler._cuda_metrics) > 0

        # Validate metrics structure
        for metrics in profiler._cuda_metrics:
            assert isinstance(metrics, CUDAMetrics)
            assert 0 <= metrics.gpu_utilization <= 100
            assert metrics.memory_used_mb >= 0
            assert metrics.memory_total_mb > 0
            assert metrics.power_draw_watts >= 0
            assert metrics.temperature_c >= 0

    def test_memory_snapshot(self, temp_log_dir, simple_model):
        """Test memory snapshot capture."""
        profiler = GPUProfiler(
            device='cuda:0',
            log_dir=temp_log_dir,
            memory_profiling=True
        )
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        # Run enough batches to trigger memory snapshots (every 10 batches)
        with torch.no_grad():
            for i in range(15):
                input_tensor = torch.randn(32, 3, 64, 64).cuda()
                with profiler.profile_batch(batch_size=32):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Check memory snapshots were captured
        assert len(profiler._memory_snapshots) > 0

        # Validate snapshot structure
        for snapshot in profiler._memory_snapshots:
            assert snapshot.allocated_mb >= 0
            assert snapshot.reserved_mb >= snapshot.allocated_mb
            assert 0 <= snapshot.fragmentation_ratio <= 1

    def test_realtime_metrics(self, temp_log_dir):
        """Test real-time metrics retrieval."""
        profiler = GPUProfiler(
            device='cuda:0',
            log_dir=temp_log_dir,
            enable_nvml=True
        )

        if not profiler.enable_nvml:
            pytest.skip("NVML not available")

        metrics = profiler.get_realtime_metrics()

        assert 'cuda' in metrics
        assert 'memory' in metrics

        # Validate CUDA metrics
        cuda_metrics = metrics['cuda']
        assert 'gpu_utilization' in cuda_metrics
        assert 'memory_used_mb' in cuda_metrics
        assert 'power_draw_watts' in cuda_metrics

        # Validate memory metrics
        memory_metrics = metrics['memory']
        assert 'allocated_mb' in memory_metrics
        assert 'reserved_mb' in memory_metrics

    def test_generate_report(self, temp_log_dir, simple_model):
        """Test profiling report generation."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        # Run some batches
        with torch.no_grad():
            for i in range(20):
                input_tensor = torch.randn(32, 3, 64, 64).cuda()
                with profiler.profile_batch(batch_size=32):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Generate report
        report = profiler.generate_report()

        assert isinstance(report, ProfilingSession)
        assert report.total_batches == 20
        assert report.total_samples == 20 * 32
        assert report.avg_batch_size == 32
        assert report.duration_seconds > 0
        assert report.avg_throughput > 0
        assert report.gpu_name is not None
        assert len(report.batch_metrics) == 20

    def test_export_report(self, temp_log_dir, simple_model):
        """Test report export to JSON."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        # Run batches
        with torch.no_grad():
            for i in range(5):
                input_tensor = torch.randn(32, 3, 64, 64).cuda()
                with profiler.profile_batch(batch_size=32):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Export report
        report_path = profiler.export_report()

        assert report_path.exists()
        assert report_path.suffix == '.json'

        # Load and validate JSON
        import json
        with open(report_path) as f:
            report_data = json.load(f)

        assert 'session_id' in report_data
        assert 'total_batches' in report_data
        assert 'avg_throughput' in report_data

    def test_context_manager(self, temp_log_dir, simple_model):
        """Test profiler as context manager."""
        model = simple_model.cuda().eval()

        with GPUProfiler(device='cuda:0', log_dir=temp_log_dir) as profiler:
            # Profiling should be started automatically
            assert profiler._profiling is True

            with torch.no_grad():
                for i in range(5):
                    input_tensor = torch.randn(32, 3, 64, 64).cuda()
                    with profiler.profile_batch(batch_size=32):
                        output = model(input_tensor)

        # Profiling should be stopped automatically
        # (profiler is out of scope, but we can't check its state)

    def test_mixed_precision_profiling(self, temp_log_dir):
        """Test profiling with mixed precision inference."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = create_model_for_game('gomoku').cuda().eval()

        profiler.start_profiling()

        batch_size = 32
        input_tensor = torch.randn(batch_size, 36, 15, 15).cuda()

        # FP32 baseline
        with profiler.profile_batch(batch_size=batch_size):
            with torch.no_grad():
                policy, value = model(input_tensor)

        # FP16 mixed precision
        with profiler.profile_batch(batch_size=batch_size):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    policy_fp16, value_fp16 = model(input_tensor)

        profiler.stop_profiling()

        assert len(profiler._batch_metrics) == 2

        # FP16 should typically be faster (though not guaranteed in all cases)
        fp32_time = profiler._batch_metrics[0].total_time_ms
        fp16_time = profiler._batch_metrics[1].total_time_ms

        print(f"FP32 time: {fp32_time:.3f}ms, FP16 time: {fp16_time:.3f}ms")

    def test_batch_size_scaling(self, temp_log_dir, simple_model):
        """Test profiling across different batch sizes."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        batch_sizes = [8, 16, 32, 64]
        profiler.start_profiling()

        with torch.no_grad():
            for batch_size in batch_sizes:
                input_tensor = torch.randn(batch_size, 3, 64, 64).cuda()
                with profiler.profile_batch(batch_size=batch_size):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Check all batch sizes were profiled
        assert len(profiler._batch_metrics) == len(batch_sizes)

        # Larger batches should have higher throughput
        throughputs = [m.samples_per_second for m in profiler._batch_metrics]
        print(f"Throughputs by batch size: {list(zip(batch_sizes, throughputs))}")

    @pytest.mark.skipif(
        not hasattr(torch.profiler, 'profile'),
        reason="torch.profiler not available"
    )
    def test_torch_profiler_integration(self, temp_log_dir, simple_model):
        """Test integration with torch.profiler."""
        profiler = GPUProfiler(
            device='cuda:0',
            log_dir=temp_log_dir,
            enable_torch_profiler=True,
            tensorboard_export=False  # Disable for test speed
        )
        model = simple_model.cuda().eval()

        profiler.start_profiling(
            profile_tensorboard=False,
            active_steps=5
        )

        with torch.no_grad():
            for i in range(10):
                input_tensor = torch.randn(32, 3, 64, 64).cuda()
                with profiler.profile_batch(batch_size=32):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Torch profiler should have been active
        assert profiler._torch_profiler is None  # Should be stopped now

    def test_profiler_without_nvml(self, temp_log_dir, simple_model):
        """Test profiler works without NVML (fallback mode)."""
        profiler = GPUProfiler(
            device='cuda:0',
            log_dir=temp_log_dir,
            enable_nvml=False  # Explicitly disable
        )
        model = simple_model.cuda().eval()

        profiler.start_profiling()

        with torch.no_grad():
            for i in range(5):
                input_tensor = torch.randn(32, 3, 64, 64).cuda()
                with profiler.profile_batch(batch_size=32):
                    output = model(input_tensor)

        profiler.stop_profiling()

        # Should still generate report without NVML metrics
        report = profiler.generate_report()
        assert report.total_batches == 5

        # CUDA metrics should be empty
        assert len(profiler._cuda_metrics) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUProfilerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_profiling_session(self, temp_log_dir):
        """Test generating report from empty session."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)

        profiler.start_profiling()
        time.sleep(0.1)
        profiler.stop_profiling()

        # Should generate report even with no batches
        report = profiler.generate_report()
        assert report.total_batches == 0
        assert report.total_samples == 0

    def test_double_start(self, temp_log_dir):
        """Test starting profiler twice."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)

        profiler.start_profiling()
        profiler.start_profiling()  # Should be no-op

        # Should still work normally
        profiler.stop_profiling()

    def test_stop_without_start(self, temp_log_dir):
        """Test stopping profiler that wasn't started."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)

        # Should be safe no-op
        profiler.stop_profiling()

    def test_profile_batch_without_start(self, temp_log_dir, simple_model):
        """Test using profile_batch without starting profiler."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)
        model = simple_model.cuda().eval()

        input_tensor = torch.randn(32, 3, 64, 64).cuda()

        # Should be no-op
        with profiler.profile_batch(batch_size=32):
            with torch.no_grad():
                output = model(input_tensor)

        # No metrics should be recorded
        assert len(profiler._batch_metrics) == 0

    def test_report_without_profiling(self, temp_log_dir):
        """Test generating report without running profiler."""
        profiler = GPUProfiler(device='cuda:0', log_dir=temp_log_dir)

        with pytest.raises(RuntimeError, match="No profiling session"):
            profiler.generate_report()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
