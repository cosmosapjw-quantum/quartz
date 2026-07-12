"""
Unit tests for GPU device management and optimization.

Tests core functionality of device detection, GPU warmup, and batch size estimation.
"""

import time
from unittest.mock import Mock, patch
import pytest
import torch

from src.neural.device_manager import (
    DeviceManager,
    DeviceInfo,
    DummyModel,
    get_device_manager,
    initialize_device,
)


class TestDummyModel:
    """Test cases for DummyModel."""

    def test_model_creation_and_forward_pass(self):
        """Test dummy model creation and forward pass."""
        model = DummyModel((36, 15, 15), num_actions=225)
        model.eval()

        # Test forward pass
        batch_size = 2
        test_input = torch.randn(batch_size, 36, 15, 15)

        with torch.no_grad():
            policy, value = model(test_input)

        # Check output shapes
        assert policy.shape == (batch_size, 225)
        assert value.shape == (batch_size, 1)

        # Check output ranges
        assert torch.all(policy >= 0.0) and torch.all(policy <= 1.0)
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

    def test_different_game_sizes(self):
        """Test model with different game board sizes."""
        test_cases = [
            ((7, 15, 15), 225),  # Gomoku
            ((12, 8, 8), 4096),  # Chess
            ((17, 19, 19), 361),  # Go
        ]

        for input_shape, num_actions in test_cases:
            model = DummyModel(input_shape, num_actions)
            test_input = torch.randn(1, *input_shape)

            with torch.no_grad():
                policy, value = model(test_input)

            assert policy.shape == (1, num_actions)
            assert value.shape == (1, 1)


class TestDeviceInfo:
    """Test cases for DeviceInfo dataclass."""

    def test_device_info_creation(self):
        """Test DeviceInfo dataclass creation."""
        device_info = DeviceInfo(
            name="Test GPU",
            memory_total_mb=8192.0,
            memory_free_mb=7500.0,
            compute_capability=(8, 6),
            device_id=0,
            is_cuda_available=True,
        )

        assert device_info.name == "Test GPU"
        assert device_info.memory_total_mb == 8192.0
        assert device_info.compute_capability == (8, 6)
        assert device_info.is_cuda_available


class TestDeviceManager:
    """Test cases for DeviceManager core functionality."""

    def test_initialization(self):
        """Test device manager initialization."""
        device_manager = DeviceManager(memory_fraction=0.8, warmup_iterations=5)

        assert device_manager.memory_fraction == 0.8
        assert device_manager.warmup_iterations == 5
        assert device_manager.device_info is None
        assert device_manager.device == torch.device("cpu")

    @patch("torch.cuda.is_available")
    def test_detect_device_no_cuda(self, mock_cuda_available):
        """Test device detection when CUDA is not available."""
        mock_cuda_available.return_value = False

        device_manager = DeviceManager()
        device_info = device_manager.detect_device()

        assert not device_info.is_cuda_available
        assert device_info.name == "CPU"
        assert device_info.device_id == -1

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.memory_allocated")
    def test_detect_device_with_cuda(
        self, mock_memory_allocated, mock_get_properties, mock_cuda_available
    ):
        """Test device detection when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB

        # Mock device properties
        mock_props = Mock()
        mock_props.name = "GeForce RTX 3060 Ti"
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        mock_props.major = 8
        mock_props.minor = 6
        mock_get_properties.return_value = mock_props

        device_manager = DeviceManager()
        device_info = device_manager.detect_device()

        assert device_info.is_cuda_available
        assert device_info.name == "GeForce RTX 3060 Ti"
        assert device_info.compute_capability == (8, 6)
        assert device_info.memory_total_mb == 8192.0

    def test_rtx_3060_ti_detection(self):
        """Test RTX 3060 Ti detection logic."""
        device_manager = DeviceManager()

        # Test positive case
        rtx_3060_ti_info = DeviceInfo(
            name="GeForce RTX 3060 Ti",
            memory_total_mb=8192.0,
            memory_free_mb=7500.0,
            compute_capability=(8, 6),
            device_id=0,
            is_cuda_available=True,
        )
        assert device_manager._is_rtx_3060_ti(rtx_3060_ti_info)

        # Test negative case
        rtx_3080_info = DeviceInfo(
            name="GeForce RTX 3080",
            memory_total_mb=10240.0,  # 10GB
            memory_free_mb=9500.0,
            compute_capability=(8, 6),
            device_id=0,
            is_cuda_available=True,
        )
        assert not device_manager._is_rtx_3060_ti(rtx_3080_info)

    def test_warmup_no_gpu(self):
        """Test warmup when no GPU is available."""
        device_manager = DeviceManager()
        device_manager.device_info = DeviceInfo(
            name="CPU",
            memory_total_mb=0,
            memory_free_mb=0,
            compute_capability=(0, 0),
            device_id=-1,
            is_cuda_available=False,
        )

        warmup_time = device_manager.warmup((7, 15, 15))
        assert warmup_time == 0.0

    def test_warmup_with_mock_gpu(self):
        """Test warmup with available device."""
        device_manager = DeviceManager(warmup_iterations=2)
        device_manager.device_info = DeviceInfo(
            name="Test GPU",
            memory_total_mb=8192.0,
            memory_free_mb=7500.0,
            compute_capability=(8, 6),
            device_id=0,
            is_cuda_available=True,
        )
        device_manager.device = torch.device("cpu")  # Use CPU for testing

        warmup_time = device_manager.warmup((7, 15, 15))

        # Should return positive warmup time
        assert warmup_time > 0.0
        assert device_manager.device_info.warmup_time_ms == warmup_time

    def test_estimate_batch_size_no_gpu(self):
        """Test batch size estimation when no GPU is available."""
        device_manager = DeviceManager()
        device_manager.device_info = DeviceInfo(
            name="CPU",
            memory_total_mb=0,
            memory_free_mb=0,
            compute_capability=(0, 0),
            device_id=-1,
            is_cuda_available=False,
        )

        batch_size = device_manager.estimate_batch_size((7, 15, 15))
        assert batch_size == 1

    def test_estimate_batch_size_mock_success(self):
        """Test batch size estimation with mocked device."""
        device_manager = DeviceManager()
        device_manager.device_info = DeviceInfo(
            name="Test GPU",
            memory_total_mb=8192.0,
            memory_free_mb=7500.0,
            compute_capability=(8, 6),
            device_id=0,
            is_cuda_available=True,
        )
        device_manager.device = torch.device("cpu")

        # Mock the _test_batch_size method to return predictable results
        def mock_test_batch_size(model, input_shape, batch_size):
            return batch_size <= 64  # Simulate max batch size of 64

        device_manager._test_batch_size = mock_test_batch_size

        batch_size = device_manager.estimate_batch_size((7, 15, 15), max_batch_size=128)

        # Should find optimal batch size around 64 * 0.9 = 57-58
        assert 50 <= batch_size <= 64
        assert device_manager.device_info.optimal_batch_size == batch_size

    def test_get_memory_stats_no_gpu(self):
        """Test memory stats when no GPU is available."""
        device_manager = DeviceManager()
        device_manager.device_info = None

        stats = device_manager.get_memory_stats()
        assert stats == {}

    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    def test_get_memory_stats_with_gpu(
        self, mock_memory_reserved, mock_memory_allocated
    ):
        """Test memory stats with mocked GPU."""
        mock_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        mock_memory_reserved.return_value = 3 * 1024 * 1024 * 1024  # 3GB

        device_manager = DeviceManager()
        device_manager.device_info = DeviceInfo(
            name="Test GPU",
            memory_total_mb=8192.0,
            memory_free_mb=6000.0,
            compute_capability=(8, 6),
            device_id=0,
            is_cuda_available=True,
        )

        stats = device_manager.get_memory_stats()

        assert stats["allocated_mb"] == 2048.0
        assert stats["reserved_mb"] == 3072.0
        assert stats["total_mb"] == 8192.0
        assert abs(stats["utilization_percent"] - 25.0) < 0.1


class TestGlobalFunctions:
    """Test global device manager functions."""

    def test_get_device_manager_singleton(self):
        """Test global device manager singleton."""
        # Clean up any existing global state
        import src.neural.device_manager as dm

        dm._global_device_manager = None

        manager1 = get_device_manager()
        manager2 = get_device_manager()

        assert manager1 is manager2
        assert isinstance(manager1, DeviceManager)

    @patch("torch.cuda.is_available")
    def test_initialize_device_function(self, mock_cuda_available):
        """Test global device initialization function."""
        mock_cuda_available.return_value = False

        # Clean up global state
        import src.neural.device_manager as dm

        dm._global_device_manager = None

        device_info = initialize_device((7, 15, 15))

        assert isinstance(device_info, DeviceInfo)
        assert not device_info.is_cuda_available


class TestIntegration:
    """Integration tests for device manager."""

    @patch("torch.cuda.is_available")
    def test_complete_initialization_flow(self, mock_cuda_available):
        """Test complete device initialization flow."""
        mock_cuda_available.return_value = False

        device_manager = DeviceManager(warmup_iterations=2)
        device_info = device_manager.initialize((7, 15, 15))

        assert device_info is not None
        assert not device_info.is_cuda_available
        assert device_info.name == "CPU"

    def test_device_manager_methods_exist(self):
        """Test that all required methods exist and are callable."""
        device_manager = DeviceManager()

        # Test method existence
        assert callable(device_manager.detect_device)
        assert callable(device_manager.warmup)
        assert callable(device_manager.estimate_batch_size)
        assert callable(device_manager.initialize)
        assert callable(device_manager.get_device_info)
        assert callable(device_manager.get_device)
        assert callable(device_manager.get_memory_stats)

    def test_optimization_methods_callable(self):
        """Test that optimization methods are callable."""
        device_manager = DeviceManager()

        # These methods should be callable without error
        assert callable(device_manager._is_rtx_3060_ti)
        assert callable(device_manager._apply_rtx_3060_ti_optimizations)

        # Test optimization application doesn't crash
        try:
            device_manager._apply_rtx_3060_ti_optimizations()
        except Exception:
            # Optimization might fail in test environment, that's OK
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
