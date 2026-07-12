"""
Neural Network Model Validation
===============================

Comprehensive validation for neural network models to ensure integrity,
compatibility, and correctness before deployment or inference.
"""

import torch
import torch.nn as nn
import logging
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np

from src.utils.errors import (
    ModelError, ModelLoadError, ModelValidationError,
    with_error_handling, error_reporter
)


class ModelValidator:
    """Comprehensive neural network model validation."""

    def __init__(self, expected_input_shape: Tuple[int, ...],
                 expected_output_shapes: Dict[str, Tuple[int, ...]],
                 device: torch.device):
        """
        Initialize model validator.

        Args:
            expected_input_shape: Expected input tensor shape (without batch dimension)
            expected_output_shapes: Expected output shapes for each output head
            device: Device for validation testing
        """
        self.expected_input_shape = expected_input_shape
        self.expected_output_shapes = expected_output_shapes
        self.device = device
        self.logger = logging.getLogger(__name__)

    @with_error_handling()
    def validate_model_file(self, model_path: str) -> Dict[str, Any]:
        """
        Validate model file before loading.

        Args:
            model_path: Path to model file

        Returns:
            Dictionary with validation results

        Raises:
            ModelLoadError: If file validation fails
        """
        path = Path(model_path)

        if not path.exists():
            raise ModelLoadError(f"Model file does not exist", model_path=model_path)

        if not path.is_file():
            raise ModelLoadError(f"Model path is not a file", model_path=model_path)

        # Check file size (reasonable bounds)
        file_size = path.stat().st_size
        if file_size == 0:
            raise ModelLoadError(f"Model file is empty", model_path=model_path)

        if file_size > 2 * 1024**3:  # 2GB limit
            raise ModelLoadError(
                f"Model file too large: {file_size / 1024**3:.1f}GB > 2GB limit",
                model_path=model_path
            )

        # Calculate file checksum for integrity
        checksum = self._calculate_checksum(model_path)

        return {
            "path": str(path),
            "size_bytes": file_size,
            "size_mb": file_size / (1024**2),
            "checksum": checksum
        }

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    @with_error_handling()
    def validate_loaded_model(self, model: nn.Module,
                            model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of loaded model.

        Args:
            model: Loaded PyTorch model
            model_path: Optional path for error reporting

        Returns:
            Dictionary with validation results

        Raises:
            ModelValidationError: If validation fails
        """
        context = {"model_path": model_path} if model_path else {}

        # Basic model structure validation
        model_info = self._validate_model_structure(model, context)

        # Device compatibility validation
        device_info = self._validate_device_compatibility(model, context)

        # Forward pass validation
        forward_info = self._validate_forward_pass(model, context)

        # Output validation
        output_info = self._validate_model_outputs(model, context)

        # Memory usage validation
        memory_info = self._validate_memory_usage(model, context)

        return {
            "model_structure": model_info,
            "device_compatibility": device_info,
            "forward_pass": forward_info,
            "output_validation": output_info,
            "memory_usage": memory_info,
            "validation_passed": True
        }

    def _validate_model_structure(self, model: nn.Module,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic model structure and parameters."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Check for reasonable parameter count (for AlphaZero ResNet)
            expected_range = (10_000_000, 50_000_000)  # 10M - 50M parameters
            if not (expected_range[0] <= total_params <= expected_range[1]):
                raise ModelValidationError(
                    f"Parameter count outside expected range",
                    expected=f"{expected_range[0]:,} - {expected_range[1]:,}",
                    actual=f"{total_params:,}"
                )

            # Check model is in appropriate mode
            is_training = model.training
            if is_training:
                self.logger.warning("Model is in training mode, switching to eval for validation")
                model.eval()

            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
                "was_training_mode": is_training,
                "parameter_count_valid": True
            }
        except Exception as e:
            raise ModelValidationError(f"Model structure validation failed: {e}") from e

    def _validate_device_compatibility(self, model: nn.Module,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model can be moved to target device."""
        try:
            original_device = next(model.parameters()).device

            # Try moving to target device
            start_time = time.time()
            model.to(self.device)
            transfer_time = time.time() - start_time

            # Verify all parameters are on correct device
            devices = {p.device for p in model.parameters()}
            if len(devices) > 1:
                raise ModelValidationError(
                    f"Model parameters on multiple devices: {devices}",
                    expected=str(self.device),
                    actual=str(devices)
                )

            if next(model.parameters()).device != self.device:
                raise ModelValidationError(
                    f"Failed to move model to target device",
                    expected=str(self.device),
                    actual=str(next(model.parameters()).device)
                )

            return {
                "original_device": str(original_device),
                "target_device": str(self.device),
                "transfer_time_ms": transfer_time * 1000,
                "device_transfer_successful": True
            }
        except Exception as e:
            raise ModelValidationError(f"Device compatibility validation failed: {e}") from e

    def _validate_forward_pass(self, model: nn.Module,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model forward pass with test input."""
        try:
            # Create test input
            batch_size = 4  # Small batch for testing
            test_input = torch.randn(
                batch_size, *self.expected_input_shape,
                device=self.device, dtype=torch.float32
            )

            # Test forward pass
            start_time = time.time()
            with torch.no_grad():
                outputs = model(test_input)
            forward_time = time.time() - start_time

            # Validate outputs structure
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
                raise ModelValidationError(
                    f"Expected model to return (policy, value) tuple",
                    expected="tuple of length 2",
                    actual=f"{type(outputs)} of length {len(outputs) if hasattr(outputs, '__len__') else 'unknown'}"
                )

            policy, value = outputs

            return {
                "forward_pass_successful": True,
                "forward_time_ms": forward_time * 1000,
                "throughput_samples_per_sec": batch_size / forward_time,
                "test_batch_size": batch_size,
                "policy_shape": list(policy.shape),
                "value_shape": list(value.shape)
            }
        except Exception as e:
            raise ModelValidationError(f"Forward pass validation failed: {e}") from e

    def _validate_model_outputs(self, model: nn.Module,
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model output shapes and value ranges."""
        try:
            # Test with different batch sizes
            test_results = []
            batch_sizes = [1, 8, 32]

            for batch_size in batch_sizes:
                test_input = torch.randn(
                    batch_size, *self.expected_input_shape,
                    device=self.device, dtype=torch.float32
                )

                with torch.no_grad():
                    policy, value = model(test_input)

                # Validate policy output
                expected_policy_shape = (batch_size, self.expected_output_shapes["policy"][0])
                if policy.shape != expected_policy_shape:
                    raise ModelValidationError(
                        f"Invalid policy shape for batch size {batch_size}",
                        expected=expected_policy_shape,
                        actual=policy.shape
                    )

                # Validate value output
                expected_value_shape = (batch_size, 1)
                if value.shape != expected_value_shape:
                    raise ModelValidationError(
                        f"Invalid value shape for batch size {batch_size}",
                        expected=expected_value_shape,
                        actual=value.shape
                    )

                # Validate policy probabilities (should be non-negative)
                if (policy < 0).any():
                    raise ModelValidationError(
                        f"Policy contains negative values (batch size {batch_size})",
                        actual=f"min: {policy.min().item():.6f}"
                    )

                # Validate value range
                if not (-1.1 <= value.min().item() <= value.max().item() <= 1.1):
                    raise ModelValidationError(
                        f"Value predictions outside expected range [-1, 1] (batch size {batch_size})",
                        expected="[-1.0, 1.0]",
                        actual=f"[{value.min().item():.3f}, {value.max().item():.3f}]"
                    )

                test_results.append({
                    "batch_size": batch_size,
                    "policy_shape": list(policy.shape),
                    "value_shape": list(value.shape),
                    "policy_range": [policy.min().item(), policy.max().item()],
                    "value_range": [value.min().item(), value.max().item()],
                    "passed": True
                })

            return {
                "output_validation_successful": True,
                "batch_size_tests": test_results
            }
        except Exception as e:
            raise ModelValidationError(f"Output validation failed: {e}") from e

    def _validate_memory_usage(self, model: nn.Module,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model memory usage is reasonable."""
        try:
            if self.device.type == 'cuda':
                # GPU memory validation
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(self.device)

                # Test with larger batch to see memory scaling
                large_batch = 64
                test_input = torch.randn(
                    large_batch, *self.expected_input_shape,
                    device=self.device, dtype=torch.float32
                )

                with torch.no_grad():
                    outputs = model(test_input)

                peak_memory = torch.cuda.max_memory_allocated(self.device)
                memory_used = peak_memory - initial_memory

                # Check memory usage is reasonable (< 4GB for inference)
                memory_limit = 4 * 1024**3  # 4GB
                if memory_used > memory_limit:
                    raise ModelValidationError(
                        f"Model uses too much memory",
                        expected=f"< {memory_limit / 1024**3:.1f}GB",
                        actual=f"{memory_used / 1024**3:.1f}GB"
                    )

                del test_input, outputs
                torch.cuda.empty_cache()

                return {
                    "memory_validation_successful": True,
                    "peak_memory_mb": memory_used / (1024**2),
                    "test_batch_size": large_batch,
                    "device": "cuda"
                }
            else:
                # CPU memory validation (simplified)
                return {
                    "memory_validation_successful": True,
                    "device": "cpu",
                    "note": "CPU memory validation skipped"
                }
        except Exception as e:
            raise ModelValidationError(f"Memory usage validation failed: {e}") from e

    @with_error_handling()
    def validate_model_from_path(self, model_path: str) -> Dict[str, Any]:
        """
        Complete model validation pipeline from file path.

        Args:
            model_path: Path to model file

        Returns:
            Complete validation results

        Raises:
            ModelLoadError: If model loading fails
            ModelValidationError: If validation fails
        """
        try:
            # File validation
            file_info = self.validate_model_file(model_path)

            # Load model
            self.logger.info(f"Loading model from {model_path}")
            model = self._load_model_safely(model_path)

            # Model validation
            validation_info = self.validate_loaded_model(model, model_path)

            return {
                "file_validation": file_info,
                "model_validation": validation_info,
                "overall_validation_passed": True
            }
        except Exception as e:
            error_reporter.report_error(e, {"model_path": model_path})
            raise

    def _load_model_safely(self, model_path: str) -> nn.Module:
        """Load model with comprehensive error handling."""
        try:
            # Use map_location to handle device compatibility
            model_data = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False  # Allow loading full model objects
            )

            # Handle different save formats
            if isinstance(model_data, nn.Module):
                model = model_data
            elif isinstance(model_data, dict) and 'model' in model_data:
                model = model_data['model']
            elif isinstance(model_data, dict) and 'state_dict' in model_data:
                # Would need model architecture to load state_dict
                raise ModelLoadError(
                    "Model saved as state_dict - requires model architecture",
                    model_path=model_path
                )
            else:
                raise ModelLoadError(
                    f"Unrecognized model format: {type(model_data)}",
                    model_path=model_path
                )

            return model

        except torch.serialization.pickle.UnpicklingError as e:
            raise ModelLoadError(f"Failed to unpickle model: {e}", model_path=model_path) from e
        except RuntimeError as e:
            if "CUDA" in str(e):
                raise ModelLoadError(f"CUDA error loading model: {e}", model_path=model_path) from e
            raise ModelLoadError(f"Runtime error loading model: {e}", model_path=model_path) from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading model: {e}", model_path=model_path) from e


def create_gomoku_validator(device: torch.device) -> ModelValidator:
    """Create model validator for Gomoku models."""
    return ModelValidator(
        expected_input_shape=(36, 15, 15),  # Gomoku feature planes
        expected_output_shapes={
            "policy": (225,),  # 15x15 board positions
            "value": (1,)      # Single value prediction
        },
        device=device
    )


def create_chess_validator(device: torch.device) -> ModelValidator:
    """Create model validator for Chess models."""
    return ModelValidator(
        expected_input_shape=(30, 8, 8),  # Chess feature planes
        expected_output_shapes={
            "policy": (4096,),  # Chess move encoding
            "value": (1,)       # Single value prediction
        },
        device=device
    )


def create_go_validator(device: torch.device, board_size: int = 19) -> ModelValidator:
    """Create model validator for Go models."""
    return ModelValidator(
        expected_input_shape=(25, board_size, board_size),  # Go feature planes
        expected_output_shapes={
            "policy": (board_size * board_size + 1,),  # Board positions + pass
            "value": (1,)  # Single value prediction
        },
        device=device
    )