"""
Neural network package for the AlphaZero engine.

Provides GPU device management, neural network models, and inference
optimization for high-performance game AI training and evaluation.
"""

from .device_manager import (
    DeviceManager,
    DeviceInfo,
    DummyModel,
    get_device_manager,
    initialize_device,
)

__all__ = [
    "DeviceManager",
    "DeviceInfo",
    "DummyModel",
    "get_device_manager",
    "initialize_device",
]
