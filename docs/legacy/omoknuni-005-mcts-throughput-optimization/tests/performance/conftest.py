"""
Pytest configuration and fixtures for performance tests.

This module provides shared fixtures for the benchmark harness according to
spec.md v2.0 Task T001.
"""

import pytest
from tests.performance.fixtures import (
    default_benchmark_config,
    comprehensive_benchmark_config,
    thread_scaling_configs,
    batch_size_configs,
    timeout_configs,
    ablation_configs,
    multi_game_configs,
    sample_telemetry_data,
)

# Re-export all fixtures for pytest discovery
__all__ = [
    'default_benchmark_config',
    'comprehensive_benchmark_config',
    'thread_scaling_configs',
    'batch_size_configs',
    'timeout_configs',
    'ablation_configs',
    'multi_game_configs',
    'sample_telemetry_data',
]
