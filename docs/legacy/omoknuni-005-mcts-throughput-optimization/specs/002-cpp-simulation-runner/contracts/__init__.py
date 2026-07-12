"""
API Contracts for C++ MCTS Simulation Runner (Spec 002)

This package contains the formal API contracts and interface definitions for the
C++ simulation runner implementation. These contracts serve as executable
specifications validated by the contract test suite.

Modules:
    simulation_runner_api: Core SimulationRunner interface contract
    tree_extensions_api: MCTSTree move storage extension contract
    cpp_inference_callback: PyInferenceCallback bridge contract
    threading_model: Threading and GIL release contract (documentation)
    performance_contracts: Performance targets and metrics (documentation)

Usage:
    These contracts are imported by tests/contract/ test suites to validate
    implementation compliance. Any changes to these contracts must be accompanied
    by corresponding implementation updates and test updates.

Version: 002-cpp-simulation-runner (2025-10-02)
"""

__all__ = [
    "simulation_runner_api",
    "tree_extensions_api",
    "cpp_inference_callback",
]

__version__ = "002-cpp-simulation-runner"
__status__ = "READY FOR IMPLEMENTATION"
