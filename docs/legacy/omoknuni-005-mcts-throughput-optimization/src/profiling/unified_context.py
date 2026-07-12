"""
Unified Profiling Context Manager

Provides a convenient context manager for profiling both C++ and Python
components simultaneously.

Example:
    >>> from src.profiling.unified_context import UnifiedProfilingContext
    >>>
    >>> with UnifiedProfilingContext("benchmark", output_dir="results") as profiler:
    ...     # Both C++ and Python profiling active
    ...     mcts.search(state, simulations=800)
    ...
    >>> # Results automatically exported on context exit

Author: MCTS Performance Team
Date: 2025-10-15
"""

from pathlib import Path
from typing import Optional
from contextlib import contextmanager

try:
    import mcts_py
except ImportError:
    raise ImportError(
        "mcts_py not found. Build C++ extensions first:\n"
        "  export CXXFLAGS='-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3'\n"
        "  pip install -e . --force-reinstall --no-deps"
    )

from src.profiling import ProfilingSession, ProfilerConfig


class UnifiedProfilingContext:
    """Context manager for unified C++/Python profiling

    Automatically manages:
    - C++ profiler lifecycle (start/stop/export)
    - Python profiler lifecycle
    - Result export to organized directory
    - Validation (optional)

    Args:
        session_name: Name for this profiling session
        output_dir: Directory for profiling results (default: profiling_results)
        validate_first: Run validation before profiling (default: False)
        cpp_level: C++ profiling level (default: FULL)
        enable_python: Enable Python profiling (default: True)

    Example:
        >>> with UnifiedProfilingContext("mcts_benchmark") as ctx:
        ...     # Your MCTS code here
        ...     mcts.search(state, simulations=800)
        ...
        >>> # Results in profiling_results/:
        >>> #   - cpp_profiling.json
        >>> #   - cpp_trace.json
        >>> #   - cpp_report.md
        >>> #   - python_profiling.json
    """

    def __init__(
        self,
        session_name: str,
        output_dir: str = "profiling_results",
        validate_first: bool = False,
        cpp_level: Optional[mcts_py.ProfileLevel] = None,
        enable_python: bool = True
    ):
        self.session_name = session_name
        self.output_dir = Path(output_dir)
        self.validate_first = validate_first
        self.cpp_level = cpp_level or mcts_py.ProfileLevel.FULL
        self.enable_python = enable_python

        self.cpp_profiler = None
        self.python_session = None

    def __enter__(self):
        """Start profiling session"""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optional validation
        if self.validate_first:
            print("Running validation...")
            if not mcts_py.run_profiling_validation():
                raise RuntimeError("Profiling validation failed! Fix issues before profiling.")
            print("✅ Validation passed\n")

        # Start C++ profiler
        print(f"Starting C++ profiler (level: {self.cpp_level.name})...")
        self.cpp_profiler = mcts_py.EnhancedProfiler.instance()
        self.cpp_profiler.set_enabled(True)
        self.cpp_profiler.set_level(self.cpp_level)
        self.cpp_profiler.start_session(self.session_name)
        print("✅ C++ profiler active\n")

        # Start Python profiler
        if self.enable_python:
            print("Starting Python profiler...")
            config = ProfilerConfig(
                enable_gil_profiling=True,
                enable_inference_profiling=True,
                enable_cpp_instrumentation=True,
                enable_thread_profiling=True
            )
            self.python_session = ProfilingSession(config)
            self.python_session.__enter__()
            print("✅ Python profiler active\n")

        print(f"🚀 Profiling session '{self.session_name}' started")
        print(f"   Output directory: {self.output_dir.absolute()}\n")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and export results"""
        print("\n" + "="*60)
        print("EXPORTING PROFILING RESULTS")
        print("="*60)

        try:
            # Stop and export C++ profiler
            if self.cpp_profiler:
                print("\nExporting C++ results...")
                self.cpp_profiler.stop_session()

                cpp_json = self.output_dir / "cpp_profiling.json"
                cpp_trace = self.output_dir / "cpp_trace.json"
                cpp_report = self.output_dir / "cpp_report.md"

                self.cpp_profiler.export_json(str(cpp_json))
                self.cpp_profiler.export_chrome_trace(str(cpp_trace))
                self.cpp_profiler.export_markdown(str(cpp_report))

                print(f"  ✅ JSON:      {cpp_json}")
                print(f"  ✅ Trace:     {cpp_trace}")
                print(f"  ✅ Report:    {cpp_report}")

                # Print summary to console
                print("\n--- C++ Profiling Summary ---")
                self.cpp_profiler.print_summary()

            # Stop and export Python profiler
            if self.python_session:
                print("\nExporting Python results...")
                self.python_session.__exit__(exc_type, exc_val, exc_tb)

                python_json = self.output_dir / "python_profiling.json"
                metrics = self.python_session.get_all_metrics()

                import json
                with open(python_json, 'w') as f:
                    json.dump(metrics, f, indent=2)

                print(f"  ✅ JSON:      {python_json}")

            print("\n" + "="*60)
            print("✅ PROFILING COMPLETE")
            print("="*60)
            print(f"\nResults saved to: {self.output_dir.absolute()}")
            print("\nNext steps:")
            print("1. Review cpp_report.md for comprehensive analysis")
            print("2. Open cpp_trace.json in chrome://tracing")
            print("3. Inspect cpp_profiling.json for raw metrics\n")

        except Exception as e:
            print(f"\n❌ Error exporting results: {e}")
            import traceback
            traceback.print_exc()

        # Don't suppress exceptions from the with block
        return False


@contextmanager
def profile_mcts(
    session_name: str = "mcts_profile",
    validate: bool = False,
    output_dir: str = "profiling_results"
):
    """Convenience context manager for MCTS profiling

    Args:
        session_name: Name for profiling session
        validate: Run validation first
        output_dir: Output directory for results

    Example:
        >>> from src.profiling.unified_context import profile_mcts
        >>>
        >>> with profile_mcts("benchmark", validate=True):
        ...     mcts.search(state, simulations=800)
    """
    with UnifiedProfilingContext(
        session_name=session_name,
        output_dir=output_dir,
        validate_first=validate,
        cpp_level=mcts_py.ProfileLevel.FULL,
        enable_python=True
    ) as ctx:
        yield ctx
