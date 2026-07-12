#!/usr/bin/env python3
"""
Memory Leak Detection System for AlphaZero Engine
=================================================

Comprehensive memory leak detection with valgrind integration, Python memory profiling,
and automated testing for both Python and C++ components.

This script provides:
- Python memory profiling with tracemalloc and psutil
- Valgrind integration for C++ extension leak detection
- GPU memory leak monitoring with CUDA tools
- Automated leak detection during CI/CD and development
- Detailed reporting and analysis with trend detection

Usage:
    python scripts/check_memory_leaks.py --python --duration 300
    python scripts/check_memory_leaks.py --valgrind --component mcts
    python scripts/check_memory_leaks.py --gpu --cuda-memcheck
    python scripts/check_memory_leaks.py --all --output leak_report.json
"""

import argparse
import json
import logging
import os
import psutil
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('memory_leak_detector')

# Try to import CUDA monitoring if available
try:
    import pynvml
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    pynvml = None

# Try to import torch for GPU memory monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    timestamp: float
    python_rss_mb: float
    python_vms_mb: float
    gpu_allocated_mb: float = 0.0
    gpu_cached_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0
    fd_count: int = 0
    tracemalloc_current_mb: float = 0.0
    tracemalloc_peak_mb: float = 0.0


@dataclass
class LeakDetectionResult:
    """Result of memory leak detection analysis."""
    duration_seconds: float
    total_samples: int
    memory_growth_mb: float
    growth_rate_mb_per_hour: float
    leak_detected: bool
    leak_threshold_mb: float
    components_tested: List[str]
    valgrind_reports: List[Dict[str, Any]]
    python_profile_summary: Dict[str, Any]
    gpu_leaks: List[Dict[str, Any]]
    snapshots: List[MemorySnapshot]
    recommendations: List[str]


class PythonMemoryProfiler:
    """Python memory profiling with tracemalloc and psutil."""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.process = psutil.Process()
        self.tracemalloc_enabled = False

    def start_profiling(self) -> None:
        """Start memory profiling."""
        try:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            logger.info("Started tracemalloc profiling")
        except Exception as e:
            logger.warning(f"Failed to start tracemalloc: {e}")
            self.tracemalloc_enabled = False

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        memory_info = self.process.memory_info()

        # Get tracemalloc stats if available
        tracemalloc_current = 0.0
        tracemalloc_peak = 0.0
        if self.tracemalloc_enabled:
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current = current / 1024**2
                tracemalloc_peak = peak / 1024**2
            except Exception as e:
                logger.debug(f"Failed to get tracemalloc memory: {e}")

        # Get GPU memory if available
        gpu_allocated = 0.0
        gpu_cached = 0.0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_allocated = torch.cuda.memory_allocated() / 1024**2
                gpu_cached = torch.cuda.memory_reserved() / 1024**2
            except Exception as e:
                logger.debug(f"Failed to get CUDA memory: {e}")

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            python_rss_mb=memory_info.rss / 1024**2,
            python_vms_mb=memory_info.vms / 1024**2,
            gpu_allocated_mb=gpu_allocated,
            gpu_cached_mb=gpu_cached,
            process_count=len(self.process.children(recursive=True)) + 1,
            thread_count=self.process.num_threads(),
            fd_count=self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            tracemalloc_current_mb=tracemalloc_current,
            tracemalloc_peak_mb=tracemalloc_peak
        )

        self.snapshots.append(snapshot)
        return snapshot

    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return summary."""
        summary = {}

        if self.tracemalloc_enabled:
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                summary['top_memory_usage'] = []
                for index, stat in enumerate(top_stats[:10]):
                    summary['top_memory_usage'].append({
                        'rank': index + 1,
                        'size_mb': stat.size / 1024**2,
                        'count': stat.count,
                        'filename': stat.traceback.format()[0] if stat.traceback else 'unknown'
                    })

                tracemalloc.stop()
                logger.info("Stopped tracemalloc profiling")
            except Exception as e:
                logger.warning(f"Failed to get tracemalloc summary: {e}")

        summary['total_snapshots'] = len(self.snapshots)
        summary['profiling_duration'] = (
            self.snapshots[-1].timestamp - self.snapshots[0].timestamp
            if len(self.snapshots) >= 2 else 0.0
        )

        return summary

    def analyze_growth(self, threshold_mb_per_hour: float = 10.0) -> Tuple[bool, float]:
        """Analyze memory growth for leak detection.

        Returns:
            Tuple of (leak_detected, growth_rate_mb_per_hour)
        """
        if len(self.snapshots) < 2:
            return False, 0.0

        # Calculate linear regression for memory growth
        times = np.array([s.timestamp for s in self.snapshots])
        rss_values = np.array([s.python_rss_mb for s in self.snapshots])

        # Convert to hours for rate calculation
        duration_hours = (times[-1] - times[0]) / 3600.0
        if duration_hours <= 0:
            return False, 0.0

        # Linear regression to find growth rate
        coeffs = np.polyfit(times - times[0], rss_values, 1)
        growth_rate_mb_per_sec = coeffs[0]
        growth_rate_mb_per_hour = growth_rate_mb_per_sec * 3600.0

        # Total growth over the measurement period
        total_growth_mb = rss_values[-1] - rss_values[0]

        leak_detected = abs(growth_rate_mb_per_hour) > threshold_mb_per_hour

        logger.info(f"Memory growth analysis:")
        logger.info(f"  Duration: {duration_hours:.2f} hours")
        logger.info(f"  Total growth: {total_growth_mb:.2f} MB")
        logger.info(f"  Growth rate: {growth_rate_mb_per_hour:.2f} MB/hour")
        logger.info(f"  Leak detected: {leak_detected}")

        return leak_detected, growth_rate_mb_per_hour


class ValgrindIntegration:
    """Valgrind integration for C++ extension leak detection."""

    def __init__(self):
        self.valgrind_available = self._check_valgrind_available()

    def _check_valgrind_available(self) -> bool:
        """Check if valgrind is available on the system."""
        try:
            result = subprocess.run(['valgrind', '--version'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Valgrind available: {result.stdout.strip()}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        logger.warning("Valgrind not available - C++ leak detection disabled")
        return False

    def run_valgrind_check(self, component: str, test_script: Optional[str] = None) -> Dict[str, Any]:
        """Run valgrind memory check on a specific component.

        Args:
            component: Component to test ('mcts', 'games', 'python_bindings')
            test_script: Optional test script to run under valgrind

        Returns:
            Dictionary containing valgrind results
        """
        if not self.valgrind_available:
            return {'error': 'Valgrind not available'}

        # Create temporary valgrind log file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.valgrind', delete=False) as f:
            log_file = f.name

        try:
            # Build valgrind command
            valgrind_cmd = [
                'valgrind',
                '--tool=memcheck',
                '--leak-check=full',
                '--show-leak-kinds=all',
                '--track-origins=yes',
                '--verbose',
                f'--log-file={log_file}',
                '--xml=no'  # Use text output for easier parsing
            ]

            # Determine what to test
            if test_script:
                cmd = valgrind_cmd + ['python', test_script]
            elif component == 'python_bindings':
                cmd = valgrind_cmd + ['python', '-c', '''
import sys
sys.path.append(".")
from examples.python_bindings_demo import run_basic_demo
run_basic_demo()
''']
            elif component == 'mcts':
                # Test MCTS C++ components through Python bindings
                cmd = valgrind_cmd + ['python', '-c', '''
import sys
sys.path.append(".")
try:
    from tests.unit.test_tree_layout import test_tree_memory_layout
    test_tree_memory_layout()
except ImportError:
    print("MCTS tests not available")
''']
            else:
                return {'error': f'Unknown component: {component}'}

            logger.info(f"Running valgrind on {component}: {' '.join(cmd)}")

            # Run valgrind with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.getcwd()
            )

            # Read valgrind log
            with open(log_file, 'r') as f:
                valgrind_output = f.read()

            # Parse valgrind output
            analysis = self._parse_valgrind_output(valgrind_output)

            return {
                'component': component,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'valgrind_log': valgrind_output,
                'analysis': analysis,
                'log_file': log_file  # Keep for manual inspection
            }

        except subprocess.TimeoutExpired:
            return {
                'component': component,
                'error': 'Valgrind check timed out',
                'timeout': True
            }
        except Exception as e:
            return {
                'component': component,
                'error': str(e)
            }

    def _parse_valgrind_output(self, output: str) -> Dict[str, Any]:
        """Parse valgrind output to extract key information."""
        analysis = {
            'heap_summary': {},
            'leak_summary': {},
            'error_count': 0,
            'leaks_possible': 0,
            'leaks_definitely': 0,
            'leaks_indirectly': 0,
            'errors': []
        }

        lines = output.split('\n')
        in_heap_summary = False
        in_leak_summary = False

        for line in lines:
            line = line.strip()

            # Remove valgrind process ID prefix (e.g., "==12345==")
            if line.startswith('==') and '==' in line[2:]:
                line = line.split('==', 2)[-1].strip()

            # Error counting
            if 'ERROR SUMMARY:' in line:
                try:
                    error_count = int(line.split('ERROR SUMMARY:')[1].split()[0])
                    analysis['error_count'] = error_count
                except (IndexError, ValueError):
                    pass

            # Heap summary parsing
            if 'HEAP SUMMARY:' in line:
                in_heap_summary = True
                continue
            elif in_heap_summary and line.startswith('in use at exit:'):
                try:
                    parts = line.split()
                    bytes_used = int(parts[4].replace(',', ''))
                    blocks_used = int(parts[7].replace(',', ''))
                    analysis['heap_summary']['bytes_in_use'] = bytes_used
                    analysis['heap_summary']['blocks_in_use'] = blocks_used
                except (IndexError, ValueError):
                    pass
                in_heap_summary = False

            # Leak summary parsing
            if 'LEAK SUMMARY:' in line:
                in_leak_summary = True
                continue
            elif in_leak_summary:
                if 'definitely lost:' in line:
                    try:
                        bytes_lost = int(line.split()[2].replace(',', ''))
                        analysis['leaks_definitely'] = bytes_lost
                    except (IndexError, ValueError):
                        pass
                elif 'indirectly lost:' in line:
                    try:
                        bytes_lost = int(line.split()[2].replace(',', ''))
                        analysis['leaks_indirectly'] = bytes_lost
                    except (IndexError, ValueError):
                        pass
                elif 'possibly lost:' in line:
                    try:
                        bytes_lost = int(line.split()[2].replace(',', ''))
                        analysis['leaks_possible'] = bytes_lost
                    except (IndexError, ValueError):
                        pass
                elif line == '' or 'still reachable:' in line:
                    in_leak_summary = False

            # Collect individual errors
            if line.startswith('==') and ('Invalid' in line or 'Conditional' in line or 'Syscall' in line):
                analysis['errors'].append(line)

        return analysis


class GPUMemoryProfiler:
    """GPU memory leak detection for CUDA operations."""

    def __init__(self):
        self.cuda_available = CUDA_AVAILABLE and TORCH_AVAILABLE
        self.gpu_snapshots: List[Dict[str, float]] = []

    def start_monitoring(self) -> bool:
        """Start GPU memory monitoring."""
        if not self.cuda_available:
            logger.warning("CUDA or PyTorch not available - GPU monitoring disabled")
            return False

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache before monitoring
                logger.info("Started GPU memory monitoring")
                return True
        except Exception as e:
            logger.warning(f"Failed to start GPU monitoring: {e}")

        return False

    def take_gpu_snapshot(self) -> Dict[str, float]:
        """Take a GPU memory snapshot."""
        snapshot = {
            'timestamp': time.time(),
            'allocated_mb': 0.0,
            'cached_mb': 0.0,
            'max_allocated_mb': 0.0
        }

        if self.cuda_available and torch.cuda.is_available():
            try:
                snapshot['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
                snapshot['cached_mb'] = torch.cuda.memory_reserved() / 1024**2
                snapshot['max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
            except Exception as e:
                logger.debug(f"Failed to get GPU memory snapshot: {e}")

        self.gpu_snapshots.append(snapshot)
        return snapshot

    def analyze_gpu_leaks(self) -> List[Dict[str, Any]]:
        """Analyze GPU memory for potential leaks."""
        leaks = []

        if len(self.gpu_snapshots) < 2:
            return leaks

        # Check for consistent growth in allocated memory
        allocated_values = [s['allocated_mb'] for s in self.gpu_snapshots]
        cached_values = [s['cached_mb'] for s in self.gpu_snapshots]

        if len(allocated_values) >= 3:
            # Look for monotonic increase in allocated memory
            increasing_count = 0
            for i in range(1, len(allocated_values)):
                if allocated_values[i] > allocated_values[i-1]:
                    increasing_count += 1

            if increasing_count > len(allocated_values) * 0.7:  # >70% increases
                total_growth = allocated_values[-1] - allocated_values[0]
                duration_hours = (
                    self.gpu_snapshots[-1]['timestamp'] - self.gpu_snapshots[0]['timestamp']
                ) / 3600.0

                if total_growth > 10.0:  # >10MB growth
                    leaks.append({
                        'type': 'gpu_memory_leak',
                        'total_growth_mb': total_growth,
                        'growth_rate_mb_per_hour': total_growth / max(duration_hours, 0.001),
                        'severity': 'high' if total_growth > 100.0 else 'medium'
                    })

        return leaks

    def cleanup_gpu_memory(self) -> None:
        """Cleanup GPU memory after monitoring."""
        if self.cuda_available and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.info("Cleaned up GPU memory")
            except Exception as e:
                logger.debug(f"GPU cleanup failed: {e}")


class MemoryLeakDetector:
    """Main memory leak detection coordinator."""

    def __init__(self):
        self.python_profiler = PythonMemoryProfiler()
        self.valgrind = ValgrindIntegration()
        self.gpu_profiler = GPUMemoryProfiler()

    def run_comprehensive_check(self,
                               duration_seconds: float = 300,
                               sample_interval: float = 5.0,
                               leak_threshold_mb_per_hour: float = 10.0,
                               components: Optional[List[str]] = None,
                               workload_function: Optional[callable] = None) -> LeakDetectionResult:
        """Run comprehensive memory leak detection.

        Args:
            duration_seconds: How long to run the test
            sample_interval: Seconds between memory snapshots
            leak_threshold_mb_per_hour: Threshold for leak detection
            components: List of components to test with valgrind
            workload_function: Function to run during testing

        Returns:
            LeakDetectionResult with comprehensive analysis
        """
        logger.info(f"Starting comprehensive leak detection for {duration_seconds}s")

        components = components or ['python_bindings']

        # Start profiling
        self.python_profiler.start_profiling()
        gpu_monitoring_active = self.gpu_profiler.start_monitoring()

        # Background monitoring thread
        monitoring_active = threading.Event()
        monitoring_active.set()

        def background_monitor():
            while monitoring_active.is_set():
                self.python_profiler.take_snapshot()
                if gpu_monitoring_active:
                    self.gpu_profiler.take_gpu_snapshot()
                time.sleep(sample_interval)

        monitor_thread = threading.Thread(target=background_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        try:
            # Run workload if provided
            if workload_function:
                logger.info("Running provided workload function")
                workload_function()
            else:
                logger.info("Running default memory stress workload")
                self._run_default_workload(duration_seconds)

            # Wait for monitoring to complete
            time.sleep(max(0, duration_seconds - (time.time() - self.python_profiler.snapshots[0].timestamp if self.python_profiler.snapshots else 0)))

        finally:
            # Stop monitoring
            monitoring_active.clear()
            monitor_thread.join(timeout=10)

        # Stop profiling
        python_summary = self.python_profiler.stop_profiling()

        # Analyze results
        leak_detected, growth_rate = self.python_profiler.analyze_growth(leak_threshold_mb_per_hour)
        gpu_leaks = self.gpu_profiler.analyze_gpu_leaks()

        # Run valgrind checks
        valgrind_reports = []
        for component in components:
            if component != 'skip_valgrind':  # Allow skipping valgrind
                report = self.valgrind.run_valgrind_check(component)
                valgrind_reports.append(report)

        # Calculate total memory growth
        snapshots = self.python_profiler.snapshots
        total_growth = (snapshots[-1].python_rss_mb - snapshots[0].python_rss_mb
                       if len(snapshots) >= 2 else 0.0)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            leak_detected, growth_rate, valgrind_reports, gpu_leaks
        )

        # Cleanup
        self.gpu_profiler.cleanup_gpu_memory()

        return LeakDetectionResult(
            duration_seconds=duration_seconds,
            total_samples=len(snapshots),
            memory_growth_mb=total_growth,
            growth_rate_mb_per_hour=growth_rate,
            leak_detected=leak_detected or bool(gpu_leaks),
            leak_threshold_mb=leak_threshold_mb_per_hour,
            components_tested=components,
            valgrind_reports=valgrind_reports,
            python_profile_summary=python_summary,
            gpu_leaks=gpu_leaks,
            snapshots=snapshots,
            recommendations=recommendations
        )

    def _run_default_workload(self, duration_seconds: float) -> None:
        """Run default memory stress workload."""
        end_time = time.time() + duration_seconds
        iteration = 0

        while time.time() < end_time:
            iteration += 1

            # Simulate typical AlphaZero workload
            try:
                # Memory allocation patterns
                large_arrays = [np.random.randn(1000, 1000) for _ in range(5)]

                # Dictionary and list operations
                data_dict = {f'key_{i}': np.random.randn(100) for i in range(100)}

                # String operations
                text_data = [''.join([chr(65 + (i % 26)) for i in range(1000)]) for _ in range(50)]

                # GPU operations if available
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_tensors = [torch.randn(100, 100).cuda() for _ in range(10)]
                    result = sum(t.sum() for t in gpu_tensors)
                    del gpu_tensors

                # Cleanup some data (but not all - to test GC)
                if iteration % 10 == 0:
                    del large_arrays[:2]

            except Exception as e:
                logger.debug(f"Workload iteration {iteration} failed: {e}")

            time.sleep(0.1)  # Brief pause

        logger.info(f"Completed {iteration} workload iterations")

    def _generate_recommendations(self,
                                 leak_detected: bool,
                                 growth_rate: float,
                                 valgrind_reports: List[Dict],
                                 gpu_leaks: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        if leak_detected:
            recommendations.append(f"Python memory leak detected: {growth_rate:.2f} MB/hour growth rate")
            recommendations.append("Review object lifecycle management and ensure proper cleanup")
            recommendations.append("Consider using weak references for caches and circular references")

        for report in valgrind_reports:
            if 'analysis' in report and report['analysis'].get('leaks_definitely', 0) > 0:
                component = report.get('component', 'unknown')
                bytes_lost = report['analysis']['leaks_definitely']
                recommendations.append(f"C++ memory leak in {component}: {bytes_lost} bytes definitely lost")
                recommendations.append(f"Review {component} for missing delete/free calls")

        if gpu_leaks:
            for leak in gpu_leaks:
                recommendations.append(f"GPU memory leak detected: {leak['total_growth_mb']:.2f} MB growth")
                recommendations.append("Ensure torch.cuda.empty_cache() is called appropriately")
                recommendations.append("Review CUDA tensor lifecycle and avoid keeping references")

        if not recommendations:
            recommendations.append("No significant memory leaks detected")
            recommendations.append("Continue monitoring in production environments")

        return recommendations


def main():
    """Main entry point for memory leak detection."""
    parser = argparse.ArgumentParser(
        description='Comprehensive memory leak detection for AlphaZero engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --python --duration 600 --output report.json
  %(prog)s --valgrind --component mcts --component games
  %(prog)s --gpu --cuda-memcheck
  %(prog)s --all --threshold 5.0
        """
    )

    parser.add_argument('--python', action='store_true',
                       help='Run Python memory leak detection')
    parser.add_argument('--valgrind', action='store_true',
                       help='Run valgrind C++ leak detection')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU memory leak detection')
    parser.add_argument('--all', action='store_true',
                       help='Run all leak detection methods')

    parser.add_argument('--duration', type=float, default=300,
                       help='Duration in seconds (default: 300)')
    parser.add_argument('--interval', type=float, default=5.0,
                       help='Sampling interval in seconds (default: 5.0)')
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Leak threshold in MB/hour (default: 10.0)')

    parser.add_argument('--component', action='append',
                       choices=['mcts', 'games', 'python_bindings'],
                       help='Components to test with valgrind (can be repeated)')
    parser.add_argument('--output', type=str,
                       help='Output file for detailed results (JSON format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine what to run
    if not any([args.python, args.valgrind, args.gpu, args.all]):
        args.python = True  # Default to Python detection

    if args.all:
        args.python = args.valgrind = args.gpu = True

    # Set up components to test
    components = args.component or ['python_bindings']
    if not args.valgrind:
        components = ['skip_valgrind']  # Skip valgrind if not requested

    # Run leak detection
    detector = MemoryLeakDetector()

    try:
        result = detector.run_comprehensive_check(
            duration_seconds=args.duration,
            sample_interval=args.interval,
            leak_threshold_mb_per_hour=args.threshold,
            components=components
        )

        # Print summary
        print("\n" + "="*60)
        print("MEMORY LEAK DETECTION SUMMARY")
        print("="*60)
        print(f"Duration: {result.duration_seconds:.1f} seconds")
        print(f"Samples collected: {result.total_samples}")
        print(f"Memory growth: {result.memory_growth_mb:.2f} MB")
        print(f"Growth rate: {result.growth_rate_mb_per_hour:.2f} MB/hour")
        print(f"Leak detected: {'YES' if result.leak_detected else 'NO'}")
        print(f"Components tested: {', '.join(result.components_tested)}")

        if result.gpu_leaks:
            print(f"GPU leaks detected: {len(result.gpu_leaks)}")

        print("\nRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

        # Save detailed results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.output}")

        # Exit with appropriate code
        sys.exit(1 if result.leak_detected else 0)

    except KeyboardInterrupt:
        print("\nMemory leak detection interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Memory leak detection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()