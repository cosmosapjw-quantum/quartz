"""
Bottleneck Analyzer
===================

Automated detection and analysis of performance bottlenecks.

Detects:
- State cloning waste (review.txt lines 37-54)
- Thread contention (review.txt lines 71-136)
- Feature extraction bottlenecks (review.txt lines 22-34)
- Python coordination overhead (review.txt lines 59-62)
- Thread affinity issues (review.txt lines 244-250)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class Bottleneck:
    """Identified performance bottleneck"""
    name: str
    category: str  # "cpu", "gpu", "sync", "memory", "coordination"
    severity: str  # "critical", "high", "medium", "low"
    percentage_of_total: float
    impact_description: str
    recommendations: List[str]
    evidence: Dict[str, Any]

class BottleneckAnalyzer:
    """
    Automated bottleneck detection and analysis.

    Uses heuristics based on review.txt findings to identify:
    1. State cloning waste (>15% of time)
    2. Thread idle time (>40% idle)
    3. Feature extraction slowness (>5ms per batch)
    4. Python overhead (>30% of time)
    5. CAS retry storms (>30% retry rate)
    """

    def __init__(self):
        # Thresholds for bottleneck detection (tuned from review.txt)
        self.thresholds = {
            # State cloning (review.txt identifies 2-3× per sim as critical)
            'state_cloning_critical_pct': 15.0,
            'state_cloning_high_pct': 10.0,

            # Thread idle time (review.txt shows 60% idle)
            'thread_idle_critical_pct': 40.0,
            'thread_idle_high_pct': 30.0,

            # Feature extraction (review.txt shows 7.5ms target <1ms)
            'feature_extraction_critical_ms': 5.0,
            'feature_extraction_high_ms': 3.0,

            # Python overhead (review.txt shows 67% in Python)
            'python_overhead_critical_pct': 40.0,
            'python_overhead_high_pct': 25.0,

            # CAS retry rate
            'cas_retry_critical_rate': 0.3,
            'cas_retry_high_rate': 0.2,
        }

    def analyze(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze metrics and detect bottlenecks.

        Args:
            metrics: Unified profiling metrics

        Returns:
            Bottleneck analysis results with:
                - bottlenecks: List of detected bottlenecks
                - severity_counts: Counts by severity
                - total_bottlenecks: Total number of bottlenecks
                - critical_count: Number of critical bottlenecks
        """
        bottlenecks = []

        # Get session duration
        session_duration = metrics.get('session_duration', 1.0)

        # Analyze Python metrics
        if 'python_metrics' in metrics or 'thread_metrics' in metrics:
            bottlenecks.extend(self._analyze_python_metrics(metrics, session_duration))

        # Analyze C++ metrics (if available)
        if 'cpp_metrics' in metrics:
            bottlenecks.extend(self._analyze_cpp_metrics(metrics['cpp_metrics'], session_duration))

        # Sort by severity and impact
        bottlenecks.sort(key=lambda b: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[b.severity],
            -b.percentage_of_total
        ))

        # Generate summary
        severity_counts = {
            'critical': sum(1 for b in bottlenecks if b.severity == 'critical'),
            'high': sum(1 for b in bottlenecks if b.severity == 'high'),
            'medium': sum(1 for b in bottlenecks if b.severity == 'medium'),
            'low': sum(1 for b in bottlenecks if b.severity == 'low'),
        }

        return {
            'bottlenecks': [self._bottleneck_to_dict(b) for b in bottlenecks],
            'severity_counts': severity_counts,
            'total_bottlenecks': len(bottlenecks),
            'critical_count': severity_counts['critical'],
            'analysis_summary': self._generate_summary(bottlenecks, severity_counts),
        }

    def _analyze_python_metrics(self, metrics: Dict[str, Any], session_duration: float) -> List[Bottleneck]:
        """Analyze Python metrics for bottlenecks"""
        bottlenecks = []

        # Get Python summary
        python_summary = metrics.get('thread_metrics', {})
        if not python_summary and 'python_metrics' in metrics:
            python_summary = metrics['python_metrics'].get('thread_metrics', {})

        # Check state cloning
        if 'state_cloning' in python_summary:
            bottleneck = self._check_state_cloning(python_summary['state_cloning'], session_duration)
            if bottleneck:
                bottlenecks.append(bottleneck)

        # Check feature extraction
        if 'feature_extraction' in python_summary:
            bottleneck = self._check_feature_extraction(python_summary['feature_extraction'])
            if bottleneck:
                bottlenecks.append(bottleneck)

        # Check Python coordination overhead
        if 'function_stats' in python_summary:
            bottleneck = self._check_python_overhead(python_summary['function_stats'], session_duration)
            if bottleneck:
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _check_state_cloning(self, clone_stats: Dict[str, Any], session_duration: float) -> Optional[Bottleneck]:
        """
        Check for state cloning waste bottleneck.

        Evidence from review.txt (lines 37-54):
        - 2-3× clones per simulation
        - Clones at: select_leaf, submit_request, run_continuous
        """
        total_clone_time = clone_stats.get('total_time', 0.0)
        clone_count = clone_stats.get('total_clones', 0)

        if total_clone_time == 0.0:
            return None

        clone_pct = 100.0 * total_clone_time / session_duration

        if clone_pct > self.thresholds['state_cloning_high_pct']:
            severity = "critical" if clone_pct > self.thresholds['state_cloning_critical_pct'] else "high"

            return Bottleneck(
                name="State Cloning Waste",
                category="memory",
                severity=severity,
                percentage_of_total=clone_pct,
                impact_description=f"State cloning consumes {clone_pct:.1f}% of total execution time ({clone_count} total clones)",
                recommendations=[
                    "🔴 CRITICAL: Implement state pooling with per-thread caches (review.txt lines 164-176)",
                    "Replace clone() with copy_from() where possible",
                    "Pass states by reference to AsyncInferenceQueue (review.txt lines 177-188)",
                    "Precompute legal moves before queuing to avoid state retention (review.txt lines 189-200)",
                    "Target: Reduce to 1× clone per simulation (current: 2-3×)",
                ],
                evidence={
                    'total_time_s': total_clone_time,
                    'percentage': clone_pct,
                    'clone_count': clone_count,
                    'avg_time_ms': clone_stats.get('avg_time', 0.0) * 1000,
                    'review_txt_reference': "lines 37-54",
                }
            )
        return None

    def _check_feature_extraction(self, extraction_stats: Dict[str, Any]) -> Optional[Bottleneck]:
        """
        Check for feature extraction bottleneck.

        Evidence from review.txt (lines 22-34):
        - 7.5ms per batch of 64 states (should be <1ms)
        - OpenMP not parallelizing
        """
        avg_time_ms = extraction_stats.get('avg_time', 0.0) * 1000
        total_extractions = extraction_stats.get('total_extractions', 0)

        if avg_time_ms == 0.0:
            return None

        if avg_time_ms > self.thresholds['feature_extraction_high_ms']:
            severity = "critical" if avg_time_ms > self.thresholds['feature_extraction_critical_ms'] else "high"

            recommendations = [
                f"⚠️  Feature extraction taking {avg_time_ms:.1f}ms per call (target: <1ms)",
                "🔴 CRITICAL: Verify OpenMP is enabled with -fopenmp flag",
                "Check that #pragma omp parallel for is active at dlpack_bridge.cpp:431-434",
                "Verify OpenMP thread count matches CPU cores",
                "Use vectorized feature extraction if available",
                "Target: <1ms per batch of 64 states",
            ]

            # Check if OpenMP is likely disabled
            if avg_time_ms > 5.0:
                recommendations.insert(0, "❌ LIKELY CAUSE: OpenMP NOT parallelizing (review.txt lines 22-34)")

            return Bottleneck(
                name="Feature Extraction Bottleneck",
                category="cpu",
                severity=severity,
                percentage_of_total=0.0,  # Would need total time to calculate
                impact_description=f"Feature extraction takes {avg_time_ms:.1f}ms per call ({total_extractions} total, target: <1ms)",
                recommendations=recommendations,
                evidence={
                    'avg_time_ms': avg_time_ms,
                    'total_extractions': total_extractions,
                    'review_txt_reference': "lines 22-34",
                }
            )
        return None

    def _check_python_overhead(self, function_stats: Dict[str, Any], session_duration: float) -> Optional[Bottleneck]:
        """
        Check for Python coordination overhead.

        Evidence from review.txt (lines 59-62):
        - 67% of runtime in Python/GIL overhead
        """
        # Calculate total Python time
        total_python_time = sum(
            stats.get('total_time', 0.0)
            for stats in function_stats.values()
        )

        python_pct = 100.0 * total_python_time / session_duration

        if python_pct > self.thresholds['python_overhead_high_pct']:
            severity = "critical" if python_pct > self.thresholds['python_overhead_critical_pct'] else "high"

            # Identify top offenders
            top_functions = sorted(
                function_stats.items(),
                key=lambda x: x[1].get('total_time', 0.0),
                reverse=True
            )[:5]

            top_functions_str = "\n".join([
                f"    - {name}: {stats.get('total_time', 0)*1000:.2f}ms"
                for name, stats in top_functions
            ])

            return Bottleneck(
                name="Python Coordination Overhead",
                category="coordination",
                severity=severity,
                percentage_of_total=python_pct,
                impact_description=f"Python coordination consumes {python_pct:.1f}% of total time\n  Top offenders:\n{top_functions_str}",
                recommendations=[
                    "Move hot loops to C++ (review.txt lines 258-307)",
                    "Use DLPack zero-copy for tensor creation (eliminate numpy overhead)",
                    "Batch Python operations to reduce GIL acquisitions",
                    "Return numpy arrays directly instead of Python lists (review.txt lines 279-280)",
                    "Target: <20% Python overhead (review.txt shows 67%)",
                ],
                evidence={
                    'python_time_s': total_python_time,
                    'percentage': python_pct,
                    'top_functions': [
                        {'name': name, 'time_ms': stats.get('total_time', 0)*1000}
                        for name, stats in top_functions
                    ],
                    'review_txt_reference': "lines 59-62",
                }
            )
        return None

    def _analyze_cpp_metrics(self, cpp_metrics: Dict[str, Any], session_duration: float) -> List[Bottleneck]:
        """Analyze C++ metrics for bottlenecks"""
        bottlenecks = []

        # Check for thread contention
        if 'ThreadIdleTotal' in cpp_metrics:
            bottleneck = self._check_thread_contention(cpp_metrics, session_duration)
            if bottleneck:
                bottlenecks.append(bottleneck)

        # Check for CAS retry storms
        if 'CAS_RetryCount' in cpp_metrics:
            bottleneck = self._check_cas_retries(cpp_metrics)
            if bottleneck:
                bottlenecks.append(bottleneck)

        return bottlenecks

    def _check_thread_contention(self, metrics: Dict[str, Any], session_duration: float) -> Optional[Bottleneck]:
        """
        Check for thread contention bottleneck.

        Evidence from review.txt (lines 71-136):
        - 60% idle time during 2.5s search
        - Global mutex contention in allocate_nodes()
        - Spin-wait in ContinuousSimulationRunner
        """
        idle_time = metrics.get('ThreadIdleTotal', {}).get('total_ns', 0) / 1e9
        idle_pct = 100.0 * idle_time / session_duration

        if idle_pct > self.thresholds['thread_idle_high_pct']:
            severity = "critical" if idle_pct > self.thresholds['thread_idle_critical_pct'] else "high"

            return Bottleneck(
                name="Thread Contention and Idle Time",
                category="sync",
                severity=severity,
                percentage_of_total=idle_pct,
                impact_description=f"Threads are idle {idle_pct:.1f}% of the time (review.txt shows 60%)",
                recommendations=[
                    "🔴 Replace spin-wait with condition variable (review.txt lines 212-224)",
                    "Reduce allocation_mutex_ contention by batching (review.txt lines 225-236)",
                    "Optimize thread affinity to avoid cross-CCD (review.txt lines 244-250)",
                    "Consider reducing virtual loss magnitude if CAS rate is high",
                    "Target: <20% idle time (current: 60%)",
                ],
                evidence={
                    'idle_time_s': idle_time,
                    'idle_percentage': idle_pct,
                    'review_txt_reference': "lines 71-136",
                }
            )
        return None

    def _check_cas_retries(self, metrics: Dict[str, Any]) -> Optional[Bottleneck]:
        """Check for excessive CAS retries"""
        retry_count = metrics.get('CAS_RetryCount', {}).get('value', 0)
        success_count = metrics.get('CAS_SuccessRate', {}).get('value', 0) or retry_count + 1
        total_attempts = retry_count + success_count
        retry_rate = retry_count / total_attempts if total_attempts > 0 else 0.0

        if retry_rate > self.thresholds['cas_retry_high_rate']:
            severity = "critical" if retry_rate > self.thresholds['cas_retry_critical_rate'] else "medium"

            return Bottleneck(
                name="Excessive CAS Retries",
                category="sync",
                severity=severity,
                percentage_of_total=0.0,
                impact_description=f"CAS operations have {retry_rate*100:.1f}% retry rate",
                recommendations=[
                    "Reduce thread count to decrease contention",
                    "Use WU-UCT virtual loss (visit-only) instead of classic",
                    "Batch atomic updates where possible",
                    "Consider lock-free data structures",
                ],
                evidence={
                    'retry_count': retry_count,
                    'retry_rate': retry_rate,
                }
            )
        return None

    def _bottleneck_to_dict(self, bottleneck: Bottleneck) -> Dict[str, Any]:
        """Convert Bottleneck to dictionary"""
        return {
            'name': bottleneck.name,
            'category': bottleneck.category,
            'severity': bottleneck.severity,
            'percentage': bottleneck.percentage_of_total,
            'description': bottleneck.impact_description,
            'recommendations': bottleneck.recommendations,
            'evidence': bottleneck.evidence,
        }

    def _generate_summary(self, bottlenecks: List[Bottleneck], severity_counts: Dict[str, int]) -> str:
        """Generate textual summary of analysis"""
        if not bottlenecks:
            return "✅ No critical bottlenecks detected!"

        lines = []
        lines.append(f"Found {len(bottlenecks)} bottleneck(s):")
        lines.append(f"  - Critical: {severity_counts['critical']}")
        lines.append(f"  - High: {severity_counts['high']}")
        lines.append(f"  - Medium: {severity_counts['medium']}")
        lines.append(f"  - Low: {severity_counts['low']}")

        if severity_counts['critical'] > 0:
            lines.append("\n🔴 CRITICAL ISSUES:")
            for bottleneck in [b for b in bottlenecks if b.severity == 'critical']:
                lines.append(f"  - {bottleneck.name}: {bottleneck.impact_description}")

        return "\n".join(lines)

def print_bottleneck_analysis(analysis: Dict[str, Any]):
    """Print human-readable bottleneck analysis"""
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)

    print(f"\n{analysis.get('analysis_summary', 'No summary available')}")

    bottlenecks = analysis.get('bottlenecks', [])
    if not bottlenecks:
        print("\n✅ No bottlenecks detected!")
        return

    for bottleneck in bottlenecks[:10]:  # Top 10
        severity_marker = {
            'critical': '🔴',
            'high': '🟡',
            'medium': '🟠',
            'low': '⚪',
        }.get(bottleneck['severity'], '❓')

        print(f"\n{severity_marker} {bottleneck['name']} [{bottleneck['category'].upper()}]")
        print(f"  Severity: {bottleneck['severity'].upper()}")
        if bottleneck['percentage'] > 0:
            print(f"  Impact: {bottleneck['percentage']:.1f}% of total time")
        print(f"\n  {bottleneck['description']}")
        print("\n  Recommendations:")
        for i, rec in enumerate(bottleneck['recommendations'][:5], 1):
            print(f"    {i}. {rec}")

        # Show evidence
        if bottleneck.get('evidence', {}).get('review_txt_reference'):
            print(f"\n  📖 Reference: review.txt {bottleneck['evidence']['review_txt_reference']}")

    print("\n" + "="*80)
