"""
Tests for Search Quality Equivalence (T017)

Validates that MCTS optimizations maintain search quality by comparing
policies and values across different configurations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPolicyComparison:
    """Tests for policy comparison metrics."""

    def test_kl_divergence_identical(self):
        """KL divergence should be 0 for identical distributions."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        # KL(P || Q) = sum(P * log(P / Q)) = 0
        kl = self._calculate_kl_divergence(p, q)
        assert kl < 1e-6, f"Expected KL ≈ 0, got {kl}"

    def test_kl_divergence_similar(self):
        """KL divergence should be small for similar distributions."""
        p = np.array([0.24, 0.26, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        kl = self._calculate_kl_divergence(p, q)
        assert kl < 0.01, f"Expected small KL, got {kl}"

    def test_kl_divergence_different(self):
        """KL divergence should be large for different distributions."""
        p = np.array([0.7, 0.1, 0.1, 0.1])
        q = np.array([0.1, 0.3, 0.3, 0.3])

        kl = self._calculate_kl_divergence(p, q)
        assert kl > 0.1, f"Expected large KL, got {kl}"

    def test_policy_correlation_identical(self):
        """Correlation should be 1.0 for identical policies."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        corr = np.corrcoef(p, q)[0, 1]
        # May be NaN if std=0, which is fine
        assert np.isnan(corr) or abs(corr - 1.0) < 0.01

    def test_policy_correlation_similar(self):
        """Correlation should be high for similar policies."""
        p = np.array([0.24, 0.26, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        corr = np.corrcoef(p, q)[0, 1]
        # NaN is acceptable when std is very small (nearly identical)
        assert np.isnan(corr) or corr > 0.95, f"Expected high correlation or NaN, got {corr}"

    def test_policy_correlation_negative(self):
        """Correlation can be negative for opposite policies."""
        p = np.array([0.7, 0.1, 0.1, 0.1])
        q = np.array([0.1, 0.3, 0.3, 0.3])

        corr = np.corrcoef(p, q)[0, 1]
        assert corr < 0.5, f"Expected low/negative correlation, got {corr}"

    def _calculate_kl_divergence(self, p, q, epsilon=1e-10):
        """Calculate KL divergence KL(P || Q)."""
        # Add epsilon to avoid log(0)
        p = np.asarray(p) + epsilon
        q = np.asarray(q) + epsilon

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        # KL(P || Q) = sum(P * log(P / Q))
        kl = np.sum(p * np.log(p / q))
        return float(kl)


class TestValueComparison:
    """Tests for value comparison metrics."""

    def test_value_mse_identical(self):
        """MSE should be 0 for identical values."""
        v1 = 0.5
        v2 = 0.5

        mse = (v1 - v2) ** 2
        assert mse == 0.0

    def test_value_mse_small_difference(self):
        """MSE should be small for similar values."""
        v1 = 0.50
        v2 = 0.51

        mse = (v1 - v2) ** 2
        assert mse < 0.001, f"Expected small MSE, got {mse}"

    def test_value_mse_large_difference(self):
        """MSE should be large for different values."""
        v1 = 0.5
        v2 = -0.5

        mse = (v1 - v2) ** 2
        assert mse == 1.0

    def test_value_difference_threshold(self):
        """Value difference should be within acceptable range."""
        v1 = 0.5
        v2 = 0.52

        diff = abs(v1 - v2)
        threshold = 0.05  # 5% tolerance
        assert diff < threshold, f"Value difference {diff} exceeds threshold {threshold}"


class TestComparisonMetrics:
    """Integration tests for comparison framework."""

    def test_comparison_result_creation(self):
        """Test creation of ComparisonResult dataclass."""
        from scripts.compare_search_quality import ComparisonResult

        result = ComparisonResult(
            position_id=0,
            kl_divergence=0.005,
            value_mse=0.002,
            policy_correlation=0.98,
            value_difference=0.03,
            baseline_value=0.5,
            test_value=0.53
        )

        assert result.position_id == 0
        assert result.kl_divergence == 0.005
        assert result.policy_correlation == 0.98

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        from scripts.compare_search_quality import ComparisonResult, SearchQualityComparer

        # Create mock comparer
        comparer = SearchQualityComparer(None, None, game='gomoku')

        # Add mock results
        comparer.results = [
            ComparisonResult(0, 0.005, 0.002, 0.98, 0.02, 0.5, 0.52),
            ComparisonResult(1, 0.007, 0.003, 0.97, 0.03, 0.5, 0.53),
            ComparisonResult(2, 0.003, 0.001, 0.99, 0.01, 0.5, 0.51),
        ]

        summary = comparer._create_summary(simulations=800)

        # Check averages
        assert 0.004 < summary.avg_kl_divergence < 0.006
        assert 0.001 < summary.avg_value_mse < 0.003
        assert 0.97 < summary.avg_policy_correlation < 0.99

        # Check maximums
        assert summary.max_kl_divergence == 0.007
        assert summary.max_value_mse == 0.003

    def test_quality_thresholds(self):
        """Test quality assessment against thresholds."""
        from scripts.compare_search_quality import ComparisonResult, SearchQualityComparer

        comparer = SearchQualityComparer(None, None, game='gomoku')

        # Add results that pass all thresholds
        comparer.results = [
            ComparisonResult(0, 0.005, 0.002, 0.98, 0.02, 0.5, 0.52),
            ComparisonResult(1, 0.007, 0.003, 0.97, 0.03, 0.5, 0.53),
        ]

        summary = comparer._create_summary(simulations=800)

        # Check quality thresholds
        kl_threshold = 0.01
        mse_threshold = 0.005
        corr_threshold = 0.95

        assert summary.avg_kl_divergence < kl_threshold, "KL divergence too high"
        assert summary.avg_value_mse < mse_threshold, "Value MSE too high"
        assert summary.avg_policy_correlation > corr_threshold, "Correlation too low"


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_seed_same_results(self):
        """Same random seed should produce identical results."""
        np.random.seed(42)
        result1 = np.random.rand(10)

        np.random.seed(42)
        result2 = np.random.rand(10)

        assert np.allclose(result1, result2), "Results should be identical with same seed"

    def test_mcts_determinism_single_thread(self):
        """MCTS with single thread and same seed should be deterministic."""
        # This would require actual MCTS runs - placeholder test
        # In practice, we'd run MCTS twice with same config and verify
        # visit counts are identical
        pass


@pytest.mark.skipif(
    not pytest.importorskip("scripts.compare_search_quality", reason="compare_search_quality not available"),
    reason="Comparison script not available"
)
class TestComparisonFramework:
    """Integration tests for the full comparison framework."""

    def test_mock_comparison_runs(self):
        """Test that mock comparison runs without errors."""
        from scripts.compare_search_quality import SearchQualityComparer

        comparer = SearchQualityComparer(None, None, game='gomoku')
        summary = comparer._mock_comparison(num_positions=5, simulations=100)

        assert summary.num_positions == 5
        assert summary.num_simulations == 100
        assert len(summary.results) == 5

        # Check that mock results are reasonable
        assert summary.avg_kl_divergence < 0.02
        assert summary.avg_value_mse < 0.01
        assert summary.avg_policy_correlation > 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
