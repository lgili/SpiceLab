"""Tests for the statistical analysis helpers module."""

import math

import pytest
from spicelab.analysis.stats import (
    compute_stats,
    create_metric_extractor,
)


class TestComputeStats:
    """Tests for compute_stats function."""

    def test_basic_stats(self):
        """Basic statistical computations."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_stats(values)

        assert stats.n == 5
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_std_calculation(self):
        """Standard deviation uses n-1 denominator."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = compute_stats(values)

        # Mean = 5.0, sample std = 2.138...
        assert abs(stats.mean - 5.0) < 1e-10
        # Sample variance = sum((x-mean)^2) / (n-1) = 32/7 = 4.571...
        expected_std = math.sqrt(32 / 7)
        assert abs(stats.std - expected_std) < 1e-10

    def test_single_value(self):
        """Single value has zero std."""
        stats = compute_stats([42.0])
        assert stats.n == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_sigma3_bounds(self):
        """3-sigma bounds are computed correctly."""
        values = [100.0] * 100  # All same value
        stats = compute_stats(values)

        assert stats.sigma3_low == stats.mean  # std is 0
        assert stats.sigma3_high == stats.mean

    def test_percentiles(self):
        """Percentiles are computed correctly."""
        # Create 100 values from 1 to 100
        values = list(range(1, 101))
        stats = compute_stats([float(v) for v in values])

        assert stats.p1 == pytest.approx(1.99, rel=0.01)
        assert stats.p5 == pytest.approx(5.95, rel=0.01)
        assert stats.median == pytest.approx(50.5, rel=0.01)
        assert stats.p95 == pytest.approx(95.05, rel=0.01)
        assert stats.p99 == pytest.approx(99.01, rel=0.01)

    def test_empty_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="values must not be empty"):
            compute_stats([])


class TestStatisticsMethods:
    """Tests for Statistics dataclass methods."""

    def test_cpk_centered(self):
        """Cpk for centered process."""
        # Create normal-ish distribution centered at 100
        values = [100.0 + (i - 50) * 0.1 for i in range(100)]
        stats = compute_stats(values)

        # With LSL=95, USL=105 and process centered at 100
        cpk = stats.cpk(95.0, 105.0)
        # For well-centered process, Cpk should be reasonable
        assert cpk > 0

    def test_cpk_shifted(self):
        """Cpk for off-center process."""
        # Process shifted toward upper limit
        values = [104.0 + (i - 50) * 0.1 for i in range(100)]
        stats = compute_stats(values)

        cpk = stats.cpk(95.0, 105.0)
        # Cpk should be limited by proximity to USL
        assert cpk < stats.cpk(90.0, 110.0)  # Wider limits = higher Cpk

    def test_cpk_zero_std(self):
        """Cpk with zero std."""
        stats = compute_stats([100.0] * 10)
        assert stats.cpk(95.0, 105.0) == float("inf")
        # Out of spec
        assert stats.cpk(101.0, 105.0) == 0.0

    def test_yield_estimate(self):
        """Yield estimate for normal distribution."""
        # Generate somewhat normal distribution
        import random

        random.seed(42)
        values = [random.gauss(100, 1) for _ in range(1000)]
        stats = compute_stats(values)

        # 3-sigma should be ~99.7%
        yield_3s = stats.yield_estimate(stats.mean - 3 * stats.std, stats.mean + 3 * stats.std)
        assert yield_3s > 0.99

        # 1-sigma should be ~68%
        yield_1s = stats.yield_estimate(stats.mean - stats.std, stats.mean + stats.std)
        assert 0.65 < yield_1s < 0.72

    def test_repr(self):
        """String representation is reasonable."""
        stats = compute_stats([1.0, 2.0, 3.0])
        repr_str = repr(stats)
        assert "Statistics" in repr_str
        assert "n=3" in repr_str
        assert "mean=" in repr_str


class TestCreateMetricExtractor:
    """Tests for create_metric_extractor factory."""

    def test_creates_callable(self):
        """Factory returns a callable."""
        extractor = create_metric_extractor("V(vout)")
        assert callable(extractor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
