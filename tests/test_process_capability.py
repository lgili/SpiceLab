"""Tests for process capability metrics on MonteCarloResult."""

from __future__ import annotations

import pytest
from spicelab.analysis.stats import compute_stats


class TestStatisticsCpk:
    """Tests for Statistics.cpk() method."""

    def test_cpk_centered_process(self):
        """Cpk should be high for centered process with low std."""
        # Mean = 5.0, std = 0.1, spec limits = 4.5 to 5.5
        # Cpk = (5.5 - 5.0) / (3 * 0.1) = 0.5 / 0.3 = 1.667
        # This is a well-centered process
        values = [5.0 + 0.1 * (i - 50) / 50 for i in range(101)]  # Range ~4.9 to ~5.1
        stats = compute_stats(values)

        # With a tighter distribution, Cpk should be > 1
        cpk = stats.cpk(4.5, 5.5)
        assert cpk > 1.0

    def test_cpk_off_center_process(self):
        """Cpk should be lower for off-center process."""
        # Process centered at 4.8 instead of 5.0
        # Even with same std, Cpk is limited by distance to nearest spec
        values = [4.8 + 0.05 * (i - 50) / 50 for i in range(101)]
        stats = compute_stats(values)

        lsl, usl = 4.5, 5.5
        cpk = stats.cpk(lsl, usl)

        # Cpk is min of CPU and CPL
        # CPU = (5.5 - 4.8) / (3 * std) ≈ high
        # CPL = (4.8 - 4.5) / (3 * std) ≈ lower (limiting)
        assert cpk > 0

    def test_cpk_zero_std(self):
        """Cpk should be inf when std is 0 and mean is within spec."""
        values = [5.0]  # Single value, std = 0
        stats = compute_stats(values)

        cpk = stats.cpk(4.5, 5.5)
        assert cpk == float("inf")

    def test_cpk_out_of_spec(self):
        """Cpk should be 0 when mean is outside spec limits."""
        values = [6.0]  # Single value outside spec
        stats = compute_stats(values)

        cpk = stats.cpk(4.5, 5.5)
        assert cpk == 0.0

    def test_cpk_known_value(self):
        """Test Cpk with known statistical values."""
        # Create a distribution with known mean=5.0 and std=0.1
        # USL=5.4, LSL=4.6, so range is ±0.4 from mean
        # CPU = (5.4 - 5.0) / (3 * 0.1) = 0.4 / 0.3 = 1.33
        # CPL = (5.0 - 4.6) / (3 * 0.1) = 0.4 / 0.3 = 1.33
        # Cpk = min(1.33, 1.33) = 1.33

        # Generate values with approximate std of 0.1
        import random

        random.seed(42)
        values = [random.gauss(5.0, 0.1) for _ in range(10000)]
        stats = compute_stats(values)

        cpk = stats.cpk(4.6, 5.4)

        # With 10k samples, Cpk should be close to 1.33
        assert 1.2 < cpk < 1.5


class TestStatisticsYield:
    """Tests for Statistics.yield_estimate() method."""

    def test_yield_centered_tight_process(self):
        """Yield should be high for centered process with tight std."""
        import random

        random.seed(42)
        values = [random.gauss(5.0, 0.05) for _ in range(1000)]
        stats = compute_stats(values)

        # Wide spec limits, tight process -> high yield
        yield_est = stats.yield_estimate(4.5, 5.5)
        assert yield_est > 0.99  # Should be > 99%

    def test_yield_wide_process(self):
        """Yield should be lower for process with wide std."""
        import random

        random.seed(42)
        values = [random.gauss(5.0, 0.2) for _ in range(1000)]
        stats = compute_stats(values)

        # Spec limits = mean ± 2.5*sigma -> ~98.76% yield
        yield_est = stats.yield_estimate(4.5, 5.5)
        assert 0.85 < yield_est < 0.995  # Should be reasonable but not perfect

    def test_yield_zero_std(self):
        """Yield should be 100% when std is 0 and mean is within spec."""
        values = [5.0]
        stats = compute_stats(values)

        yield_est = stats.yield_estimate(4.5, 5.5)
        assert yield_est == 1.0


class TestComputeStats:
    """Tests for compute_stats function."""

    def test_basic_stats(self):
        """compute_stats should calculate correct statistics."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_stats(values)

        assert stats.n == 5
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.median == 3.0

    def test_single_value(self):
        """compute_stats should handle single value."""
        values = [42.0]
        stats = compute_stats(values)

        assert stats.n == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_empty_raises(self):
        """compute_stats should raise on empty input."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_stats([])

    def test_sigma3_bounds(self):
        """sigma3_low and sigma3_high should be mean ± 3*std."""
        import random

        random.seed(42)
        values = [random.gauss(100, 10) for _ in range(1000)]
        stats = compute_stats(values)

        assert abs(stats.sigma3_low - (stats.mean - 3 * stats.std)) < 1e-10
        assert abs(stats.sigma3_high - (stats.mean + 3 * stats.std)) < 1e-10


class TestProcessCapabilityIntegration:
    """Integration tests for process capability workflow."""

    def test_sigma_level_conversion(self):
        """sigma_level should be Cpk * 3."""
        from spicelab.analysis.stats import sigma_level

        assert sigma_level(1.0) == 3.0
        assert sigma_level(1.33) == pytest.approx(3.99, rel=0.01)
        assert sigma_level(2.0) == 6.0

    def test_cpk_to_dpmo_relationship(self):
        """Cpk values should correspond to expected DPMO."""
        # Cpk 1.0 -> ~2700 DPMO (3 sigma, 99.73%)
        # Cpk 1.33 -> ~63 DPMO (4 sigma, 99.99%)
        # Cpk 2.0 -> ~0.002 DPMO (6 sigma, 99.9999998%)

        import random

        random.seed(42)
        values = [random.gauss(0, 1) for _ in range(10000)]
        stats = compute_stats(values)

        # 3-sigma limits: -3 to 3
        cpk_3sigma = stats.cpk(-3, 3)
        yield_3sigma = stats.yield_estimate(-3, 3)

        # Should be close to Cpk=1.0 and yield=99.73%
        assert 0.9 < cpk_3sigma < 1.1
        assert 0.99 < yield_3sigma < 0.999
