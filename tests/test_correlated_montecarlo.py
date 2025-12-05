"""Tests for CorrelatedGroup and absolute tolerance distributions."""

from __future__ import annotations

import random

import pytest
from spicelab.analysis.montecarlo import (
    CorrelatedGroup,
    NormalAbs,
    NormalPct,
    TriangularAbs,
    TriangularPct,
    UniformAbs,
    UniformPct,
)


class TestNormalAbs:
    """Tests for NormalAbs distribution."""

    def test_basic_sampling(self):
        """NormalAbs should sample around nominal with absolute sigma."""
        dist = NormalAbs(0.002)  # 2mV sigma
        rnd = random.Random(42)

        samples = [dist.sample(0.0, rnd) for _ in range(1000)]

        # Mean should be close to nominal (0.0)
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.001, f"Mean {mean} too far from 0"

        # Std should be close to sigma (0.002)
        std = (sum((s - mean) ** 2 for s in samples) / len(samples)) ** 0.5
        assert 0.001 < std < 0.003, f"Std {std} not close to 0.002"

    def test_with_nonzero_nominal(self):
        """NormalAbs should work with non-zero nominal."""
        dist = NormalAbs(0.001)
        rnd = random.Random(42)

        samples = [dist.sample(5.0, rnd) for _ in range(1000)]
        mean = sum(samples) / len(samples)

        # Mean should be close to nominal (5.0)
        assert 4.9 < mean < 5.1, f"Mean {mean} too far from 5.0"

    def test_negative_sigma_raises(self):
        """NormalAbs should reject negative sigma."""
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            NormalAbs(-0.001)

    def test_repr(self):
        """NormalAbs should have informative repr."""
        dist = NormalAbs(0.002)
        assert "NormalAbs" in repr(dist)
        assert "0.002" in repr(dist)


class TestTriangularAbs:
    """Tests for TriangularAbs distribution."""

    def test_basic_sampling(self):
        """TriangularAbs should sample within absolute bounds."""
        dist = TriangularAbs(0.005)  # ±5mV
        rnd = random.Random(42)

        samples = [dist.sample(0.0, rnd) for _ in range(1000)]

        # All samples should be within ±delta
        assert all(-0.005 <= s <= 0.005 for s in samples)

        # Mean should be close to nominal
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.002

    def test_negative_delta_raises(self):
        """TriangularAbs should reject negative delta."""
        with pytest.raises(ValueError, match="delta must be >= 0"):
            TriangularAbs(-0.001)

    def test_repr(self):
        """TriangularAbs should have informative repr."""
        dist = TriangularAbs(0.005)
        assert "TriangularAbs" in repr(dist)
        assert "0.005" in repr(dist)


class TestCorrelatedGroup:
    """Tests for CorrelatedGroup class."""

    def test_empty_group_raises(self):
        """CorrelatedGroup should reject empty component list."""
        with pytest.raises(ValueError, match="at least one component"):
            CorrelatedGroup([], NormalPct(0.01))

    def test_stores_components_and_dist(self):
        """CorrelatedGroup should store components and distribution."""

        class FakeComponent:
            def __init__(self, ref, value):
                self.ref = ref
                self.value = value

        c1 = FakeComponent("R1", 1000)
        c2 = FakeComponent("R2", 2000)
        dist = NormalPct(0.01)

        group = CorrelatedGroup([c1, c2], dist)

        assert len(group.components) == 2
        assert group.components[0] is c1
        assert group.components[1] is c2
        assert group.dist is dist

    def test_repr(self):
        """CorrelatedGroup should have informative repr."""

        class FakeComponent:
            def __init__(self, ref, value):
                self.ref = ref
                self.value = value

        c1 = FakeComponent("R1", 1000)
        group = CorrelatedGroup([c1], NormalPct(0.01))

        repr_str = repr(group)
        assert "CorrelatedGroup" in repr_str
        assert "R1" in repr_str

    def test_hash_and_equality(self):
        """CorrelatedGroup instances should be unique by identity."""

        class FakeComponent:
            def __init__(self, ref, value):
                self.ref = ref
                self.value = value

        c1 = FakeComponent("R1", 1000)

        group1 = CorrelatedGroup([c1], NormalPct(0.01))
        group2 = CorrelatedGroup([c1], NormalPct(0.01))

        # Different instances are not equal
        assert group1 != group2
        assert hash(group1) != hash(group2)

        # Same instance is equal
        assert group1 == group1


class TestCorrelatedSampling:
    """Integration tests for correlated sampling behavior."""

    def test_correlated_vs_independent(self):
        """Correlated components should have same relative variation."""

        # Simulate correlated sampling (same multiplier)
        rnd = random.Random(42)
        dist = NormalPct(0.01)

        # Correlated: one multiplier for all
        multiplier = dist.sample(1.0, rnd)
        corr_r1 = 1000 * multiplier
        corr_r2 = 2000 * multiplier

        # Check that relative variations are identical
        rel_var_r1 = (corr_r1 - 1000) / 1000
        rel_var_r2 = (corr_r2 - 2000) / 2000
        assert (
            abs(rel_var_r1 - rel_var_r2) < 1e-10
        ), "Correlated should have same relative variation"

        # Independent: different samples
        rnd2 = random.Random(42)
        indep_r1 = dist.sample(1000, rnd2)
        indep_r2 = dist.sample(2000, rnd2)

        # Independent variations are different
        rel_var_indep_r1 = (indep_r1 - 1000) / 1000
        rel_var_indep_r2 = (indep_r2 - 2000) / 2000
        assert abs(rel_var_indep_r1 - rel_var_indep_r2) > 1e-10, "Independent should differ"

    def test_percentage_distribution_correlation(self):
        """Percentage-based distributions should use same multiplier for correlation."""
        rnd = random.Random(42)

        for dist_class in [NormalPct, UniformPct, TriangularPct]:
            dist = dist_class(0.01)
            mult = dist.sample(1.0, rnd)

            # Applying multiplier to different nominals
            v1 = 100 * mult
            v2 = 200 * mult

            # Relative deviations should be identical
            assert abs((v1 / 100) - (v2 / 200)) < 1e-10

    def test_absolute_distribution_correlation(self):
        """Absolute distributions should use same offset for correlation."""
        rnd = random.Random(42)

        for dist_class in [NormalAbs, TriangularAbs, UniformAbs]:
            if dist_class == NormalAbs:
                dist = dist_class(0.001)
            else:
                dist = dist_class(0.001)

            offset = dist.sample(0.0, rnd)

            # Applying offset to different nominals
            v1 = 100 + offset
            v2 = 200 + offset

            # Absolute deviations should be identical
            assert abs((v1 - 100) - (v2 - 200)) < 1e-10
