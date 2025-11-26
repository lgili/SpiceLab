"""Tests for the Worst-Case Analysis (WCA) module."""

import pytest
from spicelab.analysis.wca import (
    WcaCorner,
    WcaResult,
    tolerance_to_normal,
    tolerance_to_uniform,
)


class TestToleranceHelpers:
    """Tests for tolerance conversion functions."""

    def test_tolerance_to_normal_default(self):
        """Default 3-sigma conversion."""
        # 1% tolerance -> 0.333% sigma
        result = tolerance_to_normal(0.01)
        assert abs(result - 0.01 / 3) < 1e-10

    def test_tolerance_to_normal_custom_sigma(self):
        """Custom sigma multiplier."""
        # 1% tolerance with 2-sigma
        result = tolerance_to_normal(0.01, sigma_multiplier=2.0)
        assert abs(result - 0.005) < 1e-10

    def test_tolerance_to_normal_negative_raises(self):
        """Negative tolerance should raise."""
        with pytest.raises(ValueError, match="tolerance must be >= 0"):
            tolerance_to_normal(-0.01)

    def test_tolerance_to_normal_zero_sigma_raises(self):
        """Zero sigma multiplier should raise."""
        with pytest.raises(ValueError, match="sigma_multiplier must be > 0"):
            tolerance_to_normal(0.01, sigma_multiplier=0)

    def test_tolerance_to_uniform(self):
        """Uniform tolerance is passthrough."""
        assert tolerance_to_uniform(0.01) == 0.01
        assert tolerance_to_uniform(0.05) == 0.05

    def test_tolerance_to_uniform_negative_raises(self):
        """Negative tolerance should raise."""
        with pytest.raises(ValueError, match="tolerance must be >= 0"):
            tolerance_to_uniform(-0.01)


class TestWcaResultMethods:
    """Tests for WcaResult data class methods."""

    def test_empty_corners_raises(self):
        """find_extreme with empty corners should raise."""
        result = WcaResult(
            corners=[],
            nominal_combo={},
            tolerances={},
        )
        with pytest.raises(ValueError, match="No corners to search"):
            result.find_extreme(lambda c: 0.0)


class TestWcaCornerDataclass:
    """Tests for WcaCorner dataclass."""

    def test_corner_name(self):
        """Corner name is stored correctly."""

        class FakeHandle:
            def dataset(self):
                return {}

            def attrs(self):
                return {}

        corner = WcaCorner(
            combo={"R1": 1000.0, "R2": 2000.0},
            corner_signs={"R1": 1, "R2": -1},
            handle=FakeHandle(),  # type: ignore
            corner_name="R1+, R2-",
        )

        assert corner.corner_name == "R1+, R2-"
        assert corner.combo["R1"] == 1000.0
        assert corner.corner_signs["R1"] == 1
        assert corner.corner_signs["R2"] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
