"""Tests for slew rate measurement (Sprint 6 - M7).

Tests for SlewRateSpec measurement including:
- Rising edge slew rate
- Falling edge slew rate
- Both edges (worst case)
- Different units
"""

from __future__ import annotations

import numpy as np
import pytest

from spicelab.analysis import SlewRateSpec, measure
from spicelab.io.raw_reader import Trace, TraceSet


def _make_traceset(data: dict[str, np.ndarray]) -> TraceSet:
    """Create a TraceSet from a dictionary of arrays."""
    traces = [Trace(name=name, unit=None, values=values) for name, values in data.items()]
    return TraceSet(traces)


def _make_step_response(
    t_start: float = 0.0,
    t_end: float = 100e-6,
    n_points: int = 1000,
    rise_time: float = 10e-6,
    fall_time: float = 15e-6,
    v_low: float = 0.0,
    v_high: float = 5.0,
) -> TraceSet:
    """Create a step response waveform with controlled rise/fall times.

    The waveform goes: low -> rising -> high -> falling -> low
    """
    t = np.linspace(t_start, t_end, n_points)

    # Create a trapezoid waveform
    # Rise starts at t_end/4, fall starts at 3*t_end/4
    t_rise_start = t_end / 4
    t_rise_end = t_rise_start + rise_time
    t_fall_start = 3 * t_end / 4
    t_fall_end = t_fall_start + fall_time

    y = np.zeros_like(t)

    # Initial low
    y[t < t_rise_start] = v_low

    # Rising edge (linear ramp)
    rising_mask = (t >= t_rise_start) & (t <= t_rise_end)
    y[rising_mask] = v_low + (v_high - v_low) * (t[rising_mask] - t_rise_start) / rise_time

    # High plateau
    y[(t > t_rise_end) & (t < t_fall_start)] = v_high

    # Falling edge (linear ramp)
    falling_mask = (t >= t_fall_start) & (t <= t_fall_end)
    y[falling_mask] = v_high - (v_high - v_low) * (t[falling_mask] - t_fall_start) / fall_time

    # Final low
    y[t > t_fall_end] = v_low

    return _make_traceset({"time": t, "V(out)": y})


class TestSlewRateBasic:
    """Basic slew rate measurement tests."""

    def test_rising_edge_slew_rate(self) -> None:
        """Test measuring rising edge slew rate."""
        # 5V swing in 10us = 0.5 V/us
        ts = _make_step_response(rise_time=10e-6, v_low=0, v_high=5)

        results = measure(
            ts,
            [SlewRateSpec(name="sr_rising", signal="V(out)", edge="rising", units="V/us")],
            return_as="python",
        )

        assert len(results) == 1
        result = results[0]
        assert result["measure"] == "sr_rising"
        assert result["type"] == "slew_rate"
        assert result["units"] == "V/us"

        # Expected: 5V / 10us = 0.5 V/us
        # Allow 10% tolerance due to threshold filtering
        assert pytest.approx(result["value"], rel=0.15) == 0.5
        assert result["rising_sr"] > 0
        assert result["falling_sr"] == 0  # Not measured for rising only

    def test_falling_edge_slew_rate(self) -> None:
        """Test measuring falling edge slew rate."""
        # 5V swing in 15us = 0.333 V/us
        ts = _make_step_response(fall_time=15e-6, v_low=0, v_high=5)

        results = measure(
            ts,
            [SlewRateSpec(name="sr_falling", signal="V(out)", edge="falling", units="V/us")],
            return_as="python",
        )

        assert len(results) == 1
        result = results[0]
        assert result["edge"] == "falling"

        # Expected: 5V / 15us = 0.333 V/us
        assert pytest.approx(result["value"], rel=0.15) == 0.333
        assert result["falling_sr"] > 0
        assert result["rising_sr"] == 0

    def test_both_edges_returns_minimum(self) -> None:
        """Test that 'both' returns the minimum (worst case) slew rate."""
        # Rising: 5V / 10us = 0.5 V/us
        # Falling: 5V / 20us = 0.25 V/us (slower)
        ts = _make_step_response(rise_time=10e-6, fall_time=20e-6, v_low=0, v_high=5)

        results = measure(
            ts,
            [SlewRateSpec(name="sr_both", signal="V(out)", edge="both", units="V/us")],
            return_as="python",
        )

        result = results[0]

        # Should return falling (slower = 0.25)
        assert pytest.approx(result["value"], rel=0.15) == 0.25
        assert result["rising_sr"] > result["falling_sr"]


class TestSlewRateUnits:
    """Test slew rate unit conversions."""

    def test_volts_per_second(self) -> None:
        """Test V/s units."""
        # 5V in 10us = 500000 V/s
        ts = _make_step_response(rise_time=10e-6, v_low=0, v_high=5)

        results = measure(
            ts,
            [SlewRateSpec(name="sr", signal="V(out)", edge="rising", units="V/s")],
            return_as="python",
        )

        # Expected: 5V / 10e-6s = 500000 V/s
        assert pytest.approx(results[0]["value"], rel=0.15) == 500000
        assert results[0]["units"] == "V/s"

    def test_volts_per_nanosecond(self) -> None:
        """Test V/ns units."""
        # 5V in 10us = 0.0005 V/ns
        ts = _make_step_response(rise_time=10e-6, v_low=0, v_high=5)

        results = measure(
            ts,
            [SlewRateSpec(name="sr", signal="V(out)", edge="rising", units="V/ns")],
            return_as="python",
        )

        # Expected: 5V / 10000ns = 0.0005 V/ns
        assert pytest.approx(results[0]["value"], rel=0.15) == 0.0005


class TestSlewRateEdgeCases:
    """Test edge cases and error handling."""

    def test_flat_signal_returns_zero(self) -> None:
        """Test that a flat signal returns zero slew rate."""
        t = np.linspace(0, 100e-6, 1000)
        y = np.ones_like(t) * 2.5  # Constant voltage

        ts = _make_traceset({"time": t, "V(out)": y})

        results = measure(
            ts,
            [SlewRateSpec(name="sr", signal="V(out)")],
            return_as="python",
        )

        assert results[0]["value"] == 0.0

    def test_custom_thresholds(self) -> None:
        """Test custom threshold levels."""
        ts = _make_step_response(rise_time=10e-6, v_low=0, v_high=5)

        # Use 20-80% thresholds instead of 10-90%
        results = measure(
            ts,
            [
                SlewRateSpec(
                    name="sr_narrow",
                    signal="V(out)",
                    edge="rising",
                    threshold_low=0.2,
                    threshold_high=0.8,
                )
            ],
            return_as="python",
        )

        # Should still measure approximately the same slew rate
        # (for linear ramp, dV/dt is constant)
        assert results[0]["value"] > 0

    def test_short_signal_raises(self) -> None:
        """Test that a signal with only 1 point raises an error."""
        ts = _make_traceset({"time": np.array([0.0]), "V(out)": np.array([1.0])})

        with pytest.raises(ValueError, match="too short"):
            measure(ts, [SlewRateSpec(name="sr", signal="V(out)")], return_as="python")


class TestSlewRateRealistic:
    """Realistic op-amp slew rate scenarios."""

    def test_opamp_like_response(self) -> None:
        """Test slew rate with an exponential-like step response."""
        t = np.linspace(0, 50e-6, 1000)

        # Simulate an op-amp step response with slew limiting
        # Use a combination of slew-limited and exponential response
        tau = 5e-6  # Time constant
        sr_limit = 1e6  # 1 V/us slew limit
        v_final = 5.0

        y = np.zeros_like(t)
        for i, ti in enumerate(t):
            # Ideal exponential response
            ideal = v_final * (1 - np.exp(-ti / tau))
            # Slew-limited response
            slew_limited = sr_limit * ti
            # Take minimum (slew limiting)
            y[i] = min(ideal, slew_limited, v_final)

        ts = _make_traceset({"time": t, "V(out)": y})

        results = measure(
            ts,
            [SlewRateSpec(name="sr", signal="V(out)", edge="rising", units="V/us")],
            return_as="python",
        )

        # Should measure close to 1 V/us during slew-limited region
        assert results[0]["value"] > 0.5  # At least 0.5 V/us
        assert results[0]["value"] < 2.0  # Less than 2 V/us
