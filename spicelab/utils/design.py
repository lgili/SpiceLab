"""Circuit design helpers and calculators.

This module provides common circuit design calculations including:

- Voltage dividers
- Op-amp gain calculations
- RC/RL time constants
- Sensor linearization helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .e_series import ESeries, find_best_match, nearest_value

# -----------------------------------------------------------------------------
# Voltage Divider
# -----------------------------------------------------------------------------


@dataclass
class VoltageDividerResult:
    """Result of voltage divider design."""

    r_top: float
    r_bottom: float
    v_out: float
    ratio: float
    error: float  # Relative error from target


def voltage_divider_ratio(r_top: float, r_bottom: float) -> float:
    """Calculate voltage divider ratio Vout/Vin = R_bottom / (R_top + R_bottom)."""
    return r_bottom / (r_top + r_bottom)


def design_voltage_divider(
    v_in: float,
    v_out_target: float,
    r_total_target: float = 10e3,
    series: ESeries = "E24",
) -> VoltageDividerResult:
    """Design a voltage divider using standard resistor values.

    Args:
        v_in: Input voltage
        v_out_target: Desired output voltage
        r_total_target: Target total resistance (R_top + R_bottom)
        series: E-series to use for resistor selection

    Returns:
        VoltageDividerResult with selected resistors and achieved output
    """
    ratio_target = v_out_target / v_in

    # Calculate ideal values
    r_bottom_ideal = r_total_target * ratio_target
    r_top_ideal = r_total_target - r_bottom_ideal

    # Find nearest standard values
    r_bottom = nearest_value(r_bottom_ideal, series)
    r_top = nearest_value(r_top_ideal, series)

    # Calculate actual output
    actual_ratio = voltage_divider_ratio(r_top, r_bottom)
    v_out_actual = v_in * actual_ratio
    error = abs(v_out_actual - v_out_target) / v_out_target if v_out_target != 0 else 0

    return VoltageDividerResult(
        r_top=r_top,
        r_bottom=r_bottom,
        v_out=v_out_actual,
        ratio=actual_ratio,
        error=error,
    )


# -----------------------------------------------------------------------------
# Op-Amp Gain Calculations
# -----------------------------------------------------------------------------


def opamp_noninverting_gain(r_feedback: float, r_ground: float) -> float:
    """Calculate non-inverting op-amp gain: G = 1 + Rf/Rg."""
    return 1.0 + r_feedback / r_ground


def opamp_inverting_gain(r_feedback: float, r_input: float) -> float:
    """Calculate inverting op-amp gain: G = -Rf/Rin."""
    return -r_feedback / r_input


@dataclass
class OpAmpGainResult:
    """Result of op-amp gain stage design."""

    r_feedback: float
    r_ground_or_input: float
    gain: float
    error: float


def design_noninverting_gain(
    gain_target: float,
    r_ground_target: float = 10e3,
    series: ESeries = "E24",
) -> OpAmpGainResult:
    """Design non-inverting op-amp stage for target gain.

    Args:
        gain_target: Desired voltage gain (must be >= 1)
        r_ground_target: Target value for ground resistor
        series: E-series for resistor selection

    Returns:
        OpAmpGainResult with selected resistors
    """
    if gain_target < 1:
        raise ValueError("Non-inverting gain must be >= 1")

    r_ground = nearest_value(r_ground_target, series)

    # G = 1 + Rf/Rg => Rf = Rg * (G - 1)
    r_feedback_ideal = r_ground * (gain_target - 1)
    match = find_best_match(r_feedback_ideal, series)
    r_feedback = match.value

    actual_gain = opamp_noninverting_gain(r_feedback, r_ground)
    error = abs(actual_gain - gain_target) / gain_target if gain_target != 0 else 0

    return OpAmpGainResult(
        r_feedback=r_feedback,
        r_ground_or_input=r_ground,
        gain=actual_gain,
        error=error,
    )


def design_inverting_gain(
    gain_target: float,
    r_input_target: float = 10e3,
    series: ESeries = "E24",
) -> OpAmpGainResult:
    """Design inverting op-amp stage for target gain (magnitude).

    Args:
        gain_target: Desired voltage gain magnitude (positive number)
        r_input_target: Target value for input resistor
        series: E-series for resistor selection

    Returns:
        OpAmpGainResult with selected resistors (gain will be negative)
    """
    r_input = nearest_value(r_input_target, series)

    # |G| = Rf/Rin => Rf = Rin * |G|
    r_feedback_ideal = r_input * abs(gain_target)
    match = find_best_match(r_feedback_ideal, series)
    r_feedback = match.value

    actual_gain = opamp_inverting_gain(r_feedback, r_input)
    error = abs(abs(actual_gain) - abs(gain_target)) / abs(gain_target) if gain_target != 0 else 0

    return OpAmpGainResult(
        r_feedback=r_feedback,
        r_ground_or_input=r_input,
        gain=actual_gain,
        error=error,
    )


# -----------------------------------------------------------------------------
# RC/RL Time Constants
# -----------------------------------------------------------------------------


def rc_time_constant(r: float, c: float) -> float:
    """Calculate RC time constant in seconds."""
    return r * c


def rc_cutoff_frequency(r: float, c: float) -> float:
    """Calculate RC filter -3dB cutoff frequency in Hz."""
    tau = rc_time_constant(r, c)
    return 1.0 / (2.0 * np.pi * tau)


def rl_time_constant(r: float, l: float) -> float:
    """Calculate RL time constant in seconds."""
    return l / r


@dataclass
class RCFilterResult:
    """Result of RC filter design."""

    r: float
    c: float
    cutoff_hz: float
    time_constant: float
    error: float


def design_rc_lowpass(
    cutoff_hz: float,
    r_target: float = 10e3,
    r_series: ESeries = "E24",
    c_series: Literal["E12", "E24"] = "E12",
) -> RCFilterResult:
    """Design RC low-pass filter for target cutoff frequency.

    Args:
        cutoff_hz: Target -3dB cutoff frequency in Hz
        r_target: Target resistance value
        r_series: E-series for resistor
        c_series: Series for capacitor (typically E12)

    Returns:
        RCFilterResult with selected components
    """
    from .e_series import nearest_capacitor

    r = nearest_value(r_target, r_series)

    # f_c = 1/(2*pi*R*C) => C = 1/(2*pi*R*f_c)
    c_ideal = 1.0 / (2.0 * np.pi * r * cutoff_hz)
    c = nearest_capacitor(c_ideal, c_series)

    actual_cutoff = rc_cutoff_frequency(r, c)
    tau = rc_time_constant(r, c)
    error = abs(actual_cutoff - cutoff_hz) / cutoff_hz if cutoff_hz != 0 else 0

    return RCFilterResult(
        r=r,
        c=c,
        cutoff_hz=actual_cutoff,
        time_constant=tau,
        error=error,
    )


# -----------------------------------------------------------------------------
# Sensor Helpers
# -----------------------------------------------------------------------------


def pt100_resistance(temp_c: float, r0: float = 100.0) -> float:
    """Calculate PT100/PT1000 resistance using Callendar-Van Dusen equation.

    Args:
        temp_c: Temperature in Celsius
        r0: Resistance at 0°C (100 for PT100, 1000 for PT1000)

    Returns:
        Resistance in Ohms
    """
    a = 3.9083e-3
    b = -5.775e-7
    c = -4.183e-12

    if temp_c >= 0:
        return r0 * (1 + a * temp_c + b * temp_c**2)
    # Below 0°C, add cubic term
    return r0 * (1 + a * temp_c + b * temp_c**2 + c * (temp_c - 100) * temp_c**3)


def pt1000_resistance(temp_c: float) -> float:
    """Calculate PT1000 resistance at given temperature."""
    return pt100_resistance(temp_c, r0=1000.0)


def ntc_resistance(temp_c: float, r25: float = 10e3, beta: float = 3950.0) -> float:
    """Calculate NTC thermistor resistance using beta equation.

    Args:
        temp_c: Temperature in Celsius
        r25: Resistance at 25°C
        beta: Beta coefficient (typically 3000-4000)

    Returns:
        Resistance in Ohms
    """
    t_kelvin = temp_c + 273.15
    t25_kelvin = 298.15  # 25°C
    return float(r25 * np.exp(beta * (1.0 / t_kelvin - 1.0 / t25_kelvin)))


def ntc_temperature(resistance: float, r25: float = 10e3, beta: float = 3950.0) -> float:
    """Calculate temperature from NTC resistance.

    Args:
        resistance: Measured resistance in Ohms
        r25: Resistance at 25°C
        beta: Beta coefficient

    Returns:
        Temperature in Celsius
    """
    t25_kelvin = 298.15
    t_kelvin = 1.0 / (1.0 / t25_kelvin + (1.0 / beta) * np.log(resistance / r25))
    return float(t_kelvin - 273.15)


# -----------------------------------------------------------------------------
# Power and Current Calculations
# -----------------------------------------------------------------------------


def power_dissipation(voltage: float, current: float) -> float:
    """Calculate power dissipation P = V * I."""
    return voltage * current


def resistor_power(voltage: float, resistance: float) -> float:
    """Calculate power dissipated in resistor P = V²/R."""
    return voltage**2 / resistance


def resistor_current(voltage: float, resistance: float) -> float:
    """Calculate current through resistor I = V/R."""
    return voltage / resistance


def max_resistor_voltage(power_rating: float, resistance: float) -> float:
    """Calculate maximum voltage for resistor given power rating V = sqrt(P*R)."""
    return float(np.sqrt(power_rating * resistance))


# -----------------------------------------------------------------------------
# Decibel Conversions
# -----------------------------------------------------------------------------


def db_to_linear(db: float) -> float:
    """Convert decibels to linear ratio (voltage/amplitude)."""
    return 10 ** (db / 20.0)


def linear_to_db(ratio: float) -> float:
    """Convert linear ratio to decibels."""
    return float(20.0 * np.log10(ratio))


def db_power_to_linear(db: float) -> float:
    """Convert power decibels to linear ratio."""
    return 10 ** (db / 10.0)


def linear_to_db_power(ratio: float) -> float:
    """Convert linear power ratio to decibels."""
    return float(10.0 * np.log10(ratio))
