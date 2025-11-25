"""Parameter Sweep Automation

Demonstrates automated parameter sweeping for design exploration
and optimization, with visualization of results.

Run: python examples/automation/parameter_sweep.py
"""

import math
from dataclasses import dataclass

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net


@dataclass
class SweepPoint:
    """Single point in a parameter sweep."""

    param_value: float
    output_value: float
    additional_metrics: dict | None = None


@dataclass
class SweepResult:
    """Complete sweep result."""

    param_name: str
    param_unit: str
    output_name: str
    output_unit: str
    points: list[SweepPoint]


def build_rc_filter(r: float, c: float) -> Circuit:
    """Build an RC lowpass filter."""
    circuit = Circuit("rc_filter")

    v_in = Vac("in", ac_mag=1.0)
    r1 = Resistor("1", resistance=r)
    c1 = Capacitor("1", capacitance=c)

    circuit.add(v_in, r1, c1)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    return circuit


def calculate_cutoff_freq(r: float, c: float) -> float:
    """Calculate RC filter cutoff frequency."""
    return 1 / (2 * math.pi * r * c)


def calculate_impedance_at_freq(r: float, c: float, freq: float) -> float:
    """Calculate filter impedance at a given frequency."""
    xc = 1 / (2 * math.pi * freq * c)
    return math.sqrt(r**2 + xc**2)


def calculate_gain_at_freq(r: float, c: float, freq: float) -> float:
    """Calculate filter gain (magnitude) at a given frequency."""
    fc = calculate_cutoff_freq(r, c)
    f_ratio = freq / fc
    return 1 / math.sqrt(1 + f_ratio**2)


class ParameterSweeper:
    """Automated parameter sweeper."""

    def __init__(self):
        self.results: list[SweepResult] = []

    def linear_sweep(
        self,
        param_name: str,
        start: float,
        stop: float,
        num_points: int,
        eval_func,
        param_unit: str = "",
        output_name: str = "Output",
        output_unit: str = "",
    ) -> SweepResult:
        """Perform a linear parameter sweep."""
        step = (stop - start) / (num_points - 1)
        points = []

        for i in range(num_points):
            param_value = start + i * step
            output_value = eval_func(param_value)
            points.append(SweepPoint(param_value, output_value))

        result = SweepResult(
            param_name=param_name,
            param_unit=param_unit,
            output_name=output_name,
            output_unit=output_unit,
            points=points,
        )
        self.results.append(result)
        return result

    def log_sweep(
        self,
        param_name: str,
        start: float,
        stop: float,
        points_per_decade: int,
        eval_func,
        param_unit: str = "",
        output_name: str = "Output",
        output_unit: str = "",
    ) -> SweepResult:
        """Perform a logarithmic parameter sweep."""
        decades = math.log10(stop / start)
        num_points = int(decades * points_per_decade) + 1
        log_start = math.log10(start)
        log_step = decades / (num_points - 1) if num_points > 1 else 0
        points = []

        for i in range(num_points):
            param_value = 10 ** (log_start + i * log_step)
            output_value = eval_func(param_value)
            points.append(SweepPoint(param_value, output_value))

        result = SweepResult(
            param_name=param_name,
            param_unit=param_unit,
            output_name=output_name,
            output_unit=output_unit,
            points=points,
        )
        self.results.append(result)
        return result

    def find_value_at_target(self, result: SweepResult, target_output: float) -> float | None:
        """Find parameter value that gives target output (linear interpolation)."""
        for i in range(len(result.points) - 1):
            p1 = result.points[i]
            p2 = result.points[i + 1]

            # Check if target is between these points
            if (p1.output_value <= target_output <= p2.output_value) or (
                p2.output_value <= target_output <= p1.output_value
            ):
                # Linear interpolation
                t = (target_output - p1.output_value) / (p2.output_value - p1.output_value)
                return p1.param_value + t * (p2.param_value - p1.param_value)

        return None


def format_engineering(value: float, unit: str = "") -> str:
    """Format value with engineering prefix."""
    if value == 0:
        return f"0 {unit}"

    prefixes = [
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1, ""),
        (1e-3, "m"),
        (1e-6, "µ"),
        (1e-9, "n"),
        (1e-12, "p"),
    ]

    for threshold, prefix in prefixes:
        if abs(value) >= threshold:
            return f"{value/threshold:.3g} {prefix}{unit}"

    return f"{value:.3g} {unit}"


def main():
    """Demonstrate parameter sweep automation."""
    print("=" * 60)
    print("Automation: Parameter Sweep Analysis")
    print("=" * 60)

    sweeper = ParameterSweeper()

    # Fixed capacitor, sweep resistor
    c_nom = 100e-9  # 100nF

    print(f"""
   RC Lowpass Filter Parameter Sweep
   ──────────────────────────────────
   Fixed: C = {format_engineering(c_nom, 'F')}
   Sweep: R from 100Ω to 100kΩ
""")

    # Sweep resistance and calculate cutoff frequency
    result_fc = sweeper.log_sweep(
        param_name="Resistance",
        start=100,
        stop=100_000,
        points_per_decade=10,
        eval_func=lambda r: calculate_cutoff_freq(r, c_nom),
        param_unit="Ω",
        output_name="Cutoff Frequency",
        output_unit="Hz",
    )

    # Display results
    print("   Resistance vs Cutoff Frequency:")
    print("   " + "-" * 50)
    for point in result_fc.points[::5]:  # Show every 5th point
        print(
            f"   R = {format_engineering(point.param_value, 'Ω'):>12s} -> "
            f"fc = {format_engineering(point.output_value, 'Hz'):>12s}"
        )

    # Find R for specific cutoff frequencies
    print("\n   Target Cutoff Frequency Design:")
    print("   " + "-" * 50)

    target_freqs = [100, 1000, 10_000]
    for fc_target in target_freqs:
        r_required = sweeper.find_value_at_target(result_fc, fc_target)
        if r_required:
            print(
                f"   fc = {fc_target:,} Hz requires R = " f"{format_engineering(r_required, 'Ω')}"
            )

    # Build and validate a circuit
    circuit = build_rc_filter(10_000, c_nom)

    # Frequency response sweep
    print("\n   Frequency Response (R=10kΩ, C=100nF):")
    print("   " + "-" * 50)

    r_nom = 10_000
    fc = calculate_cutoff_freq(r_nom, c_nom)

    result_freq = sweeper.log_sweep(
        param_name="Frequency",
        start=1,
        stop=100_000,
        points_per_decade=5,
        eval_func=lambda f: 20 * math.log10(calculate_gain_at_freq(r_nom, c_nom, f)),
        param_unit="Hz",
        output_name="Gain",
        output_unit="dB",
    )

    for point in result_freq.points[::2]:  # Show every other point
        print(
            f"   f = {format_engineering(point.param_value, 'Hz'):>12s} -> "
            f"Gain = {point.output_value:>7.2f} dB"
        )

    print(f"""
   Summary:
   ──────────────────────────────────
   Calculated fc = {format_engineering(fc, 'Hz')}
   -3dB point should be at fc

   Sweep Types Available:
   ┌────────────────────────────────────────────────────────┐
   │ linear_sweep: Uniform spacing (voltage, temperature)  │
   │ log_sweep: Decade spacing (frequency, impedance)      │
   └────────────────────────────────────────────────────────┘

   Applications:
   - Filter design: fc vs R or C
   - Amplifier design: Gain vs feedback ratio
   - Power supply: Output vs load current
   - Thermal: Temperature vs power dissipation
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
