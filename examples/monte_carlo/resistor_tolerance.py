"""Basic Resistor Tolerance Analysis

Demonstrates Monte Carlo analysis of a simple voltage divider
to understand how component tolerances affect output voltage.

Run: python examples/monte_carlo/resistor_tolerance.py
"""

import random
from dataclasses import dataclass

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


@dataclass
class ToleranceSpec:
    """Component tolerance specification."""

    nominal: float
    tolerance: float  # As fraction (0.01 = 1%)
    distribution: str = "uniform"  # "uniform", "gaussian", "truncated_gaussian"


def apply_tolerance(nominal: float, tolerance: float, distribution: str = "uniform") -> float:
    """Apply tolerance variation to a nominal value.

    Args:
        nominal: Nominal component value
        tolerance: Tolerance as fraction (0.01 = 1%)
        distribution: "uniform", "gaussian", or "truncated_gaussian"

    Returns:
        Value with tolerance applied
    """
    if distribution == "uniform":
        # Uniform distribution within tolerance bounds
        variation = random.uniform(-tolerance, tolerance)
    elif distribution == "gaussian":
        # Gaussian with 3σ = tolerance (99.7% within bounds)
        sigma = tolerance / 3
        variation = random.gauss(0, sigma)
    elif distribution == "truncated_gaussian":
        # Gaussian but clipped to tolerance bounds
        sigma = tolerance / 3
        variation = random.gauss(0, sigma)
        variation = max(-tolerance, min(tolerance, variation))
    else:
        variation = 0

    return nominal * (1 + variation)


def build_voltage_divider(r1_value: float, r2_value: float, vin: float = 10.0) -> Circuit:
    """Build a voltage divider circuit.

    Args:
        r1_value: Top resistor value
        r2_value: Bottom resistor value
        vin: Input voltage

    Returns:
        Circuit with voltage divider
    """
    circuit = Circuit("voltage_divider")

    v1 = Vdc("1", vin)
    r1 = Resistor("1", resistance=r1_value)
    r2 = Resistor("2", resistance=r2_value)

    circuit.add(v1, r1, r2)

    # Nets
    v_in = Net("vin")
    v_out = Net("vout")

    circuit.connect(v1.ports[0], v_in)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], v_in)
    circuit.connect(r1.ports[1], v_out)
    circuit.connect(r2.ports[0], v_out)
    circuit.connect(r2.ports[1], GND)

    return circuit


def calculate_divider_output(r1: float, r2: float, vin: float = 10.0) -> float:
    """Calculate voltage divider output analytically.

    Args:
        r1: Top resistor value
        r2: Bottom resistor value
        vin: Input voltage

    Returns:
        Output voltage
    """
    return vin * r2 / (r1 + r2)


def run_monte_carlo(
    r1_spec: ToleranceSpec,
    r2_spec: ToleranceSpec,
    vin: float,
    n_runs: int,
) -> list[float]:
    """Run Monte Carlo simulation.

    Args:
        r1_spec: R1 tolerance specification
        r2_spec: R2 tolerance specification
        vin: Input voltage
        n_runs: Number of Monte Carlo runs

    Returns:
        List of output voltages
    """
    results = []

    for _ in range(n_runs):
        r1_val = apply_tolerance(r1_spec.nominal, r1_spec.tolerance, r1_spec.distribution)
        r2_val = apply_tolerance(r2_spec.nominal, r2_spec.tolerance, r2_spec.distribution)
        vout = calculate_divider_output(r1_val, r2_val, vin)
        results.append(vout)

    return results


def analyze_results(results: list[float], nominal: float) -> dict:
    """Analyze Monte Carlo results.

    Args:
        results: List of simulation results
        nominal: Expected nominal value

    Returns:
        Dictionary with statistics
    """
    n = len(results)
    mean = sum(results) / n
    variance = sum((x - mean) ** 2 for x in results) / n
    std_dev = variance**0.5

    min_val = min(results)
    max_val = max(results)
    range_val = max_val - min_val

    # Calculate percentiles
    sorted_results = sorted(results)
    p01 = sorted_results[int(0.01 * n)]
    p99 = sorted_results[int(0.99 * n)]
    p05 = sorted_results[int(0.05 * n)]
    p95 = sorted_results[int(0.95 * n)]

    # Error from nominal
    mean_error = (mean - nominal) / nominal * 100
    max_error = max(abs(min_val - nominal), abs(max_val - nominal)) / nominal * 100

    return {
        "n_samples": n,
        "nominal": nominal,
        "mean": mean,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "p01": p01,
        "p99": p99,
        "p05": p05,
        "p95": p95,
        "mean_error_pct": mean_error,
        "max_error_pct": max_error,
        "3sigma_range": (mean - 3 * std_dev, mean + 3 * std_dev),
    }


def main():
    """Demonstrate resistor tolerance Monte Carlo analysis."""
    print("=" * 60)
    print("Monte Carlo: Resistor Tolerance Analysis")
    print("=" * 60)

    # Design parameters
    vin = 10.0  # 10V input
    r1_nom = 10_000  # 10kΩ
    r2_nom = 10_000  # 10kΩ
    vout_nom = calculate_divider_output(r1_nom, r2_nom, vin)

    # Build nominal circuit
    circuit = build_voltage_divider(r1_nom, r2_nom, vin)

    print(f"""
   Voltage Divider Monte Carlo Analysis

   Circuit:
   Vin ──[R1]──┬── Vout
               │
              [R2]
               │
              GND

   Nominal values:
   - Vin = {vin} V
   - R1 = {r1_nom/1000:.0f} kΩ
   - R2 = {r2_nom/1000:.0f} kΩ
   - Vout = Vin × R2/(R1+R2) = {vout_nom:.3f} V
""")

    # Run Monte Carlo with different tolerances
    n_runs = 10000
    random.seed(42)  # Reproducibility

    print(f"   Running {n_runs} Monte Carlo iterations...\n")

    scenarios = [
        ("1% resistors", 0.01),
        ("5% resistors", 0.05),
        ("10% resistors", 0.10),
    ]

    for name, tol in scenarios:
        r1_spec = ToleranceSpec(r1_nom, tol)
        r2_spec = ToleranceSpec(r2_nom, tol)

        results = run_monte_carlo(r1_spec, r2_spec, vin, n_runs)
        stats = analyze_results(results, vout_nom)

        print(f"   {name}:")
        print(f"   - Mean: {stats['mean']:.4f} V (error: {stats['mean_error_pct']:+.3f}%)")
        print(f"   - Std Dev: {stats['std_dev']*1000:.2f} mV")
        print(f"   - Range: {stats['min']:.4f} to {stats['max']:.4f} V")
        print(f"   - Max Error: ±{stats['max_error_pct']:.2f}%")
        print(f"   - 3σ Range: {stats['3sigma_range'][0]:.4f} to {stats['3sigma_range'][1]:.4f} V")
        print()

    # Compare distributions
    print("   Distribution Comparison (5% tolerance):")
    print("   " + "-" * 50)

    distributions = ["uniform", "gaussian", "truncated_gaussian"]
    r1_nom_spec = 10_000
    r2_nom_spec = 10_000
    tol = 0.05

    for dist in distributions:
        r1_spec = ToleranceSpec(r1_nom_spec, tol, dist)
        r2_spec = ToleranceSpec(r2_nom_spec, tol, dist)
        results = run_monte_carlo(r1_spec, r2_spec, vin, n_runs)
        stats = analyze_results(results, vout_nom)

        print(
            f"   {dist:20s}: σ = {stats['std_dev']*1000:.2f} mV, "
            f"range = {stats['range']*1000:.1f} mV"
        )

    print("""
   Key Observations:
   ┌────────────────────────────────────────────────────────┐
   │ 1. Output tolerance ≈ component tolerance (for equal R)│
   │ 2. Mean stays close to nominal (errors tend to cancel) │
   │ 3. Gaussian distribution gives narrower spread         │
   │ 4. Uniform has more samples near extremes              │
   └────────────────────────────────────────────────────────┘

   For tighter output tolerance:
   - Use lower tolerance components
   - Use ratio-matched resistors
   - Consider active regulation
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
