"""Manufacturing Yield Prediction

Demonstrates how to predict manufacturing yield based on
component tolerances and specification limits.

Run: python examples/monte_carlo/yield_analysis.py
"""

import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def apply_tolerance(nominal: float, tolerance: float) -> float:
    """Apply random tolerance (Gaussian, 3σ = tolerance)."""
    sigma = tolerance / 3
    return nominal * (1 + random.gauss(0, sigma))


def build_voltage_reference(r1: float, r2: float, r3: float, vref: float = 5.0) -> Circuit:
    """Build a voltage reference divider network."""
    circuit = Circuit("voltage_reference")

    v_ref = Vdc("ref", vref)
    r_1 = Resistor("1", resistance=r1)
    r_2 = Resistor("2", resistance=r2)
    r_3 = Resistor("3", resistance=r3)

    circuit.add(v_ref, r_1, r_2, r_3)

    vref_net = Net("vref")
    v1 = Net("v1")
    vout = Net("vout")

    circuit.connect(v_ref.ports[0], vref_net)
    circuit.connect(v_ref.ports[1], GND)
    circuit.connect(r_1.ports[0], vref_net)
    circuit.connect(r_1.ports[1], v1)
    circuit.connect(r_2.ports[0], v1)
    circuit.connect(r_2.ports[1], vout)
    circuit.connect(r_3.ports[0], vout)
    circuit.connect(r_3.ports[1], GND)

    return circuit


def calculate_output(r1: float, r2: float, r3: float, vref: float = 5.0) -> float:
    """Calculate output voltage of divider network."""
    # Parallel R2||R3 then divider with R1
    r_parallel = (r2 * r3) / (r2 + r3)
    return vref * r_parallel / (r1 + r_parallel)


def run_yield_analysis(
    r1_nom: float,
    r2_nom: float,
    r3_nom: float,
    tolerance: float,
    vref: float,
    spec_min: float,
    spec_max: float,
    n_runs: int,
) -> dict:
    """Run yield analysis Monte Carlo."""
    results = []
    passes = 0

    for _ in range(n_runs):
        r1 = apply_tolerance(r1_nom, tolerance)
        r2 = apply_tolerance(r2_nom, tolerance)
        r3 = apply_tolerance(r3_nom, tolerance)
        vout = calculate_output(r1, r2, r3, vref)
        results.append(vout)

        if spec_min <= vout <= spec_max:
            passes += 1

    mean = sum(results) / n_runs
    std = (sum((x - mean) ** 2 for x in results) / n_runs) ** 0.5
    yield_pct = passes / n_runs * 100

    return {
        "mean": mean,
        "std": std,
        "min": min(results),
        "max": max(results),
        "yield_pct": yield_pct,
        "rejects": n_runs - passes,
        "cpk": min(spec_max - mean, mean - spec_min) / (3 * std) if std > 0 else float("inf"),
    }


def main():
    """Demonstrate yield analysis."""
    print("=" * 60)
    print("Monte Carlo: Manufacturing Yield Analysis")
    print("=" * 60)

    random.seed(42)
    n_runs = 100000

    # Design: 2.5V output from 5V reference
    vref = 5.0
    target = 2.5
    r1_nom = 10_000
    r2_nom = 20_000
    r3_nom = 20_000

    vout_nom = calculate_output(r1_nom, r2_nom, r3_nom, vref)

    circuit = build_voltage_reference(r1_nom, r2_nom, r3_nom, vref)

    print(f"""
   Voltage Reference Yield Analysis

   Circuit:
   Vref ──[R1]──┬──[R2]──┬── (load)
                │        │
               (tap)    [R3]
                         │
                        GND

   Nominal: R1 = {r1_nom/1000:.0f}kΩ, R2 = {r2_nom/1000:.0f}kΩ, R3 = {r3_nom/1000:.0f}kΩ
   Nominal Vout = {vout_nom:.3f} V (target: {target:.1f} V)
""")

    # Analyze different spec limits with 1% components
    spec_scenarios = [
        ("±1% spec", target * 0.99, target * 1.01),
        ("±2% spec", target * 0.98, target * 1.02),
        ("±5% spec", target * 0.95, target * 1.05),
    ]

    print("   Yield vs Specification (1% components):")
    print("   " + "-" * 55)

    for name, spec_min, spec_max in spec_scenarios:
        result = run_yield_analysis(r1_nom, r2_nom, r3_nom, 0.01, vref, spec_min, spec_max, n_runs)
        print(
            f"   {name:12s}: Yield = {result['yield_pct']:.2f}%, "
            f"Cpk = {result['cpk']:.2f}, "
            f"Rejects = {result['rejects']:,}"
        )

    # Analyze different component tolerances with ±2% spec
    print("\n   Yield vs Component Tolerance (±2% spec):")
    print("   " + "-" * 55)

    spec_min = target * 0.98
    spec_max = target * 1.02

    tol_scenarios = [
        ("0.1% components", 0.001),
        ("0.5% components", 0.005),
        ("1% components", 0.01),
        ("2% components", 0.02),
        ("5% components", 0.05),
    ]

    for name, tol in tol_scenarios:
        result = run_yield_analysis(r1_nom, r2_nom, r3_nom, tol, vref, spec_min, spec_max, n_runs)
        print(f"   {name:18s}: Yield = {result['yield_pct']:6.2f}%, " f"Cpk = {result['cpk']:.2f}")

    print("""
   Process Capability Index (Cpk) Interpretation:
   ┌─────────┬────────────────────────────────────────┐
   │ Cpk     │ Meaning                                │
   ├─────────┼────────────────────────────────────────┤
   │ < 1.0   │ Process not capable (high defects)     │
   │ 1.0-1.33│ Marginally capable                     │
   │ 1.33-1.5│ Capable (typical manufacturing)        │
   │ 1.5-2.0 │ Very capable (good quality)            │
   │ > 2.0   │ Six Sigma quality                      │
   └─────────┴────────────────────────────────────────┘

   Design Guidelines:
   - Target Cpk ≥ 1.33 for production
   - Use spec limits 3× wider than component tolerance
   - Consider 100% testing for Cpk < 1.0
   - Ratio-matched resistors improve yield significantly
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
