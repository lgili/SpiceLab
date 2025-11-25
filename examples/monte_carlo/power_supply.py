"""Power Supply Output Voltage Distribution

Analyzes how component tolerances affect linear regulator
and voltage divider based power supply accuracy.

Run: python examples/monte_carlo/power_supply.py
"""

import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def build_adjustable_regulator(r1: float, r2: float) -> Circuit:
    """Build adjustable regulator feedback network (e.g., LM317)."""
    circuit = Circuit("adjustable_regulator")

    v_ref = Vdc("ref", 1.25)  # Internal 1.25V reference
    res1 = Resistor("1", resistance=r1)  # Upper resistor
    res2 = Resistor("2", resistance=r2)  # Lower resistor (240Ω typical)

    circuit.add(v_ref, res1, res2)

    vref_net = Net("vref")
    vout = Net("vout")

    circuit.connect(v_ref.ports[0], vref_net)
    circuit.connect(v_ref.ports[1], GND)
    circuit.connect(res2.ports[0], vref_net)
    circuit.connect(res2.ports[1], GND)
    circuit.connect(res1.ports[0], vout)
    circuit.connect(res1.ports[1], vref_net)

    return circuit


def calculate_lm317_output(r1: float, r2: float, vref: float = 1.25) -> float:
    """Calculate LM317 output: Vout = Vref × (1 + R1/R2) + Iadj × R1."""
    # Ignoring Iadj term (typically 50µA, small contribution)
    return vref * (1 + r1 / r2)


def simulate_regulator(
    r1_nom: float,
    r2_nom: float,
    r1_tol: float,
    r2_tol: float,
    vref_nom: float,
    vref_tol: float,
    n_runs: int,
) -> list[float]:
    """Simulate adjustable regulator output."""
    results = []
    for _ in range(n_runs):
        r1 = r1_nom * (1 + random.gauss(0, r1_tol / 3))
        r2 = r2_nom * (1 + random.gauss(0, r2_tol / 3))
        vref = vref_nom * (1 + random.gauss(0, vref_tol / 3))
        vout = calculate_lm317_output(r1, r2, vref)
        results.append(vout)
    return results


def stats(data: list[float]) -> dict:
    """Calculate statistics."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return {"mean": mean, "std": std, "min": min(data), "max": max(data)}


def main():
    """Demonstrate power supply analysis."""
    print("=" * 60)
    print("Monte Carlo: Power Supply Output Distribution")
    print("=" * 60)

    random.seed(42)
    n_runs = 10000

    # LM317 adjustable regulator design for 5V output
    # Vout = 1.25 × (1 + R1/R2)
    # For 5V: R1/R2 = (5/1.25) - 1 = 3
    # Using R2 = 240Ω (typical), R1 = 720Ω

    vout_target = 5.0
    vref_nom = 1.25
    r2_nom = 240
    r1_nom = r2_nom * (vout_target / vref_nom - 1)

    vout_nom = calculate_lm317_output(r1_nom, r2_nom, vref_nom)
    circuit = build_adjustable_regulator(r1_nom, r2_nom)

    print(f"""
   Adjustable Regulator (LM317) Output Analysis

   Design: Vout = Vref × (1 + R1/R2)
   Target: {vout_target}V
   Nominal: Vref = {vref_nom}V, R1 = {r1_nom:.0f}Ω, R2 = {r2_nom}Ω
   Calculated Vout = {vout_nom:.3f}V
""")

    # Test different resistor tolerances
    scenarios = [
        ("1% R, 4% Vref", 0.01, 0.01, 0.04),
        ("5% R, 4% Vref", 0.05, 0.05, 0.04),
        ("1% matched R, 4% Vref", 0.01, 0.01, 0.04),
        ("0.1% R, 4% Vref", 0.001, 0.001, 0.04),
    ]

    print("   Output voltage variation:")
    print("   " + "-" * 55)

    for name, r1_tol, r2_tol, vref_t in scenarios:
        results = simulate_regulator(r1_nom, r2_nom, r1_tol, r2_tol, vref_nom, vref_t, n_runs)
        s = stats(results)
        error_pct = (s["max"] - s["min"]) / vout_target * 100
        print(f"   {name:25s}: {s['mean']:.3f}V ± {s['std']*1000:.1f}mV (±{error_pct/2:.1f}%)")

    print("""
   Key Observations:
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Reference tolerance dominates output accuracy           │
   │ 2. Resistor ratio (not absolute) determines Vout           │
   │ 3. Matched/ratiometric resistors improve accuracy          │
   │ 4. LDO with integrated divider (e.g., LM7805) is simpler   │
   └─────────────────────────────────────────────────────────────┘

   Design Tips:
   - Use 1% or better resistors for the feedback network
   - Consider fixed-output regulators when possible
   - For precision: use external precision reference
   - Temperature effects add ~50-100ppm/°C to resistor error

   Error Budget (5V output):
   - Vref ±4%: ±200mV
   - R ratio ±2%: ±75mV (since Vout ∝ (1 + R1/R2) and ratio adds)
   - Total RSS: ~215mV typical
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
