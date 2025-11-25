"""Amplifier Gain Distribution Analysis

Analyzes how resistor tolerances affect op-amp amplifier gain.

Run: python examples/monte_carlo/amplifier_gain.py
"""

import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def apply_tolerance(nominal: float, tolerance: float) -> float:
    """Apply random tolerance to a value."""
    return nominal * (1 + random.uniform(-tolerance, tolerance))


def build_inverting_amp(rf: float, rin: float) -> Circuit:
    """Build an inverting amplifier."""
    circuit = Circuit("inverting_amp")

    v_in = Vdc("in", 1.0)
    u1 = create_component("opamp.ideal", "1")
    r_in = Resistor("in", resistance=rin)
    r_f = Resistor("f", resistance=rf)

    circuit.add(v_in, u1, r_in, r_f)

    vin = Net("vin")
    v_inv = Net("v_inv")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_in.ports[0], vin)
    circuit.connect(r_in.ports[1], v_inv)
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)
    circuit.connect(r_f.ports[0], v_inv)
    circuit.connect(r_f.ports[1], vout)

    return circuit


def calculate_gain(rf: float, rin: float) -> float:
    """Calculate inverting amplifier gain."""
    return -rf / rin


def run_monte_carlo(
    rf_nom: float,
    rin_nom: float,
    rf_tol: float,
    rin_tol: float,
    n_runs: int,
) -> list[float]:
    """Run Monte Carlo for amplifier gain."""
    results = []
    for _ in range(n_runs):
        rf = apply_tolerance(rf_nom, rf_tol)
        rin = apply_tolerance(rin_nom, rin_tol)
        gain = abs(calculate_gain(rf, rin))
        results.append(gain)
    return results


def stats(data: list[float]) -> dict:
    """Calculate statistics."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return {"mean": mean, "std": std, "min": min(data), "max": max(data)}


def main():
    """Demonstrate amplifier gain Monte Carlo analysis."""
    print("=" * 60)
    print("Monte Carlo: Amplifier Gain Distribution")
    print("=" * 60)

    random.seed(42)
    n_runs = 10000

    # Design: Gain = -10 (inverting amplifier)
    gain_target = 10
    rin_nom = 10_000
    rf_nom = gain_target * rin_nom

    circuit = build_inverting_amp(rf_nom, rin_nom)

    print(f"""
   Inverting Amplifier Gain Analysis

   Circuit:
                  Rf
              ┌──[R]──┐
              │       │
   Vin ──[Rin]┴──[-]──┴── Vout
                 [+]
                  │
                 GND

   Nominal: Rin = {rin_nom/1000:.0f}kΩ, Rf = {rf_nom/1000:.0f}kΩ
   Target Gain = |Rf/Rin| = {gain_target}
""")

    scenarios = [
        ("0.1% matched pair", 0.001, 0.001),
        ("1% standard", 0.01, 0.01),
        ("1% Rin, 5% Rf", 0.01, 0.05),
        ("5% standard", 0.05, 0.05),
    ]

    print("   Gain variation analysis:")
    print("   " + "-" * 50)

    for name, rin_tol, rf_tol in scenarios:
        results = run_monte_carlo(rf_nom, rin_nom, rf_tol, rin_tol, n_runs)
        s = stats(results)
        error_pct = (s["max"] - s["min"]) / s["mean"] * 100
        print(f"   {name:20s}: Gain = {s['mean']:.3f} ± {s['std']:.4f} " f"(±{error_pct/2:.2f}%)")

    print("""
   Key Insights:
   ┌───────────────────────────────────────────────────────────┐
   │ 1. Gain tolerance ≈ √(tol_Rf² + tol_Rin²) [RSS]          │
   │ 2. Using matched/ratio pairs significantly improves gain  │
   │ 3. For precision gain, use 0.1% resistors or laser trim   │
   │ 4. Temperature tracking also matters for precision apps   │
   └───────────────────────────────────────────────────────────┘
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
