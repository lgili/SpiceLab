"""Voltage Reference Accuracy Analysis

Analyzes accuracy of voltage reference circuits including
initial tolerance, temperature effects, and load regulation.

Run: python examples/monte_carlo/voltage_reference.py
"""

import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def build_shunt_reference(vref: float, r_series: float) -> Circuit:
    """Build a shunt voltage reference circuit."""
    circuit = Circuit("shunt_reference")

    v_supply = Vdc("supply", 12.0)
    r_s = Resistor("s", resistance=r_series)
    r_load = Resistor("load", resistance=10_000)

    circuit.add(v_supply, r_s, r_load)

    vsupply = Net("vsupply")
    vout = Net("vout")

    circuit.connect(v_supply.ports[0], vsupply)
    circuit.connect(v_supply.ports[1], GND)
    circuit.connect(r_s.ports[0], vsupply)
    circuit.connect(r_s.ports[1], vout)
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit


def simulate_reference(
    vref_nom: float,
    vref_init_tol: float,  # Initial accuracy
    tempco: float,  # ppm/°C
    temp_delta: float,  # Temperature change from 25°C
    load_reg: float,  # mV/mA load regulation
    load_current: float,  # mA
    n_runs: int,
) -> list[float]:
    """Simulate voltage reference with all error sources."""
    results = []

    for _ in range(n_runs):
        # Initial tolerance (usually specified at 25°C)
        v = vref_nom * (1 + random.gauss(0, vref_init_tol / 3))

        # Temperature variation
        temp_error = tempco * 1e-6 * temp_delta * vref_nom
        v += random.gauss(0, abs(temp_error) / 3)

        # Load regulation error
        load_error = load_reg * load_current * 1e-3
        v += random.gauss(0, abs(load_error) / 3)

        results.append(v)

    return results


def stats(data: list[float]) -> dict:
    """Calculate statistics."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return {"mean": mean, "std": std, "min": min(data), "max": max(data)}


def main():
    """Demonstrate voltage reference analysis."""
    print("=" * 60)
    print("Monte Carlo: Voltage Reference Accuracy Analysis")
    print("=" * 60)

    random.seed(42)
    n_runs = 10000

    # Reference specifications
    vref_nom = 2.5  # 2.5V reference
    vref_init_tol = 0.01  # ±1% initial accuracy
    tempco_typ = 50  # 50 ppm/°C
    load_reg = 0.5  # 0.5 mV/mA

    circuit = build_shunt_reference(vref_nom, 1000)

    print(f"""
   Voltage Reference Error Analysis

   Reference: {vref_nom} V nominal
   Initial Accuracy: ±{vref_init_tol*100:.1f}%
   Temperature Coefficient: {tempco_typ} ppm/°C
   Load Regulation: {load_reg} mV/mA
""")

    # Different operating conditions
    conditions = [
        ("25°C, 0mA (ideal)", 0, 0),
        ("25°C, 10mA load", 0, 10),
        ("0°C to 70°C, 1mA", 45, 1),  # 45°C span from 25°C midpoint
        ("-40°C to 85°C, 10mA", 62.5, 10),  # Industrial range
    ]

    print("   Vout variation under different conditions:")
    print("   " + "-" * 55)

    for name, temp_delta, load in conditions:
        results = simulate_reference(
            vref_nom, vref_init_tol, tempco_typ, temp_delta, load_reg, load, n_runs
        )
        s = stats(results)
        error_pct = (s["max"] - s["min"]) / vref_nom * 100
        print(f"   {name:25s}: {s['mean']:.4f} ± {s['std']*1000:.2f}mV (±{error_pct/2:.2f}%)")

    # Compare reference grades
    print("\n   Reference Grade Comparison (25°C, 1mA, ΔT=50°C):")
    print("   " + "-" * 55)

    grades = [
        ("Economy (±2%, 100ppm)", 0.02, 100),
        ("Standard (±1%, 50ppm)", 0.01, 50),
        ("Precision (±0.5%, 25ppm)", 0.005, 25),
        ("High-Precision (±0.1%, 10ppm)", 0.001, 10),
    ]

    for name, init_tol, tempco in grades:
        results = simulate_reference(vref_nom, init_tol, tempco, 50, 0.5, 1, n_runs)
        s = stats(results)
        error_mv = (s["max"] - s["min"]) * 1000 / 2
        print(f"   {name:30s}: ±{error_mv:.1f}mV")

    print("""
   Error Budget Analysis:
   ┌────────────────────────────────────────────────────────┐
   │ Error Source          │ Calculation                   │
   ├───────────────────────┼───────────────────────────────┤
   │ Initial Accuracy      │ Vref × tolerance              │
   │ Temperature Drift     │ Vref × tempco × ΔT            │
   │ Load Regulation       │ Load_reg × ΔI                 │
   │ Long-term Drift       │ Typically 50-100 ppm/1000hr   │
   │ Line Regulation       │ Typically 0.01-0.1%/V         │
   └───────────────────────┴───────────────────────────────┘

   Total Error (RSS): √(sum of squared individual errors)

   Reference Selection Guidelines:
   - General purpose: Standard grade sufficient
   - Precision ADC/DAC: Precision or better
   - Calibration: High-precision with calibration
   - Battery monitoring: Consider low Iq references
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
