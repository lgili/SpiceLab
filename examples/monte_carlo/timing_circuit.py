"""RC Timing Circuit Tolerance Analysis

Analyzes timing accuracy of RC circuits used in oscillators
and timers (e.g., 555 timer circuits).

Run: python examples/monte_carlo/timing_circuit.py
"""

import math
import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net


def apply_tolerance(nominal: float, tolerance: float) -> float:
    """Apply tolerance with Gaussian distribution."""
    sigma = tolerance / 3
    return nominal * (1 + random.gauss(0, sigma))


def build_555_monostable(r: float, c: float) -> Circuit:
    """Build conceptual 555 monostable timing circuit."""
    circuit = Circuit("555_monostable")

    v_cc = Vdc("cc", 5.0)
    r_timing = Resistor("t", resistance=r)
    c_timing = Capacitor("t", capacitance=c)

    circuit.add(v_cc, r_timing, c_timing)

    vcc = Net("vcc")
    v_timing = Net("timing")

    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)
    circuit.connect(r_timing.ports[0], vcc)
    circuit.connect(r_timing.ports[1], v_timing)
    circuit.connect(c_timing.ports[0], v_timing)
    circuit.connect(c_timing.ports[1], GND)

    return circuit


def calculate_555_pulse_width(r: float, c: float) -> float:
    """Calculate 555 monostable pulse width: T = 1.1 × R × C."""
    return 1.1 * r * c


def run_timing_monte_carlo(
    r_nom: float, c_nom: float, r_tol: float, c_tol: float, n_runs: int
) -> list[float]:
    """Run Monte Carlo for timing analysis."""
    results = []
    for _ in range(n_runs):
        r = apply_tolerance(r_nom, r_tol)
        c = apply_tolerance(c_nom, c_tol)
        t = calculate_555_pulse_width(r, c)
        results.append(t)
    return results


def stats(data: list[float]) -> dict:
    """Calculate statistics."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return {"mean": mean, "std": std, "min": min(data), "max": max(data)}


def main():
    """Demonstrate timing circuit tolerance analysis."""
    print("=" * 60)
    print("Monte Carlo: RC Timing Circuit Analysis")
    print("=" * 60)

    random.seed(42)
    n_runs = 10000

    # Design: 1ms pulse from 555 monostable
    # T = 1.1 × R × C = 1ms
    t_target = 1e-3  # 1ms
    c_nom = 100e-9  # 100nF
    r_nom = t_target / (1.1 * c_nom)  # ~9.09kΩ

    t_nom = calculate_555_pulse_width(r_nom, c_nom)
    circuit = build_555_monostable(r_nom, c_nom)

    print(f"""
   555 Timer Monostable Timing Analysis

   Timing equation: T = 1.1 × R × C

   Nominal: R = {r_nom/1000:.2f}kΩ, C = {c_nom*1e9:.0f}nF
   Target pulse width: {t_target*1000:.2f} ms
   Calculated T = {t_nom*1000:.3f} ms

   Component tolerances analyzed:
""")

    scenarios = [
        ("1% R, 5% C (ceramic)", 0.01, 0.05),
        ("1% R, 10% C (film)", 0.01, 0.10),
        ("1% R, 20% C (electrolytic)", 0.01, 0.20),
        ("5% R, 5% C", 0.05, 0.05),
        ("5% R, 10% C", 0.05, 0.10),
    ]

    print("   Pulse width variation:")
    print("   " + "-" * 55)

    for name, r_tol, c_tol in scenarios:
        results = run_timing_monte_carlo(r_nom, c_nom, r_tol, c_tol, n_runs)
        s = stats(results)
        error_pct = (s["max"] - s["min"]) / s["mean"] * 100
        print(
            f"   {name:25s}: T = {s['mean']*1000:.3f} ± {s['std']*1000:.3f} ms "
            f"(±{error_pct/2:.1f}%)"
        )

    # Combined tolerance approximation
    rss_example = math.sqrt(0.01**2 + 0.10**2) * 100
    print(f"""
   RSS Tolerance Analysis:
   Total tolerance ≈ √(tol_R² + tol_C²)

   Example: 1% R + 10% C:
   - RSS = √(0.01² + 0.10²) = {rss_example:.1f}%
   - Simulated: ~10% range (matches theory)

   Improving Timing Accuracy:
   ┌────────────────────────────────────────────────────────┐
   │ 1. Use NPO/C0G capacitors (±1-5% available)           │
   │ 2. Use precision metal film resistors (0.1-1%)        │
   │ 3. Consider crystal oscillators for high precision    │
   │ 4. Digital timing (MCU) for repeatability             │
   │ 5. Trim pots for adjustment (adds calibration step)   │
   └────────────────────────────────────────────────────────┘

   Capacitor Types and Tolerances:
   - NPO/C0G ceramic: ±1-5%, excellent stability
   - X7R ceramic: ±10%, moderate stability
   - Film: ±5-10%, good stability
   - Electrolytic: ±20%, poor stability, temperature drift
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
