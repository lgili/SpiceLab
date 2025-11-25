"""Filter Frequency Response with Tolerances

Demonstrates how component tolerances affect filter cutoff
frequency and Q factor in an RC and LC filter.

Run: python examples/monte_carlo/filter_variations.py
"""

import math
import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vac
from spicelab.core.net import GND, Net


def apply_tolerance(nominal: float, tolerance: float) -> float:
    """Apply random tolerance to a value."""
    return nominal * (1 + random.uniform(-tolerance, tolerance))


def build_rc_lowpass(r_value: float, c_value: float) -> Circuit:
    """Build an RC lowpass filter."""
    circuit = Circuit("rc_lowpass")

    v_in = Vac("in", ac_mag=1.0)
    r1 = Resistor("1", resistance=r_value)
    c1 = Capacitor("1", capacitance=c_value)

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


def build_rlc_bandpass(r: float, ind: float, c: float) -> Circuit:
    """Build a series RLC bandpass filter."""
    circuit = Circuit("rlc_bandpass")

    v_in = Vac("in", ac_mag=1.0)
    r1 = Resistor("1", resistance=r)
    l1 = Inductor("1", inductance=ind)
    c1 = Capacitor("1", capacitance=c)
    r_load = Resistor("load", resistance=r)  # Matched load

    circuit.add(v_in, r1, l1, c1, r_load)

    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Series RLC
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v1)
    circuit.connect(l1.ports[0], v1)
    circuit.connect(l1.ports[1], v2)
    circuit.connect(c1.ports[0], v2)
    circuit.connect(c1.ports[1], vout)

    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit


def calculate_rc_cutoff(r: float, c: float) -> float:
    """Calculate RC filter cutoff frequency."""
    return 1 / (2 * math.pi * r * c)


def calculate_rlc_params(r: float, ind: float, c: float) -> tuple[float, float]:
    """Calculate RLC filter center frequency and Q."""
    f0 = 1 / (2 * math.pi * math.sqrt(ind * c))
    Q = (1 / r) * math.sqrt(ind / c)
    return f0, Q


def run_rc_monte_carlo(
    r_nom: float,
    c_nom: float,
    r_tol: float,
    c_tol: float,
    n_runs: int,
) -> list[float]:
    """Run Monte Carlo for RC filter cutoff frequency."""
    results = []
    for _ in range(n_runs):
        r = apply_tolerance(r_nom, r_tol)
        c = apply_tolerance(c_nom, c_tol)
        fc = calculate_rc_cutoff(r, c)
        results.append(fc)
    return results


def run_rlc_monte_carlo(
    r_nom: float,
    l_nom: float,
    c_nom: float,
    r_tol: float,
    l_tol: float,
    c_tol: float,
    n_runs: int,
) -> tuple[list[float], list[float]]:
    """Run Monte Carlo for RLC filter center frequency and Q."""
    f0_results = []
    q_results = []
    for _ in range(n_runs):
        r = apply_tolerance(r_nom, r_tol)
        ind = apply_tolerance(l_nom, l_tol)
        c = apply_tolerance(c_nom, c_tol)
        f0, Q = calculate_rlc_params(r, ind, c)
        f0_results.append(f0)
        q_results.append(Q)
    return f0_results, q_results


def stats(data: list[float]) -> dict:
    """Calculate basic statistics."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
    return {
        "mean": mean,
        "std": std,
        "min": min(data),
        "max": max(data),
        "range_pct": (max(data) - min(data)) / mean * 100,
    }


def main():
    """Demonstrate filter variation analysis."""
    print("=" * 60)
    print("Monte Carlo: Filter Frequency Response Variations")
    print("=" * 60)

    random.seed(42)
    n_runs = 10000

    # RC Lowpass Filter Analysis
    print("\n1. RC Lowpass Filter")
    print("-" * 40)

    r_nom = 10_000  # 10kΩ
    c_nom = 15.9e-9  # ~15.9nF for 1kHz cutoff
    fc_nom = calculate_rc_cutoff(r_nom, c_nom)

    circuit_rc = build_rc_lowpass(r_nom, c_nom)

    print(f"""
   Circuit: Vin ──[R]──┬── Vout
                       │
                      [C]
                       │
                      GND

   Nominal: R = {r_nom/1000:.0f}kΩ, C = {c_nom*1e9:.1f}nF
   Nominal fc = {fc_nom:.1f} Hz
""")

    # Test different tolerance combinations
    scenarios = [
        ("1% R, 5% C", 0.01, 0.05),
        ("1% R, 10% C", 0.01, 0.10),
        ("5% R, 5% C", 0.05, 0.05),
        ("5% R, 10% C", 0.05, 0.10),
    ]

    print("   fc variation with component tolerances:")
    print("   " + "-" * 50)

    for name, r_tol, c_tol in scenarios:
        results = run_rc_monte_carlo(r_nom, c_nom, r_tol, c_tol, n_runs)
        s = stats(results)
        print(
            f"   {name:15s}: fc = {s['mean']:.1f} ± {s['std']:.1f} Hz "
            f"(±{s['range_pct']/2:.1f}%)"
        )

    # RLC Bandpass Filter Analysis
    print("\n2. RLC Bandpass Filter")
    print("-" * 40)

    r_nom = 50  # 50Ω
    l_nom = 100e-6  # 100µH
    c_nom = 253e-12  # ~253pF for 1MHz
    f0_nom, Q_nom = calculate_rlc_params(r_nom, l_nom, c_nom)

    circuit_rlc = build_rlc_bandpass(r_nom, l_nom, c_nom)

    print(f"""
   Circuit: Vin ──[R]──[L]──[C]──┬── Vout
                                 │
                               [Rload]
                                 │
                                GND

   Nominal: R = {r_nom}Ω, L = {l_nom*1e6:.0f}µH, C = {c_nom*1e12:.0f}pF
   Nominal f0 = {f0_nom/1e6:.3f} MHz, Q = {Q_nom:.1f}
""")

    # RLC tolerance scenarios
    scenarios_rlc = [
        ("1% all", 0.01, 0.01, 0.01),
        ("5% all", 0.05, 0.05, 0.05),
        ("1% R, 5% L, 10% C", 0.01, 0.05, 0.10),
        ("5% R, 10% L, 10% C", 0.05, 0.10, 0.10),
    ]

    print("   f0 and Q variation:")
    print("   " + "-" * 55)

    for name, r_tol, l_tol, c_tol in scenarios_rlc:
        f0_results, q_results = run_rlc_monte_carlo(
            r_nom, l_nom, c_nom, r_tol, l_tol, c_tol, n_runs
        )
        s_f0 = stats(f0_results)
        s_q = stats(q_results)
        print(
            f"   {name:20s}: f0 = {s_f0['mean']/1e6:.3f}±{s_f0['std']/1e3:.1f}kHz, "
            f"Q = {s_q['mean']:.1f}±{s_q['std']:.2f}"
        )

    print("""
   Key Observations:
   ┌─────────────────────────────────────────────────────────────┐
   │ RC Filter:                                                  │
   │ - fc varies as 1/(R×C), so tolerances add                   │
   │ - 1% R + 5% C ≈ 6% fc variation (RSS: ~5.1%)                │
   │ - Capacitor tolerance usually dominates                     │
   │                                                             │
   │ RLC Filter:                                                 │
   │ - f0 ∝ 1/√(LC), so tolerance effects are halved             │
   │ - Q is sensitive to R (directly proportional)               │
   │ - Use precision R for tight Q control                       │
   │                                                             │
   │ Mitigation:                                                 │
   │ - Use tighter tolerance capacitors (NPO/C0G for RF)         │
   │ - Match ratios instead of absolutes where possible          │
   │ - Consider active filters for critical applications         │
   └─────────────────────────────────────────────────────────────┘
""")

    result_rc = circuit_rc.validate()
    result_rlc = circuit_rlc.validate()
    print(f"   RC Filter Validation: {'VALID' if result_rc.is_valid else 'INVALID'}")
    print(f"   RLC Filter Validation: {'VALID' if result_rlc.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
