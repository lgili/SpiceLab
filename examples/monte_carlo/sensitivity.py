"""Component Sensitivity Analysis

Determines which components have the greatest impact on
circuit performance using Monte Carlo correlation analysis.

Run: python examples/monte_carlo/sensitivity.py
"""

import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def apply_tolerance(nominal: float, tolerance: float) -> float:
    """Apply random tolerance."""
    return nominal * (1 + random.uniform(-tolerance, tolerance))


def build_sallen_key_lowpass(r1: float, r2: float, c1: float, c2: float) -> Circuit:
    """Build a Sallen-Key lowpass filter."""
    circuit = Circuit("sallen_key_lp")

    v_in = Vdc("in", 1.0)
    u1 = create_component("opamp.ideal", "1")
    res1 = Resistor("1", resistance=r1)
    res2 = Resistor("2", resistance=r2)
    cap1 = Capacitor("1", capacitance=c1)
    cap2 = Capacitor("2", capacitance=c2)

    circuit.add(v_in, u1, res1, res2, cap1, cap2)

    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(res1.ports[0], vin)
    circuit.connect(res1.ports[1], v1)
    circuit.connect(res2.ports[0], v1)
    circuit.connect(res2.ports[1], v2)
    circuit.connect(cap1.ports[0], v1)
    circuit.connect(cap1.ports[1], vout)
    circuit.connect(u1.ports[0], vout)  # Unity gain buffer
    circuit.connect(u1.ports[1], v2)
    circuit.connect(u1.ports[2], vout)
    circuit.connect(cap2.ports[0], v2)
    circuit.connect(cap2.ports[1], GND)

    return circuit


def calculate_sallen_key_params(r1: float, r2: float, c1: float, c2: float) -> tuple[float, float]:
    """Calculate Sallen-Key fc and Q."""
    import math

    fc = 1 / (2 * math.pi * math.sqrt(r1 * r2 * c1 * c2))
    Q = math.sqrt(r1 * r2 * c1 * c2) / (c2 * (r1 + r2))
    return fc, Q


def run_sensitivity_analysis(
    r1_nom: float,
    r2_nom: float,
    c1_nom: float,
    c2_nom: float,
    tolerance: float,
    n_runs: int,
) -> dict:
    """Run sensitivity analysis Monte Carlo."""
    # Store component values and outputs
    r1_vals, r2_vals, c1_vals, c2_vals = [], [], [], []
    fc_vals, q_vals = [], []

    for _ in range(n_runs):
        r1 = apply_tolerance(r1_nom, tolerance)
        r2 = apply_tolerance(r2_nom, tolerance)
        c1 = apply_tolerance(c1_nom, tolerance)
        c2 = apply_tolerance(c2_nom, tolerance)

        r1_vals.append(r1)
        r2_vals.append(r2)
        c1_vals.append(c1)
        c2_vals.append(c2)

        fc, Q = calculate_sallen_key_params(r1, r2, c1, c2)
        fc_vals.append(fc)
        q_vals.append(Q)

    # Calculate correlations
    def correlation(x: list[float], y: list[float]) -> float:
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True)) / n
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
        return cov / (std_x * std_y) if std_x * std_y > 0 else 0

    return {
        "fc_corr": {
            "R1": correlation(r1_vals, fc_vals),
            "R2": correlation(r2_vals, fc_vals),
            "C1": correlation(c1_vals, fc_vals),
            "C2": correlation(c2_vals, fc_vals),
        },
        "q_corr": {
            "R1": correlation(r1_vals, q_vals),
            "R2": correlation(r2_vals, q_vals),
            "C1": correlation(c1_vals, q_vals),
            "C2": correlation(c2_vals, q_vals),
        },
        "fc_mean": sum(fc_vals) / n_runs,
        "q_mean": sum(q_vals) / n_runs,
    }


def main():
    """Demonstrate sensitivity analysis."""
    print("=" * 60)
    print("Monte Carlo: Component Sensitivity Analysis")
    print("=" * 60)

    random.seed(42)
    n_runs = 10000

    # Sallen-Key lowpass filter design (1kHz, Q=0.707 Butterworth)
    import math

    fc_target = 1000
    # Target Q = 0.707 for Butterworth response

    # For unity-gain Sallen-Key: fc = 1/(2π√(R1R2C1C2)), Q = √(R1R2C1C2)/(C2(R1+R2))
    # Simplify: R1 = R2 = R, C1 = 2C, C2 = C for Q = 0.707
    R = 10_000
    C = 1 / (2 * math.pi * fc_target * R * math.sqrt(2))  # Approximate

    r1_nom = R
    r2_nom = R
    c1_nom = 2 * C
    c2_nom = C

    fc_nom, Q_nom = calculate_sallen_key_params(r1_nom, r2_nom, c1_nom, c2_nom)

    circuit = build_sallen_key_lowpass(r1_nom, r2_nom, c1_nom, c2_nom)

    print(f"""
   Sallen-Key Lowpass Filter Sensitivity

   Circuit:
   Vin ──[R1]──┬──[R2]──┬──[+]──┬── Vout
               │        │  [U1] │
              [C1]     [C2] [-]─┘
               │        │
              Vout     GND

   Nominal: R1 = R2 = {r1_nom/1000:.0f}kΩ
            C1 = {c1_nom*1e9:.2f}nF, C2 = {c2_nom*1e9:.2f}nF
   Nominal fc = {fc_nom:.1f} Hz, Q = {Q_nom:.3f}
""")

    # Run sensitivity analysis
    result = run_sensitivity_analysis(r1_nom, r2_nom, c1_nom, c2_nom, 0.05, n_runs)

    print("   Sensitivity to fc (correlation coefficients):")
    print("   " + "-" * 40)
    for comp, corr in sorted(result["fc_corr"].items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"   {comp:4s}: {sign}{abs(corr):.3f} {bar}")

    print("\n   Sensitivity to Q (correlation coefficients):")
    print("   " + "-" * 40)
    for comp, corr in sorted(result["q_corr"].items(), key=lambda x: abs(x[1]), reverse=True):
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr > 0 else "-"
        print(f"   {comp:4s}: {sign}{abs(corr):.3f} {bar}")

    print("""
   Interpretation:
   - Correlation ≈ -0.5 for all on fc: Each component contributes
     equally to fc (fc ∝ 1/√(R1×R2×C1×C2))
   - C2 has largest effect on Q: Q ∝ 1/C2
   - Negative correlation: increasing component decreases output

   Design Recommendations:
   1. For tight fc control: Use precision C (NPO/C0G ceramic)
   2. For tight Q control: Use precision C2
   3. For cost optimization: Use standard R (less sensitive)

   Sensitivity Formula (analytical):
   ∂fc/∂R1 × R1/fc = -0.5 (each component has -0.5 sensitivity)
   ∂Q/∂C2 × C2/Q ≈ -0.5 to -1.0 depending on design
""")

    result_val = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result_val.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
