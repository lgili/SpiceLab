"""Worst-Case Corner Analysis

Combines Monte Carlo with worst-case corner analysis to find
the extreme operating conditions of a circuit.

Run: python examples/monte_carlo/worst_case_corners.py
"""

import itertools
import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def apply_tolerance(nominal: float, tolerance: float) -> float:
    """Apply random tolerance."""
    return nominal * (1 + random.uniform(-tolerance, tolerance))


def build_divider(r1: float, r2: float, vin: float) -> Circuit:
    """Build voltage divider circuit."""
    circuit = Circuit("voltage_divider")

    v_in = Vdc("in", vin)
    res1 = Resistor("1", resistance=r1)
    res2 = Resistor("2", resistance=r2)

    circuit.add(v_in, res1, res2)

    v_input = Net("vin")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], v_input)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(res1.ports[0], v_input)
    circuit.connect(res1.ports[1], vout)
    circuit.connect(res2.ports[0], vout)
    circuit.connect(res2.ports[1], GND)

    return circuit


def calculate_output(r1: float, r2: float, vin: float) -> float:
    """Calculate divider output."""
    return vin * r2 / (r1 + r2)


def worst_case_corners(
    r1_nom: float,
    r2_nom: float,
    vin_nom: float,
    r1_tol: float,
    r2_tol: float,
    vin_tol: float,
) -> dict:
    """Evaluate all worst-case corners.

    Each component can be at +tol or -tol, giving 2^n combinations.
    """
    corners = []

    # Generate all combinations of + and - tolerance
    for r1_sign, r2_sign, vin_sign in itertools.product([-1, 1], repeat=3):
        r1 = r1_nom * (1 + r1_sign * r1_tol)
        r2 = r2_nom * (1 + r2_sign * r2_tol)
        vin = vin_nom * (1 + vin_sign * vin_tol)
        vout = calculate_output(r1, r2, vin)

        r1_str = "+" if r1_sign > 0 else "-"
        r2_str = "+" if r2_sign > 0 else "-"
        vin_str = "+" if vin_sign > 0 else "-"
        corner_name = f"R1{r1_str}, R2{r2_str}, Vin{vin_str}"
        corners.append({"name": corner_name, "vout": vout, "r1": r1, "r2": r2, "vin": vin})

    # Sort by output
    corners.sort(key=lambda x: x["vout"])

    return {
        "all_corners": corners,
        "min": corners[0],
        "max": corners[-1],
        "nominal": calculate_output(r1_nom, r2_nom, vin_nom),
    }


def monte_carlo_vs_corners(
    r1_nom: float,
    r2_nom: float,
    vin_nom: float,
    r1_tol: float,
    r2_tol: float,
    vin_tol: float,
    n_runs: int,
) -> dict:
    """Compare Monte Carlo results with worst-case corners."""
    # Monte Carlo
    mc_results = []
    for _ in range(n_runs):
        r1 = apply_tolerance(r1_nom, r1_tol)
        r2 = apply_tolerance(r2_nom, r2_tol)
        vin = apply_tolerance(vin_nom, vin_tol)
        vout = calculate_output(r1, r2, vin)
        mc_results.append(vout)

    mc_min = min(mc_results)
    mc_max = max(mc_results)
    mc_mean = sum(mc_results) / n_runs

    # Worst-case corners
    corners = worst_case_corners(r1_nom, r2_nom, vin_nom, r1_tol, r2_tol, vin_tol)

    return {
        "mc_min": mc_min,
        "mc_max": mc_max,
        "mc_mean": mc_mean,
        "wc_min": corners["min"]["vout"],
        "wc_max": corners["max"]["vout"],
        "nominal": corners["nominal"],
    }


def main():
    """Demonstrate worst-case corner analysis."""
    print("=" * 60)
    print("Monte Carlo: Worst-Case Corner Analysis")
    print("=" * 60)

    random.seed(42)
    n_runs = 100000

    # Design parameters
    vin_nom = 5.0
    r1_nom = 10_000
    r2_nom = 10_000
    vout_nom = calculate_output(r1_nom, r2_nom, vin_nom)

    circuit = build_divider(r1_nom, r2_nom, vin_nom)

    print(f"""
   Worst-Case vs Monte Carlo Analysis

   Circuit: Voltage Divider
   Nominal: Vin = {vin_nom}V, R1 = R2 = {r1_nom/1000:.0f}kΩ
   Nominal Vout = {vout_nom:.2f}V
""")

    # Analyze with different tolerances
    scenarios = [
        ("5% R, 2% Vin", 0.05, 0.05, 0.02),
        ("10% R, 5% Vin", 0.10, 0.10, 0.05),
        ("1% R, 1% Vin", 0.01, 0.01, 0.01),
    ]

    for name, r1_tol, r2_tol, vin_tol in scenarios:
        result = monte_carlo_vs_corners(r1_nom, r2_nom, vin_nom, r1_tol, r2_tol, vin_tol, n_runs)

        print(f"   {name}:")
        print(f"   {'Method':20s} {'Min':>10s} {'Max':>10s} {'Range':>10s}")
        print("   " + "-" * 52)
        mc_range = result["mc_max"] - result["mc_min"]
        wc_range = result["wc_max"] - result["wc_min"]
        print(
            f"   {'Monte Carlo':20s} {result['mc_min']:10.4f} "
            f"{result['mc_max']:10.4f} {mc_range:10.4f}"
        )
        print(
            f"   {'Worst-Case Corners':20s} {result['wc_min']:10.4f} "
            f"{result['wc_max']:10.4f} {wc_range:10.4f}"
        )
        print(f"   {'Ratio (MC/WC)':20s} {mc_range/wc_range*100:>9.1f}%")
        print()

    # Show all corners for one case
    print("   Detailed Corner Analysis (5% R, 2% Vin):")
    print("   " + "-" * 45)
    corners = worst_case_corners(r1_nom, r2_nom, vin_nom, 0.05, 0.05, 0.02)

    for c in corners["all_corners"]:
        print(f"   {c['name']:18s}: Vout = {c['vout']:.4f} V")

    print("""
   Key Insights:
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. Worst-case gives absolute bounds (all tolerances at max)│
   │ 2. Monte Carlo shows statistical distribution              │
   │ 3. MC range is typically 60-80% of WC range               │
   │ 4. WC is conservative - probability of hitting corners low │
   │ 5. Use WC for safety-critical, MC for typical performance  │
   └─────────────────────────────────────────────────────────────┘

   When to Use Each Method:
   - Worst-Case: Safety limits, guaranteed specs, mil-spec
   - Monte Carlo: Yield estimation, typical performance
   - Combined: Design margin analysis (WC bounds, MC distribution)

   Corner Naming Convention:
   R1+, R2-, Vin+ means:
   - R1 at positive tolerance limit
   - R2 at negative tolerance limit
   - Vin at positive tolerance limit
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
