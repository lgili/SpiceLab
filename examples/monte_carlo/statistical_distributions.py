"""Statistical Distributions for Monte Carlo

Compares different statistical distributions for modeling
component tolerances and their effect on simulation results.

Run: python examples/monte_carlo/statistical_distributions.py
"""

import random

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def uniform(nominal: float, tolerance: float) -> float:
    """Uniform distribution within ±tolerance."""
    return nominal * (1 + random.uniform(-tolerance, tolerance))


def gaussian(nominal: float, tolerance: float) -> float:
    """Gaussian with 3σ = tolerance (99.7% within bounds)."""
    sigma = tolerance / 3
    return nominal * (1 + random.gauss(0, sigma))


def truncated_gaussian(nominal: float, tolerance: float) -> float:
    """Gaussian clipped to ±tolerance bounds."""
    sigma = tolerance / 3
    variation = random.gauss(0, sigma)
    variation = max(-tolerance, min(tolerance, variation))
    return nominal * (1 + variation)


def triangular(nominal: float, tolerance: float) -> float:
    """Triangular distribution centered at nominal."""
    # Mode at 0 (center)
    return nominal * (1 + random.triangular(-tolerance, tolerance, 0))


def beta_distribution(
    nominal: float, tolerance: float, alpha: float = 2, beta_param: float = 2
) -> float:
    """Beta distribution (U-shaped when α=β<1, bell when α=β>1)."""
    # Map beta(0,1) to (-tolerance, +tolerance)
    x = random.betavariate(alpha, beta_param)
    return nominal * (1 + (2 * x - 1) * tolerance)


def build_circuit(r1: float, r2: float) -> Circuit:
    """Build test circuit."""
    circuit = Circuit("test")
    v_in = Vdc("in", 5.0)
    res1 = Resistor("1", resistance=r1)
    res2 = Resistor("2", resistance=r2)
    circuit.add(v_in, res1, res2)

    vin = Net("vin")
    vout = Net("vout")
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(res1.ports[0], vin)
    circuit.connect(res1.ports[1], vout)
    circuit.connect(res2.ports[0], vout)
    circuit.connect(res2.ports[1], GND)

    return circuit


def calculate_vout(r1: float, r2: float, vin: float = 5.0) -> float:
    """Calculate output voltage."""
    return vin * r2 / (r1 + r2)


def run_distribution_comparison(
    r1_nom: float, r2_nom: float, tolerance: float, n_runs: int
) -> dict:
    """Compare different distributions."""
    distributions = {
        "Uniform": uniform,
        "Gaussian (3σ)": gaussian,
        "Trunc. Gaussian": truncated_gaussian,
        "Triangular": triangular,
        "Beta (2,2)": lambda n, t: beta_distribution(n, t, 2, 2),
    }

    results = {}

    for name, dist_func in distributions.items():
        vout_values = []
        for _ in range(n_runs):
            r1 = dist_func(r1_nom, tolerance)
            r2 = dist_func(r2_nom, tolerance)
            vout = calculate_vout(r1, r2)
            vout_values.append(vout)

        mean = sum(vout_values) / n_runs
        std = (sum((x - mean) ** 2 for x in vout_values) / n_runs) ** 0.5
        min_val = min(vout_values)
        max_val = max(vout_values)

        # Count values in different ranges
        within_1sigma = sum(1 for v in vout_values if abs(v - mean) <= std) / n_runs
        within_2sigma = sum(1 for v in vout_values if abs(v - mean) <= 2 * std) / n_runs
        within_3sigma = sum(1 for v in vout_values if abs(v - mean) <= 3 * std) / n_runs

        results[name] = {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val,
            "within_1sigma": within_1sigma * 100,
            "within_2sigma": within_2sigma * 100,
            "within_3sigma": within_3sigma * 100,
        }

    return results


def main():
    """Demonstrate statistical distribution comparison."""
    print("=" * 60)
    print("Monte Carlo: Statistical Distributions Comparison")
    print("=" * 60)

    random.seed(42)
    n_runs = 100000

    r1_nom = 10_000
    r2_nom = 10_000
    tolerance = 0.05  # 5%

    circuit = build_circuit(r1_nom, r2_nom)
    vout_nom = calculate_vout(r1_nom, r2_nom)

    print(f"""
   Comparing Distribution Effects on Monte Carlo Results

   Circuit: Voltage Divider (R1 = R2 = {r1_nom/1000:.0f}kΩ, ±{tolerance*100:.0f}%)
   Nominal Vout = {vout_nom:.2f} V
   Simulations: {n_runs:,}
""")

    results = run_distribution_comparison(r1_nom, r2_nom, tolerance, n_runs)

    # Summary table
    print("   Distribution Comparison:")
    print("   " + "-" * 70)
    print(f"   {'Distribution':<18} {'Mean':>8} {'Std':>10} {'Range':>10} {'Within 3σ':>10}")
    print("   " + "-" * 70)

    for name, r in results.items():
        print(
            f"   {name:<18} {r['mean']:.4f}V {r['std']*1000:.2f}mV "
            f"{r['range']*1000:.2f}mV {r['within_3sigma']:>8.1f}%"
        )

    print("""
   Distribution Characteristics:
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Distribution     │ Shape          │ Best For                       │
   ├──────────────────┼────────────────┼────────────────────────────────┤
   │ Uniform          │ Flat           │ Unknown distribution, WC test  │
   │ Gaussian         │ Bell curve     │ Natural variations, many mfg   │
   │ Trunc. Gaussian  │ Clipped bell   │ Binned/screened components     │
   │ Triangular       │ Triangle       │ Simple approximation           │
   │ Beta             │ Adjustable     │ Custom shapes, real measured   │
   └──────────────────┴────────────────┴────────────────────────────────┘

   Gaussian Properties (Ideal):
   - 68.3% within ±1σ
   - 95.4% within ±2σ
   - 99.7% within ±3σ

   Manufacturing Reality:
   - Components often screened (truncated at spec limits)
   - Distribution may not be perfectly Gaussian
   - Measure actual distribution from sample batches if critical
   - Uniform is most conservative (highest probability at extremes)

   Recommendation:
   - Use Gaussian for typical analysis
   - Use Uniform for conservative/worst-case
   - Measure real distribution for high-volume/critical products
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
