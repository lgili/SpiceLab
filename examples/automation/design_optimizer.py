"""Design Optimization Workflow

Demonstrates automated design optimization to find component
values that meet specified targets and constraints.

Run: python examples/automation/design_optimizer.py
"""

import random
from dataclasses import dataclass

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


@dataclass
class DesignSpec:
    """Design specification with targets and constraints."""

    name: str
    target: float
    tolerance: float  # As fraction (0.01 = ±1%)
    weight: float = 1.0  # Importance weight for optimization


@dataclass
class ComponentConstraint:
    """Constraint on component values."""

    name: str
    min_value: float
    max_value: float
    preferred_values: list[float] | None = None  # E-series values


@dataclass
class OptimizationResult:
    """Result of optimization."""

    best_params: dict[str, float]
    best_score: float
    specs_met: dict[str, bool]
    iterations: int


# Standard E24 resistor series (common 5% values)
E24_SERIES = [
    1.0,
    1.1,
    1.2,
    1.3,
    1.5,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.7,
    3.0,
    3.3,
    3.6,
    3.9,
    4.3,
    4.7,
    5.1,
    5.6,
    6.2,
    6.8,
    7.5,
    8.2,
    9.1,
]


def get_nearest_e24(value: float) -> float:
    """Find nearest E24 standard value."""
    if value <= 0:
        return E24_SERIES[0]

    # Find decade multiplier
    decade = 1
    while value >= 10:
        value /= 10
        decade *= 10
    while value < 1:
        value *= 10
        decade /= 10

    # Find nearest E24 value
    nearest = min(E24_SERIES, key=lambda x: abs(x - value))
    return nearest * decade


def build_voltage_divider(r1: float, r2: float, vin: float = 12.0) -> Circuit:
    """Build a voltage divider circuit."""
    circuit = Circuit("voltage_divider")

    v_in = Vdc("in", vin)
    res1 = Resistor("1", resistance=r1)
    res2 = Resistor("2", resistance=r2)

    circuit.add(v_in, res1, res2)

    vin_net = Net("vin")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin_net)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(res1.ports[0], vin_net)
    circuit.connect(res1.ports[1], vout)
    circuit.connect(res2.ports[0], vout)
    circuit.connect(res2.ports[1], GND)

    return circuit


def calculate_divider(r1: float, r2: float, vin: float = 12.0) -> dict:
    """Calculate voltage divider outputs."""
    vout = vin * r2 / (r1 + r2)
    current = vin / (r1 + r2)
    power_r1 = current**2 * r1
    power_r2 = current**2 * r2
    impedance = r1 * r2 / (r1 + r2)  # Thevenin output impedance

    return {
        "vout": vout,
        "current": current,
        "power_r1": power_r1,
        "power_r2": power_r2,
        "power_total": power_r1 + power_r2,
        "output_impedance": impedance,
    }


class DesignOptimizer:
    """Optimizer for circuit design."""

    def __init__(self):
        self.specs: list[DesignSpec] = []
        self.constraints: dict[str, ComponentConstraint] = {}

    def add_spec(self, spec: DesignSpec):
        """Add a design specification."""
        self.specs.append(spec)

    def add_constraint(self, constraint: ComponentConstraint):
        """Add a component constraint."""
        self.constraints[constraint.name] = constraint

    def evaluate(self, params: dict[str, float], eval_func) -> tuple[float, dict]:
        """Evaluate a design against specifications."""
        # Calculate outputs
        outputs = eval_func(**params)

        # Calculate score (lower is better)
        score = 0.0
        specs_met = {}

        for spec in self.specs:
            if spec.name in outputs:
                actual = outputs[spec.name]
                error = abs(actual - spec.target) / spec.target
                specs_met[spec.name] = error <= spec.tolerance

                # Weighted penalty for not meeting spec
                if error > spec.tolerance:
                    score += spec.weight * (error - spec.tolerance) * 100
                else:
                    score += spec.weight * error

        return score, specs_met

    def optimize_grid_search(
        self,
        param_ranges: dict[str, tuple[float, float, int]],
        eval_func,
        use_standard_values: bool = False,
    ) -> OptimizationResult:
        """Grid search optimization."""
        best_params = {}
        best_score = float("inf")
        best_specs_met = {}
        iterations = 0

        # Generate grid points
        param_values = {}
        for name, (start, stop, num_points) in param_ranges.items():
            step = (stop - start) / (num_points - 1)
            values = [start + i * step for i in range(num_points)]

            if use_standard_values and name in self.constraints:
                # Snap to nearest standard values
                values = [get_nearest_e24(v) for v in values]
                values = list(set(values))  # Remove duplicates

            param_values[name] = values

        # Grid search
        def recursive_search(params, remaining_names):
            nonlocal best_params, best_score, best_specs_met, iterations

            if not remaining_names:
                iterations += 1
                score, specs_met = self.evaluate(params, eval_func)
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    best_specs_met = specs_met.copy()
                return

            name = remaining_names[0]
            for value in param_values[name]:
                params[name] = value
                recursive_search(params, remaining_names[1:])

        recursive_search({}, list(param_ranges.keys()))

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            specs_met=best_specs_met,
            iterations=iterations,
        )

    def optimize_random_search(
        self,
        param_ranges: dict[str, tuple[float, float]],
        eval_func,
        max_iterations: int = 1000,
        use_standard_values: bool = False,
    ) -> OptimizationResult:
        """Random search optimization."""
        best_params = {}
        best_score = float("inf")
        best_specs_met = {}

        for _ in range(max_iterations):
            # Generate random parameters
            params = {}
            for name, (min_val, max_val) in param_ranges.items():
                value = random.uniform(min_val, max_val)
                if use_standard_values:
                    value = get_nearest_e24(value)
                params[name] = value

            score, specs_met = self.evaluate(params, eval_func)
            if score < best_score:
                best_score = score
                best_params = params.copy()
                best_specs_met = specs_met.copy()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            specs_met=best_specs_met,
            iterations=max_iterations,
        )


def main():
    """Demonstrate design optimization."""
    print("=" * 60)
    print("Automation: Design Optimizer")
    print("=" * 60)

    random.seed(42)

    # Design problem: Voltage divider
    # From 12V, we need 3.3V output for MCU
    # Constraints: Low power (<1mW), output impedance <1kΩ

    vin = 12.0
    vout_target = 3.3

    print(f"""
   Design Problem: Voltage Divider for MCU Power
   ─────────────────────────────────────────────
   Input: Vin = {vin}V
   Target: Vout = {vout_target}V ±1%
   Constraints:
   - Total power < 1mW
   - Output impedance < 1kΩ
   - Use standard E24 resistor values
""")

    optimizer = DesignOptimizer()

    # Add specifications
    optimizer.add_spec(DesignSpec("vout", vout_target, 0.01, weight=10.0))
    optimizer.add_spec(DesignSpec("power_total", 0.5e-3, 1.0, weight=5.0))
    optimizer.add_spec(DesignSpec("output_impedance", 500, 1.0, weight=2.0))

    # Add constraints
    optimizer.add_constraint(ComponentConstraint("r1", 1_000, 100_000))
    optimizer.add_constraint(ComponentConstraint("r2", 1_000, 100_000))

    # Grid search
    print("   Grid Search Optimization:")
    print("   " + "-" * 50)

    result_grid = optimizer.optimize_grid_search(
        param_ranges={
            "r1": (1_000, 100_000, 20),
            "r2": (1_000, 100_000, 20),
        },
        eval_func=lambda r1, r2: calculate_divider(r1, r2, vin),
        use_standard_values=True,
    )

    r1_opt = result_grid.best_params["r1"]
    r2_opt = result_grid.best_params["r2"]
    outputs = calculate_divider(r1_opt, r2_opt, vin)

    print(f"   Iterations: {result_grid.iterations}")
    print(f"   Best R1: {r1_opt/1000:.2f} kΩ")
    print(f"   Best R2: {r2_opt/1000:.2f} kΩ")
    print(f"   Vout: {outputs['vout']:.3f} V (target: {vout_target}V)")
    print(f"   Power: {outputs['power_total']*1000:.3f} mW")
    print(f"   Output Z: {outputs['output_impedance']:.0f} Ω")
    print(f"   Specs met: {result_grid.specs_met}")

    # Random search for comparison
    print("\n   Random Search Optimization:")
    print("   " + "-" * 50)

    result_random = optimizer.optimize_random_search(
        param_ranges={
            "r1": (1_000, 100_000),
            "r2": (1_000, 100_000),
        },
        eval_func=lambda r1, r2: calculate_divider(r1, r2, vin),
        max_iterations=1000,
        use_standard_values=True,
    )

    r1_rand = result_random.best_params["r1"]
    r2_rand = result_random.best_params["r2"]
    outputs_rand = calculate_divider(r1_rand, r2_rand, vin)

    print(f"   Iterations: {result_random.iterations}")
    print(f"   Best R1: {r1_rand/1000:.2f} kΩ")
    print(f"   Best R2: {r2_rand/1000:.2f} kΩ")
    print(f"   Vout: {outputs_rand['vout']:.3f} V")
    print(f"   Score: {result_random.best_score:.4f}")

    # Build and validate final circuit
    circuit = build_voltage_divider(r1_opt, r2_opt, vin)

    print("""
   Optimization Methods:
   ┌────────────────────────────────────────────────────────┐
   │ Grid Search: Exhaustive, finds global optimum          │
   │ Random Search: Fast, good for high-dimensional spaces  │
   │ Gradient Descent: Fast convergence, needs derivatives  │
   │ Genetic Algorithm: Good for complex constraints        │
   └────────────────────────────────────────────────────────┘

   Standard Value Support:
   - E24 series for 5% resistors
   - E96 series for 1% resistors
   - Snap-to-nearest ensures manufacturable designs
""")

    result = circuit.validate()
    print(f"   Circuit Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
