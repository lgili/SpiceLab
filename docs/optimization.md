# Optimization & Design of Experiments

SpiceLab provides comprehensive optimization capabilities for circuit design, including:

- **Parameter optimization** with scipy (Nelder-Mead, L-BFGS-B, Differential Evolution)
- **Multi-objective optimization** with genetic algorithms (NSGA-II)
- **Design of Experiments** (factorial, LHS, Sobol, CCD, Box-Behnken)
- **Sensitivity analysis** (Morris, Sobol, local gradients, OAT)
- **Corner & PVT analysis** (process-voltage-temperature sweeps)

## Quick Start

```python
from spicelab.optimization import (
    CircuitOptimizer,
    ParameterBounds,
    full_factorial,
    morris_analysis,
    PVTSweep,
)

# Basic parameter optimization
opt = CircuitOptimizer(circuit, simulate_func)
opt.add_parameter("R1", 1e3, 100e3, scale="log")
opt.add_parameter("C1", 1e-12, 1e-9, scale="log")

def objective(params, dataset):
    # Maximize bandwidth (return negative for minimization)
    bw = measure(dataset, "bandwidth", output_node="vout")
    return -bw.value

result = opt.optimize(objective, method="nelder-mead")
print(f"Optimal: R1={result.parameters['R1']:.2f}, C1={result.parameters['C1']:.2e}")
```

## Available Optimizers

### Local Optimizers (scipy)

| Method | Description | Best For |
|--------|-------------|----------|
| `nelder-mead` | Simplex method (gradient-free) | General purpose, noisy functions |
| `powell` | Conjugate direction method | Smooth functions |
| `l-bfgs-b` | Limited-memory BFGS with bounds | Large-scale, smooth functions |

### Global Optimizers (scipy)

| Method | Description | Best For |
|--------|-------------|----------|
| `differential-evolution` | Evolutionary algorithm | Global search, multimodal |
| `dual-annealing` | Simulated annealing variant | Hard global optimization |

### Genetic Algorithms

| Class | Description | Best For |
|-------|-------------|----------|
| `GeneticOptimizer` | Single-objective GA | Discrete/mixed optimization |
| `NSGA2Optimizer` | Multi-objective NSGA-II | Trade-off analysis, Pareto fronts |

## Parameter Bounds

Define optimization parameters with bounds and scaling:

```python
from spicelab.optimization import ParameterBounds

# Linear scale (default)
r_bounds = ParameterBounds("R1", 100, 10000)

# Log scale (for values spanning orders of magnitude)
c_bounds = ParameterBounds("C1", 1e-12, 1e-9, scale="log")

# Custom initial value
r_bounds = ParameterBounds("R1", 100, 10000, initial=1000)
```

## Multi-Start Optimization

Avoid local minima with multiple starting points:

```python
from spicelab.optimization import MultiStartOptimizer, NelderMeadOptimizer

base_optimizer = NelderMeadOptimizer()
multi = MultiStartOptimizer(
    base_optimizer,
    n_starts=10,
    sampling="lhs",  # Latin Hypercube Sampling
    seed=42
)

result = multi.optimize(objective, bounds)
print(f"Best of {result.n_starts} starts: {result.best_result.value}")
print(f"Success rate: {result.success_rate:.1%}")
```

## Multi-Objective Optimization

Find Pareto-optimal trade-offs with NSGA-II:

```python
from spicelab.optimization import NSGA2Optimizer

def multi_objective(params):
    # Return tuple of objectives to minimize
    gain = simulate_gain(params)
    power = simulate_power(params)
    return (-gain, power)  # Maximize gain, minimize power

optimizer = NSGA2Optimizer(
    population_size=50,
    n_generations=100
)

result = optimizer.optimize(
    multi_objective,
    bounds,
    n_objectives=2
)

# Get Pareto front
print(f"Found {len(result.pareto_front)} Pareto solutions")

# Get knee point (best compromise)
knee = result.pareto_front.get_knee_point()

# Get extreme solutions
min_power = result.pareto_front.get_extreme(1, minimize=True)
max_gain = result.pareto_front.get_extreme(0, minimize=True)
```

## Design of Experiments

### Factorial Designs

```python
from spicelab.optimization import full_factorial, fractional_factorial

bounds = [
    ParameterBounds("R1", 1e3, 10e3),
    ParameterBounds("R2", 1e3, 10e3),
    ParameterBounds("C1", 1e-9, 10e-9),
]

# Full factorial (3^3 = 27 runs)
design = full_factorial(bounds, levels=3)

# Fractional factorial (fewer runs)
design = fractional_factorial(bounds, resolution=3)
```

### Latin Hypercube Sampling

```python
from spicelab.optimization import latin_hypercube

# Space-filling design
design = latin_hypercube(
    bounds,
    n_samples=50,
    criterion="maximin",  # Maximize minimum distance
    seed=42
)
```

### Response Surface Methods

```python
from spicelab.optimization import (
    central_composite,
    box_behnken,
    run_doe,
    fit_response_surface
)

# Central Composite Design
design = central_composite(bounds, alpha="face", center_points=5)

# Box-Behnken (requires 3+ factors)
design = box_behnken(bounds, center_points=3)

# Run experiments
result = run_doe(design, objective_func)

# Fit response surface
surface = fit_response_surface(result, model_type="quadratic")
print(f"R-squared: {surface.r_squared:.4f}")

# Find optimum on surface
optimum = surface.get_optimum(bounds, minimize=True)
```

## Sensitivity Analysis

### Morris Method (Screening)

Identify influential parameters with minimal evaluations:

```python
from spicelab.optimization import morris_analysis

result = morris_analysis(
    objective,
    bounds,
    n_trajectories=20,
    n_levels=4,
    seed=42
)

# Ranking by importance
ranking = result.get_ranking()
print(f"Most important: {ranking[0]}")

# Check for nonlinearity/interactions
for name in result.factor_names:
    if result.sigma[name] > result.mu_star[name]:
        print(f"{name}: significant nonlinear/interaction effects")
```

### Sobol Indices (Variance-Based)

Quantify parameter contributions to output variance:

```python
from spicelab.optimization import sobol_analysis

result = sobol_analysis(
    objective,
    bounds,
    n_samples=1024,
    calc_second_order=True,
    seed=42
)

# First-order indices (main effects)
for name, s1 in result.S1.items():
    print(f"{name}: S1={s1:.3f}, ST={result.ST[name]:.3f}")

# Total-order indices include interactions
# ST - S1 indicates interaction strength
```

### Local Sensitivity

Gradient-based sensitivity at a point:

```python
from spicelab.optimization import local_sensitivity

result = local_sensitivity(
    objective,
    bounds,
    point={"R1": 5000, "C1": 1e-9},
    step_size=0.01
)

# Gradients
for name, grad in result.gradients.items():
    print(f"d(output)/d({name}) = {grad:.4f}")
```

### One-at-a-Time (OAT)

Sweep each parameter individually:

```python
from spicelab.optimization import oat_analysis

result = oat_analysis(objective, bounds, n_points=21)

# Get sweep data for plotting
for name, (param_values, responses) in result.sweeps.items():
    plt.plot(param_values, responses, label=name)
```

## Corner & PVT Analysis

### Corner Definition

```python
from spicelab.optimization import CornerDefinition

corners = CornerDefinition()
corners.add_parameter("process", ["slow", "typical", "fast"])
corners.add_parameter("voltage", [3.0, 3.3, 3.6])
corners.add_parameter("temperature", [-40, 25, 85])

# Generate all 27 corners
all_corners = corners.generate_all()

# Generate only extremes (8 corners)
extreme_corners = corners.generate_extremes()

# Get typical corner
typical = corners.generate_typical()
```

### PVT Sweep

```python
from spicelab.optimization import PVTSweep

pvt = PVTSweep()
pvt.set_process(["slow", "typical", "fast"])
pvt.set_voltage(3.3, tolerance=0.1, n_points=3)  # 3.0V, 3.3V, 3.6V
pvt.set_temperature("military")  # -40, 25, 85

conditions = pvt.generate_all()
print(f"Total PVT conditions: {len(conditions)}")
```

### Corner Analysis

```python
from spicelab.optimization import CornerAnalysis, print_corner_report

analysis = CornerAnalysis(circuit, simulate_func)

def objective(params, dataset):
    return measure(dataset, "gain").value

results = analysis.run(corners, objective, verbose=True)

# Get worst/best case
print(f"Worst case: {results.worst_case.corner.name}")
print(f"Best case: {results.best_case.corner.name}")

# Statistics
stats = results.get_statistics()
print(f"Range: {stats['min']:.2f} to {stats['max']:.2f}")

# Generate report
print(print_corner_report(results))
```

### Statistical Corner Analysis

Monte Carlo analysis for yield estimation:

```python
from spicelab.optimization import statistical_corner_analysis

result = statistical_corner_analysis(
    objective,
    bounds,
    n_samples=10000,
    distribution="normal",  # or "uniform"
    seed=42
)

# Yield within spec
yield_pct = result.get_yield(spec_min=0.9, spec_max=1.1)
print(f"Yield: {yield_pct:.1%}")

# Process capability
cpk = result.get_cpk(spec_min=0.9, spec_max=1.1)
print(f"Cpk: {cpk:.2f}")

# Percentiles
print(f"1st percentile: {result.percentiles[1]:.4f}")
print(f"99th percentile: {result.percentiles[99]:.4f}")
```

### Worst-Case Analysis

Find worst-case parameter combinations:

```python
from spicelab.optimization import worst_case_analysis

result = worst_case_analysis(
    objective,
    bounds,
    maximize=True,  # Find maximum (worst if higher is worse)
    method="extremes",  # Check all 2^n corners
)

print(f"Worst value: {result.worst_value}")
print(f"Worst parameters: {result.worst_params}")
print(f"Sensitivity at worst: {result.sensitivity}")
```

## Constraints

Add constraints to optimization:

```python
from spicelab.optimization import (
    make_inequality_constraint,
    make_equality_constraint,
    make_bound_constraint
)

# Inequality: R1 >= 2 * R2
constraints = [
    make_inequality_constraint(
        lambda p: p["R1"] - 2 * p["R2"],
        name="R1 >= 2*R2"
    )
]

# Equality: R1 * C1 = 1e-6 (within tolerance)
constraints.append(
    make_equality_constraint(
        lambda p: p["R1"] * p["C1"] - 1e-6,
        tolerance=1e-9,
        name="RC = 1us"
    )
)

# Use with GA (scipy optimizers have limited constraint support)
result = ga_optimizer.optimize(
    objective, bounds, constraints=constraints
)
```

## Convergence Tracking

Monitor optimization progress:

```python
from spicelab.optimization import ConvergenceTracker

tracker = ConvergenceTracker(
    tolerance=1e-6,
    patience=10,
    min_improvement=0.001
)

for iteration in range(max_iterations):
    value = evaluate(params)
    tracker.record(iteration, value, params)

    if tracker.is_converged():
        print("Converged!")
        break

    if tracker.is_stagnant():
        print("Stagnant - consider restarting")
```

## Complete Example: Op-Amp Optimization

```python
from spicelab import Circuit
from spicelab.optimization import (
    CircuitOptimizer,
    NSGA2Optimizer,
    morris_analysis,
    PVTSweep,
    CornerAnalysis,
)

# Build circuit
circuit = Circuit("opamp_stage")
# ... add components ...

def simulate(circuit):
    return run_ac_analysis(circuit, start=1, stop=1e9)

# 1. Sensitivity screening
bounds = [
    ParameterBounds("R1", 1e3, 100e3),
    ParameterBounds("R2", 1e3, 100e3),
    ParameterBounds("C1", 1e-12, 100e-12),
]

morris_result = morris_analysis(
    lambda p: simulate_and_measure(p)["bandwidth"],
    bounds,
    n_trajectories=20
)
print(f"Most influential: {morris_result.get_ranking()[:2]}")

# 2. Multi-objective optimization
def objectives(params):
    result = simulate_with_params(params)
    return (-result["gain"], result["power"], -result["bandwidth"])

optimizer = NSGA2Optimizer(population_size=50, n_generations=100)
result = optimizer.optimize(objectives, bounds, n_objectives=3)

# 3. Select design point
best = result.pareto_front.get_knee_point()

# 4. PVT verification
pvt = PVTSweep()
pvt.set_process(["slow", "typical", "fast"])
pvt.set_voltage(3.3, tolerance=0.1)
pvt.set_temperature("military")

corners = pvt.to_corner_definition()
analysis = CornerAnalysis(circuit, simulate)

pvt_results = analysis.run(corners, measure_performance)
print(f"Worst-case gain: {pvt_results.worst_case.value:.2f} dB")
```

## API Reference

### Core Classes

- `CircuitOptimizer` - High-level interface for circuit optimization
- `ParameterBounds` - Defines parameter bounds and scaling
- `OptimizationResult` - Container for optimization results
- `OptimizationConfig` - Configuration for optimization runs

### Optimizers

- `NelderMeadOptimizer`, `PowellOptimizer`, `LBFGSBOptimizer`
- `DifferentialEvolutionOptimizer`, `DualAnnealingOptimizer`
- `GeneticOptimizer`, `NSGA2Optimizer`
- `MultiStartOptimizer`

### Design of Experiments

- `full_factorial`, `fractional_factorial`
- `latin_hypercube`, `sobol_sequence`
- `central_composite`, `box_behnken`
- `fit_response_surface`, `run_doe`

### Sensitivity Analysis

- `morris_analysis` - Morris screening method
- `sobol_analysis` - Sobol variance-based indices
- `local_sensitivity` - Local gradient analysis
- `oat_analysis` - One-at-a-time sweeps

### Corner Analysis

- `CornerDefinition`, `Corner`, `CornerAnalysis`
- `PVTSweep`, `PVTCondition`
- `statistical_corner_analysis`
- `worst_case_analysis`
