# Optimization Cookbook

Practical recipes for common circuit optimization tasks.

## Recipe 1: RC Filter Bandwidth Optimization

Optimize resistor and capacitor values to achieve a target bandwidth.

```python
from spicelab import Circuit
from spicelab.library import resistor, capacitor
from spicelab.optimization import CircuitOptimizer, OptimizationConfig

# Build RC filter
circuit = Circuit("rc_filter")
circuit.add(resistor("R1", "vin", "vout", 1e3))
circuit.add(capacitor("C1", "vout", "gnd", 1e-9))
circuit.add_ground("gnd")

def simulate(circuit):
    return circuit.ac(start=1, stop=1e9, points=100)

# Objective: -3dB bandwidth = 10kHz
TARGET_BW = 10e3

def objective(params, dataset):
    # Find -3dB frequency
    gain_db = 20 * np.log10(np.abs(dataset["vout"]))
    f3db_idx = np.argmin(np.abs(gain_db - (gain_db[0] - 3)))
    f3db = float(dataset.coords["frequency"][f3db_idx])
    return (f3db - TARGET_BW) ** 2

# Optimize
opt = CircuitOptimizer(circuit, simulate)
opt.add_parameter("R1", 100, 100e3, scale="log")
opt.add_parameter("C1", 1e-12, 1e-6, scale="log")

config = OptimizationConfig(max_iterations=100, verbose=True)
result = opt.optimize(objective, method="nelder-mead", config=config)

print(f"Optimal R1: {result.parameters['R1']:.0f} ohms")
print(f"Optimal C1: {result.parameters['C1']*1e9:.2f} nF")
print(f"Theoretical: RC = 1/(2*pi*fc) = {1/(2*np.pi*TARGET_BW)*1e6:.2f} us")
```

## Recipe 2: Multi-Objective Amplifier Design

Trade off gain, bandwidth, and power consumption.

```python
from spicelab.optimization import NSGA2Optimizer, ParameterBounds

bounds = [
    ParameterBounds("Ibias", 10e-6, 1e-3, scale="log"),
    ParameterBounds("Rload", 1e3, 100e3, scale="log"),
    ParameterBounds("W", 1e-6, 100e-6),  # Transistor width
]

def multi_objective(params):
    circuit = build_amplifier(params)
    result = simulate(circuit)

    gain = measure_gain(result)      # dB, maximize
    bw = measure_bandwidth(result)   # Hz, maximize
    power = measure_power(result)    # W, minimize

    return (-gain, -bw, power)  # All objectives minimized

optimizer = NSGA2Optimizer(
    population_size=100,
    n_generations=200,
    crossover_prob=0.9,
    mutation_prob=0.1
)

result = optimizer.optimize(
    multi_objective,
    bounds,
    n_objectives=3,
    config=OptimizationConfig(seed=42)
)

# Analyze Pareto front
print(f"Found {len(result.pareto_front)} Pareto-optimal solutions")

# Get best compromise (knee point)
best = result.pareto_front.get_knee_point()
print(f"Best compromise: {best}")

# Get solution with maximum gain
max_gain = result.pareto_front.get_extreme(0, minimize=True)

# Get solution with minimum power
min_power = result.pareto_front.get_extreme(2, minimize=True)
```

## Recipe 3: Sensitivity Screening

Identify which parameters most affect circuit performance.

```python
from spicelab.optimization import morris_analysis, print_sensitivity_report

bounds = [
    ParameterBounds("R1", 900, 1100),       # ±10% tolerance
    ParameterBounds("R2", 1800, 2200),
    ParameterBounds("C1", 0.9e-9, 1.1e-9),
    ParameterBounds("C2", 0.9e-9, 1.1e-9),
    ParameterBounds("Vcc", 4.75, 5.25),     # ±5% supply
    ParameterBounds("temp", -40, 85),       # Temperature range
]

def measure_output(params):
    circuit = build_with_params(params)
    result = simulate(circuit)
    return measure_key_metric(result)

# Morris screening (fast, ~200 evaluations for 6 params)
result = morris_analysis(
    measure_output,
    bounds,
    n_trajectories=30,
    seed=42
)

# Get ranking
print("Parameter importance ranking:")
for i, name in enumerate(result.get_ranking()):
    print(f"  {i+1}. {name}: mu*={result.mu_star[name]:.3f}")

# Identify parameters with nonlinear effects
print("\nParameters with nonlinear/interaction effects:")
for name in result.factor_names:
    if result.sigma[name] > 0.5 * result.mu_star[name]:
        print(f"  {name}: sigma={result.sigma[name]:.3f}")

# Full report
print(print_sensitivity_report(result))
```

## Recipe 4: Response Surface Optimization

Use Design of Experiments to build a model and find optimum.

```python
from spicelab.optimization import (
    central_composite,
    run_doe,
    fit_response_surface,
    ParameterBounds
)

bounds = [
    ParameterBounds("R1", 1e3, 10e3),
    ParameterBounds("R2", 1e3, 10e3),
    ParameterBounds("C1", 1e-9, 10e-9),
]

# Create Central Composite Design
design = central_composite(bounds, alpha="face", center_points=5)
print(f"Design requires {design.n_runs} simulations")

def objective(params):
    circuit = build_circuit(params)
    result = simulate(circuit)
    return measure_error(result)

# Run all experiments
doe_result = run_doe(design, objective)

# Fit quadratic response surface
surface = fit_response_surface(doe_result, model_type="quadratic")
print(f"R-squared: {surface.r_squared:.4f}")

# Print coefficient significance
for name, coef in sorted(surface.coefficients.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name}: {coef:.4g}")

# Find optimum on surface (fast, no simulation needed)
optimum = surface.get_optimum(bounds, minimize=True, n_samples=10000)
print(f"\nPredicted optimum: {optimum}")
print(f"Predicted value: {surface.predict(optimum):.4f}")

# Verify with actual simulation
actual = objective(optimum)
print(f"Actual value: {actual:.4f}")
```

## Recipe 5: PVT Corner Analysis

Verify design across Process-Voltage-Temperature corners.

```python
from spicelab.optimization import (
    PVTSweep,
    CornerAnalysis,
    print_corner_report
)

# Define PVT sweep
pvt = PVTSweep()
pvt.set_process(["slow", "typical", "fast"])
pvt.set_voltage(3.3, tolerance=0.1, n_points=3)  # 3.0, 3.3, 3.6V
pvt.set_temperature("military")  # -40, 25, 85°C

print(f"Total corners: {pvt.n_conditions}")

# Define how to apply corner to circuit
def apply_corner(circuit, corner):
    modified = circuit.copy()
    # Apply process corner
    if corner.parameters["process"] == "slow":
        modified.set_parameter("mobility_factor", 0.8)
    elif corner.parameters["process"] == "fast":
        modified.set_parameter("mobility_factor", 1.2)

    # Apply voltage
    modified.set_parameter("Vdd", corner.parameters["voltage"])

    # Apply temperature
    modified.set_parameter("temp", corner.parameters["temperature"])

    return modified

# Run corner analysis
analysis = CornerAnalysis(circuit, simulate, apply_corner)
corners = pvt.to_corner_definition()

def measure_spec(params, dataset):
    gain = measure_gain(dataset)
    return gain  # Return gain in dB

results = analysis.run(corners, measure_spec, verbose=True)

# Check specs
SPEC_MIN = 40  # dB
SPEC_MAX = 60  # dB

passing = [r for r in results.results if SPEC_MIN <= r.value <= SPEC_MAX]
print(f"\nPassing corners: {len(passing)}/{results.n_corners}")

# Identify failing corners
failing = [r for r in results.results if not (SPEC_MIN <= r.value <= SPEC_MAX)]
for r in failing:
    print(f"  FAIL: {r.corner.name} = {r.value:.1f} dB")

# Full report
print(print_corner_report(results))
```

## Recipe 6: Yield Analysis

Estimate manufacturing yield with Monte Carlo simulation.

```python
from spicelab.optimization import (
    statistical_corner_analysis,
    ParameterBounds
)

# Component tolerances
bounds = [
    ParameterBounds("R1", 950, 1050),     # 1k ±5%
    ParameterBounds("R2", 1900, 2100),    # 2k ±5%
    ParameterBounds("C1", 0.8e-9, 1.2e-9), # 1nF ±20%
]

def measure_output(params):
    circuit = build_circuit(params)
    result = simulate(circuit)
    return measure_key_spec(result)

# Monte Carlo with normal distribution (realistic for components)
result = statistical_corner_analysis(
    measure_output,
    bounds,
    n_samples=10000,
    distribution="normal",  # 3-sigma at bounds
    seed=42,
    percentiles=(0.1, 1, 5, 50, 95, 99, 99.9)
)

# Yield calculation
SPEC_MIN = 0.95
SPEC_MAX = 1.05

yield_pct = result.get_yield(spec_min=SPEC_MIN, spec_max=SPEC_MAX)
print(f"Yield: {yield_pct:.2%}")

# Process capability
cpk = result.get_cpk(spec_min=SPEC_MIN, spec_max=SPEC_MAX)
print(f"Cpk: {cpk:.3f}")

if cpk >= 1.33:
    print("Process is capable (Cpk >= 1.33)")
elif cpk >= 1.0:
    print("Process is marginal (1.0 <= Cpk < 1.33)")
else:
    print("Process is NOT capable (Cpk < 1.0)")

# Distribution summary
print(f"\nMean: {result.mean:.4f}")
print(f"Std:  {result.std:.4f}")
print(f"Min:  {result.min:.4f}")
print(f"Max:  {result.max:.4f}")

# Percentiles
print("\nPercentiles:")
for p, v in sorted(result.percentiles.items()):
    print(f"  {p:5.1f}%: {v:.4f}")
```

## Recipe 7: Worst-Case Design

Find the combination of tolerances that produces the worst output.

```python
from spicelab.optimization import worst_case_analysis, ParameterBounds

bounds = [
    ParameterBounds("R1", 950, 1050),
    ParameterBounds("R2", 1900, 2100),
    ParameterBounds("C1", 0.8e-9, 1.2e-9),
    ParameterBounds("Vcc", 4.75, 5.25),
]

def measure_error(params):
    circuit = build_circuit(params)
    result = simulate(circuit)
    target = 2.5  # Expected output
    actual = measure_output(result)
    return abs(actual - target)

# Find worst case (maximum error)
result = worst_case_analysis(
    measure_error,
    bounds,
    maximize=True,
    method="extremes"  # Check all 2^n corners
)

print(f"Worst-case error: {result.worst_value:.4f}")
print(f"\nWorst-case parameters:")
for name, value in result.worst_params.items():
    print(f"  {name}: {value}")

print(f"\nSensitivity at worst case:")
for name, sens in sorted(result.sensitivity.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name}: {sens:.4f}")
```

## Recipe 8: Convergence Monitoring

Track optimization progress and detect stagnation.

```python
from spicelab.optimization import (
    ConvergenceTracker,
    NelderMeadOptimizer,
    OptimizationConfig
)

# Custom convergence callback
tracker = ConvergenceTracker(
    tolerance=1e-6,
    patience=20,
    min_improvement=0.001
)

def callback(iteration, value, params):
    tracker.record(iteration, value, params)

    # Print progress
    if iteration % 10 == 0:
        data = tracker.data
        print(f"Iter {iteration}: value={value:.6f}, best={data.best_value:.6f}")

    # Check for early stopping
    if tracker.is_stagnant():
        print("Warning: optimization stagnant")
        return True  # Signal to stop

    if tracker.is_converged():
        print("Converged!")
        return True

    return False  # Continue

# Run optimization with callback
optimizer = NelderMeadOptimizer()
config = OptimizationConfig(max_iterations=500, callback=callback)
result = optimizer.optimize(objective, bounds, config=config)

# Analyze convergence
data = tracker.data
print(f"\nFinal value: {data.final_value:.6f}")
print(f"Best value: {data.best_value:.6f}")
print(f"Total improvement: {data.improvement:.6f}")
print(f"Iterations: {data.n_iterations}")
```

## Recipe 9: Constrained Optimization with GA

Optimize with complex constraints using genetic algorithms.

```python
from spicelab.optimization import (
    GeneticOptimizer,
    make_inequality_constraint,
    OptimizationConfig
)

bounds = [
    ParameterBounds("R1", 1e3, 100e3),
    ParameterBounds("R2", 1e3, 100e3),
    ParameterBounds("R3", 1e3, 100e3),
]

# Constraints
constraints = [
    # R1 + R2 <= 50k (total resistance limit)
    make_inequality_constraint(
        lambda p: 50e3 - (p["R1"] + p["R2"]),
        name="R1+R2 <= 50k"
    ),
    # R1 >= 0.5 * R2 (ratio constraint)
    make_inequality_constraint(
        lambda p: p["R1"] - 0.5 * p["R2"],
        name="R1 >= 0.5*R2"
    ),
    # R3 <= R1 + R2 (ordering constraint)
    make_inequality_constraint(
        lambda p: (p["R1"] + p["R2"]) - p["R3"],
        name="R3 <= R1+R2"
    ),
]

optimizer = GeneticOptimizer(
    population_size=100,
    n_generations=200,
    crossover_prob=0.8,
    mutation_prob=0.15
)

config = OptimizationConfig(seed=42, verbose=True)
result = optimizer.optimize(
    objective,
    bounds,
    constraints=constraints,
    config=config
)

# Verify constraints
params = result.parameters
print(f"R1 + R2 = {params['R1'] + params['R2']:.0f} (limit: 50k)")
print(f"R1/R2 = {params['R1']/params['R2']:.2f} (min: 0.5)")
print(f"R3 = {params['R3']:.0f}, R1+R2 = {params['R1']+params['R2']:.0f}")
```

## Recipe 10: Sobol Sensitivity with Interactions

Deep sensitivity analysis including parameter interactions.

```python
from spicelab.optimization import sobol_analysis, ParameterBounds

bounds = [
    ParameterBounds("R1", 1e3, 10e3),
    ParameterBounds("R2", 1e3, 10e3),
    ParameterBounds("C1", 1e-9, 10e-9),
]

result = sobol_analysis(
    objective,
    bounds,
    n_samples=2048,
    calc_second_order=True,  # Include pairwise interactions
    seed=42
)

# Main effects
print("First-order indices (main effects):")
total_s1 = sum(result.S1.values())
for name in result.get_main_effects_ranking():
    s1 = result.S1[name]
    print(f"  {name}: S1={s1:.3f} ({s1/total_s1*100:.1f}% of main effects)")

# Total effects (including interactions)
print("\nTotal-order indices:")
for name in result.factor_names:
    st = result.ST[name]
    interaction = result.get_interaction_strength(name)
    print(f"  {name}: ST={st:.3f}, interaction strength={interaction:.3f}")

# Second-order interactions
if result.S2:
    print("\nSecond-order indices (pairwise interactions):")
    for (p1, p2), s2 in sorted(result.S2.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(s2) > 0.01:
            print(f"  {p1} x {p2}: S2={s2:.3f}")
```
