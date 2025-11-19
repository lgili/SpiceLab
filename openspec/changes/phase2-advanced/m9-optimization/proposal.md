# M9: Optimization & Design of Experiments

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 8-10 weeks
**Dependencies:** M7 (measurements for objectives), M2 (performance baseline), M4 (error handling)

## Problem Statement

SpiceLab lacks systematic optimization capabilities for circuit design. Users must manually tune component values through trial-and-error, which is inefficient, non-optimal, and doesn't scale to multi-parameter optimization. There's no support for Design of Experiments (DoE), sensitivity analysis, or systematic corner analysis—all critical for production-ready circuit design.

### Current Gaps
- ❌ No automatic parameter optimization (scipy.optimize integration)
- ❌ No genetic algorithms for multi-modal optimization
- ❌ No multi-objective optimization (conflicting specs)
- ❌ No Design of Experiments (factorial, Taguchi, Latin hypercube)
- ❌ No sensitivity analysis (parameter importance)
- ❌ No automated corner analysis
- ❌ No PVT (Process/Voltage/Temperature) sweep helpers
- ❌ Users rely on manual tuning (hours to days)

### Impact
- **Productivity:** Days wasted on manual parameter tuning
- **Optimality:** Suboptimal designs (local minima)
- **Robustness:** No systematic yield/corner analysis
- **Professional Use:** Cannot meet industry requirements for design optimization

## Objectives

1. **Integrate scipy.optimize** for gradient-based and derivative-free optimization
2. **Implement genetic algorithms** (DEAP, NSGA-II) for multi-modal problems
3. **Support multi-objective optimization** with Pareto front exploration
4. **Build DoE framework** (factorial, Taguchi, Latin hypercube, Sobol sequences)
5. **Add sensitivity analysis** (SALib integration, Morris, Sobol indices)
6. **Automate corner analysis** (fast/slow, high/low, PVT sweeps)
7. **Create PVT sweep helpers** for process/voltage/temperature variations
8. **Achieve production-ready optimization** framework with <1hr typical runtime

## Technical Design

### 1. Optimization Framework Architecture

```python
# spicelab/optimization/base.py
from typing import Protocol, Callable, Any
from pydantic import BaseModel
import numpy as np

class OptimizationResult(BaseModel):
    """Result of optimization."""
    success: bool
    optimal_params: dict[str, float]
    optimal_value: float
    n_iterations: int
    n_simulations: int
    convergence_history: list[float]
    final_measurements: dict[str, Any]

class ObjectiveFunction(Protocol):
    """Protocol for optimization objectives."""

    def __call__(self, params: dict[str, float]) -> float:
        """Evaluate objective (minimize)."""
        ...

class Optimizer(Protocol):
    """Protocol for optimizers."""

    def optimize(
        self,
        objective: ObjectiveFunction,
        bounds: dict[str, tuple[float, float]],
        constraints: list[Callable] | None = None,
    ) -> OptimizationResult:
        """Run optimization."""
        ...
```

### 2. Scipy.optimize Integration

```python
# spicelab/optimization/scipy_opt.py
from scipy.optimize import minimize, differential_evolution
from spicelab.core.circuit import Circuit
from spicelab.measurements import MeasurementRegistry

class CircuitOptimizer:
    """Optimize circuit parameters to meet specifications."""

    def __init__(
        self,
        circuit: Circuit,
        analyses: list[AnalysisSpec],
        measurements: list[tuple[str, dict]],  # [(measurement_name, kwargs)]
        engine: str = "ngspice",
    ):
        self.circuit = circuit
        self.analyses = analyses
        self.measurements = measurements
        self.engine = engine
        self.n_simulations = 0

    def minimize(
        self,
        objective: str | Callable,
        parameters: dict[str, tuple[float, float]],  # {param: (min, max)}
        method: str = "Nelder-Mead",  # scipy.optimize methods
        constraints: list | None = None,
        options: dict | None = None,
    ) -> OptimizationResult:
        """Minimize objective function."""

        # Build parameter vector
        param_names = list(parameters.keys())
        bounds = [parameters[name] for name in param_names]

        # Define objective wrapper
        def objective_wrapper(x: np.ndarray) -> float:
            # Update circuit with parameters
            param_dict = {name: val for name, val in zip(param_names, x)}
            self._update_circuit(param_dict)

            # Run simulation
            result = run_simulation(self.circuit, self.analyses, engine=self.engine)
            self.n_simulations += 1

            # Evaluate measurements
            measurements = self._evaluate_measurements(result)

            # Calculate objective
            if callable(objective):
                obj_value = objective(measurements)
            else:
                # Objective is measurement name
                obj_value = measurements[objective]

            return obj_value

        # Run optimization
        result = minimize(
            objective_wrapper,
            x0=[(b[0] + b[1]) / 2 for b in bounds],  # Midpoint start
            bounds=bounds,
            method=method,
            constraints=constraints,
            options=options or {},
        )

        optimal_params = {name: val for name, val in zip(param_names, result.x)}

        return OptimizationResult(
            success=result.success,
            optimal_params=optimal_params,
            optimal_value=result.fun,
            n_iterations=result.nit,
            n_simulations=self.n_simulations,
            convergence_history=[],  # Not available from scipy
            final_measurements=self._evaluate_final(optimal_params),
        )

    def differential_evolution(
        self,
        objective: str | Callable,
        parameters: dict[str, tuple[float, float]],
        population_size: int = 15,
        max_iterations: int = 1000,
        **kwargs
    ) -> OptimizationResult:
        """Global optimization with differential evolution."""

        param_names = list(parameters.keys())
        bounds = [parameters[name] for name in param_names]

        def objective_wrapper(x: np.ndarray) -> float:
            param_dict = {name: val for name, val in zip(param_names, x)}
            self._update_circuit(param_dict)

            result = run_simulation(self.circuit, self.analyses, engine=self.engine)
            self.n_simulations += 1

            measurements = self._evaluate_measurements(result)

            if callable(objective):
                return objective(measurements)
            else:
                return measurements[objective]

        result = differential_evolution(
            objective_wrapper,
            bounds=bounds,
            popsize=population_size,
            maxiter=max_iterations,
            **kwargs
        )

        optimal_params = {name: val for name, val in zip(param_names, result.x)}

        return OptimizationResult(
            success=result.success,
            optimal_params=optimal_params,
            optimal_value=result.fun,
            n_iterations=result.nit,
            n_simulations=self.n_simulations,
            convergence_history=[],
            final_measurements=self._evaluate_final(optimal_params),
        )

    def _update_circuit(self, params: dict[str, float]):
        """Update circuit with parameter values."""
        for param_name, value in params.items():
            # param_name format: "R1.resistance", "C1.capacitance"
            comp_ref, attr = param_name.split('.')
            comp = self.circuit.get_component(comp_ref)
            setattr(comp, attr, value)

    def _evaluate_measurements(self, result: SimulationResult) -> dict[str, float]:
        """Evaluate all measurements."""
        ds = result.dataset()
        measurements = {}

        for meas_name, meas_kwargs in self.measurements:
            meas_cls = MeasurementRegistry.get(meas_name)
            meas = meas_cls(**meas_kwargs)
            meas_result = meas.measure(ds)
            measurements[meas_name] = meas_result.value

        return measurements

    def _evaluate_final(self, params: dict[str, float]) -> dict[str, Any]:
        """Get all measurements at optimal point."""
        self._update_circuit(params)
        result = run_simulation(self.circuit, self.analyses, engine=self.engine)
        return self._evaluate_measurements(result)
```

### 3. Genetic Algorithms with DEAP

```python
# spicelab/optimization/genetic.py
from deap import base, creator, tools, algorithms
import random

class GeneticOptimizer:
    """Genetic algorithm optimization using DEAP."""

    def __init__(
        self,
        circuit: Circuit,
        analyses: list[AnalysisSpec],
        measurements: list[tuple[str, dict]],
        engine: str = "ngspice",
    ):
        self.circuit = circuit
        self.analyses = analyses
        self.measurements = measurements
        self.engine = engine

        # Setup DEAP
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize
            creator.create("Individual", list, fitness=creator.FitnessMin)

    def optimize(
        self,
        objective: str | Callable,
        parameters: dict[str, tuple[float, float]],
        population_size: int = 50,
        n_generations: int = 100,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
    ) -> OptimizationResult:
        """Run genetic algorithm optimization."""

        param_names = list(parameters.keys())
        bounds = [parameters[name] for name in param_names]

        toolbox = base.Toolbox()

        # Attribute generator
        for i, (low, high) in enumerate(bounds):
            toolbox.register(f"attr_{i}", random.uniform, low, high)

        # Individual and population
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            tuple(getattr(toolbox, f"attr_{i}") for i in range(len(bounds))),
            n=1,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        def evaluate(individual):
            param_dict = {name: val for name, val in zip(param_names, individual)}
            self._update_circuit(param_dict)

            result = run_simulation(self.circuit, self.analyses, engine=self.engine)
            measurements = self._evaluate_measurements(result)

            if callable(objective):
                obj_value = objective(measurements)
            else:
                obj_value = measurements[objective]

            return (obj_value,)

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run evolution
        population = toolbox.population(n=population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        population, logbook = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=n_generations,
            stats=stats,
            verbose=False,
        )

        # Best individual
        best = tools.selBest(population, k=1)[0]
        optimal_params = {name: val for name, val in zip(param_names, best)}

        return OptimizationResult(
            success=True,
            optimal_params=optimal_params,
            optimal_value=best.fitness.values[0],
            n_iterations=n_generations,
            n_simulations=population_size * n_generations,
            convergence_history=[record['min'] for record in logbook],
            final_measurements=self._evaluate_final(optimal_params),
        )

class MultiObjectiveOptimizer:
    """Multi-objective optimization with NSGA-II."""

    def optimize(
        self,
        objectives: list[str | Callable],  # Multiple objectives
        parameters: dict[str, tuple[float, float]],
        population_size: int = 100,
        n_generations: int = 200,
    ) -> list[OptimizationResult]:
        """Return Pareto front of solutions."""

        # NSGA-II implementation
        # Returns list of non-dominated solutions
        ...
```

### 4. Design of Experiments (DoE)

```python
# spicelab/optimization/doe.py
from pyDOE2 import lhs, ff2n, pbdesign
import numpy as np

class DesignOfExperiments:
    """Design of Experiments framework."""

    @staticmethod
    def factorial(
        factors: dict[str, tuple[float, float]],  # {name: (low, high)}
        levels: int = 2,  # 2-level factorial
    ) -> np.ndarray:
        """Full factorial design."""
        n_factors = len(factors)

        if levels == 2:
            # 2^k design
            design_coded = ff2n(n_factors)  # Returns -1, +1 coded
        else:
            # General factorial
            raise NotImplementedError("Only 2-level factorial supported")

        # Decode to actual values
        factor_names = list(factors.keys())
        bounds = np.array([factors[name] for name in factor_names])

        design = np.zeros_like(design_coded)
        for i in range(n_factors):
            low, high = bounds[i]
            design[:, i] = np.where(
                design_coded[:, i] == -1,
                low,
                high
            )

        return design

    @staticmethod
    def latin_hypercube(
        factors: dict[str, tuple[float, float]],
        n_samples: int = 100,
        criterion: str = "maximin",  # "center", "maximin", "centermaximin"
    ) -> np.ndarray:
        """Latin Hypercube Sampling."""
        n_factors = len(factors)

        # Generate LHS design
        design_normalized = lhs(n_factors, samples=n_samples, criterion=criterion)

        # Scale to actual bounds
        factor_names = list(factors.keys())
        bounds = np.array([factors[name] for name in factor_names])

        design = bounds[:, 0] + design_normalized * (bounds[:, 1] - bounds[:, 0])

        return design

    @staticmethod
    def sobol_sequence(
        factors: dict[str, tuple[float, float]],
        n_samples: int = 1024,  # Power of 2
    ) -> np.ndarray:
        """Sobol quasi-random sequence for better coverage."""
        from scipy.stats import qmc

        n_factors = len(factors)
        sampler = qmc.Sobol(d=n_factors, scramble=True)
        design_normalized = sampler.random(n_samples)

        # Scale
        factor_names = list(factors.keys())
        bounds = np.array([factors[name] for name in factor_names])

        design = bounds[:, 0] + design_normalized * (bounds[:, 1] - bounds[:, 0])

        return design
```

### 5. Sensitivity Analysis

```python
# spicelab/optimization/sensitivity.py
from SALib.analyze import morris, sobol
from SALib.sample import morris as morris_sample
from SALib.sample import saltelli

class SensitivityAnalysis:
    """Sensitivity analysis with SALib."""

    def __init__(
        self,
        circuit: Circuit,
        analyses: list[AnalysisSpec],
        measurement: tuple[str, dict],
        engine: str = "ngspice",
    ):
        self.circuit = circuit
        self.analyses = analyses
        self.measurement = measurement
        self.engine = engine

    def morris_analysis(
        self,
        factors: dict[str, tuple[float, float]],
        n_trajectories: int = 10,
    ) -> dict:
        """Morris one-at-a-time (OAT) sensitivity analysis."""

        # Define problem
        problem = {
            'num_vars': len(factors),
            'names': list(factors.keys()),
            'bounds': [factors[name] for name in factors.keys()],
        }

        # Generate samples
        samples = morris_sample.sample(problem, N=n_trajectories, num_levels=4)

        # Evaluate
        outputs = []
        for sample in samples:
            param_dict = {name: val for name, val in zip(problem['names'], sample)}
            output = self._evaluate_sample(param_dict)
            outputs.append(output)

        # Analyze
        Si = morris.analyze(problem, np.array(samples), np.array(outputs))

        return {
            'mu': Si['mu'],  # Mean absolute sensitivity
            'mu_star': Si['mu_star'],  # Mean absolute (preferred)
            'sigma': Si['sigma'],  # Standard deviation (interactions)
            'mu_star_conf': Si['mu_star_conf'],  # Confidence interval
        }

    def sobol_analysis(
        self,
        factors: dict[str, tuple[float, float]],
        n_samples: int = 1024,
    ) -> dict:
        """Sobol variance-based sensitivity analysis."""

        problem = {
            'num_vars': len(factors),
            'names': list(factors.keys()),
            'bounds': [factors[name] for name in factors.keys()],
        }

        # Generate samples (Saltelli sampling)
        samples = saltelli.sample(problem, N=n_samples)

        # Evaluate
        outputs = []
        for sample in samples:
            param_dict = {name: val for name, val in zip(problem['names'], sample)}
            output = self._evaluate_sample(param_dict)
            outputs.append(output)

        # Analyze
        Si = sobol.analyze(problem, np.array(outputs))

        return {
            'S1': Si['S1'],  # First-order indices
            'ST': Si['ST'],  # Total-order indices
            'S2': Si['S2'],  # Second-order indices
        }

    def _evaluate_sample(self, param_dict: dict[str, float]) -> float:
        """Evaluate single sample."""
        # Update circuit
        for param_name, value in param_dict.items():
            comp_ref, attr = param_name.split('.')
            comp = self.circuit.get_component(comp_ref)
            setattr(comp, attr, value)

        # Run simulation
        result = run_simulation(self.circuit, self.analyses, engine=self.engine)

        # Measure
        meas_name, meas_kwargs = self.measurement
        meas_cls = MeasurementRegistry.get(meas_name)
        meas = meas_cls(**meas_kwargs)
        meas_result = meas.measure(result.dataset())

        return meas_result.value
```

### 6. Corner Analysis and PVT Sweeps

```python
# spicelab/optimization/corners.py

class CornerAnalysis:
    """Automated corner analysis."""

    def __init__(
        self,
        circuit: Circuit,
        analyses: list[AnalysisSpec],
        measurements: list[tuple[str, dict]],
        engine: str = "ngspice",
    ):
        self.circuit = circuit
        self.analyses = analyses
        self.measurements = measurements
        self.engine = engine

    def run_corners(
        self,
        parameters: dict[str, tuple[float, float]],  # {param: (min, max)}
        include_typical: bool = True,
    ) -> dict[str, dict]:
        """Run all corners (fast/slow, high/low combinations)."""

        corners = self._generate_corners(parameters, include_typical)

        results = {}
        for corner_name, corner_params in corners.items():
            # Update circuit
            for param_name, value in corner_params.items():
                comp_ref, attr = param_name.split('.')
                comp = self.circuit.get_component(comp_ref)
                setattr(comp, attr, value)

            # Simulate
            result = run_simulation(self.circuit, self.analyses, engine=self.engine)

            # Measure
            measurements = {}
            for meas_name, meas_kwargs in self.measurements:
                meas_cls = MeasurementRegistry.get(meas_name)
                meas = meas_cls(**meas_kwargs)
                meas_result = meas.measure(result.dataset())
                measurements[meas_name] = meas_result.value

            results[corner_name] = {
                'parameters': corner_params,
                'measurements': measurements,
            }

        return results

    def _generate_corners(
        self,
        parameters: dict[str, tuple[float, float]],
        include_typical: bool,
    ) -> dict[str, dict]:
        """Generate corner combinations."""
        corners = {}

        # Typical
        if include_typical:
            corners['typical'] = {
                name: (low + high) / 2
                for name, (low, high) in parameters.items()
            }

        # Fast/slow corners (2^n combinations)
        from itertools import product

        param_names = list(parameters.keys())
        n_params = len(param_names)

        for combo in product([0, 1], repeat=n_params):
            corner_name = '_'.join(
                'F' if c == 0 else 'S' for c in combo
            )  # F=fast(min), S=slow(max)

            corner_params = {}
            for i, name in enumerate(param_names):
                low, high = parameters[name]
                corner_params[name] = low if combo[i] == 0 else high

            corners[corner_name] = corner_params

        return corners

class PVTSweep:
    """Process/Voltage/Temperature sweep helper."""

    def __init__(
        self,
        circuit: Circuit,
        analyses: list[AnalysisSpec],
        measurements: list[tuple[str, dict]],
        engine: str = "ngspice",
    ):
        self.circuit = circuit
        self.analyses = analyses
        self.measurements = measurements
        self.engine = engine

    def sweep_pvt(
        self,
        process: dict[str, tuple[float, float]] | None = None,  # Component variations
        voltage: dict[str, tuple[float, float]] | None = None,  # Supply variations
        temperature: tuple[float, float] = (-40, 125),  # Celsius
        n_points: int = 3,  # Per dimension
    ) -> xr.Dataset:
        """Run PVT sweep."""

        # Generate sweep points
        sweep_points = self._generate_pvt_points(
            process, voltage, temperature, n_points
        )

        results = []
        for point in sweep_points:
            # Apply PVT conditions
            self._apply_pvt(point)

            # Simulate
            result = run_simulation(self.circuit, self.analyses, engine=self.engine)

            # Measure
            measurements = self._measure(result)
            measurements['pvt'] = point

            results.append(measurements)

        # Convert to xarray Dataset
        return self._to_dataset(results)

    def _generate_pvt_points(self, process, voltage, temperature, n_points):
        """Generate PVT sweep points."""
        # Create grid of P, V, T values
        ...
```

## Implementation Plan

(Weeks 1-10 breakdown similar to previous milestones)

## Success Metrics

- [ ] **scipy.optimize integration** functional
- [ ] **Genetic algorithms** (DEAP) working
- [ ] **Multi-objective optimization** (Pareto front)
- [ ] **DoE methods:** factorial, LHS, Sobol
- [ ] **Sensitivity analysis:** Morris, Sobol indices
- [ ] **Corner analysis** automated
- [ ] **PVT sweeps** functional
- [ ] Typical optimization: <1 hour
- [ ] Test coverage: **95%+**
- [ ] Documentation complete

## Dependencies

- M7 (measurements as objectives)
- M2 (performance baseline)
- scipy
- DEAP
- SALib
- pyDOE2

## References

- [Scipy Optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [SALib](https://salib.readthedocs.io/)
- [Design of Experiments](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm)
