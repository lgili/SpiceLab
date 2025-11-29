"""Integration tests for the optimization framework (M9).

These tests verify that multiple optimization components work together
correctly in realistic scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from spicelab.optimization import (
    ParameterBounds,
    OptimizationConfig,
)
from spicelab.optimization.doe import (
    full_factorial,
    latin_hypercube,
    central_composite,
    run_doe,
    fit_response_surface,
)
from spicelab.optimization.sensitivity import (
    morris_analysis,
    sobol_analysis,
    local_sensitivity,
    oat_analysis,
)
from spicelab.optimization.corner import (
    CornerDefinition,
    CornerAnalysisResult,
    CornerResult,
    Corner,
    PVTSweep,
    statistical_corner_analysis,
    worst_case_analysis,
)
from spicelab.optimization.genetic import (
    GeneticOptimizer,
    NSGA2Optimizer,
)

# Check if scipy is available
try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

scipy_required = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


# =============================================================================
# Test Functions
# =============================================================================

def quadratic_bowl(params: dict[str, float]) -> float:
    """Simple quadratic bowl function: minimum at (0, 0)."""
    return params["x"] ** 2 + params["y"] ** 2


def linear_response(params: dict[str, float]) -> float:
    """Linear response: y = 2*R1 + 3*R2 + 0.5*C1."""
    return 2 * params["R1"] + 3 * params["R2"] + 0.5 * params["C1"]


def nonlinear_circuit(params: dict[str, float]) -> float:
    """Simulated circuit response with nonlinear behavior."""
    R = params["R"]
    C = params["C"]
    # RC time constant affects bandwidth
    tau = R * C
    # Simulate bandwidth as 1/(2*pi*tau) with some noise
    bw = 1 / (2 * np.pi * tau)
    return bw


def multi_objective_amp(params: dict[str, float]) -> tuple[float, float]:
    """Simulated amplifier with gain/power trade-off."""
    Ibias = params["Ibias"]
    Rload = params["Rload"]

    # Higher bias = more gain, more power
    gain = 10 * np.log10(Ibias * Rload)  # dB
    power = Ibias ** 2 * Rload  # W

    return (-gain, power)  # Minimize both


# =============================================================================
# Integration Test: DoE + RSM + Optimization
# =============================================================================


class TestDoEToOptimization:
    """Test DoE -> Response Surface -> Optimization workflow."""

    def test_factorial_to_rsm(self) -> None:
        """Test full factorial -> fit RSM -> predict."""
        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]

        # 1. Run factorial design
        design = full_factorial(bounds, levels=5)
        result = run_doe(design, quadratic_bowl)

        assert result.n_runs == 25

        # 2. Fit response surface
        surface = fit_response_surface(result, model_type="quadratic")

        assert surface.r_squared > 0.99  # Perfect quadratic fit

        # 3. Find optimum on surface
        optimum = surface.get_optimum(bounds, minimize=True)

        # Should be near (0, 0)
        assert abs(optimum["x"]) < 1
        assert abs(optimum["y"]) < 1

    def test_ccd_to_rsm(self) -> None:
        """Test CCD -> fit RSM -> verify."""
        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]

        # CCD is designed for quadratic models
        design = central_composite(bounds, alpha="face", center_points=3)
        result = run_doe(design, quadratic_bowl)

        surface = fit_response_surface(result, model_type="quadratic")

        # Check coefficients
        assert abs(surface.coefficients.get("x^2", 0) - 1) < 0.1
        assert abs(surface.coefficients.get("y^2", 0) - 1) < 0.1

    def test_lhs_exploration(self) -> None:
        """Test LHS for design space exploration."""
        bounds = [
            ParameterBounds("R", 1e3, 10e3),
            ParameterBounds("C", 1e-9, 10e-9),
        ]

        # LHS for space-filling
        design = latin_hypercube(bounds, n_samples=30, criterion="maximin", seed=42)

        # Verify good coverage
        r_values = [p["R"] for p in design.points]
        c_values = [p["C"] for p in design.points]

        # Should span most of the range
        assert max(r_values) - min(r_values) > 0.8 * (10e3 - 1e3)
        assert max(c_values) - min(c_values) > 0.8 * (10e-9 - 1e-9)


# =============================================================================
# Integration Test: Sensitivity -> Optimization
# =============================================================================


class TestSensitivityToOptimization:
    """Test Sensitivity Analysis -> Optimization workflow."""

    def test_morris_screening(self) -> None:
        """Test Morris for screening important parameters."""
        bounds = [
            ParameterBounds("R1", 0, 10),   # coef = 2
            ParameterBounds("R2", 0, 10),   # coef = 3
            ParameterBounds("C1", 0, 10),   # coef = 0.5
        ]

        result = morris_analysis(linear_response, bounds, n_trajectories=15, seed=42)

        # R2 should be most important (highest coefficient)
        ranking = result.get_ranking()
        assert ranking[0] == "R2"
        assert ranking[1] == "R1"
        assert ranking[2] == "C1"

    def test_sobol_quantification(self) -> None:
        """Test Sobol for quantifying parameter importance."""
        bounds = [
            ParameterBounds("R1", 0, 10),
            ParameterBounds("R2", 0, 10),
        ]

        def simple_linear(params: dict[str, float]) -> float:
            return 2 * params["R1"] + 3 * params["R2"]

        result = sobol_analysis(simple_linear, bounds, n_samples=512, seed=42)

        # Check indices are in valid range
        for name in result.factor_names:
            assert 0 <= result.S1[name] <= 1
            assert 0 <= result.ST[name] <= 1

        # R2 has larger coefficient, should have higher S1
        assert result.S1["R2"] > result.S1["R1"]

    def test_local_gradient(self) -> None:
        """Test local sensitivity at a point."""
        bounds = [
            ParameterBounds("x", -10, 10),
            ParameterBounds("y", -10, 10),
        ]

        result = local_sensitivity(quadratic_bowl, bounds, point={"x": 2, "y": 3})

        # Gradient of x^2 + y^2 at (2, 3) is (4, 6)
        assert abs(result.gradients["x"] - 4) < 0.5
        assert abs(result.gradients["y"] - 6) < 0.5

    def test_oat_sweeps(self) -> None:
        """Test OAT for parameter sweeps."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        result = oat_analysis(quadratic_bowl, bounds, n_points=11)

        # Check sweep data exists
        assert "x" in result.sweeps
        assert "y" in result.sweeps

        # Check sweep covers range
        x_vals, _ = result.sweeps["x"]
        assert min(x_vals) == 0
        assert max(x_vals) == 10


# =============================================================================
# Integration Test: Corner Analysis + PVT
# =============================================================================


class TestCornerPVTIntegration:
    """Test Corner and PVT analysis integration."""

    def test_corner_definition_workflow(self) -> None:
        """Test complete corner definition workflow."""
        # 1. Define corners
        corners = CornerDefinition()
        corners.add_parameter("process", ["slow", "typical", "fast"])
        corners.add_parameter("voltage", [3.0, 3.3, 3.6])
        corners.add_parameter("temp", [-40, 25, 85])

        assert corners.n_corners == 27

        # 2. Generate all corners
        all_corners = corners.generate_all()
        assert len(all_corners) == 27

        # 3. Generate extremes
        extremes = corners.generate_extremes()
        assert len(extremes) == 8  # 2^3

        # 4. Generate typical
        typical = corners.generate_typical()
        assert typical.parameters["process"] == "typical"
        assert typical.parameters["voltage"] == 3.3

    def test_pvt_sweep_integration(self) -> None:
        """Test PVT sweep with corner definition."""
        pvt = PVTSweep()
        pvt.set_process(["slow", "typical", "fast"])
        pvt.set_voltage(3.3, tolerance=0.1, n_points=3)
        pvt.set_temperature("military")

        assert pvt.n_conditions == 27

        # Convert to corner definition
        definition = pvt.to_corner_definition()
        assert definition.n_corners == 27

        # Generate extremes
        extremes = pvt.generate_extremes()
        assert len(extremes) == 8

    def test_corner_analysis_result(self) -> None:
        """Test corner analysis result processing."""
        definition = CornerDefinition()
        definition.add_parameter("x", [1, 2, 3])

        # Simulate results
        results = [
            CornerResult(Corner(f"c{i}", {"x": i}), value=float(i**2))
            for i in [1, 2, 3]
        ]

        analysis = CornerAnalysisResult(results, definition)

        # Statistics
        stats = analysis.get_statistics()
        assert stats["min"] == 1.0
        assert stats["max"] == 9.0
        assert stats["mean"] == pytest.approx(14 / 3)

        # Worst/best
        assert analysis.worst_case.value == 9.0
        assert analysis.best_case.value == 1.0

    def test_statistical_corner_with_yield(self) -> None:
        """Test statistical corner with yield calculation."""
        bounds = [
            ParameterBounds("x", 0, 10),
        ]

        def objective(params: dict[str, float]) -> float:
            return params["x"]  # Linear: uniform output

        result = statistical_corner_analysis(
            objective, bounds, n_samples=1000, seed=42
        )

        # For uniform input, output should also be uniform
        assert result.mean == pytest.approx(5, abs=0.5)

        # Yield within [2, 8] should be about 60%
        yield_val = result.get_yield(spec_min=2, spec_max=8)
        assert 0.5 < yield_val < 0.7

    def test_worst_case_integration(self) -> None:
        """Test worst-case analysis."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        # Find maximum of x + y
        def objective(params: dict[str, float]) -> float:
            return params["x"] + params["y"]

        result = worst_case_analysis(
            objective, bounds, maximize=True, method="extremes"
        )

        assert result.worst_value == 20.0
        assert result.worst_params["x"] == 10
        assert result.worst_params["y"] == 10


# =============================================================================
# Integration Test: Genetic Algorithms
# =============================================================================


class TestGAIntegration:
    """Test genetic algorithm integration."""

    def test_ga_optimization(self) -> None:
        """Test GA on simple problem."""
        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]

        optimizer = GeneticOptimizer(
            population_size=30,
            n_generations=50,
        )

        config = OptimizationConfig(seed=42, verbose=False)
        result = optimizer.optimize(quadratic_bowl, bounds, config=config)

        # Should find near (0, 0)
        assert result.success
        assert abs(result.parameters["x"]) < 1
        assert abs(result.parameters["y"]) < 1

    def test_nsga2_pareto(self) -> None:
        """Test NSGA-II multi-objective."""
        bounds = [
            ParameterBounds("Ibias", 1e-6, 1e-3, scale="log"),
            ParameterBounds("Rload", 1e3, 100e3, scale="log"),
        ]

        optimizer = NSGA2Optimizer(
            population_size=30,
            n_generations=30,
        )

        config = OptimizationConfig(seed=42, verbose=False)
        result = optimizer.optimize(
            multi_objective_amp, bounds, n_objectives=2, config=config
        )

        # Should find multiple Pareto solutions
        assert len(result.pareto_front) >= 2

        # Get knee point
        knee = result.pareto_front.get_knee_point()
        assert "Ibias" in knee
        assert "Rload" in knee


# =============================================================================
# Integration Test: Complete Workflow
# =============================================================================


class TestCompleteWorkflow:
    """Test complete optimization workflow."""

    def test_screening_to_optimization(self) -> None:
        """Test: Screening -> DoE -> RSM -> Optimization."""
        # Problem: find minimum of 2*x^2 + 3*y^2 + z^2

        def objective(params: dict[str, float]) -> float:
            return 2 * params["x"]**2 + 3 * params["y"]**2 + params["z"]**2

        bounds = [
            ParameterBounds("x", -10, 10),
            ParameterBounds("y", -10, 10),
            ParameterBounds("z", -10, 10),
        ]

        # 1. Morris screening
        morris = morris_analysis(objective, bounds, n_trajectories=10, seed=42)
        ranking = morris.get_ranking()

        # y should be most important (coef = 3)
        assert ranking[0] == "y"

        # 2. DoE on important parameters
        important_bounds = [
            ParameterBounds("x", -10, 10),
            ParameterBounds("y", -10, 10),
        ]
        design = central_composite(important_bounds, center_points=3)

        def reduced_objective(params: dict[str, float]) -> float:
            return objective({"x": params["x"], "y": params["y"], "z": 0})

        doe_result = run_doe(design, reduced_objective)

        # 3. Fit RSM
        surface = fit_response_surface(doe_result, model_type="quadratic")
        assert surface.r_squared > 0.95

        # 4. Find optimum
        optimum = surface.get_optimum(important_bounds, minimize=True)
        assert abs(optimum["x"]) < 2
        assert abs(optimum["y"]) < 2

    def test_pvt_with_yield(self) -> None:
        """Test: PVT corners -> Yield analysis."""
        # Problem: circuit gain depends on PVT

        def gain_model(params: dict[str, float]) -> float:
            process = params.get("process_factor", 1.0)
            voltage = params.get("voltage", 3.3)
            temp = params.get("temp", 25)

            # Simplified gain model
            base_gain = 40  # dB
            gain = base_gain * process * (voltage / 3.3) * (1 - 0.001 * (temp - 25))
            return gain

        # 1. Define PVT
        bounds = [
            ParameterBounds("process_factor", 0.8, 1.2),  # ±20% process
            ParameterBounds("voltage", 3.0, 3.6),          # ±10% voltage
            ParameterBounds("temp", -40, 85),              # Temperature range
        ]

        # 2. Corner analysis (extremes)
        corners = CornerDefinition()
        corners.add_parameter("process_factor", [0.8, 1.0, 1.2])
        corners.add_parameter("voltage", [3.0, 3.3, 3.6])
        corners.add_parameter("temp", [-40, 25, 85])

        # Simulate all corners
        results = []
        for corner in corners.generate_all():
            value = gain_model(corner.parameters)
            results.append(CornerResult(corner, value))

        analysis = CornerAnalysisResult(results, corners)

        # Check range
        stats = analysis.get_statistics()
        assert stats["min"] > 20  # Minimum gain
        assert stats["max"] < 60  # Maximum gain

        # 3. Statistical yield
        stat_result = statistical_corner_analysis(
            gain_model, bounds, n_samples=1000, seed=42
        )

        # Spec: 35-45 dB
        yield_val = stat_result.get_yield(spec_min=35, spec_max=45)
        assert 0 < yield_val < 1

    def test_multi_objective_to_corner(self) -> None:
        """Test: Multi-objective -> Select point -> Verify at corners."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        def objectives(params: dict[str, float]) -> tuple[float, float]:
            # Minimize x^2, minimize (x-5)^2
            return (params["x"]**2, (params["x"] - 5)**2)

        # 1. Multi-objective optimization
        optimizer = NSGA2Optimizer(population_size=30, n_generations=30)
        result = optimizer.optimize(objectives, bounds, n_objectives=2)

        # 2. Select compromise
        knee = result.pareto_front.get_knee_point()

        # 3. Verify at extreme values of y (which doesn't affect objectives)
        for y_val in [0, 10]:
            test_params = {"x": knee["x"], "y": y_val}
            obj_values = objectives(test_params)
            # Should still be on Pareto front
            assert obj_values[0] < 30  # x^2 bounded
            assert obj_values[1] < 30  # (x-5)^2 bounded


# =============================================================================
# Integration Test: scipy optimizers
# =============================================================================


@scipy_required
class TestScipyIntegration:
    """Test scipy optimizer integration."""

    def test_multistart_with_lhs(self) -> None:
        """Test multi-start with LHS sampling."""
        from spicelab.optimization import MultiStartOptimizer, NelderMeadOptimizer

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]

        base = NelderMeadOptimizer()
        multi = MultiStartOptimizer(base, n_starts=5, sampling="lhs", seed=42)

        config = OptimizationConfig(max_iterations=50, verbose=False)
        result = multi.optimize(quadratic_bowl, bounds, config=config)

        # All starts should find the optimum
        assert result.n_starts == 5
        assert result.best_result.value < 0.1
        assert result.success_rate >= 0.8

    def test_convergence_tracking(self) -> None:
        """Test optimization with convergence tracking."""
        from spicelab.optimization import NelderMeadOptimizer, ConvergenceTracker

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]

        optimizer = NelderMeadOptimizer()
        config = OptimizationConfig(max_iterations=100)
        result = optimizer.optimize(quadratic_bowl, bounds, config=config)

        # Check history
        assert len(result.history) > 0
        # History should show improvement
        first_value = result.history[0][1]
        last_value = result.history[-1][1]
        assert last_value <= first_value
