"""Tests for the optimization framework (M9).

Tests cover:
- ParameterBounds
- OptimizationResult
- Scipy optimizers (Nelder-Mead, Powell, L-BFGS-B, DE)
- CircuitObjective
- CircuitOptimizer
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from spicelab.optimization import (
    CircuitObjective,
    OptimizationConfig,
    OptimizationResult,
    ParameterBounds,
)
from spicelab.optimization.doe import (
    DoEResult,
    ExperimentalDesign,
    ResponseSurface,
    box_behnken,
    central_composite,
    fit_response_surface,
    fractional_factorial,
    full_factorial,
    latin_hypercube,
    run_doe,
    sobol_sequence,
)
from spicelab.optimization.sensitivity import (
    LocalSensitivity,
    MorrisResult,
    OATResult,
    SobolResult,
    generate_tornado_data,
    local_sensitivity,
    morris_analysis,
    oat_analysis,
    print_sensitivity_report,
    sobol_analysis,
)
from spicelab.optimization.genetic import (
    GAConfig,
    GeneticOptimizer,
    MultiObjectiveResult,
    NSGA2Optimizer,
    ParetoFront,
    get_genetic_optimizer,
)
from spicelab.optimization.utils import (
    ConvergenceData,
    ConvergenceTracker,
    make_bound_constraint,
    make_equality_constraint,
    make_inequality_constraint,
)

# Check if scipy is available
try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import scipy-dependent classes only if available
if HAS_SCIPY:
    from spicelab.optimization import (
        CircuitOptimizer,
        DifferentialEvolutionOptimizer,
        LBFGSBOptimizer,
        MultiStartOptimizer,
        MultiStartResult,
        NelderMeadOptimizer,
        PowellOptimizer,
        analyze_sensitivity,
        compute_hessian_diagonal,
        get_scipy_optimizer,
        list_scipy_optimizers,
    )

scipy_required = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


# =============================================================================
# Test ParameterBounds
# =============================================================================


class TestParameterBounds:
    """Tests for ParameterBounds class."""

    def test_basic_creation(self) -> None:
        """Test basic bounds creation."""
        bounds = ParameterBounds("R1", 100, 10000)
        assert bounds.name == "R1"
        assert bounds.lower == 100
        assert bounds.upper == 10000
        assert bounds.scale == "linear"
        # Default initial should be midpoint
        assert bounds.initial == 5050

    def test_log_scale(self) -> None:
        """Test log scale bounds."""
        bounds = ParameterBounds("R1", 100, 10000, scale="log")
        assert bounds.scale == "log"
        # Default initial should be geometric mean
        assert bounds.initial == pytest.approx(1000)

    def test_custom_initial(self) -> None:
        """Test custom initial value."""
        bounds = ParameterBounds("R1", 100, 10000, initial=2000)
        assert bounds.initial == 2000

    def test_invalid_bounds(self) -> None:
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError, match="Lower bound must be < upper"):
            ParameterBounds("R1", 10000, 100)

    def test_invalid_initial(self) -> None:
        """Test that invalid initial raises error."""
        with pytest.raises(ValueError, match="Initial value must be within bounds"):
            ParameterBounds("R1", 100, 10000, initial=50)

    def test_normalize_linear(self) -> None:
        """Test linear normalization."""
        bounds = ParameterBounds("R1", 0, 100)
        assert bounds.normalize(0) == 0
        assert bounds.normalize(50) == 0.5
        assert bounds.normalize(100) == 1.0

    def test_normalize_log(self) -> None:
        """Test log normalization."""
        bounds = ParameterBounds("R1", 1, 1000, scale="log")
        assert bounds.normalize(1) == pytest.approx(0)
        assert bounds.normalize(1000) == pytest.approx(1)
        # 31.6 is approximately 10^1.5, which is midpoint in log space
        assert bounds.normalize(31.622) == pytest.approx(0.5, rel=0.01)

    def test_denormalize_linear(self) -> None:
        """Test linear denormalization."""
        bounds = ParameterBounds("R1", 0, 100)
        assert bounds.denormalize(0) == 0
        assert bounds.denormalize(0.5) == 50
        assert bounds.denormalize(1.0) == 100

    def test_denormalize_log(self) -> None:
        """Test log denormalization."""
        bounds = ParameterBounds("R1", 1, 1000, scale="log")
        assert bounds.denormalize(0) == pytest.approx(1)
        assert bounds.denormalize(1) == pytest.approx(1000)
        assert bounds.denormalize(0.5) == pytest.approx(31.622, rel=0.01)


# =============================================================================
# Test OptimizationResult
# =============================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_basic_creation(self) -> None:
        """Test basic result creation."""
        result = OptimizationResult(
            success=True,
            value=1.5,
            parameters={"R1": 1000, "R2": 2000},
            n_iterations=10,
            n_evaluations=50,
        )
        assert result.success is True
        assert result.value == 1.5
        assert result.parameters["R1"] == 1000
        assert result.n_iterations == 10
        assert result.n_evaluations == 50

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        result = OptimizationResult(
            success=True,
            value=1.5,
            parameters={"R1": 1000},
            n_iterations=10,
            n_evaluations=50,
            message="Converged",
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["value"] == 1.5
        assert d["parameters"]["R1"] == 1000
        assert d["message"] == "Converged"

    def test_str_representation(self) -> None:
        """Test string representation."""
        result = OptimizationResult(
            success=True,
            value=1.5,
            parameters={"R1": 1000},
            n_iterations=10,
            n_evaluations=50,
        )
        s = str(result)
        assert "SUCCESS" in s
        assert "1.5" in s
        assert "R1" in s


# =============================================================================
# Test Scipy Optimizers
# =============================================================================


def rosenbrock(params: dict[str, float]) -> float:
    """Rosenbrock function for testing optimization."""
    x = params["x"]
    y = params["y"]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def sphere(params: dict[str, float]) -> float:
    """Simple sphere function for testing."""
    x = params["x"]
    y = params["y"]
    return x**2 + y**2


@scipy_required
class TestScipyOptimizers:
    """Tests for scipy-based optimizers."""

    @pytest.fixture
    def sphere_bounds(self) -> list[ParameterBounds]:
        """Bounds for sphere function optimization."""
        return [
            ParameterBounds("x", -5, 5, initial=2),
            ParameterBounds("y", -5, 5, initial=2),
        ]

    @pytest.fixture
    def rosenbrock_bounds(self) -> list[ParameterBounds]:
        """Bounds for Rosenbrock function optimization."""
        return [
            ParameterBounds("x", -2, 2, initial=0),
            ParameterBounds("y", -2, 2, initial=0),
        ]

    def test_nelder_mead_sphere(self, sphere_bounds: list[ParameterBounds]) -> None:
        """Test Nelder-Mead on sphere function."""
        optimizer = NelderMeadOptimizer()
        config = OptimizationConfig(max_iterations=100, tolerance=1e-6)

        result = optimizer.optimize(sphere, sphere_bounds, config=config)

        assert result.success
        assert result.value < 1e-4  # Should be close to 0
        assert abs(result.parameters["x"]) < 0.1
        assert abs(result.parameters["y"]) < 0.1

    def test_powell_sphere(self, sphere_bounds: list[ParameterBounds]) -> None:
        """Test Powell on sphere function."""
        optimizer = PowellOptimizer()
        config = OptimizationConfig(max_iterations=100)

        result = optimizer.optimize(sphere, sphere_bounds, config=config)

        assert result.success
        assert result.value < 1e-4

    def test_lbfgsb_sphere(self, sphere_bounds: list[ParameterBounds]) -> None:
        """Test L-BFGS-B on sphere function."""
        optimizer = LBFGSBOptimizer()
        config = OptimizationConfig(max_iterations=100)

        result = optimizer.optimize(sphere, sphere_bounds, config=config)

        assert result.success
        assert result.value < 1e-4

    def test_differential_evolution(self, sphere_bounds: list[ParameterBounds]) -> None:
        """Test differential evolution on sphere function."""
        optimizer = DifferentialEvolutionOptimizer(popsize=10)
        config = OptimizationConfig(max_iterations=50, seed=42)

        result = optimizer.optimize(sphere, sphere_bounds, config=config)

        assert result.success
        assert result.value < 0.01

    def test_nelder_mead_rosenbrock(self, rosenbrock_bounds: list[ParameterBounds]) -> None:
        """Test Nelder-Mead on Rosenbrock (harder problem)."""
        optimizer = NelderMeadOptimizer()
        config = OptimizationConfig(max_iterations=500, tolerance=1e-8)

        result = optimizer.optimize(rosenbrock, rosenbrock_bounds, config=config)

        # Nelder-Mead should get close to (1, 1) for Rosenbrock
        assert result.value < 0.1  # May not fully converge
        assert result.n_evaluations > 0

    def test_optimizer_history(self, sphere_bounds: list[ParameterBounds]) -> None:
        """Test that optimization history is recorded."""
        optimizer = NelderMeadOptimizer()
        config = OptimizationConfig(max_iterations=50)

        result = optimizer.optimize(sphere, sphere_bounds, config=config)

        # Should have some history entries
        assert len(result.history) > 0
        # History should show improvement
        if len(result.history) > 1:
            first_value = result.history[0][1]
            last_value = result.history[-1][1]
            # Last should be as good or better (lower)
            assert last_value <= first_value + 1e-6

    def test_get_scipy_optimizer(self) -> None:
        """Test optimizer factory function."""
        nm = get_scipy_optimizer("nelder-mead")
        assert isinstance(nm, NelderMeadOptimizer)

        de = get_scipy_optimizer("differential-evolution")
        assert isinstance(de, DifferentialEvolutionOptimizer)

    def test_get_invalid_optimizer(self) -> None:
        """Test that invalid optimizer raises error."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_scipy_optimizer("invalid-method")

    def test_list_optimizers(self) -> None:
        """Test listing available optimizers."""
        optimizers = list_scipy_optimizers()
        assert "nelder-mead" in optimizers
        assert "differential-evolution" in optimizers
        assert "l-bfgs-b" in optimizers


# =============================================================================
# Test CircuitObjective
# =============================================================================


class TestCircuitObjective:
    """Tests for CircuitObjective class."""

    def test_basic_evaluation(self) -> None:
        """Test basic objective evaluation."""
        # Mock circuit
        mock_circuit = MagicMock()
        mock_circuit.copy.return_value = mock_circuit
        mock_circuit.get_component.return_value = MagicMock()

        # Mock simulation
        def mock_simulate(circuit: object) -> xr.Dataset:
            return xr.Dataset({"vout": (["time"], [1, 2, 3])}, coords={"time": [0, 1, 2]})

        # Mock evaluation
        def mock_evaluate(params: dict[str, float], dataset: xr.Dataset) -> float:
            return params["R1"] / 1000  # Simple objective

        obj = CircuitObjective(mock_circuit, mock_simulate, mock_evaluate)
        value = obj({"R1": 5000})

        assert value == 5.0
        assert obj.n_evaluations == 1

    def test_caching(self) -> None:
        """Test that results are cached."""
        mock_circuit = MagicMock()
        mock_circuit.copy.return_value = mock_circuit
        mock_circuit.get_component.return_value = MagicMock()

        eval_count = [0]

        def mock_simulate(circuit: object) -> xr.Dataset:
            eval_count[0] += 1
            return xr.Dataset()

        def mock_evaluate(params: dict[str, float], dataset: xr.Dataset) -> float:
            return params["R1"]

        obj = CircuitObjective(mock_circuit, mock_simulate, mock_evaluate, cache_results=True)

        # First call
        obj({"R1": 1000})
        assert eval_count[0] == 1

        # Second call with same params - should use cache
        obj({"R1": 1000})
        assert eval_count[0] == 1  # No additional simulation
        assert obj.n_cache_hits == 1

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        mock_circuit = MagicMock()
        mock_circuit.copy.return_value = mock_circuit
        mock_circuit.get_component.return_value = MagicMock()

        def mock_simulate(circuit: object) -> xr.Dataset:
            return xr.Dataset()

        def mock_evaluate(params: dict[str, float], dataset: xr.Dataset) -> float:
            return params["R1"]

        obj = CircuitObjective(mock_circuit, mock_simulate, mock_evaluate)
        obj({"R1": 1000})
        obj.clear_cache()

        # Cache should be empty - will evaluate again
        obj({"R1": 1000})
        assert obj.n_evaluations == 2


# =============================================================================
# Test CircuitOptimizer
# =============================================================================


@scipy_required
class TestCircuitOptimizer:
    """Tests for CircuitOptimizer high-level interface."""

    def test_add_parameters(self) -> None:
        """Test adding parameters."""
        mock_circuit = MagicMock()

        def mock_simulate(circuit: object) -> xr.Dataset:
            return xr.Dataset()

        opt = CircuitOptimizer(mock_circuit, mock_simulate)
        opt.add_parameter("R1", 100, 10000, scale="log")
        opt.add_parameter("C1", 1e-12, 1e-9)

        assert len(opt.bounds) == 2
        assert opt.bounds[0].name == "R1"
        assert opt.bounds[0].scale == "log"
        assert opt.bounds[1].name == "C1"

    def test_method_chaining(self) -> None:
        """Test method chaining for parameter addition."""
        mock_circuit = MagicMock()

        def mock_simulate(circuit: object) -> xr.Dataset:
            return xr.Dataset()

        opt = (
            CircuitOptimizer(mock_circuit, mock_simulate)
            .add_parameter("R1", 100, 10000)
            .add_parameter("R2", 100, 10000)
            .add_parameter("C1", 1e-12, 1e-9)
        )

        assert len(opt.bounds) == 3

    def test_full_optimization(self) -> None:
        """Test full optimization workflow with mock circuit."""
        # This test simulates a complete optimization

        mock_circuit = MagicMock()
        mock_circuit.copy.return_value = mock_circuit
        mock_component = MagicMock()
        mock_circuit.get_component.return_value = mock_component

        # Simulate RC filter with R*C time constant
        def mock_simulate(circuit: object) -> xr.Dataset:
            # Just return a dataset - actual values come from objective
            return xr.Dataset({"tau": ([], 0.001)})

        opt = CircuitOptimizer(mock_circuit, mock_simulate)
        opt.add_parameter("R1", 100, 10000, initial=1000)
        opt.add_parameter("C1", 1e-12, 1e-9, initial=1e-10)

        # Objective: R * C should equal target (1e-6)
        def objective(params: dict[str, float], dataset: xr.Dataset) -> float:
            tau = params["R1"] * params["C1"]
            target = 1e-6
            return (tau - target) ** 2

        config = OptimizationConfig(max_iterations=50, verbose=False)
        result = opt.optimize(objective, method="nelder-mead", config=config)

        # Should find some reasonable solution
        assert result.n_evaluations > 0
        assert "n_simulations" in result.metadata


# =============================================================================
# Test OptimizationConfig
# =============================================================================


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OptimizationConfig()
        assert config.max_iterations == 100
        assert config.max_evaluations == 1000
        assert config.tolerance == 1e-6
        assert config.verbose is False
        assert config.parallel is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = OptimizationConfig(
            max_iterations=500,
            tolerance=1e-8,
            verbose=True,
            seed=42,
        )
        assert config.max_iterations == 500
        assert config.tolerance == 1e-8
        assert config.verbose is True
        assert config.seed == 42


# =============================================================================
# Test Constraint Builders
# =============================================================================


class TestConstraintBuilders:
    """Tests for constraint builder functions."""

    def test_inequality_constraint(self) -> None:
        """Test inequality constraint creation."""
        # R1 >= 2 * R2
        constraint = make_inequality_constraint(
            lambda p: p["R1"] - 2 * p["R2"],
            "R1 >= 2*R2",
        )

        # Should be satisfied when R1 > 2*R2
        assert constraint({"R1": 3000, "R2": 1000}) >= 0
        # Should not be satisfied when R1 < 2*R2
        assert constraint({"R1": 1000, "R2": 1000}) < 0

    def test_equality_constraint(self) -> None:
        """Test equality constraint creation."""
        # R1 * C1 = 1e-6
        constraint = make_equality_constraint(
            lambda p: p["R1"] * p["C1"] - 1e-6,
            tolerance=1e-9,
            name="RC = 1us",
        )

        # Should be satisfied within tolerance
        assert constraint({"R1": 1000, "C1": 1e-9}) >= 0  # RC = 1e-6
        # Should not be satisfied outside tolerance
        assert constraint({"R1": 1000, "C1": 2e-9}) < 0  # RC = 2e-6

    def test_bound_constraint_lower(self) -> None:
        """Test lower bound constraint."""
        constraints = make_bound_constraint("R1", lower=100)

        assert len(constraints) == 1
        assert constraints[0]({"R1": 200}) >= 0
        assert constraints[0]({"R1": 50}) < 0

    def test_bound_constraint_upper(self) -> None:
        """Test upper bound constraint."""
        constraints = make_bound_constraint("R1", upper=10000)

        assert len(constraints) == 1
        assert constraints[0]({"R1": 5000}) >= 0
        assert constraints[0]({"R1": 15000}) < 0

    def test_bound_constraint_both(self) -> None:
        """Test both bound constraints."""
        constraints = make_bound_constraint("R1", lower=100, upper=10000)

        assert len(constraints) == 2
        # Within bounds
        assert all(c({"R1": 1000}) >= 0 for c in constraints)
        # Below lower
        assert any(c({"R1": 50}) < 0 for c in constraints)
        # Above upper
        assert any(c({"R1": 20000}) < 0 for c in constraints)


# =============================================================================
# Test Convergence Tracking
# =============================================================================


class TestConvergenceData:
    """Tests for ConvergenceData class."""

    def test_empty_data(self) -> None:
        """Test empty convergence data."""
        data = ConvergenceData()
        assert data.n_iterations == 0
        assert np.isnan(data.final_value)
        assert np.isnan(data.best_value)

    def test_add_points(self) -> None:
        """Test adding convergence points."""
        data = ConvergenceData()
        data.add_point(0, 10.0, {"x": 1.0})
        data.add_point(1, 8.0, {"x": 0.8})
        data.add_point(2, 5.0, {"x": 0.5})

        assert data.n_iterations == 3
        assert data.final_value == 5.0
        assert data.best_value == 5.0
        assert data.improvement == 5.0

    def test_best_value_tracking(self) -> None:
        """Test that best value is tracked correctly even with fluctuations."""
        data = ConvergenceData()
        data.add_point(0, 10.0, {"x": 1.0})
        data.add_point(1, 5.0, {"x": 0.5})
        data.add_point(2, 8.0, {"x": 0.8})  # Goes up
        data.add_point(3, 3.0, {"x": 0.3})

        # Best values should be monotonically non-increasing
        assert data.best_values == [10.0, 5.0, 5.0, 3.0]
        assert data.best_value == 3.0

    def test_to_dict(self) -> None:
        """Test serialization."""
        data = ConvergenceData()
        data.add_point(0, 10.0, {"x": 1.0})
        data.add_point(1, 5.0, {"x": 0.5})

        d = data.to_dict()
        assert "iterations" in d
        assert "values" in d
        assert "best_values" in d


class TestConvergenceTracker:
    """Tests for ConvergenceTracker class."""

    def test_record_and_check(self) -> None:
        """Test recording iterations."""
        tracker = ConvergenceTracker(tolerance=1e-6)
        tracker.record(0, 10.0, {"x": 1.0})
        tracker.record(1, 5.0, {"x": 0.5})
        tracker.record(2, 2.0, {"x": 0.2})

        assert tracker.data.n_iterations == 3
        assert not tracker.is_stagnant()

    def test_convergence_detection(self) -> None:
        """Test convergence detection."""
        tracker = ConvergenceTracker(tolerance=0.1)

        # Add points that converge
        for i in range(10):
            tracker.record(i, 1.0 + 0.001 * (10 - i), {"x": 1.0})

        assert tracker.is_converged()

    def test_stagnation_detection(self) -> None:
        """Test stagnation detection."""
        tracker = ConvergenceTracker(patience=5, min_improvement=0.01)

        # Add stagnant points
        for i in range(10):
            tracker.record(i, 1.0, {"x": 1.0})

        assert tracker.is_stagnant()

    def test_reset(self) -> None:
        """Test tracker reset."""
        tracker = ConvergenceTracker()
        tracker.record(0, 10.0, {"x": 1.0})
        tracker.reset()

        assert tracker.data.n_iterations == 0


# =============================================================================
# Test Multi-Start Optimization
# =============================================================================


@scipy_required
class TestMultiStartOptimizer:
    """Tests for MultiStartOptimizer."""

    def test_multi_start_sphere(self) -> None:
        """Test multi-start on sphere function."""
        base_optimizer = NelderMeadOptimizer()
        multi = MultiStartOptimizer(base_optimizer, n_starts=5, sampling="random", seed=42)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(max_iterations=50, verbose=False)

        result = multi.optimize(sphere, bounds, config=config)

        assert isinstance(result, MultiStartResult)
        assert result.n_starts == 5
        assert result.best_result.value < 0.1  # Should find near optimum
        assert len(result.all_results) == 5

    def test_multi_start_lhs_sampling(self) -> None:
        """Test LHS sampling."""
        base_optimizer = NelderMeadOptimizer()
        multi = MultiStartOptimizer(base_optimizer, n_starts=8, sampling="lhs", seed=42)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(max_iterations=30, verbose=False)

        result = multi.optimize(sphere, bounds, config=config)

        # LHS should provide good coverage
        assert result.n_starts == 8
        assert len(result.start_points) == 8

    def test_multi_start_grid_sampling(self) -> None:
        """Test grid sampling."""
        base_optimizer = NelderMeadOptimizer()
        multi = MultiStartOptimizer(base_optimizer, n_starts=9, sampling="grid", seed=42)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(max_iterations=30, verbose=False)

        result = multi.optimize(sphere, bounds, config=config)

        assert result.n_starts == 9

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        base_optimizer = NelderMeadOptimizer()
        multi = MultiStartOptimizer(base_optimizer, n_starts=5, seed=42)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(max_iterations=50, verbose=False)

        result = multi.optimize(sphere, bounds, config=config)

        # All should succeed for sphere
        assert result.success_rate > 0.0

    def test_unique_optima(self) -> None:
        """Test finding unique optima."""
        base_optimizer = NelderMeadOptimizer()
        multi = MultiStartOptimizer(base_optimizer, n_starts=5, seed=42)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(max_iterations=50, verbose=False)

        result = multi.optimize(sphere, bounds, config=config)
        unique = result.get_unique_optima(tolerance=0.1)

        # Sphere has single optimum, so should find ~1 unique
        assert len(unique) >= 1


# =============================================================================
# Test Analysis Utilities
# =============================================================================


@scipy_required
class TestAnalysisUtilities:
    """Tests for analysis utility functions."""

    def test_analyze_sensitivity(self) -> None:
        """Test sensitivity analysis at optimum."""
        # Optimize sphere first
        optimizer = NelderMeadOptimizer()
        bounds = [
            ParameterBounds("x", -5, 5, initial=1),
            ParameterBounds("y", -5, 5, initial=1),
        ]
        config = OptimizationConfig(max_iterations=50)
        result = optimizer.optimize(sphere, bounds, config=config)

        # Analyze sensitivity
        sensitivities = analyze_sensitivity(result, sphere, bounds, perturbation=0.01)

        assert "x" in sensitivities
        assert "y" in sensitivities
        # At optimum (0, 0), sensitivities should be small
        # But result may not be exactly at optimum

    def test_compute_hessian_diagonal(self) -> None:
        """Test Hessian diagonal computation."""
        # Optimize sphere first
        optimizer = NelderMeadOptimizer()
        bounds = [
            ParameterBounds("x", -5, 5, initial=1),
            ParameterBounds("y", -5, 5, initial=1),
        ]
        config = OptimizationConfig(max_iterations=50)
        result = optimizer.optimize(sphere, bounds, config=config)

        # Compute Hessian diagonal
        hess_diag = compute_hessian_diagonal(result, sphere, bounds, perturbation=0.01)

        assert "x" in hess_diag
        assert "y" in hess_diag
        # For sphere (x^2 + y^2), Hessian diagonal should be ~2 for each
        assert abs(hess_diag["x"] - 2.0) < 0.5
        assert abs(hess_diag["y"] - 2.0) < 0.5


# =============================================================================
# Test Genetic Algorithms
# =============================================================================


class TestGeneticOptimizer:
    """Tests for GeneticOptimizer class."""

    def test_ga_sphere(self) -> None:
        """Test GA on sphere function."""
        optimizer = GeneticOptimizer(
            population_size=30,
            n_generations=50,
            crossover_prob=0.8,
            mutation_prob=0.1,
        )

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(seed=42, verbose=False)

        result = optimizer.optimize(sphere, bounds, config=config)

        assert result.success
        assert result.value < 0.5  # Should find near optimum
        assert abs(result.parameters["x"]) < 1.0
        assert abs(result.parameters["y"]) < 1.0

    def test_ga_rosenbrock(self) -> None:
        """Test GA on Rosenbrock (harder problem)."""
        optimizer = GeneticOptimizer(
            population_size=50,
            n_generations=100,
        )

        bounds = [
            ParameterBounds("x", -2, 2),
            ParameterBounds("y", -2, 2),
        ]
        config = OptimizationConfig(seed=42, verbose=False)

        result = optimizer.optimize(rosenbrock, bounds, config=config)

        # GA should make progress on Rosenbrock
        assert result.n_evaluations > 0
        assert result.value < 10  # Should improve from random

    def test_ga_with_constraints(self) -> None:
        """Test GA with constraint penalties."""
        optimizer = GeneticOptimizer(population_size=30, n_generations=30)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]

        # Constraint: x + y >= 1
        constraints = [make_inequality_constraint(lambda p: p["x"] + p["y"] - 1)]
        config = OptimizationConfig(seed=42, verbose=False)

        result = optimizer.optimize(sphere, bounds, constraints=constraints, config=config)

        # Solution should satisfy constraint
        assert result.parameters["x"] + result.parameters["y"] >= 0.9  # Allow small tolerance

    def test_ga_history(self) -> None:
        """Test that GA records history."""
        optimizer = GeneticOptimizer(population_size=20, n_generations=20)

        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        config = OptimizationConfig(seed=42)

        result = optimizer.optimize(sphere, bounds, config=config)

        assert len(result.history) > 0
        # History should show improvement
        if len(result.history) > 1:
            assert result.history[-1][1] <= result.history[0][1]

    def test_get_genetic_optimizer(self) -> None:
        """Test factory function."""
        ga = get_genetic_optimizer("ga")
        assert isinstance(ga, GeneticOptimizer)

        nsga = get_genetic_optimizer("nsga2")
        assert isinstance(nsga, NSGA2Optimizer)


class TestNSGA2Optimizer:
    """Tests for NSGA2Optimizer class."""

    def test_nsga2_simple(self) -> None:
        """Test NSGA-II on simple bi-objective problem."""
        # Objectives: minimize x^2, minimize (x-2)^2

        def bi_objective(params: dict[str, float]) -> tuple[float, float]:
            x = params["x"]
            return (x**2, (x - 2) ** 2)

        optimizer = NSGA2Optimizer(
            population_size=30,
            n_generations=30,
        )

        bounds = [ParameterBounds("x", -1, 3)]
        config = OptimizationConfig(seed=42, verbose=False)

        result = optimizer.optimize(bi_objective, bounds, n_objectives=2, config=config)

        assert isinstance(result, MultiObjectiveResult)
        assert result.success
        assert len(result.pareto_front) > 0

    def test_nsga2_pareto_front(self) -> None:
        """Test Pareto front properties."""
        # 2D problem: minimize f1 = x^2 + y^2, f2 = (x-1)^2 + (y-1)^2

        def two_spheres(params: dict[str, float]) -> tuple[float, float]:
            x, y = params["x"], params["y"]
            return (x**2 + y**2, (x - 1) ** 2 + (y - 1) ** 2)

        optimizer = NSGA2Optimizer(population_size=50, n_generations=50)
        bounds = [
            ParameterBounds("x", -1, 2),
            ParameterBounds("y", -1, 2),
        ]
        config = OptimizationConfig(seed=42, verbose=False)

        result = optimizer.optimize(two_spheres, bounds, n_objectives=2, config=config)

        # Should find multiple Pareto solutions
        assert len(result.pareto_front) >= 2

        # Test knee point
        knee = result.pareto_front.get_knee_point()
        assert "x" in knee
        assert "y" in knee

    def test_pareto_front_extremes(self) -> None:
        """Test getting extreme solutions from Pareto front."""
        pareto = ParetoFront(
            solutions=[{"x": 0}, {"x": 0.5}, {"x": 1}],
            objectives=[(0.0, 1.0), (0.25, 0.25), (1.0, 0.0)],
            n_objectives=2,
        )

        # Get extreme for objective 0 (minimize)
        min_obj0 = pareto.get_extreme(0, minimize=True)
        assert min_obj0["x"] == 0  # First solution has min obj0

        # Get extreme for objective 1 (minimize)
        min_obj1 = pareto.get_extreme(1, minimize=True)
        assert min_obj1["x"] == 1  # Third solution has min obj1

    def test_pareto_front_to_dict(self) -> None:
        """Test Pareto front serialization."""
        pareto = ParetoFront(
            solutions=[{"x": 1.0}, {"x": 2.0}],
            objectives=[(1.0, 2.0), (2.0, 1.0)],
            n_objectives=2,
        )

        d = pareto.to_dict()
        assert "solutions" in d
        assert "objectives" in d
        assert d["n_objectives"] == 2

    def test_multi_objective_result(self) -> None:
        """Test MultiObjectiveResult properties."""
        pareto = ParetoFront(
            solutions=[{"x": 0.5}],
            objectives=[(0.25, 0.25)],
            n_objectives=2,
        )

        result = MultiObjectiveResult(
            pareto_front=pareto,
            n_generations=10,
            n_evaluations=100,
        )

        assert result.success
        compromise = result.get_best_compromise()
        assert compromise["x"] == 0.5


class TestGAConfig:
    """Tests for GAConfig class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = GAConfig()
        assert config.population_size == 50
        assert config.n_generations == 100
        assert config.crossover_prob == 0.8
        assert config.mutation_prob == 0.1
        assert config.tournament_size == 3
        assert config.elitism == 2
        assert config.seed is None
        assert config.verbose is False

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = GAConfig(
            population_size=100,
            n_generations=200,
            crossover_prob=0.9,
            mutation_prob=0.05,
            seed=42,
        )
        assert config.population_size == 100
        assert config.n_generations == 200
        assert config.crossover_prob == 0.9
        assert config.mutation_prob == 0.05
        assert config.seed == 42


# =============================================================================
# Test Design of Experiments
# =============================================================================


class TestFullFactorial:
    """Tests for full factorial design."""

    def test_two_level(self) -> None:
        """Test 2-level full factorial."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = full_factorial(bounds, levels=2)

        assert design.n_runs == 4  # 2^2
        assert design.n_factors == 2
        assert design.design_type == "full_factorial_2^2"

        # Check all corners present
        values_x = {p["x"] for p in design.points}
        values_y = {p["y"] for p in design.points}
        assert values_x == {0, 10}
        assert values_y == {0, 20}

    def test_three_level(self) -> None:
        """Test 3-level full factorial."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = full_factorial(bounds, levels=3)

        assert design.n_runs == 9  # 3^2
        values_x = {p["x"] for p in design.points}
        assert 5.0 in values_x  # Midpoint

    def test_to_array(self) -> None:
        """Test conversion to numpy array."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = full_factorial(bounds, levels=2)

        arr = design.to_array()
        assert arr.shape == (4, 2)

    def test_to_normalized(self) -> None:
        """Test conversion to normalized array."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = full_factorial(bounds, levels=2)

        norm = design.to_normalized()
        assert norm.min() >= 0
        assert norm.max() <= 1


class TestFractionalFactorial:
    """Tests for fractional factorial design."""

    def test_small_design(self) -> None:
        """Test fractional factorial for small number of factors."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = fractional_factorial(bounds, resolution=3)

        # Should fall back to full factorial for 2 factors
        assert design.n_runs == 4

    def test_large_design(self) -> None:
        """Test fractional factorial for more factors."""
        bounds = [ParameterBounds(f"x{i}", 0, 10) for i in range(5)]
        design = fractional_factorial(bounds, resolution=3)

        # Should be less than full factorial (2^5 = 32)
        assert design.n_runs < 32
        assert design.n_factors == 5


class TestLatinHypercube:
    """Tests for Latin Hypercube Sampling."""

    def test_basic_lhs(self) -> None:
        """Test basic LHS generation."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = latin_hypercube(bounds, n_samples=10, seed=42)

        assert design.n_runs == 10
        assert design.design_type == "lhs_maximin"

    def test_lhs_coverage(self) -> None:
        """Test that LHS covers the space."""
        bounds = [ParameterBounds("x", 0, 100)]
        design = latin_hypercube(bounds, n_samples=10, seed=42)

        values = [p["x"] for p in design.points]
        # Should have values spread across the range
        assert min(values) < 20
        assert max(values) > 80

    def test_lhs_criteria(self) -> None:
        """Test different LHS criteria."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]

        for criterion in ["random", "maximin", "correlation"]:
            design = latin_hypercube(bounds, n_samples=10, criterion=criterion, seed=42)
            assert design.n_runs == 10
            assert criterion in design.design_type


class TestSobolSequence:
    """Tests for Sobol sequence."""

    def test_basic_sobol(self) -> None:
        """Test basic Sobol sequence generation."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = sobol_sequence(bounds, n_samples=16)

        assert design.n_runs == 16
        assert design.design_type == "sobol"

    def test_sobol_coverage(self) -> None:
        """Test Sobol sequence space coverage."""
        bounds = [ParameterBounds("x", 0, 100)]
        design = sobol_sequence(bounds, n_samples=32)

        values = sorted([p["x"] for p in design.points])
        # Sobol should give better coverage than random
        # Check for reasonable spread
        assert max(values) - min(values) > 50


class TestCentralComposite:
    """Tests for Central Composite Design."""

    def test_ccd_face(self) -> None:
        """Test face-centered CCD."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = central_composite(bounds, alpha="face", center_points=3)

        # 2^2 factorial + 2*2 axial + 3 center = 4 + 4 + 3 = 11
        assert design.n_runs == 11

    def test_ccd_contains_center(self) -> None:
        """Test that CCD contains center points."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = central_composite(bounds, center_points=3)

        center_count = sum(1 for p in design.points if p["x"] == 5 and p["y"] == 10)
        assert center_count == 3


class TestBoxBehnken:
    """Tests for Box-Behnken design."""

    def test_basic_bb(self) -> None:
        """Test basic Box-Behnken design."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
            ParameterBounds("z", 0, 30),
        ]
        design = box_behnken(bounds, center_points=3)

        # BB for 3 factors: 12 edge points + 3 center = 15
        assert design.n_runs == 15

    def test_bb_requires_3_factors(self) -> None:
        """Test that BB requires at least 3 factors."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        with pytest.raises(ValueError, match="at least 3 factors"):
            box_behnken(bounds)


class TestResponseSurface:
    """Tests for response surface modeling."""

    def test_fit_linear(self) -> None:
        """Test fitting linear response surface."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 20),
        ]
        design = full_factorial(bounds, levels=3)

        # Linear response: y = 2x + 3y + 5
        def objective(params: dict[str, float]) -> float:
            return 2 * params["x"] + 3 * params["y"] + 5

        result = run_doe(design, objective)
        surface = fit_response_surface(result, model_type="linear")

        assert surface.r_squared > 0.99  # Perfect linear fit
        assert "intercept" in surface.coefficients
        assert abs(surface.coefficients["x"] - 2) < 0.1
        assert abs(surface.coefficients["y"] - 3) < 0.1

    def test_fit_quadratic(self) -> None:
        """Test fitting quadratic response surface."""
        bounds = [
            ParameterBounds("x", -5, 5),
            ParameterBounds("y", -5, 5),
        ]
        design = central_composite(bounds, center_points=5)

        # Quadratic response: f = x^2 + y^2
        def objective(params: dict[str, float]) -> float:
            return params["x"] ** 2 + params["y"] ** 2

        result = run_doe(design, objective)
        surface = fit_response_surface(result, model_type="quadratic")

        assert surface.r_squared > 0.95
        # Should have quadratic terms
        assert "x^2" in surface.coefficients
        assert "y^2" in surface.coefficients

    def test_predict(self) -> None:
        """Test response surface prediction."""
        surface = ResponseSurface(
            coefficients={"intercept": 5, "x": 2, "y": 3},
            factor_names=["x", "y"],
            model_type="linear",
            r_squared=1.0,
        )

        pred = surface.predict({"x": 1, "y": 2})
        assert pred == 5 + 2 * 1 + 3 * 2  # 13

    def test_get_optimum(self) -> None:
        """Test finding optimum of response surface."""
        bounds = [
            ParameterBounds("x", -10, 10),
            ParameterBounds("y", -10, 10),
        ]
        surface = ResponseSurface(
            coefficients={"intercept": 0, "x^2": 1, "y^2": 1},
            factor_names=["x", "y"],
            model_type="quadratic",
            r_squared=1.0,
        )

        opt = surface.get_optimum(bounds, minimize=True, n_samples=1000)
        # Optimum should be near (0, 0)
        assert abs(opt["x"]) < 2
        assert abs(opt["y"]) < 2


class TestRunDoe:
    """Tests for DoE runner."""

    def test_run_doe(self) -> None:
        """Test running DoE experiments."""
        bounds = [ParameterBounds("x", 0, 10)]
        design = full_factorial(bounds, levels=3)

        def objective(params: dict[str, float]) -> float:
            return params["x"] ** 2

        result = run_doe(design, objective)

        assert isinstance(result, DoEResult)
        assert result.n_runs == 3
        assert len(result.responses) == 3

    def test_get_best(self) -> None:
        """Test getting best result from DoE."""
        bounds = [ParameterBounds("x", 0, 10)]
        design = full_factorial(bounds, levels=3)

        def objective(params: dict[str, float]) -> float:
            return (params["x"] - 5) ** 2  # Minimum at x=5

        result = run_doe(design, objective)
        best_params, best_value = result.get_best(minimize=True)

        assert best_params["x"] == 5
        assert best_value == 0


# =============================================================================
# Test Sensitivity Analysis
# =============================================================================


def linear_model_2d(params: dict[str, float]) -> float:
    """2D Linear model: y = 2*x1 + 3*x2."""
    return 2 * params["x1"] + 3 * params["x2"]


def linear_model_3d(params: dict[str, float]) -> float:
    """3D Linear model: y = 2*x1 + 3*x2 + x3."""
    return 2 * params["x1"] + 3 * params["x2"] + params["x3"]


def nonlinear_model(params: dict[str, float]) -> float:
    """Nonlinear model with interaction: y = x1^2 + x1*x2 + x2."""
    return params["x1"] ** 2 + params["x1"] * params["x2"] + params["x2"]


class TestMorrisAnalysis:
    """Tests for Morris sensitivity analysis."""

    def test_morris_basic(self) -> None:
        """Test basic Morris analysis."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = morris_analysis(linear_model_3d, bounds, n_trajectories=10, seed=42)

        assert isinstance(result, MorrisResult)
        assert len(result.factor_names) == 3
        assert "x1" in result.mu_star
        assert "x2" in result.mu_star
        assert "x3" in result.mu_star

    def test_morris_ranking(self) -> None:
        """Test that Morris correctly ranks factors."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = morris_analysis(linear_model_3d, bounds, n_trajectories=20, seed=42)
        ranking = result.get_ranking()

        # x2 should be most important (coef=3), then x1 (coef=2), then x3 (coef=1)
        assert ranking[0] == "x2"
        assert ranking[1] == "x1"
        assert ranking[2] == "x3"

    def test_morris_detects_nonlinearity(self) -> None:
        """Test that Morris detects nonlinear/interaction effects."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = morris_analysis(nonlinear_model, bounds, n_trajectories=20, seed=42)

        # x1 has nonlinear effect (x1^2) and interaction (x1*x2)
        # Should have higher sigma for x1
        assert result.sigma["x1"] > 0

    def test_morris_is_influential(self) -> None:
        """Test influential factor detection."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = morris_analysis(linear_model_3d, bounds, n_trajectories=15, seed=42)

        # All factors should be influential (all have non-zero coefficients)
        assert result.is_influential("x1")
        assert result.is_influential("x2")
        assert result.is_influential("x3")

    def test_morris_to_dict(self) -> None:
        """Test conversion to dictionary."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = morris_analysis(linear_model_3d, bounds, n_trajectories=5, seed=42)
        d = result.to_dict()

        assert "mu" in d
        assert "mu_star" in d
        assert "sigma" in d


class TestSobolAnalysis:
    """Tests for Sobol sensitivity analysis."""

    def test_sobol_basic(self) -> None:
        """Test basic Sobol analysis."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = sobol_analysis(linear_model_3d, bounds, n_samples=256, seed=42)

        assert isinstance(result, SobolResult)
        assert len(result.factor_names) == 3
        assert "x1" in result.S1
        assert "x1" in result.ST

    def test_sobol_linear_model(self) -> None:
        """Test Sobol on linear model (no interactions)."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = sobol_analysis(linear_model_2d, bounds, n_samples=512, seed=42)

        # For linear model, S1 should be close to ST (no interactions)
        for name in result.factor_names:
            interaction = result.get_interaction_strength(name)
            assert interaction < 0.2  # Small interaction effect

    def test_sobol_ranking(self) -> None:
        """Test Sobol ranking of factors."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = sobol_analysis(linear_model_3d, bounds, n_samples=512, seed=42)
        ranking = result.get_main_effects_ranking()

        # x2 has highest coefficient (3), should be first
        assert ranking[0] == "x2"

    def test_sobol_indices_in_range(self) -> None:
        """Test that Sobol indices are in valid range [0, 1]."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = sobol_analysis(linear_model_2d, bounds, n_samples=256, seed=42)

        for name in result.factor_names:
            assert 0 <= result.S1[name] <= 1
            assert 0 <= result.ST[name] <= 1

    def test_sobol_second_order(self) -> None:
        """Test second-order Sobol indices."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = sobol_analysis(
            nonlinear_model, bounds, n_samples=256, calc_second_order=True, seed=42
        )

        assert result.S2 is not None
        assert ("x1", "x2") in result.S2

    def test_sobol_to_dict(self) -> None:
        """Test conversion to dictionary."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = sobol_analysis(linear_model_2d, bounds, n_samples=128, seed=42)
        d = result.to_dict()

        assert "S1" in d
        assert "ST" in d


class TestLocalSensitivity:
    """Tests for local sensitivity analysis."""

    def test_local_basic(self) -> None:
        """Test basic local sensitivity."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = local_sensitivity(linear_model_3d, bounds)

        assert isinstance(result, LocalSensitivity)
        assert len(result.gradients) == 3

    def test_local_gradients(self) -> None:
        """Test that local gradients are correct for linear model."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = local_sensitivity(linear_model_3d, bounds)

        # Gradients should match coefficients: 2, 3, 1
        assert abs(result.gradients["x1"] - 2) < 0.1
        assert abs(result.gradients["x2"] - 3) < 0.1
        assert abs(result.gradients["x3"] - 1) < 0.1

    def test_local_at_point(self) -> None:
        """Test local sensitivity at specific point."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        point = {"x1": 2.0, "x2": 3.0}
        result = local_sensitivity(linear_model_2d, bounds, point=point)

        # Should evaluate at the specified point
        assert result.point["x1"] == 2.0
        assert result.point["x2"] == 3.0

    def test_local_ranking(self) -> None:
        """Test ranking by sensitivity."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = local_sensitivity(linear_model_3d, bounds)
        ranking = result.get_ranking(normalized=False)

        # x2 has highest gradient (3)
        assert ranking[0] == "x2"


class TestOATAnalysis:
    """Tests for one-at-a-time analysis."""

    def test_oat_basic(self) -> None:
        """Test basic OAT analysis."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = oat_analysis(linear_model_2d, bounds, n_points=5)

        assert isinstance(result, OATResult)
        assert "x1" in result.sweeps
        assert "x2" in result.sweeps

    def test_oat_sweeps(self) -> None:
        """Test OAT sweep data."""
        bounds = [ParameterBounds("x1", 0, 10)]

        # Use a simple 1D linear function for this test
        def linear_1d(params: dict[str, float]) -> float:
            return 2 * params["x1"]

        result = oat_analysis(linear_1d, bounds, n_points=11)

        params, responses = result.sweeps["x1"]
        assert len(params) == 11
        assert len(responses) == 11

    def test_oat_sensitivity(self) -> None:
        """Test OAT sensitivity estimation."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = oat_analysis(linear_model_2d, bounds, n_points=11)

        # Sensitivity should approximate the coefficients
        assert abs(result.get_sensitivity("x1") - 2) < 0.5
        assert abs(result.get_sensitivity("x2") - 3) < 0.5

    def test_oat_ranking(self) -> None:
        """Test OAT ranking."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
            ParameterBounds("x3", 0, 10),
        ]

        result = oat_analysis(linear_model_3d, bounds, n_points=11)
        ranking = result.get_ranking()

        # x2 should have largest range (coefficient 3)
        assert ranking[0] == "x2"


class TestVisualizationHelpers:
    """Tests for visualization helpers."""

    def test_tornado_data_morris(self) -> None:
        """Test tornado data generation for Morris."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        morris_result = morris_analysis(linear_model_2d, bounds, n_trajectories=5, seed=42)
        tornado = generate_tornado_data(morris_result)

        assert "data" in tornado
        assert len(tornado["data"]) == 2

    def test_tornado_data_sobol(self) -> None:
        """Test tornado data generation for Sobol."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        sobol_result = sobol_analysis(linear_model_2d, bounds, n_samples=128, seed=42)
        tornado = generate_tornado_data(sobol_result)

        assert "data" in tornado
        assert len(tornado["data"]) == 2

    def test_print_sensitivity_report_morris(self) -> None:
        """Test Morris report generation."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = morris_analysis(linear_model_2d, bounds, n_trajectories=5, seed=42)
        report = print_sensitivity_report(result)

        assert "Morris" in report
        assert "x1" in report
        assert "x2" in report

    def test_print_sensitivity_report_sobol(self) -> None:
        """Test Sobol report generation."""
        bounds = [
            ParameterBounds("x1", 0, 10),
            ParameterBounds("x2", 0, 10),
        ]

        result = sobol_analysis(linear_model_2d, bounds, n_samples=128, seed=42)
        report = print_sensitivity_report(result)

        assert "Sobol" in report
        assert "S1" in report
        assert "ST" in report


# =============================================================================
# Test Corner Analysis
# =============================================================================

from spicelab.optimization.corner import (
    Corner,
    CornerAnalysis,
    CornerAnalysisResult,
    CornerDefinition,
    CornerResult,
    PVTCondition,
    PVTSweep,
    StatisticalCornerResult,
    WorstCaseResult,
    print_corner_report,
    statistical_corner_analysis,
    worst_case_analysis,
)


class TestCorner:
    """Tests for Corner class."""

    def test_basic_corner(self) -> None:
        """Test basic corner creation."""
        corner = Corner(
            name="test_corner",
            parameters={"R1": 1000, "C1": 1e-9},
        )
        assert corner.name == "test_corner"
        assert corner.parameters["R1"] == 1000
        assert corner.parameters["C1"] == 1e-9

    def test_corner_str(self) -> None:
        """Test corner string representation."""
        corner = Corner(name="test", parameters={"x": 1, "y": 2})
        s = str(corner)
        assert "test" in s
        assert "x=1" in s
        assert "y=2" in s

    def test_corner_to_dict(self) -> None:
        """Test corner serialization."""
        corner = Corner(
            name="test",
            parameters={"x": 1},
            metadata={"type": "typical"},
        )
        d = corner.to_dict()
        assert d["name"] == "test"
        assert d["parameters"]["x"] == 1
        assert d["metadata"]["type"] == "typical"


class TestCornerDefinition:
    """Tests for CornerDefinition class."""

    def test_add_parameter(self) -> None:
        """Test adding parameters."""
        definition = CornerDefinition()
        definition.add_parameter("process", ["slow", "typical", "fast"])
        definition.add_parameter("voltage", [3.0, 3.3, 3.6])

        assert definition.n_parameters == 2
        assert "process" in definition.parameter_names
        assert "voltage" in definition.parameter_names

    def test_method_chaining(self) -> None:
        """Test method chaining."""
        definition = (
            CornerDefinition()
            .add_parameter("process", ["slow", "typical", "fast"])
            .add_parameter("voltage", [3.0, 3.3, 3.6])
        )
        assert definition.n_parameters == 2

    def test_n_corners(self) -> None:
        """Test corner count calculation."""
        definition = CornerDefinition()
        definition.add_parameter("process", ["slow", "typical", "fast"])  # 3
        definition.add_parameter("voltage", [3.0, 3.3, 3.6])  # 3
        definition.add_parameter("temperature", [-40, 25, 85])  # 3

        assert definition.n_corners == 27  # 3 * 3 * 3

    def test_generate_all(self) -> None:
        """Test generating all corners."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 1])
        definition.add_parameter("y", [0, 1])

        corners = definition.generate_all()
        assert len(corners) == 4  # 2 * 2

        # Check all combinations present
        params_set = {tuple(sorted(c.parameters.items())) for c in corners}
        expected = {
            (("x", 0), ("y", 0)),
            (("x", 0), ("y", 1)),
            (("x", 1), ("y", 0)),
            (("x", 1), ("y", 1)),
        }
        assert params_set == expected

    def test_generate_extremes(self) -> None:
        """Test generating extreme corners only."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 5, 10])
        definition.add_parameter("y", [0, 5, 10])

        extremes = definition.generate_extremes()
        assert len(extremes) == 4  # 2^2 extremes

        # Check only min/max values used
        for corner in extremes:
            assert corner.parameters["x"] in [0, 10]
            assert corner.parameters["y"] in [0, 10]

    def test_generate_typical(self) -> None:
        """Test generating typical corner."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 5, 10])
        definition.add_parameter("y", [0, 50, 100])

        typical = definition.generate_typical()
        assert typical.parameters["x"] == 5
        assert typical.parameters["y"] == 50

    def test_add_pvt(self) -> None:
        """Test adding standard PVT parameters."""
        definition = CornerDefinition()
        definition.add_pvt(voltage=[3.0, 3.3, 3.6])

        assert "process" in definition.parameter_names
        assert "voltage" in definition.parameter_names
        assert "temperature" in definition.parameter_names

        assert definition.n_corners == 27  # 3 * 3 * 3


class TestCornerResult:
    """Tests for CornerResult class."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        corner = Corner(name="test", parameters={"x": 1})
        result = CornerResult(corner=corner, value=42.0, success=True)

        assert result.corner.name == "test"
        assert result.value == 42.0
        assert result.success is True

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        corner = Corner(name="test", parameters={"x": 1})
        result = CornerResult(
            corner=corner,
            value=42.0,
            metrics={"gain": 10.0},
        )

        d = result.to_dict()
        assert d["value"] == 42.0
        assert d["metrics"]["gain"] == 10.0


class TestCornerAnalysisResult:
    """Tests for CornerAnalysisResult class."""

    def test_worst_best_case(self) -> None:
        """Test worst/best case detection."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 1])

        results = [
            CornerResult(Corner("c1", {"x": 0}), value=10.0),
            CornerResult(Corner("c2", {"x": 1}), value=20.0),
        ]
        analysis_result = CornerAnalysisResult(results, definition)

        assert analysis_result.worst_case is not None
        assert analysis_result.worst_case.value == 20.0
        assert analysis_result.best_case is not None
        assert analysis_result.best_case.value == 10.0

    def test_statistics(self) -> None:
        """Test statistical summary."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 1, 2])

        results = [
            CornerResult(Corner("c1", {"x": 0}), value=10.0),
            CornerResult(Corner("c2", {"x": 1}), value=20.0),
            CornerResult(Corner("c3", {"x": 2}), value=30.0),
        ]
        analysis_result = CornerAnalysisResult(results, definition)

        stats = analysis_result.get_statistics()
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["mean"] == 20.0
        assert stats["range"] == 20.0

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 1])

        results = [
            CornerResult(Corner("c1", {"x": 0}), value=10.0, success=True),
            CornerResult(Corner("c2", {"x": 1}), value=float("nan"), success=False),
        ]
        analysis_result = CornerAnalysisResult(results, definition)

        assert analysis_result.success_rate == 0.5

    def test_get_by_parameter(self) -> None:
        """Test filtering by parameter value."""
        definition = CornerDefinition()
        definition.add_parameter("process", ["slow", "fast"])
        definition.add_parameter("temp", [-40, 85])

        results = [
            CornerResult(Corner("c1", {"process": "slow", "temp": -40}), value=10.0),
            CornerResult(Corner("c2", {"process": "slow", "temp": 85}), value=15.0),
            CornerResult(Corner("c3", {"process": "fast", "temp": -40}), value=20.0),
            CornerResult(Corner("c4", {"process": "fast", "temp": 85}), value=25.0),
        ]
        analysis_result = CornerAnalysisResult(results, definition)

        slow_results = analysis_result.get_by_parameter("process", "slow")
        assert len(slow_results) == 2

    def test_sensitivity_to_parameter(self) -> None:
        """Test parameter sensitivity calculation."""
        definition = CornerDefinition()
        definition.add_parameter("process", ["slow", "fast"])

        results = [
            CornerResult(Corner("c1", {"process": "slow"}), value=10.0),
            CornerResult(Corner("c2", {"process": "fast"}), value=20.0),
        ]
        analysis_result = CornerAnalysisResult(results, definition)

        sens = analysis_result.sensitivity_to_parameter("process")
        assert sens["slow"] == 10.0
        assert sens["fast"] == 20.0


class TestPVTSweep:
    """Tests for PVTSweep class."""

    def test_basic_pvt(self) -> None:
        """Test basic PVT setup."""
        pvt = PVTSweep()
        pvt.set_process(["slow", "typical", "fast"])
        pvt.set_voltage(3.3, tolerance=0.1, n_points=3)
        pvt.set_temperature("military")

        assert pvt.n_conditions == 27  # 3 * 3 * 3

    def test_generate_all(self) -> None:
        """Test generating all PVT conditions."""
        pvt = PVTSweep()
        pvt.set_process(["slow", "fast"])
        pvt.set_voltage_values([3.0, 3.6])
        pvt.set_temperature([-40, 85])

        conditions = pvt.generate_all()
        assert len(conditions) == 8  # 2 * 2 * 2

    def test_generate_extremes(self) -> None:
        """Test generating extreme PVT conditions."""
        pvt = PVTSweep()
        pvt.set_process(["slow", "typical", "fast"])
        pvt.set_voltage_values([3.0, 3.3, 3.6])
        pvt.set_temperature([-40, 25, 85])

        extremes = pvt.generate_extremes()
        assert len(extremes) == 8  # 2^3 extremes

    def test_temp_presets(self) -> None:
        """Test temperature presets."""
        pvt = PVTSweep()

        pvt.set_temperature("military")
        assert -40 in pvt._temperature
        assert 85 in pvt._temperature

        pvt.set_temperature("commercial")
        assert 0 in pvt._temperature
        assert 70 in pvt._temperature

    def test_to_corner_definition(self) -> None:
        """Test conversion to CornerDefinition."""
        pvt = PVTSweep()
        pvt.set_process(["slow", "fast"])
        pvt.set_voltage_values([3.0, 3.6])
        pvt.set_temperature([-40, 85])

        definition = pvt.to_corner_definition()
        assert definition.n_corners == 8


class TestPVTCondition:
    """Tests for PVTCondition class."""

    def test_basic_condition(self) -> None:
        """Test basic PVT condition."""
        cond = PVTCondition("slow", 3.3, 25)
        assert cond.process == "slow"
        assert cond.voltage == 3.3
        assert cond.temperature == 25
        assert "slow" in cond.name

    def test_to_corner(self) -> None:
        """Test conversion to Corner."""
        cond = PVTCondition("fast", 3.6, 85)
        corner = cond.to_corner()

        assert corner.parameters["process"] == "fast"
        assert corner.parameters["voltage"] == 3.6
        assert corner.parameters["temperature"] == 85


class TestStatisticalCornerAnalysis:
    """Tests for statistical corner analysis."""

    def test_basic_statistical(self) -> None:
        """Test basic statistical analysis."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        def objective(params: dict[str, float]) -> float:
            return params["x"] + params["y"]

        result = statistical_corner_analysis(
            objective, bounds, n_samples=100, seed=42
        )

        assert isinstance(result, StatisticalCornerResult)
        assert result.n_samples == 100
        assert 0 <= result.mean <= 20
        assert result.std > 0

    def test_percentiles(self) -> None:
        """Test percentile calculation."""
        bounds = [ParameterBounds("x", 0, 100)]

        def objective(params: dict[str, float]) -> float:
            return params["x"]

        result = statistical_corner_analysis(
            objective, bounds, n_samples=1000, seed=42, percentiles=(5, 50, 95)
        )

        assert 5 in result.percentiles
        assert 50 in result.percentiles
        assert 95 in result.percentiles

        # For uniform, 50th percentile should be near 50
        assert 40 < result.percentiles[50] < 60

    def test_yield_calculation(self) -> None:
        """Test yield calculation."""
        bounds = [ParameterBounds("x", 0, 100)]

        def objective(params: dict[str, float]) -> float:
            return params["x"]

        result = statistical_corner_analysis(
            objective, bounds, n_samples=1000, seed=42
        )

        # Yield within [25, 75] should be about 50%
        yield_val = result.get_yield(spec_min=25, spec_max=75)
        assert 0.4 < yield_val < 0.6

    def test_cpk_calculation(self) -> None:
        """Test Cpk calculation."""
        bounds = [ParameterBounds("x", 0, 100)]

        def objective(params: dict[str, float]) -> float:
            return params["x"]

        result = statistical_corner_analysis(
            objective, bounds, n_samples=1000, seed=42
        )

        cpk = result.get_cpk(spec_min=0, spec_max=100)
        assert cpk > 0


class TestWorstCaseAnalysis:
    """Tests for worst-case analysis."""

    def test_extremes_method(self) -> None:
        """Test worst-case with extremes method."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        # Maximize x + y
        def objective(params: dict[str, float]) -> float:
            return params["x"] + params["y"]

        result = worst_case_analysis(
            objective, bounds, maximize=True, method="extremes"
        )

        assert isinstance(result, WorstCaseResult)
        assert result.worst_value == 20.0  # max = 10 + 10
        assert result.worst_params["x"] == 10
        assert result.worst_params["y"] == 10

    def test_minimize(self) -> None:
        """Test worst-case minimization."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        def objective(params: dict[str, float]) -> float:
            return params["x"] + params["y"]

        result = worst_case_analysis(
            objective, bounds, maximize=False, method="extremes"
        )

        assert result.worst_value == 0.0  # min = 0 + 0
        assert result.worst_params["x"] == 0
        assert result.worst_params["y"] == 0

    def test_statistical_method(self) -> None:
        """Test worst-case with statistical method."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        def objective(params: dict[str, float]) -> float:
            return params["x"] + params["y"]

        result = worst_case_analysis(
            objective, bounds, maximize=True, method="statistical", n_samples=500, seed=42
        )

        # Should find a value close to maximum
        assert result.worst_value > 15.0

    def test_sensitivity_at_worst(self) -> None:
        """Test sensitivity calculation at worst case."""
        bounds = [
            ParameterBounds("x", 0, 10),
            ParameterBounds("y", 0, 10),
        ]

        def objective(params: dict[str, float]) -> float:
            return 2 * params["x"] + 3 * params["y"]

        # Use minimize=True so worst case is at (0, 0), allowing positive perturbation
        result = worst_case_analysis(
            objective, bounds, maximize=False, method="extremes"
        )

        # Sensitivities should approximate coefficients
        assert abs(result.sensitivity["x"] - 2) < 1
        assert abs(result.sensitivity["y"] - 3) < 1


class TestPrintCornerReport:
    """Tests for corner report generation."""

    def test_basic_report(self) -> None:
        """Test basic report generation."""
        definition = CornerDefinition()
        definition.add_parameter("x", [0, 1])

        results = [
            CornerResult(Corner("c1", {"x": 0}), value=10.0),
            CornerResult(Corner("c2", {"x": 1}), value=20.0),
        ]
        analysis_result = CornerAnalysisResult(results, definition)

        report = print_corner_report(analysis_result)

        assert "CORNER ANALYSIS REPORT" in report
        assert "Corners evaluated: 2" in report
        assert "Worst case" in report
        assert "Best case" in report
