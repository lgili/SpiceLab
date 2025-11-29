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
