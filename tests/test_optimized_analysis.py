"""Tests for optimized Monte Carlo and WCA analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from spicelab.analysis.montecarlo import UniformPct
from spicelab.analysis.optimized import (
    OptimizationConfig,
    OptimizationStats,
    OptimizedMonteCarlo,
    OptimizedWCA,
    create_monte_carlo_runner,
    create_wca_runner,
    quick_monte_carlo,
    quick_wca,
)
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_circuit() -> Circuit:
    """Create a simple RC circuit for testing."""
    circuit = Circuit("test_rc")
    vin = Vdc("Vin", 1.0)
    r = Resistor("R1", "1k")
    c = Capacitor("C1", "1u")

    circuit.add(vin, r, c)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r.ports[0], Net("in"))
    circuit.connect(r.ports[1], Net("out"))
    circuit.connect(c.ports[0], Net("out"))
    circuit.connect(c.ports[1], GND)

    return circuit


@pytest.fixture
def resistors(simple_circuit: Circuit) -> tuple[Resistor, ...]:
    """Extract resistors from circuit."""
    return tuple(c for c in simple_circuit._components if isinstance(c, Resistor))


# =============================================================================
# OptimizationConfig Tests
# =============================================================================


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = OptimizationConfig()
        assert config.cache is True
        assert config.parallel is True
        assert config.incremental is True
        assert config.workers > 0  # Auto-detected

    def test_fast_preset(self) -> None:
        """Fast preset enables all optimizations."""
        config = OptimizationConfig.fast()
        assert config.cache is True
        assert config.parallel is True
        assert config.incremental is True
        assert config.binning_precision == 2

    def test_accurate_preset(self) -> None:
        """Accurate preset disables binning."""
        config = OptimizationConfig.accurate()
        assert config.cache is True
        assert config.parallel is True
        assert config.binning_precision is None

    def test_minimal_preset(self) -> None:
        """Minimal preset disables all optimizations."""
        config = OptimizationConfig.minimal()
        assert config.cache is False
        assert config.parallel is False
        assert config.incremental is False
        assert config.workers == 1

    def test_memory_efficient_preset(self) -> None:
        """Memory efficient preset limits cache size."""
        config = OptimizationConfig.memory_efficient()
        assert config.cache is True
        assert config.incremental is False
        assert config.cache_max_size_mb == 100
        assert config.workers == 2


class TestOptimizationStats:
    """Tests for OptimizationStats."""

    def test_empty_stats(self) -> None:
        """Empty stats have default values."""
        stats = OptimizationStats()
        assert stats.total_simulations == 0
        assert stats.cache_hits == 0
        assert stats.cache_hit_rate == 0.0

    def test_cache_hit_rate(self) -> None:
        """Cache hit rate is calculated correctly."""
        stats = OptimizationStats(cache_hits=80, cache_misses=20)
        assert stats.cache_hit_rate == 0.8

    def test_speedup_factor_no_cache(self) -> None:
        """Speedup is 1x when no cache hits."""
        stats = OptimizationStats(
            total_simulations=10,
            cache_hits=0,
            cache_misses=10,
            total_time_ms=1000,
            simulation_time_ms=1000,
        )
        assert stats.speedup_factor == pytest.approx(1.0)

    def test_stats_str(self) -> None:
        """Stats have readable string representation."""
        stats = OptimizationStats(total_simulations=100, cache_hits=80)
        s = str(stats)
        assert "total_simulations=100" in s
        assert "cache_hits=80" in s


# =============================================================================
# OptimizedMonteCarlo Tests
# =============================================================================


class TestOptimizedMonteCarlo:
    """Tests for OptimizedMonteCarlo."""

    def test_init_with_defaults(self) -> None:
        """Can initialize with default settings."""
        mc = OptimizedMonteCarlo()
        assert mc.config.cache is True
        assert mc.config.parallel is True

    def test_init_with_custom_config(self) -> None:
        """Can initialize with custom config."""
        config = OptimizationConfig.minimal()
        mc = OptimizedMonteCarlo(config=config)
        assert mc.config.cache is False
        assert mc.config.parallel is False

    def test_init_with_kwargs(self) -> None:
        """Can initialize with keyword arguments."""
        mc = OptimizedMonteCarlo(
            cache=False,
            parallel=True,
            workers=2,
        )
        assert mc.config.cache is False
        assert mc.config.parallel is True
        assert mc.config.workers == 2

    def test_context_manager(self) -> None:
        """Works as context manager."""
        with OptimizedMonteCarlo() as mc:
            assert mc is not None

    def test_bin_value(self) -> None:
        """Value binning works correctly."""
        mc = OptimizedMonteCarlo(binning_precision=2)
        assert mc._bin_value(1.2345) == 1.23
        assert mc._bin_value(1.2367) == 1.24

    def test_bin_value_no_binning(self) -> None:
        """No binning when precision is None."""
        mc = OptimizedMonteCarlo(binning_precision=None)
        assert mc._bin_value(1.2345) == 1.2345

    def test_cache_dir_creation(self) -> None:
        """Cache directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache"
            mc = OptimizedMonteCarlo(cache_dir=cache_path)
            result = mc._get_cache_dir()
            assert result == cache_path
            assert result.exists()

    def test_temp_cache_dir(self) -> None:
        """Temp directory is used when no cache_dir specified."""
        mc = OptimizedMonteCarlo(cache_dir=None)
        path = mc._get_cache_dir()
        assert path.exists()
        assert "spicelab_mc_" in str(path)


class TestOptimizedMonteCarloIntegration:
    """Integration tests for OptimizedMonteCarlo (require ngspice)."""

    @pytest.fixture(autouse=True)
    def check_ngspice(self) -> None:
        """Skip if ngspice not available."""
        try:
            from spicelab.spice.registry import get_active_adapter

            get_active_adapter()
        except Exception:
            pytest.skip("ngspice not available")

    def test_run_with_tolerances(
        self, simple_circuit: Circuit, resistors: tuple[Resistor, ...]
    ) -> None:
        """Can run Monte Carlo with tolerance spec."""
        mc = OptimizedMonteCarlo(config=OptimizationConfig.minimal())

        tolerances = {resistors[0]: 0.01}

        result = mc.run_with_tolerances(
            circuit=simple_circuit,
            tolerances=tolerances,
            n=3,
            analyses=[AnalysisSpec("op")],
            distribution="uniform",
        )

        assert len(result.samples) == 3
        assert len(result.runs) == 3

    def test_run_with_mapping(
        self, simple_circuit: Circuit, resistors: tuple[Resistor, ...]
    ) -> None:
        """Can run Monte Carlo with distribution mapping."""
        mc = OptimizedMonteCarlo(config=OptimizationConfig.minimal())

        mapping = {resistors[0]: UniformPct(0.01)}

        result = mc.run(
            circuit=simple_circuit,
            mapping=mapping,
            n=3,
            analyses=[AnalysisSpec("op")],
        )

        assert len(result.samples) == 3
        assert len(result.runs) == 3


# =============================================================================
# OptimizedWCA Tests
# =============================================================================


class TestOptimizedWCA:
    """Tests for OptimizedWCA."""

    def test_init_with_defaults(self) -> None:
        """Can initialize with default settings."""
        wca = OptimizedWCA()
        assert wca.config.cache is True
        assert wca.config.parallel is True

    def test_init_with_custom_config(self) -> None:
        """Can initialize with custom config."""
        config = OptimizationConfig.minimal()
        wca = OptimizedWCA(config=config)
        assert wca.config.cache is False
        assert wca.config.parallel is False

    def test_context_manager(self) -> None:
        """Works as context manager."""
        with OptimizedWCA() as wca:
            assert wca is not None


class TestOptimizedWCAIntegration:
    """Integration tests for OptimizedWCA (require ngspice)."""

    @pytest.fixture(autouse=True)
    def check_ngspice(self) -> None:
        """Skip if ngspice not available."""
        try:
            from spicelab.spice.registry import get_active_adapter

            get_active_adapter()
        except Exception:
            pytest.skip("ngspice not available")

    def test_run_wca(self, simple_circuit: Circuit, resistors: tuple[Resistor, ...]) -> None:
        """Can run WCA analysis."""
        wca = OptimizedWCA(config=OptimizationConfig.minimal())

        tolerances = {resistors[0]: 0.01}

        result = wca.run(
            circuit=simple_circuit,
            tolerances=tolerances,
            analyses=[AnalysisSpec("op")],
            include_nominal=True,
        )

        # 1 component = 2 corners
        assert len(result.corners) == 2
        assert result.nominal_run is not None


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for quick_monte_carlo and quick_wca."""

    @pytest.fixture(autouse=True)
    def check_ngspice(self) -> None:
        """Skip if ngspice not available."""
        try:
            from spicelab.spice.registry import get_active_adapter

            get_active_adapter()
        except Exception:
            pytest.skip("ngspice not available")

    def test_quick_monte_carlo(
        self, simple_circuit: Circuit, resistors: tuple[Resistor, ...]
    ) -> None:
        """quick_monte_carlo works with minimal args."""
        result = quick_monte_carlo(
            circuit=simple_circuit,
            tolerances={resistors[0]: 0.01},
            n=3,
            analyses=[AnalysisSpec("op")],
            optimize=False,  # Faster for test
        )

        assert len(result.samples) == 3
        assert len(result.runs) == 3

    def test_quick_monte_carlo_distributions(
        self, simple_circuit: Circuit, resistors: tuple[Resistor, ...]
    ) -> None:
        """quick_monte_carlo supports different distributions."""
        for dist in ["normal", "uniform", "triangular"]:
            result = quick_monte_carlo(
                circuit=simple_circuit,
                tolerances={resistors[0]: 0.01},
                n=2,
                analyses=[AnalysisSpec("op")],
                distribution=dist,  # type: ignore
                optimize=False,
            )
            assert len(result.runs) == 2

    def test_quick_wca(self, simple_circuit: Circuit, resistors: tuple[Resistor, ...]) -> None:
        """quick_wca works with minimal args."""
        result = quick_wca(
            circuit=simple_circuit,
            tolerances={resistors[0]: 0.01},
            analyses=[AnalysisSpec("op")],
            optimize=False,
        )

        assert len(result.corners) == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for create_monte_carlo_runner and create_wca_runner."""

    def test_create_mc_runner_presets(self) -> None:
        """Can create MC runner with all presets."""
        for preset in ["fast", "accurate", "minimal", "memory_efficient"]:
            mc = create_monte_carlo_runner(preset)  # type: ignore
            assert mc is not None

    def test_create_mc_runner_with_overrides(self) -> None:
        """Can override preset values."""
        mc = create_monte_carlo_runner("fast", workers=8, verbose=True)
        assert mc.config.workers == 8
        assert mc.config.verbose is True

    def test_create_mc_runner_invalid_preset(self) -> None:
        """Invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            create_monte_carlo_runner("invalid")  # type: ignore

    def test_create_mc_runner_invalid_override(self) -> None:
        """Invalid override key raises error."""
        with pytest.raises(ValueError, match="Unknown config option"):
            create_monte_carlo_runner("fast", invalid_option=True)

    def test_create_wca_runner_presets(self) -> None:
        """Can create WCA runner with all presets."""
        for preset in ["fast", "accurate", "minimal", "memory_efficient"]:
            wca = create_wca_runner(preset)  # type: ignore
            assert wca is not None

    def test_create_wca_runner_with_overrides(self) -> None:
        """Can override preset values."""
        wca = create_wca_runner("fast", workers=4)
        assert wca.config.workers == 4


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Test that all exports are importable."""

    def test_import_from_analysis(self) -> None:
        """Can import from spicelab.analysis."""
        from spicelab.analysis import (
            OptimizationConfig,
            OptimizationStats,
            OptimizedMonteCarlo,
            OptimizedWCA,
            create_monte_carlo_runner,
            create_wca_runner,
            quick_monte_carlo,
            quick_wca,
        )

        assert OptimizationConfig is not None
        assert OptimizationStats is not None
        assert OptimizedMonteCarlo is not None
        assert OptimizedWCA is not None
        assert quick_monte_carlo is not None
        assert quick_wca is not None
        assert create_monte_carlo_runner is not None
        assert create_wca_runner is not None

    def test_import_from_optimized(self) -> None:
        """Can import directly from optimized module."""
        from spicelab.analysis.optimized import (
            OptimizationConfig,
            OptimizedMonteCarlo,
            quick_monte_carlo,
        )

        assert OptimizationConfig is not None
        assert OptimizedMonteCarlo is not None
        assert quick_monte_carlo is not None
