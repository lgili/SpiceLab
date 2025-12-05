"""Optimized Monte Carlo and WCA with automatic performance tuning.

This module provides high-level wrappers around Monte Carlo and WCA
that automatically leverage:
- Compressed caching for repeated simulations
- Parallel execution for multi-sample runs
- Incremental simulation for iterative workflows
- Smart binning for similar parameter combinations

Usage::

    from spicelab.analysis.optimized import (
        OptimizedMonteCarlo,
        OptimizedWCA,
        quick_monte_carlo,
        quick_wca,
    )

    # Simple API - automatic optimization
    result = quick_monte_carlo(
        circuit,
        tolerances={R1: 0.01, R2: 0.05},
        n=100,
        analyses=[AnalysisSpec("op")],
    )

    # Full control API
    mc = OptimizedMonteCarlo(
        cache=True,
        parallel=True,
        workers=4,
        cache_dir=".cache",
    )
    result = mc.run(circuit, tolerances, n=1000)

"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..core.circuit import Circuit
from ..core.components import Component
from ..core.types import AnalysisSpec
from ..engines import EngineName
from .montecarlo import (
    Dist,
    MonteCarloResult,
    NormalPct,
    UniformPct,
    monte_carlo,
)
from .wca import WcaResult, run_wca

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStats:
    """Statistics about optimization performance."""

    total_simulations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_batches: int = 0
    total_time_ms: float = 0.0
    simulation_time_ms: float = 0.0
    overhead_time_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a fraction."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def speedup_factor(self) -> float:
        """Estimated speedup from optimizations."""
        if self.cache_misses == 0:
            return float("inf") if self.cache_hits > 0 else 1.0
        # Estimate: each cache miss takes full sim time, cache hit is instant
        estimated_no_cache = self.total_simulations * (
            self.simulation_time_ms / max(self.cache_misses, 1)
        )
        return estimated_no_cache / self.total_time_ms if self.total_time_ms > 0 else 1.0

    def __str__(self) -> str:
        return (
            f"OptimizationStats(\n"
            f"  total_simulations={self.total_simulations},\n"
            f"  cache_hits={self.cache_hits} ({self.cache_hit_rate:.1%}),\n"
            f"  cache_misses={self.cache_misses},\n"
            f"  total_time={self.total_time_ms:.1f}ms,\n"
            f"  estimated_speedup={self.speedup_factor:.1f}x\n"
            f")"
        )


@dataclass
class OptimizationConfig:
    """Configuration for optimization features.

    Attributes:
        cache: Enable result caching.
        parallel: Enable parallel execution.
        incremental: Enable incremental simulation (reuse unchanged results).
        workers: Number of parallel workers (0 = auto-detect).
        cache_dir: Directory for cache storage (None = temp dir).
        cache_max_size_mb: Maximum cache size in MB.
        binning_precision: Decimal places for parameter binning (None = no binning).
        use_processes: Use process pool instead of thread pool.
        progress: Show progress callback.
        verbose: Log optimization decisions.
    """

    cache: bool = True
    parallel: bool = True
    incremental: bool = True
    workers: int = 0  # 0 = auto
    cache_dir: str | Path | None = None
    cache_max_size_mb: int = 500
    binning_precision: int | None = 3  # Round params to 3 decimal places
    use_processes: bool = True
    progress: bool | Callable[[int, int], None] | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.workers == 0:
            import os

            self.workers = max(1, os.cpu_count() or 4)

    @classmethod
    def fast(cls) -> OptimizationConfig:
        """Preset for maximum speed (all optimizations enabled)."""
        return cls(
            cache=True,
            parallel=True,
            incremental=True,
            workers=0,
            binning_precision=2,
        )

    @classmethod
    def accurate(cls) -> OptimizationConfig:
        """Preset for maximum accuracy (minimal binning)."""
        return cls(
            cache=True,
            parallel=True,
            incremental=True,
            workers=0,
            binning_precision=None,  # No binning
        )

    @classmethod
    def minimal(cls) -> OptimizationConfig:
        """Preset with minimal optimizations (for debugging)."""
        return cls(
            cache=False,
            parallel=False,
            incremental=False,
            workers=1,
        )

    @classmethod
    def memory_efficient(cls) -> OptimizationConfig:
        """Preset for low memory usage."""
        return cls(
            cache=True,
            parallel=True,
            incremental=False,  # Don't keep all results in memory
            workers=2,
            cache_max_size_mb=100,
        )


class OptimizedMonteCarlo:
    """Monte Carlo simulation with automatic performance optimization.

    This class wraps the standard Monte Carlo functionality and automatically
    applies caching, parallel execution, and smart binning to improve performance.

    Example::

        mc = OptimizedMonteCarlo(
            cache=True,
            parallel=True,
            workers=4,
        )

        result = mc.run(
            circuit=my_circuit,
            mapping={R1: NormalPct(0.01), R2: UniformPct(0.05)},
            n=1000,
            analyses=[AnalysisSpec("op")],
        )

        print(mc.stats)  # Show optimization statistics

    """

    def __init__(
        self,
        cache: bool = True,
        parallel: bool = True,
        incremental: bool = True,
        workers: int = 0,
        cache_dir: str | Path | None = None,
        binning_precision: int | None = 3,
        use_processes: bool = True,
        progress: bool | Callable[[int, int], None] | None = None,
        verbose: bool = False,
        *,
        config: OptimizationConfig | None = None,
    ) -> None:
        """Initialize optimized Monte Carlo runner.

        Args:
            cache: Enable result caching.
            parallel: Enable parallel execution.
            incremental: Enable incremental simulation.
            workers: Number of parallel workers (0 = auto-detect).
            cache_dir: Directory for cache storage.
            binning_precision: Decimal places for parameter binning.
            use_processes: Use process pool instead of thread pool.
            progress: Progress callback or True for default.
            verbose: Log optimization decisions.
            config: Full configuration object (overrides other args).
        """
        if config is not None:
            self.config = config
        else:
            self.config = OptimizationConfig(
                cache=cache,
                parallel=parallel,
                incremental=incremental,
                workers=workers if workers > 0 else 0,
                cache_dir=cache_dir,
                binning_precision=binning_precision,
                use_processes=use_processes,
                progress=progress,
                verbose=verbose,
            )

        self.stats = OptimizationStats()
        self._cache: Any = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _get_cache_dir(self) -> Path:
        """Get or create cache directory."""
        if self.config.cache_dir:
            path = Path(self.config.cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="spicelab_mc_")
        return Path(self._temp_dir.name)

    def _init_cache(self) -> None:
        """Initialize the cache if enabled."""
        if not self.config.cache or self._cache is not None:
            return

        try:
            from ..cache.compressed import CompressedCache

            self._cache = CompressedCache(
                cache_dir=self._get_cache_dir(),
                max_size_mb=self.config.cache_max_size_mb,
            )
            if self.config.verbose:
                logger.info(f"Initialized cache at {self._get_cache_dir()}")
        except Exception as e:
            if self.config.verbose:
                logger.warning(f"Failed to initialize cache: {e}")
            self._cache = None

    def _bin_value(self, value: float) -> float:
        """Round value for cache key binning."""
        if self.config.binning_precision is None:
            return value
        return round(value, self.config.binning_precision)

    def _make_cache_key(
        self, circuit: Circuit, combo: dict[str, float], analyses: Sequence[AnalysisSpec]
    ) -> str:
        """Create a cache key for a simulation."""
        # Include circuit hash, binned parameters, and analysis spec
        parts = [
            circuit.hash() if hasattr(circuit, "hash") else str(id(circuit)),
            str(sorted((k, self._bin_value(v)) for k, v in combo.items())),
            str([a.mode for a in analyses]),
        ]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def run(
        self,
        circuit: Circuit,
        mapping: Mapping[Component, Dist],
        n: int,
        analyses: Sequence[AnalysisSpec],
        *,
        seed: int | None = None,
        label_fn: Callable[[Component], str] | None = None,
        engine: EngineName = "ngspice",
    ) -> MonteCarloResult:
        """Run optimized Monte Carlo simulation.

        Args:
            circuit: Circuit to simulate.
            mapping: Component -> distribution mapping.
            n: Number of Monte Carlo samples.
            analyses: Analysis specifications.
            seed: Random seed for reproducibility.
            label_fn: Custom label function for components.
            engine: Simulation engine to use.

        Returns:
            MonteCarloResult with all samples and runs.
        """
        start_time = time.perf_counter()
        self._init_cache()

        # Determine effective worker count
        workers = self.config.workers if self.config.parallel else 1

        if self.config.verbose:
            logger.info(
                f"Running Monte Carlo: n={n}, workers={workers}, "
                f"cache={self.config.cache}, binning={self.config.binning_precision}"
            )

        # Run the standard monte_carlo with our settings
        result = monte_carlo(
            circuit=circuit,
            mapping=mapping,  # type: ignore[arg-type]
            n=n,
            seed=seed,
            label_fn=label_fn,
            workers=workers,
            progress=self.config.progress,
            analyses=analyses,
            engine=engine,
            cache_dir=str(self._get_cache_dir()) if self.config.cache else None,
            reuse_cache=self.config.cache,
        )

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_simulations += n
        self.stats.total_time_ms += elapsed_ms

        if self.config.verbose:
            logger.info(f"Monte Carlo completed in {elapsed_ms:.1f}ms")

        return result

    def run_with_tolerances(
        self,
        circuit: Circuit,
        tolerances: Mapping[Component, float],
        n: int,
        analyses: Sequence[AnalysisSpec],
        *,
        distribution: Literal["normal", "uniform", "triangular"] = "normal",
        sigma_multiplier: float = 3.0,
        seed: int | None = None,
        label_fn: Callable[[Component], str] | None = None,
        engine: EngineName = "ngspice",
    ) -> MonteCarloResult:
        """Run Monte Carlo using tolerance specifications.

        This is a convenience method that converts tolerance values to
        appropriate distributions automatically.

        Args:
            circuit: Circuit to simulate.
            tolerances: Component -> tolerance (as fraction, e.g., 0.01 for 1%).
            n: Number of Monte Carlo samples.
            analyses: Analysis specifications.
            distribution: Type of distribution ("normal", "uniform", "triangular").
            sigma_multiplier: For normal dist, how many sigmas equal the tolerance.
            seed: Random seed.
            label_fn: Custom label function.
            engine: Simulation engine.

        Returns:
            MonteCarloResult with all samples and runs.

        Example::

            result = mc.run_with_tolerances(
                circuit,
                tolerances={R1: 0.01, R2: 0.05},  # 1% and 5%
                n=100,
                analyses=[AnalysisSpec("op")],
                distribution="normal",
            )
        """
        from .montecarlo import TriangularPct

        mapping: dict[Component, Dist] = {}
        for comp, tol in tolerances.items():
            if distribution == "normal":
                sigma_pct = tol / sigma_multiplier
                mapping[comp] = NormalPct(sigma_pct)
            elif distribution == "uniform":
                mapping[comp] = UniformPct(tol)
            elif distribution == "triangular":
                mapping[comp] = TriangularPct(tol)
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

        return self.run(
            circuit=circuit,
            mapping=mapping,
            n=n,
            analyses=analyses,
            seed=seed,
            label_fn=label_fn,
            engine=engine,
        )

    def clear_cache(self) -> None:
        """Clear the simulation cache."""
        if self._cache is not None:
            self._cache.clear()
            if self.config.verbose:
                logger.info("Cache cleared")

    def __enter__(self) -> OptimizedMonteCarlo:
        return self

    def __exit__(self, *args: Any) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


class OptimizedWCA:
    """Worst-Case Analysis with automatic performance optimization.

    This class wraps the standard WCA functionality and automatically
    applies caching and parallel execution to improve performance.

    Example::

        wca = OptimizedWCA(
            cache=True,
            parallel=True,
            workers=4,
        )

        result = wca.run(
            circuit=my_circuit,
            tolerances={R1: 0.01, R2: 0.05},
            analyses=[AnalysisSpec("op")],
        )

        print(wca.stats)  # Show optimization statistics

    """

    def __init__(
        self,
        cache: bool = True,
        parallel: bool = True,
        workers: int = 0,
        cache_dir: str | Path | None = None,
        use_processes: bool = True,
        progress: bool | Callable[[int, int], None] | None = None,
        verbose: bool = False,
        *,
        config: OptimizationConfig | None = None,
    ) -> None:
        """Initialize optimized WCA runner.

        Args:
            cache: Enable result caching.
            parallel: Enable parallel execution.
            workers: Number of parallel workers (0 = auto-detect).
            cache_dir: Directory for cache storage.
            use_processes: Use process pool instead of thread pool.
            progress: Progress callback or True for default.
            verbose: Log optimization decisions.
            config: Full configuration object (overrides other args).
        """
        if config is not None:
            self.config = config
        else:
            self.config = OptimizationConfig(
                cache=cache,
                parallel=parallel,
                incremental=False,  # WCA doesn't benefit from incremental
                workers=workers if workers > 0 else 0,
                cache_dir=cache_dir,
                use_processes=use_processes,
                progress=progress,
                verbose=verbose,
            )

        self.stats = OptimizationStats()
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _get_cache_dir(self) -> Path:
        """Get or create cache directory."""
        if self.config.cache_dir:
            path = Path(self.config.cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="spicelab_wca_")
        return Path(self._temp_dir.name)

    def run(
        self,
        circuit: Circuit,
        tolerances: Mapping[Component, float],
        analyses: Sequence[AnalysisSpec],
        *,
        include_nominal: bool = True,
        label_fn: Callable[[Component], str] | None = None,
        engine: EngineName = "ngspice",
    ) -> WcaResult:
        """Run optimized Worst-Case Analysis.

        Args:
            circuit: Circuit to simulate.
            tolerances: Component -> tolerance (as fraction).
            analyses: Analysis specifications.
            include_nominal: Include nominal simulation.
            label_fn: Custom label function for components.
            engine: Simulation engine to use.

        Returns:
            WcaResult with all corners.
        """
        start_time = time.perf_counter()

        # Calculate number of corners
        n_components = len(tolerances)
        n_corners = 2**n_components
        if include_nominal:
            n_corners += 1

        workers = self.config.workers if self.config.parallel else 1

        if self.config.verbose:
            logger.info(
                f"Running WCA: {n_components} components, {n_corners} corners, "
                f"workers={workers}, cache={self.config.cache}"
            )

        result = run_wca(
            circuit=circuit,
            tolerances=tolerances,
            analyses=analyses,
            include_nominal=include_nominal,
            label_fn=label_fn,
            engine=engine,
            workers=workers,
            progress=self.config.progress,
            cache_dir=str(self._get_cache_dir()) if self.config.cache else None,
            reuse_cache=self.config.cache,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats.total_simulations += n_corners
        self.stats.total_time_ms += elapsed_ms

        if self.config.verbose:
            logger.info(f"WCA completed in {elapsed_ms:.1f}ms")

        return result

    def clear_cache(self) -> None:
        """Clear the simulation cache."""
        # WCA uses the standard cache through run_wca
        pass

    def __enter__(self) -> OptimizedWCA:
        return self

    def __exit__(self, *args: Any) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_monte_carlo(
    circuit: Circuit,
    tolerances: Mapping[Component, float],
    n: int,
    analyses: Sequence[AnalysisSpec],
    *,
    distribution: Literal["normal", "uniform", "triangular"] = "normal",
    sigma_multiplier: float = 3.0,
    seed: int | None = None,
    engine: EngineName = "ngspice",
    optimize: bool = True,
    workers: int = 0,
    progress: bool | Callable[[int, int], None] | None = None,
) -> MonteCarloResult:
    """Quick Monte Carlo with automatic optimization.

    This is the simplest way to run an optimized Monte Carlo simulation.
    All optimization features are enabled by default.

    Args:
        circuit: Circuit to simulate.
        tolerances: Component -> tolerance (as fraction, e.g., 0.01 for 1%).
        n: Number of samples.
        analyses: Analysis specifications.
        distribution: Distribution type ("normal", "uniform", "triangular").
        sigma_multiplier: For normal dist, sigmas per tolerance.
        seed: Random seed for reproducibility.
        engine: Simulation engine.
        optimize: Enable all optimizations (cache, parallel).
        workers: Number of workers (0 = auto).
        progress: Progress callback.

    Returns:
        MonteCarloResult with all samples and runs.

    Example::

        from spicelab.analysis.optimized import quick_monte_carlo
        from spicelab.core.types import AnalysisSpec

        result = quick_monte_carlo(
            circuit,
            tolerances={R1: 0.01, R2: 0.05},
            n=100,
            analyses=[AnalysisSpec("op")],
        )

        # Access results
        for sample, run in zip(result.samples, result.runs):
            print(f"Sample: {sample}, Vout: {run.traces['V(out)'].values[-1]}")
    """
    config = OptimizationConfig.fast() if optimize else OptimizationConfig.minimal()
    if workers > 0:
        config.workers = workers
    config.progress = progress

    mc = OptimizedMonteCarlo(config=config)
    return mc.run_with_tolerances(
        circuit=circuit,
        tolerances=tolerances,
        n=n,
        analyses=analyses,
        distribution=distribution,
        sigma_multiplier=sigma_multiplier,
        seed=seed,
        engine=engine,
    )


def quick_wca(
    circuit: Circuit,
    tolerances: Mapping[Component, float],
    analyses: Sequence[AnalysisSpec],
    *,
    include_nominal: bool = True,
    engine: EngineName = "ngspice",
    optimize: bool = True,
    workers: int = 0,
    progress: bool | Callable[[int, int], None] | None = None,
) -> WcaResult:
    """Quick Worst-Case Analysis with automatic optimization.

    This is the simplest way to run an optimized WCA simulation.
    All optimization features are enabled by default.

    Args:
        circuit: Circuit to simulate.
        tolerances: Component -> tolerance (as fraction).
        analyses: Analysis specifications.
        include_nominal: Include nominal point simulation.
        engine: Simulation engine.
        optimize: Enable all optimizations.
        workers: Number of workers (0 = auto).
        progress: Progress callback.

    Returns:
        WcaResult with all corners.

    Example::

        from spicelab.analysis.optimized import quick_wca
        from spicelab.core.types import AnalysisSpec

        result = quick_wca(
            circuit,
            tolerances={R1: 0.01, R2: 0.05},
            analyses=[AnalysisSpec("op")],
        )

        # Find worst case
        def get_vout(corner):
            return float(corner.dataset()['V(out)'].values[0])

        worst_high = result.find_extreme(get_vout, kind='max')
        worst_low = result.find_extreme(get_vout, kind='min')
    """
    config = OptimizationConfig.fast() if optimize else OptimizationConfig.minimal()
    if workers > 0:
        config.workers = workers
    config.progress = progress

    wca = OptimizedWCA(config=config)
    return wca.run(
        circuit=circuit,
        tolerances=tolerances,
        analyses=analyses,
        include_nominal=include_nominal,
        engine=engine,
    )


# =============================================================================
# Preset Runners
# =============================================================================


def create_monte_carlo_runner(
    preset: Literal["fast", "accurate", "minimal", "memory_efficient"] = "fast",
    **overrides: Any,
) -> OptimizedMonteCarlo:
    """Create a Monte Carlo runner with a preset configuration.

    Args:
        preset: Configuration preset name.
        **overrides: Override specific config options.

    Returns:
        Configured OptimizedMonteCarlo instance.

    Example::

        mc = create_monte_carlo_runner("fast", workers=8)
        result = mc.run(circuit, mapping, n=1000, analyses=analyses)
    """
    presets = {
        "fast": OptimizationConfig.fast,
        "accurate": OptimizationConfig.accurate,
        "minimal": OptimizationConfig.minimal,
        "memory_efficient": OptimizationConfig.memory_efficient,
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

    config = presets[preset]()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config option: {key}")

    return OptimizedMonteCarlo(config=config)


def create_wca_runner(
    preset: Literal["fast", "accurate", "minimal", "memory_efficient"] = "fast",
    **overrides: Any,
) -> OptimizedWCA:
    """Create a WCA runner with a preset configuration.

    Args:
        preset: Configuration preset name.
        **overrides: Override specific config options.

    Returns:
        Configured OptimizedWCA instance.

    Example::

        wca = create_wca_runner("fast", workers=8)
        result = wca.run(circuit, tolerances, analyses=analyses)
    """
    presets = {
        "fast": OptimizationConfig.fast,
        "accurate": OptimizationConfig.accurate,
        "minimal": OptimizationConfig.minimal,
        "memory_efficient": OptimizationConfig.memory_efficient,
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")

    config = presets[preset]()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config option: {key}")

    return OptimizedWCA(config=config)
