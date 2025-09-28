from __future__ import annotations

import math
import random
import shutil
import statistics as stats
from collections.abc import Callable

import pytest
from spicelab.analysis import (
    AnalysisResult,
    Dist,
    LogNormalPct,
    MonteCarloResult,
    NormalPct,
    TriangularPct,
    UniformAbs,
    UniformPct,
    monte_carlo,
)
from spicelab.core.circuit import Circuit
from spicelab.core.components import Idc, Resistor
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")
RESISTOR_NOMINAL = 1000.0


def _build_dc_resistor() -> tuple[Circuit, Resistor]:
    circuit = Circuit("mc_dist_res")
    node = Net("n1")
    current = Idc("I1", 1.0)
    resistor = Resistor("R1", RESISTOR_NOMINAL)
    circuit.add(current, resistor)
    circuit.connect(current.ports[0], node)
    circuit.connect(current.ports[1], GND)
    circuit.connect(resistor.ports[0], node)
    circuit.connect(resistor.ports[1], GND)
    return circuit, resistor


def _node_voltage(result: AnalysisResult) -> float:
    trace = result.traces["V(n1)"]
    return float(trace.values[-1])


def _run_distribution(dist: Dist, *, n: int, seed: int) -> MonteCarloResult:
    circuit, resistor = _build_dc_resistor()
    result = monte_carlo(
        circuit,
        mapping={resistor: dist},
        n=n,
        seed=seed,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
    )
    return result


def _series_to_list(series):
    if hasattr(series, "tolist"):
        return series.tolist()
    return list(series)


@pytest.mark.skipif(not ng, reason="ngspice not installed")
@pytest.mark.parametrize(
    "_label, dist_factory, expected_std, bounds",
    [
        (
            "normal_pct",
            lambda: NormalPct(0.05),
            RESISTOR_NOMINAL * 0.05,
            None,
        ),
        (
            "uniform_abs",
            lambda: UniformAbs(50.0),
            50.0 / math.sqrt(3.0),
            (RESISTOR_NOMINAL - 50.0, RESISTOR_NOMINAL + 50.0),
        ),
        (
            "uniform_pct",
            lambda: UniformPct(0.10),
            (RESISTOR_NOMINAL * 0.10) / math.sqrt(3.0),
            (RESISTOR_NOMINAL * 0.90, RESISTOR_NOMINAL * 1.10),
        ),
        (
            "triangular_pct",
            lambda: TriangularPct(0.15),
            (RESISTOR_NOMINAL * 0.15) / math.sqrt(6.0),
            (RESISTOR_NOMINAL * 0.85, RESISTOR_NOMINAL * 1.15),
        ),
        (
            "lognormal_pct",
            lambda: LogNormalPct(0.10),
            RESISTOR_NOMINAL * math.sqrt(math.exp(0.10**2) - 1.0),
            (RESISTOR_NOMINAL * 0.60, RESISTOR_NOMINAL * 1.60),
        ),
    ],
)
def test_monte_carlo_distributions_reflect_sampling(
    _label: str,
    dist_factory: Callable[[], Dist],
    expected_std: float,
    bounds: tuple[float, float] | None,
) -> None:
    seed = 4242
    n = 64
    dist = dist_factory()
    mc = _run_distribution(dist, n=n, seed=seed)

    assert len(mc.samples) == n
    assert len(mc.runs) == n

    sample_key = next(iter(mc.samples[0].keys()))
    samples = [sample[sample_key] for sample in mc.samples]

    rng = random.Random(seed)
    expected_sequence = [dist.sample(RESISTOR_NOMINAL, rng) for _ in range(n)]
    for actual, expected in zip(samples, expected_sequence, strict=False):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-9)

    series_mean = stats.mean(samples)
    series_std = stats.pstdev(samples)
    mean_tol = max(1e-6, expected_std * 0.25)
    std_tol = max(1e-6, expected_std * 0.25)
    assert abs(series_mean - RESISTOR_NOMINAL) <= mean_tol
    assert abs(series_std - expected_std) <= std_tol

    if bounds is not None:
        lo, hi = bounds
        assert min(samples) >= lo - 1e-6
        assert max(samples) <= hi + 1e-6

    voltages = [float(run.traces["V(n1)"].values[-1]) for run in mc.runs]
    for sample, voltage in zip(samples, voltages, strict=False):
        assert math.isclose(voltage, -sample, rel_tol=1e-9, abs_tol=1e-6)

    df = mc.to_dataframe(metric=_node_voltage, param_prefix="param_")
    param_column = f"param_{sample_key}"
    df_samples = _series_to_list(df[param_column])
    df_metric = _series_to_list(df["metric"])
    assert df_samples == pytest.approx(samples)
    for sample, metric_value in zip(samples, df_metric, strict=False):
        assert math.isclose(metric_value, -sample, rel_tol=1e-9, abs_tol=1e-6)
