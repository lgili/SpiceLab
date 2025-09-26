from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation
from spicelab.engines.exceptions import EngineBinaryNotFound


def _simple_circuit() -> Circuit:
    circuit = Circuit("local_op")
    v1 = Vdc("1", 5.0)
    r1 = Resistor("1", "1k")
    circuit.add(v1, r1)
    circuit.connect(v1.ports[0], r1.ports[0])
    circuit.connect(r1.ports[1], GND)
    circuit.connect(v1.ports[1], GND)
    return circuit


def _set_env(var: str, value: str | None) -> tuple[str, str | None]:
    previous = os.environ.get(var)
    if value is not None:
        os.environ[var] = value
    return var, previous


def _restore_env(var: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = previous


@pytest.mark.local
@pytest.mark.engine
@pytest.mark.skipif(
    not os.environ.get("SPICELAB_TEST_LTSPICE"),
    reason="Set SPICELAB_TEST_LTSPICE to the LTspice executable path to run locally",
)
def test_ltspice_local_op_smoke() -> None:
    exe = os.environ["SPICELAB_TEST_LTSPICE"]
    if not Path(exe).exists():
        pytest.skip(f"LTspice executable not found at {exe}")

    var, previous = _set_env("SPICELAB_LTSPICE", exe)
    try:
        try:
            handle = run_simulation(
                _simple_circuit(),
                [AnalysisSpec("op", {})],
                engine="ltspice",
            )
        except EngineBinaryNotFound as exc:
            pytest.skip(f"LTspice not available: {exc}")

        ds = handle.dataset()
        voltage_vars = [name for name in ds.data_vars if name.startswith("V(")]
        assert voltage_vars, f"No voltage variables found: {list(ds.data_vars)}"
        vals = ds[voltage_vars[0]].values
        assert vals.size > 0
        assert math.isfinite(float(vals[0]))
    finally:
        _restore_env(var, previous)


@pytest.mark.local
@pytest.mark.engine
@pytest.mark.skipif(
    not os.environ.get("SPICELAB_TEST_XYCE"),
    reason="Set SPICELAB_TEST_XYCE to the Xyce executable path to run locally",
)
def test_xyce_local_op_smoke() -> None:
    exe = os.environ["SPICELAB_TEST_XYCE"]
    if not Path(exe).exists():
        pytest.skip(f"Xyce executable not found at {exe}")

    var, previous = _set_env("SPICELAB_XYCE", exe)
    try:
        try:
            handle = run_simulation(
                _simple_circuit(),
                [AnalysisSpec("op", {})],
                engine="xyce",
            )
        except EngineBinaryNotFound as exc:
            pytest.skip(f"Xyce not available: {exc}")

        ds = handle.dataset()
        voltage_vars = [name for name in ds.data_vars if name.startswith("V(")]
        assert voltage_vars, f"No voltage variables found: {list(ds.data_vars)}"
        vals = ds[voltage_vars[0]].values
        assert vals.size > 0
        assert math.isfinite(float(vals[0]))
    finally:
        _restore_env(var, previous)
