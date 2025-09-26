from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Any, cast

import pytest
from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

if TYPE_CHECKING:
    pass

ng = shutil.which("ngspice")


def _simple() -> tuple[Circuit, Resistor]:
    c = Circuit("op_post")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c, R1


def test_stack_pick_columns() -> None:
    if not ng:
        pytest.skip("ngspice not installed")
    pd = cast(Any, pytest.importorskip("pandas"))
    c, R = _simple()
    sweep = run_value_sweep(
        c,
        component=R,
        values=["1k", "2k"],
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
    )

    frames: list[Any] = []
    for run in sweep.runs:
        df = run.handle.dataset().to_dataframe().reset_index()
        df["R"] = run.value
        frames.append(df)

    stacked = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    assert not stacked.empty
    xcol = "time" if "time" in stacked.columns else stacked.columns[0]
    assert xcol in stacked.columns
    assert "R" in stacked.columns
