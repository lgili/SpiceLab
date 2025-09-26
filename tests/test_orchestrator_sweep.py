from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec, SweepSpec
from spicelab.engines import run_simulation
from spicelab.orchestrator import Job, run_job
from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.registry import get_run_directives, set_run_directives

ASCII_TEMPLATE = """Title:  transient
Date:   Thu Sep  1 12:00:00 2025
Plotname: Transient
Flags: real
No. Variables: 2
No. Points: 1
Variables:
        0       time    time
        1       v(vout) voltage
Values:
        0       0.0     {value}
"""


def _runner_with_values(values: list[float]) -> Callable[[str, Sequence[str]], RunResult]:
    vals = list(values)

    def _runner(netlist: str, directives: Sequence[str]) -> RunResult:
        if not vals:
            raise RuntimeError("runner called more times than expected")
        val = vals.pop(0)
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w", encoding="utf-8") as f:
            f.write(ASCII_TEMPLATE.format(value=val))
        log = os.path.join(td, "ngspice.log")
        with open(log, "w", encoding="utf-8") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    return _runner


def _rc_circuit() -> Circuit:
    c = Circuit("rc")
    v1 = Vdc("1", 1.0)
    r1 = Resistor("1", "1k")
    c.add(v1, r1)
    c.connect(v1.ports[0], r1.ports[0])
    c.connect(r1.ports[1], GND)
    c.connect(v1.ports[1], GND)
    return c


def test_run_simulation_sweep_returns_job_result(tmp_path: Path) -> None:
    old = get_run_directives()
    try:
        set_run_directives(_runner_with_values([1.0, 2.0]))
        circuit = _rc_circuit()
        sweep = SweepSpec(variables={"1": ["1k", "2k"]})
        result = run_simulation(
            circuit,
            [AnalysisSpec("tran", {"tstep": "1u", "tstop": "1u"})],
            sweep=sweep,
            engine="ngspice",
            cache_dir=tmp_path,
        )
        from spicelab.orchestrator import JobResult

        assert isinstance(result, JobResult)
        assert len(result.runs) == 2
        values = []
        for run in result.runs:
            ds = run.handle.dataset()
            values.append(float(ds["V(vout)"].values[-1]))
        assert values == [1.0, 2.0]

        # second run should hit cache even if the runner raises
        set_run_directives(_runner_with_values([]))
        circuit2 = _rc_circuit()
        sweep2 = SweepSpec(variables={"1": ["1k", "2k"]})
        job = Job(
            circuit=circuit2,
            analyses=[AnalysisSpec("tran", {"tstep": "1u", "tstop": "1u"})],
            sweep=sweep2,
            engine="ngspice",
        )
        cached = run_job(job, cache_dir=tmp_path)
        assert [run.from_cache for run in cached.runs] == [True, True]
    finally:
        set_run_directives(old)
