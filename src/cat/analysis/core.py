from __future__ import annotations

from dataclasses import dataclass

from ..core.circuit import Circuit
from ..io.log_reader import read_errors
from ..io.raw_reader import TraceSet, parse_ngspice_ascii_raw
from ..spice.base import RunResult as BaseRunResult
from ..spice.registry import get_run_directives


@dataclass(frozen=True)
class AnalysisResult:
    run: BaseRunResult
    traces: TraceSet


class _BaseAnalysis:
    def _directives(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def run(self, circuit: Circuit) -> AnalysisResult:
        net = circuit.build_netlist()
        run_directives = get_run_directives()
        res = run_directives(net, self._directives())
        if res.returncode != 0:
            try:
                with open(res.artifacts.log_path, encoding="utf-8", errors="ignore") as f:
                    errs = read_errors(f.read())
                msg = f"NGSpice exited with code {res.returncode}. Errors: {errs}"
            except Exception:
                msg = f"NGSpice exited with code {res.returncode}. stderr: {res.stderr[:200]}"
            raise RuntimeError(msg)
        if not res.artifacts.raw_path:
            raise RuntimeError("NGSpice produced no RAW path")
        traces = parse_ngspice_ascii_raw(res.artifacts.raw_path)
        return AnalysisResult(run=res, traces=traces)


class OP(_BaseAnalysis):
    def _directives(self) -> list[str]:
        return [".op"]


class TRAN(_BaseAnalysis):
    def __init__(self, tstep: str, tstop: str, tstart: str | None = None) -> None:
        self.tstep = tstep
        self.tstop = tstop
        self.tstart = tstart

    def _directives(self) -> list[str]:
        if self.tstart:
            return [f".tran {self.tstep} {self.tstop} {self.tstart}"]
        return [f".tran {self.tstep} {self.tstop}"]


class AC(_BaseAnalysis):
    def __init__(self, sweep_type: str, n: int, fstart: float, fstop: float) -> None:
        self.sweep_type = sweep_type
        self.n = n
        self.fstart = fstart
        self.fstop = fstop

    def _directives(self) -> list[str]:
        return [f".ac {self.sweep_type} {self.n} {self.fstart} {self.fstop}"]


class DC(_BaseAnalysis):
    def __init__(self, src_ref: str, start: float, stop: float, step: float) -> None:
        self.src_ref = src_ref
        self.start = start
        self.stop = stop
        self.step = step

    def _directives(self) -> list[str]:
        return [f".dc V{self.src_ref} {self.start} {self.stop} {self.step}"]
