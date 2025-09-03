from __future__ import annotations

from dataclasses import dataclass

from ..core.circuit import Circuit
from ..io.raw_reader import TraceSet, parse_ngspice_ascii_raw
from ..spice import ngspice_cli
from ..spice.base import RunResult


@dataclass(frozen=True)
class AnalysisResult:
    """Resultado de uma análise: artefatos do run e os traços parseados."""

    run: RunResult
    traces: TraceSet


class _BaseAnalysis:
    def _directives(self) -> list[str]:  # pragma: no cover (interface)
        raise NotImplementedError

    def run(self, circuit: Circuit) -> AnalysisResult:
        """Gera o netlist, roda o NGSpice e retorna TraceSet."""
        net = circuit.build_netlist()
        res = ngspice_cli.run_directives(net, self._directives())
        if res.returncode != 0:
            raise RuntimeError(f"NGSpice exited with code {res.returncode}")
        if not res.artifacts.raw_path:
            raise RuntimeError("NGSpice produced no RAW path")
        traces = parse_ngspice_ascii_raw(res.artifacts.raw_path)
        return AnalysisResult(run=res, traces=traces)


class OP(_BaseAnalysis):
    """Ponto de operação (.op)"""

    def _directives(self) -> list[str]:
        return [
            ".op",
            ".option filetype=ascii",
            ".save all",
        ]


class TRAN(_BaseAnalysis):
    """Transiente (.tran tstep tstop)"""

    def __init__(self, tstep: str, tstop: str) -> None:
        self.tstep = tstep
        self.tstop = tstop

    def _directives(self) -> list[str]:
        return [
            f".tran {self.tstep} {self.tstop}",
            ".option filetype=ascii",
            ".save all",
        ]


class AC(_BaseAnalysis):
    """Análise em frequência (.ac {lin|dec|oct} N f_start f_stop)"""

    def __init__(self, sweep: str, points: int, f_start: float, f_stop: float) -> None:
        if sweep not in {"lin", "dec", "oct"}:
            raise ValueError("sweep must be one of {'lin','dec','oct'}")
        self.sweep = sweep
        self.points = points
        self.f_start = f_start
        self.f_stop = f_stop

    def _directives(self) -> list[str]:
        return [
            f".ac {self.sweep} {self.points} {self.f_start} {self.f_stop}",
            ".option filetype=ascii",
            ".save all",
        ]


class DC(_BaseAnalysis):
    """Varredura DC nativa do SPICE (ex.: .dc V1 0 5 0.1).

    `source_ref` é o **ref** da fonte de tensão/corrente (sem a letra),
    ex.: para Vdc("1", 5.0) use source_ref="1" → gera ".dc V1 ..."
    """

    def __init__(self, source_ref: str, start: float, stop: float, step: float) -> None:
        self.source_ref = source_ref
        self.start = start
        self.stop = stop
        self.step = step

    def _directives(self) -> list[str]:
        return [
            f".dc V{self.source_ref} {self.start} {self.stop} {self.step}",
            ".option filetype=ascii",
            ".save all",
        ]
