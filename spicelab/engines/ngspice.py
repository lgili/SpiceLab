from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec, ResultHandle, ResultMeta, SweepSpec
from ..io.readers import read_ltspice_raw
from ..spice.registry import get_active_adapter
from .base import EngineFeatures, Simulator
from .result import DatasetResultHandle


def _spec_to_directives(spec: AnalysisSpec) -> list[str]:
    mode = spec.mode
    a = spec.args
    if mode == "op":
        return [".op"]
    if mode == "tran":
        tstep = a.get("tstep")
        tstop = a.get("tstop")
        tstart = a.get("tstart")
        if tstep is None or tstop is None:
            raise ValueError("TRAN requires 'tstep' and 'tstop' args")
        if tstart is not None:
            return [f".tran {tstep} {tstop} {tstart}"]
        return [f".tran {tstep} {tstop}"]
    if mode == "ac":
        sweep_type = a.get("sweep_type", "dec")
        n = a.get("n")
        fstart = a.get("fstart")
        fstop = a.get("fstop")
        if n is None or fstart is None or fstop is None:
            raise ValueError("AC requires 'n', 'fstart', 'fstop' args")
        return [f".ac {sweep_type} {n} {fstart} {fstop}"]
    if mode == "dc":
        src = a.get("src")
        start = a.get("start")
        stop = a.get("stop")
        step = a.get("step")
        if src is None or start is None or stop is None or step is None:
            raise ValueError("DC requires 'src', 'start', 'stop', 'step' args")
        # Allow either `V1` or bare reference like `in` -> NGSpice expects `V<name>`
        src_ref = src if src.upper().startswith("V") else f"V{src}"
        return [f".dc {src_ref} {start} {stop} {step}"]
    if mode == "noise":
        # Placeholder: NGSpice noise syntax is richer; fill later
        raise NotImplementedError("noise analysis mapping not implemented yet")
    raise NotImplementedError(f"Unsupported analysis mode: {mode}")


class NgSpiceSimulator(Simulator):
    def __init__(self) -> None:
        self._features = EngineFeatures(name="ngspice-cli", supports_noise=False)

    def features(self) -> EngineFeatures:
        return self._features

    def run(
        self,
        circuit: object,
        analyses: Sequence[AnalysisSpec],
        sweep: SweepSpec | None = None,
    ) -> ResultHandle:
        if sweep is not None and sweep.variables:
            # Not yet implemented in this thin adapter; will be added in sweep orchestrator
            raise NotImplementedError("SweepSpec is not yet supported in NgSpiceSimulator.run()")

        directives: list[str] = []
        for spec in analyses:
            directives.extend(_spec_to_directives(spec))

        adapter = get_active_adapter()
        # Accept both Circuit and compatible objects with build_netlist()
        if not hasattr(circuit, "build_netlist"):
            raise TypeError("circuit must provide build_netlist()")
        net = circuit.build_netlist()
        res = adapter.run_directives(net, directives)
        if res.returncode != 0:
            raise RuntimeError(f"ngspice failed, return code {res.returncode}")
        if not res.artifacts.raw_path:
            raise RuntimeError("ngspice did not produce a RAW file path")

        ds = read_ltspice_raw(res.artifacts.raw_path)
        meta = ResultMeta(
            engine="ngspice",
            analyses=list(analyses),
            attrs={
                "workdir": res.artifacts.workdir,
                "log_path": res.artifacts.log_path,
                "netlist_path": res.artifacts.netlist_path,
            },
        )
        return DatasetResultHandle(ds, meta)


__all__ = ["NgSpiceSimulator"]
