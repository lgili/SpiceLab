from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec, Probe, ResultHandle, ResultMeta, SweepSpec, circuit_hash
from ..io.readers import read_xyce_table
from ..spice.xyce_cli import DEFAULT_ADAPTER
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
        src_ref = src if str(src).upper().startswith("V") else f"V{src}"
        return [f".dc {src_ref} {start} {stop} {step}"]
    raise NotImplementedError(f"Unsupported analysis mode: {mode}")


class XyceSimulator(Simulator):
    def __init__(self) -> None:
        self._features = EngineFeatures(name="xyce-cli", supports_parallel=True)

    def features(self) -> EngineFeatures:
        return self._features

    def run(
        self,
        circuit: object,
        analyses: Sequence[AnalysisSpec],
        sweep: SweepSpec | None = None,
        probes: list[Probe] | None = None,
    ) -> ResultHandle:
        if sweep is not None and sweep.variables:
            raise NotImplementedError("SweepSpec is not supported yet in XyceSimulator.run()")

        directives: list[str] = []
        for spec in analyses:
            directives.extend(_spec_to_directives(spec))

        if not hasattr(circuit, "build_netlist"):
            raise TypeError("circuit must provide build_netlist()")
        net = circuit.build_netlist()

        res = DEFAULT_ADAPTER.run_directives(net, directives)
        if res.returncode != 0:
            raise RuntimeError(f"Xyce failed, return code {res.returncode}")
        if not res.artifacts.raw_path:
            raise RuntimeError("Xyce did not produce a .prn/.csv file")

        ds = read_xyce_table(res.artifacts.raw_path)
        try:
            if "time" in ds:
                ds = ds.set_coords("time")
            if "freq" in ds:
                ds = ds.set_coords("freq")
        except Exception:
            pass

        nl_hash = circuit_hash(circuit, extra={"analyses": [spec.mode for spec in analyses]})
        meta = ResultMeta(
            engine="xyce",
            engine_version=None,
            netlist_hash=nl_hash,
            analyses=list(analyses),
            probes=list(probes) if probes else [],
            attrs={
                "workdir": res.artifacts.workdir,
                "log_path": res.artifacts.log_path,
                "netlist_path": res.artifacts.netlist_path,
                "analysis_modes": [spec.mode for spec in analyses],
                "analysis_params": [{"mode": spec.mode, **spec.args} for spec in analyses],
            },
        )
        return DatasetResultHandle(ds, meta)


__all__ = ["XyceSimulator"]
