from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec, Probe, ResultHandle, ResultMeta, SweepSpec, circuit_hash
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
        probes: list[Probe] | None = None,
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
        # Best-effort engine version from log
        eng_ver: str | None = None
        try:
            with open(res.artifacts.log_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip().lower()
                    if s.startswith("ngspice") and ("version" in s or "ngspice" in s):
                        eng_ver = line.strip()
                        break
        except Exception:
            eng_ver = None

        # Deterministic netlist hash including analysis modes
        nl_hash = circuit_hash(circuit, extra={"analyses": [spec.mode for spec in analyses]})

        # Ensure likely independent variable is a coordinate
        try:
            if "time" in ds:
                ds = ds.set_coords("time")
            if "freq" in ds:
                ds = ds.set_coords("freq")
        except Exception:
            pass

        # If this is a DC analysis, normalize the sweep coordinate name to 'sweep'
        # and record the original label and source name in attrs for convenience.
        dc_specs = [s for s in analyses if s.mode == "dc"]
        original_sweep_name: str | None = None
        try:
            if dc_specs:
                # find first coord that isn't index/time/freq
                for cname in list(ds.coords):
                    cl = str(cname).lower()
                    if cl not in {"index", "time", "freq", "frequency"}:
                        original_sweep_name = str(cname)
                        if original_sweep_name != "sweep":
                            ds = ds.rename({original_sweep_name: "sweep"})
                            ds = ds.set_coords("sweep")
                        break
        except Exception:
            pass

        # Attach convenience sweep attrs on the dataset itself for DC
        try:
            if dc_specs:
                src_val = dc_specs[0].args.get("src")
                src_name = str(src_val) if src_val is not None else None
                # Heuristic unit by source type
                unit: str | None = None
                if src_name:
                    up = src_name.upper()
                    if up.startswith("V"):
                        unit = "V"
                    elif up.startswith("I"):
                        unit = "A"
                ds.attrs["sweep_src"] = src_name
                ds.attrs["sweep_unit"] = unit
                if original_sweep_name:
                    ds.attrs["sweep_label"] = original_sweep_name
        except Exception:
            pass

        # Normalize analysis metadata for downstream consumers
        analysis_modes = [spec.mode for spec in analyses]
        analysis_params = [{"mode": spec.mode, **spec.args} for spec in analyses]

        meta = ResultMeta(
            engine="ngspice",
            engine_version=eng_ver,
            netlist_hash=nl_hash,
            analyses=list(analyses),
            probes=list(probes) if probes else [],
            attrs={
                "workdir": res.artifacts.workdir,
                "log_path": res.artifacts.log_path,
                "netlist_path": res.artifacts.netlist_path,
                # Normalized analysis summaries
                "analysis_modes": analysis_modes,
                "analysis_params": analysis_params,
                # DC sweep convenience
                **(
                    {
                        "dc_sweep_label": original_sweep_name,
                        "dc_src": dc_specs[0].args.get("src"),
                    }
                    if dc_specs
                    else {}
                ),
            },
        )
        return DatasetResultHandle(ds, meta)


__all__ = ["NgSpiceSimulator"]
