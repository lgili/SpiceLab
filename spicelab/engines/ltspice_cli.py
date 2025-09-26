"""LTspice simulator that runs the official CLI in headless mode."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec, Probe, ResultHandle, ResultMeta, SweepSpec, circuit_hash
from ..io.readers import read_ltspice_raw
from ..spice.ltspice_cli import DEFAULT_ADAPTER
from .base import EngineFeatures, Simulator
from .directives import specs_to_directives
from .result import DatasetResultHandle


class LtSpiceCLISimulator(Simulator):
    """Execute analyses using the LTspice command-line interface."""

    def __init__(self) -> None:
        self._features = EngineFeatures(
            name="ltspice-cli",
            supports_noise=True,
            supports_callbacks=False,
            supports_shared_lib=False,
            supports_verilog_a=False,
            supports_parallel=False,
        )

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
            raise NotImplementedError("SweepSpec is not supported yet in LtSpiceCLISimulator.run()")

        directives = specs_to_directives(analyses)

        if not hasattr(circuit, "build_netlist"):
            raise TypeError("circuit must provide build_netlist()")
        net = circuit.build_netlist()

        res = DEFAULT_ADAPTER.run_directives(net, directives)
        if res.returncode != 0:
            log_hint = f"; see {res.artifacts.log_path}" if res.artifacts.log_path else ""
            raise RuntimeError(f"LTspice failed (return code {res.returncode}){log_hint}")
        if not res.artifacts.raw_path:
            log_hint = f"; see {res.artifacts.log_path}" if res.artifacts.log_path else ""
            raise RuntimeError(
                f"LTspice did not produce a RAW file path (ASCII may be required){log_hint}"
            )

        ds = read_ltspice_raw(res.artifacts.raw_path)
        try:
            if "time" in ds:
                ds = ds.set_coords("time")
            if "freq" in ds:
                ds = ds.set_coords("freq")
        except Exception:
            pass

        dc_specs = [s for s in analyses if s.mode == "dc"]
        original_sweep_name: str | None = None
        try:
            if dc_specs:
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

        nl_hash = circuit_hash(circuit, extra={"analyses": [spec.mode for spec in analyses]})
        feats = self.features()
        meta = ResultMeta(
            engine="ltspice",
            engine_version=None,
            netlist_hash=nl_hash,
            analyses=list(analyses),
            probes=list(probes) if probes else [],
            attrs={
                "workdir": res.artifacts.workdir,
                "log_path": res.artifacts.log_path,
                "netlist_path": res.artifacts.netlist_path,
                "engine_features": {
                    "supports_callbacks": feats.supports_callbacks,
                    "supports_shared_lib": feats.supports_shared_lib,
                    "supports_noise": feats.supports_noise,
                    "supports_verilog_a": feats.supports_verilog_a,
                    "supports_parallel": feats.supports_parallel,
                },
                "analysis_modes": [spec.mode for spec in analyses],
                "analysis_params": [{"mode": spec.mode, **spec.args} for spec in analyses],
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


__all__ = ["LtSpiceCLISimulator"]
