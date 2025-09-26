"""Xyce simulator that shells out to the Xyce CLI."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec, Probe, ResultHandle, ResultMeta, SweepSpec, circuit_hash
from ..io.readers import read_xyce_table
from ..spice.xyce_cli import DEFAULT_ADAPTER, _build_print_directives
from .base import EngineFeatures, Simulator
from .directives import specs_to_directives
from .result import DatasetResultHandle


class XyceCLISimulator(Simulator):
    """Execute analyses using the official Xyce command-line binary."""

    def __init__(self) -> None:
        self._features = EngineFeatures(
            name="xyce-cli",
            supports_parallel=True,
            supports_noise=True,
            supports_verilog_a=True,
            supports_callbacks=False,
            supports_shared_lib=False,
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
            raise NotImplementedError("SweepSpec is not supported yet in XyceCLISimulator.run()")

        directives = list(specs_to_directives(analyses))
        directives.extend(_build_print_directives(analyses, probes))

        if not hasattr(circuit, "build_netlist"):
            raise TypeError("circuit must provide build_netlist()")
        net = circuit.build_netlist()

        res = DEFAULT_ADAPTER.run_directives(net, directives)
        if res.returncode != 0:
            raise RuntimeError(
                f"Xyce failed (code {res.returncode}); inspect {res.artifacts.log_path} for details"
            )
        if not res.artifacts.raw_path:
            raise RuntimeError("Xyce did not produce a .prn/.csv file")

        ds = read_xyce_table(res.artifacts.raw_path)
        try:
            for name in list(ds.data_vars):
                if name.startswith("Re(") and name.endswith(")"):
                    base = name[3:-1]
                    imag_name = f"Im({base})"
                    if imag_name in ds.data_vars and base not in ds.data_vars:
                        ds[base] = ds[name] + 1j * ds[imag_name]
        except Exception:
            pass
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
            engine="xyce",
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


__all__ = ["XyceCLISimulator"]
