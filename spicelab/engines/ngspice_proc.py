"""NGSpice simulator that shells out to the CLI (process mode).

This module contains the concrete :class:`Simulator` implementation that backs
``engine="ngspice"`` during the M3 milestone.  The previous implementation lived
in :mod:`spicelab.engines.ngspice`; that module now re-exports the
``NgSpiceProcSimulator`` class defined here for backwards compatibility.

The simulator is intentionally lightweight:

* Constructs a deck from ``Circuit.build_netlist()``
* Delegates execution to the active adapter registered in
  :mod:`spicelab.spice.registry` (``ngspice_cli`` by default)
* Normalises the resulting dataset via the unified readers delivered in M2
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.types import AnalysisSpec, Probe, ResultHandle, ResultMeta, SweepSpec, circuit_hash
from ..io.readers import read_ngspice_raw
from ..spice.registry import get_active_adapter
from .base import EngineFeatures, Simulator
from .directives import specs_to_directives
from .result import DatasetResultHandle


class NgSpiceProcSimulator(Simulator):
    """Execute analyses by invoking the ``ngspice`` binary via subprocess."""

    def __init__(self) -> None:
        self._features = EngineFeatures(
            name="ngspice-cli",
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
            raise NotImplementedError(
                "SweepSpec is not yet supported in NgSpiceProcSimulator.run()"
            )

        directives = specs_to_directives(analyses)

        adapter = get_active_adapter()
        if not hasattr(circuit, "build_netlist"):
            raise TypeError("circuit must provide build_netlist()")
        net = circuit.build_netlist()
        res = adapter.run_directives(net, directives)
        if res.returncode != 0:
            raise RuntimeError(f"ngspice failed, return code {res.returncode}")
        if not res.artifacts.raw_path:
            raise RuntimeError("ngspice did not produce a RAW file path")

        ds = read_ngspice_raw(res.artifacts.raw_path)

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

        nl_hash = circuit_hash(circuit, extra={"analyses": [spec.mode for spec in analyses]})

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

        try:
            if dc_specs:
                src_val = dc_specs[0].args.get("src")
                src_name = str(src_val) if src_val is not None else None
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

        analysis_modes = [spec.mode for spec in analyses]
        analysis_params = [{"mode": spec.mode, **spec.args} for spec in analyses]

        feats = self.features()
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
                "engine_features": {
                    "supports_callbacks": feats.supports_callbacks,
                    "supports_shared_lib": feats.supports_shared_lib,
                    "supports_noise": feats.supports_noise,
                    "supports_verilog_a": feats.supports_verilog_a,
                    "supports_parallel": feats.supports_parallel,
                },
                "analysis_modes": analysis_modes,
                "analysis_params": analysis_params,
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


__all__ = ["NgSpiceProcSimulator"]
