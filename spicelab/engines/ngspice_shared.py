"""Shared-library driven ngspice simulator with co-simulation callbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import xarray as xr
else:  # pragma: no cover - optional dependency guard
    try:
        xr = cast(Any, __import__("xarray"))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xarray is required to use NgSpiceSharedSimulator; install it with "
            "`pip install xarray`."
        ) from exc

from ..core.types import AnalysisSpec, Probe, ResultHandle, ResultMeta, SweepSpec, circuit_hash
from ..spice.ngspice_shared import (
    ExternalSourceRequest,
    NgSpiceSharedAdapter,
    SharedCallbacks,
)
from .base import EngineFeatures, Simulator
from .directives import specs_to_directives
from .result import DatasetResultHandle

__all__ = ["NgSpiceSharedSimulator", "SharedSimulationConfig"]


@dataclass(frozen=True)
class SharedSimulationConfig:
    """High-level configuration recognised by :class:`NgSpiceSharedSimulator`."""

    callbacks: SharedCallbacks | None = None
    external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] | None = None


class NgSpiceSharedSimulator(Simulator):
    """Execute transient analyses through the libngspice shared library."""

    def __init__(self, adapter: NgSpiceSharedAdapter | None = None) -> None:
        self._adapter = adapter or NgSpiceSharedAdapter()
        self._features = EngineFeatures(
            name="ngspice-shared",
            supports_noise=True,
            supports_callbacks=True,
            supports_shared_lib=True,
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
            raise NotImplementedError("SweepSpec is not supported with ngspice shared yet")
        if len(analyses) != 1:
            raise NotImplementedError(
                "ngspice shared simulator currently expects exactly one analysis"
            )
        analysis = analyses[0]
        if analysis.mode != "tran":
            raise NotImplementedError(
                "ngspice shared simulator currently supports only transient analyses"
            )

        config = _extract_shared_config(analysis)

        if not hasattr(circuit, "build_netlist"):
            raise TypeError("circuit must provide build_netlist()")
        net = circuit.build_netlist()
        directives = specs_to_directives([analysis])

        adapter_result = self._adapter.run(
            net,
            directives,
            config.callbacks,
            config.external_sources,
        )

        dataset = _ensure_coord_dataset(adapter_result.dataset)

        nl_hash = circuit_hash(circuit, extra={"analyses": [analysis.mode]})
        analysis_args_clean = {
            k: v for k, v in analysis.args.items() if k not in {"callbacks", "external_sources"}
        }

        attrs = {
            "engine_features": {
                "supports_callbacks": self._features.supports_callbacks,
                "supports_shared_lib": self._features.supports_shared_lib,
                "supports_noise": self._features.supports_noise,
                "supports_verilog_a": self._features.supports_verilog_a,
                "supports_parallel": self._features.supports_parallel,
            },
            "analysis_modes": [analysis.mode],
            "analysis_params": [{"mode": analysis.mode, **analysis_args_clean}],
            "log_lines": list(adapter_result.log_lines),
        }
        if config.callbacks is not None:
            enabled = [
                name
                for name, cb in {
                    "on_init": config.callbacks.on_init,
                    "on_exit": config.callbacks.on_exit,
                    "on_message": config.callbacks.on_message,
                    "on_tran_point": config.callbacks.on_tran_point,
                }.items()
                if cb is not None
            ]
            attrs["callbacks"] = enabled
        if config.external_sources:
            attrs["external_sources"] = sorted(config.external_sources.keys())

        meta = ResultMeta(
            engine="ngspice-shared",
            engine_version=None,
            netlist_hash=nl_hash,
            analyses=[analysis],
            probes=list(probes or []),
            attrs=attrs,
        )
        return DatasetResultHandle(dataset, meta)


def _extract_shared_config(analysis: AnalysisSpec) -> SharedSimulationConfig:
    args = dict(analysis.args)
    callbacks_raw = args.pop("callbacks", None)
    external_raw = args.pop("external_sources", None)

    callbacks: SharedCallbacks | None = None
    if callbacks_raw:
        callbacks = SharedCallbacks(
            on_init=callbacks_raw.get("on_init"),
            on_exit=callbacks_raw.get("on_exit"),
            on_message=callbacks_raw.get("on_message"),
            on_tran_point=callbacks_raw.get("on_tran_point"),
        )
    external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] | None = None
    if external_raw:
        external_sources = dict(external_raw)
    return SharedSimulationConfig(callbacks=callbacks, external_sources=external_sources)


def _ensure_coord_dataset(obj: Any) -> Any:
    if isinstance(obj, xr.Dataset):
        ds = obj
    elif hasattr(obj, "to_dataset"):
        ds = obj.to_dataset()
    else:
        raise TypeError(
            "Shared adapter must return an xarray.Dataset (or object exposing to_dataset())"
        )
    try:
        if "time" in ds and "time" not in ds.coords:
            ds = ds.set_coords("time")
    except Exception:
        pass
    return ds
