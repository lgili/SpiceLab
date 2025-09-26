from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import EngineFeatures, Simulator
from .result import DatasetResultHandle

__all__ = [
    "EngineFeatures",
    "Simulator",
    "DatasetResultHandle",
    "NgSpiceSimulator",
    "NgSpiceProcSimulator",
    "NgSpiceSharedSimulator",
    "LtSpiceSimulator",
    "LtSpiceCLISimulator",
    "XyceSimulator",
    "XyceCLISimulator",
    "get_simulator",
    "run_simulation",
    "EngineName",
    "Job",
    "JobResult",
    "run_job",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "NgSpiceSimulator": (".ngspice", "NgSpiceSimulator"),
    "NgSpiceProcSimulator": (".ngspice_proc", "NgSpiceProcSimulator"),
    "NgSpiceSharedSimulator": (".ngspice_shared", "NgSpiceSharedSimulator"),
    "LtSpiceSimulator": (".ltspice", "LtSpiceSimulator"),
    "LtSpiceCLISimulator": (".ltspice_cli", "LtSpiceCLISimulator"),
    "XyceSimulator": (".xyce", "XyceSimulator"),
    "XyceCLISimulator": (".xyce_cli", "XyceCLISimulator"),
    "get_simulator": (".orchestrator", "get_simulator"),
    "run_simulation": (".orchestrator", "run_simulation"),
    "EngineName": (".orchestrator", "EngineName"),
    "Job": ("..orchestrator", "Job"),
    "JobResult": ("..orchestrator", "JobResult"),
    "run_job": ("..orchestrator", "run_job"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'spicelab.engines' has no attribute {name!r}") from exc

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value
