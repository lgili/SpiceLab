from __future__ import annotations

from importlib import import_module
from typing import cast

from .base import Simulator

__all__ = ["create_simulator"]

_SIMULATOR_MODULES = {
    "ngspice": ("spicelab.engines.ngspice", "NgSpiceSimulator"),
    "ngspice-cli": ("spicelab.engines.ngspice_proc", "NgSpiceProcSimulator"),
    "ngspice-shared": ("spicelab.engines.ngspice_shared", "NgSpiceSharedSimulator"),
    "ltspice": ("spicelab.engines.ltspice", "LtSpiceSimulator"),
    "ltspice-cli": ("spicelab.engines.ltspice_cli", "LtSpiceCLISimulator"),
    "xyce": ("spicelab.engines.xyce", "XyceSimulator"),
    "xyce-cli": ("spicelab.engines.xyce_cli", "XyceCLISimulator"),
}

_ENGINE_ALIASES = {
    "ng": "ngspice",
    "ngspice": "ngspice",
    "ngspice-cli": "ngspice-cli",
    "ngspice-shared": "ngspice-shared",
    "lt": "ltspice",
    "ltspice": "ltspice",
    "ltspice-cli": "ltspice-cli",
    "xy": "xyce",
    "xyce": "xyce",
    "xyce-cli": "xyce-cli",
}


def _load_simulator(name: str) -> type[Simulator]:
    module_path, attr = _SIMULATOR_MODULES[name]
    module = import_module(module_path)
    cls = getattr(module, attr)
    if not isinstance(cls, type):
        raise TypeError(f"Attribute '{attr}' in module '{module_path}' is not a class")
    return cast(type[Simulator], cls)


def create_simulator(name: str) -> Simulator:
    key = _ENGINE_ALIASES.get(name.lower())
    if key is None or key not in _SIMULATOR_MODULES:
        allowed = ", ".join(sorted(_SIMULATOR_MODULES))
        raise ValueError(f"Unknown engine '{name}'. Expected one of: {allowed}")
    simulator_cls = _load_simulator(key)
    return simulator_cls()
