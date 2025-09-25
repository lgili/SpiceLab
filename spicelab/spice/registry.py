from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from .base import RunResult, SimulatorAdapter
from .ltspice_cli import DEFAULT_ADAPTER as LTSPICE_DEFAULT_ADAPTER
from .ngspice_cli import DEFAULT_ADAPTER as NGSPICE_DEFAULT_ADAPTER
from .xyce_cli import DEFAULT_ADAPTER as XYCE_DEFAULT_ADAPTER

Runner = Callable[[str, Sequence[str]], RunResult]


def _adapter_to_runner(adapter: SimulatorAdapter) -> Runner:
    def runner(netlist_text: str, directives: Sequence[str]) -> RunResult:
        return adapter.run_directives(netlist_text, directives)

    return runner


class _FunctionAdapter(SimulatorAdapter):
    def __init__(self, func: Runner) -> None:
        self._func = func

    def run_directives(self, netlist_text: str, directives: Sequence[str]) -> RunResult:
        return self._func(netlist_text, directives)


class _SimulatorRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SimulatorAdapter] = {}
        self._aliases: dict[str, str] = {}
        self._active_name: str | None = None
        self._active_adapter: SimulatorAdapter = _FunctionAdapter(
            NGSPICE_DEFAULT_ADAPTER.run_directives
        )
        self._active_runner: Runner = _adapter_to_runner(self._active_adapter)

    # -- registration -----------------------------------------------------------------
    def register(
        self,
        name: str,
        adapter: SimulatorAdapter,
        *,
        aliases: Sequence[str] = (),
        make_default: bool = False,
    ) -> None:
        self._adapters[name] = adapter
        for alias in aliases:
            self._aliases[alias] = name
        if make_default or self._active_name is None:
            self._activate(name)

    def available(self) -> list[str]:
        return list(self._adapters.keys())

    # -- activation -------------------------------------------------------------------
    def use(self, key: str) -> SimulatorAdapter:
        name = self._resolve(key)
        self._activate(name)
        return self._active_adapter

    def set_adapter(self, adapter: SimulatorAdapter) -> None:
        self._active_name = None
        self._active_adapter = adapter
        self._active_runner = _adapter_to_runner(adapter)

    def set_callable(self, func: Runner) -> None:
        self._active_name = None
        self._active_adapter = _FunctionAdapter(func)
        self._active_runner = func

    # -- accessors --------------------------------------------------------------------
    def get_runner(self) -> Runner:
        return self._active_runner

    def get_adapter(self) -> SimulatorAdapter:
        return self._active_adapter

    # -- helpers ----------------------------------------------------------------------
    def _activate(self, name: str) -> None:
        adapter = self._adapters[name]
        self._active_name = name
        self._active_adapter = adapter
        self._active_runner = _adapter_to_runner(adapter)

    def _resolve(self, key: str) -> str:
        if key in self._adapters:
            return key
        if key in self._aliases:
            return self._aliases[key]
        raise KeyError(f"Simulator adapter '{key}' is not registered")


_registry = _SimulatorRegistry()
_registry.register(
    "ngspice-cli",
    NGSPICE_DEFAULT_ADAPTER,
    aliases=("ngspice", "default"),
    make_default=True,
)
_registry.register(
    "ltspice-cli",
    LTSPICE_DEFAULT_ADAPTER,
    aliases=("ltspice",),
)
_registry.register(
    "xyce-cli",
    XYCE_DEFAULT_ADAPTER,
    aliases=("xyce",),
)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def register_adapter(
    name: str,
    adapter: SimulatorAdapter,
    *,
    aliases: Sequence[str] = (),
    make_default: bool = False,
) -> None:
    """Register a named simulator adapter."""

    _registry.register(name, adapter, aliases=aliases, make_default=make_default)


def list_adapters() -> list[str]:
    """Return the list of registered adapter names."""

    return _registry.available()


def use_adapter(name: str) -> SimulatorAdapter:
    """Activate a registered adapter by name (or alias)."""

    return _registry.use(name)


def get_active_adapter() -> SimulatorAdapter:
    """Return the adapter currently used by analyses."""

    return _registry.get_adapter()


def set_run_directives(handler: Runner | SimulatorAdapter | str) -> None:
    """Compatibility helper to override the active runner.

    Accepts:
        - a callable ``Runner`` (legacy behaviour)
        - a ``SimulatorAdapter`` instance
        - the name/alias of a registered adapter
    """

    if isinstance(handler, str):
        use_adapter(handler)
        return

    if hasattr(handler, "run_directives"):
        adapter = cast(SimulatorAdapter, handler)
        _registry.set_adapter(adapter)
        return

    _registry.set_callable(handler)


def get_run_directives() -> Runner:
    """Return the currently active runner callable (legacy API)."""

    return _registry.get_runner()
