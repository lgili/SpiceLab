from __future__ import annotations

from collections.abc import Callable, Sequence

from . import ngspice_cli
from .base import RunResult

Runner = Callable[[str, Sequence[str]], RunResult]


def _default_runner(net: str, dirs: Sequence[str]) -> RunResult:
    return ngspice_cli.run_directives(net, dirs)


_runner: Runner = _default_runner


def set_run_directives(func: Runner) -> None:
    global _runner
    _runner = func


def get_run_directives() -> Runner:
    return _runner
