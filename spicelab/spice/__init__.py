"""Public exports for :mod:`spicelab.spice` used by docs tooling.

Expose a small, explicit surface for registry helpers.
"""

from .registry import (
    Runner,
    get_active_adapter,
    get_run_directives,
    list_adapters,
    register_adapter,
    set_run_directives,
    use_adapter,
)

__all__ = [
    "Runner",
    "set_run_directives",
    "get_run_directives",
    "get_active_adapter",
    "register_adapter",
    "use_adapter",
    "list_adapters",
]
