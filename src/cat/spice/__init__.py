"""Public exports for :mod:`cat.spice` used by docs tooling.

Expose a small, explicit surface for registry helpers.
"""

from .registry import Runner, get_run_directives, set_run_directives

__all__ = ["set_run_directives", "get_run_directives", "Runner"]
