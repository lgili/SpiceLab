"""Compatibility layer for the previous 'spicelab' import path.

This package forwards imports to the new 'spicelab' package so that existing
code using 'import spicelab' or 'from spicelab.* import ...' continues to work.

It installs a lightweight meta path finder that maps any 'spicelab.<subpath>' import
to 'spicelab.<subpath>' at import time, avoiding the need to create many stub
modules on disk. Top-level symbols are re-exported from spicelab for convenience.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType
from typing import Any

# Re-export the top-level API from spicelab
from spicelab import *  # noqa: F401,F403


def __getattr__(name: str) -> Any:  # pragma: no cover - simple passthrough
    return getattr(importlib.import_module("spicelab"), name)


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(set(dir(importlib.import_module("spicelab"))))


class _CatProxyLoader(importlib.abc.Loader):
    def __init__(self, target_name: str) -> None:
        self._target = target_name

    def create_module(self, spec):  # type: ignore[override]
        # Defer to default module creation
        return None

    def exec_module(self, module: ModuleType) -> None:  # type: ignore[override]
        target = importlib.import_module(self._target)
        # Alias the target module under the 'spicelab.*' name
        sys.modules[module.__name__] = target


class _CatProxyFinder(importlib.abc.MetaPathFinder):
    prefix = "spicelab."

    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if not fullname.startswith(self.prefix):
            return None
        mapped = "spicelab." + fullname[len(self.prefix) :]
        target_spec = importlib.util.find_spec(mapped)
        if target_spec is None:
            return None
        # Build a new spec for the alias module
        loader = _CatProxyLoader(mapped)
        is_pkg = target_spec.submodule_search_locations is not None
        spec = importlib.machinery.ModuleSpec(fullname, loader, is_package=is_pkg)
        if is_pkg:
            # Reuse the target package's search locations so submodules resolve
            spec.submodule_search_locations = target_spec.submodule_search_locations  # type: ignore[attr-defined]
        return spec


def _install_proxy() -> None:
    # Install our finder once at import time, if not already installed
    for f in sys.meta_path:
        if isinstance(f, _CatProxyFinder):
            return
    sys.meta_path.insert(0, _CatProxyFinder())


_install_proxy()
