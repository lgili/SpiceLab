from __future__ import annotations

import importlib
import os
from typing import Any, cast

matplotlib: Any | None
plt: Any | None
try:
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(fig: Any, name: str) -> str:
    ensure_dir(".")
    out = os.path.abspath(name)
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")
    if plt is not None:
        cast(Any, plt).close(fig)
    return out
