from __future__ import annotations

import importlib
import os
from pathlib import Path
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
    target = Path(name)

    # Handle Plotly-based VizFigure
    if hasattr(fig, "to_image") and hasattr(fig, "to_html"):
        try:
            saved_path = fig.to_image(target)
        except Exception:
            html_path = target.with_suffix(".html")
            fig.to_html(html_path, include_plotlyjs="cdn")
            saved_path = html_path
        print(f"[saved] {saved_path.resolve()}")
        return str(saved_path.resolve())

    # Fallback to Matplotlib Figure API
    out = target.resolve()
    if hasattr(fig, "savefig"):
        fig.savefig(out, dpi=150)
        print(f"[saved] {out}")
        if plt is not None:
            cast(Any, plt).close(fig)
        return str(out)

    raise TypeError(f"Unsupported figure type: {type(fig)!r}")
