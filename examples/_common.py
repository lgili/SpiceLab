from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from spicelab.engines import run_simulation
    from spicelab.engines.exceptions import EngineBinaryNotFound, EngineSharedLibraryNotFound
except ModuleNotFoundError:  # pragma: no cover - when running examples directly
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from spicelab.engines import run_simulation  # noqa: E402
    from spicelab.engines.exceptions import (  # noqa: E402
        EngineBinaryNotFound,
        EngineSharedLibraryNotFound,
    )

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


# --------------------------------------------------------------------------------------
# CLI helpers shared by the runnable examples
# --------------------------------------------------------------------------------------


def _default_engine(env_default: str) -> str:
    for var in ("SPICELAB_EXAMPLE_ENGINE", "SPICELAB_ENGINE"):
        val = os.getenv(var)
        if val:
            return val
    return env_default


def add_engine_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str = "ngspice",
    required: bool = False,
) -> None:
    parser.add_argument(
        "--engine",
        default=_default_engine(default) if not required else None,
        required=required,
        help=(
            "Engine name to use (ngspice, ltspice, xyce, ngspice-shared). "
            "Defaults to SPICELAB_EXAMPLE_ENGINE/SPICELAB_ENGINE or 'ngspice'."
        ),
    )


def ensure_examples_package() -> None:
    """Ensure the project root is on sys.path when running a script directly."""

    if __package__ in {None, ""}:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))


def run_or_fail(
    circuit: object,
    analyses: Sequence[object],
    *,
    engine: str,
    sweep: object | None = None,
    probes: Sequence[object] | None = None,
    **kwargs: Any,
) -> Any:
    """Run a simulation and exit the script gracefully on configuration errors."""

    try:
        return run_simulation(
            circuit,
            analyses,
            sweep=sweep,
            probes=probes,
            engine=engine,
            **kwargs,
        )
    except EngineBinaryNotFound as exc:
        print(exc)
        raise SystemExit(1) from exc
    except EngineSharedLibraryNotFound as exc:
        print(exc)
        raise SystemExit(1) from exc
    except FileNotFoundError as exc:
        print(f"Engine '{engine}' not available: {exc}")
        raise SystemExit(1) from exc


def resolve_engine(engine: str | None, *, fallback: str = "ngspice") -> str:
    if engine:
        return engine
    return _default_engine(fallback)


def parser_with_engine(description: str, *, default: str = "ngspice") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_engine_argument(parser, default=default, required=False)
    return parser


def print_header(title: str, engine: str | None = None) -> None:
    banner = f"== {title} =="
    if engine:
        banner += f" [engine: {engine}]"
    print(banner)


def with_examples_package(fn: Callable[[], None]) -> None:
    ensure_examples_package()
    fn()
