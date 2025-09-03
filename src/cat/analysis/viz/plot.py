from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from ...io.raw_reader import TraceSet


def _ensure_pyplot() -> Any:
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc


def plot_traces(ts: TraceSet, ys: Sequence[str] | None = None, title: str | None = None) -> Any:
    """
    Plota Y(s) vs eixo X do TraceSet. `ys=None` => plota todos (menos o eixo X).
    Retorna a figure do matplotlib.
    """
    plt = _ensure_pyplot()
    x = ts.x.values
    xname = ts.x.name
    names = [n for n in ts.names if n != xname] if ys is None else list(ys)

    fig = plt.figure()
    ax = fig.gca()
    for n in names:
        ax.plot(x, ts[n].values, label=n)
    ax.set_xlabel(xname)
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sweep_df(
    df: Any,
    x: str,
    y: str,
    hue: str,
    title: str | None = None,
) -> Any:
    """
    Plota um DataFrame empilhado por parâmetro (ex.: retornado por stack_step_to_df).
    Uma curva por valor distinto de `hue`.
    """
    plt = _ensure_pyplot()
    fig = plt.figure()
    ax = fig.gca()
    for val, g in df.groupby(hue):
        ax.plot(g[x].values, g[y].values, label=f"{hue}={val}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
