from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...io.raw_reader import TraceSet


def _ensure_pyplot() -> Any:
    """Importa `matplotlib.pyplot` sob demanda, com erro amigável se não instalado."""
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc


def _pick_x(ts: TraceSet) -> tuple[NDArray[Any], str]:
    """Escolhe o eixo X automaticamente e retorna (valores, nome).

    Preferência: atributo `ts.x` (se existir e for válido) → 'time' → 'frequency' → primeira coluna.
    """
    # Compatibilidade: algumas versões de TraceSet podem expor `x` como uma Trace
    x_attr = getattr(ts, "x", None)
    if x_attr is not None:
        try:
            return np.asarray(x_attr.values, dtype=float), getattr(x_attr, "name", "x")
        except Exception:  # pragma: no cover — cair para heurística de nomes
            pass

    names_lower = [n.lower() for n in ts.names]
    if "time" in names_lower:
        name = ts.names[names_lower.index("time")]
        return np.asarray(ts[name].values, dtype=float), name
    if "frequency" in names_lower:
        name = ts.names[names_lower.index("frequency")]
        return np.asarray(ts[name].values, dtype=float), name
    # fallback: primeira coluna
    first = ts.names[0]
    return np.asarray(ts[first].values, dtype=float), first


def plot_traces(
    ts: TraceSet,
    ys: Sequence[str] | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    grid: bool = True,
    tight: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plota uma ou mais *traces* de um :class:`TraceSet` contra o eixo X escolhido.

    - Se ``ys=None``, plota todas as colunas exceto o eixo X.
    - Retorna a *figure* do Matplotlib (``ax.figure`` quando um ``ax`` é fornecido).
    """
    plt = _ensure_pyplot()

    x, xname = _pick_x(ts)
    names = [n for n in ts.names if n != xname] if ys is None else list(ys)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure

    for n in names:
        y = ts[n].values
        ax.plot(x, y, label=n)

    ax.set_xlabel(xlabel or xname)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.3)
    if tight:
        fig.tight_layout()
    return fig


def plot_bode(
    ts: TraceSet,
    y: str,
    *,
    unwrap_phase: bool = True,
    title_mag: str | None = None,
    title_phase: str | None = None,
) -> tuple[Any, Any]:
    """Plota Bode (magnitude em dB e fase em graus) para uma *trace* complexa ``y``.

    Retorna ``(fig_mag, fig_phase)``.
    """
    plt = _ensure_pyplot()

    x, xname = _pick_x(ts)
    z = np.asarray(ts[y].values)
    if not np.iscomplexobj(z):
        raise ValueError(f"Trace '{y}' is not complex; AC/Bode requires complex values.")

    mag_db = 20.0 * np.log10(np.abs(z))
    phase = np.angle(z, deg=True)
    if unwrap_phase:
        phase = np.unwrap(np.deg2rad(phase))
        phase = np.rad2deg(phase)

    # Magnitude
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.plot(x, mag_db, label=f"|{y}| [dB]")
    ax1.set_xlabel(xname)
    ax1.set_ylabel("Magnitude [dB]")
    if title_mag:
        ax1.set_title(title_mag)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    # Fase
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(x, phase, label=f"∠{y} [deg]")
    ax2.set_xlabel(xname)
    ax2.set_ylabel("Phase [deg]")
    if title_phase:
        ax2.set_title(title_phase)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()

    return fig1, fig2


def plot_sweep_df(
    df: Any,
    x: str,
    y: str,
    hue: str,
    title: str | None = None,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    grid: bool = True,
    tight: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plota um DataFrame empilhado por parâmetro (ex.: retornado por ``stack_step_to_df``).

    Uma curva por valor distinto de ``hue``.
    """
    plt = _ensure_pyplot()

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure

    for val, g in df.groupby(hue):
        ax.plot(g[x].values, g[y].values, label=f"{hue}={val}")

    ax.set_xlabel(xlabel or x)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.3)
    if tight:
        fig.tight_layout()

    return fig
