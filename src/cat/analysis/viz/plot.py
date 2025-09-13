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


def plot_mc_metric_hist(
    metrics: list[float] | None = None,
    title: str | None = None,
    bins: int | None = 50,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    tight: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plota um histograma para uma lista de métricas de Monte Carlo.

    - `metrics`: lista de valores (se None, raise).
    - retorna a figure.
    """
    plt = _ensure_pyplot()
    if metrics is None:
        raise ValueError("metrics list is required")

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure

    ax.hist(metrics, bins=bins, alpha=0.8)
    ax.set_xlabel(xlabel or "metric")
    ax.set_ylabel(ylabel or "count")
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if tight:
        fig.tight_layout()
    return fig


def plot_param_vs_metric(
    samples: list[dict[str, float]],
    metrics: list[float],
    param: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    tight: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plota um scatter de um parâmetro Monte Carlo vs uma métrica.

    - `samples`: lista de dicts (as returned by MonteCarloResult.samples)
    - `metrics`: lista de scalar metric values, same length as samples
    - `param`: the key in each sample to use as x
    """
    plt = _ensure_pyplot()

    xs = [s.get(param, 0.0) for s in samples]
    ys = list(metrics)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure

    ax.scatter(xs, ys, alpha=0.8)
    ax.set_xlabel(xlabel or param)
    ax.set_ylabel(ylabel or "metric")
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if tight:
        fig.tight_layout()
    return fig


def plot_mc_kde(
    metrics: list[float],
    *,
    title: str | None = None,
    bandwidth: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    tight: bool = True,
    ax: Any | None = None,
) -> Any:
    """Plot a KDE-smoothed distribution for a metric list.

    Will try to use `scipy.stats.gaussian_kde` if available, otherwise falls
    back to a histogram.
    """
    plt = _ensure_pyplot()

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    else:
        fig = ax.figure

    xs = list(metrics)
    try:
        from scipy.stats import gaussian_kde  # type: ignore

        kde = gaussian_kde(xs, bw_method=bandwidth)
        import numpy as _np

        x_min, x_max = min(xs), max(xs)
        x_grid = _np.linspace(x_min, x_max, 256)
        ax.plot(x_grid, kde(x_grid))
    except Exception:
        # fallback: histogram
        ax.hist(xs, bins=50, alpha=0.8)

    ax.set_xlabel(xlabel or "metric")
    ax.set_ylabel(ylabel or "density")
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3)
    if tight:
        fig.tight_layout()
    return fig


def plot_params_matrix(
    samples: list[dict[str, float]],
    params: list[str] | None = None,
    *,
    figsize: tuple[int, int] | None = None,
    tight: bool = True,
) -> Any:
    """Create a pairwise scatter matrix for sampled parameters.

    - `samples`: list of dicts as produced by MonteCarloResult.samples
    - `params`: optional list of parameter names to include (default: all keys from first sample)
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        # Fallback: build small matrix of subplots with matplotlib
        plt = _ensure_pyplot()
        if not samples:
            raise ValueError("no samples provided") from None
        keys = params or list(samples[0].keys())
        n = len(keys)
        fig = plt.figure(figsize=figsize or (3 * n, 3 * n))
        axes = [[fig.add_subplot(n, n, i * n + j + 1) for j in range(n)] for i in range(n)]
        xs = [[s.get(k, 0.0) for s in samples] for k in keys]
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    ax.hist(xs[i], bins=30, alpha=0.8)
                else:
                    ax.scatter(xs[j], xs[i], s=8, alpha=0.6)
                if i == n - 1:
                    ax.set_xlabel(keys[j])
                if j == 0:
                    ax.set_ylabel(keys[i])
        if tight:
            fig.tight_layout()
        return fig

    # pandas available: use DataFrame.plot.scatter_matrix via pandas.plotting
    df = pd.DataFrame(samples)
    if params is not None:
        df = df.loc[:, params]
    try:
        from pandas.plotting import scatter_matrix  # type: ignore

        fig = None
        if figsize is not None:
            plt = _ensure_pyplot()
            plt.figure(figsize=figsize)
        axes = scatter_matrix(df, alpha=0.6, diagonal="hist")
        # scatter_matrix returns a 2D array-like of Axes; index as [0][0]
        fig = axes[0][0].figure
        if tight:
            fig.tight_layout()
        return fig
    except Exception:
        # fallback to manual matrix
        return plot_params_matrix(samples, params=params, figsize=figsize, tight=tight)
