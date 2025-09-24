from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

from .core import AnalysisResult
from .step import StepResult


def _ensure_pandas() -> Any:
    try:
        return importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for stacking sweep results") from exc


def _pick_columns(df: Any, names: Sequence[str] | None) -> Any:
    if names is None:
        return df
    missing = [n for n in names if n not in df.columns]
    if missing:
        raise KeyError(f"Missing traces in DataFrame: {missing!r}. Available: {list(df.columns)!r}")
    return df[list(names)]


def stack_runs_to_df(
    runs: Sequence[AnalysisResult],
    params_list: Sequence[Mapping[str, str | float]] | None = None,
    y: Sequence[str] | None = None,
    with_x: bool = True,
    run_index_name: str = "run_idx",
) -> Any:
    """
    Empilha uma lista de `AnalysisResult` em um único DataFrame, adicionando
    as colunas de parâmetros (`params_list[i]`) por run.

    - `y`: lista de nomes de traços a manter (ex.: ["v(out)"]). Se None, mantém todos.
    - `with_x`: inclui a coluna do eixo X (primeira coluna do TraceSet), tipicamente
    "time"/"frequency".
    """
    pd = _ensure_pandas()
    frames: list[Any] = []
    for i, res in enumerate(runs):
        df = res.traces.to_dataframe()
        x_name = res.traces.x.name
        keep = list(df.columns)
        if y is not None:
            keep = [x_name] + list(y) if with_x else list(y)
            df = _pick_columns(df, keep)
        else:
            if not with_x:
                keep = [c for c in keep if c != x_name]
                df = df[keep]
        # parâmetros deste run
        params = params_list[i] if params_list is not None else {}
        for k, v in params.items():
            df[k] = v
        df[run_index_name] = i
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def stack_step_to_df(
    step: StepResult,
    y: Sequence[str] | None = None,
    with_x: bool = True,
    run_index_name: str = "run_idx",
) -> Any:
    """
    Versão prática para `StepResult`: empilha `step.runs` com as colunas de `step.grid`.
    """
    return stack_runs_to_df(step.runs, step.grid, y=y, with_x=with_x, run_index_name=run_index_name)
