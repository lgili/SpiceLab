from __future__ import annotations

import numpy as np

from ..io.raw_reader import TraceSet
from .metrics_tran import overshoot as _ov


def _vals(ts: TraceSet, name: str) -> np.ndarray:
    tr = ts[name]
    if tr is None:
        raise ValueError(f"Trace {name!r} not found")
    if hasattr(tr, "values"):
        return np.asarray(tr.values, dtype=float)
    return np.asarray(tr, dtype=float)


def gain_db_from_traces(ts: TraceSet, y_name: str) -> float:
    """
    "Ganho" DC aproximado, usando a diferença final - inicial do traço (em dB do delta).
    É um helper simples para testes; não é igual a ganho AC.
    """
    arr = _vals(ts, y_name)
    if arr.size == 0:
        return -300.0
    v = float(arr[-1] - arr[0])
    if v == 0.0:
        return -300.0
    return float(20.0 * np.log10(abs(v)))


def overshoot_pct(ts: TraceSet, y_name: str) -> float:
    """Overshoot em % relativo ao degrau."""
    return float(_ov(ts, y_name).overshoot * 100.0)


def peak(ts: TraceSet, y_name: str) -> float:
    return float(np.max(_vals(ts, y_name)))
