from __future__ import annotations

import numpy as np

from ..io.raw_reader import TraceSet


def peak(ts: TraceSet, y: str) -> tuple[float, float]:
    """Retorna (t_peak, y_peak) do traço `y`."""
    x = ts.x.values
    v = ts[y].values
    idx = int(np.argmax(v))
    return float(x[idx]), float(v[idx])


def _final_value(v: np.ndarray, tail_frac: float = 0.1) -> float:
    n = len(v)
    k = max(1, int(n * tail_frac))
    return float(np.mean(v[-k:]))


def settling_time(ts: TraceSet, y: str, tol: float = 0.02) -> float:
    """
    Tempo de estabelecimento: primeiro instante t após o qual o sinal fica
    dentro de ±tol da média final e não sai mais até o final.
    """
    x = ts.x.values
    v = ts[y].values
    vf = _final_value(v)
    lo, hi = vf * (1.0 - tol), vf * (1.0 + tol)

    # encontra primeiro índice a partir do qual todos os pontos estão no intervalo
    for i, val in enumerate(v):
        if lo <= val <= hi and np.all((v[i:] >= lo) & (v[i:] <= hi)):
            return float(x[i])
    return float(x[-1])


def overshoot_pct(ts: TraceSet, y: str) -> float:
    """Overshoot(%) = (peak - final) / final * 100 (se final > 0), senão 0."""
    v = ts[y].values
    vf = _final_value(v)
    if vf <= 0:
        return 0.0
    vp = float(np.max(v))
    return (vp - vf) / vf * 100.0


def gain_db_from_traces(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (x, gain_db). Útil para AC:
      - se `y_in` for None, assume 1V de referência.
      - caso contrário, usa ganho = |y_out| / |y_in|.
    """
    x = ts.x.values
    vout = ts[y_out].values
    if y_in is None:
        ratio = vout
    else:
        vin = ts[y_in].values
        ratio = vout / (vin + eps)
    gain_db = 20.0 * np.log10(np.maximum(np.abs(ratio), eps))
    return x.copy(), gain_db


def bandwidth_3db(
    ts: TraceSet,
    y_mag: str,
    ref_value: float | None = None,
) -> float | None:
    """
    Frequência de -3 dB relativa ao nível de referência.
    - Se `ref_value` for None, usa o valor em `y_mag` no primeiro ponto.
    - Supõe eixo X = frequência crescente.
    """
    f = ts.x.values
    mag = ts[y_mag].values
    ref = float(mag[0]) if ref_value is None else float(ref_value)
    target = ref / (10 ** (3.0 / 20.0))  # -3 dB
    # primeiro ponto onde mag cai abaixo de target
    below = np.where(mag <= target)[0]
    if len(below) == 0:
        return None
    i = int(below[0])
    return float(f[i])
