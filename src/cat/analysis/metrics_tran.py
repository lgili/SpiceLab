from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.raw_reader import TraceSet


def _vals(ts: TraceSet, name: str) -> np.ndarray:
    """Obtém os valores de um traço como numpy array."""
    tr = ts[name]
    if tr is None:
        raise KeyError(f"Trace {name!r} not found")
    if hasattr(tr, "values"):
        return np.asarray(getattr(tr, "values"))
    return np.asarray(tr)


def _pick_voltage_trace_name(ts: TraceSet, preferred: str) -> str:
    """
    Tenta usar `preferred`; se não existir, escolhe um traço de tensão disponível.
    Preferência: v(n2) -> v(n1) -> primeiro nome que começar com 'v('.
    """
    names = getattr(ts, "names", [])
    if preferred in names:
        return preferred
    if "v(n2)" in names:
        return "v(n2)"
    if "v(n1)" in names:
        return "v(n1)"
    for nm in names:
        if isinstance(nm, str) and nm.startswith("v("):
            return nm
    return preferred  # deixa estourar erro claro depois, se não existir


def _get_xy(ts: TraceSet, y_name: str, x_name: str = "time") -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (x, y) como ndarrays float para análise de transiente.
    Faz fallback do nome do traço de tensão, se necessário.
    """
    x = _vals(ts, x_name).astype(float)
    real_name = _pick_voltage_trace_name(ts, y_name) if y_name.lower().startswith("v(") else y_name
    y_raw = _vals(ts, real_name)
    if np.iscomplexobj(y_raw):
        y = np.real(y_raw).astype(float)
    else:
        y = np.asarray(y_raw, dtype=float)
    if y.ndim == 0:
        y = np.atleast_1d(y)
    return x, y


def peak(ts: TraceSet, y_name: str) -> float:
    """Valor máximo de y(t)."""
    _, y = _get_xy(ts, y_name)
    return float(np.max(y))


@dataclass(frozen=True)
class OvershootResult:
    overshoot: float  # relativo (0.1 = 10%)
    y_final: float
    y_initial: float
    y_peak: float


def overshoot(ts: TraceSet, y_name: str) -> OvershootResult:
    """
    Overshoot relativo ao degrau: (Vmax - Vfinal) / (Vfinal - Vinicial).
    """
    x, y = _get_xy(ts, y_name)
    _ = x  # reservado para futuros usos
    n = y.shape[0]
    if n < 2:
        return OvershootResult(
            overshoot=0.0,
            y_final=float(y[-1]),
            y_initial=float(y[0]),
            y_peak=float(y[0]),
        )
    y0 = float(y[0])
    yf = float(y[-1])
    yp = float(np.max(y))
    denom = yf - y0
    if np.isclose(denom, 0.0):
        return OvershootResult(overshoot=0.0, y_final=yf, y_initial=y0, y_peak=yp)
    ov = max(0.0, (yp - yf) / denom)
    return OvershootResult(overshoot=ov, y_final=yf, y_initial=y0, y_peak=yp)


def overshoot_pct(ts: TraceSet, y_name: str) -> float:
    """Overshoot em % do degrau."""
    return float(overshoot(ts, y_name).overshoot * 100.0)


@dataclass(frozen=True)
class SettlingResult:
    t_settle: float | None  # instante em que entra e permanece dentro da banda
    idx: int | None  # índice do primeiro ponto que satisfaz até o fim
    band: float  # tolerância absoluta usada
    y_final: float  # valor final (última amostra)


def settling_time(ts: TraceSet, y_name: str, tol: float = 0.02) -> SettlingResult:
    """
    Primeiro instante em que |y(t) - y_final| <= tol*|y_final - y_initial|
    e permanece assim até o final.
    """
    x, y = _get_xy(ts, y_name)
    y0 = float(y[0])
    yf = float(y[-1])
    band = abs(tol * (yf - y0))
    if band == 0.0:
        return SettlingResult(t_settle=None, idx=None, band=band, y_final=yf)

    err = np.abs(y - yf)
    inside = err <= band
    suffix_ok = np.flip(np.cumsum(np.flip(~inside)) == 0)
    ok = inside & suffix_ok
    idxs = np.where(ok)[0]
    if idxs.size == 0:
        return SettlingResult(t_settle=None, idx=None, band=band, y_final=yf)
    i = int(idxs[0])
    return SettlingResult(t_settle=float(x[i]), idx=i, band=band, y_final=yf)


# ============
# Rise / Fall
# ============


@dataclass(frozen=True)
class RiseFall:
    """Tempos de subida/descida entre frações de nível (ex.: 10%→90%)."""

    trise: float | None
    tfall: float | None


def _cross_time(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """Cruzamento linear (primeiro) de y(t) = level."""
    for i in range(1, len(x)):
        y0, y1 = y[i - 1], y[i]
        if (y0 < level <= y1) or (y0 > level >= y1):
            x0, x1 = x[i - 1], x[i]
            if np.isclose(y1, y0):
                return float(x0)
            w = (level - y0) / (y1 - y0)
            return float(x0 + w * (x1 - x0))
    return None


def _interp_time_rise(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """
    Interpolação robusta para subida: usa envelope não-decrescente (cumulative max).
    Retorna None se nível estiver fora do alcance.
    """
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if not (y_min <= level <= y_max):
        return None
    y_env = np.maximum.accumulate(y)  # não-decrescente
    return float(np.interp(level, y_env, x))


def _interp_time_fall(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """
    Interpolação robusta para descida: usa envelope não-crescente (cumulative min).
    Retorna None se nível estiver fora do alcance.
    """
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if not (y_min <= level <= y_max):
        return None
    y_env = np.minimum.accumulate(y[::-1])[::-1]  # não-crescente ao longo de x
    return float(np.interp(level, y_env, x))


def _discrete_time_first_at_or_above(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """Fallback discreto: primeiro x[i] com y[i] >= level (ou None)."""
    idxs = np.where(y >= level)[0]
    if idxs.size == 0:
        return None
    return float(x[int(idxs[0])])


def rise_time(
    ts: TraceSet,
    y_name: str,
    frac_low: float = 0.1,
    frac_high: float = 0.9,
) -> RiseFall:
    """
    Tempo 10–90% (por padrão). Retorna RiseFall com .trise (e .tfall=None).

    Estratégia robusta:
      1) Prioriza níveis com base em y0→yf (degrau observado). Se não subir, usa span global.
      2) Tenta cruzamento linear; se falhar, interpola no envelope; se falhar, usa fallback
         discreto (primeiro índice com y >= nível).
    """
    x, y = _get_xy(ts, y_name)
    y0 = float(y[0])
    yf = float(y[-1])
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span_end = yf - y0
    span_global = y_max - y_min

    if span_end > 1e-15:
        lo = y0 + frac_low * span_end
        hi = y0 + frac_high * span_end
    elif span_global > 1e-15:
        lo = y_min + frac_low * span_global
        hi = y_min + frac_high * span_global
    else:
        return RiseFall(trise=None, tfall=None)

    # 1) cruzamento linear
    t_lo = _cross_time(x, y, lo)
    t_hi = _cross_time(x, y, hi)

    # 2) envelope (não-decrescente)
    if t_lo is None:
        t_lo = _interp_time_rise(x, y, lo)
    if t_hi is None:
        t_hi = _interp_time_rise(x, y, hi)

    # 3) discreto (primeiro índice com y >= nível)
    if t_lo is None:
        t_lo = _discrete_time_first_at_or_above(x, y, lo)
    if t_hi is None:
        t_hi = _discrete_time_first_at_or_above(x, y, hi)

    if t_lo is None or t_hi is None:
        return RiseFall(trise=None, tfall=None)

    dt = float(t_hi - t_lo)
    if dt < 0.0:
        dt = 0.0
    return RiseFall(trise=dt, tfall=None)


def fall_time(
    ts: TraceSet,
    y_name: str,
    frac_low: float = 0.1,
    frac_high: float = 0.9,
) -> RiseFall:
    """
    Tempo 90–10% (por padrão). Retorna RiseFall com .tfall (e .trise=None).

    Estratégia robusta análoga à de subida, mas do nível alto para o baixo.
    """
    x, y = _get_xy(ts, y_name)
    y0 = float(y[0])
    yf = float(y[-1])
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span_end = y0 - yf
    span_global = y_max - y_min

    if span_end > 1e-15:
        hi = yf + (1.0 - frac_high) * span_end  # próximo de y0
        lo = yf + (1.0 - frac_low) * span_end
    elif span_global > 1e-15:
        hi = y_max - frac_high * span_global
        lo = y_max - frac_low * span_global
    else:
        return RiseFall(trise=None, tfall=None)

    t_hi = _cross_time(x, y, hi)
    t_lo = _cross_time(x, y, lo)

    if t_hi is None:
        t_hi = _interp_time_fall(x, y, hi)
    if t_lo is None:
        t_lo = _interp_time_fall(x, y, lo)

    if t_hi is None:
        t_hi = _discrete_time_first_at_or_above(x, y[::-1], hi)
    if t_lo is None:
        t_lo = _discrete_time_first_at_or_above(x, y[::-1], lo)

    if t_hi is None or t_lo is None:
        return RiseFall(trise=None, tfall=None)

    dt = float(t_lo - t_hi)
    if dt < 0.0:
        dt = 0.0
    return RiseFall(trise=None, tfall=dt)
