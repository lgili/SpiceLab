from __future__ import annotations

import numpy as np

from ..io.raw_reader import TraceSet


def _unwrap_deg(phase_deg: np.ndarray) -> np.ndarray:
    # unwrap opera em rad; converte ida/volta
    return np.unwrap(np.deg2rad(phase_deg)) * (180.0 / np.pi)


def _first_crossing_x(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    """
    Retorna a primeira abscissa x onde y cruza 'target' (por interpolação linear).
    Se não cruzar, retorna None.
    """
    d = y - target
    s = np.sign(d)
    idx = np.where(s[:-1] * s[1:] <= 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, x1 = float(x[i]), float(x[i + 1])
    y0, y1 = float(y[i]), float(y[i + 1])
    if y1 == y0:
        return x0
    t = (target - y0) / (y1 - y0)
    return x0 + t * (x1 - x0)


def ac_gain_phase(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
    eps: float = 1e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (f, mag_db, phase_deg) do ganho de 'y_out' relativo a 'y_in'.
    - Se y_in for None, assume 1 V de referência.
    - Exige traço complexo para fase; se não houver, fase retorna zeros.
    """
    f = ts.x.values
    out_t = ts[y_out]
    if y_in is None:
        mag = out_t.magnitude()
        ph = out_t.phase_deg()
    else:
        in_t = ts[y_in]
        if out_t._complex is not None and in_t._complex is not None:
            ratio_c = out_t._complex / (in_t._complex + eps)
            mag = np.abs(ratio_c)
            ph = np.angle(ratio_c, deg=True)
        else:
            mag = out_t.values / (in_t.values + eps)
            ph = np.zeros_like(mag, dtype=float)
    mag_db = 20.0 * np.log10(np.maximum(mag, eps))
    return f, mag_db, ph


def crossover_freq_0db(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """Primeira frequência onde |G| cruza 0 dB."""
    f, mag_db, _ = ac_gain_phase(ts, y_out, y_in)
    return _first_crossing_x(f, mag_db, 0.0)


def phase_crossover_freq(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """Primeira frequência onde fase cruza −180° (com unwrap)."""
    f, _, ph = ac_gain_phase(ts, y_out, y_in)
    ph_u = _unwrap_deg(ph)
    return _first_crossing_x(f, ph_u, -180.0)


def phase_margin(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """
    Margem de fase (graus): 180° + fase(wc), onde wc é a frequência de cruzamento 0 dB.
    Retorna None se não houver cruzamento.
    """
    wc = crossover_freq_0db(ts, y_out, y_in)
    if wc is None:
        return None
    f, _, ph = ac_gain_phase(ts, y_out, y_in)
    ph_u = _unwrap_deg(ph)
    return _interp_at_x(f, ph_u, wc) + 180.0


def gain_margin_db(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """
    Margem de ganho (dB): −|G(jw_180)| (valor positivo indica margem).
    w_180 é a frequência onde fase cruza −180°. Se não cruzar, retorna None.
    """
    w180 = phase_crossover_freq(ts, y_out, y_in)
    if w180 is None:
        return None
    f, mag_db, _ = ac_gain_phase(ts, y_out, y_in)
    g_at = _interp_at_x(f, mag_db, w180)
    return -g_at


def _interp_at_x(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Interpolação linear y(xq) assumindo x crescente."""
    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[-1]):
        return float(y[-1])
    i = int(np.searchsorted(x, xq) - 1)
    x0, x1 = float(x[i]), float(x[i + 1])
    y0, y1 = float(y[i]), float(y[i + 1])
    if x1 == x0:
        return y0
    t = (xq - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
