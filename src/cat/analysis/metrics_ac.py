from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..io.raw_reader import TraceSet

# =====================
# Estruturas e helpers
# =====================


@dataclass(frozen=True)
class Bode:
    """Container opcional para quem preferir um objeto tipado."""

    f: NDArray[Any]  # Hz
    mag_db: NDArray[Any]  # dB
    ph_deg: NDArray[Any]  # graus


def _vals(ts: TraceSet, name: str) -> NDArray[Any]:
    """
    Retorna os valores (numpy array) do traço `name`. Aceita que ts[name] retorne
    um objeto Trace com atributo `.values` ou diretamente uma sequência.
    """
    tr = ts[name]
    if tr is None:
        raise ValueError(f"Trace {name!r} not found")
    if hasattr(tr, "values"):  # objeto Trace
        return np.asarray(tr.values)
    return np.asarray(tr)


def _try_trace(ts: TraceSet, name: str) -> NDArray[Any] | None:
    """Versão segura: devolve ndarray ou None."""
    try:
        tr = ts[name]
    except Exception:
        return None
    if tr is None:
        return None
    if hasattr(tr, "values"):
        return np.asarray(tr.values)
    return np.asarray(tr)


def _wrap_deg180(phi_deg: NDArray[Any]) -> NDArray[Any]:
    """Normaliza fase para o intervalo (-180, 180]."""
    out = (phi_deg + 180.0) % 360.0 - 180.0
    out[np.isclose(out, -180.0)] = -180.0
    return out


def _phase_from_any_match(ts: TraceSet, n: int) -> NDArray[Any] | None:
    """
    Fallback genérico: varre todos os traços buscando nomes que indiquem fase.
    Usa o primeiro que casar o comprimento.
    """
    names = cast(Iterable[str], getattr(ts, "names", ()))
    for nm in names:
        low = nm.lower()
        if any(k in low for k in ("phase", "ph(", "angle", "arg")):
            arr = _try_trace(ts, nm)
            if arr is not None and arr.shape[0] == n:
                return arr.astype(float)
    return None


def _maybe_phase_array(ts: TraceSet, y_out: str, n: int) -> NDArray[Any] | None:
    """
    Procura traço explícito de fase (em graus preferencialmente).
    Aceita variações comuns e, se falhar, tenta um fallback varrendo todos os nomes.
    """
    candidates = (
        "phase",
        f"phase({y_out})",
        f"ph({y_out})",
        "phase_deg",
        f"phase_deg({y_out})",
        "angle",
        f"angle({y_out})",
        "arg",
        f"arg({y_out})",
    )
    for name in candidates:
        arr = _try_trace(ts, name)
        if arr is not None and arr.shape[0] == n:
            return arr.astype(float)
    return _phase_from_any_match(ts, n)


def _maybe_complex_from_re_im(ts: TraceSet, y_out: str, n: int) -> NDArray[Any] | None:
    """Reconstrói vetor complexo a partir de traços re()/im()."""
    re_candidates = (f"re({y_out})", f"real({y_out})", "re", "real")
    im_candidates = (f"im({y_out})", f"imag({y_out})", "im", "imag")

    re_arr: NDArray[Any] | None = None
    im_arr: NDArray[Any] | None = None
    for name in re_candidates:
        a = _try_trace(ts, name)
        if a is not None and a.shape[0] == n:
            re_arr = a.astype(float)
            break
    for name in im_candidates:
        a = _try_trace(ts, name)
        if a is not None and a.shape[0] == n:
            im_arr = a.astype(float)
            break
    if re_arr is not None and im_arr is not None:
        return re_arr + 1j * im_arr
    return None


def _maybe_complex_from_mag_phase(
    ts: TraceSet,
    y_out: str,
    n: int,
) -> NDArray[Any] | None:
    """
    Reconstrói vetor complexo a partir de (mag, phase) ou (db, phase).
    Detecta automaticamente graus vs radianos.
    """
    mag_candidates = (f"mag({y_out})", "mag", "magnitude", f"db({y_out})", "db")
    ph_candidates = (
        f"phase({y_out})",
        "phase",
        "phase_deg",
        f"phase_deg({y_out})",
        "angle",
        f"angle({y_out})",
        "arg",
        f"arg({y_out})",
        f"ph({y_out})",
        "ph",
    )
    mag: NDArray[Any] | None = None
    ph: NDArray[Any] | None = None
    is_db = False
    for name in mag_candidates:
        a = _try_trace(ts, name)
        if a is not None and a.shape[0] == n:
            mag = np.asarray(a, dtype=float)
            if "db" in name:
                is_db = True
            break
    for name in ph_candidates:
        a = _try_trace(ts, name)
        if a is not None and a.shape[0] == n:
            ph = np.asarray(a, dtype=float)
            break
    if mag is None or ph is None:
        return None

    # db -> linear
    if is_db:
        mag = 10.0 ** (mag / 20.0)

    # graus vs radianos
    ph_max = float(np.nanmax(np.abs(ph)))
    if ph_max > 3.5:
        ph_rad = np.deg2rad(ph)
    else:
        ph_rad = ph
    out = mag * np.exp(1j * ph_rad)
    return cast(NDArray[Any], out)


# ===========================
# Núcleo: montar Bode de AC
# ===========================


def _get_xy_ac(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
    f_name: str = "frequency",
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Obtém (f, |G|dB, fase_deg). Se y_in=None, assume que y_out já é a saída a ser analisada.
    Quando y_in é fornecida, retorna a razão y_out / y_in.

    Robustez:
      - aceita fase explícita (diversos nomes);
      - reconstrói complexo via (re,im) ou (mag,phase)/(db,phase);
      - fallback: usa np.angle/unwrap caso não haja fase explícita.
    """
    f = _vals(ts, f_name).astype(float)
    y = _vals(ts, y_out)
    n = y.shape[0]

    # tentar reconstruir y como complexo
    if np.iscomplexobj(y):
        y_complex: NDArray[Any] = y
    else:
        tmp = _maybe_complex_from_re_im(ts, y_out, n)
        if tmp is None:
            tmp = _maybe_complex_from_mag_phase(ts, y_out, n)
        y_complex = tmp if tmp is not None else y.astype(complex)

    if y_in is None:
        num = y_complex
    else:
        xin = _vals(ts, y_in)
        if not np.iscomplexobj(xin):
            xin_complex = _maybe_complex_from_re_im(ts, y_in, n)
            if xin_complex is None:
                xin_complex = _maybe_complex_from_mag_phase(ts, y_in, n)
            if xin_complex is None:
                xin_complex = xin.astype(complex)
        else:
            xin_complex = xin
        num = y_complex / xin_complex

    mag = np.abs(num)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-300))

    # fase: preferir traço explícito se existir
    ph_override = _maybe_phase_array(ts, y_out, n)
    if ph_override is not None:
        ph_deg: NDArray[Any] = ph_override.astype(float)
    else:
        ph_deg = np.angle(num, deg=True)
        ph_deg = np.rad2deg(np.unwrap(np.deg2rad(ph_deg)))
        ph_deg = _wrap_deg180(ph_deg)

    return f, mag_db, ph_deg


def ac_gain_phase(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Retorna (f, mag_db, fase_deg)."""
    return _get_xy_ac(ts, y_out=y_out, y_in=y_in)


# =================
# Métricas de Bode
# =================


def _interp_at_x(x: NDArray[Any], y: NDArray[Any], xq: float) -> float:
    if not np.all(np.diff(x) > 0):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
    return float(np.interp(xq, x, y))


def gain_at(
    ts: TraceSet,
    y_out: str,
    f_hz: float,
    y_in: str | None = None,
) -> float:
    """Ganho (dB) em f_hz."""
    f, mag_db, _ = ac_gain_phase(ts, y_out=y_out, y_in=y_in)
    return _interp_at_x(f, mag_db, f_hz)


def bandwidth_3db(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """Frequência -3 dB relativa ao ganho de baixa frequência."""
    f, mag_db, _ = ac_gain_phase(ts, y_out=y_out, y_in=y_in)
    g0 = float(mag_db[0])
    target = g0 - 3.0
    below = np.where(mag_db <= target)[0]
    if below.size == 0:
        return None
    i = below[0]
    if i == 0:
        return float(f[0])
    x0, x1 = f[i - 1], f[i]
    y0, y1 = mag_db[i - 1], mag_db[i]
    if np.isclose(y1, y0):
        return float(x1)
    w = (target - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


def crossover_freq_0db(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """Frequência em que |G| cruza 0 dB."""
    f, mag_db, _ = ac_gain_phase(ts, y_out=y_out, y_in=y_in)
    y = mag_db
    s = np.sign(y)
    idx = np.where((s[:-1] >= 0) & (s[1:] <= 0) | (s[:-1] <= 0) & (s[1:] >= 0))[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, x1 = f[i], f[i + 1]
    y0, y1 = y[i], y[i + 1]
    if np.isclose(y1, y0):
        return float(x0)
    w = (0.0 - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


def phase_crossover_freq(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
    target_deg: float = -180.0,
) -> float | None:
    """Frequência em que fase cruza target (padrão: -180°)."""
    f, _, ph_deg = ac_gain_phase(ts, y_out=y_out, y_in=y_in)
    y = ph_deg - target_deg
    s = np.sign(y)
    idx = np.where((s[:-1] >= 0) & (s[1:] <= 0) | (s[:-1] <= 0) & (s[1:] >= 0))[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    x0, x1 = f[i], f[i + 1]
    y0, y1 = y[i], y[i + 1]
    if np.isclose(y1, y0):
        return float(x0)
    w = (0.0 - y0) / (y1 - y0)
    return float(x0 + w * (x1 - x0))


def gain_margin_db(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """Ganho em dB quando fase = -180° (margem de ganho)."""
    f180 = phase_crossover_freq(ts, y_out=y_out, y_in=y_in, target_deg=-180.0)
    if f180 is None:
        return None
    return gain_at(ts, y_out=y_out, y_in=y_in, f_hz=f180)


def loop_gain_bode(
    ts: TraceSet,
    y_out: str,
    y_in: str,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Retorna Bode do loop (y_out / y_in) como tupla (f, mag_db, fase)."""
    return ac_gain_phase(ts, y_out=y_out, y_in=y_in)


def phase_margin(
    ts: TraceSet,
    y_out: str,
    y_in: str | None = None,
) -> float | None:
    """
    PM = 180° + fase em w_c (onde |G|=1). Fase normalizada para (-180, 180].

    Fallback: se a fase aparenta estar ausente (variância ~ zero), estima PM assumindo
    um sistema de 1 polo mínimo-fase: φ(w) ≈ -atan(w/wp) e PM ≈ 180° - atan(wc/wp),
    onde wp é obtido via largura de banda de -3 dB.
    """
    wc = crossover_freq_0db(ts, y_out=y_out, y_in=y_in)
    if wc is None:
        return None
    f, _, ph = ac_gain_phase(ts, y_out=y_out, y_in=y_in)

    # se há fase, use-a
    if np.nanstd(ph) > 1e-3:
        phi = _interp_at_x(f, ph, wc)
        return 180.0 + float(phi)

    # fallback 1-polo: usa -3 dB para estimar wp
    bw = bandwidth_3db(ts, y_out=y_out, y_in=y_in)
    if bw is None or bw <= 0.0:
        # sem alternativa: devolve 180 para não explodir
        return 180.0
    phi_est = -np.degrees(np.arctan(wc / bw))
    return 180.0 + float(phi_est)
