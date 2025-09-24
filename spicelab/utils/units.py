from __future__ import annotations

import re
from typing import Final

_SUFFIXES: Final[dict[str, float]] = {
    # high
    "t": 1e12,
    "g": 1e9,
    "meg": 1e6,  # comum em spice
    "m": 1e-3,  # cuidado: 'm' = milli
    "k": 1e3,
    "": 1.0,
    # low
    "u": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

_NUM_RE = re.compile(
    r"""
    ^\s*
    (?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
    (?P<suf>[a-zA-Zµ]{0,3})?
    \s*$
    """,
    re.VERBOSE,
)


def to_float(val: str | float) -> float:
    """Converte '1k' -> 1000.0, '100n' -> 1e-7, também aceita float."""
    if isinstance(val, float):
        return val
    s = val.strip()
    m = _NUM_RE.match(s)
    if not m:
        raise ValueError(f"Invalid numeric with suffix: {val!r}")
    num = float(m.group("num"))
    suf = (m.group("suf") or "").lower()
    if suf not in _SUFFIXES:
        # permite 'meg' e 'k', mas não 'K' separado
        if suf == "megs":
            suf = "meg"
        else:
            raise ValueError(f"Unknown suffix: {suf!r} in {val!r}")
    return num * _SUFFIXES[suf]


def format_eng(x: float) -> str:
    """Formata em engenharia usando sufixos (k, M/meg, m, u, n, p)."""
    if x == 0.0:
        return "0"
    import math

    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    exp = max(min(exp, 12), -15)
    # preferir "meg" para 1e6
    table = {
        12: "t",
        9: "g",
        6: "meg",
        3: "k",
        0: "",
        -3: "m",
        -6: "u",
        -9: "n",
        -12: "p",
        -15: "f",
    }
    suf = table.get(exp, "")
    scaled = x / (10**exp)
    # remover zeros desnecessários
    s = f"{scaled:.6g}"
    return f"{s}{suf}"
