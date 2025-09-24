from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

ESeries = Literal["E12", "E24", "E48", "E96"]


def _base(series: ESeries) -> list[float]:
    N = int(series[1:])
    return [10 ** (k / N) for k in range(N)]


def enumerate_values(series: ESeries, decade_min: int = -3, decade_max: int = 6) -> Iterable[float]:
    base = _base(series)
    for d in range(decade_min, decade_max + 1):
        for b in base:
            yield round(b * (10**d), 12)


def round_to_series(x: float, series: ESeries = "E24") -> float:
    best, err = None, math.inf
    for v in enumerate_values(series, -6, 9):
        rel = abs(v - x) / x
        if rel < err:
            best, err = v, rel
    return float(best if best is not None else x)
