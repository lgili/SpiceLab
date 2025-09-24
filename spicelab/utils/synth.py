from __future__ import annotations

import math
from dataclasses import dataclass

from .e_series import ESeries, round_to_series


@dataclass(frozen=True)
class RCDesign:
    R: float
    C: float
    fc: float
    tau: float


def design_rc_lowpass(fc: float, prefer_R: bool = True, series: ESeries = "E24") -> RCDesign:
    tau = 1.0 / (2 * math.pi * fc)
    if prefer_R:
        R = round_to_series(10_000.0, series)
        C = round_to_series(tau / R, series)
    else:
        C = round_to_series(100e-9, series)
        R = round_to_series(tau / C, series)
    return RCDesign(R=R, C=C, fc=1.0 / (2 * math.pi * R * C), tau=R * C)
