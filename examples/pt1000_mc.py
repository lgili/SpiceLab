from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from cat.analysis import OP, NormalPct, monte_carlo
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND, Net

# --- RTD model (PT1000, IEC 60751 alpha ~ 0.00385) ---


def pt1000_r(t_c: float, r0: float = 1000.0, alpha: float = 0.00385) -> float:
    return r0 * (1.0 + alpha * (t_c - 0.0))


@dataclass(frozen=True)
class PT1000Params:
    vdd: float = 3.3
    r_pu_nom: float = 3900.0  # pull-up from VDD to sense
    r_gain_top_nom: float = 4300.0  # Rf (top) for non-inverting gain
    r_gain_bot_nom: float = 2000.0  # Rg (bot) for non-inverting gain
    r0: float = 1000.0
    alpha: float = 0.00385


def build_pt1000_chain(
    r_rtd: float, p: PT1000Params
) -> tuple[Circuit, Resistor, Resistor, Resistor]:
    """Builds a simplified PT1000 front-end:

    - VDD -- Rpu -- Vsense -- Rrtd -- GND
    - Non-inverting amplifier: Vout = (1 + Rtop/Rbot) * Vsense

    Returns (circuit, Rpu, Rtop, Rbot). The Rrtd is created but not varied by MC here.
    """
    c = Circuit("pt1000_chain")
    vin = Net("vin")  # sense node
    vout = Net("vout")

    VDD = Vdc("dd", p.vdd)
    c.add(VDD)
    c.connect(VDD.ports[0], Net("vdd"))
    c.connect(VDD.ports[1], GND)

    # Pull-up and RTD
    Rpu = Resistor("pu", p.r_pu_nom)
    Rrtd = Resistor("rtd", r_rtd)
    c.add(Rpu, Rrtd)
    c.connect(Rpu.ports[0], Net("vdd"))
    c.connect(Rpu.ports[1], vin)
    c.connect(Rrtd.ports[0], vin)
    c.connect(Rrtd.ports[1], GND)

    # Non-inverting amplifier using ideal OA and two resistors
    from cat.core.components import OpAmpIdeal

    oa = OpAmpIdeal("1", gain=1e6)
    Rtop = Resistor("t", p.r_gain_top_nom)
    Rbot = Resistor("b", p.r_gain_bot_nom)
    c.add(oa, Rtop, Rbot)
    # + input at sense
    c.connect(oa.ports[0], vin)
    # output node
    c.connect(oa.ports[2], vout)
    # feedback: Vout -> Rtop -> OA- ; OA- -> Rbot -> GND
    c.connect(Rtop.ports[0], vout)
    c.connect(Rtop.ports[1], oa.ports[1])
    c.connect(Rbot.ports[0], oa.ports[1])
    c.connect(Rbot.ports[1], GND)

    return c, Rpu, Rtop, Rbot


def invert_transfer(vout: float, p: PT1000Params) -> float:
    """Given Vout, estimate temperature assuming nominal resistors.

    Vs = Vout / Gain_nom;  Rrtd_est = Vs*Rpu / (VDD - Vs);  T = (R/R0-1)/alpha.
    """
    gain_nom = 1.0 + p.r_gain_top_nom / p.r_gain_bot_nom
    vs = vout / gain_nom
    if vs >= p.vdd:  # avoid div by zero
        vs = p.vdd * 0.999999
    r_est = vs * p.r_pu_nom / max(p.vdd - vs, 1e-12)
    t_est = (r_est / p.r0 - 1.0) / p.alpha
    return t_est


def run_mc(
    t_true: float, n: int = 1000, sigma_pct: float = 0.01, seed: int = 123, *, progress: bool = True
) -> None:
    p = PT1000Params()
    r_rtd = pt1000_r(t_true, r0=p.r0, alpha=p.alpha)
    c, Rpu, Rtop, Rbot = build_pt1000_chain(r_rtd, p)

    # Vary all non-RTD resistors by 1% (1-sigma)
    mapping = {Rpu: NormalPct(sigma_pct), Rtop: NormalPct(sigma_pct), Rbot: NormalPct(sigma_pct)}

    mc = monte_carlo(
        circuit=c,
        mapping=mapping,
        n=n,
        analysis_factory=lambda: OP(),
        seed=seed,
        workers=1,
        progress=progress,
    )

    # Post-process: pick V(vout) from OP result (single point) and estimate temperature
    vouts = []
    for run in mc.runs:
        ts = run.traces
        vouts.append(float(ts["v(vout)"].values[-1]))
    vouts_arr = np.asarray(vouts)
    t_est = np.array([invert_transfer(v, p) for v in vouts_arr])
    err = t_est - t_true

    # Plot histogram of temperature error
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(err, bins=40, edgecolor="black", alpha=0.85)
    ax.set_title(f"PT1000 MC — Temperature error @ {t_true:.1f} °C (n={n})")
    ax.set_xlabel("Error [°C]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("pt1000_mc_error_hist.png", dpi=150)
    print("[saved] pt1000_mc_error_hist.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="PT1000 Monte Carlo example")
    ap.add_argument("--temp", type=float, default=100.0, help="True temperature in °C")
    ap.add_argument("--n", type=int, default=1000, help="Number of Monte Carlo runs")
    ap.add_argument("--sigma", type=float, default=0.01, help="Resistor 1-sigma (relative)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    args = ap.parse_args()
    run_mc(t_true=args.temp, n=args.n, sigma_pct=args.sigma, progress=not args.no_progress)


if __name__ == "__main__":
    main()
