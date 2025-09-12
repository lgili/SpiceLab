from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

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
    vdd = Net("vdd")

    VDD = Vdc("dd", p.vdd)
    c.add(VDD)
    c.connect(VDD.ports[0], vdd)
    c.connect(VDD.ports[1], GND)

    # Pull-up and RTD
    Rpu = Resistor("pu", p.r_pu_nom)
    Rrtd = Resistor("rtd", r_rtd)
    c.add(Rpu, Rrtd)
    c.connect(Rpu.ports[0], vdd)
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
    t_true: float,
    n: int = 1000,
    sigma_pct: float = 0.01,
    seed: int = 123,
    *,
    progress: bool = True,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    p = PT1000Params()
    r_rtd = pt1000_r(t_true, r0=p.r0, alpha=p.alpha)
    c, Rpu, Rtop, Rbot = build_pt1000_chain(r_rtd, p)

    # Optional quick connectivity check (uncomment if needed)
    # print("=== Connectivity Summary ===\n" + c.summary())
    # from cat.analysis import OP as _OP
    # _res = _OP().run(c)
    # print("OP traces:", _res.traces.names)
    # print("OP V(vin)=", float(_res.traces["v(vin)"].values[-1]))
    # print("OP V(vout)=", float(_res.traces["v(vout)"].values[-1]))

    # Vary all non-RTD resistors by 1% (1-sigma)
    mapping = {Rpu: NormalPct(sigma_pct), Rtop: NormalPct(sigma_pct), Rbot: NormalPct(sigma_pct)}

    mc = monte_carlo(
        circuit=c,
        mapping=mapping,
        n=n,
        analysis_factory=lambda: OP(),
        seed=seed,
        workers=workers,
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

    # Print quick stats for debugging
    def _stats(a: np.ndarray) -> str:
        return f"min={a.min():.6g} mean={a.mean():.6g} max={a.max():.6g} std={a.std(ddof=1):.6g}"

    print(f"Vout stats: {_stats(vouts_arr)}")
    print(f"Temp err stats [°C]: {_stats(err)}")

    return vouts_arr, t_est, err, mc.samples


def main() -> None:
    ap = argparse.ArgumentParser(description="PT1000 Monte Carlo example")
    ap.add_argument("--temp", type=float, default=100.0, help="True temperature in °C")
    ap.add_argument("--n", type=int, default=1000, help="Number of Monte Carlo runs")
    ap.add_argument("--sigma", type=float, default=0.01, help="Resistor 1-sigma (relative)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    ap.add_argument("--workers", type=int, default=1, help="MC workers (threads)")
    ap.add_argument("--bins", type=int, default=60, help="Histogram bins")
    ap.add_argument(
        "--temps",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of temperatures (°C) to overlay (defaults to --temp)",
    )
    ap.add_argument("--scatter", action="store_true", help="Plot scatter diagnostics")
    ap.add_argument(
        "--check", action="store_true", help="Sanity check OP and connectivity before MC"
    )
    args = ap.parse_args()

    temps = args.temps if args.temps else [args.temp]

    if args.check:
        # Run a quick connectivity + OP sanity check at the first temperature
        p = PT1000Params()
        r_rtd = pt1000_r(temps[0], r0=p.r0, alpha=p.alpha)
        c, *_ = build_pt1000_chain(r_rtd, p)
        print("=== Connectivity Summary ===\n" + c.summary())
        from cat.analysis import OP as _OP  # local import to avoid any circulars

        _res = _OP().run(c)
        print("OP traces:", _res.traces.names)
        v_vin = float(_res.traces["v(vin)"].values[-1])
        v_vout = float(_res.traces["v(vout)"].values[-1])
        print(f"OP V(vin)={v_vin}")
        print(f"OP V(vout)={v_vout}")
        lo, hi = 0.5, PT1000Params().vdd  # expect ~1..3.3 V per design; use 0.5..VDD as guard
        if not (lo <= v_vout <= hi):
            print(
                f"[ERROR] OP V(vout) out of expected range [{lo}, {hi}] V. Aborting MC.",
                file=sys.stderr,
            )
            sys.exit(2)

    results: dict[float, dict[str, Any]] = {}
    for t in temps:
        vouts, t_est, err, samples = run_mc(
            t_true=t,
            n=args.n,
            sigma_pct=args.sigma,
            seed=123,
            progress=not args.no_progress,
            workers=args.workers,
        )
        results[t] = {"vouts": vouts, "t_est": t_est, "err": err, "samples": samples}

    # 1) Vout distributions
    fig1 = plt.figure()
    ax1 = fig1.gca()
    for t, r in results.items():
        ax1.hist(r["vouts"], bins=args.bins, alpha=0.5, edgecolor="black", label=f"T={t:.1f}°C")
    ax1.set_title("PT1000 MC — Vout distributions")
    ax1.set_xlabel("Vout [V]")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("pt1000_mc_vout_hist.png", dpi=150)
    print("[saved] pt1000_mc_vout_hist.png")

    # 2) Temperature error distributions
    fig2 = plt.figure()
    ax2 = fig2.gca()
    for t, r in results.items():
        ax2.hist(r["err"], bins=args.bins, alpha=0.5, edgecolor="black", label=f"T={t:.1f}°C")
    ax2.set_title("PT1000 MC — Temperature error")
    ax2.set_xlabel("Error [°C]")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("pt1000_mc_error_hist.png", dpi=150)
    print("[saved] pt1000_mc_error_hist.png")

    # 3) Optional scatter diagnostics using the last temperature run
    if args.scatter:
        last_t = temps[-1]
        rlast = results[last_t]
        samples = rlast["samples"]
        err = rlast["err"]
        keys = []
        for k in ("Resistor.pu", "Resistor.t", "Resistor.b"):
            if samples and (k in samples[0]):
                keys.append(k)
        if keys:
            fig3, axs = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3))
            if len(keys) == 1:
                axs = [axs]
            for ax, k in zip(axs, keys, strict=False):
                x = np.array([s[k] for s in samples], dtype=float)
                ax.scatter(x, err, s=8, alpha=0.5)
                ax.set_xlabel(k)
                ax.set_ylabel("Temp error [°C]")
                ax.grid(True, alpha=0.3)
            fig3.tight_layout()
            fig3.savefig("pt1000_mc_scatter.png", dpi=150)
            print("[saved] pt1000_mc_scatter.png")


if __name__ == "__main__":
    main()
