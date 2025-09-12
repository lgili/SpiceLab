from __future__ import annotations

import matplotlib.pyplot as plt
from _common import savefig

from cat import GND, Circuit, opamp_inverting
from cat.analysis import AC, ac_gain_phase, gain_margin_db, phase_margin
from cat.core.components import VA, R
from cat.core.net import Net
from cat.spice import ngspice_cli


def build() -> Circuit:
    c = Circuit("ex_opamp_stability")

    # Input source (small-signal AC = 1 V)
    vin_src = VA(ac_mag=1.0)
    c.add(vin_src)

    # Output/load node
    vout = Net("vout")
    load = R("10k")
    c.add(load)
    c.connect(load.ports[0], vout)
    c.connect(load.ports[1], GND)

    # Drive non-inverting topology via helper (here we use inverting for a clear loop)
    # Gain = -Rf/Rin = -100k/10k = -10
    opamp_inverting(c, inp=vin_src.ports[0], out=vout, ref=GND, Rin="10k", Rf="100k", gain=1e6)

    # Tie source reference to ground
    c.connect(vin_src.ports[1], GND)

    return c


def main() -> None:
    c = build()
    # AC sweep
    res = AC("dec", 201, 10.0, 1e6).run(c)
    f, mag_db, ph_deg = ac_gain_phase(res.traces, "v(vout)")

    # Margins at loop crossover
    pm = phase_margin(res.traces, "v(vout)")
    gm = gain_margin_db(res.traces, "v(vout)")
    print("phase_margin:", pm, "deg")
    print("gain_margin:", gm, "dB")

    # Plot Bode
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.semilogx(f, mag_db)
    ax1.set_ylabel("Mag (dB)")
    ax1.grid(True, which="both")

    ax2 = fig.add_subplot(212)
    ax2.semilogx(f, ph_deg)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both")

    fig.suptitle("Op-amp (ideal) — Closed-loop Bode")
    fig.tight_layout()
    savefig(fig, "opamp_bode.png")
    ngspice_cli.cleanup_artifacts(res.run.artifacts)


if __name__ == "__main__":
    main()
