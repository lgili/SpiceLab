from __future__ import annotations

import shutil

from cat.analysis import OP, NormalPct, monte_carlo
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND, Net


def _rc() -> tuple[Circuit, Resistor]:
    c = Circuit("mc_df_rc")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", 1000.0)
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    vin = Net("vin")
    vout = Net("vout")
    c.connect(V1.ports[0], vin)  # vin
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)  # vout
    c.connect(C1.ports[0], vout)
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c, R1


def main() -> None:
    if not shutil.which("ngspice"):
        print("[warn] ngspice not found; this example may skip real simulation.")

    c, R1 = _rc()

    def metric_map(res):
        # expose last value of vout as metric
        return {"vout": float(res.traces["v(vout)"].values[-1])}

    mc = monte_carlo(
        c,
        mapping={R1: NormalPct(0.05)},
        n=64,
        analysis_factory=lambda: OP(),
        seed=42,
        workers=None,
    )

    df = mc.to_dataframe(metric=metric_map, param_prefix="param_")
    print(df.head())

    try:
        # optional: quick histogram if matplotlib is installed
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        ax.hist(df["vout"].values, bins=20, edgecolor="black", alpha=0.8)
        ax.set_title("Monte Carlo — vout distribution")
        ax.set_xlabel("vout [V]")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig("monte_carlo_df_hist.png", dpi=150)
        print("[saved] monte_carlo_df_hist.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
