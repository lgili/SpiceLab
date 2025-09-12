from __future__ import annotations

import shutil

from cat import GND, Circuit, opamp_inverting
from cat.analysis import AC, TRAN, ac_gain_phase
from cat.core.components import R, V


def inverting_demo() -> None:
    c = Circuit("opamp_inverting_demo")

    # Create named nodes via Nets from component pins
    V1 = V(1.0)
    c.add(V1)
    vin = V1.ports[0]  # source positive pin

    # Define an output node using a load resistor
    load = R("1k")
    c.add(load)
    vout = load.ports[0]
    c.connect(load.ports[1], GND)

    # Inverting amplifier: gain = -Rf/Rin = -100k/10k = -10
    _u, _rin, _rf = opamp_inverting(c, inp=vin, out=vout, ref=GND, Rin="10k", Rf="100k", gain=1e6)

    # Tie V1 negative to ground
    c.connect(V1.ports[1], GND)

    # Optionally run analyses if ngspice is available
    if shutil.which("ngspice"):
        res_tran = TRAN("10us", "5ms").run(c)
        print("tran traces:", res_tran.traces.names)

        res_ac = AC("dec", 20, 10.0, 1e5).run(c)
        f, mag_db, _ = ac_gain_phase(res_ac.traces, "v(n1)")
        print("ac points:", len(f), "| first mag[dB]=", mag_db[0])


if __name__ == "__main__":
    inverting_demo()
