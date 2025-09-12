from __future__ import annotations

import matplotlib.pyplot as plt
from _common import savefig

from cat.analysis import AC, ac_gain_phase
from cat.core.circuit import Circuit
from cat.core.components import VA, Capacitor, Resistor
from cat.core.net import GND
from cat.spice import ngspice_cli

# RC com fonte AC (1 V)
c = Circuit("ex_ac_bode")
V1 = VA(ac_mag=1.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")

c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])  # vin
c.connect(R1.ports[1], C1.ports[0])  # vout
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

res = AC("dec", 50, 10.0, 1e6).run(c)
f, mag_db, phase_deg = ac_gain_phase(res.traces, "v(out)")

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.semilogx(f, mag_db)
ax1.set_ylabel("Mag (dB)")
ax1.grid(True, which="both")

ax2 = fig.add_subplot(212)
ax2.semilogx(f, phase_deg)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Phase (deg)")
ax2.grid(True, which="both")

fig.suptitle("RC Bode")
fig.tight_layout()
savefig(fig, "ac_bode.png")
ngspice_cli.cleanup_artifacts(res.run.artifacts)
