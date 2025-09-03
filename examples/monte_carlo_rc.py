from __future__ import annotations

import matplotlib.pyplot as plt
from _common import savefig

from cat.analysis import OP, NormalPct, monte_carlo, stack_runs_to_df
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND

# Circuito simples: V -> R -> GND
c = Circuit("ex_mc_rc")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", 1000.0)  # nominal numérico (por ora)
c.add(V1, R1)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], GND)
c.connect(V1.ports[1], GND)

# Monte Carlo: R ~ Normal(5%)
mc = monte_carlo(
    c,
    mapping={R1: NormalPct(0.05)},
    n=100,
    analysis_factory=lambda: OP(),
    seed=1234,
)

# Empilha em DF (inclui colunas dos samples)
df = stack_runs_to_df(mc.runs, mc.samples, y=None, with_x=True)
print("DF columns:", list(df.columns))
print("Rows:", df.shape[0])

# Histograma dos valores amostrados
vals = [s["Resistor.1"] for s in mc.samples]
fig = plt.figure()
ax = fig.gca()
ax.hist(vals, bins=20, edgecolor="black")
ax.set_title("Monte Carlo: R samples (Normal 5%)")
ax.set_xlabel("R (ohms)")
ax.set_ylabel("count")
fig.tight_layout()
savefig(fig, "mc_r_hist.png")

# (Opcional) salvar CSV
df.to_csv("mc_op_stack.csv", index=False)
print("[saved] mc_op_stack.csv")
