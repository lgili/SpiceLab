from __future__ import annotations

from spicelab.analysis import OP, ParamGrid, stack_step_to_df, step_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND

# RC parametrizado: V=VIN, R={R}, C={C}
c = Circuit("ex_step_grid")
V1 = Vdc("1", "{VIN}")
R1 = Resistor("1", "{R}")
C1 = Capacitor("1", "{C}")
c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])  # vin
c.connect(R1.ports[1], C1.ports[0])  # vout
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

grid: ParamGrid = {
    "VIN": [1.0, 5.0],
    "R": ["1k", "2k"],
    "C": ["100n", "220n"],
}
res = step_grid(c, grid, analysis_factory=lambda: OP(), order=["VIN", "R", "C"])
df = stack_step_to_df(res, y=None, with_x=True)

# Como OP geralmente não tem "time", apenas cheque as colunas disponíveis:
print("DF columns:", list(df.columns))
print("Unique combos:", df[["VIN", "R", "C"]].drop_duplicates().shape[0])

# (Opcional) salvar CSV
df.to_csv("step_grid_op.csv", index=False)
print("[saved] step_grid_op.csv")
