from __future__ import annotations

import pandas as pd
from spicelab.analysis.sweep_grid import run_param_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

# RC parametrizado: V=VIN, R={R}, C={C}
c = Circuit("ex_step_grid")
V1 = Vdc("VIN", 1.0)
R1 = Resistor("R", "1k")
C1 = Capacitor("C", "100n")
c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])  # vin
c.connect(R1.ports[1], C1.ports[0])  # vout
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

grid = {
    "VIN": [1.0, 5.0],
    "R": ["1k", "2k"],
    "C": ["100n", "220n"],
}

result = run_param_grid(
    circuit=c,
    variables=[(V1, grid["VIN"]), (R1, grid["R"]), (C1, grid["C"])],
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
)

frames: list[pd.DataFrame] = []
for run in result.runs:
    df = run.handle.dataset().to_dataframe().reset_index()
    for key, value in run.combo.items():
        df[key] = value
    frames.append(df)

stacked = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

print("DF columns:", list(stacked.columns))
if not stacked.empty:
    print(
        "Unique combos:",
        stacked[["VIN", "R", "C"]].drop_duplicates().shape[0],
    )
    stacked.to_csv("step_grid_op.csv", index=False)
    print("[saved] step_grid_op.csv")
