## AnalogMux8 — Enable Ports Mode

`AnalogMux8` can operate in dynamic mode by exposing enable ports `en0..en7`.
Each channel uses a voltage‑controlled switch (`S...`) followed by a series
resistor `r_series`. When an enable input is high, the corresponding channel
conducts with `RON` (from the switch) and `r_series` in series; other channels
remain high‑Z (switch `ROFF`).

Options

- `enable_ports=True`: expose `en0..en7` and emit `S...` elements.
- `emit_model=True`: also emit a default `.model SW_<ref> SW(...)` for the switches.
- `sw_model`: custom model name (defaults to `SW_<ref>`).

Control polarity

- Switch control is `V(cp) - V(cn)`; the mux ties `cn` to ground so driving
  `enX` high enables channel `X`.

Example (OP)

```
from cat.core.circuit import Circuit
from cat.core.components import AnalogMux8, Vdc, Resistor
from cat.core.net import GND, Net
from cat.analysis import OP

c = Circuit("mux_en_ports")
vin = Net("vin")
vout = Net("vout")

# 5 V source at mux input
V1 = Vdc("1", 5.0)
M = AnalogMux8(ref="MU1", r_series=100.0, enable_ports=True, emit_model=True)
RL = Resistor("L", 1000.0)

c.add(V1, M, RL)
c.connect(V1.ports[0], vin); c.connect(V1.ports[1], GND)
c.connect(M.ports[0], vin)           # in
c.connect(M.ports[1 + 2], vout)      # out2 -> vout
# Terminate other outputs if needed; for OP leave them floating or to GND

# Load at vout
c.connect(RL.ports[0], vout); c.connect(RL.ports[1], GND)

# Drive enables: en2=1 V (on), others 0 V (off)
for i in range(8):
    V_en = Vdc(f"EN{i}", 1.0 if i == 2 else 0.0)
    c.add(V_en)
    c.connect(V_en.ports[0], M.ports[1 + 8 + i])  # en ports follow the 8 outputs
    c.connect(V_en.ports[1], GND)

r = OP().run(c)
print("V(vout)=", float(r.traces["v(vout)"].values[-1]))
```

Expected result

- With `SW` model `RON=10` and `r_series=100`, and `RL=1k`, a static divider
  yields `Vout ≈ 5 V × 1k / (1k + 10 + 100)`.

Notes

- For time‑varying enables use `Vpulse` sources on selected `enX` nets and
  run a `TRAN` analysis to observe switching.
- If you prefer your own switch model, pass `sw_model="MY_SW"` and add a
  `.model MY_SW SW(...)` via `c.add_directive()`.
