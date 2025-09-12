# Importing LTspice Netlists

CAT can import SPICE netlists exported from LTspice (View → SPICE Netlist) and
run analyses using the same Python API.

## Supported features
- `.include` and `.param` (expanded recursively)
- Devices: R, C, L, V/I (DC/AC), `PULSE`, `SIN`, `PWL`
- Controlled sources: E/G/F/H
- Diode `D`
- Switches: `S` (VSWITCH), `W` (ISWITCH)
- Simple `.SUBCKT` flattening (one level, params via `{NAME}` placeholders)

## Quick example
```python
from cat.io.ltspice_parser import from_ltspice_file
from cat.analysis import run_tran

c = from_ltspice_file("./my_filter.cir")  # exported plain netlist
res_df = run_tran(c, "1us", "2ms", return_df=True)
print(res_df.head())
```

## Notes
- `.asc` schematic files are not netlists — export the SPICE netlist first.
- For switches, add your `.model` directives via `circuit.add_directive(".model ...")`
  if the model is not already present in the netlist.
- SUBCKT flattening: the importer expands a single level of subcircuit instances.
  Parameters are substituted when referenced as `{PARAM}` in the subckt body.
- If you see errors around unknown cards, file an issue with a minimal netlist.
