# LTspice Integration

CAT interoperates with LTspice in two complementary ways:

1. Import **netlists** exported via *View → SPICE Netlist* (`.cir`/`.net`).
2. Round-trip **schematics** (`.asc`) for a curated symbol library, generating
   LTspice-friendly projects straight from your Python models.

## Netlists (`.cir` / `.net`)

Use `spicelab.io.ltspice_parser.from_ltspice_file` to parse a netlist and obtain a
`Circuit` object that can be analysed, modified or exported again.

### Supported cards

- `.include` / `.param` expansion (recursive)
- Devices: R, C, L, independent V/I (DC/AC), `PULSE`, `SIN`, `PWL`
- Controlled sources: E, G, F, H
- Diodes `D`
- Voltage/current controlled switches: `S` (VSWITCH), `W` (ISWITCH)
- One-level `.SUBCKT` flattening with `{PARAM}` substitution

```python
from spicelab.io.ltspice_parser import from_ltspice_file
from spicelab.analysis import run_tran

c = from_ltspice_file("./my_filter.cir")
tran = run_tran(c, "1us", "2ms", return_df=True)
print(tran.head())
```

> ℹ️ **Tips**
> - Keep `.model` statements together with the circuit or re-add them via
>   `circuit.add_directive(".model ...")`.
> - Unknown cards will raise a descriptive `ValueError`. If something critical is
>   missing, please open an issue with a minimal example.

## Schematics (`.asc`)

The module `spicelab.io.ltspice_asc` introduces helpers to load, generate and save
schematic files:

- `parse_asc(path_or_text)` → `AscSchematic`
- `schematic_to_circuit(AscSchematic)` → `Circuit`
- `circuit_to_schematic(Circuit, include_wires=True)` → `AscSchematic`
- `circuit_to_asc_text` / `save_circuit_as_asc`

### When should I use `SpiceLine`?

The exporter attaches `SYMATTR SpiceLine ...` to every symbol with the exact
SPICE card produced by spicelab. When these attributes are present the importer uses
those directives verbatim, guaranteeing an exact round-trip.

If a schematic lacks `SpiceLine`, spicelab falls back to analysing wires and symbol
pins (union-find). This allows importing hand-drawn LTspice schematics, but the
result may require manual clean-up if multiple nets share the same coordinates.

### Supported symbols (export)

| Symbol    | CAT component                     |
|-----------|----------------------------------|
| `res`     | `Resistor`                       |
| `cap`     | `Capacitor`                      |
| `ind`     | `Inductor`                       |
| `voltage` | `Vdc`                            |
| `current` | `Idc`                            |
| `dio`     | `Diode`                          |
| `vcvs`    | `VCVS`                           |
| `vccs`    | `VCCS`                           |
| `cccs`    | `CCCS`                           |
| `ccvs`    | `CCVS`                           |

Orientation is auto-selected when practical (for example, capacitors tied to
GND are placed vertically).

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.io.ltspice_asc import circuit_to_asc_text, parse_asc, schematic_to_circuit

vin = Net("vin")
vout = Net("vout")

c = Circuit("rc")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "10n")

c.add(V1, R1, C1)
c.connect(V1.ports[0], vin)
c.connect(R1.ports[0], vin)
c.connect(R1.ports[1], vout)
c.connect(C1.ports[0], vout)
c.connect(C1.ports[1], GND)
c.connect(V1.ports[1], GND)

asc_text = circuit_to_asc_text(c)
print(asc_text)

round_trip = schematic_to_circuit(parse_asc(asc_text))
print(round_trip.build_netlist())
```

### Limitations

- Unsupported symbols raise `ValueError`. You can still edit the exported `.asc`
  in LTspice and swap symbols manually.
- The geometry fallback assumes orthogonal wires; schematics using unusual
  routing may need touch-ups before importing.
- `include_wires=False` produces schematic stubs that rely entirely on
  `SpiceLine`. This is convenient for programmatic generation but results in
  schematics without conductor art.

### Converting existing `.asc`

```python
from spicelab.io.ltspice_asc import circuit_from_asc
from spicelab.analysis import run_ac

c = circuit_from_asc("./filter.asc")
res = run_ac(c, "dec", 201, 10, 1e5)
print(res.traces.names)
```

If your schematic has symbols that CAT does not understand, consider exporting a
netlist instead or filing an issue with a small reproduction.

See also [Component Library](components-library.md) if you want to register manufacturer
specific devices (for example a diode with a recommended `.model` line) and
re-use them across multiple projects.
