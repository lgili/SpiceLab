## ADDED Requirements

### Requirement: BJT Transistor Support
The system SHALL provide a `BJT` component class for bipolar junction transistors (NPN and PNP types).

The BJT class SHALL:
- Accept a reference designator, model name, and optional type (NPN/PNP)
- Provide ports: `c` (collector), `b` (base), `e` (emitter), and optionally `s` (substrate)
- Generate SPICE card in format: `Q<ref> <c> <b> <e> [<s>] <model>`
- Support optional instance parameters (area, multiplier)

#### Scenario: Create NPN transistor
- **WHEN** user creates `BJT("1", model="2N2222")`
- **THEN** component has ports c, b, e accessible
- **THEN** `spice_card()` returns `Q1 <c_node> <b_node> <e_node> 2N2222`

#### Scenario: Create PNP transistor with substrate
- **WHEN** user creates `BJT("2", model="2N2907", substrate=True)`
- **THEN** component has ports c, b, e, s accessible
- **THEN** `spice_card()` includes substrate node

---

### Requirement: MOSFET Transistor Support
The system SHALL provide a `MOSFET` component class for MOS field-effect transistors (NMOS and PMOS types).

The MOSFET class SHALL:
- Accept a reference designator, model name, and optional type (NMOS/PMOS)
- Provide ports: `d` (drain), `g` (gate), `s` (source), `b` (bulk/body)
- Generate SPICE card in format: `M<ref> <d> <g> <s> <b> <model> [W=<w>] [L=<l>]`
- Support W (width) and L (length) parameters
- Support optional instance parameters (multiplier, fingers)

#### Scenario: Create NMOS transistor
- **WHEN** user creates `MOSFET("1", model="NMOS_BASIC", w="10u", l="1u")`
- **THEN** component has ports d, g, s, b accessible
- **THEN** `spice_card()` returns `M1 <d> <g> <s> <b> NMOS_BASIC W=10u L=1u`

#### Scenario: Create PMOS with bulk tied to source
- **WHEN** user creates `MOSFET("2", model="PMOS_BASIC")` and connects b to s
- **THEN** circuit netlist shows bulk and source on same node

---

### Requirement: JFET Transistor Support
The system SHALL provide a `JFET` component class for junction field-effect transistors (NJF and PJF types).

The JFET class SHALL:
- Accept a reference designator and model name
- Provide ports: `d` (drain), `g` (gate), `s` (source)
- Generate SPICE card in format: `J<ref> <d> <g> <s> <model>`

#### Scenario: Create N-channel JFET
- **WHEN** user creates `JFET("1", model="2N5457")`
- **THEN** component has ports d, g, s accessible
- **THEN** `spice_card()` returns `J1 <d> <g> <s> 2N5457`

---

### Requirement: Zener Diode Support
The system SHALL provide a `ZenerDiode` component class for voltage reference diodes.

The ZenerDiode class SHALL:
- Accept a reference designator and model name
- Provide ports: `a` (anode), `c` (cathode)
- Generate SPICE card in format: `D<ref> <a> <c> <model>`
- Use the same format as regular diode but with Zener model

#### Scenario: Create Zener diode
- **WHEN** user creates `ZenerDiode("1", model="1N4733")` (5.1V Zener)
- **THEN** component has ports a, c accessible
- **THEN** `spice_card()` returns `D1 <a> <c> 1N4733`

---

### Requirement: Mutual Inductance Support
The system SHALL provide a `MutualInductance` component class for magnetically coupled inductors.

The MutualInductance class SHALL:
- Accept a reference designator, two inductor references, and coupling coefficient
- NOT have physical ports (references existing inductors)
- Generate SPICE card in format: `K<ref> <L1_ref> <L2_ref> <coupling>`
- Support coupling coefficient from 0 to 1

#### Scenario: Create coupled inductors
- **WHEN** user creates `MutualInductance("1", l1="L1", l2="L2", coupling=0.99)`
- **THEN** `spice_card()` returns `K1 L1 L2 0.99`

#### Scenario: Coupling coefficient validation
- **WHEN** user creates `MutualInductance("1", l1="L1", l2="L2", coupling=1.5)`
- **THEN** system raises `ValueError` for invalid coupling

---

### Requirement: Ideal Transformer Support
The system SHALL provide a `Transformer` component class for ideal transformers.

The Transformer class SHALL:
- Accept a reference designator, turns ratio, and optional coupling
- Provide ports: `p1` (primary +), `p2` (primary -), `s1` (secondary +), `s2` (secondary -)
- Generate SPICE netlist using coupled inductors (L + K) internally
- Support turns ratio specification

#### Scenario: Create 1:10 step-up transformer
- **WHEN** user creates `Transformer("1", turns_ratio=10)`
- **THEN** component has ports p1, p2, s1, s2 accessible
- **THEN** `spice_card()` returns internal L1, L2, and K elements

---

### Requirement: Transmission Line Support
The system SHALL provide transmission line component classes for signal integrity analysis.

The system SHALL provide:
- `TLine` class for lossless transmission lines (T element)
- `TLineLossy` class for lossy transmission lines (O element)
- `TLineRC` class for uniform RC lines (U element)

Each transmission line class SHALL:
- Provide four ports: `p1p`, `p1n` (port 1), `p2p`, `p2n` (port 2)
- Accept characteristic impedance and delay/length parameters
- Generate appropriate SPICE card format

#### Scenario: Create lossless transmission line
- **WHEN** user creates `TLine("1", z0=50, td="1n")`
- **THEN** component has ports p1p, p1n, p2p, p2n accessible
- **THEN** `spice_card()` returns `T1 <p1p> <p1n> <p2p> <p2n> Z0=50 TD=1n`

---

### Requirement: Behavioral Voltage Source Support
The system SHALL provide a `BVoltage` component class for arbitrary voltage expressions.

The BVoltage class SHALL:
- Accept a reference designator and voltage expression string
- Provide ports: `p` (positive), `n` (negative)
- Generate SPICE card in format: `B<ref> <p> <n> V=<expression>`
- Pass expression directly to SPICE engine (no parsing)

#### Scenario: Create voltage doubler expression
- **WHEN** user creates `BVoltage("1", expr="V(in)*2")`
- **THEN** component has ports p, n accessible
- **THEN** `spice_card()` returns `B1 <p> <n> V=V(in)*2`

#### Scenario: Create conditional voltage
- **WHEN** user creates `BVoltage("2", expr="IF(V(ctrl)>2.5, 5, 0)")`
- **THEN** `spice_card()` returns `B2 <p> <n> V=IF(V(ctrl)>2.5, 5, 0)`

---

### Requirement: Behavioral Current Source Support
The system SHALL provide a `BCurrent` component class for arbitrary current expressions.

The BCurrent class SHALL:
- Accept a reference designator and current expression string
- Provide ports: `p` (positive), `n` (negative)
- Generate SPICE card in format: `B<ref> <p> <n> I=<expression>`

#### Scenario: Create current mirror expression
- **WHEN** user creates `BCurrent("1", expr="I(Vref)*10")`
- **THEN** `spice_card()` returns `B1 <p> <n> I=I(Vref)*10`

---

### Requirement: Subcircuit Instance Support
The system SHALL provide a `SubcktInstance` component class for instantiating subcircuits.

The SubcktInstance class SHALL:
- Accept a reference designator, subcircuit name, and node connections
- Provide dynamically created ports based on subcircuit definition
- Generate SPICE card in format: `X<ref> <nodes...> <subckt_name> [params]`
- Support parameter passing to subcircuit instances

#### Scenario: Instantiate op-amp subcircuit
- **WHEN** user creates `SubcktInstance("1", name="LM741", nodes=["inp", "inn", "vcc", "vee", "out"])`
- **THEN** component has 5 ports accessible
- **THEN** `spice_card()` returns `X1 <inp> <inn> <vcc> <vee> <out> LM741`

#### Scenario: Instantiate with parameters
- **WHEN** user creates `SubcktInstance("2", name="RES", nodes=["a", "b"], params={"R": "1k"})`
- **THEN** `spice_card()` returns `X2 <a> <b> RES R=1k`

---

### Requirement: Component Helper Factories
The system SHALL provide convenience factory functions for creating components with auto-generated reference designators.

The system SHALL provide:
- `Q(model)` → BJT
- `M(model, w=None, l=None)` → MOSFET
- `J(model)` → JFET
- `DZ(model)` → ZenerDiode
- `K(l1, l2, coupling)` → MutualInductance
- `BV(expr)` → BVoltage
- `BI(expr)` → BCurrent
- `X(name, nodes, params=None)` → SubcktInstance
- `TLINE(z0, td)` → TLine

#### Scenario: Create BJT with auto-ref
- **WHEN** user calls `Q("2N2222")` twice
- **THEN** first call returns BJT with ref="1"
- **THEN** second call returns BJT with ref="2"

---

### Requirement: ASC Parser Component Recognition
The system SHALL update the ASC file parser to recognize and create instances of new component types.

The parser SHALL:
- Map LTspice symbol names to SpiceLab component classes
- Extract component parameters from symbol attributes
- Handle both standard and LTspice-specific component variants
- Create appropriate ideal models when vendor models are not available

#### Scenario: Parse schematic with NPN transistor
- **WHEN** parser reads .asc file containing `SYMBOL npn` element
- **THEN** parser creates `BJT` instance with correct model reference
- **THEN** parser connects ports to correct nets

#### Scenario: Parse schematic with MOSFET
- **WHEN** parser reads .asc file containing `SYMBOL nmos` element
- **THEN** parser creates `MOSFET` instance with W and L parameters if specified
