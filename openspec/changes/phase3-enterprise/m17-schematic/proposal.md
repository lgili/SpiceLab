# M17: Schematic Integration

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 10-12 weeks
**Dependencies:** M3 (component library), M8 (model management), M14 (PDK for IC schematics)

## Problem Statement

SpiceLab requires users to define circuits programmatically in Python, which is powerful but creates a barrier for electrical engineers accustomed to graphical schematic capture tools. Integration with popular EDA (Electronic Design Automation) tools would enable bidirectional workflows: import existing schematics into SpiceLab for analysis, and export SpiceLab circuits to visual formats for documentation and layout.

### Current Gaps
- ❌ No KiCad schematic import (community prototype exists but incomplete)
- ❌ No LTspice ASC parser (only basic support)
- ❌ No Altium Designer integration
- ❌ No Eagle/Fusion 360 Electronics import
- ❌ No gEDA/gschem support
- ❌ No schematic generator (Python → visual schematic)
- ❌ Limited symbol library (<50 symbols)
- ❌ No netlist round-trip verification

### Impact
- **User Adoption:** Engineers prefer visual schematics
- **Workflow Integration:** Isolated from existing design flows
- **Collaboration:** Cannot share schematics with non-Python users
- **Documentation:** No visual circuit diagrams for reports

## Objectives

1. **KiCad importer** - Production-ready KiCad schematic (.kicad_sch) parser
2. **LTspice ASC complete parser** - Full support for all components and subcircuits
3. **Altium Designer export** - Generate Altium-compatible schematics
4. **Eagle/Fusion import** - Parse Eagle XML schematics
5. **gEDA/gschem support** - Import gEDA schematics
6. **Schematic generator** - Python circuit → SVG/PNG schematic
7. **Symbol library** - 1000+ standard component symbols
8. **Target:** Seamless EDA tool integration, bidirectional workflows

## Technical Design

### 1. KiCad Schematic Importer (Production-Ready)

**KiCad file format (.kicad_sch is S-expression format):**

```python
# spicelab/integrations/kicad/parser.py
import sexpdata
from pathlib import Path
from dataclasses import dataclass

@dataclass
class KiCadSymbol:
    """KiCad schematic symbol."""
    lib_id: str  # e.g., "Device:R"
    reference: str  # e.g., "R1"
    value: str  # e.g., "10k"
    position: tuple[float, float]
    properties: dict[str, str]

@dataclass
class KiCadWire:
    """KiCad wire/net."""
    points: list[tuple[float, float]]
    net_name: str | None = None

class KiCadSchematicParser:
    """Parse KiCad 6.0+ schematic files."""

    def __init__(self, schematic_path: Path):
        self.schematic_path = schematic_path
        self.symbols: list[KiCadSymbol] = []
        self.wires: list[KiCadWire] = []
        self.global_labels: dict[str, tuple[float, float]] = {}

    def parse(self) -> 'Circuit':
        """Parse KiCad schematic and convert to SpiceLab Circuit."""

        with open(self.schematic_path, 'r') as f:
            sexp = sexpdata.load(f)

        # Parse symbols
        for item in sexp:
            if isinstance(item, list) and item[0] == 'symbol':
                symbol = self._parse_symbol(item)
                if symbol:
                    self.symbols.append(symbol)

            elif isinstance(item, list) and item[0] == 'wire':
                wire = self._parse_wire(item)
                self.wires.append(wire)

            elif isinstance(item, list) and item[0] == 'global_label':
                label = self._parse_global_label(item)
                self.global_labels.update(label)

        # Convert to SpiceLab Circuit
        circuit = self._build_circuit()

        return circuit

    def _parse_symbol(self, sexp: list) -> KiCadSymbol | None:
        """Parse symbol S-expression."""
        # Extract lib_id, reference, value, position
        lib_id = self._find_property(sexp, 'lib_id')
        reference = self._find_property(sexp, 'Reference')
        value = self._find_property(sexp, 'Value')

        if not all([lib_id, reference, value]):
            return None

        # Parse position
        position_data = self._find_tag(sexp, 'at')
        position = (float(position_data[1]), float(position_data[2]))

        properties = self._extract_properties(sexp)

        return KiCadSymbol(lib_id, reference, value, position, properties)

    def _build_circuit(self) -> 'Circuit':
        """Convert parsed KiCad data to SpiceLab Circuit."""
        from spicelab import Circuit, Resistor, Capacitor, Inductor, VoltageSource

        circuit = Circuit(name=self.schematic_path.stem)

        # Map KiCad symbols to SpiceLab components
        for symbol in self.symbols:
            component = self._kicad_to_spicelab(symbol)
            if component:
                circuit.add_component(component)

        # Infer net connectivity from wires and labels
        net_map = self._build_netlist_from_wires()

        # Apply net connections to components
        for component in circuit.components:
            # Map component pins to nets based on position and wires
            ...

        return circuit

    def _kicad_to_spicelab(self, symbol: KiCadSymbol):
        """Convert KiCad symbol to SpiceLab component."""
        lib_id = symbol.lib_id.lower()

        if 'device:r' in lib_id:
            return Resistor(
                ref=symbol.reference,
                resistance=self._parse_value(symbol.value)
            )
        elif 'device:c' in lib_id:
            return Capacitor(
                ref=symbol.reference,
                capacitance=self._parse_value(symbol.value)
            )
        elif 'device:l' in lib_id:
            return Inductor(
                ref=symbol.reference,
                inductance=self._parse_value(symbol.value)
            )
        # ... handle more component types

    def _parse_value(self, value_str: str) -> float:
        """Parse component value with SI units (10k, 100nF, etc.)."""
        from spicelab.core.units import parse_si_value
        return parse_si_value(value_str)
```

### 2. LTspice ASC Complete Parser

```python
# spicelab/integrations/ltspice/asc_parser.py
import re
from pathlib import Path

class LTspiceASCParser:
    """Parse LTspice ASC (schematic) files."""

    def __init__(self, asc_path: Path):
        self.asc_path = asc_path
        self.components: list = []
        self.wires: list = []
        self.texts: list = []

    def parse(self) -> 'Circuit':
        """Parse ASC file and convert to Circuit."""

        with open(self.asc_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('SYMBOL'):
                component = self._parse_symbol(lines, i)
                self.components.append(component)

            elif line.startswith('WIRE'):
                wire = self._parse_wire(line)
                self.wires.append(wire)

            elif line.startswith('TEXT'):
                text = self._parse_text(lines, i)
                self.texts.append(text)

            i += 1

        return self._build_circuit()

    def _parse_symbol(self, lines: list[str], start_idx: int) -> dict:
        """Parse SYMBOL block."""
        # Example:
        # SYMBOL res 128 96 R0
        # SYMATTR InstName R1
        # SYMATTR Value 10k

        match = re.match(r'SYMBOL\s+(\S+)\s+([\d-]+)\s+([\d-]+)\s+(\S+)', lines[start_idx])
        if not match:
            return {}

        symbol_type, x, y, orientation = match.groups()

        # Parse following SYMATTR lines
        attributes = {}
        i = start_idx + 1
        while i < len(lines) and lines[i].strip().startswith('SYMATTR'):
            attr_match = re.match(r'SYMATTR\s+(\S+)\s+(.*)', lines[i].strip())
            if attr_match:
                attr_name, attr_value = attr_match.groups()
                attributes[attr_name] = attr_value.strip()
            i += 1

        return {
            'type': symbol_type,
            'position': (int(x), int(y)),
            'orientation': orientation,
            'attributes': attributes
        }

    def _build_circuit(self) -> 'Circuit':
        """Convert parsed ASC to SpiceLab Circuit."""
        from spicelab import Circuit

        circuit = Circuit(name=self.asc_path.stem)

        for comp_data in self.components:
            component = self._ltspice_to_spicelab(comp_data)
            if component:
                circuit.add_component(component)

        # Build netlist from wires
        # ...

        return circuit
```

### 3. Schematic Generator (Python → Visual)

```python
# spicelab/integrations/schematic_gen.py
import schemdraw
import schemdraw.elements as elm

class SchematicGenerator:
    """Generate visual schematics from SpiceLab circuits."""

    def __init__(self, circuit: 'Circuit'):
        self.circuit = circuit
        self.drawing = schemdraw.Drawing()

    def generate_svg(self, output_path: Path):
        """Generate SVG schematic."""

        # Auto-layout algorithm
        layout = self._auto_layout()

        # Draw components
        for component in self.circuit.components:
            self._draw_component(component, layout[component.ref])

        # Draw wires
        for net in self.circuit.nets:
            self._draw_net(net)

        # Save
        self.drawing.save(str(output_path))

    def _draw_component(self, component, position: tuple[float, float]):
        """Draw a single component."""
        from spicelab import Resistor, Capacitor, VoltageSource

        if isinstance(component, Resistor):
            self.drawing += elm.Resistor().at(position).label(f"{component.ref}\\n{component.resistance}")

        elif isinstance(component, Capacitor):
            self.drawing += elm.Capacitor().at(position).label(f"{component.ref}\\n{component.capacitance}")

        elif isinstance(component, VoltageSource):
            self.drawing += elm.SourceV().at(position).label(f"{component.ref}\\n{component.voltage}V")

        # ... more component types

    def _auto_layout(self) -> dict[str, tuple[float, float]]:
        """Automatic component placement algorithm."""
        # Use graph-based layout (Sugiyama, force-directed)
        ...
```

### 4. Symbol Library Management

```python
# spicelab/integrations/symbols.py
from pathlib import Path
import json

class SymbolLibrary:
    """Manage component symbol library."""

    def __init__(self, library_path: Path):
        self.library_path = library_path
        self.symbols: dict[str, dict] = {}
        self._load_symbols()

    def _load_symbols(self):
        """Load symbol definitions from library."""
        # Load from JSON/YAML symbol database
        with open(self.library_path / "symbols.json", 'r') as f:
            self.symbols = json.load(f)

    def get_symbol(self, component_type: str) -> dict:
        """Get symbol definition for component type."""
        return self.symbols.get(component_type, self.symbols['default'])

    def add_symbol(self, name: str, definition: dict):
        """Add custom symbol to library."""
        self.symbols[name] = definition

    def export_kicad_lib(self, output_path: Path):
        """Export symbols as KiCad library."""
        # Generate .kicad_sym file
        ...

    def export_ltspice_lib(self, output_path: Path):
        """Export symbols as LTspice library."""
        # Generate .asy files
        ...
```

## Implementation Plan

### Phase 1: KiCad Integration (Weeks 1-3)
- [ ] KiCad S-expression parser
- [ ] Symbol-to-component mapping (50+ types)
- [ ] Wire/net connectivity builder
- [ ] Subcircuit handling
- [ ] Test with 20+ real KiCad schematics
- [ ] Export to KiCad format (round-trip)

### Phase 2: LTspice Integration (Weeks 4-5)
- [ ] Complete ASC parser (all components)
- [ ] Subcircuit (.asc hierarchy) support
- [ ] SPICE directive extraction
- [ ] Parameter sweep configuration import
- [ ] Test with LTspice example library

### Phase 3: Altium & Eagle (Weeks 6-7)
- [ ] Altium SchDoc parser (binary format)
- [ ] Eagle XML parser
- [ ] Component library mapping
- [ ] Export to Altium format

### Phase 4: Schematic Generator (Weeks 8-9)
- [ ] Auto-layout algorithm (graph-based)
- [ ] Schemdraw integration
- [ ] SVG/PNG/PDF export
- [ ] Customizable styles and themes
- [ ] 50+ auto-generated examples

### Phase 5: Symbol Library (Weeks 10-11)
- [ ] 1000+ standard symbols
- [ ] Symbol database (JSON/SQLite)
- [ ] Custom symbol editor
- [ ] Export to KiCad/LTspice formats

### Phase 6: Testing & Documentation (Week 12)
- [ ] Integration test suite (100+ schematics)
- [ ] Netlist round-trip verification
- [ ] EDA tool migration guide
- [ ] 20+ schematic examples
- [ ] Video tutorials

## Success Metrics

### Must Have
- [ ] KiCad import working (100+ schematics tested)
- [ ] LTspice ASC complete parser
- [ ] Schematic generator (SVG/PNG)
- [ ] 1000+ symbol library
- [ ] Round-trip netlist verification

### Should Have
- [ ] Altium integration
- [ ] Eagle/Fusion import
- [ ] Auto-layout optimization
- [ ] Custom symbol editor

### Nice to Have
- [ ] gEDA/gschem support
- [ ] EAGLE ULP script generator
- [ ] Web-based schematic viewer

## Dependencies

- M3 (Component Library) - component types
- M8 (Model Management) - component models
- M14 (PDK) - IC schematic symbols

## References

- [KiCad File Formats](https://dev-docs.kicad.org/en/file-formats/)
- [LTspice ASC Format](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html)
- [Schemdraw Documentation](https://schemdraw.readthedocs.io/)
- [Altium Designer Scripting](https://www.altium.com/documentation/altium-designer/scripting)
