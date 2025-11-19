# M14: PDK Integration

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 10-12 weeks
**Dependencies:** M3 (component library), M8 (model management), M13 (mixed-signal for digital cells)

## Problem Statement

SpiceLab currently lacks integration with Process Design Kits (PDKs), which are essential for IC (Integrated Circuit) design. PDKs contain transistor models, parasitic extraction rules, design rule constraints, and foundry-specific technology parameters. Without PDK support, SpiceLab cannot be used for real-world chip design targeting specific fabrication processes.

### Current Gaps
- ❌ No PDK loader/parser (SkyWater, GlobalFoundries, TSMC formats)
- ❌ No process corner automation (TT, FF, SS, FS, SF)
- ❌ No parasitic extraction integration (SPEF, DSPF)
- ❌ No DRC (Design Rule Check) integration
- ❌ No LVS (Layout Versus Schematic) integration
- ❌ No technology scaling helpers (180nm → 130nm → 65nm, etc.)
- ❌ No vendor-specific device models (BSIM, PSP, HSPICE Level)

### Impact
- **IC Design:** Cannot target real fabrication processes
- **Industry Adoption:** IC designers cannot use SpiceLab
- **Open-Source Hardware:** Cannot leverage open PDKs (SkyWater, GF)
- **Academic Use:** Universities teaching IC design need PDK support

## Objectives

1. **SkyWater PDK 130nm support** - Full integration with open-source PDK
2. **GlobalFoundries PDK integration** - GF180MCU (180nm) support
3. **TSMC model support** - Handle NDA-protected models gracefully
4. **Custom PDK loader** - Generic framework for any PDK
5. **Process corner automation** - TT, FF, SS, FS, SF corner simulations
6. **Parasitic extraction** - SPEF/DSPF file integration
7. **DRC/LVS hooks** - Integration points for Magic, KLayout
8. **Target:** 3+ open PDKs supported, enable real IC design workflows

## Technical Design

### 1. PDK Architecture

**PDK Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│                    SpiceLab PDK System                      │
├─────────────────────────────────────────────────────────────┤
│  PDK Registry                                               │
│  ┌───────────────┬───────────────┬───────────────┐         │
│  │ SkyWater 130  │ GF 180MCU     │ Custom PDK    │         │
│  │               │               │               │         │
│  │ • Devices     │ • Devices     │ • Devices     │         │
│  │ • Models      │ • Models      │ • Models      │         │
│  │ • Corners     │ • Corners     │ • Corners     │         │
│  │ • Rules       │ • Rules       │ • Rules       │         │
│  └───────────────┴───────────────┴───────────────┘         │
│                          ↓                                  │
│  Process Corner Manager                                     │
│  ┌─────────────────────────────────────────────┐           │
│  │ TT (Typical)  FF (Fast)  SS (Slow)          │           │
│  │ FS (Fast-Slow)  SF (Slow-Fast)              │           │
│  └─────────────────────────────────────────────┘           │
│                          ↓                                  │
│  Device Library (PDK-specific)                              │
│  ┌─────────────────────────────────────────────┐           │
│  │ NMOS, PMOS, Resistors, Capacitors, Diodes  │           │
│  │ Varactors, BJTs, JFETs, MOSCAPs             │           │
│  └─────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 2. PDK Data Model

```python
# spicelab/pdk/base.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Literal
from pydantic import BaseModel

class ProcessCorner(BaseModel):
    """Process corner specification."""
    name: str  # "tt", "ff", "ss", "fs", "sf"
    nmos_corner: str  # "typical", "fast", "slow"
    pmos_corner: str  # "typical", "fast", "slow"
    temperature: float  # Celsius
    voltage: float  # Supply voltage (V)
    model_files: list[Path]  # SPICE model files for this corner

class DeviceModel(BaseModel):
    """PDK device model."""
    name: str  # e.g., "nfet_01v8"
    device_type: Literal["nmos", "pmos", "resistor", "capacitor", "diode", "bjt"]
    model_name: str  # SPICE model name
    min_length: float  # Minimum channel length (m)
    min_width: float  # Minimum channel width (m)
    max_length: float | None = None
    max_width: float | None = None
    parameters: dict[str, float] = field(default_factory=dict)

class PDKInfo(BaseModel):
    """Process Design Kit metadata."""
    name: str  # "sky130", "gf180mcu", "tsmc180"
    foundry: str  # "SkyWater", "GlobalFoundries", "TSMC"
    process_node: str  # "130nm", "180nm", "65nm"
    technology: str  # "CMOS", "BiCMOS", "SOI"
    version: str  # "1.0.0"
    license: str  # "Apache-2.0", "NDA", "Proprietary"
    pdk_root: Path  # Root directory of PDK files

class PDK(Protocol):
    """Protocol for all PDKs."""

    info: PDKInfo
    devices: dict[str, DeviceModel]
    corners: dict[str, ProcessCorner]

    def get_device(self, name: str) -> DeviceModel:
        """Get device model by name."""
        ...

    def get_corner(self, corner: str) -> ProcessCorner:
        """Get process corner definition."""
        ...

    def load_models(self, corner: str) -> str:
        """Generate SPICE .lib statements for corner."""
        ...
```

### 3. SkyWater PDK 130nm Integration

**SkyWater PDK structure:**
```
sky130_fd_pr/  (Foundry Devices - Primitives)
├── models/
│   ├── sky130.lib.spice       # Top-level library
│   ├── sky130_fd_pr__nfet_01v8.pm3.spice  # NMOS 1.8V
│   ├── sky130_fd_pr__pfet_01v8.pm3.spice  # PMOS 1.8V
│   ├── corners/
│   │   ├── tt.spice           # Typical-Typical
│   │   ├── ff.spice           # Fast-Fast
│   │   └── ss.spice           # Slow-Slow
├── cells/
│   ├── nfet_01v8/             # Standard 1.8V NMOS
│   ├── pfet_01v8/             # Standard 1.8V PMOS
│   ├── nfet_01v8_lvt/         # Low-Vt NMOS
│   └── ...
└── tech/
    ├── drc/                   # Design Rule Checks
    ├── lvs/                   # Layout vs Schematic
    └── pex/                   # Parasitic Extraction
```

**Implementation:**
```python
# spicelab/pdk/sky130.py
from pathlib import Path
import re

class SkyWater130PDK:
    """SkyWater 130nm PDK integration."""

    def __init__(self, pdk_root: Path):
        self.info = PDKInfo(
            name="sky130",
            foundry="SkyWater",
            process_node="130nm",
            technology="CMOS",
            version="1.0.0",
            license="Apache-2.0",
            pdk_root=pdk_root
        )

        self.devices: dict[str, DeviceModel] = {}
        self.corners: dict[str, ProcessCorner] = {}

        self._load_devices()
        self._load_corners()

    def _load_devices(self):
        """Load device models from PDK."""
        # Standard devices
        self.devices["nfet_01v8"] = DeviceModel(
            name="nfet_01v8",
            device_type="nmos",
            model_name="sky130_fd_pr__nfet_01v8",
            min_length=0.15e-6,  # 150nm
            min_width=0.42e-6,   # 420nm
            parameters={
                "mult": 1,  # Multiplier
                "nf": 1,    # Number of fingers
            }
        )

        self.devices["pfet_01v8"] = DeviceModel(
            name="pfet_01v8",
            device_type="pmos",
            model_name="sky130_fd_pr__pfet_01v8",
            min_length=0.15e-6,
            min_width=0.42e-6,
            parameters={"mult": 1, "nf": 1}
        )

        # Low-Vt variants
        self.devices["nfet_01v8_lvt"] = DeviceModel(
            name="nfet_01v8_lvt",
            device_type="nmos",
            model_name="sky130_fd_pr__nfet_01v8_lvt",
            min_length=0.15e-6,
            min_width=0.42e-6,
            parameters={"mult": 1, "nf": 1}
        )

        # High voltage devices
        self.devices["nfet_g5v0d10v5"] = DeviceModel(
            name="nfet_g5v0d10v5",
            device_type="nmos",
            model_name="sky130_fd_pr__nfet_g5v0d10v5",
            min_length=0.5e-6,
            min_width=1.0e-6,
            parameters={"mult": 1}
        )

        # Passive devices
        self.devices["res_high_po"] = DeviceModel(
            name="res_high_po",
            device_type="resistor",
            model_name="sky130_fd_pr__res_high_po",
            min_length=0.5e-6,
            min_width=0.35e-6,
            parameters={"rsh": 320}  # Sheet resistance (Ω/sq)
        )

        self.devices["cap_mim"] = DeviceModel(
            name="cap_mim",
            device_type="capacitor",
            model_name="sky130_fd_pr__cap_mim_m3_1",
            min_length=2.0e-6,
            min_width=2.0e-6,
            parameters={"cj": 2e-15}  # fF/µm²
        )

    def _load_corners(self):
        """Load process corners."""
        models_dir = self.info.pdk_root / "models" / "corners"

        # Typical-Typical (27°C, 1.8V)
        self.corners["tt"] = ProcessCorner(
            name="tt",
            nmos_corner="typical",
            pmos_corner="typical",
            temperature=27.0,
            voltage=1.8,
            model_files=[models_dir / "tt.spice"]
        )

        # Fast-Fast (high temp, high voltage)
        self.corners["ff"] = ProcessCorner(
            name="ff",
            nmos_corner="fast",
            pmos_corner="fast",
            temperature=-40.0,  # Cold = faster
            voltage=1.95,       # +5% voltage
            model_files=[models_dir / "ff.spice"]
        )

        # Slow-Slow (low temp, low voltage)
        self.corners["ss"] = ProcessCorner(
            name="ss",
            nmos_corner="slow",
            pmos_corner="slow",
            temperature=125.0,  # Hot = slower
            voltage=1.65,       # -5% voltage
            model_files=[models_dir / "ss.spice"]
        )

        # Fast-Slow (NMOS fast, PMOS slow)
        self.corners["fs"] = ProcessCorner(
            name="fs",
            nmos_corner="fast",
            pmos_corner="slow",
            temperature=27.0,
            voltage=1.8,
            model_files=[models_dir / "fs.spice"]
        )

        # Slow-Fast (NMOS slow, PMOS fast)
        self.corners["sf"] = ProcessCorner(
            name="sf",
            nmos_corner="slow",
            pmos_corner="fast",
            temperature=27.0,
            voltage=1.8,
            model_files=[models_dir / "sf.spice"]
        )

    def get_device(self, name: str) -> DeviceModel:
        """Get device model by name."""
        if name not in self.devices:
            raise ValueError(
                f"Device '{name}' not found in SkyWater 130nm PDK. "
                f"Available: {list(self.devices.keys())}"
            )
        return self.devices[name]

    def get_corner(self, corner: str) -> ProcessCorner:
        """Get process corner."""
        if corner not in self.corners:
            raise ValueError(
                f"Corner '{corner}' not found. "
                f"Available: {list(self.corners.keys())}"
            )
        return self.corners[corner]

    def load_models(self, corner: str) -> str:
        """Generate SPICE .lib statements."""
        corner_obj = self.get_corner(corner)

        lines = [
            f"* SkyWater 130nm PDK - {corner.upper()} corner",
            f".temp {corner_obj.temperature}",
            f".param VDD={corner_obj.voltage}"
        ]

        for model_file in corner_obj.model_files:
            lines.append(f'.lib "{model_file}" {corner}')

        return "\n".join(lines)
```

### 4. GlobalFoundries 180nm MCU Integration

```python
# spicelab/pdk/gf180mcu.py
class GlobalFoundries180PDK:
    """GlobalFoundries 180nm MCU PDK."""

    def __init__(self, pdk_root: Path):
        self.info = PDKInfo(
            name="gf180mcu",
            foundry="GlobalFoundries",
            process_node="180nm",
            technology="CMOS",
            version="1.0.0",
            license="Apache-2.0",
            pdk_root=pdk_root
        )

        self.devices: dict[str, DeviceModel] = {}
        self.corners: dict[str, ProcessCorner] = {}

        self._load_devices()
        self._load_corners()

    def _load_devices(self):
        """Load GF 180nm devices."""
        # 3.3V core devices
        self.devices["nfet_03v3"] = DeviceModel(
            name="nfet_03v3",
            device_type="nmos",
            model_name="nfet_03v3",
            min_length=0.28e-6,  # 280nm
            min_width=0.22e-6,   # 220nm
        )

        self.devices["pfet_03v3"] = DeviceModel(
            name="pfet_03v3",
            device_type="pmos",
            model_name="pfet_03v3",
            min_length=0.28e-6,
            min_width=0.22e-6,
        )

        # 6V I/O devices
        self.devices["nfet_06v0"] = DeviceModel(
            name="nfet_06v0",
            device_type="nmos",
            model_name="nfet_06v0",
            min_length=0.55e-6,
            min_width=0.3e-6,
        )

        # Poly resistors
        self.devices["res_poly"] = DeviceModel(
            name="res_poly",
            device_type="resistor",
            model_name="res_poly",
            min_length=0.5e-6,
            min_width=0.5e-6,
            parameters={"rsh": 120}  # Ω/sq
        )

        # MIM capacitors
        self.devices["cap_mim"] = DeviceModel(
            name="cap_mim",
            device_type="capacitor",
            model_name="cap_mim",
            min_length=5.0e-6,
            min_width=5.0e-6,
            parameters={"cj": 1.5e-15}  # fF/µm²
        )

    def _load_corners(self):
        """Load GF 180nm corners."""
        models_dir = self.info.pdk_root / "models"

        self.corners["tt"] = ProcessCorner(
            name="tt",
            nmos_corner="typical",
            pmos_corner="typical",
            temperature=25.0,
            voltage=3.3,
            model_files=[models_dir / "sm141064.ngspice" / "typical.ngspice"]
        )

        # Similar for ff, ss, fs, sf...
```

### 5. Custom PDK Loader

**Generic PDK loader for arbitrary formats:**

```python
# spicelab/pdk/loader.py
from pathlib import Path
import yaml
import json

class PDKLoader:
    """Load custom PDKs from configuration files."""

    @staticmethod
    def from_yaml(config_path: Path) -> PDK:
        """Load PDK from YAML configuration."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        info = PDKInfo(**config['info'])

        devices = {
            name: DeviceModel(**dev_config)
            for name, dev_config in config.get('devices', {}).items()
        }

        corners = {
            name: ProcessCorner(**corner_config)
            for name, corner_config in config.get('corners', {}).items()
        }

        return CustomPDK(info, devices, corners)

    @staticmethod
    def from_json(config_path: Path) -> PDK:
        """Load PDK from JSON configuration."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        return PDKLoader.from_yaml(config_path)  # Same structure

    @staticmethod
    def auto_detect(pdk_root: Path) -> PDK:
        """Auto-detect PDK type from directory structure."""
        # Check for SkyWater
        if (pdk_root / "libraries" / "sky130_fd_pr").exists():
            return SkyWater130PDK(pdk_root)

        # Check for GlobalFoundries
        if (pdk_root / "gf180mcu_fd_pr").exists():
            return GlobalFoundries180PDK(pdk_root)

        # Try to load custom config
        for config_file in ["pdk.yaml", "pdk.json", "tech.yaml"]:
            config_path = pdk_root / config_file
            if config_path.exists():
                if config_path.suffix == ".yaml":
                    return PDKLoader.from_yaml(config_path)
                elif config_path.suffix == ".json":
                    return PDKLoader.from_json(config_path)

        raise ValueError(f"Could not detect PDK type in {pdk_root}")

class CustomPDK:
    """Custom PDK loaded from config."""

    def __init__(
        self,
        info: PDKInfo,
        devices: dict[str, DeviceModel],
        corners: dict[str, ProcessCorner]
    ):
        self.info = info
        self.devices = devices
        self.corners = corners

    def get_device(self, name: str) -> DeviceModel:
        return self.devices[name]

    def get_corner(self, corner: str) -> ProcessCorner:
        return self.corners[corner]

    def load_models(self, corner: str) -> str:
        corner_obj = self.corners[corner]
        lines = [f"* {self.info.name} - {corner}"]

        for model_file in corner_obj.model_files:
            lines.append(f'.include "{model_file}"')

        return "\n".join(lines)
```

### 6. Process Corner Automation

```python
# spicelab/pdk/corners.py
from dataclasses import dataclass
import xarray as xr

@dataclass
class CornerSweep:
    """Automated corner sweep configuration."""
    corners: list[str]  # ["tt", "ff", "ss", "fs", "sf"]
    analyses: list[str]  # Analysis types to run
    parallel: bool = True  # Run corners in parallel

class CornerSimulator:
    """Automate process corner simulations."""

    def __init__(self, pdk: PDK):
        self.pdk = pdk

    async def run_corner_sweep(
        self,
        circuit: 'Circuit',
        sweep: CornerSweep,
        **kwargs
    ) -> dict[str, xr.Dataset]:
        """Run simulation across all process corners."""
        results = {}

        for corner_name in sweep.corners:
            corner = self.pdk.get_corner(corner_name)

            # Inject corner models into circuit
            circuit_with_corner = self._inject_corner(circuit, corner)

            # Run simulation
            result = await self._simulate(circuit_with_corner, sweep.analyses, **kwargs)
            results[corner_name] = result

        return results

    def _inject_corner(self, circuit: 'Circuit', corner: ProcessCorner) -> 'Circuit':
        """Inject corner-specific models and parameters."""
        # Clone circuit
        new_circuit = circuit.copy()

        # Add model includes
        model_includes = self.pdk.load_models(corner.name)
        new_circuit.add_raw(model_includes)

        # Set temperature
        new_circuit.add_raw(f".temp {corner.temperature}")

        # Set supply voltage
        new_circuit.add_raw(f".param VDD={corner.voltage}")

        return new_circuit

    def analyze_corners(self, results: dict[str, xr.Dataset]) -> dict:
        """Analyze corner simulation results."""
        analysis = {
            "worst_case": {},
            "best_case": {},
            "nominal": results.get("tt")
        }

        # Find worst/best for each measurement
        for measurement in results["tt"].data_vars:
            values = {corner: results[corner][measurement] for corner in results}

            worst_corner = min(values, key=lambda c: values[c].min())
            best_corner = max(values, key=lambda c: values[c].max())

            analysis["worst_case"][measurement] = {
                "corner": worst_corner,
                "value": values[worst_corner]
            }

            analysis["best_case"][measurement] = {
                "corner": best_corner,
                "value": values[best_corner]
            }

        return analysis
```

### 7. Parasitic Extraction Integration

**SPEF (Standard Parasitic Exchange Format) parser:**

```python
# spicelab/pdk/parasitic.py
from dataclasses import dataclass
import re

@dataclass
class ParasiticRC:
    """Parasitic resistance and capacitance."""
    net_name: str
    resistance: float  # Ohms
    capacitance: float  # Farads
    coupling_caps: dict[str, float] = None  # Coupling to other nets

class SPEFParser:
    """Parse SPEF files for parasitic extraction."""

    def __init__(self, spef_file: Path):
        self.spef_file = spef_file
        self.parasitic_map: dict[str, ParasiticRC] = {}

    def parse(self) -> dict[str, ParasiticRC]:
        """Parse SPEF file and extract parasitics."""
        with open(self.spef_file, 'r') as f:
            content = f.read()

        # Parse header
        design_name = self._parse_header(content)

        # Parse name map
        name_map = self._parse_name_map(content)

        # Parse D_NET sections (parasitic networks)
        nets = re.findall(r'\*D_NET (.*?) \*END', content, re.DOTALL)

        for net_section in nets:
            parasitic = self._parse_net_parasitics(net_section, name_map)
            if parasitic:
                self.parasitic_map[parasitic.net_name] = parasitic

        return self.parasitic_map

    def _parse_net_parasitics(self, net_section: str, name_map: dict) -> ParasiticRC:
        """Parse parasitics for a single net."""
        # Extract net name
        net_match = re.search(r'\*([^\s]+)', net_section)
        if not net_match:
            return None

        net_name = name_map.get(net_match.group(1), net_match.group(1))

        # Extract total capacitance
        cap_match = re.search(r'\*TOTAL_CAP\s+([\d.E+-]+)', net_section)
        total_cap = float(cap_match.group(1)) if cap_match else 0.0

        # Extract resistance (sum of segments)
        res_matches = re.findall(r'\d+:(\d+)\s+([\d.E+-]+)', net_section)
        total_res = sum(float(r[1]) for r in res_matches)

        # Extract coupling capacitances
        coupling_caps = {}
        coupling_matches = re.findall(r'\d+:(\d+):(\d+)\s+([\d.E+-]+)', net_section)
        for match in coupling_matches:
            coupled_net = name_map.get(match[1], match[1])
            coupling_caps[coupled_net] = float(match[2])

        return ParasiticRC(
            net_name=net_name,
            resistance=total_res,
            capacitance=total_cap,
            coupling_caps=coupling_caps
        )

class ParasiticInjector:
    """Inject parasitics into SpiceLab circuit."""

    def __init__(self, parasitics: dict[str, ParasiticRC]):
        self.parasitics = parasitics

    def inject(self, circuit: 'Circuit') -> 'Circuit':
        """Add parasitic R and C to circuit."""
        new_circuit = circuit.copy()

        for net_name, parasitic in self.parasitics.items():
            # Add series resistance
            if parasitic.resistance > 0:
                new_circuit.add_component(
                    Resistor(
                        ref=f"R_par_{net_name}",
                        resistance=parasitic.resistance
                    )
                )

            # Add ground capacitance
            if parasitic.capacitance > 0:
                new_circuit.add_component(
                    Capacitor(
                        ref=f"C_par_{net_name}",
                        capacitance=parasitic.capacitance
                    )
                )

            # Add coupling capacitances
            if parasitic.coupling_caps:
                for coupled_net, coupling_cap in parasitic.coupling_caps.items():
                    new_circuit.add_component(
                        Capacitor(
                            ref=f"C_coup_{net_name}_{coupled_net}",
                            capacitance=coupling_cap
                        )
                    )

        return new_circuit
```

### 8. DRC/LVS Integration Hooks

```python
# spicelab/pdk/verification.py
from pathlib import Path
import subprocess

class LayoutVerification:
    """Integration with DRC/LVS tools."""

    def __init__(self, pdk: PDK):
        self.pdk = pdk

    def run_drc(
        self,
        gds_file: Path,
        tool: Literal["magic", "klayout"] = "klayout"
    ) -> dict:
        """Run Design Rule Check."""
        if tool == "magic":
            return self._run_magic_drc(gds_file)
        elif tool == "klayout":
            return self._run_klayout_drc(gds_file)
        else:
            raise ValueError(f"Unknown DRC tool: {tool}")

    def _run_klayout_drc(self, gds_file: Path) -> dict:
        """Run KLayout DRC."""
        drc_script = self.pdk.info.pdk_root / "tech" / "drc" / f"{self.pdk.info.name}.lydrc"

        cmd = [
            "klayout",
            "-b",  # Batch mode
            "-r", str(drc_script),
            "-rd", f"input={gds_file}",
            "-rd", "output=drc_report.xml"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse DRC report
        violations = self._parse_drc_report("drc_report.xml")

        return {
            "passed": len(violations) == 0,
            "violations": violations
        }

    def run_lvs(
        self,
        gds_file: Path,
        netlist_file: Path,
        tool: Literal["magic", "netgen"] = "netgen"
    ) -> dict:
        """Run Layout Versus Schematic check."""
        if tool == "netgen":
            return self._run_netgen_lvs(gds_file, netlist_file)
        else:
            raise ValueError(f"Unknown LVS tool: {tool}")

    def _run_netgen_lvs(self, gds_file: Path, netlist_file: Path) -> dict:
        """Run Netgen LVS."""
        cmd = [
            "netgen",
            "-batch",
            "lvs",
            str(gds_file),
            str(netlist_file),
            self.pdk.info.name,
            "lvs_report.txt"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse LVS report
        passed = "Circuits match uniquely" in result.stdout

        return {
            "passed": passed,
            "report": result.stdout
        }
```

## Implementation Plan

### Phase 1: PDK Infrastructure (Weeks 1-3)
- [ ] Design PDK data model (PDKInfo, DeviceModel, ProcessCorner)
- [ ] Implement PDK base protocol
- [ ] Create PDK registry system
- [ ] Write custom PDK loader (YAML/JSON)
- [ ] Add auto-detection for known PDKs
- [ ] Write tests for PDK infrastructure

### Phase 2: SkyWater 130nm (Weeks 4-5)
- [ ] Implement SkyWater130PDK class
- [ ] Load standard devices (nfet, pfet, variants)
- [ ] Load passive devices (resistors, capacitors)
- [ ] Define all 5 process corners (tt, ff, ss, fs, sf)
- [ ] Test with real SkyWater PDK files
- [ ] Write 10+ SkyWater-based circuit examples

### Phase 3: GlobalFoundries 180nm (Weeks 6-7)
- [ ] Implement GlobalFoundries180PDK class
- [ ] Load 3.3V and 6V devices
- [ ] Load passive components
- [ ] Define process corners
- [ ] Test with GF180MCU PDK
- [ ] Write 5+ GF180-based examples

### Phase 4: Process Corner Automation (Weeks 8-9)
- [ ] Implement CornerSimulator class
- [ ] Automated corner sweep runner
- [ ] Parallel corner simulation
- [ ] Corner analysis tools (worst/best case)
- [ ] Statistical corner analysis
- [ ] Write corner sweep examples

### Phase 5: Parasitic Extraction (Weeks 10-11)
- [ ] Implement SPEF parser
- [ ] Implement DSPF parser (optional)
- [ ] ParasiticInjector for circuit modification
- [ ] Test with real SPEF files
- [ ] Performance comparison (pre/post parasitic)

### Phase 6: DRC/LVS Integration (Week 12)
- [ ] LayoutVerification class
- [ ] KLayout DRC integration
- [ ] Magic DRC integration
- [ ] Netgen LVS integration
- [ ] Write verification workflow examples
- [ ] Documentation and tutorials

## Success Metrics

### Functionality (Must Have)
- [ ] **3+ PDKs supported** (SkyWater 130, GF 180, Custom)
- [ ] **5 process corners** automated (tt, ff, ss, fs, sf)
- [ ] **SPEF parser** working for real extraction files
- [ ] **DRC/LVS** integration hooks functional
- [ ] **100+ PDK devices** available (across all PDKs)

### Performance
- [ ] Corner sweep **<2x overhead** vs single corner
- [ ] SPEF parsing **<1s** for 10k nets
- [ ] PDK loading **<500ms** on first use

### Documentation
- [ ] IC design tutorial (inverter → amplifier → ADC)
- [ ] 20+ PDK-based examples
- [ ] Migration guide from commercial tools
- [ ] PDK installation guide

## Dependencies

- M3 (Component Library) - base component framework
- M8 (Model Management) - for storing PDK models
- M13 (Mixed-Signal) - for digital standard cells

## References

- [SkyWater PDK Documentation](https://skywater-pdk.readthedocs.io/)
- [GlobalFoundries 180MCU](https://github.com/google/gf180mcu-pdk)
- [SPEF Format Specification](https://www.ispd.cc/spef/)
- [Magic VLSI Layout Tool](http://opencircuitdesign.com/magic/)
- [KLayout DRC](https://www.klayout.de/doc/about/drc_ref.html)
