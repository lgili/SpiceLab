# M3: Component Library Expansion

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 6-8 weeks
**Dependencies:** M1 (validation framework), M2 (performance for large catalogs)

## Problem Statement

SpiceLab currently has a minimal component library (~30 basic components) with no vendor model integration. Users must manually create or source component models for realistic simulations, which is time-consuming and error-prone. This severely limits the library's practical utility for professional circuit design.

### Current Gaps
- ❌ Library has only ~30 basic components (R, C, L, diodes, BJT, MOSFET)
- ❌ No vendor models (TI, Analog Devices, Infineon, STM, NXP)
- ❌ No subcircuit templates (OpAmp, LDO, Buck converter, etc.)
- ❌ No behavioral models (ideal switches, limiters, controlled sources)
- ❌ No model downloader/cacher for vendor websites
- ❌ No model validation (SPICE syntax checking)
- ❌ Users create everything from scratch

### Impact
- **User Experience:** Frustration from lack of ready-to-use components
- **Adoption:** Users choose PySpice/LTspice for better model libraries
- **Productivity:** Hours wasted finding/converting vendor models
- **Reliability:** Unvalidated user models lead to simulation errors

## Objectives

1. **Expand basic library** to 100+ components (E-series resistors, ceramic caps, inductors)
2. **Integrate vendor models** from TI, Analog Devices, Infineon, STM, NXP (50+ models)
3. **Build model registry** with downloader/cacher for automated fetching
4. **Create subcircuit templates** (OpAmp, LDO, Buck, Boost, ADC, etc.)
5. **Add behavioral models** (ideal switches, limiters, controlled sources)
6. **Implement model validation** (SPICE syntax checking, parameter verification)
7. **Document all models** with datasheets and example circuits

## Technical Design

### 1. Expanded Basic Component Library

**Strategy:** Provide comprehensive passive/active component catalog with standard values.

#### E-Series Resistors
```python
# spicelab/library/passive.py
from typing import Literal

E_SERIES_E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E_SERIES_E24 = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
]
E_SERIES_E96 = [...]  # Full E96 series

class ESeriesResistor:
    """E-series resistor with standard values."""

    @staticmethod
    def E12(value: float, decade: int = 3) -> Resistor:
        """Create E12 series resistor (1kΩ decade).

        Args:
            value: Base value from E12 series (1.0-8.2)
            decade: Power of 10 multiplier (e.g., 3 = kΩ)

        Example:
            >>> r = ESeriesResistor.E12(4.7, 3)  # 4.7kΩ
            >>> r.resistance
            4700.0
        """
        if value not in E_SERIES_E12:
            raise ValueError(f"{value} not in E12 series: {E_SERIES_E12}")
        resistance = value * (10 ** decade)
        return Resistor(f"R_{value}_{decade}", resistance)

    @staticmethod
    def nearest_E12(target: float) -> Resistor:
        """Find nearest E12 resistor to target value."""
        # Find closest E12 value
        decade = int(np.log10(target))
        normalized = target / (10 ** decade)
        closest = min(E_SERIES_E12, key=lambda x: abs(x - normalized))
        return ESeriesResistor.E12(closest, decade)

# Convenience factory functions
def R_E12(value: float, decade: int = 3) -> Resistor:
    """Shorthand for E12 resistor."""
    return ESeriesResistor.E12(value, decade)

def R_E24(value: float, decade: int = 3) -> Resistor:
    """Shorthand for E24 resistor."""
    return ESeriesResistor.E24(value, decade)
```

#### Ceramic Capacitors Catalog
```python
# spicelab/library/passive.py
from dataclasses import dataclass

@dataclass
class CapacitorSpec:
    """Ceramic capacitor specification."""
    value: float  # Farads
    voltage_rating: float  # Volts
    dielectric: str  # X7R, X5R, C0G, etc.
    tolerance: float  # Fractional (0.1 = 10%)
    package: str  # 0402, 0603, 0805, etc.

CERAMIC_CAP_CATALOG = {
    "C0G_100pF_50V_0603": CapacitorSpec(100e-12, 50, "C0G", 0.05, "0603"),
    "X7R_1uF_25V_0805": CapacitorSpec(1e-6, 25, "X7R", 0.1, "0805"),
    "X5R_10uF_16V_1206": CapacitorSpec(10e-6, 16, "X5R", 0.2, "1206"),
    # ... 100+ standard values
}

class CeramicCapacitor(Capacitor):
    """Ceramic capacitor with realistic specs."""

    def __init__(
        self,
        ref: str,
        spec_name: str | None = None,
        capacitance: float | None = None,
        voltage_rating: float = 50,
        dielectric: str = "X7R",
    ):
        if spec_name:
            spec = CERAMIC_CAP_CATALOG[spec_name]
            capacitance = spec.value
            self.voltage_rating = spec.voltage_rating
            self.dielectric = spec.dielectric
            self.tolerance = spec.tolerance
        else:
            self.voltage_rating = voltage_rating
            self.dielectric = dielectric
            self.tolerance = 0.1

        super().__init__(ref, capacitance)

# Usage
c1 = CeramicCapacitor("C1", "X7R_1uF_25V_0805")
c2 = CeramicCapacitor("C2", capacitance=100e-9, dielectric="C0G")
```

### 2. Vendor Model Registry

**Strategy:** Create database of vendor models with automated downloading.

#### Model Registry Database
```python
# spicelab/models/database.py
import sqlite3
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

class VendorModel(BaseModel):
    """Vendor SPICE model metadata."""
    vendor: str  # "TI", "ADI", "Infineon", etc.
    part_number: str  # "LM358", "OP27", etc.
    category: str  # "opamp", "ldo", "mosfet", etc.
    description: str
    url: str  # Download URL
    spice_file: str  # Subcircuit filename
    verified: bool = False
    checksum: Optional[str] = None

class ModelDatabase:
    """Local database of vendor models."""

    def __init__(self, db_path: Path = Path.home() / ".spicelab" / "models.db"):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY,
                vendor TEXT NOT NULL,
                part_number TEXT NOT NULL,
                category TEXT,
                description TEXT,
                url TEXT,
                spice_file TEXT,
                verified BOOLEAN,
                checksum TEXT,
                downloaded_at TIMESTAMP,
                UNIQUE(vendor, part_number)
            )
        """)
        conn.commit()
        conn.close()

    def add_model(self, model: VendorModel):
        """Add model to registry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO models
            (vendor, part_number, category, description, url, spice_file, verified, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.vendor,
            model.part_number,
            model.category,
            model.description,
            model.url,
            model.spice_file,
            model.verified,
            model.checksum,
        ))
        conn.commit()
        conn.close()

    def search(
        self,
        vendor: Optional[str] = None,
        category: Optional[str] = None,
        part_number: Optional[str] = None
    ) -> list[VendorModel]:
        """Search model database."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if vendor:
            query += " AND vendor = ?"
            params.append(vendor)
        if category:
            query += " AND category = ?"
            params.append(category)
        if part_number:
            query += " AND part_number LIKE ?"
            params.append(f"%{part_number}%")

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert to VendorModel objects
        models = []
        for row in rows:
            models.append(VendorModel(
                vendor=row[1],
                part_number=row[2],
                category=row[3],
                description=row[4],
                url=row[5],
                spice_file=row[6],
                verified=bool(row[7]),
                checksum=row[8],
            ))
        return models
```

#### Model Downloader
```python
# spicelab/models/downloader.py
import requests
import hashlib
from pathlib import Path

class ModelDownloader:
    """Download and cache vendor SPICE models."""

    def __init__(self, cache_dir: Path = Path.home() / ".spicelab" / "cache"):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model: VendorModel) -> Path:
        """Download model file (with caching)."""
        cached_path = self.cache_dir / model.vendor / model.spice_file

        # Check cache first
        if cached_path.exists():
            # Verify checksum if available
            if model.checksum and self._verify_checksum(cached_path, model.checksum):
                return cached_path

        # Download from vendor
        print(f"Downloading {model.part_number} from {model.vendor}...")
        response = requests.get(model.url, timeout=30)
        response.raise_for_status()

        # Save to cache
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(response.content)

        # Update checksum
        checksum = self._compute_checksum(cached_path)
        # TODO: Update database with checksum

        return cached_path

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum."""
        sha256 = hashlib.sha256()
        sha256.update(path.read_bytes())
        return sha256.hexdigest()

    def _verify_checksum(self, path: Path, expected: str) -> bool:
        """Verify file checksum."""
        actual = self._compute_checksum(path)
        return actual == expected

# Usage
db = ModelDatabase()
downloader = ModelDownloader()

# Search for TI opamps
models = db.search(vendor="TI", category="opamp")
for model in models:
    spice_file = downloader.download(model)
    print(f"Downloaded: {spice_file}")
```

### 3. Subcircuit Templates

**Strategy:** Provide reusable subcircuit templates for common blocks.

#### Generic OpAmp Template
```python
# spicelab/library/templates.py
from spicelab.core.circuit import Circuit, Subcircuit
from spicelab.core.components import VoltageSource, Resistor, Capacitor

class OpAmpTemplate(Subcircuit):
    """Generic operational amplifier template."""

    def __init__(
        self,
        ref: str,
        gain: float = 1e5,
        bandwidth: float = 1e6,  # Hz
        slew_rate: float = 1e6,   # V/s
        input_impedance: float = 1e12,
        output_impedance: float = 50,
    ):
        super().__init__(ref)
        self.gain = gain
        self.bandwidth = bandwidth
        self.slew_rate = slew_rate

        # Build internal circuit (behavioral model)
        self._build_internal()

    def _build_internal(self):
        """Build behavioral opamp model."""
        # Simplified opamp: VCVS + RC for bandwidth limiting
        # Real models are more complex (e.g., Boyle model)

        # Input stage (high impedance)
        self.add_port("in+", "non-inverting input")
        self.add_port("in-", "inverting input")
        self.add_port("out", "output")

        # Gain stage (voltage-controlled voltage source)
        self.add_component(f"E{self.ref}", "VCVS",
                          gain=self.gain,
                          input_ports=["in+", "in-"],
                          output_ports=["out_internal", "0"])

        # Bandwidth limiting (RC)
        gbw = self.gain * self.bandwidth
        r_bw = 1e3
        c_bw = 1 / (2 * np.pi * gbw * r_bw)
        self.add(Resistor(f"R_bw_{self.ref}", r_bw))
        self.add(Capacitor(f"C_bw_{self.ref}", c_bw))

        # Output stage (low impedance)
        self.add(Resistor(f"R_out_{self.ref}", self.output_impedance))

    def spice_subckt(self) -> str:
        """Generate SPICE subcircuit."""
        return f"""
.SUBCKT {self.ref} in+ in- out
* Generic OpAmp (gain={self.gain}, BW={self.bandwidth}Hz)
Rin+ in+ 0 1e12
Rin- in- 0 1e12
E1 out_int 0 in+ in- {self.gain}
R1 out_int out_rc 1k
C1 out_rc 0 {1/(2*np.pi*self.gain*self.bandwidth*1e3)}
Rout out_rc out {self.output_impedance}
.ENDS
"""

# Usage
opamp = OpAmpTemplate("U1", gain=1e5, bandwidth=1e6)
circuit.add_subcircuit(opamp)
circuit.connect(opamp.ports["in+"], signal)
circuit.connect(opamp.ports["in-"], feedback)
circuit.connect(opamp.ports["out"], output)
```

#### LDO Regulator Template
```python
# spicelab/library/templates.py

class LDOTemplate(Subcircuit):
    """Generic LDO regulator template."""

    def __init__(
        self,
        ref: str,
        v_out: float,  # Output voltage
        i_max: float = 1.0,  # Maximum current (A)
        dropout: float = 0.3,  # Dropout voltage (V)
        psrr: float = 60,  # PSRR (dB)
    ):
        super().__init__(ref)
        self.v_out = v_out
        self.i_max = i_max
        self.dropout = dropout
        self.psrr = psrr
        self._build_internal()

    def _build_internal(self):
        """Build behavioral LDO model."""
        # Ports: VIN, VOUT, GND, EN (enable)
        self.add_port("vin", "input voltage")
        self.add_port("vout", "output voltage")
        self.add_port("gnd", "ground")
        self.add_port("en", "enable (optional)")

        # TODO: Implement behavioral model
        # - Pass element (MOSFET or equivalent)
        # - Error amplifier (OpAmp comparing VOUT to VREF)
        # - Feedback divider
        # - Current limiting
        # - PSRR filter

    def spice_subckt(self) -> str:
        """Generate SPICE subcircuit for LDO."""
        return f"""
.SUBCKT {self.ref} vin vout gnd en
* Generic LDO (Vout={self.v_out}V, Imax={self.i_max}A)
* Simplified model - replace with vendor model for accuracy
.ENDS
"""
```

### 4. Behavioral Models

**Strategy:** Provide ideal components for system-level simulation.

#### Ideal Switch
```python
# spicelab/library/behavioral.py

class IdealSwitch:
    """Ideal voltage-controlled switch."""

    def __init__(
        self,
        ref: str,
        r_on: float = 1e-3,   # On resistance (Ω)
        r_off: float = 1e12,  # Off resistance (Ω)
        v_threshold: float = 2.5,  # Threshold voltage (V)
    ):
        self.ref = ref
        self.r_on = r_on
        self.r_off = r_off
        self.v_threshold = v_threshold

    def spice_card(self) -> str:
        """Generate SPICE switch card."""
        return f"""
* Ideal Switch {self.ref}
S{self.ref} n1 n2 ctrl_node 0 SWITCH_MODEL_{self.ref}
.MODEL SWITCH_MODEL_{self.ref} VSWITCH (RON={self.r_on} ROFF={self.r_off} VON={self.v_threshold} VOFF={self.v_threshold-0.1})
"""

# Ideal diode (zero forward drop, infinite reverse resistance)
class IdealDiode:
    """Ideal diode model."""

    def __init__(self, ref: str, v_forward: float = 0.0):
        self.ref = ref
        self.v_forward = v_forward

    def spice_card(self) -> str:
        return f"""
D{self.ref} anode cathode IDEAL_DIODE_{self.ref}
.MODEL IDEAL_DIODE_{self.ref} D (IS=1e-15 RS=1e-3 N=0.01 VJ=0)
"""
```

### 5. Model Validation

**Strategy:** Validate SPICE models before use (syntax, parameter ranges).

#### SPICE Syntax Validator
```python
# spicelab/validators/model_validation.py
import re
from pathlib import Path

class ModelValidator:
    """Validate SPICE model files."""

    def validate_file(self, path: Path) -> tuple[bool, list[str]]:
        """Validate SPICE model file.

        Returns:
            (is_valid, errors)
        """
        errors = []
        content = path.read_text()

        # Check for basic SPICE syntax
        if not self._has_valid_header(content):
            errors.append("Missing or invalid SPICE header")

        # Check for .MODEL or .SUBCKT definitions
        if not re.search(r'\.(MODEL|SUBCKT)', content, re.IGNORECASE):
            errors.append("No .MODEL or .SUBCKT found")

        # Check for matching .ENDS
        subckt_count = len(re.findall(r'\.SUBCKT', content, re.IGNORECASE))
        ends_count = len(re.findall(r'\.ENDS', content, re.IGNORECASE))
        if subckt_count != ends_count:
            errors.append(f"Unmatched .SUBCKT/.ENDS ({subckt_count} vs {ends_count})")

        # Check for invalid characters
        if re.search(r'[^\x00-\x7F]', content):
            errors.append("Non-ASCII characters found")

        # Validate parameter syntax
        param_errors = self._validate_parameters(content)
        errors.extend(param_errors)

        return (len(errors) == 0, errors)

    def _has_valid_header(self, content: str) -> bool:
        """Check for valid SPICE header (optional * comment line)."""
        lines = content.strip().split('\n')
        return len(lines) > 0

    def _validate_parameters(self, content: str) -> list[str]:
        """Validate parameter definitions."""
        errors = []
        # Check for .PARAM statements with invalid syntax
        param_lines = re.findall(r'\.PARAM\s+(.+)', content, re.IGNORECASE)
        for param in param_lines:
            # Basic validation: param=value
            if '=' not in param:
                errors.append(f"Invalid .PARAM syntax: {param}")
        return errors

# Usage
validator = ModelValidator()
is_valid, errors = validator.validate_file(Path("models/LM358.lib"))
if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### 6. Pre-Bundled Model Catalog

**Strategy:** Ship with curated set of verified models.

#### Bundled Models Structure
```
spicelab/models/catalog/
├── ti/
│   ├── opamps/
│   │   ├── LM358.lib
│   │   ├── TL071.lib
│   │   └── OPA2134.lib
│   ├── regulators/
│   │   ├── LM317.lib
│   │   └── TPS54331.lib
│   └── index.json
├── adi/
│   ├── opamps/
│   │   ├── OP27.lib
│   │   └── AD8065.lib
│   └── index.json
├── infineon/
│   ├── mosfets/
│   │   ├── IRF540.lib
│   │   └── BSS138.lib
│   └── index.json
└── manifest.json  # Master index
```

#### Model Catalog Loader
```python
# spicelab/models/catalog.py
import json
from pathlib import Path

CATALOG_DIR = Path(__file__).parent / "catalog"

class ModelCatalog:
    """Access bundled SPICE models."""

    def __init__(self):
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load master manifest."""
        manifest_path = CATALOG_DIR / "manifest.json"
        return json.loads(manifest_path.read_text())

    def list_vendors(self) -> list[str]:
        """List available vendors."""
        return list(self.manifest.keys())

    def list_models(self, vendor: str, category: str | None = None) -> list[str]:
        """List models for vendor."""
        vendor_dir = CATALOG_DIR / vendor
        index_path = vendor_dir / "index.json"
        index = json.loads(index_path.read_text())

        if category:
            return index.get(category, [])
        else:
            # Return all models
            all_models = []
            for cat_models in index.values():
                all_models.extend(cat_models)
            return all_models

    def get_model_path(self, vendor: str, part_number: str) -> Path:
        """Get path to model file."""
        # Search in vendor directory
        vendor_dir = CATALOG_DIR / vendor
        for category_dir in vendor_dir.iterdir():
            if category_dir.is_dir():
                model_path = category_dir / f"{part_number}.lib"
                if model_path.exists():
                    return model_path

        raise FileNotFoundError(f"Model {part_number} not found in {vendor} catalog")

# Usage
catalog = ModelCatalog()
vendors = catalog.list_vendors()  # ['ti', 'adi', 'infineon']
ti_opamps = catalog.list_models("ti", "opamps")  # ['LM358', 'TL071', ...]
model_path = catalog.get_model_path("ti", "LM358")
```

## Implementation Plan

### Week 1: Infrastructure
- [ ] Create `spicelab/library/` package structure
- [ ] Setup model database schema (SQLite)
- [ ] Implement ModelDatabase class
- [ ] Create model downloader with caching
- [ ] Add model validator (syntax checking)

### Week 2: Basic Components
- [ ] Implement E-series resistor catalog (E12, E24, E96)
- [ ] Create ceramic capacitor catalog (100+ values)
- [ ] Add inductor catalog (standard values)
- [ ] Implement diode catalog (1N4148, 1N4007, etc.)
- [ ] Add BJT catalog (2N2222, 2N3904, BC547, etc.)
- [ ] Create MOSFET catalog (2N7000, IRF540, BSS138, etc.)

### Week 3-4: Vendor Model Integration
- [ ] Research vendor model sources (TI, ADI, Infineon, etc.)
- [ ] Download and validate 50+ vendor models
- [ ] Create model manifest (JSON index)
- [ ] Organize models by vendor/category
- [ ] Add to model database
- [ ] Write tests for model loading

### Week 5: Subcircuit Templates
- [ ] Implement GenericOpAmp template
- [ ] Create LDO regulator template
- [ ] Add buck converter template
- [ ] Implement boost converter template
- [ ] Create flyback template
- [ ] Add ADC/DAC behavioral models

### Week 6: Behavioral Models
- [ ] Implement ideal switch
- [ ] Create ideal diode
- [ ] Add ideal transformer
- [ ] Implement controlled sources (VCVS, VCCS, CCVS, CCCS)
- [ ] Create limiters (voltage/current)
- [ ] Add slew rate limiter

### Week 7: Documentation
- [ ] Document all components with examples
- [ ] Create component catalog reference
- [ ] Write vendor model usage guide
- [ ] Add subcircuit template tutorials
- [ ] Document behavioral model limitations

### Week 8: Integration & Testing
- [ ] Write unit tests for all components
- [ ] Test vendor model loading
- [ ] Validate subcircuit templates
- [ ] Create example circuits using new components
- [ ] Performance test (catalog loading time)

## Success Metrics

### Component Counts
- [ ] **100+** basic components (passive + active)
- [ ] **50+** vendor models integrated
- [ ] **10+** subcircuit templates
- [ ] **10+** behavioral models

### Quality
- [ ] All models validated (SPICE syntax)
- [ ] All components documented with examples
- [ ] All components have unit tests
- [ ] Model catalog loads in <100ms

### User Experience
- [ ] One-line component creation (e.g., `R_E12(4.7, 3)`)
- [ ] Vendor models downloadable with single command
- [ ] Comprehensive searchable catalog
- [ ] Autocomplete support for VSCode/PyCharm

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Vendor model licensing | HIGH | Only include models with permissive licenses, provide downloader for others |
| Model file size | MEDIUM | Keep catalog <10MB, download large models on-demand |
| Model accuracy | HIGH | Validate all models, include disclaimers, reference vendor datasheets |
| Vendor URL changes | MEDIUM | Regular automated checks, fallback to local cache |
| SPICE dialect incompatibility | MEDIUM | Test models with NGSpice/LTspice/Xyce, document compatibility |

## Dependencies

**Required:**
- M1 (validation framework for model checking)
- M2 (performance for large catalogs)
- requests (`pip install requests`)
- sqlite3 (stdlib)

**Optional:**
- beautifulsoup4 (for web scraping vendor models)

## Future Enhancements

- **M8:** Advanced model management (versioning, encryption)
- **M13:** Mixed-signal models (Verilog-AMS)
- **M14:** PDK integration (SkyWater, GlobalFoundries)
- **M17:** Schematic import (KiCad component mapping)

## References

- [TI SPICE Models](https://www.ti.com/support/spice-models.html)
- [Analog Devices Models](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html)
- [Infineon Models](https://www.infineon.com/cms/en/tools/spice-models/)
- [E-Series Values](https://en.wikipedia.org/wiki/E_series_of_preferred_numbers)
- [SPICE Subcircuit Tutorial](http://bwrcs.eecs.berkeley.edu/Classes/IcBook/SPICE/UserGuide/subckt_ug.html)
