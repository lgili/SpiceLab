# M8: Model Management System

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 10-12 weeks
**Dependencies:** M3 (component library foundations), M1 (validation framework)

## Problem Statement

SpiceLab lacks a systematic way to manage, version, and access SPICE models. Users must manually download models from vendor websites, manage local copies, handle different SPICE dialects, and ensure model compatibility. This creates significant friction for professional circuit design and limits the library's practical utility.

### Current Gaps
- ❌ No local model database (users manage files manually)
- ❌ No model versioning or metadata tracking
- ❌ No automatic model fetching from vendor websites
- ❌ No Verilog-A parser/compiler integration
- ❌ No IBIS model support
- ❌ No encrypted model support (vendor NDAs)
- ❌ No model verification suite (syntax, parameters, behavior)
- ❌ Users waste hours finding and configuring models

### Impact
- **User Experience:** Frustration from model management complexity
- **Productivity:** Hours wasted downloading and organizing models
- **Reliability:** Unverified models lead to simulation errors
- **Adoption:** Users choose tools with integrated model libraries
- **Professional Use:** Cannot handle vendor NDA-protected models

## Objectives

1. **Build local model database** with SQLite backend for metadata and search
2. **Implement model versioning** with change tracking and rollback
3. **Create automatic model downloader** from vendor websites (TI, ADI, Infineon, etc.)
4. **Integrate Verilog-A support** with parser and compiler to SPICE
5. **Add IBIS model parser** for I/O buffer characterization
6. **Support encrypted models** with vendor NDA compliance
7. **Build model verification suite** (syntax, parameter validation, behavioral checks)
8. **Achieve 500+ vendor models** in searchable, categorized database

## Technical Design

### 1. Model Database Architecture

**Strategy:** SQLite database with rich metadata, full-text search, and efficient caching.

#### Database Schema
```python
# spicelab/models/schema.py
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Vendor(Base):
    """Vendor information."""
    __tablename__ = 'vendors'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)  # "TI", "ADI", etc.
    website = Column(String)
    api_endpoint = Column(String, nullable=True)  # If vendor has API
    models = relationship("Model", back_populates="vendor")

class Model(Base):
    """SPICE model metadata."""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    vendor_id = Column(Integer, ForeignKey('vendors.id'))
    part_number = Column(String, nullable=False, index=True)
    category = Column(String, index=True)  # "opamp", "ldo", "mosfet", etc.
    subcategory = Column(String, nullable=True)
    description = Column(Text)

    # File information
    filename = Column(String)  # Local filename
    file_path = Column(String)  # Absolute path to model file
    file_hash = Column(String)  # SHA256 for integrity
    file_size = Column(Integer)  # Bytes

    # Model metadata
    model_type = Column(String)  # "subcircuit", "primitive", "verilog-a", "ibis"
    spice_dialect = Column(String)  # "ngspice", "ltspice", "hspice", "xyce"
    encrypted = Column(Boolean, default=False)

    # Version information
    version = Column(String)
    release_date = Column(DateTime)

    # Download information
    source_url = Column(String)
    downloaded_at = Column(DateTime)
    last_verified = Column(DateTime)
    verified = Column(Boolean, default=False)

    # Relationships
    vendor = relationship("Vendor", back_populates="models")
    parameters = relationship("ModelParameter", back_populates="model")
    pins = relationship("ModelPin", back_populates="model")

class ModelParameter(Base):
    """Model parameters (for subcircuits)."""
    __tablename__ = 'model_parameters'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    name = Column(String, nullable=False)
    default_value = Column(String)
    unit = Column(String)
    description = Column(Text)

    model = relationship("Model", back_populates="parameters")

class ModelPin(Base):
    """Model pins (for subcircuits)."""
    __tablename__ = 'model_pins'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    pin_number = Column(Integer)
    pin_name = Column(String)
    description = Column(String)

    model = relationship("Model", back_populates="pins")

class ModelVersion(Base):
    """Model version history."""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    version = Column(String)
    file_hash = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    changelog = Column(Text)
```

#### Database Manager
```python
# spicelab/models/database.py
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from typing import Optional

class ModelDatabase:
    """Central model database manager."""

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            db_path = Path.home() / ".spicelab" / "models.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

        # Session factory
        self.Session = sessionmaker(bind=self.engine)

    def add_model(
        self,
        vendor_name: str,
        part_number: str,
        category: str,
        file_path: Path,
        **kwargs
    ) -> Model:
        """Add model to database."""
        session = self.Session()

        try:
            # Get or create vendor
            vendor = session.query(Vendor).filter_by(name=vendor_name).first()
            if not vendor:
                vendor = Vendor(name=vendor_name)
                session.add(vendor)
                session.flush()

            # Create model
            model = Model(
                vendor_id=vendor.id,
                part_number=part_number,
                category=category,
                file_path=str(file_path.absolute()),
                filename=file_path.name,
                file_hash=self._compute_hash(file_path),
                file_size=file_path.stat().st_size,
                downloaded_at=datetime.datetime.utcnow(),
                **kwargs
            )

            session.add(model)
            session.commit()
            return model

        finally:
            session.close()

    def search(
        self,
        vendor: Optional[str] = None,
        category: Optional[str] = None,
        part_number: Optional[str] = None,
        verified_only: bool = False,
    ) -> list[Model]:
        """Search models."""
        session = self.Session()

        try:
            query = session.query(Model)

            if vendor:
                query = query.join(Vendor).filter(Vendor.name == vendor)
            if category:
                query = query.filter(Model.category == category)
            if part_number:
                query = query.filter(Model.part_number.contains(part_number))
            if verified_only:
                query = query.filter(Model.verified == True)

            return query.all()

        finally:
            session.close()

    def get_model(self, part_number: str, vendor: Optional[str] = None) -> Optional[Model]:
        """Get specific model."""
        session = self.Session()

        try:
            query = session.query(Model).filter(Model.part_number == part_number)

            if vendor:
                query = query.join(Vendor).filter(Vendor.name == vendor)

            return query.first()

        finally:
            session.close()

    @staticmethod
    def _compute_hash(file_path: Path) -> str:
        """Compute SHA256 hash."""
        import hashlib
        sha256 = hashlib.sha256()
        sha256.update(file_path.read_bytes())
        return sha256.hexdigest()
```

### 2. Automatic Model Downloader

**Strategy:** Web scraping and API integration for vendor websites.

#### Vendor-Specific Downloaders
```python
# spicelab/models/downloaders/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel

class DownloadResult(BaseModel):
    """Result of model download."""
    success: bool
    file_path: Path | None = None
    error: str | None = None
    metadata: dict = {}

class VendorDownloader(ABC):
    """Base class for vendor-specific downloaders."""

    vendor_name: str
    base_url: str

    @abstractmethod
    def search_models(self, part_number: str) -> list[dict]:
        """Search for models by part number."""
        pass

    @abstractmethod
    def download_model(self, part_number: str, cache_dir: Path) -> DownloadResult:
        """Download model file."""
        pass

# spicelab/models/downloaders/ti.py
import requests
from bs4 import BeautifulSoup

class TIDownloader(VendorDownloader):
    """Texas Instruments model downloader."""

    vendor_name = "TI"
    base_url = "https://www.ti.com"

    def search_models(self, part_number: str) -> list[dict]:
        """Search TI website for models."""
        # TI has a search API
        search_url = f"{self.base_url}/search?searchTerm={part_number}"

        response = requests.get(search_url, timeout=30)
        response.raise_for_status()

        # Parse results (simplified - actual implementation more complex)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for item in soup.find_all('div', class_='product-item'):
            results.append({
                'part_number': item.find('span', class_='part-number').text,
                'description': item.find('p', class_='description').text,
                'model_url': item.find('a', class_='model-link')['href'],
            })

        return results

    def download_model(self, part_number: str, cache_dir: Path) -> DownloadResult:
        """Download SPICE model from TI."""
        # Search for model
        results = self.search_models(part_number)

        if not results:
            return DownloadResult(
                success=False,
                error=f"Model {part_number} not found on TI website"
            )

        # Get model download link
        model_url = results[0]['model_url']
        full_url = f"{self.base_url}{model_url}"

        # Download
        response = requests.get(full_url, timeout=60)
        response.raise_for_status()

        # Save to cache
        file_path = cache_dir / f"{part_number}_ti.lib"
        file_path.write_bytes(response.content)

        return DownloadResult(
            success=True,
            file_path=file_path,
            metadata={
                'source_url': full_url,
                'description': results[0]['description'],
            }
        )

# Similar for ADI, Infineon, STM, NXP, etc.
```

#### Unified Downloader Manager
```python
# spicelab/models/downloader.py

class ModelDownloader:
    """Unified model downloader for all vendors."""

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".spicelab" / "cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Register vendor downloaders
        self.downloaders = {
            'TI': TIDownloader(),
            'ADI': ADIDownloader(),
            'Infineon': InfineonDownloader(),
            # ... more vendors
        }

    def download(
        self,
        part_number: str,
        vendor: str | None = None,
        force: bool = False
    ) -> DownloadResult:
        """Download model (with caching)."""

        # Check cache first
        if not force:
            cached = self._check_cache(part_number, vendor)
            if cached:
                return DownloadResult(success=True, file_path=cached)

        # Determine vendor
        if vendor:
            vendors_to_try = [vendor]
        else:
            vendors_to_try = list(self.downloaders.keys())

        # Try each vendor
        for vendor_name in vendors_to_try:
            downloader = self.downloaders.get(vendor_name)
            if not downloader:
                continue

            try:
                result = downloader.download_model(part_number, self.cache_dir)
                if result.success:
                    return result
            except Exception as e:
                # Try next vendor
                continue

        return DownloadResult(
            success=False,
            error=f"Could not find model {part_number} from any vendor"
        )

    def _check_cache(self, part_number: str, vendor: str | None) -> Path | None:
        """Check if model exists in cache."""
        pattern = f"{part_number}*.lib"
        if vendor:
            pattern = f"{part_number}_{vendor.lower()}.lib"

        matches = list(self.cache_dir.glob(pattern))
        return matches[0] if matches else None
```

### 3. Verilog-A Integration

**Strategy:** Use existing Verilog-A compiler (OpenVAF) or build parser.

```python
# spicelab/models/verilog_a.py
import subprocess
from pathlib import Path

class VerilogACompiler:
    """Compile Verilog-A to SPICE subcircuit."""

    def __init__(self, compiler_path: Path | None = None):
        # Try to find OpenVAF compiler
        if compiler_path is None:
            compiler_path = self._find_openvaf()

        self.compiler_path = compiler_path

    def compile(
        self,
        va_file: Path,
        output_file: Path | None = None,
        target: str = "ngspice"  # "ngspice", "xyce", "hspice"
    ) -> Path:
        """Compile Verilog-A to SPICE."""

        if output_file is None:
            output_file = va_file.with_suffix('.lib')

        # Run compiler
        cmd = [
            str(self.compiler_path),
            str(va_file),
            "-o", str(output_file),
            "--target", target,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Verilog-A compilation failed: {result.stderr}")

        return output_file

    @staticmethod
    def _find_openvaf() -> Path | None:
        """Find OpenVAF compiler."""
        # Check common locations
        candidates = [
            Path("/usr/local/bin/openvaf"),
            Path("/usr/bin/openvaf"),
            Path.home() / ".local/bin/openvaf",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Try which
        try:
            result = subprocess.run(["which", "openvaf"], capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except:
            pass

        return None

    def is_available(self) -> bool:
        """Check if compiler is available."""
        return self.compiler_path is not None and self.compiler_path.exists()
```

### 4. IBIS Model Support

**Strategy:** Parse IBIS files and convert to SPICE subcircuits.

```python
# spicelab/models/ibis.py
from pathlib import Path
import re

class IBISParser:
    """Parse IBIS (I/O Buffer Information Specification) files."""

    def __init__(self, ibis_file: Path):
        self.ibis_file = ibis_file
        self.data = self._parse()

    def _parse(self) -> dict:
        """Parse IBIS file."""
        content = self.ibis_file.read_text()

        data = {
            'component': self._parse_component(content),
            'models': self._parse_models(content),
            'pins': self._parse_pins(content),
        }

        return data

    def _parse_component(self, content: str) -> dict:
        """Parse [Component] section."""
        match = re.search(r'\[Component\]\s+(\w+)', content)
        if match:
            return {'name': match.group(1)}
        return {}

    def _parse_models(self, content: str) -> list[dict]:
        """Parse [Model] sections."""
        models = []

        # Find all [Model] sections
        for match in re.finditer(r'\[Model\]\s+(\w+)', content):
            model_name = match.group(1)
            # Extract model data (VI curves, etc.)
            models.append({'name': model_name})

        return models

    def _parse_pins(self, content: str) -> list[dict]:
        """Parse pin definitions."""
        pins = []

        # Parse pin table
        pin_section = re.search(r'\[Pin\](.*?)\n\n', content, re.DOTALL)
        if pin_section:
            for line in pin_section.group(1).split('\n'):
                if line.strip() and not line.startswith('|'):
                    parts = line.split()
                    if len(parts) >= 4:
                        pins.append({
                            'number': parts[0],
                            'name': parts[1],
                            'model': parts[2],
                            'r_pin': parts[3],
                        })

        return pins

    def to_spice(self, output_file: Path) -> Path:
        """Convert IBIS to SPICE subcircuit."""
        # Simplified conversion - real implementation much more complex
        spice = [
            f"* IBIS model converted from {self.ibis_file.name}",
            f".SUBCKT {self.data['component']['name']} {' '.join([p['name'] for p in self.data['pins']])}",
            "* TODO: Implement VI curve models",
            ".ENDS",
        ]

        output_file.write_text('\n'.join(spice))
        return output_file
```

### 5. Encrypted Model Support

**Strategy:** Decrypt using vendor-provided tools, cache decrypted version.

```python
# spicelab/models/encrypted.py

class EncryptedModelHandler:
    """Handle encrypted vendor models."""

    def __init__(self, decryption_key_dir: Path | None = None):
        if decryption_key_dir is None:
            decryption_key_dir = Path.home() / ".spicelab" / "keys"

        self.key_dir = decryption_key_dir
        self.key_dir.mkdir(parents=True, exist_ok=True)

    def is_encrypted(self, model_file: Path) -> bool:
        """Check if model is encrypted."""
        # Common encrypted model markers
        content = model_file.read_text(errors='ignore')

        markers = [
            '***ENCRYPTED***',
            'BEGIN ENCRYPTED',
            '.PROTECT',
            'ENCRYPTED MODEL',
        ]

        return any(marker in content for marker in markers)

    def decrypt(
        self,
        encrypted_file: Path,
        vendor: str,
        output_file: Path | None = None
    ) -> Path:
        """Decrypt model using vendor tool."""

        if output_file is None:
            output_file = encrypted_file.with_suffix('.decrypted.lib')

        # Find vendor decryption tool
        decryptor = self._find_decryptor(vendor)

        if not decryptor:
            raise RuntimeError(f"No decryptor found for vendor {vendor}")

        # Run decryption (vendor-specific)
        # Most vendors provide command-line tools
        cmd = [str(decryptor), str(encrypted_file), str(output_file)]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            raise RuntimeError(f"Decryption failed: {result.stderr}")

        return output_file

    def _find_decryptor(self, vendor: str) -> Path | None:
        """Find vendor decryption tool."""
        # Vendor-specific tool locations
        tools = {
            'Infineon': 'infineon_decrypt',
            'STM': 'stm_decrypt',
            # ... more vendors
        }

        tool_name = tools.get(vendor)
        if not tool_name:
            return None

        # Search PATH
        import shutil
        return shutil.which(tool_name)
```

### 6. Model Verification Suite

```python
# spicelab/models/verification.py

class ModelVerifier:
    """Verify model integrity and behavior."""

    def verify_syntax(self, model_file: Path) -> tuple[bool, list[str]]:
        """Verify SPICE syntax."""
        errors = []

        content = model_file.read_text()

        # Check for .SUBCKT/.ENDS matching
        subckt_count = len(re.findall(r'\.SUBCKT', content, re.IGNORECASE))
        ends_count = len(re.findall(r'\.ENDS', content, re.IGNORECASE))

        if subckt_count != ends_count:
            errors.append(f"Unmatched .SUBCKT/.ENDS ({subckt_count} vs {ends_count})")

        # Check for .MODEL definitions
        models = re.findall(r'\.MODEL\s+(\w+)', content, re.IGNORECASE)

        # Check for parameter syntax
        # ... more checks

        return (len(errors) == 0, errors)

    def verify_parameters(self, model_file: Path) -> tuple[bool, list[str]]:
        """Verify parameter values are reasonable."""
        errors = []

        # Extract parameters
        # Check ranges (e.g., VT should be 0.3-1.5V for MOSFETs)

        return (len(errors) == 0, errors)

    def verify_behavior(
        self,
        model_file: Path,
        test_circuit: Circuit,
        expected_results: dict
    ) -> tuple[bool, dict]:
        """Verify model behavior through simulation."""
        # Run test simulation
        # Compare against expected results
        # Return pass/fail + metrics

        return (True, {})
```

## Implementation Plan

### Week 1-2: Database Infrastructure
- [ ] Design database schema
- [ ] Implement SQLAlchemy models
- [ ] Create ModelDatabase class
- [ ] Add search and query methods
- [ ] Write database tests

### Week 3-4: Downloader Framework
- [ ] Create base VendorDownloader class
- [ ] Implement TI downloader
- [ ] Implement ADI downloader
- [ ] Implement Infineon downloader
- [ ] Add caching layer
- [ ] Write downloader tests

### Week 5-6: Verilog-A Integration
- [ ] Research Verilog-A compilers (OpenVAF)
- [ ] Implement VerilogACompiler wrapper
- [ ] Test with sample Verilog-A models
- [ ] Add error handling
- [ ] Document Verilog-A workflow

### Week 7-8: IBIS Support
- [ ] Implement IBIS parser
- [ ] Create IBIS to SPICE converter
- [ ] Test with vendor IBIS files
- [ ] Add validation

### Week 9: Encrypted Models
- [ ] Implement encrypted model detection
- [ ] Add vendor decryption tool integration
- [ ] Test with encrypted models
- [ ] Document NDA compliance

### Week 10: Model Verification
- [ ] Implement syntax verifier
- [ ] Add parameter validator
- [ ] Create behavioral test framework
- [ ] Build test circuit library

### Week 11-12: Model Population & Documentation
- [ ] Download 500+ vendor models
- [ ] Verify all models
- [ ] Populate database
- [ ] Write comprehensive documentation
- [ ] Create user guide

## Success Metrics

### Database
- [ ] **500+ vendor models** in database
- [ ] Search response time: <100ms
- [ ] Full-text search functional
- [ ] Metadata completeness: >90%

### Downloaders
- [ ] **5+ vendor** downloaders implemented
- [ ] Success rate: >95% for available models
- [ ] Cache hit rate: >80%

### Model Support
- [ ] Verilog-A compilation functional
- [ ] IBIS parsing for 100+ models
- [ ] Encrypted model support for 2+ vendors

### Quality
- [ ] All models syntax-verified
- [ ] **95%+ test coverage**
- [ ] Zero critical bugs

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Vendor website changes | HIGH | Automated monitoring, fallback to manual download |
| Model licensing issues | HIGH | Legal review, permissive licenses only |
| Encrypted model access | MEDIUM | Document vendor tool requirements |
| Verilog-A compiler availability | MEDIUM | Bundle compiler or provide installation guide |
| Model accuracy | HIGH | Verification suite, user feedback |

## Dependencies

**Required:**
- M3 (component library)
- M1 (validation framework)
- SQLAlchemy
- requests
- BeautifulSoup4

**Optional:**
- OpenVAF (Verilog-A)
- Vendor decryption tools

## Future Enhancements

- M12: Plugin system for custom model importers
- M14: PDK integration (SkyWater, GF)
- Cloud model repository
- Community model contributions

## References

- [TI SPICE Models](https://www.ti.com/support/spice-models.html)
- [Analog Devices Models](https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html)
- [IBIS Specification](https://ibis.org/)
- [OpenVAF](https://openvaf.semimod.de/)
- [Verilog-AMS](https://www.accellera.org/downloads/standards/vams)
