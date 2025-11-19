# SpiceLab Project Context

## Purpose

SpiceLab is a typed Python library for SPICE circuit simulation that aims to become the **industry-standard open-source solution** for circuit design, analysis, and verification.

### Vision
To provide a powerful, flexible, and production-grade Python interface for SPICE simulation that rivals commercial tools while remaining open-source and community-driven.

### Mission
- Enable engineers and researchers to design circuits programmatically with type-safe APIs
- Support multiple SPICE engines (NGSpice, LTspice, Xyce, QSPICE) through a unified interface
- Provide advanced analysis capabilities (Monte Carlo, optimization, DoE, yield analysis)
- Foster an ecosystem of components, models, and plugins
- Maintain production-grade quality (95%+ test coverage, strict type checking, comprehensive docs)

## Tech Stack

### Core
- **Python 3.10+** - Modern Python with latest type hints
- **Pydantic v2** - Data validation and settings management
- **MyPy** - Strict static type checking (no `Any` allowed)
- **Ruff** - Fast linting and formatting

### Data Processing
- **xarray** - N-dimensional labeled datasets (primary)
- **pandas** - Tabular data (export/bridge)
- **polars** - Fast DataFrames (export/bridge)
- **numpy** - Numerical computing foundation

### Simulation Engines
- **NGSpice** - Default open-source engine (CLI + libngspice shared)
- **LTspice** - Analog Devices CLI interface
- **Xyce** - Sandia high-performance simulator
- **QSPICE** - (planned) New free simulator from Qorvo

### Testing & Quality
- **pytest** - Test framework
- **pytest-cov** - Coverage reporting (target: 95%+)
- **Hypothesis** - Property-based testing
- **pytest-benchmark** - Performance regression testing

### Visualization (optional)
- **Plotly** - Interactive plots (Bode, transient, etc.)
- **Graphviz** - Circuit topology diagrams
- **Rich/Textual** - Terminal UI components

### Documentation
- **MkDocs Material** - Documentation site
- **mkdocstrings** - API reference generation
- **Jupyter** - Interactive examples and tutorials

## Project Conventions

### Code Style

#### Python Style
- **Line length:** 100 characters (Ruff configured)
- **Imports:** isort compatible, grouped (stdlib, third-party, local)
- **Formatting:** Ruff format (Black-compatible)
- **Docstrings:** NumPy style with type hints in signatures
- **Type hints:** Required for all public APIs, strict MyPy compliance

#### Naming Conventions
```python
# Classes: PascalCase
class Circuit: ...
class Resistor: ...

# Functions/methods: snake_case
def run_simulation(...): ...
def build_netlist(...): ...

# Constants: UPPER_SNAKE_CASE
GND = Net("0")
DEFAULT_TOLERANCE = 0.05

# Private: _leading_underscore
def _internal_helper(...): ...
class _CacheManager: ...

# Protocol/ABC: Explicit suffix or inheritance
class Simulator(Protocol): ...
class ComponentBase(ABC): ...
```

#### File Organization
```
spicelab/
├── core/           # Core abstractions (Circuit, Component, Net, etc.)
├── engines/        # Simulator backends
├── analysis/       # Analysis utilities (sweeps, Monte Carlo, measurements)
├── io/             # File readers/writers (RAW, log, HDF5)
├── library/        # Component catalog and registry
├── models/         # Device model management
├── optimization/   # Optimization and DoE
├── viz/            # Visualization helpers
├── cli/            # Command-line interfaces
└── utils/          # Utilities (units, logging, etc.)
```

### Architecture Patterns

#### 1. Protocol-Based Design
Use structural typing (Protocols) over inheritance for flexibility:
```python
from typing import Protocol

class Simulator(Protocol):
    def run(self, circuit: Circuit, analyses: list[Analysis]) -> ResultHandle: ...
    def features(self) -> EngineFeatures: ...
```

#### 2. Immutable Data Structures
Use frozen dataclasses for hashable, thread-safe objects:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Net:
    name: str | None = None
```

#### 3. Pydantic Models for Validation
Use Pydantic for data validation and serialization:
```python
from pydantic import BaseModel, Field

class Analysis(BaseModel):
    type: Literal["op", "dc", "ac", "tran", "noise"]
    params: dict[str, Any]

    @field_validator("params")
    def validate_params(cls, v, info):
        # Custom validation logic
        ...
```

#### 4. Lazy Evaluation
Defer expensive operations until results are needed:
```python
class ResultHandle(Protocol):
    def dataset(self) -> xr.Dataset:
        """Lazy load simulation results."""
        ...
```

#### 5. Deterministic Hashing
Enable caching with content-based hashing:
```python
def circuit_hash(circuit: Circuit, extra: dict | None = None) -> str:
    """Generate stable hash for circuit + context."""
    netlist = circuit.build_netlist()
    canonical = json.dumps({"netlist": netlist, "extra": extra}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

#### 6. Factory Pattern
Use registries for extensible component creation:
```python
COMPONENT_REGISTRY: dict[str, ComponentFactory] = {}

def register_component(name: str, factory: ComponentFactory):
    COMPONENT_REGISTRY[name] = factory

def create_component(name: str, *args, **kwargs) -> Component:
    return COMPONENT_REGISTRY[name](*args, **kwargs)
```

### Testing Strategy

#### Coverage Requirements
- **Minimum:** 90% branch coverage
- **Target:** 95%+ branch coverage
- **Critical paths:** 100% coverage (netlist generation, simulation execution)

#### Test Organization
```
tests/
├── unit/               # Fast, isolated unit tests
├── integration/        # Multi-component integration tests
├── engines/            # Engine-specific tests (marked with @pytest.mark.engine)
├── regression/         # Regression test suite
├── property/           # Property-based tests (Hypothesis)
└── benchmarks/         # Performance benchmarks
```

#### Test Markers
```python
@pytest.mark.unit           # Fast unit tests (no external deps)
@pytest.mark.integration    # Integration tests
@pytest.mark.engine         # Requires SPICE engine installed
@pytest.mark.slow           # Tests >1 second
@pytest.mark.benchmark      # Performance benchmarks
```

#### Test Naming
```python
def test_resistor_spice_card_generation():
    """Test that Resistor generates correct SPICE card."""
    ...

def test_circuit_netlist_with_floating_nodes_raises():
    """Test that circuits with floating nodes raise ValidationError."""
    ...
```

#### Property-Based Testing
Use Hypothesis for edge case discovery:
```python
from hypothesis import given, strategies as st

@given(resistance=st.floats(min_value=1e-6, max_value=1e12))
def test_resistor_accepts_valid_resistance(resistance):
    r = Resistor("R1", resistance)
    assert r.resistance == resistance
```

### Git Workflow

#### Branch Strategy
- **main** - Stable, production-ready code
- **develop** - Integration branch for features
- **feature/M{N}-{description}** - Feature branches for milestones
- **fix/{issue-number}-{description}** - Bug fixes
- **refactor/{description}** - Refactoring work

#### Commit Conventions (Conventional Commits)
```
feat(core): add lazy component instantiation
fix(engines): handle ngspice stderr parsing edge case
docs(tutorial): add Monte Carlo example
test(analysis): add property tests for sweep grid
perf(io): optimize RAW file parsing with memmap
refactor(library): extract component registry to separate module
chore(ci): update GitHub Actions to use uv
```

#### PR Requirements
- ✅ All tests pass (pytest)
- ✅ Coverage does not decrease
- ✅ MyPy strict passes
- ✅ Ruff linting passes
- ✅ Documentation updated (if API changes)
- ✅ Changelog entry added
- ✅ At least 1 approval from maintainer

#### Release Process
1. Update CHANGELOG.md
2. Bump version in pyproject.toml (semver)
3. Create git tag (e.g., `v0.4.0`)
4. Push tag → GitHub Actions builds and publishes to PyPI
5. Create GitHub release with changelog excerpt

## Domain Context

### SPICE Simulation Fundamentals

**SPICE (Simulation Program with Integrated Circuit Emphasis)** is a circuit simulation standard. Key concepts:

- **Netlist:** Text representation of circuit topology and component values
- **Nodes:** Connection points between components (numbered or named)
- **Ground (GND):** Reference node (always node 0)
- **Analyses:** Operating point (op), DC sweep (dc), AC frequency (ac), Transient (tran), Noise
- **Directives:** `.model`, `.subckt`, `.include`, `.param`, `.lib`
- **Probes:** Signals to measure (voltages, currents)

### Circuit Design Patterns

#### Passive Filters
- RC lowpass/highpass
- LC resonant circuits
- Multi-stage filters (Butterworth, Chebyshev)

#### Amplifiers
- Common emitter/source/collector
- Differential pairs
- Operational amplifier circuits

#### Power Electronics
- Linear regulators (LDO)
- Switching regulators (Buck, Boost, Buck-Boost)
- Power factor correction (PFC)

#### Mixed-Signal
- ADC/DAC interfaces
- Sample-and-hold circuits
- Comparators and level shifters

### Key Metrics and Measurements

- **Gain:** Output/Input ratio (dB or V/V)
- **Bandwidth:** Frequency range (-3dB points)
- **Phase Margin:** Stability metric for feedback systems
- **PSRR:** Power supply rejection ratio
- **THD:** Total harmonic distortion
- **Settling Time:** Time to reach steady state
- **Overshoot:** Peak excursion beyond final value
- **Slew Rate:** Maximum rate of change

## Important Constraints

### Performance Constraints
- **Netlist building:** Must handle 10,000+ components in <1 second
- **Simulation caching:** Must use deterministic hashing (no time-based cache invalidation)
- **Memory usage:** Must support streaming for large result files (>1GB)
- **Parallel execution:** Must support multi-core Monte Carlo runs

### Compatibility Constraints
- **Python version:** 3.10+ (use latest type hints)
- **SPICE engines:** Must support NGSpice 35+, LTspice XVII+, Xyce 7.0+
- **Operating systems:** Linux, macOS, Windows (native, no WSL requirement)
- **Netlist compatibility:** Must parse/generate SPICE3f5 compatible netlists

### Security Constraints
- **No code execution:** Never `eval()` or `exec()` user input
- **Path traversal:** Validate all file paths (no `..` in paths)
- **Model sources:** Warn when downloading models from untrusted sources

### Licensing Constraints
- **Core library:** MIT license (permissive, commercial-friendly)
- **Dependencies:** Must use compatible licenses (MIT, BSD, Apache 2.0)
- **Vendor models:** Respect vendor redistribution terms

## External Dependencies

### SPICE Engines (Required - User Installed)
- **NGSpice:** https://ngspice.sourceforge.io/
  - CLI: `ngspice` binary
  - Shared library: `libngspice.so/.dylib/.dll` (optional, for callbacks)
- **LTspice:** https://www.analog.com/ltspice
  - CLI: `LTspice` (macOS/Linux) or `XVIIx64.exe` (Windows)
- **Xyce:** https://xyce.sandia.gov/
  - CLI: `Xyce` binary

### Python Dependencies (Core)
See `pyproject.toml` for exact versions. Key dependencies:
- `numpy>=1.26` - Numerical arrays
- `pandas>=2.1` - DataFrame compatibility
- `xarray>=2025.6.1` - N-D datasets
- `pydantic>=2.7` - Validation

### Optional Dependencies
- `[viz]`: plotly, kaleido (visualization)
- `[data]`: polars (fast DataFrames)
- `[ai]`: openai (LLM integration)
- `[opt]`: scipy (optimization, signal processing)

### Environment Variables
- `SPICELAB_NGSPICE` - Path to ngspice binary
- `SPICELAB_NGSPICE_SHARED` - Path to libngspice shared library
- `SPICELAB_LTSPICE` - Path to LTspice CLI
- `SPICELAB_XYCE` - Path to Xyce binary
- `SPICELAB_ENGINE` - Default engine name
- `SPICELAB_CACHE_DIR` - Simulation cache directory (default: `~/.cache/spicelab`)

## Development Roadmap

See `CRITICAL_ANALYSIS.md` for comprehensive analysis.

### Current Phase: FOUNDATIONS (Months 1-6)
Focus on stability, performance, and developer experience to establish production-grade quality.

### Milestones Overview
- **M1-M6:** Foundations (stability, performance, components, DX, docs, community)
- **M7-M12:** Advanced features (measurements, models, optimization, I/O, UX, plugins)
- **M13-M18:** Enterprise (mixed-signal, PDK, distributed, yield, schematic, RBAC)
- **M19-M20:** Industry leadership (ecosystem maturity, advanced R&D)

### Success Metrics (24 months)
- **Adoption:** 5,000 GitHub stars, 200k PyPI downloads/month
- **Quality:** 99% test coverage, zero critical bugs
- **Ecosystem:** 1,000 components, 500 vendor models
- **Community:** 100 contributors, 100 companies using

---

**Last Updated:** 2025-01-19
**Project Status:** Active Development
**Maintainer:** Luiz Carlos Gili (@lgili)
