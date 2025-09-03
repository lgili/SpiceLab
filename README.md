# PyCircuitKit (CAT) — Circuit Analysis Toolkit

Modern, strongly-typed Python toolkit to **define**, **simulate** (.OP / .TRAN / .AC) and **analyze** electronic circuits with a clean, Pythonic API. CAT targets real engineering workflows: parameter sweeps, **Monte Carlo**, worst‑case (soon), and painless result handling in NumPy/Pandas.

> **Status:** MVP scaffold — strongly-typed Circuit DSL (Style 1: Ports & Nets), NGSpice (CLI) smoke runner for `.op`, utilities for E-series rounding and basic RC design helpers. Roadmap includes AC/DC/TRAN, parsers, Monte-Carlo & Worst-Case, DSL Styles 2 & 3, and LTspice adapter.

---

## ✨ Features (MVP)

- **Zero-string connectivity:** connect **Port ↔ Net** objects (type-safe, IDE-friendly).
- **Core components:** `Resistor`, `Capacitor`, `Vdc` (more soon).
- **Netlist builder:** generate SPICE netlists with topology validation.
- **NGSpice (CLI) runner:** headless `.op` smoke execution (skips automatically if NGSpice is missing).
- **Utilities for design:**
  - **E-series** enumerator & rounding (E12/E24/E48/E96).
  - **RC low-pass** design helper by target `f_c`.

### What’s new in this MVP
- **.TRAN end‑to‑end**: run transient with NGSpice and parse traces into a `TraceSet`.
- **Monte Carlo (beta)**: vary component values by distributions; stack results into Pandas.
- **LTspice netlist import (beta)**: parse SPICE netlists exported from LTspice to a `Circuit` and run with NGSpice.

---

## 🧰 Installation

### Requirements
- Python **3.10+**
- pip / virtualenv (or **uv** / **poetry**)
- **NGSpice** (recommended for simulation; optional for building & unit tests)

> LTspice support is planned. For now, NGSpice is the reference backend.

### macOS

```bash
# 1) Install NGSpice
brew install ngspice

# 2) (Recommended) Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell so `uv` is on PATH

# 3) Clone and set up the project
git clone https://github.com/<your-org>/pycircuitkit.git
cd pycircuitkit
uv sync --all-extras --dev
```


Alternative without uv:
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[opt]"  # installs project (editable) + optional deps if defined
pip install -r dev-requirements.txt  # if you keep one (pytest, mypy, ruff, etc.)

Linux (Ubuntu/Debian)
# 1) Install NGSpice
sudo apt update
sudo apt install -y ngspice

# 2) Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3) Clone and set up
git clone https://github.com/<your-org>/pycircuitkit.git
cd pycircuitkit
uv sync --all-extras --dev

Windows
# 1) Install NGSpice
# Download installer from: https://ngspice.sourceforge.io/download.html
# Add the ngspice bin folder (e.g. C:\Program Files\Spice64\bin) to your PATH.

# 2) Install uv (recommended)
# In PowerShell:
irm https://astral.sh/uv/install.ps1 | iex

# 3) Clone and set up
git clone https://github.com/<your-org>/pycircuitkit.git
cd pycircuitkit
uv sync --all-extras --dev

Alternative without uv:
py -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e ".[opt]"
pip install -r dev-requirements.txt

If ngspice is not on your PATH, CAT’s tests that require it will auto-skip.


## 🚀 Quick Start (User Guide)

### 1) Define a circuit (Style 1: Ports & Nets) and run **.TRAN**
```python
from cat.core.circuit import Circuit
from cat.core.net import Net, GND
from cat.core.components import Vdc, Resistor, Capacitor
from cat.analysis import TRAN

# Circuit: RC low‑pass
c = Circuit("rc_lowpass")
vin  = Net("vin")
vout = Net("vout")

V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")

c.add(V1, R1, C1)
c.connect(V1.ports[0], vin)
c.connect(V1.ports[1], GND)
c.connect(R1.ports[0], vin)
c.connect(R1.ports[1], vout)
c.connect(C1.ports[0], vout)
c.connect(C1.ports[1], GND)

res = TRAN("10us", "5ms").run(c)
ts = res.traces
print("traces:", ts.names)
```

### 2) **Monte Carlo** on the same circuit
```python
from cat.analysis import OP, monte_carlo, NormalPct

# Varia apenas R1 com 5% (sigma) — 16 amostras
mc = monte_carlo(
    circuit=c,
    mapping={R1: NormalPct(0.05)},
    n=16,
    analysis_factory=lambda: OP(),
    seed=123,
)
# Empilha em DataFrame (se quiser)
from cat.analysis import stack_runs_to_df
print(stack_runs_to_df(mc.runs).head())
```

### 3) Importar netlist do **LTspice** e simular
Export no LTspice: *View → SPICE Netlist* → salve como `.cir`/`.net`.
```python
from cat.io.ltspice_parser import from_ltspice_file
from cat.analysis import TRAN

c2 = from_ltspice_file("./my_filter.cir")
res2 = TRAN("1us", "2ms").run(c2)
print(res2.traces.names)
```

### 4) Utilities (E‑series & RC helper)
```python
from cat.utils.e_series import round_to_series
from cat.utils.synth import design_rc_lowpass

print(round_to_series(12700, "E96"))
print(design_rc_lowpass(fc=159.155, prefer_R=True, series="E24"))
```

## 📦 Project Layout
src/cat/
  core/          # Nets, Ports, Components, Circuit, NetlistBuilder
  spice/         # Simulator adapters (MVP: ngspice_cli)
  io/            # Parsers (stubs in MVP; AC/DC/TRAN parsers next)
  utils/         # e_series, synth (design helpers), logging
tests/           # pytest suite (unit + smoke if ngspice available)


Planned:
	•	analysis/ for OP/AC/DC/TRAN engines
	•	dsl/flow.py (Style 2, operators for series/parallel)
	•	dsl/schematic.py (Style 3, context manager “schematic”)
	•	spice/ltspice.py adapter

⸻

🛠️ Developer Guide

Tooling

We ship configs for ruff (lint & format), mypy (strict typing), pytest (tests & coverage), and pre-commit.

Install dev tools:
uv sync --all-extras --dev
pre-commit install

Run checks locally:
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q

CI (GitHub Actions) runs the same steps. The NGSpice smoke test is skipped if ngspice isn’t installed on the runner.

Adding Components

Create a class in cat/core/components.py:

from dataclasses import dataclass
from .net import Port, PortRole
from .components import Component

@dataclass
class Inductor(Component):
    def __post_init__(self) -> None:
        object.__setattr__(self, "_ports",
            (Port(self, "a", PortRole.NODE), Port(self, "b", PortRole.NODE)))

    def spice_card(self, net_of) -> str:
        a, b = self.ports
        return f"L{self.ref} {net_of(a)} {net_of(b)} {self.value}"

	•	Keep ports strongly-typed (PortRole).
	•	Implement spice_card(net_of) to render the final card.
	•	Add tests in tests/ (netlist presence & validation).

Coding Standards
	•	Type hints everywhere (mypy --strict must pass).
	•	Keep functions pure where possible; avoid global state.
	•	Small, composable modules. Prefer dataclasses for DTOs.
	•	Use Ruff to keep formatting consistent.

Tests
	•	Unit tests for:
	•	Netlist generation & validation
	•	Component cards
	•	Utils (E-series, synth)
	•	Integration (smoke) tests for NGSpice (skippable if missing).

Run:

uv run pytest -q --cov=cat --cov-report=term-missing

⚙️ Configuration

Environment variables (reserved, to be expanded):
	•	CAT_SPICE_NGSPICE — override path to ngspice executable.
	•	CAT_LOG_LEVEL — set logger level (INFO, DEBUG, …) for CAT.

Example:
export CAT_SPICE_NGSPICE=/opt/tools/ngspice/bin/ngspice
export CAT_LOG_LEVEL=DEBUG

🧭 Roadmap (Short)
	1.	Engines: OP/AC/DC/TRAN with unified result object (TraceSet) and Pandas export.
	2.	Parsers: NGSpice ASCII/RAW → structured traces (V(node), I(R1)).
	3.	DSL Style 2: Operators (>> series, | parallel) for fast topologies.
	4.	DSL Style 3: with schematic(): context and <</>> wiring sugar.
	5.	Sweeps: Native .STEP + Python multi-param sweeps.
	6.	Monte-Carlo: Distributions, samplers, parallel execution, metrics to DataFrame.
	7.	Worst-Case: Corners + constrained optimization (scipy).
	8.	LTspice adapter: CLI backend + RAW normalizer.
	9.	Docs website: MkDocs Material with runnable examples.

⸻

❓ Troubleshooting
	•	ngspice executable not found
Install it and ensure it’s on PATH (see OS-specific steps).
Quick check: which ngspice (Linux/macOS) or where ngspice (Windows).
	•	Unconnected port: X.Y
You created a component but didn’t connect all ports. Wire every Port to a Net or another Port.
	•	CI fails on mypy/ruff
Run the commands locally (uv run ruff check ., uv run mypy src) and fix warnings before pushing.

⸻

🤝 Contributing
	•	Fork → feature branch → PR
	•	Keep PRs focused and covered by tests.
	•	Follow the code style and typing rules.
	•	Add/Update docs or examples where relevant.

Good first issues: adding basic components (L, I sources, diodes), small utils, unit tests.

⸻

📄 License

MIT — see LICENSE.

⸻

🔗 Citation / Acknowledgments

CAT builds on concepts used across the SPICE ecosystem and Python scientific stack (NumPy, Pandas). We’ll add proper acknowledgments as dependencies and integrations grow.
