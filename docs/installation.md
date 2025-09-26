# Installation

Circuit Toolkit is currently published under the package name `spicelab`.
Use Python 3.10 or newer.

## Using uv (recommended)
```bash
uv venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\activate.ps1  # Windows PowerShell
uv pip install -e .
```

Add extras as needed:
```bash
uv pip install -e '.[viz]'      # Plotly helpers
uv pip install -e '.[data]'     # xarray / polars shortcuts
```

## Using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Install optional dev tools:
```bash
pip install pytest pytest-cov mypy ruff
```

## SPICE engines
The orchestrator auto-detects engines via environment variables or the system
PATH. Install whichever simulators you need:

- **NGSpice** – `brew install ngspice` · `sudo apt install ngspice`
- **LTspice** – Analog Devices installer (macOS via `brew install --cask ltspice`)
- **Xyce** – https://xyce.sandia.gov/

Set `SPICELAB_NGSPICE`, `SPICELAB_LTSPICE`, or `SPICELAB_XYCE` to absolute paths
if binaries are not on PATH. Missing engines raise informative errors with
install hints, and tests tagged `@pytest.mark.engine` auto-skip when the binary
is absent.
