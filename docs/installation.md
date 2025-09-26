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

### libngspice (shared-library callbacks)

The co-simulation API (`NgSpiceSharedSimulator`) relies on the dynamic
`libngspice` library so Python callbacks can observe every transient step and
drive external sources. Install the shared library alongside the CLI:

| Platform | Command(s) | Default library location |
|----------|------------|---------------------------|
| macOS (Homebrew) | `brew install libngspice` | `$(brew --prefix libngspice)/lib/libngspice.dylib` |
| Debian/Ubuntu | `sudo apt install libngspice0-dev` | `/usr/lib/x86_64-linux-gnu/libngspice.so` |
| Arch Linux | `sudo pacman -S libngspice` | `/usr/lib/libngspice.so` |
| Windows | Extract `ngspice.dll` from the official installer and place it next to your Python project |

Point Circuit Toolkit at the library with `SPICELAB_NGSPICE_SHARED`:

```bash
export SPICELAB_NGSPICE_SHARED="$(brew --prefix libngspice)/lib/libngspice.dylib"
# Windows (PowerShell)
# setx SPICELAB_NGSPICE_SHARED "C:\\path\\to\\ngspice.dll"
```

Verify that callbacks can load the library before running co-simulation code:

```bash
python - <<'PY'
from spicelab.spice.ngspice_shared_backend import load_default_backend
load_default_backend()  # raises EngineSharedLibraryNotFound on failure
print("libngspice ready")
PY
```

If the shared library is missing the toolkit transparently falls back to the
process-based adapter (`engine="ngspice"`), which still runs simulations but
does not expose live callbacks or external sources.
