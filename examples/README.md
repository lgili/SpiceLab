# CAT Examples

Minimal examples showing how to use the library with NGSpice.

## Prerequisites

- `ngspice` available on PATH (macOS: `brew install ngspice`, Linux: apt/dnf/pacman, Windows: install and add to PATH).
- Install the package in editable mode from the repo root and required plotting libraries:
  ```bash
  uv run --active pip install -e .
  uv run --active pip install matplotlib pandas
  # optional Plotly demo
  uv run --active pip install -e '.[viz]'
  ```

## How to run

From the `examples/` directory, run any script with the active uv environment. Examples:

- Run the getting-started tutorial (AC, DC, TRAN):
  ```bash
  uv run --active python getting_started.py
  ```

- Run AC Bode plot example:
  ```bash
  uv run --active python ac_bode.py
  ```

- Run DC sweep example:
  ```bash
  uv run --active python dc_sweep.py
  ```

- Run parameter step grid example:
  ```bash
  uv run --active python step_sweep_grid.py
  ```

- Run Monte Carlo examples:
  ```bash
  uv run --active python monte_carlo_rc.py
  uv run --active python monte_carlo_demo.py
  ```

- Run the Plotly visualization demo (requires the ``viz`` extra):
  ```bash
  uv run --active python plotly_viz_demo.py --outdir ./plotly_output
  ```

Each script saves PNG graphs into the current working directory (or into a script-specific output folder).

---

### `examples/_common.py`
This module provides a small helper `savefig(fig, path)` used by the examples to save figures without requiring a display.
# CAT Examples

Exemplos mínimos usando `cat` com NGSpice.

## Pré-requisitos

- `ngspice` no PATH (macOS via `brew install ngspice`, Linux via apt/dnf/pacman; Windows: instalar NGSpice e adicionar ao PATH).
- Ambiente com a lib instalada (no repo raiz):
  ```bash
  uv run --active pip install -e .
  uv run --active pip install matplotlib pandas
  uv run --active pip install -e '.[viz]'  # opcional, para o demo Plotly

  Como rodar

Dentro da pasta examples:

# OP + TRAN de um RC
uv run --active python rc_op_tran.py

# Bode (AC) com fonte de 1 V
uv run --active python ac_bode.py

# Varredura DC (0→5V)
uv run --active python dc_sweep.py

# STEP param em grade (produto cartesiano)
uv run --active python step_sweep_grid.py

# Monte Carlo (normal 5% no R)
uv run --active python monte_carlo_rc.py

# Demo Plotly (HTML interativo)
uv run --active python plotly_viz_demo.py --outdir ./plotly_output

Cada script gera gráficos .png na pasta.

---

### `examples/_common.py`
```python
from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")  # renderiza sem precisar de display
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(fig: Any, name: str) -> str:
    ensure_dir(".")
    out = os.path.abspath(name)
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")
    plt.close(fig)
    return out
