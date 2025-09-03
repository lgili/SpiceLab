# CAT Examples

Exemplos mínimos usando `cat` com NGSpice.

## Pré-requisitos

- `ngspice` no PATH (macOS via `brew install ngspice`, Linux via apt/dnf/pacman; Windows: instalar NGSpice e adicionar ao PATH).
- Ambiente com a lib instalada (no repo raiz):
  ```bash
  uv run --active pip install -e .
  uv run --active pip install matplotlib pandas

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
