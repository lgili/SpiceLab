from __future__ import annotations

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")  # renderiza sem display quando salvar figuras
import matplotlib.pyplot as plt  # noqa: E402


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(fig: Any, name: str) -> str:
    ensure_dir(".")
    out = os.path.abspath(name)
    fig.savefig(out, dpi=150)
    print(f"[saved] {out}")
    plt.close(fig)
    return out
