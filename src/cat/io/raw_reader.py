from __future__ import annotations

import importlib
from dataclasses import dataclass
from math import hypot
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Um traço: nome, unidade (quando existir) e vetor de valores (np.ndarray)."""

    name: str
    unit: str | None
    values: np.ndarray
    _complex: np.ndarray | None = None  # apenas AC: vetor complex

    def magnitude(self) -> np.ndarray:
        if self._complex is not None:
            return cast(np.ndarray, np.abs(self._complex))
        return cast(np.ndarray, np.abs(self.values))

    def real(self) -> np.ndarray:
        if self._complex is not None:
            return self._complex.real
        return self.values

    def imag(self) -> np.ndarray:
        if self._complex is not None:
            return self._complex.imag
        return np.zeros_like(self.values, dtype=float)

    def phase_deg(self) -> np.ndarray:
        if self._complex is not None:
            return np.angle(self._complex, deg=True)
        return np.zeros_like(self.values, dtype=float)


class TraceSet:
    """
    Conjunto de traços indexado por nome. O primeiro traço é o eixo X (time/freq).

    Acesso:
        ts["V(out)"] -> Trace
        ts.x -> Trace (primeira coluna)
        ts.names -> lista de nomes
        ts.to_dataframe() -> pandas.DataFrame (se pandas instalado)
    """

    def __init__(self, traces: list[Trace]) -> None:
        if not traces:
            raise ValueError("TraceSet requires at least one trace")
        self._traces = traces
        self._by_name: dict[str, Trace] = {t.name: t for t in traces}

        # valida tamanhos
        n = len(self._traces[0].values)
        for t in self._traces[1:]:
            if len(t.values) != n:
                raise ValueError("All traces must have same length")

    @property
    def x(self) -> Trace:
        return self._traces[0]

    @property
    def names(self) -> list[str]:
        return [t.name for t in self._traces]

    def __getitem__(self, key: str) -> Trace:
        try:
            return self._by_name[key]
        except KeyError as e:
            raise KeyError(f"Trace '{key}' not found. Available: {self.names}") from e

    def to_dataframe(self) -> Any:
        # Evita depender de stubs do pandas: import dinâmico via importlib, tipado como Any
        try:
            pd: Any = importlib.import_module("pandas")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for to_dataframe()") from exc
        data = {t.name: t.values for t in self._traces}
        return pd.DataFrame(data)


def _strip_prefix(line: str) -> str:
    """Remove espaços/tabs à esquerda (NGSpice ASCII costuma indentar)."""
    return line.lstrip(" \t")


def _parse_header(lines: list[str]) -> tuple[dict[str, int | str], int]:
    """
    Lê cabeçalho até a linha 'Variables:' (inclusive).

    Retorna:
      - meta (dict com chaves úteis)
      - idx (próxima linha a ser lida)
    """
    meta: dict[str, int | str] = {}
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("Title:"):
            meta["title"] = s.split("Title:", 1)[1].strip()
        elif s.startswith("Date:"):
            meta["date"] = s.split("Date:", 1)[1].strip()
        elif s.startswith("Plotname:"):
            meta["plotname"] = s.split("Plotname:", 1)[1].strip()
        elif s.startswith("Flags:"):
            meta["flags"] = s.split("Flags:", 1)[1].strip()
        elif s.startswith("No. Variables:"):
            meta["nvars"] = int(s.split("No. Variables:", 1)[1].strip())
        elif s.startswith("No. Points:"):
            meta["npoints"] = int(s.split("No. Points:", 1)[1].strip())
        elif s.startswith("Variables:"):
            i += 1
            break
        i += 1
    if "nvars" not in meta or "npoints" not in meta:
        raise ValueError("Invalid NGSpice ASCII RAW: missing counts")
    return meta, i


def _parse_variables(
    lines: list[str],
    start: int,
    nvars: int,
) -> tuple[list[tuple[str, str | None]], int]:
    """
    Lê bloco Variables.

    Retorna lista [(nome, unidade), ...] e índice da próxima linha (após 'Values:').
    Formato típico por linha: "<idx> <name> <type>" (ex.: "0 time time" ou "2 v(out) voltage").
    """
    vars_meta: list[tuple[str, str | None]] = []
    i = start
    for _ in range(nvars):
        s = _strip_prefix(lines[i])
        parts = s.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid variable line: {s!r}")
        name = parts[1]
        unit: str | None = parts[2] if len(parts) >= 3 else None
        vars_meta.append((name, unit))
        i += 1
    # encontrar "Values:"
    while i < len(lines) and not lines[i].strip().startswith("Values:"):
        i += 1
    if i >= len(lines):
        raise ValueError("Invalid NGSpice ASCII RAW: missing 'Values:' section")
    i += 1  # pula 'Values:'
    return vars_meta, i


def _to_val(tok: str) -> tuple[float, complex | None]:
    """
    Converte token NGSpice em (valor_float, valor_complex|None).
    - Em análises AC (Flags: complex) os valores vêm como "re,im".
    """
    if "," in tok:
        re_s, im_s = tok.split(",", 1)
        re = float(re_s)
        im = float(im_s)
        return float(hypot(re, im)), complex(re, im)
    val = float(tok)
    return val, None


def _parse_values(
    lines: list[str],
    start: int,
    nvars: int,
    npoints: int,
) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """
    Lê matriz (npoints x nvars) de valores e retorna também colunas complexas (se houver).
    """
    data = np.empty((npoints, nvars), dtype=float)
    complex_cols: list[list[complex] | None] = [None] * nvars

    i = start
    for row in range(npoints):
        if i >= len(lines):
            raise ValueError("Unexpected EOF while reading values")
        head = _strip_prefix(lines[i]).split()
        if not head:
            raise ValueError("Invalid Values entry (empty line)")
        try:
            _ = int(head[0])  # índice do ponto
        except ValueError as exc:
            raise ValueError(f"Invalid point index line: {lines[i]!r}") from exc
        tokens: list[str] = head[1:]
        i += 1
        while len(tokens) < nvars:
            if i >= len(lines):
                raise ValueError("Unexpected EOF while reading value tokens")
            tokens.extend(_strip_prefix(lines[i]).split())
            i += 1

        vals_row: list[float] = []
        for j, tok in enumerate(tokens[:nvars]):
            val, cval = _to_val(tok)
            vals_row.append(val)
            if cval is not None:
                lst = complex_cols[j]
                if lst is None:
                    lst = []
                    complex_cols[j] = lst
                lst.append(cval)
        data[row, :] = vals_row

    # converte listas para np.ndarray
    complex_arrays: list[np.ndarray | None] = []
    for col in complex_cols:
        if col is None:
            complex_arrays.append(None)
        else:
            complex_arrays.append(np.array(col, dtype=complex))

    return data, complex_arrays


def parse_ngspice_ascii_raw(path: str) -> TraceSet:
    """
    Parser robusto para NGSpice ASCII RAW.

    Retorna TraceSet onde a primeira coluna é o eixo X (tipicamente 'time' ou 'frequency').
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    meta, i0 = _parse_header(lines)
    nvars = int(meta["nvars"])
    npoints = int(meta["npoints"])
    vars_meta, i1 = _parse_variables(lines, i0, nvars)
    data, complex_cols = _parse_values(lines, i1, nvars, npoints)

    traces: list[Trace] = []
    for j, (name, unit) in enumerate(vars_meta):
        traces.append(
            Trace(
                name=name,
                unit=unit,
                values=data[:, j].copy(),
                _complex=complex_cols[j],
            )
        )
    return TraceSet(traces)


# --- Detect/dispatch RAW (ASCII vs "Binary") ---


def parse_ngspice_raw(path: str) -> TraceSet:
    """
    Dispatcher: se for ASCII, usa parse_ngspice_ascii_raw; se for 'Binary', tenta
    fallback convertendo para ASCII não invasivo (a depender da geração do arquivo).
    Por ora, detecta 'Binary' e dispara erro com mensagem clara.
    """
    with open(path, "rb") as f:
        head = f.read(256)
    if b"Binary:" in head or b"binary" in head:
        # Implementação binária completa é extensa; orientar uso de ASCII no runner.
        raise NotImplementedError(
            "Binary RAW not supported yet. Configure NGSpice to write ASCII RAW "
            "(set filetype=ascii)."
        )
    # ASCII
    return parse_ngspice_ascii_raw(path)


# --- Multi-plot (ex.: .step nativo em ASCII gera vários blocos) ---


def parse_ngspice_ascii_raw_multi(path: str) -> list[TraceSet]:
    """
    Lê um arquivo ASCII com múltiplos plots (p.ex. .step nativo) e retorna
    uma lista de TraceSet (um por bloco).
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    i = 0
    out: list[TraceSet] = []
    while i < len(lines):
        # procurar início de um bloco (Title:/Plotname:/Variables:)
        # Reutiliza as funções privadas para cada bloco
        # pular linhas vazias
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break
        # precisa ver se há um cabeçalho válido
        try:
            meta, i0 = _parse_header(lines[i:])
            nvars = int(meta["nvars"])
            npoints = int(meta["npoints"])
            vars_meta, i1 = _parse_variables(lines[i:], i0, nvars)
            data, complex_cols = _parse_values(lines[i:], i1, nvars, npoints)
        except Exception:
            # se não conseguiu, avança uma linha e tenta de novo
            i += 1
            continue

        traces: list[Trace] = []
        for j, (name, unit) in enumerate(vars_meta):
            traces.append(
                Trace(name=name, unit=unit, values=data[:, j].copy(), _complex=complex_cols[j])
            )
        out.append(TraceSet(traces))
        # avançar: i += i1 + npoints ... mas já usamos slices; então mova i para frente
        # tenta achar próximo 'Title:' após o bloco atual
        # heurística simples: move i até encontrar próxima 'Title:' ou EOF
        k = i + i1 + npoints + 4  # + margem
        i = max(i + 1, k)
    return out
