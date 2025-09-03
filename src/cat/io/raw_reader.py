from __future__ import annotations

import importlib
from dataclasses import dataclass
from math import hypot
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Um traço: nome, unidade (quando existir) e vetor de valores (np.ndarray)."""

    name: str
    unit: str | None
    values: np.ndarray


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


def _to_float(tok: str) -> float:
    """
    Converte token NGSpice em float. Em análises AC (Flags: complex) os valores são "re,im".
    Para esses, retornamos a **magnitude**: sqrt(re^2 + im^2).
    """
    if "," in tok:
        re_s, im_s = tok.split(",", 1)
        re = float(re_s)
        im = float(im_s)
        return float(hypot(re, im))
    return float(tok)


def _parse_values(
    lines: list[str],
    start: int,
    nvars: int,
    npoints: int,
) -> np.ndarray:
    """
    Lê matriz (npoints x nvars) de valores. Cada ponto começa com a linha contendo
    o índice do ponto, seguida pelos nvars valores (podem quebrar em várias linhas).
    """
    data = np.empty((npoints, nvars), dtype=float)
    i = start
    for row in range(npoints):
        if i >= len(lines):
            raise ValueError("Unexpected EOF while reading values")
        head = _strip_prefix(lines[i]).split()
        if not head:
            raise ValueError("Invalid Values entry (empty line)")
        # token 0 = índice (inteiro)
        try:
            _ = int(head[0])
        except ValueError as exc:
            raise ValueError(f"Invalid point index line: {lines[i]!r}") from exc
        tokens: list[str] = head[1:]  # valores já na primeira linha (se houver)
        i += 1
        # continuar coletando até termos nvars tokens
        while len(tokens) < nvars:
            if i >= len(lines):
                raise ValueError("Unexpected EOF while reading value tokens")
            more = _strip_prefix(lines[i]).split()
            tokens.extend(more)
            i += 1
        # preencher linha
        try:
            data[row, :] = [_to_float(tok) for tok in tokens[:nvars]]
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Non-numeric token in values: {tokens[:nvars]!r}") from exc
    return data


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
    data = _parse_values(lines, i1, nvars, npoints)

    traces = [
        Trace(name=vars_meta[j][0], unit=vars_meta[j][1], values=data[:, j].copy())
        for j in range(len(vars_meta))
    ]
    return TraceSet(traces)
