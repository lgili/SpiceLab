from __future__ import annotations

import importlib
import struct
from dataclasses import dataclass
from math import hypot
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trace:
    """Um traço: nome, unidade (quando existir) e vetor de valores (np.ndarray)."""

    name: str
    unit: str | None
    values: NDArray[Any]
    _complex: NDArray[Any] | None = None  # apenas AC: vetor complex

    def magnitude(self) -> NDArray[Any]:
        if self._complex is not None:
            return cast(NDArray[Any], np.abs(self._complex))
        return cast(NDArray[Any], np.abs(self.values))

    def real(self) -> NDArray[Any]:
        if self._complex is not None:
            return self._complex.real
        return self.values

    def imag(self) -> NDArray[Any]:
        if self._complex is not None:
            return self._complex.imag
        return np.zeros_like(self.values, dtype=float)

    def phase_deg(self) -> NDArray[Any]:
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

    def __init__(
        self,
        traces: list[Trace],
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if not traces:
            raise ValueError("TraceSet requires at least one trace")
        self._traces = traces
        self._by_name: dict[str, Trace] = {t.name: t for t in traces}
        self.meta: dict[str, Any] = meta or {}

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

    @classmethod
    def from_dataset(cls, dataset: Any) -> TraceSet:
        """Build a TraceSet from an xarray.Dataset-like object."""

        try:
            import numpy as _np
        except Exception as exc:  # pragma: no cover - numpy optional
            raise RuntimeError("numpy is required to convert dataset into TraceSet") from exc

        if not hasattr(dataset, "data_vars"):
            raise TypeError("Expected an xarray.Dataset-like object with 'data_vars'")

        coords = getattr(dataset, "coords", {})
        coord_names = ("time", "freq", "frequency", "index")
        coord_obj = None
        coord_key = None
        for name in coord_names:
            if name in coords:
                coord_obj = coords[name]
                coord_key = name
                break
        if coord_obj is not None:
            x_values = _np.asarray(getattr(coord_obj, "values", coord_obj), dtype=float)
            x_name = str(getattr(coord_obj, "name", None) or coord_key or "index")
        else:
            data_vars = list(getattr(dataset, "data_vars", {}))
            if not data_vars:
                raise ValueError("Dataset has no data variables to build TraceSet")
            first = _np.asarray(dataset[data_vars[0]].values)
            length = first.shape[0] if first.ndim > 0 else 1
            x_values = _np.arange(length, dtype=float)
            x_name = "index"

        x_values = _np.asarray(x_values, dtype=float).reshape(-1)
        axis_len = x_values.shape[0]
        traces: list[Trace] = [Trace(x_name, None, x_values)]

        for name, data in getattr(dataset, "data_vars", {}).items():
            arr = _np.asarray(data.values)
            if arr.ndim == 0:
                arr = _np.full((axis_len,), float(arr))
            elif arr.shape[0] != axis_len:
                try:
                    arr = arr.reshape(axis_len, -1)
                    arr = arr[:, 0]
                except Exception as exc:  # pragma: no cover - defensive reshape
                    raise ValueError(
                        f"Unable to align data variable '{name}' with independent axis"
                    ) from exc
            else:
                arr = arr.reshape(axis_len)
            arr = _np.real_if_close(arr)
            traces.append(Trace(str(name), None, _np.asarray(arr)))

        meta = dict(getattr(dataset, "attrs", {}))
        return cls(traces, meta=meta)


def _strip_prefix(line: str) -> str:
    """Remove espaços/tabs à esquerda (NGSpice ASCII costuma indentar)."""
    return line.lstrip(" \t")


def _parse_header(lines: list[str]) -> tuple[dict[str, Any], int]:
    """
    Lê cabeçalho até a linha 'Variables:' (inclusive).

    Retorna:
      - meta (dict com chaves úteis)
      - idx (próxima linha a ser lida)
    """
    meta: dict[str, Any] = {}
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
) -> tuple[NDArray[Any], list[NDArray[Any] | None]]:
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
    complex_arrays: list[NDArray[Any] | None] = []
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
    return TraceSet(traces, meta=meta)


# --- Detect/dispatch RAW (ASCII vs "Binary") ---


def _parse_binary_header_and_data(raw: bytes) -> tuple[dict[str, Any], bytes]:
    """Parse cabeçalho textual de um RAW binário (ngspice ou LTspice-like).

    Retorna meta e bloco binário puro.
    O cabeçalho termina na linha que inicia com 'Binary:' (incluída) —
    os bytes seguintes constituem os dados em float64 little endian.
    """
    marker = b"Binary:"  # LTspice usa 'Binary:'; ngspice similar
    pos = raw.find(marker)
    if pos == -1:
        raise ValueError("Binary RAW: missing 'Binary:' marker")
    # avançar até final da linha
    line_end = raw.find(b"\n", pos)
    if line_end == -1:
        raise ValueError("Binary RAW header malformed (no newline after 'Binary:')")
    header_bytes = raw[: line_end + 1]
    data_bytes = raw[line_end + 1 :]
    text = header_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    meta: dict[str, Any] = {}
    backannotations: list[str] = []
    for ln in lines:
        s = ln.strip()
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
        elif s.startswith("Offset:"):
            off_txt = s.split("Offset:", 1)[1].strip()
            try:
                meta["offset"] = float(off_txt)
            except ValueError:
                meta["offset"] = off_txt
        elif s.startswith("Command:"):
            meta["command"] = s.split("Command:", 1)[1].strip()
        elif s.startswith("Backannotation:"):
            backannotations.append(s.split("Backannotation:", 1)[1].strip())
        elif s.startswith("Variables:"):
            # variáveis ficam nas linhas subsequentes até 'Binary:'
            continue
    if backannotations:
        meta["backannotation"] = backannotations
    # segunda passada para pegar variáveis com índice
    vars_meta: list[tuple[str, str | None]] = []
    in_vars = False
    for ln in lines:
        st = ln.strip()
        if st.startswith("Variables:"):
            in_vars = True
            continue
        if st.startswith("Binary:"):
            break
        if in_vars and st:
            parts = st.split()
            if len(parts) >= 2 and parts[0].isdigit():
                # idx name type
                name = parts[1]
                unit = parts[2] if len(parts) >= 3 else None
                vars_meta.append((name, unit))
    if "nvars" not in meta or "npoints" not in meta:
        # Some minimal 'Binary:' markers (used only to signal unsupported early format) lack counts.
        raise NotImplementedError("Binary RAW: missing counts (nvars/npoints) in header")
    # store separately under private key (not part of original RAW header schema)
    meta["_vars_meta"] = vars_meta
    return meta, data_bytes


def _parse_binary_payload(meta: dict[str, Any], blob: bytes) -> TraceSet:
    # The meta dict may contain auxiliary list for '_vars_meta'; cast as needed.
    nvars = int(meta["nvars"])  # meta keys validated earlier
    npoints = int(meta["npoints"])  # meta keys validated earlier
    flags = str(meta.get("flags", "")).lower()
    vars_meta = meta.get("_vars_meta")
    if not isinstance(vars_meta, list):  # pragma: no cover - defensive
        raise ValueError("Binary RAW: internal vars metadata missing")

    is_complex = "complex" in flags
    # Determine expected total values and scalar size (8 or 4)
    if is_complex:
        total_values = npoints * (1 + 2 * (nvars - 1))
    else:
        total_values = npoints * nvars
    scalar_size: int | None = None
    for cand in (8, 4):
        if len(blob) >= total_values * cand:
            scalar_size = cand
            break
    if scalar_size is None:
        raise ValueError("Binary RAW: insufficient data length for counts declared")

    raw_vals = blob[: total_values * scalar_size]
    fmt = f"<{total_values}{'d' if scalar_size == 8 else 'f'}"
    values = struct.unpack(fmt, raw_vals)

    # Build layout helpers shared by both real/complex paths
    def build_point_major(
        vals: tuple[float, ...],
    ) -> tuple[list[list[float]], list[list[complex] | None]]:
        cols_pm: list[list[float]] = [[] for _ in range(nvars)]
        ccols_pm: list[list[complex] | None] = [None for _ in range(nvars)]
        itp = iter(vals)
        if is_complex:
            for _ in range(npoints):
                cols_pm[0].append(next(itp))
                for vi in range(1, nvars):
                    re = next(itp)
                    im = next(itp)
                    cols_pm[vi].append((re**2 + im**2) ** 0.5)
                    lst = ccols_pm[vi]
                    if lst is None:
                        lst = []
                        ccols_pm[vi] = lst
                    lst.append(complex(re, im))
        else:
            for _ in range(npoints):
                for vi in range(nvars):
                    cols_pm[vi].append(next(itp))
        return cols_pm, ccols_pm

    def build_variable_major(
        vals: tuple[float, ...],
    ) -> tuple[list[list[float]], list[list[complex] | None]]:
        cols_vm: list[list[float]] = [[] for _ in range(nvars)]
        ccols_vm: list[list[complex] | None] = [None for _ in range(nvars)]
        if is_complex:
            idx = 0
            cols_vm[0] = list(values[idx : idx + npoints])
            idx += npoints
            for vi in range(1, nvars):
                re_block = values[idx : idx + npoints]
                idx += npoints
                im_block = values[idx : idx + npoints]
                idx += npoints
                mag_block: list[float] = []
                cc: list[complex] = []
                for r, im in zip(re_block, im_block, strict=False):
                    mag_block.append((r**2 + im**2) ** 0.5)
                    cc.append(complex(r, im))
                cols_vm[vi] = mag_block
                ccols_vm[vi] = cc
        else:
            idx = 0
            for vi in range(nvars):
                cols_vm[vi] = list(values[idx : idx + npoints])
                idx += npoints
        return cols_vm, ccols_vm

    cols, complex_cols = build_point_major(values)
    time_like = cols[0]
    monotonic = all(time_like[i] <= time_like[i + 1] for i in range(len(time_like) - 1))
    if not monotonic:
        cols, complex_cols = build_variable_major(values)

    traces: list[Trace] = []
    import numpy as _np

    for j, (name, unit) in enumerate(vars_meta):
        c_arr = None
        if is_complex and j > 0:
            cc = complex_cols[j]
            if cc is not None:
                c_arr = _np.array(cc, dtype=complex)
        traces.append(
            Trace(
                name=name,
                unit=unit,
                values=_np.array(cols[j], dtype=float),
                _complex=c_arr,
            )
        )
    return TraceSet(traces, meta=meta)


def parse_ngspice_raw(path: str) -> TraceSet:
    """Dispatcher para RAW ASCII ou binário.

    Suporta formato binário de linha simples (LTspice/NGSpice) para casos reais e complexos.
    """
    with open(path, "rb") as f:
        blob = f.read()
    # Heurísticas:
    # 1. ASCII header: contém 'Binary:' sem null interleaving.
    # 2. UTF-16 (wide) header: contém padrão intercalado
    #    B\x00i\x00n\x00a... ou grande densidade de nulls.
    head_scan = blob[:8192]
    wide_pattern = b"B\x00i\x00n\x00a\x00r\x00y\x00:\x00"
    is_ascii_binary = b"Binary:" in head_scan or b"binary" in head_scan.lower()
    is_wide_binary = wide_pattern in head_scan

    if is_ascii_binary:
        meta, data_bytes = _parse_binary_header_and_data(blob)
        return _parse_binary_payload(meta, data_bytes)
    if is_wide_binary:
        # Localizar fim da linha 'Binary:' (padrão newline UTF-16 '\n\x00')
        marker_pos = head_scan.find(wide_pattern)
        newline_pat = b"\n\x00"
        nl_pos = head_scan.find(newline_pat, marker_pos)
        if nl_pos == -1:
            raise ValueError("UTF-16 binary RAW: newline after Binary: not found")
        header_bytes = blob[: nl_pos + 2]
        # Remover nulls para obter texto ascii simplificado
        ascii_header = header_bytes[::2].decode("utf-8", errors="ignore")
        # Reaproveitar parser ascii-binary: reconstruir bytes artificiais com ascii header e newline
        rebuilt = ascii_header.encode("utf-8") + blob[nl_pos + 2 :]
        meta, data_bytes = _parse_binary_header_and_data(rebuilt)
        return _parse_binary_payload(meta, data_bytes)
    # caso contrário tratar como ASCII
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
        out.append(TraceSet(traces, meta=meta))
        # avançar: i += i1 + npoints ... mas já usamos slices; então mova i para frente
        # tenta achar próximo 'Title:' após o bloco atual
        # heurística simples: move i até encontrar próxima 'Title:' ou EOF
        k = i + i1 + npoints + 4  # + margem
        i = max(i + 1, k)
    return out
