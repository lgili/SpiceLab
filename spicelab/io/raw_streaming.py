"""Memory-efficient streaming RAW file reader.

This module provides streaming access to large SPICE RAW files using:
- Memory-mapped I/O for efficient file access
- Chunked reading with generators for low memory footprint
- Streaming conversion to HDF5 for compressed storage

Supports both NGSpice and LTspice RAW formats (binary and ASCII).
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class VariableInfo:
    """Metadata for a single variable in a RAW file."""

    name: str
    unit: str | None
    index: int
    dtype: np.dtype[Any]
    is_complex: bool = False
    offset: int = 0  # Byte offset in data section (for binary)


@dataclass
class RAWHeader:
    """Parsed header from a RAW file."""

    title: str
    date: str
    plotname: str
    flags: str
    nvars: int
    npoints: int
    variables: list[VariableInfo]
    is_binary: bool
    is_complex: bool
    data_offset: int  # Byte offset where data starts
    encoding: str = "utf-8"
    raw_meta: dict[str, Any] = field(default_factory=dict)


class StreamingRAWReader:
    """Memory-efficient chunked RAW file reader.

    Uses memory-mapped I/O to read large RAW files without loading the
    entire file into memory. Supports streaming access via generators.

    Example::

        with StreamingRAWReader(Path("large_sim.raw")) as reader:
            # Get file info
            print(f"Variables: {reader.variable_names}")
            print(f"Points: {reader.n_points}")

            # Stream a single variable
            for chunk in reader.iter_variable("V(out)", chunk_size=10000):
                process(chunk)

            # Stream all variables together
            for chunk_dict in reader.iter_all_variables(chunk_size=10000):
                for name, data in chunk_dict.items():
                    process(name, data)

            # Convert to HDF5
            reader.to_hdf5(Path("output.h5"), compression="gzip")

    """

    def __init__(self, raw_file: Path | str, chunk_size: int = 10_000) -> None:
        """Initialize the streaming reader.

        Args:
            raw_file: Path to the RAW file
            chunk_size: Default number of points per chunk

        """
        self.raw_file = Path(raw_file)
        self.chunk_size = chunk_size
        self._file: Any = None
        self._mmap: mmap.mmap | None = None
        self._header: RAWHeader | None = None
        self._raw_bytes: bytes | None = None

    def __enter__(self) -> StreamingRAWReader:
        """Enter context and open file with memory mapping."""
        self._file = open(self.raw_file, "rb")
        try:
            # Memory-map the file for efficient random access
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except (ValueError, OSError):
            # mmap can fail on empty files or special files
            # Fall back to reading entire file
            self._file.seek(0)
            self._raw_bytes = self._file.read()
            self._mmap = None

        self._header = self._parse_header()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and close file."""
        if self._mmap is not None:
            self._mmap.close()
        if self._file is not None:
            self._file.close()
        self._mmap = None
        self._file = None
        self._raw_bytes = None

    @property
    def header(self) -> RAWHeader:
        """Get parsed header (must be in context)."""
        if self._header is None:
            raise RuntimeError("Reader not in context. Use 'with' statement.")
        return self._header

    @property
    def n_points(self) -> int:
        """Number of data points in the file."""
        return self.header.npoints

    @property
    def n_variables(self) -> int:
        """Number of variables in the file."""
        return self.header.nvars

    @property
    def variable_names(self) -> list[str]:
        """List of variable names."""
        return [v.name for v in self.header.variables]

    @property
    def is_complex(self) -> bool:
        """Whether the data contains complex values (AC analysis)."""
        return self.header.is_complex

    def get_variable_info(self, name: str) -> VariableInfo:
        """Get metadata for a variable by name."""
        for var in self.header.variables:
            if var.name == name:
                return var
        available = ", ".join(self.variable_names)
        raise KeyError(f"Variable '{name}' not found. Available: {available}")

    def _get_raw_data(self) -> bytes | memoryview:
        """Get raw data as bytes or memoryview."""
        if self._mmap is not None:
            return memoryview(self._mmap)
        elif self._raw_bytes is not None:
            return self._raw_bytes
        else:
            raise RuntimeError("No data source available")

    def _parse_header(self) -> RAWHeader:
        """Parse the RAW file header to extract metadata."""
        raw = self._get_raw_data()
        head_scan = bytes(raw[:16384]) if len(raw) > 16384 else bytes(raw)

        # Detect format: binary vs ASCII, UTF-8 vs UTF-16
        is_wide = b"B\x00i\x00n\x00a\x00r\x00y\x00:\x00" in head_scan
        is_ascii_binary = b"Binary:" in head_scan or b"binary:" in head_scan.lower()

        if is_wide:
            return self._parse_wide_binary_header(raw)
        elif is_ascii_binary:
            return self._parse_binary_header(raw)
        else:
            return self._parse_ascii_header(raw)

    def _parse_binary_header(self, raw: bytes | memoryview) -> RAWHeader:
        """Parse header from a binary RAW file (NGSpice/LTspice format)."""
        # Find Binary: marker
        raw_bytes = bytes(raw)
        marker = b"Binary:"
        marker_pos = raw_bytes.find(marker)
        if marker_pos == -1:
            raise ValueError("Binary RAW: missing 'Binary:' marker")

        # Find end of line after Binary:
        newline_pos = raw_bytes.find(b"\n", marker_pos)
        if newline_pos == -1:
            raise ValueError("Binary RAW: malformed header")

        header_text = raw_bytes[:newline_pos].decode("utf-8", errors="ignore")
        data_offset = newline_pos + 1

        return self._parse_header_text(header_text, data_offset, is_binary=True)

    def _parse_wide_binary_header(self, raw: bytes | memoryview) -> RAWHeader:
        """Parse header from UTF-16 binary RAW file (some LTspice versions)."""
        raw_bytes = bytes(raw)
        wide_marker = b"B\x00i\x00n\x00a\x00r\x00y\x00:\x00"
        marker_pos = raw_bytes.find(wide_marker)
        if marker_pos == -1:
            raise ValueError("Wide binary RAW: missing marker")

        # Find UTF-16 newline
        newline_pat = b"\n\x00"
        newline_pos = raw_bytes.find(newline_pat, marker_pos)
        if newline_pos == -1:
            raise ValueError("Wide binary RAW: malformed header")

        # Convert UTF-16 header to ASCII by removing null bytes
        header_bytes = raw_bytes[: newline_pos + 2]
        header_text = header_bytes[::2].decode("utf-8", errors="ignore")
        data_offset = newline_pos + 2

        return self._parse_header_text(header_text, data_offset, is_binary=True)

    def _parse_ascii_header(self, raw: bytes | memoryview) -> RAWHeader:
        """Parse header from ASCII RAW file."""
        raw_bytes = bytes(raw)
        text = raw_bytes.decode("utf-8", errors="ignore")
        lines = text.splitlines()

        # Find Values: marker for data start
        data_start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Values:"):
                data_start_line = i + 1
                break

        # Calculate byte offset for Values: line
        data_offset = sum(len(line) + 1 for line in lines[:data_start_line])

        header_text = "\n".join(lines[:data_start_line])
        return self._parse_header_text(header_text, data_offset, is_binary=False)

    def _parse_header_text(self, header_text: str, data_offset: int, is_binary: bool) -> RAWHeader:
        """Parse header text into RAWHeader structure."""
        lines = header_text.splitlines()
        meta: dict[str, Any] = {}
        variables: list[VariableInfo] = []
        in_variables = False

        for line in lines:
            s = line.strip()
            if not s:
                continue

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
                in_variables = True
                continue
            elif s.startswith("Binary:") or s.startswith("Values:"):
                in_variables = False
                continue
            elif in_variables:
                # Parse variable line: "index name type"
                parts = s.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    idx = int(parts[0])
                    name = parts[1]
                    unit = parts[2] if len(parts) >= 3 else None
                    variables.append(
                        VariableInfo(
                            name=name,
                            unit=unit,
                            index=idx,
                            dtype=np.dtype("float64"),
                            is_complex=False,
                        )
                    )

        flags = meta.get("flags", "").lower()
        is_complex = "complex" in flags

        # Update variable complex flag
        for var in variables:
            # First variable (time/frequency) is always real
            var.is_complex = is_complex and var.index > 0
            var.dtype = np.dtype("complex128") if var.is_complex else np.dtype("float64")

        return RAWHeader(
            title=meta.get("title", ""),
            date=meta.get("date", ""),
            plotname=meta.get("plotname", ""),
            flags=meta.get("flags", ""),
            nvars=meta.get("nvars", len(variables)),
            npoints=meta.get("npoints", 0),
            variables=variables,
            is_binary=is_binary,
            is_complex=is_complex,
            data_offset=data_offset,
            raw_meta=meta,
        )

    def _read_binary_chunk(
        self,
        start_point: int,
        n_points: int,
    ) -> dict[str, NDArray[Any]]:
        """Read a chunk of binary data for all variables."""
        header = self.header
        raw = self._get_raw_data()
        nvars = header.nvars
        is_complex = header.is_complex

        # Calculate values per point
        if is_complex:
            # First var is real, rest are complex (2 floats each)
            values_per_point = 1 + 2 * (nvars - 1)
        else:
            values_per_point = nvars

        # Try float64 first, then float32
        scalar_size = 8
        total_data_values = header.npoints * values_per_point
        expected_size = total_data_values * scalar_size
        available = len(raw) - header.data_offset

        if available < expected_size:
            scalar_size = 4
            expected_size = total_data_values * scalar_size
            if available < expected_size:
                raise ValueError(
                    f"Binary RAW: insufficient data. "
                    f"Expected {expected_size} bytes, got {available}"
                )

        # Calculate chunk offsets
        chunk_start = header.data_offset + start_point * values_per_point * scalar_size
        chunk_values = n_points * values_per_point
        chunk_bytes = chunk_values * scalar_size

        # Read chunk
        chunk_data = bytes(raw[chunk_start : chunk_start + chunk_bytes])
        fmt = f"<{chunk_values}{'d' if scalar_size == 8 else 'f'}"
        values = struct.unpack(fmt, chunk_data)

        # Parse into columns
        result: dict[str, NDArray[Any]] = {}
        if is_complex:
            # First variable is real
            time_values = [values[i * values_per_point] for i in range(n_points)]
            result[header.variables[0].name] = np.array(time_values, dtype=np.float64)

            # Rest are complex
            for vi in range(1, nvars):
                var = header.variables[vi]
                complex_values = []
                for pt in range(n_points):
                    base = pt * values_per_point + 1 + 2 * (vi - 1)
                    re = values[base]
                    im = values[base + 1]
                    complex_values.append(complex(re, im))
                result[var.name] = np.array(complex_values, dtype=np.complex128)
        else:
            for vi, var in enumerate(header.variables):
                var_values = [values[pt * nvars + vi] for pt in range(n_points)]
                result[var.name] = np.array(var_values, dtype=np.float64)

        return result

    def _read_ascii_chunk(
        self,
        start_point: int,
        n_points: int,
    ) -> dict[str, NDArray[Any]]:
        """Read a chunk of ASCII data for all variables."""
        header = self.header
        raw = self._get_raw_data()
        text = bytes(raw).decode("utf-8", errors="ignore")
        lines = text.splitlines()

        # Find Values: line
        values_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Values:"):
                values_line = i + 1
                break

        # Parse data points
        nvars = header.nvars
        result: dict[str, list[Any]] = {var.name: [] for var in header.variables}

        i = values_line
        point_idx = 0
        while i < len(lines) and point_idx < start_point + n_points:
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Check if this is a point header (starts with index)
            parts = line.split()
            if not parts:
                i += 1
                continue

            try:
                _ = int(parts[0])  # Point index
            except ValueError:
                i += 1
                continue

            # Collect tokens for this point
            tokens = parts[1:]
            i += 1
            while len(tokens) < nvars and i < len(lines):
                tokens.extend(lines[i].split())
                i += 1

            if point_idx >= start_point:
                # Parse values
                for vi, var in enumerate(header.variables):
                    if vi < len(tokens):
                        tok = tokens[vi]
                        if "," in tok:
                            re_s, im_s = tok.split(",", 1)
                            result[var.name].append(complex(float(re_s), float(im_s)))
                        else:
                            result[var.name].append(float(tok))

            point_idx += 1

        # Convert to numpy arrays
        return {
            name: np.array(values, dtype=header.variables[self.variable_names.index(name)].dtype)
            for name, values in result.items()
        }

    def iter_variable(
        self,
        name: str,
        chunk_size: int | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> Generator[NDArray[Any], None, None]:
        """Iterate over a single variable in chunks.

        Args:
            name: Variable name to read
            chunk_size: Points per chunk (default: self.chunk_size)
            start: Starting point index
            end: Ending point index (exclusive, default: all)

        Yields:
            numpy arrays of data for each chunk

        """
        _ = self.get_variable_info(name)  # Validate name exists
        chunk_size = chunk_size or self.chunk_size
        end = end if end is not None else self.n_points

        current = start
        while current < end:
            n_points = min(chunk_size, end - current)

            if self.header.is_binary:
                chunk_data = self._read_binary_chunk(current, n_points)
            else:
                chunk_data = self._read_ascii_chunk(current, n_points)

            yield chunk_data[name]
            current += n_points

    def iter_all_variables(
        self,
        chunk_size: int | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> Generator[dict[str, NDArray[Any]], None, None]:
        """Iterate over all variables in chunks.

        Args:
            chunk_size: Points per chunk (default: self.chunk_size)
            start: Starting point index
            end: Ending point index (exclusive, default: all)

        Yields:
            dict mapping variable names to numpy arrays for each chunk

        """
        chunk_size = chunk_size or self.chunk_size
        end = end if end is not None else self.n_points

        current = start
        while current < end:
            n_points = min(chunk_size, end - current)

            if self.header.is_binary:
                chunk_data = self._read_binary_chunk(current, n_points)
            else:
                chunk_data = self._read_ascii_chunk(current, n_points)

            yield chunk_data
            current += n_points

    def read_variable(self, name: str) -> NDArray[Any]:
        """Read an entire variable into memory.

        For large files, consider using iter_variable() instead.

        Args:
            name: Variable name to read

        Returns:
            numpy array with all values

        """
        chunks = list(self.iter_variable(name))
        return np.concatenate(chunks)

    def read_all(self) -> dict[str, NDArray[Any]]:
        """Read all variables into memory.

        For large files, consider using iter_all_variables() instead.

        Returns:
            dict mapping variable names to numpy arrays

        """
        result: dict[str, list[NDArray[Any]]] = {name: [] for name in self.variable_names}

        for chunk_dict in self.iter_all_variables():
            for name, data in chunk_dict.items():
                result[name].append(data)

        return {name: np.concatenate(chunks) for name, chunks in result.items()}

    def to_hdf5(
        self,
        output_file: Path | str,
        compression: str = "gzip",
        compression_level: int = 4,
        chunk_size: int | None = None,
    ) -> None:
        """Stream RAW file to HDF5 with compression.

        Args:
            output_file: Path for output HDF5 file
            compression: Compression algorithm ('gzip', 'lzf', or None)
            compression_level: Compression level (1-9 for gzip)
            chunk_size: Points per chunk for streaming

        """
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("h5py is required for HDF5 export") from exc

        output_file = Path(output_file)
        chunk_size = chunk_size or self.chunk_size
        header = self.header

        with h5py.File(output_file, "w") as h5:
            # Store metadata
            h5.attrs["title"] = header.title
            h5.attrs["date"] = header.date
            h5.attrs["plotname"] = header.plotname
            h5.attrs["flags"] = header.flags
            h5.attrs["nvars"] = header.nvars
            h5.attrs["npoints"] = header.npoints

            # Create datasets for each variable
            datasets: dict[str, Any] = {}
            for var in header.variables:
                ds = h5.create_dataset(
                    var.name,
                    shape=(header.npoints,),
                    dtype=var.dtype,
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None,
                    chunks=(min(chunk_size, header.npoints),),
                )
                if var.unit:
                    ds.attrs["unit"] = var.unit
                datasets[var.name] = ds

            # Stream data
            offset = 0
            for chunk_dict in self.iter_all_variables(chunk_size=chunk_size):
                chunk_len = len(next(iter(chunk_dict.values())))
                for name, data in chunk_dict.items():
                    datasets[name][offset : offset + chunk_len] = data
                offset += chunk_len

    def get_memory_estimate(self) -> dict[str, int]:
        """Estimate memory usage for different read strategies.

        Returns:
            dict with 'full_load' and 'chunked' memory estimates in bytes

        """
        header = self.header
        bytes_per_point = sum(16 if var.is_complex else 8 for var in header.variables)

        return {
            "full_load": header.npoints * bytes_per_point,
            "chunked": self.chunk_size * bytes_per_point,
            "n_chunks": (header.npoints + self.chunk_size - 1) // self.chunk_size,
        }


@contextmanager
def open_raw_streaming(
    raw_file: Path | str, chunk_size: int = 10_000
) -> Iterator[StreamingRAWReader]:
    """Context manager for streaming RAW file access.

    Example::

        with open_raw_streaming("large.raw") as reader:
            for chunk in reader.iter_variable("V(out)"):
                process(chunk)

    """
    reader = StreamingRAWReader(raw_file, chunk_size)
    with reader:
        yield reader


def stream_raw_to_hdf5(
    raw_file: Path | str,
    output_file: Path | str,
    compression: str = "gzip",
    chunk_size: int = 10_000,
) -> None:
    """Convenience function to stream a RAW file to HDF5.

    Args:
        raw_file: Input RAW file path
        output_file: Output HDF5 file path
        compression: Compression algorithm ('gzip', 'lzf', or None)
        chunk_size: Points per chunk for streaming

    """
    with open_raw_streaming(raw_file, chunk_size) as reader:
        reader.to_hdf5(output_file, compression=compression, chunk_size=chunk_size)


def get_raw_info(raw_file: Path | str) -> RAWHeader:
    """Get header information from a RAW file without loading data.

    Args:
        raw_file: Path to RAW file

    Returns:
        RAWHeader with file metadata

    """
    with open_raw_streaming(raw_file) as reader:
        return reader.header
