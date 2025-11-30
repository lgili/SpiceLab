"""Tests for the streaming RAW file reader."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from spicelab.io.raw_streaming import (
    RAWHeader,
    StreamingRAWReader,
    VariableInfo,
    get_raw_info,
    open_raw_streaming,
    stream_raw_to_hdf5,
)

if TYPE_CHECKING:
    pass


FIXTURES = Path(__file__).parent / "fixtures"


def _check_h5py_available() -> bool:
    """Check if h5py is available."""
    try:
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False


class TestVariableInfo:
    """Tests for VariableInfo dataclass."""

    def test_create_real_variable(self) -> None:
        """Test creating a real variable info."""
        var = VariableInfo(
            name="V(out)",
            unit="voltage",
            index=1,
            dtype=np.dtype("float64"),
            is_complex=False,
        )
        assert var.name == "V(out)"
        assert var.unit == "voltage"
        assert var.index == 1
        assert not var.is_complex

    def test_create_complex_variable(self) -> None:
        """Test creating a complex variable info."""
        var = VariableInfo(
            name="V(out)",
            unit="voltage",
            index=1,
            dtype=np.dtype("complex128"),
            is_complex=True,
        )
        assert var.is_complex
        assert var.dtype == np.dtype("complex128")


class TestRAWHeader:
    """Tests for RAWHeader dataclass."""

    def test_create_header(self) -> None:
        """Test creating a RAW header."""
        header = RAWHeader(
            title="Test Circuit",
            date="2025-01-01",
            plotname="Transient Analysis",
            flags="real",
            nvars=3,
            npoints=100,
            variables=[],
            is_binary=False,
            is_complex=False,
            data_offset=0,
        )
        assert header.title == "Test Circuit"
        assert header.nvars == 3
        assert header.npoints == 100
        assert not header.is_binary


class TestStreamingRAWReaderASCII:
    """Tests for streaming ASCII RAW files."""

    @pytest.fixture
    def tran_file(self) -> Path:
        """Path to transient RAW fixture."""
        return FIXTURES / "rc_tran_ng.raw"

    @pytest.fixture
    def ac_file(self) -> Path:
        """Path to AC RAW fixture."""
        return FIXTURES / "rc_ac_ng.raw"

    def test_context_manager(self, tran_file: Path) -> None:
        """Test using reader as context manager."""
        with StreamingRAWReader(tran_file) as reader:
            assert reader.n_points == 3
            assert reader.n_variables == 3

    def test_header_parsing(self, tran_file: Path) -> None:
        """Test header is correctly parsed."""
        with StreamingRAWReader(tran_file) as reader:
            header = reader.header
            assert header.title == "RC transient ng"
            assert header.plotname == "Transient Analysis"
            assert header.nvars == 3
            assert header.npoints == 3
            assert not header.is_binary

    def test_variable_names(self, tran_file: Path) -> None:
        """Test variable names are extracted."""
        with StreamingRAWReader(tran_file) as reader:
            names = reader.variable_names
            assert "time" in names
            assert "v(out)" in names
            assert "i(R1)" in names

    def test_get_variable_info(self, tran_file: Path) -> None:
        """Test getting variable metadata."""
        with StreamingRAWReader(tran_file) as reader:
            var = reader.get_variable_info("time")
            assert var.name == "time"
            assert var.unit == "time"
            assert var.index == 0

    def test_get_variable_info_not_found(self, tran_file: Path) -> None:
        """Test error when variable not found."""
        with StreamingRAWReader(tran_file) as reader:
            with pytest.raises(KeyError, match="not found"):
                reader.get_variable_info("nonexistent")

    def test_read_variable(self, tran_file: Path) -> None:
        """Test reading a complete variable."""
        with StreamingRAWReader(tran_file) as reader:
            time = reader.read_variable("time")
            assert len(time) == 3
            assert time[0] == pytest.approx(0.0)
            assert time[1] == pytest.approx(1e-3)
            assert time[2] == pytest.approx(2e-3)

    def test_read_all_variables(self, tran_file: Path) -> None:
        """Test reading all variables."""
        with StreamingRAWReader(tran_file) as reader:
            data = reader.read_all()
            assert "time" in data
            assert "v(out)" in data
            assert "i(R1)" in data
            assert len(data["time"]) == 3
            assert len(data["v(out)"]) == 3

    def test_iter_variable(self, tran_file: Path) -> None:
        """Test iterating over a variable in chunks."""
        with StreamingRAWReader(tran_file, chunk_size=2) as reader:
            chunks = list(reader.iter_variable("time"))
            # Should have 2 chunks: [0, 1e-3] and [2e-3]
            assert len(chunks) == 2
            assert len(chunks[0]) == 2
            assert len(chunks[1]) == 1

    def test_iter_all_variables(self, tran_file: Path) -> None:
        """Test iterating over all variables in chunks."""
        with StreamingRAWReader(tran_file, chunk_size=2) as reader:
            chunks = list(reader.iter_all_variables())
            assert len(chunks) == 2
            # First chunk has all variables
            assert "time" in chunks[0]
            assert "v(out)" in chunks[0]

    def test_ac_file_parsing(self, ac_file: Path) -> None:
        """Test parsing AC analysis file."""
        with StreamingRAWReader(ac_file) as reader:
            assert reader.n_points == 3
            assert "frequency" in reader.variable_names

    def test_memory_estimate(self, tran_file: Path) -> None:
        """Test memory usage estimation."""
        with StreamingRAWReader(tran_file) as reader:
            estimate = reader.get_memory_estimate()
            assert "full_load" in estimate
            assert "chunked" in estimate
            assert "n_chunks" in estimate
            # For small files, chunked estimate may exceed full_load
            # (chunk_size * bytes_per_point vs n_points * bytes_per_point)
            assert estimate["full_load"] > 0
            assert estimate["chunked"] > 0


class TestStreamingRAWReaderBinary:
    """Tests for streaming binary RAW files."""

    @pytest.fixture
    def binary_file(self) -> Path:
        """Path to binary RAW fixture (LTspice UTF-16)."""
        return FIXTURES / "circuit_1.raw"

    def test_binary_header_parsing(self, binary_file: Path) -> None:
        """Test parsing binary (UTF-16) file header."""
        if not binary_file.exists():
            pytest.skip("Binary fixture not available")

        with StreamingRAWReader(binary_file) as reader:
            assert reader.header.is_binary
            assert reader.n_points > 0
            assert reader.n_variables > 0

    def test_binary_variable_reading(self, binary_file: Path) -> None:
        """Test reading variables from binary file."""
        if not binary_file.exists():
            pytest.skip("Binary fixture not available")

        with StreamingRAWReader(binary_file) as reader:
            # Read first variable (should be time)
            time = reader.read_variable(reader.variable_names[0])
            assert len(time) == reader.n_points
            assert time.dtype == np.float64


class TestOpenRawStreaming:
    """Tests for open_raw_streaming context manager."""

    @pytest.fixture
    def tran_file(self) -> Path:
        """Path to transient RAW fixture."""
        return FIXTURES / "rc_tran_ng.raw"

    def test_context_manager(self, tran_file: Path) -> None:
        """Test using the convenience context manager."""
        with open_raw_streaming(tran_file) as reader:
            assert reader.n_points == 3

    def test_with_custom_chunk_size(self, tran_file: Path) -> None:
        """Test setting custom chunk size."""
        with open_raw_streaming(tran_file, chunk_size=1) as reader:
            chunks = list(reader.iter_variable("time"))
            assert len(chunks) == 3  # One chunk per point


class TestGetRawInfo:
    """Tests for get_raw_info function."""

    @pytest.fixture
    def tran_file(self) -> Path:
        """Path to transient RAW fixture."""
        return FIXTURES / "rc_tran_ng.raw"

    def test_get_info(self, tran_file: Path) -> None:
        """Test getting RAW file info."""
        header = get_raw_info(tran_file)
        assert isinstance(header, RAWHeader)
        assert header.nvars == 3
        assert header.npoints == 3


class TestStreamToHDF5:
    """Tests for HDF5 streaming conversion."""

    @pytest.fixture
    def tran_file(self) -> Path:
        """Path to transient RAW fixture."""
        return FIXTURES / "rc_tran_ng.raw"

    @pytest.mark.skipif(
        not _check_h5py_available(),
        reason="h5py not available",
    )
    def test_stream_to_hdf5(self, tran_file: Path) -> None:
        """Test streaming RAW to HDF5."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.h5"
            stream_raw_to_hdf5(tran_file, output)

            assert output.exists()

            with h5py.File(output, "r") as h5:
                assert "time" in h5
                assert "v(out)" in h5
                assert h5.attrs["npoints"] == 3
                np.testing.assert_array_almost_equal(h5["time"][:], [0.0, 1e-3, 2e-3], decimal=6)

    @pytest.mark.skipif(
        not _check_h5py_available(),
        reason="h5py not available",
    )
    def test_stream_to_hdf5_with_compression(self, tran_file: Path) -> None:
        """Test HDF5 streaming with compression."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "output.h5"

            with open_raw_streaming(tran_file) as reader:
                reader.to_hdf5(output, compression="gzip", compression_level=9)

            with h5py.File(output, "r") as h5:
                assert h5["time"].compression == "gzip"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_reader_outside_context(self) -> None:
        """Test error when accessing reader outside context."""
        reader = StreamingRAWReader(FIXTURES / "rc_tran_ng.raw")
        with pytest.raises(RuntimeError, match="not in context"):
            _ = reader.header

    def test_nonexistent_file(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            with StreamingRAWReader(Path("/nonexistent/file.raw")):
                pass

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.raw"
        empty_file.write_text("")

        # Empty file should be handled gracefully
        with StreamingRAWReader(empty_file) as reader:
            assert reader.n_points == 0
            assert reader.n_variables == 0
            data = reader.read_all()
            assert data == {}

    def test_malformed_header(self, tmp_path: Path) -> None:
        """Test handling of malformed header."""
        bad_file = tmp_path / "bad.raw"
        bad_file.write_text("Not a valid RAW file\n")

        # Malformed file without proper header returns empty
        with StreamingRAWReader(bad_file) as reader:
            assert reader.n_points == 0
            assert reader.n_variables == 0


class TestChunkedReading:
    """Tests for chunked reading behavior."""

    @pytest.fixture
    def tran_file(self) -> Path:
        """Path to transient RAW fixture."""
        return FIXTURES / "rc_tran_ng.raw"

    def test_chunk_boundaries(self, tran_file: Path) -> None:
        """Test that chunk boundaries are correct."""
        with StreamingRAWReader(tran_file, chunk_size=2) as reader:
            # Read with start/end
            chunks = list(reader.iter_variable("time", start=1, end=3))
            assert len(chunks) == 1  # 2 points from index 1 to 3
            assert len(chunks[0]) == 2

    def test_partial_chunk(self, tran_file: Path) -> None:
        """Test reading partial final chunk."""
        with StreamingRAWReader(tran_file, chunk_size=2) as reader:
            chunks = list(reader.iter_variable("time"))
            # Last chunk should be partial
            total_points = sum(len(c) for c in chunks)
            assert total_points == reader.n_points

    def test_chunk_size_larger_than_file(self, tran_file: Path) -> None:
        """Test when chunk size is larger than file."""
        with StreamingRAWReader(tran_file, chunk_size=1000) as reader:
            chunks = list(reader.iter_variable("time"))
            assert len(chunks) == 1
            assert len(chunks[0]) == reader.n_points
