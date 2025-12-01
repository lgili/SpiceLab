"""Tests for the HDF5 storage module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spicelab.storage import (
    HDF5ResultStorage,
    StorageMetadata,
    DatasetInfo,
)
from spicelab.storage.hdf5 import (
    is_hdf5_available,
    save_to_hdf5,
    load_from_hdf5,
)


def _check_h5py_available() -> bool:
    """Check if h5py is available."""
    try:
        import h5py  # noqa: F401
        return True
    except ImportError:
        return False


# Skip all tests if h5py is not available
pytestmark = pytest.mark.skipif(
    not _check_h5py_available(),
    reason="h5py not available",
)


class TestStorageMetadata:
    """Tests for StorageMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Test creating metadata."""
        meta = StorageMetadata(
            name="test_sim",
            simulation_type="tran",
            parameters={"temp": 25, "vdd": 3.3},
            description="Test simulation",
            tags=["test", "transient"],
        )
        assert meta.name == "test_sim"
        assert meta.simulation_type == "tran"
        assert meta.parameters["temp"] == 25

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        meta = StorageMetadata(
            name="test",
            parameters={"a": 1},
            tags=["tag1"],
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert '"a": 1' in d["parameters"]
        assert "tag1" in d["tags"]

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {
            "name": "test",
            "created_at": 12345.0,
            "parameters": '{"x": 10}',
            "tags": '["a", "b"]',
        }
        meta = StorageMetadata.from_dict(d)
        assert meta.name == "test"
        assert meta.created_at == 12345.0
        assert meta.parameters["x"] == 10
        assert meta.tags == ["a", "b"]


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""

    def test_create_info(self) -> None:
        """Test creating dataset info."""
        info = DatasetInfo(
            name="/result/time",
            shape=(1000,),
            dtype="float64",
            size_bytes=8000,
            compression="gzip",
        )
        assert info.name == "/result/time"
        assert info.shape == (1000,)
        assert info.compression == "gzip"

    def test_str_representation(self) -> None:
        """Test string representation."""
        info = DatasetInfo(
            name="data",
            shape=(100, 10),
            dtype="float32",
            size_bytes=4000,
        )
        s = str(info)
        assert "data" in s
        assert "(100, 10)" in s
        assert "float32" in s


class TestIsHDF5Available:
    """Tests for is_hdf5_available function."""

    def test_returns_bool(self) -> None:
        """Test that function returns boolean."""
        result = is_hdf5_available()
        assert isinstance(result, bool)
        # Since we skip tests if h5py not available, this should be True
        assert result is True


class TestHDF5ResultStorage:
    """Tests for HDF5ResultStorage class."""

    @pytest.fixture
    def storage_path(self, tmp_path: Path) -> Path:
        """Create a temporary path for storage."""
        return tmp_path / "test_results.h5"

    @pytest.fixture
    def storage(self, storage_path: Path) -> HDF5ResultStorage:
        """Create a storage instance."""
        return HDF5ResultStorage(storage_path)

    def test_create_storage(self, storage_path: Path) -> None:
        """Test creating storage."""
        storage = HDF5ResultStorage(storage_path)
        assert storage.path == storage_path
        storage.close()

    def test_context_manager(self, storage_path: Path) -> None:
        """Test using storage as context manager."""
        with HDF5ResultStorage(storage_path) as storage:
            assert storage._file is not None
        # File should be closed after context
        assert storage._file is None

    def test_save_and_load_result(self, storage: HDF5ResultStorage) -> None:
        """Test saving and loading a result."""
        time = np.linspace(0, 1, 1000)
        vout = np.sin(2 * np.pi * 10 * time)
        iload = np.cos(2 * np.pi * 10 * time) * 0.1

        storage.save_result(
            name="sim_001",
            time=time,
            data={"vout": vout, "iload": iload},
            metadata={"temp": 25},
        )

        result = storage.load_result("sim_001")

        np.testing.assert_array_equal(result["time"], time)
        np.testing.assert_array_equal(result["vout"], vout)
        np.testing.assert_array_equal(result["iload"], iload)
        assert result["metadata"].parameters["temp"] == 25

        storage.close()

    def test_save_without_time(self, storage: HDF5ResultStorage) -> None:
        """Test saving result without time array."""
        data = {"signal": np.random.randn(100)}

        storage.save_result(name="no_time", data=data)
        result = storage.load_result("no_time")

        assert "time" not in result
        np.testing.assert_array_equal(result["signal"], data["signal"])

        storage.close()

    def test_overwrite_result(self, storage: HDF5ResultStorage) -> None:
        """Test overwriting existing result."""
        storage.save_result("test", data={"a": np.array([1, 2, 3])})
        storage.save_result("test", data={"a": np.array([4, 5, 6])}, overwrite=True)

        result = storage.load_result("test")
        np.testing.assert_array_equal(result["a"], [4, 5, 6])

        storage.close()

    def test_overwrite_disabled(self, storage: HDF5ResultStorage) -> None:
        """Test that overwrite=False raises error."""
        storage.save_result("test", data={"a": np.array([1])})

        with pytest.raises(ValueError, match="already exists"):
            storage.save_result("test", data={"a": np.array([2])}, overwrite=False)

        storage.close()

    def test_load_nonexistent(self, storage: HDF5ResultStorage) -> None:
        """Test loading nonexistent result."""
        with pytest.raises(KeyError, match="not found"):
            storage.load_result("nonexistent")

        storage.close()

    def test_save_and_load_sweep(self, storage: HDF5ResultStorage) -> None:
        """Test saving and loading a parameter sweep."""
        time = np.linspace(0, 1, 100)
        temps = [-40, 25, 85]
        results = [
            {"vout": np.sin(2 * np.pi * t / 100 * time)}
            for t in temps
        ]

        storage.save_sweep(
            name="temp_sweep",
            parameter="temperature",
            values=temps,
            results=results,
            time=time,
            metadata={"vdd": 3.3},
        )

        loaded = storage.load_sweep("temp_sweep")

        assert loaded["parameter"] == "temperature"
        assert loaded["values"] == temps
        assert len(loaded["results"]) == 3
        np.testing.assert_array_equal(loaded["time"], time)
        assert loaded["metadata"]["vdd"] == 3.3

        storage.close()

    def test_save_and_load_batch(self, storage: HDF5ResultStorage) -> None:
        """Test saving and loading a batch of results."""
        batch = {
            "result_a": {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6])},
            "result_b": {"x": np.array([7, 8, 9]), "y": np.array([10, 11, 12])},
        }

        storage.save_batch(batch, group_name="my_batch", metadata={"version": 1})
        loaded = storage.load_batch("my_batch")

        assert "result_a" in loaded
        assert "result_b" in loaded
        np.testing.assert_array_equal(loaded["result_a"]["x"], [1, 2, 3])
        np.testing.assert_array_equal(loaded["result_b"]["y"], [10, 11, 12])

        storage.close()

    def test_list_results(self, storage: HDF5ResultStorage) -> None:
        """Test listing results."""
        storage.save_result("sim_a", data={"x": np.array([1])})
        storage.save_result("sim_b", data={"x": np.array([2])})
        storage.save_result("sim_c", data={"x": np.array([3])})

        results = storage.list_results()

        assert len(results) == 3
        assert "sim_a" in results
        assert "sim_b" in results
        assert "sim_c" in results

        storage.close()

    def test_delete_result(self, storage: HDF5ResultStorage) -> None:
        """Test deleting a result."""
        storage.save_result("to_delete", data={"x": np.array([1])})
        assert "to_delete" in storage.list_results()

        deleted = storage.delete_result("to_delete")
        assert deleted is True
        assert "to_delete" not in storage.list_results()

        # Delete nonexistent
        deleted = storage.delete_result("nonexistent")
        assert deleted is False

        storage.close()

    def test_get_info(self, storage: HDF5ResultStorage) -> None:
        """Test getting dataset information."""
        time = np.linspace(0, 1, 1000)
        data = np.random.randn(1000)

        storage.save_result("info_test", time=time, data={"signal": data})
        infos = storage.get_info("info_test")

        assert len(infos) >= 2  # time and signal
        names = [info.name for info in infos]
        assert any("time" in name for name in names)
        assert any("signal" in name for name in names)

        storage.close()

    def test_get_metadata(self, storage: HDF5ResultStorage) -> None:
        """Test getting metadata."""
        storage.save_result(
            "meta_test",
            data={"x": np.array([1])},
            metadata={"key": "value"},
        )

        meta = storage.get_metadata("meta_test")
        assert meta is not None
        assert meta.parameters["key"] == "value"

        # Nonexistent
        assert storage.get_metadata("nonexistent") is None

        storage.close()

    def test_size_properties(self, storage_path: Path) -> None:
        """Test size properties."""
        storage = HDF5ResultStorage(storage_path)

        # Save some data
        storage.save_result("size_test", data={"big": np.random.randn(10000)})
        storage.flush()

        assert storage.size_bytes > 0
        assert storage.size_mb > 0

        storage.close()

    def test_compression(self, storage_path: Path) -> None:
        """Test that compression works."""
        # Highly compressible data
        data = np.zeros(100000)

        # With compression
        storage_comp = HDF5ResultStorage(storage_path, compression="gzip")
        storage_comp.save_result("compressed", data={"zeros": data})
        storage_comp.close()
        compressed_size = storage_path.stat().st_size

        # Without compression
        uncompressed_path = storage_path.with_suffix(".h5.uncompressed")
        storage_uncomp = HDF5ResultStorage(uncompressed_path, compression=None)
        storage_uncomp.save_result("uncompressed", data={"zeros": data})
        storage_uncomp.close()
        uncompressed_size = uncompressed_path.stat().st_size

        # Compressed should be smaller
        assert compressed_size < uncompressed_size

    def test_flush(self, storage: HDF5ResultStorage) -> None:
        """Test flushing to disk."""
        storage.save_result("flush_test", data={"x": np.array([1, 2, 3])})
        storage.flush()  # Should not raise

        storage.close()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test save_to_hdf5 and load_from_hdf5."""
        path = tmp_path / "convenience.h5"
        time = np.linspace(0, 1, 100)
        data = {"vout": np.sin(time)}

        save_to_hdf5(path, "test_result", time=time, data=data)
        result = load_from_hdf5(path, "test_result")

        np.testing.assert_array_equal(result["time"], time)
        np.testing.assert_array_equal(result["vout"], data["vout"])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_data(self, tmp_path: Path) -> None:
        """Test saving empty data."""
        path = tmp_path / "empty.h5"
        storage = HDF5ResultStorage(path)

        storage.save_result("empty", data={})
        result = storage.load_result("empty")

        assert "metadata" in result

        storage.close()

    def test_large_arrays(self, tmp_path: Path) -> None:
        """Test saving large arrays."""
        path = tmp_path / "large.h5"
        storage = HDF5ResultStorage(path)

        # 10 million points
        large_data = np.random.randn(10_000_000)
        storage.save_result("large", data={"big": large_data})

        result = storage.load_result("large")
        np.testing.assert_array_equal(result["big"], large_data)

        storage.close()

    def test_complex_data(self, tmp_path: Path) -> None:
        """Test saving complex-valued arrays."""
        path = tmp_path / "complex.h5"
        storage = HDF5ResultStorage(path)

        complex_data = np.random.randn(100) + 1j * np.random.randn(100)
        storage.save_result("complex", data={"spectrum": complex_data})

        result = storage.load_result("complex")
        np.testing.assert_array_equal(result["spectrum"], complex_data)

        storage.close()

    def test_multidimensional_data(self, tmp_path: Path) -> None:
        """Test saving multi-dimensional arrays."""
        path = tmp_path / "multi.h5"
        storage = HDF5ResultStorage(path)

        data_2d = np.random.randn(100, 50)
        data_3d = np.random.randn(10, 20, 30)

        storage.save_result("multi", data={"2d": data_2d, "3d": data_3d})
        result = storage.load_result("multi")

        np.testing.assert_array_equal(result["2d"], data_2d)
        np.testing.assert_array_equal(result["3d"], data_3d)

        storage.close()

    def test_special_characters_in_name(self, tmp_path: Path) -> None:
        """Test result names with special characters."""
        path = tmp_path / "special.h5"
        storage = HDF5ResultStorage(path)

        # HDF5 allows most characters except /
        storage.save_result("result_with_underscore", data={"x": np.array([1])})
        storage.save_result("result-with-dash", data={"x": np.array([2])})

        assert "result_with_underscore" in storage.list_results()
        assert "result-with-dash" in storage.list_results()

        storage.close()

    def test_unicode_metadata(self, tmp_path: Path) -> None:
        """Test Unicode in metadata."""
        path = tmp_path / "unicode.h5"
        storage = HDF5ResultStorage(path)

        storage.save_result(
            "unicode_test",
            data={"x": np.array([1])},
            metadata={"description": "温度テスト: 25°C"},
        )

        result = storage.load_result("unicode_test")
        assert "温度テスト" in result["metadata"].parameters["description"]

        storage.close()


class TestCompactFunction:
    """Tests for compact function."""

    def test_compact_reclaims_space(self, tmp_path: Path) -> None:
        """Test that compact reduces file size after deletions."""
        path = tmp_path / "compact.h5"
        storage = HDF5ResultStorage(path)

        # Create large dataset
        large_data = np.random.randn(100000)
        storage.save_result("big", data={"data": large_data})
        storage.flush()
        size_before = storage.size_bytes

        # Delete and compact
        storage.delete_result("big")
        storage.compact()

        size_after = storage.size_bytes

        # Should be smaller after compacting
        assert size_after < size_before

        storage.close()
