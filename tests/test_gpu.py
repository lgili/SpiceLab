"""Tests for the GPU acceleration module."""

from __future__ import annotations

import numpy as np
import pytest

from spicelab.gpu import (
    GPUAccelerator,
    GPUInfo,
    get_gpu_info,
    gpu_fft,
    gpu_ifft,
    gpu_irfft,
    gpu_rfft,
    is_gpu_available,
)
from spicelab.gpu.accelerator import (
    BenchmarkResult,
    GPUMemoryManager,
    benchmark_batch_fft,
    benchmark_fft,
)


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_unavailable_info(self) -> None:
        """Test GPUInfo when GPU is not available."""
        info = GPUInfo(available=False)
        assert not info.available
        assert info.name == "N/A"
        assert "Not available" in str(info)

    def test_available_info(self) -> None:
        """Test GPUInfo when GPU is available."""
        info = GPUInfo(
            available=True,
            name="Test GPU",
            compute_capability=(8, 6),
            memory_total_mb=8192,
            memory_free_mb=6000,
            cuda_version="12.0",
            cupy_version="12.0.0",
        )
        assert info.available
        assert info.name == "Test GPU"
        assert "Test GPU" in str(info)
        assert "8.6" in str(info)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_speedup_calculation(self) -> None:
        """Test speedup is calculated correctly."""
        result = BenchmarkResult(
            operation="fft",
            input_shape=(1000,),
            gpu_time_ms=10.0,
            cpu_time_ms=100.0,
        )
        assert result.speedup == pytest.approx(10.0)

    def test_zero_gpu_time(self) -> None:
        """Test speedup when GPU time is zero."""
        result = BenchmarkResult(
            operation="fft",
            input_shape=(1000,),
            gpu_time_ms=0.0,
            cpu_time_ms=100.0,
        )
        # Should not raise, speedup stays at default
        assert result.speedup == 0.0


class TestIsGPUAvailable:
    """Tests for is_gpu_available function."""

    def test_returns_bool(self) -> None:
        """Test that function returns a boolean."""
        result = is_gpu_available()
        assert isinstance(result, bool)


class TestGetGPUInfo:
    """Tests for get_gpu_info function."""

    def test_returns_gpu_info(self) -> None:
        """Test that function returns GPUInfo."""
        info = get_gpu_info()
        assert isinstance(info, GPUInfo)

    def test_info_is_cached(self) -> None:
        """Test that GPU info is cached."""
        info1 = get_gpu_info()
        info2 = get_gpu_info()
        assert info1 is info2


class TestGPUFFT:
    """Tests for GPU FFT functions."""

    def test_gpu_fft_basic(self) -> None:
        """Test basic FFT operation."""
        # Simple sine wave
        n = 1024
        t = np.linspace(0, 1, n)
        signal = np.sin(2 * np.pi * 10 * t)

        result = gpu_fft(signal)

        assert result.shape == (n,)
        assert np.iscomplexobj(result)

    def test_gpu_fft_matches_numpy(self) -> None:
        """Test that GPU FFT matches NumPy FFT."""
        np.random.seed(42)
        signal = np.random.randn(1024)

        gpu_result = gpu_fft(signal)
        numpy_result = np.fft.fft(signal)

        np.testing.assert_allclose(gpu_result, numpy_result, rtol=1e-10)

    def test_gpu_ifft_inverse(self) -> None:
        """Test that IFFT is inverse of FFT."""
        np.random.seed(42)
        signal = np.random.randn(512)

        spectrum = gpu_fft(signal)
        recovered = gpu_ifft(spectrum)

        np.testing.assert_allclose(recovered.real, signal, rtol=1e-10)

    def test_gpu_rfft_real_input(self) -> None:
        """Test real FFT on real input."""
        signal = np.random.randn(1024)

        result = gpu_rfft(signal)

        # rfft returns n//2 + 1 complex values
        assert result.shape == (513,)
        assert np.iscomplexobj(result)

    def test_gpu_irfft_inverse(self) -> None:
        """Test that IRFFT is inverse of RFFT."""
        signal = np.random.randn(512)

        spectrum = gpu_rfft(signal)
        recovered = gpu_irfft(spectrum, n=512)

        np.testing.assert_allclose(recovered, signal, rtol=1e-10)

    def test_gpu_fft_with_n(self) -> None:
        """Test FFT with specified length."""
        signal = np.random.randn(100)

        result = gpu_fft(signal, n=128)

        assert result.shape == (128,)

    def test_gpu_fft_axis(self) -> None:
        """Test FFT along specific axis."""
        data = np.random.randn(10, 256)

        result = gpu_fft(data, axis=1)

        assert result.shape == (10, 256)

    def test_gpu_fft_normalization(self) -> None:
        """Test FFT with different normalizations."""
        signal = np.random.randn(256)

        result_backward = gpu_fft(signal, norm="backward")
        result_ortho = gpu_fft(signal, norm="ortho")
        result_forward = gpu_fft(signal, norm="forward")

        # Ortho normalization scales by 1/sqrt(n)
        np.testing.assert_allclose(
            result_ortho,
            result_backward / np.sqrt(256),
            rtol=1e-10,
        )


class TestGPUAccelerator:
    """Tests for GPUAccelerator class."""

    @pytest.fixture
    def accelerator(self) -> GPUAccelerator:
        """Create accelerator instance."""
        return GPUAccelerator(prefer_gpu=True, min_size_for_gpu=1024)

    def test_init_default(self) -> None:
        """Test default initialization."""
        accel = GPUAccelerator()
        assert accel.prefer_gpu is True
        assert accel.min_size_for_gpu == 1024

    def test_gpu_available_property(self, accelerator: GPUAccelerator) -> None:
        """Test gpu_available property."""
        assert accelerator.gpu_available == is_gpu_available()

    def test_fft_basic(self, accelerator: GPUAccelerator) -> None:
        """Test basic FFT through accelerator."""
        signal = np.random.randn(2048)

        result = accelerator.fft(signal)

        np.testing.assert_allclose(result, np.fft.fft(signal), rtol=1e-10)

    def test_ifft_basic(self, accelerator: GPUAccelerator) -> None:
        """Test basic IFFT through accelerator."""
        signal = np.random.randn(2048)
        spectrum = accelerator.fft(signal)

        recovered = accelerator.ifft(spectrum)

        np.testing.assert_allclose(recovered.real, signal, rtol=1e-10)

    def test_rfft_basic(self, accelerator: GPUAccelerator) -> None:
        """Test basic RFFT through accelerator."""
        signal = np.random.randn(2048)

        result = accelerator.rfft(signal)

        np.testing.assert_allclose(result, np.fft.rfft(signal), rtol=1e-10)

    def test_irfft_basic(self, accelerator: GPUAccelerator) -> None:
        """Test basic IRFFT through accelerator."""
        signal = np.random.randn(2048)
        spectrum = accelerator.rfft(signal)

        recovered = accelerator.irfft(spectrum, n=2048)

        np.testing.assert_allclose(recovered, signal, rtol=1e-10)

    def test_fft2_basic(self, accelerator: GPUAccelerator) -> None:
        """Test 2D FFT through accelerator."""
        data = np.random.randn(64, 64)

        result = accelerator.fft2(data)

        np.testing.assert_allclose(result, np.fft.fft2(data), rtol=1e-10)

    def test_batch_fft_list(self, accelerator: GPUAccelerator) -> None:
        """Test batch FFT with list of signals."""
        signals = [np.random.randn(1024) for _ in range(5)]

        results = accelerator.batch_fft(signals)

        assert len(results) == 5
        for sig, res in zip(signals, results):
            np.testing.assert_allclose(res, np.fft.fft(sig), rtol=1e-10)

    def test_batch_fft_array(self, accelerator: GPUAccelerator) -> None:
        """Test batch FFT with 2D array."""
        signals = np.random.randn(5, 1024)

        results = accelerator.batch_fft(signals)

        assert len(results) == 5
        for i, res in enumerate(results):
            np.testing.assert_allclose(res, np.fft.fft(signals[i]), rtol=1e-10)

    def test_convolve_full(self, accelerator: GPUAccelerator) -> None:
        """Test convolution with full mode."""
        a = np.random.randn(100)
        b = np.random.randn(20)

        result = accelerator.convolve(a, b, mode="full")

        expected = np.convolve(a, b, mode="full")
        np.testing.assert_allclose(result, expected, rtol=1e-8)

    def test_convolve_same(self, accelerator: GPUAccelerator) -> None:
        """Test convolution with same mode."""
        a = np.random.randn(100)
        b = np.random.randn(20)

        result = accelerator.convolve(a, b, mode="same")

        assert result.shape == a.shape

    def test_convolve_valid(self, accelerator: GPUAccelerator) -> None:
        """Test convolution with valid mode."""
        a = np.random.randn(100)
        b = np.random.randn(20)

        result = accelerator.convolve(a, b, mode="valid")

        expected_len = len(a) - len(b) + 1
        assert len(result) == expected_len

    def test_power_spectrum(self, accelerator: GPUAccelerator) -> None:
        """Test power spectrum calculation."""
        # Generate signal with known frequency
        fs = 1000
        t = np.arange(0, 1, 1 / fs)
        freq = 100
        signal = np.sin(2 * np.pi * freq * t)

        freqs, power = accelerator.power_spectrum(signal, fs=fs)

        # Find peak frequency
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]

        # Should be close to 100 Hz
        assert abs(peak_freq - freq) < 2  # Within 2 Hz

    def test_stats_tracking(self, accelerator: GPUAccelerator) -> None:
        """Test that usage stats are tracked."""
        accelerator.reset_stats()

        # Perform some operations
        for _ in range(5):
            accelerator.fft(np.random.randn(2048))

        stats = accelerator.stats
        total_calls = stats["gpu_calls"] + stats["cpu_calls"]
        assert total_calls == 5

    def test_reset_stats(self, accelerator: GPUAccelerator) -> None:
        """Test stats reset."""
        accelerator.fft(np.random.randn(2048))
        accelerator.reset_stats()

        stats = accelerator.stats
        assert stats["gpu_calls"] == 0
        assert stats["cpu_calls"] == 0
        assert stats["gpu_fallbacks"] == 0

    def test_prefer_gpu_false(self) -> None:
        """Test with GPU preference disabled."""
        accel = GPUAccelerator(prefer_gpu=False)

        signal = np.random.randn(10000)
        accel.fft(signal)

        # Should only use CPU
        assert accel.stats["gpu_calls"] == 0
        assert accel.stats["cpu_calls"] == 1

    def test_min_size_threshold(self) -> None:
        """Test minimum size threshold for GPU."""
        accel = GPUAccelerator(prefer_gpu=True, min_size_for_gpu=10000)

        # Small array - should use CPU
        accel.fft(np.random.randn(100))
        assert accel.stats["cpu_calls"] == 1

        accel.reset_stats()

        # Large array - may use GPU if available
        accel.fft(np.random.randn(20000))
        # Either GPU or CPU was used
        assert accel.stats["gpu_calls"] + accel.stats["cpu_calls"] == 1


class TestGPUMemoryManager:
    """Tests for GPUMemoryManager class."""

    @pytest.fixture
    def manager(self) -> GPUMemoryManager:
        """Create memory manager instance."""
        return GPUMemoryManager(reserve_mb=256)

    def test_init(self, manager: GPUMemoryManager) -> None:
        """Test initialization."""
        assert manager.reserve_mb == 256

    def test_available_mb(self, manager: GPUMemoryManager) -> None:
        """Test available memory check."""
        available = manager.available_mb
        # Should return float (0 if no GPU)
        assert isinstance(available, float)
        assert available >= 0

    def test_total_mb(self, manager: GPUMemoryManager) -> None:
        """Test total memory check."""
        total = manager.total_mb
        assert isinstance(total, float)
        assert total >= 0

    def test_can_fit_zero(self, manager: GPUMemoryManager) -> None:
        """Test can_fit with zero size."""
        # Zero bytes should always fit if GPU available
        result = manager.can_fit(0)
        assert isinstance(result, bool)

    def test_optimal_chunk_size(self, manager: GPUMemoryManager) -> None:
        """Test optimal chunk size calculation."""
        chunk_size = manager.optimal_chunk_size(np.dtype(np.float64))
        assert isinstance(chunk_size, int)
        assert chunk_size >= 1024  # Minimum chunk size

    def test_memory_info(self, manager: GPUMemoryManager) -> None:
        """Test memory info dictionary."""
        info = manager.memory_info()
        assert "total_mb" in info
        assert "free_mb" in info
        assert "used_mb" in info
        assert "available_mb" in info

    def test_clear_cache(self, manager: GPUMemoryManager) -> None:
        """Test cache clearing (should not raise)."""
        manager.clear_cache()  # Should not raise even without GPU


class TestBenchmarkFunctions:
    """Tests for benchmark functions."""

    def test_benchmark_fft(self) -> None:
        """Test FFT benchmark function."""
        result = benchmark_fft(size=1000, iterations=2)

        assert isinstance(result, BenchmarkResult)
        assert result.operation == "fft"
        assert result.input_shape == (1000,)
        assert result.cpu_time_ms > 0

    def test_benchmark_batch_fft(self) -> None:
        """Test batch FFT benchmark function."""
        result = benchmark_batch_fft(
            num_signals=10,
            signal_length=1000,
            iterations=2,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.operation == "batch_fft"
        assert result.input_shape == (10, 1000)
        assert result.cpu_time_ms > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_array(self) -> None:
        """Test FFT on empty array raises ValueError."""
        # NumPy FFT doesn't support empty arrays
        with pytest.raises(ValueError, match="Invalid number of FFT data points"):
            gpu_fft(np.array([]))

    def test_single_element(self) -> None:
        """Test FFT on single element."""
        result = gpu_fft(np.array([5.0]))
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)

    def test_complex_input(self) -> None:
        """Test FFT on complex input."""
        signal = np.random.randn(100) + 1j * np.random.randn(100)
        result = gpu_fft(signal)

        np.testing.assert_allclose(result, np.fft.fft(signal), rtol=1e-10)

    def test_2d_array_default_axis(self) -> None:
        """Test FFT on 2D array with default axis."""
        data = np.random.randn(10, 256)
        result = gpu_fft(data)

        # Default axis is -1
        np.testing.assert_allclose(result, np.fft.fft(data), rtol=1e-10)

    def test_large_array(self) -> None:
        """Test FFT on large array."""
        # 1M points
        signal = np.random.randn(1_000_000)
        result = gpu_fft(signal)

        assert result.shape == signal.shape

    def test_non_power_of_two(self) -> None:
        """Test FFT on non-power-of-two length."""
        signal = np.random.randn(1000)  # Not power of 2
        result = gpu_fft(signal)

        np.testing.assert_allclose(result, np.fft.fft(signal), rtol=1e-10)
