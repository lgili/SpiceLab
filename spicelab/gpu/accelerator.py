"""GPU-accelerated signal processing with automatic CPU fallback.

This module provides GPU-accelerated FFT and signal processing operations
using CuPy when available, with automatic fallback to NumPy/SciPy when
GPU is not available.

Example::

    from spicelab.gpu import GPUAccelerator, is_gpu_available

    # Check availability
    if is_gpu_available():
        info = get_gpu_info()
        print(f"Using GPU: {info.name} with {info.memory_total_mb}MB")

    # Use accelerator
    accel = GPUAccelerator()
    spectrum = accel.fft(signal)
    filtered = accel.convolve(signal, kernel)

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Try to import CuPy for GPU acceleration
_CUPY_AVAILABLE = False
_GPU_INFO: GPUInfo | None = None

try:
    import cupy as cp
    from cupy import fft as cp_fft

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore
    cp_fft = None  # type: ignore


@dataclass
class GPUInfo:
    """Information about the available GPU.

    Attributes:
        available: Whether GPU is available
        name: GPU device name
        compute_capability: CUDA compute capability (major, minor)
        memory_total_mb: Total GPU memory in MB
        memory_free_mb: Free GPU memory in MB
        cuda_version: CUDA runtime version
        cupy_version: CuPy library version

    """

    available: bool = False
    name: str = "N/A"
    compute_capability: tuple[int, int] = (0, 0)
    memory_total_mb: float = 0.0
    memory_free_mb: float = 0.0
    cuda_version: str = "N/A"
    cupy_version: str = "N/A"
    driver_version: str = "N/A"

    def __str__(self) -> str:
        """Return string representation."""
        if not self.available:
            return "GPU: Not available"
        return (
            f"GPU: {self.name}\n"
            f"  Compute Capability: {self.compute_capability[0]}.{self.compute_capability[1]}\n"
            f"  Memory: {self.memory_free_mb:.0f}/{self.memory_total_mb:.0f} MB free\n"
            f"  CUDA: {self.cuda_version}, CuPy: {self.cupy_version}"
        )


@dataclass
class BenchmarkResult:
    """Result of a GPU vs CPU benchmark.

    Attributes:
        operation: Name of the operation benchmarked
        input_shape: Shape of input data
        gpu_time_ms: GPU execution time in milliseconds
        cpu_time_ms: CPU execution time in milliseconds
        speedup: GPU speedup factor (cpu_time / gpu_time)
        gpu_used: Whether GPU was actually used

    """

    operation: str
    input_shape: tuple[int, ...]
    gpu_time_ms: float
    cpu_time_ms: float
    speedup: float = 0.0
    gpu_used: bool = False

    def __post_init__(self) -> None:
        """Calculate speedup after initialization."""
        if self.gpu_time_ms > 0:
            self.speedup = self.cpu_time_ms / self.gpu_time_ms


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CuPy is installed and a CUDA-capable GPU is detected.

    """
    if not _CUPY_AVAILABLE:
        return False
    try:
        # Try to get device count
        device_count = cp.cuda.runtime.getDeviceCount()
        return device_count > 0
    except Exception:
        return False


def get_gpu_info() -> GPUInfo:
    """Get information about the available GPU.

    Returns:
        GPUInfo object with GPU details, or unavailable info if no GPU.

    """
    global _GPU_INFO

    if _GPU_INFO is not None:
        return _GPU_INFO

    if not is_gpu_available():
        _GPU_INFO = GPUInfo(available=False)
        return _GPU_INFO

    try:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        mem_info = cp.cuda.runtime.memGetInfo()

        _GPU_INFO = GPUInfo(
            available=True,
            name=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            compute_capability=(props["major"], props["minor"]),
            memory_total_mb=mem_info[1] / (1024 * 1024),
            memory_free_mb=mem_info[0] / (1024 * 1024),
            cuda_version=".".join(str(x) for x in cp.cuda.runtime.runtimeGetVersion()),
            cupy_version=cp.__version__,
        )
    except Exception as e:
        _GPU_INFO = GPUInfo(available=False, name=f"Error: {e}")

    return _GPU_INFO


def _to_gpu(arr: ArrayLike) -> Any:
    """Transfer array to GPU memory.

    Args:
        arr: NumPy array or array-like

    Returns:
        CuPy array on GPU

    """
    if cp is None:
        raise RuntimeError("CuPy not available")
    if isinstance(arr, cp.ndarray):
        return arr
    return cp.asarray(arr)


def _to_cpu(arr: Any) -> NDArray[Any]:
    """Transfer array from GPU to CPU memory.

    Args:
        arr: CuPy array or NumPy array

    Returns:
        NumPy array on CPU

    """
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def gpu_fft(
    x: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Compute FFT using GPU if available, else CPU.

    Args:
        x: Input array
        n: Length of the transformed axis
        axis: Axis over which to compute the FFT
        norm: Normalization mode

    Returns:
        Complex array containing FFT result

    """
    if is_gpu_available():
        try:
            x_gpu = _to_gpu(x)
            result_gpu = cp_fft.fft(x_gpu, n=n, axis=axis, norm=norm)
            return _to_cpu(result_gpu)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback using numpy
    return np.fft.fft(x, n=n, axis=axis, norm=norm)


def gpu_ifft(
    x: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Compute inverse FFT using GPU if available, else CPU.

    Args:
        x: Input array
        n: Length of the transformed axis
        axis: Axis over which to compute the IFFT
        norm: Normalization mode

    Returns:
        Complex array containing IFFT result

    """
    if is_gpu_available():
        try:
            x_gpu = _to_gpu(x)
            result_gpu = cp_fft.ifft(x_gpu, n=n, axis=axis, norm=norm)
            return _to_cpu(result_gpu)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback using numpy
    return np.fft.ifft(x, n=n, axis=axis, norm=norm)


def gpu_rfft(
    x: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Compute real FFT using GPU if available, else CPU.

    Args:
        x: Input array (real-valued)
        n: Length of the transformed axis
        axis: Axis over which to compute the FFT
        norm: Normalization mode

    Returns:
        Complex array containing FFT of real input

    """
    if is_gpu_available():
        try:
            x_gpu = _to_gpu(x)
            result_gpu = cp_fft.rfft(x_gpu, n=n, axis=axis, norm=norm)
            return _to_cpu(result_gpu)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback using numpy
    return np.fft.rfft(x, n=n, axis=axis, norm=norm)


def gpu_irfft(
    x: ArrayLike,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] | None = None,
) -> NDArray[np.floating[Any]]:
    """Compute inverse real FFT using GPU if available, else CPU.

    Args:
        x: Input array (complex)
        n: Length of the output (real) axis
        axis: Axis over which to compute the IFFT
        norm: Normalization mode

    Returns:
        Real array containing inverse FFT result

    """
    if is_gpu_available():
        try:
            x_gpu = _to_gpu(x)
            result_gpu = cp_fft.irfft(x_gpu, n=n, axis=axis, norm=norm)
            return _to_cpu(result_gpu)
        except Exception:
            pass  # Fall through to CPU

    # CPU fallback using numpy
    return np.fft.irfft(x, n=n, axis=axis, norm=norm)


@dataclass
class GPUAccelerator:
    """GPU-accelerated signal processing with automatic CPU fallback.

    This class provides a unified interface for GPU-accelerated operations
    that automatically falls back to CPU when GPU is not available.

    Args:
        prefer_gpu: Whether to prefer GPU when available (default True)
        min_size_for_gpu: Minimum array size to use GPU (smaller arrays
            may be faster on CPU due to transfer overhead)

    Example::

        accel = GPUAccelerator()

        # Single FFT
        spectrum = accel.fft(signal)

        # Batch FFT
        spectra = accel.batch_fft(signals)

        # With explicit GPU control
        accel_cpu = GPUAccelerator(prefer_gpu=False)
        spectrum_cpu = accel_cpu.fft(signal)

    """

    prefer_gpu: bool = True
    min_size_for_gpu: int = 1024
    _stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize statistics."""
        self._stats = {
            "gpu_calls": 0,
            "cpu_calls": 0,
            "gpu_fallbacks": 0,
        }

    @property
    def gpu_available(self) -> bool:
        """Whether GPU is available."""
        return is_gpu_available()

    @property
    def using_gpu(self) -> bool:
        """Whether GPU will be used for operations."""
        return self.prefer_gpu and self.gpu_available

    @property
    def stats(self) -> dict[str, int]:
        """Get usage statistics."""
        return self._stats.copy()

    def _should_use_gpu(self, arr: ArrayLike) -> bool:
        """Determine if GPU should be used for this array."""
        if not self.using_gpu:
            return False
        arr_np = np.asarray(arr)
        return arr_np.size >= self.min_size_for_gpu

    def fft(
        self,
        x: ArrayLike,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Compute FFT with automatic GPU/CPU selection.

        Args:
            x: Input array
            n: Length of the transformed axis
            axis: Axis over which to compute the FFT
            norm: Normalization mode

        Returns:
            Complex array containing FFT result

        """
        if self._should_use_gpu(x):
            try:
                x_gpu = _to_gpu(x)
                result_gpu = cp_fft.fft(x_gpu, n=n, axis=axis, norm=norm)
                self._stats["gpu_calls"] += 1
                return _to_cpu(result_gpu)
            except Exception:
                self._stats["gpu_fallbacks"] += 1

        # CPU path
        self._stats["cpu_calls"] += 1
        return np.fft.fft(x, n=n, axis=axis, norm=norm)

    def ifft(
        self,
        x: ArrayLike,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Compute inverse FFT with automatic GPU/CPU selection.

        Args:
            x: Input array
            n: Length of the transformed axis
            axis: Axis over which to compute the IFFT
            norm: Normalization mode

        Returns:
            Complex array containing IFFT result

        """
        if self._should_use_gpu(x):
            try:
                x_gpu = _to_gpu(x)
                result_gpu = cp_fft.ifft(x_gpu, n=n, axis=axis, norm=norm)
                self._stats["gpu_calls"] += 1
                return _to_cpu(result_gpu)
            except Exception:
                self._stats["gpu_fallbacks"] += 1

        self._stats["cpu_calls"] += 1
        return np.fft.ifft(x, n=n, axis=axis, norm=norm)

    def rfft(
        self,
        x: ArrayLike,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Compute real FFT with automatic GPU/CPU selection.

        Args:
            x: Input array (real-valued)
            n: Length of the transformed axis
            axis: Axis over which to compute the FFT
            norm: Normalization mode

        Returns:
            Complex array containing FFT of real input

        """
        if self._should_use_gpu(x):
            try:
                x_gpu = _to_gpu(x)
                result_gpu = cp_fft.rfft(x_gpu, n=n, axis=axis, norm=norm)
                self._stats["gpu_calls"] += 1
                return _to_cpu(result_gpu)
            except Exception:
                self._stats["gpu_fallbacks"] += 1

        self._stats["cpu_calls"] += 1
        return np.fft.rfft(x, n=n, axis=axis, norm=norm)

    def irfft(
        self,
        x: ArrayLike,
        n: int | None = None,
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Compute inverse real FFT with automatic GPU/CPU selection.

        Args:
            x: Input array (complex)
            n: Length of the output (real) axis
            axis: Axis over which to compute the IFFT
            norm: Normalization mode

        Returns:
            Real array containing inverse FFT result

        """
        if self._should_use_gpu(x):
            try:
                x_gpu = _to_gpu(x)
                result_gpu = cp_fft.irfft(x_gpu, n=n, axis=axis, norm=norm)
                self._stats["gpu_calls"] += 1
                return _to_cpu(result_gpu)
            except Exception:
                self._stats["gpu_fallbacks"] += 1

        self._stats["cpu_calls"] += 1
        return np.fft.irfft(x, n=n, axis=axis, norm=norm)

    def fft2(
        self,
        x: ArrayLike,
        s: tuple[int, int] | None = None,
        axes: tuple[int, int] = (-2, -1),
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> NDArray[np.complexfloating[Any, Any]]:
        """Compute 2D FFT with automatic GPU/CPU selection.

        Args:
            x: Input array
            s: Shape of the output
            axes: Axes over which to compute the FFT
            norm: Normalization mode

        Returns:
            Complex array containing 2D FFT result

        """
        if self._should_use_gpu(x):
            try:
                x_gpu = _to_gpu(x)
                result_gpu = cp_fft.fft2(x_gpu, s=s, axes=axes, norm=norm)
                self._stats["gpu_calls"] += 1
                return _to_cpu(result_gpu)
            except Exception:
                self._stats["gpu_fallbacks"] += 1

        self._stats["cpu_calls"] += 1
        return np.fft.fft2(x, s=s, axes=axes, norm=norm)

    def batch_fft(
        self,
        signals: list[ArrayLike] | NDArray[Any],
        axis: int = -1,
        norm: Literal["backward", "ortho", "forward"] | None = None,
    ) -> list[NDArray[np.complexfloating[Any, Any]]]:
        """Compute FFT on multiple signals.

        Args:
            signals: List of signals or 2D array (signals along first axis)
            axis: Axis over which to compute FFT for each signal
            norm: Normalization mode

        Returns:
            List of FFT results

        """
        if isinstance(signals, np.ndarray) and signals.ndim == 2:
            # Process as batch
            return [self.fft(signals[i], axis=axis, norm=norm) for i in range(len(signals))]
        else:
            return [self.fft(sig, axis=axis, norm=norm) for sig in signals]

    def convolve(
        self,
        a: ArrayLike,
        v: ArrayLike,
        mode: Literal["full", "same", "valid"] = "full",
    ) -> NDArray[Any]:
        """Convolve two arrays using FFT.

        Args:
            a: First input array
            v: Second input array
            mode: Output mode ('full', 'same', 'valid')

        Returns:
            Convolution result

        """
        a_arr = np.asarray(a)
        v_arr = np.asarray(v)

        # Use FFT-based convolution for efficiency
        n = len(a_arr) + len(v_arr) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))

        if self._should_use_gpu(a_arr) or self._should_use_gpu(v_arr):
            try:
                a_gpu = _to_gpu(a_arr)
                v_gpu = _to_gpu(v_arr)

                a_fft = cp_fft.fft(a_gpu, n=n_fft)
                v_fft = cp_fft.fft(v_gpu, n=n_fft)
                result = cp_fft.ifft(a_fft * v_fft)[:n]
                self._stats["gpu_calls"] += 1

                result_cpu = _to_cpu(result).real
            except Exception:
                self._stats["gpu_fallbacks"] += 1
                result_cpu = self._cpu_convolve(a_arr, v_arr, n, n_fft)
        else:
            self._stats["cpu_calls"] += 1
            result_cpu = self._cpu_convolve(a_arr, v_arr, n, n_fft)

        # Apply mode
        if mode == "full":
            return result_cpu
        elif mode == "same":
            start = (len(v_arr) - 1) // 2
            return result_cpu[start : start + len(a_arr)]
        else:  # valid
            return result_cpu[len(v_arr) - 1 : len(a_arr)]

    def _cpu_convolve(
        self, a: NDArray[Any], v: NDArray[Any], n: int, n_fft: int
    ) -> NDArray[Any]:
        """CPU convolution using FFT."""
        a_fft = np.fft.fft(a, n=n_fft)
        v_fft = np.fft.fft(v, n=n_fft)
        result = np.fft.ifft(a_fft * v_fft)[:n]
        return result.real

    def power_spectrum(
        self,
        x: ArrayLike,
        fs: float = 1.0,
        window: str | None = "hann",
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute power spectrum of a signal.

        Args:
            x: Input signal
            fs: Sampling frequency
            window: Window function to apply

        Returns:
            Tuple of (frequencies, power spectrum)

        """
        x_arr = np.asarray(x)
        n = len(x_arr)

        # Apply window using numpy (common windows)
        if window is not None:
            if window == "hann":
                w = np.hanning(n)
            elif window == "hamming":
                w = np.hamming(n)
            elif window == "blackman":
                w = np.blackman(n)
            elif window == "bartlett":
                w = np.bartlett(n)
            elif window == "kaiser":
                w = np.kaiser(n, 14)  # Default beta
            else:
                # Try scipy as fallback for other windows
                try:
                    from scipy.signal import get_window
                    w = get_window(window, n)
                except ImportError:
                    # Default to no window if scipy not available
                    w = np.ones(n)
            x_arr = x_arr * w

        # Compute FFT
        spectrum = self.rfft(x_arr)
        power = np.abs(spectrum) ** 2 / n

        # Frequency axis
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        return freqs, power

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._stats = {
            "gpu_calls": 0,
            "cpu_calls": 0,
            "gpu_fallbacks": 0,
        }


def benchmark_fft(
    size: int = 1_000_000,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark GPU vs CPU FFT performance.

    Args:
        size: Size of test array
        iterations: Number of iterations for timing

    Returns:
        BenchmarkResult with timing information

    """
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(size).astype(np.float64)

    # CPU benchmark using numpy
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = np.fft.fft(data)
        cpu_times.append((time.perf_counter() - start) * 1000)
    cpu_time_ms = np.median(cpu_times)

    # GPU benchmark
    gpu_time_ms = cpu_time_ms  # Default if GPU not available
    gpu_used = False

    if is_gpu_available():
        try:
            # Warmup
            data_gpu = _to_gpu(data)
            _ = cp_fft.fft(data_gpu)
            cp.cuda.Stream.null.synchronize()

            gpu_times = []
            for _ in range(iterations):
                data_gpu = _to_gpu(data)
                start = time.perf_counter()
                _ = cp_fft.fft(data_gpu)
                cp.cuda.Stream.null.synchronize()
                gpu_times.append((time.perf_counter() - start) * 1000)
            gpu_time_ms = np.median(gpu_times)
            gpu_used = True
        except Exception:
            pass

    return BenchmarkResult(
        operation="fft",
        input_shape=(size,),
        gpu_time_ms=gpu_time_ms,
        cpu_time_ms=cpu_time_ms,
        gpu_used=gpu_used,
    )


def benchmark_batch_fft(
    num_signals: int = 100,
    signal_length: int = 10000,
    iterations: int = 5,
) -> BenchmarkResult:
    """Benchmark GPU vs CPU batch FFT performance.

    Args:
        num_signals: Number of signals in batch
        signal_length: Length of each signal
        iterations: Number of iterations for timing

    Returns:
        BenchmarkResult with timing information

    """
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(num_signals, signal_length).astype(np.float64)

    # CPU benchmark using numpy
    cpu_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for i in range(num_signals):
            _ = np.fft.fft(data[i])
        cpu_times.append((time.perf_counter() - start) * 1000)
    cpu_time_ms = np.median(cpu_times)

    # GPU benchmark
    gpu_time_ms = cpu_time_ms
    gpu_used = False

    if is_gpu_available():
        try:
            # Warmup
            data_gpu = _to_gpu(data[0])
            _ = cp_fft.fft(data_gpu)
            cp.cuda.Stream.null.synchronize()

            gpu_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                # Batch transfer and FFT
                data_gpu = _to_gpu(data)
                for i in range(num_signals):
                    _ = cp_fft.fft(data_gpu[i])
                cp.cuda.Stream.null.synchronize()
                gpu_times.append((time.perf_counter() - start) * 1000)
            gpu_time_ms = np.median(gpu_times)
            gpu_used = True
        except Exception:
            pass

    return BenchmarkResult(
        operation="batch_fft",
        input_shape=(num_signals, signal_length),
        gpu_time_ms=gpu_time_ms,
        cpu_time_ms=cpu_time_ms,
        gpu_used=gpu_used,
    )


class GPUMemoryManager:
    """Manage GPU memory for large datasets.

    This class helps manage GPU memory when processing datasets
    that may not fit entirely in GPU memory.

    Example::

        manager = GPUMemoryManager()

        # Check if data fits in GPU memory
        if manager.can_fit(data_size_bytes):
            result = gpu_fft(data)
        else:
            # Process in chunks
            for chunk in manager.chunk_array(data, chunk_size):
                process(chunk)

    """

    def __init__(self, reserve_mb: float = 256) -> None:
        """Initialize memory manager.

        Args:
            reserve_mb: Amount of GPU memory to keep free (in MB)

        """
        self.reserve_mb = reserve_mb

    @property
    def available_mb(self) -> float:
        """Get available GPU memory in MB."""
        if not is_gpu_available():
            return 0.0
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            return (mem_info[0] / (1024 * 1024)) - self.reserve_mb
        except Exception:
            return 0.0

    @property
    def total_mb(self) -> float:
        """Get total GPU memory in MB."""
        if not is_gpu_available():
            return 0.0
        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            return mem_info[1] / (1024 * 1024)
        except Exception:
            return 0.0

    def can_fit(self, size_bytes: int, margin: float = 1.5) -> bool:
        """Check if data of given size can fit in GPU memory.

        Args:
            size_bytes: Size of data in bytes
            margin: Safety margin multiplier (for intermediate results)

        Returns:
            True if data fits in available GPU memory

        """
        size_mb = (size_bytes * margin) / (1024 * 1024)
        return size_mb <= self.available_mb

    def optimal_chunk_size(
        self,
        dtype: np.dtype[Any],
        margin: float = 2.0,
    ) -> int:
        """Calculate optimal chunk size for processing.

        Args:
            dtype: Data type of arrays
            margin: Safety margin for intermediate results

        Returns:
            Optimal number of elements per chunk

        """
        available_bytes = self.available_mb * 1024 * 1024 / margin
        element_size = np.dtype(dtype).itemsize
        return max(1024, int(available_bytes / element_size))

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if is_gpu_available():
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:
                pass

    def memory_info(self) -> dict[str, float]:
        """Get detailed memory information.

        Returns:
            Dictionary with memory stats in MB

        """
        if not is_gpu_available():
            return {
                "total_mb": 0.0,
                "free_mb": 0.0,
                "used_mb": 0.0,
                "available_mb": 0.0,
            }

        try:
            mem_info = cp.cuda.runtime.memGetInfo()
            free_mb = mem_info[0] / (1024 * 1024)
            total_mb = mem_info[1] / (1024 * 1024)
            return {
                "total_mb": total_mb,
                "free_mb": free_mb,
                "used_mb": total_mb - free_mb,
                "available_mb": max(0, free_mb - self.reserve_mb),
            }
        except Exception:
            return {
                "total_mb": 0.0,
                "free_mb": 0.0,
                "used_mb": 0.0,
                "available_mb": 0.0,
            }
