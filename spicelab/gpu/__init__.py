"""GPU acceleration module for high-performance signal processing.

This module provides:
- GPU-accelerated FFT operations using CuPy
- Automatic CPU fallback when GPU is unavailable
- Memory management utilities for large datasets
- Benchmarking tools for performance comparison

Example::

    from spicelab.gpu import gpu_fft, is_gpu_available, GPUAccelerator

    # Check GPU availability
    if is_gpu_available():
        print("GPU acceleration enabled")

    # Perform GPU-accelerated FFT
    spectrum = gpu_fft(signal_data)

    # Use the accelerator for batch operations
    accelerator = GPUAccelerator()
    results = accelerator.batch_fft(signals)

"""

from .accelerator import (
    GPUAccelerator,
    GPUInfo,
    gpu_fft,
    gpu_ifft,
    gpu_rfft,
    gpu_irfft,
    is_gpu_available,
    get_gpu_info,
)

__all__ = [
    "GPUAccelerator",
    "GPUInfo",
    "gpu_fft",
    "gpu_ifft",
    "gpu_rfft",
    "gpu_irfft",
    "is_gpu_available",
    "get_gpu_info",
]
