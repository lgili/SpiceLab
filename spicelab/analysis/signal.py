"""Signal processing helpers for measurements and post-processing.

Lightweight, NumPy-only utilities to avoid heavy deps. Designed to work on
1D NumPy arrays and be easy to plug into xarray/polars flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

WindowKind = Literal["rect", "hann", "hamming", "blackman"]


@dataclass(frozen=True)
class FFTResult:
    freq: np.ndarray  # frequency axis (Hz)
    spectrum: np.ndarray  # complex spectrum or magnitude depending on helper


def window(n: int, kind: WindowKind = "hann") -> np.ndarray:
    """Return a window with n points.

    - rect: ones
    - hann: 0.5 * (1 - cos(2*pi*n/N))
    - hamming: 0.54 - 0.46 * cos(2*pi*n/N)
    - blackman: 0.42 - 0.5*cos(2*pi*n/N) + 0.08*cos(4*pi*n/N)
    """
    if n <= 0:
        raise ValueError("window length must be > 0")
    if kind == "rect":
        return np.ones(n, dtype=float)
    k = np.arange(n, dtype=float)
    if kind == "hann":
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / (n - 1)))
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / (n - 1))
    if kind == "blackman":
        return (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))
            + 0.08 * np.cos(4.0 * np.pi * k / (n - 1))
        )
    raise ValueError(f"unsupported window '{kind}'")


def rfft_coherent(x: np.ndarray, fs: float, *, win: WindowKind = "hann") -> FFTResult:
    """One-sided FFT with simple coherent gain correction.

    Parameters
    ----------
    x: input samples (1D)
    fs: sample rate in Hz
    win: window type

    Notes
    -----
    - Applies window and divides by sum(window) to correct coherent gain (CG).
    - Returns frequency axis (0..fs/2) and complex spectrum (one-sided, DC..Nyquist).
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    n = x.size
    if n == 0:
        raise ValueError("x must be non-empty")
    w = window(n, win)
    cg = np.sum(w)
    xw = x * w
    spec = np.fft.rfft(xw) / cg * 2.0  # scale to roughly preserve amplitude (except DC/Nyquist)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return FFTResult(freq=freq, spectrum=spec)


def amplitude_spectrum(
    x: np.ndarray, fs: float, *, win: WindowKind = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq, |X(f)|) using rfft_coherent.

    The magnitude is 2*|FFT|/sum(window) as in rfft_coherent().
    """
    res = rfft_coherent(x, fs, win=win)
    mag = np.abs(res.spectrum)
    # fix DC and Nyquist scaling (factor 2 not applied there in one-sided spectrum)
    if mag.size > 0:
        mag[0] = np.abs(np.fft.rfft(x * window(x.size, win))[0]) / np.sum(window(x.size, win))
    if mag.size > 2 and np.isclose(res.freq[-1], fs / 2):
        mag[-1] = np.abs(np.fft.rfft(x * window(x.size, win))[-1]) / np.sum(window(x.size, win))
    return res.freq, mag


def power_spectral_density(
    x: np.ndarray, fs: float, *, win: WindowKind = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq, PSD) with a simple window-energy normalization.

    PSD here is magnitude-squared normalized by ENBW of the window to yield units ~ V^2/Hz.
    """
    n = x.size
    if n == 0:
        raise ValueError("x must be non-empty")
    w = window(n, win)
    enbw = fs * np.sum(w**2) / (np.sum(w) ** 2)
    res = rfft_coherent(x, fs, win=win)
    psd = (np.abs(res.spectrum) ** 2) / enbw
    return res.freq, psd
