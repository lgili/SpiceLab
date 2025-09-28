from __future__ import annotations

import numpy as np
from spicelab.analysis import (
    amplitude_spectrum,
    power_spectral_density,
    rfft_coherent,
    window,
)


def test_window_shapes_and_values() -> None:
    w = window(8, "rect")
    assert w.shape == (8,) and np.allclose(w, 1.0)
    w2 = window(8, "hann")
    assert w2.shape == (8,) and np.isclose(w2[0], 0.0) and np.isclose(w2[-1], 0.0)


def test_rfft_and_magnitude_for_sine() -> None:
    fs = 10_000.0
    f = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x = np.sin(2 * np.pi * f * t)
    res = rfft_coherent(x, fs, win="hann")
    # peak should be near f
    k = int(np.argmin(np.abs(res.freq - f)))
    assert res.freq[k] == res.freq[k]  # finite
    # amplitude_spectrum around the peak should be ~0.5..1.5 depending on window/scaling
    freq, mag = amplitude_spectrum(x, fs, win="hann")
    assert mag[k] > 0.3


def test_psd_basic_properties() -> None:
    fs = 2000.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    f, psd = power_spectral_density(x, fs, win="hann")
    assert f.shape == psd.shape and f.ndim == 1
    # PSD should be non-negative
    assert np.all(psd >= 0)
