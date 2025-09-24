import numpy as np
from spicelab.io.raw_reader import Trace


def test_trace_ac_helpers() -> None:
    # Simula um traço com valores complexos
    vals = np.array([1.0, 2.0, 3.0])
    comp = np.array([1 + 0j, 0 + 1j, 1 - 1j])
    t = Trace("v(out)", "V", vals, _complex=comp)

    mag = t.magnitude()
    real = t.real()
    imag = t.imag()
    phase = t.phase_deg()

    assert np.allclose(mag, np.abs(comp))
    assert np.allclose(real, comp.real)
    assert np.allclose(imag, comp.imag)
    assert np.allclose(phase, np.angle(comp, deg=True))
