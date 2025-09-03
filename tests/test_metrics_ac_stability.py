import numpy as np
from cat.analysis import (
    ac_gain_phase,
    crossover_freq_0db,
    gain_margin_db,
    phase_crossover_freq,
    phase_margin,
)
from cat.io.raw_reader import Trace, TraceSet


def _synthetic_open_loop() -> TraceSet:
    """
    Modelo sintético de malha aberta com 1 polo:
      H(jw) = K / (1 + j f/fc)
    Escolhemos K=10 (20 dB), fc=1e3 Hz, então o cruzamento de 0 dB ocorre ~ 9.95 kHz.
    """
    K = 10.0
    fc = 1.0e3
    f = np.logspace(1, 5, 501)  # 10 .. 1e5
    s = 1j * (f / fc)  # aqui usamos f/fc como normalizado (fase coerente)
    H = K / (1.0 + s)
    mag = np.abs(H)
    # guardamos também módulo em "values" e complexo para fase
    return TraceSet(
        [
            Trace("frequency", "Hz", f),
            Trace("v(out)", "V", mag, _complex=H),
        ]
    )


def test_ac_stability_margins() -> None:
    ts = _synthetic_open_loop()
    f, mag_db, phase_deg = ac_gain_phase(ts, "v(out)")
    # monotonia básica
    assert f[0] < f[-1]
    assert np.isfinite(mag_db).all()

    wc = crossover_freq_0db(ts, "v(out)")
    assert wc is not None
    # deve estar por volta de 10 kHz (com aproximações)
    assert 5e3 < wc < 2e4

    pm = phase_margin(ts, "v(out)")
    assert pm is not None
    # margem de fase esperada ~ ~95° para K=10, 1 polo
    assert 60.0 < pm < 140.0

    # 1 polo nunca atinge -180°, então GM deve ser None
    w180 = phase_crossover_freq(ts, "v(out)")
    assert w180 is None
    gm = gain_margin_db(ts, "v(out)")
    assert gm is None
