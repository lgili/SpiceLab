from spicelab.core.adc import ADCModel


def test_quantize_basic() -> None:
    adc = ADCModel(vref=3.3, bits=4)
    assert adc.levels == 16
    # 0V -> 0
    assert adc.quantize(0.0) == 0
    # full-scale -> max code
    assert adc.quantize(3.3) == 15
    # mid-scale
    mid = adc.quantize(3.3 / 2.0)
    assert 7 <= mid <= 8


def test_sample_sequence() -> None:
    adc = ADCModel(vref=1.0, bits=2)
    samples = [0.0, 0.3, 0.6, 1.0]
    codes = adc.sample(samples)
    assert isinstance(codes, list)
    assert all(isinstance(c, int) for c in codes)


def test_sample_sh_settling() -> None:
    # With large source resistance and small C, settling will be small
    adc = ADCModel(vref=3.3, bits=8, rs=1e6, c_sh=1e-12, tacq=1e-6)
    samples = [1.0, 2.0, 3.0]
    codes = adc.sample_sh(samples)
    # Codes should be <= ideal sample codes because of incomplete settling
    ideal = adc.sample(samples)
    assert all(c <= ic for c, ic in zip(codes, ideal, strict=True))


def test_sample_sh_with_noise_and_seed() -> None:
    adc1 = ADCModel(vref=3.3, bits=8, noise_rms=0.01, rng_seed=42)
    adc2 = ADCModel(vref=3.3, bits=8, noise_rms=0.01, rng_seed=42)
    s = [1.0, 1.5, 2.0]
    c1 = adc1.sample_sh(s)
    c2 = adc2.sample_sh(s)
    assert c1 == c2


def test_sample_from_function_jitter_and_aperture() -> None:
    # function: simple ramp f(t)=t
    def f(t: float) -> float:
        return t

    adc_with_j = ADCModel(vref=10.0, bits=8, fs=100.0, jitter_std=1e-4, rng_seed=1)
    n = 5
    c_with_j = adc_with_j.sample_from_function(f, n_samples=n, start_time=0.0)
    # with jitter but same seed should be deterministic (same across runs)
    c_with_j2 = ADCModel(
        vref=10.0, bits=8, fs=100.0, jitter_std=1e-4, rng_seed=1
    ).sample_from_function(f, n_samples=n, start_time=0.0)
    assert c_with_j == c_with_j2
    # aperture averaging should reduce quantization variability if aperture large
    adc_ap = ADCModel(vref=10.0, bits=8, fs=100.0, jitter_std=0.0, aperture=1e-2, rng_seed=1)
    c_ap = adc_ap.sample_from_function(f, n_samples=n, start_time=0.0, aperture_samples=5)
    assert len(c_ap) == n
