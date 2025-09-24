"""ADC demo: generate a sine wave and sample with ADCModel."""

from math import pi, sin

from spicelab.core.adc import ADCModel


def generate_sine(vpp: float, freq: float, fs: float, duration: float):
    samples = int(fs * duration)
    amp = vpp / 2.0
    for n in range(samples):
        t = n / fs
        yield amp * (1 + sin(2 * pi * freq * t))  # shift so it's in [0, vpp]


def main() -> None:
    adc = ADCModel(vref=3.3, bits=12, fs=1000.0, rs=1e3, c_sh=10e-12, tacq=1e-6)
    data = list(generate_sine(3.0, 50.0, adc.fs, 0.02))
    codes = adc.sample_sh(data)
    for i, (v, c) in enumerate(zip(data, codes, strict=True)):
        print(f"{i:03d}: V={v:.3f} -> code={c} bits={adc.code_to_bits(c)}")


if __name__ == "__main__":
    main()
