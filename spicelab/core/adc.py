from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class ADCModel:
    """Simple ADC model for emulation and testing.

    Parameters
    - vref: maximum supported input voltage (e.g., 3.3)
    - bits: resolution in bits (e.g., 12)
    - fs: sampling frequency in Hz

    Behavior
    - sample(seq) -> list of integers: quantizes each sample to the ADC code
    - sample_with_bits(seq) -> list of (int, bits_str)
    - clamps input to [0, vref]
    - ideal S/H (instantaneous sample)
    """

    vref: float = 3.3
    bits: int = 12
    fs: float = 1000.0
    # input front-end model
    rs: float = 0.0  # source resistance in ohms
    c_sh: float = 1e-12  # sampling/hold capacitance in farads
    tacq: float = 1e-6  # acquisition time in seconds
    # ADC non-idealities
    offset: float = 0.0
    gain: float = 1.0
    noise_rms: float = 0.0
    jitter_std: float = 0.0
    rng_seed: int | None = None
    # aperture window in seconds (time window over which input is averaged)
    aperture: float = 0.0

    @property
    def levels(self) -> int:
        return 1 << self.bits

    def quantize(self, v: float) -> int:
        """Quantize a single voltage to ADC code (unsigned).

        Clamps to [0, vref], returns integer in [0, 2^bits - 1].
        """
        if math.isnan(v):
            raise ValueError("input is NaN")
        # clamp
        vv = max(0.0, min(self.vref, float(v)))
        # apply offset and gain
        vv = (vv + self.offset) * self.gain
        vv = max(0.0, min(self.vref, vv))
        code = int((vv / self.vref) * (self.levels - 1) + 0.5)
        # ensure within range
        return max(0, min(self.levels - 1, code))

    def code_to_bits(self, code: int) -> str:
        return format(code, f"0{self.bits}b")

    def sample(self, samples: Sequence[float]) -> list[int]:
        """Quantize a sequence of float samples.

        This assumes samples are already taken at a rate compatible with fs.
        """
        return [self.quantize(v) for v in samples]

    def _seed_rng(self) -> None:
        if self.rng_seed is not None:
            random.seed(self.rng_seed)

    def _apply_noise_and_jitter(self, v: float) -> float:
        # gaussian noise
        if self.noise_rms and self.noise_rms > 0:
            v = v + random.gauss(0.0, self.noise_rms)
        return v

    def sample_sh(self, voltages: Sequence[float], rs: float | None = None) -> list[int]:
        """Sample using a simplified sample-and-hold front-end.

        voltages: sequence of source voltages at the start of acquisition
        rs: optional per-call source resistance; if None, use self.rs

        Simplified RC settling: v_final = v_source * (1 - exp(-Tacq/RC)). We treat
        the hold capacitor charged through source resistance Rs.
        """
        self._seed_rng()
        Rs = self.rs if rs is None else rs
        codes: list[int] = []
        for v in voltages:
            # compute settling factor
            tau = Rs * self.c_sh if Rs and self.c_sh else 0.0
            if tau > 0:
                alpha = 1.0 - math.exp(-self.tacq / tau)
            else:
                alpha = 1.0
            v_settled = v * alpha
            # apply noise and jitter
            v_noisy = self._apply_noise_and_jitter(v_settled)
            codes.append(self.quantize(v_noisy))
        return codes

    def sample_from_function(
        self,
        func: Callable[[float], float],
        n_samples: int,
        start_time: float = 0.0,
        aperture_samples: int = 1,
    ) -> list[int]:
        """Sample a continuous function `func(t)` at times determined by fs.

        For each sample index k, nominal sample time is t = start_time + k / fs.
        Applies jitter (Gaussian with std `jitter_std`) to the sampling instant.
        If `aperture_samples` > 1, takes that many uniformly spaced sub-samples
        across the `aperture` window centered at the (jittered) sample time,
        averages them and then applies quantization.
        """
        self._seed_rng()
        codes: list[int] = []
        for k in range(n_samples):
            t0 = start_time + k / self.fs
            # jitter the sampling instant
            if self.jitter_std and self.jitter_std > 0:
                dt = random.gauss(0.0, self.jitter_std)
            else:
                dt = 0.0
            t_sample = t0 + dt

            # aperture averaging
            if self.aperture and aperture_samples > 1:
                half = self.aperture / 2.0
                values = []
                for m in range(aperture_samples):
                    # uniform spacing across aperture
                    tm = t_sample - half + (m + 0.5) * (self.aperture / aperture_samples)
                    values.append(func(tm))
                v = sum(values) / len(values)
            else:
                v = func(t_sample)

            # apply RC settling approximation per sample if needed (use self.rs)
            tau = (self.rs * self.c_sh) if (self.rs and self.c_sh) else 0.0
            if tau > 0:
                alpha = 1.0 - math.exp(-self.tacq / tau)
            else:
                alpha = 1.0
            v_settled = v * alpha
            v_noisy = self._apply_noise_and_jitter(v_settled)
            codes.append(self.quantize(v_noisy))
        return codes

    def sample_with_bits(self, samples: Sequence[float]) -> list[tuple[int, str]]:
        return [(c, self.code_to_bits(c)) for c in self.sample(samples)]
