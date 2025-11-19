# M16: Yield & Reliability Analysis

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 10-12 weeks
**Dependencies:** M2 (performance), M9 (optimization), M14 (PDK corners), M15 (distributed for large-scale MC)

## Problem Statement

SpiceLab lacks comprehensive yield and reliability analysis capabilities required for production IC design and safety-critical applications (automotive, aerospace, medical). Designers need tools to predict manufacturing yield, worst-case performance, aging effects, and compliance with industry standards.

### Current Gaps
- ❌ No large-scale yield analysis (10k+ Monte Carlo limited by local compute)
- ❌ No worst-case corner analysis automation
- ❌ No reliability prediction (FIT rates, MTBF calculation)
- ❌ No aging models (NBTI, HCI, EM, TDDB)
- ❌ No stress testing automation
- ❌ No compliance reports (ISO 26262, DO-254, IEC 61508)
- ❌ No design centering optimization for yield improvement

### Impact
- **Production IC Design:** Cannot predict manufacturing yield
- **Safety-Critical:** Cannot certify for automotive/aerospace
- **Quality:** No long-term reliability predictions
- **Industry Adoption:** Missing critical capabilities for serious IC design

## Objectives

1. **Large-scale yield analysis** - 10k+ Monte Carlo with statistical analysis
2. **Worst-case analysis** - Automated corner combination sweep
3. **Reliability prediction** - FIT rates, MTBF, failure probability over time
4. **Aging models** - NBTI, HCI, EM, TDDB degradation over lifetime
5. **Stress testing** - Automated overvoltage, overcurrent, temperature stress
6. **Compliance reports** - ISO 26262, DO-254, IEC 61508 report generation
7. **Yield optimization** - Design centering to maximize yield
8. **Target:** Production-ready yield analysis, automotive/aerospace compliance

## Technical Design

### 1. Yield Analysis Framework

```python
# spicelab/reliability/yield_analysis.py
from dataclasses import dataclass
import numpy as np
import xarray as xr
from scipy import stats

@dataclass
class YieldSpec:
    """Specification for yield analysis."""
    measurement: str  # What to measure (e.g., "gain", "bandwidth")
    min_value: float | None = None  # Lower spec limit
    max_value: float | None = None  # Upper spec limit
    target_value: float | None = None  # Nominal target
    name: str = ""

class YieldAnalyzer:
    """Large-scale Monte Carlo yield analysis."""

    def __init__(self, specs: list[YieldSpec]):
        self.specs = specs
        self.results: xr.Dataset | None = None
        self.yield_report: dict = {}

    async def run_yield_analysis(
        self,
        circuit_factory: Callable,
        n_iterations: int = 10000,
        distributed: bool = True
    ) -> dict:
        """Run large-scale Monte Carlo for yield prediction."""

        if distributed:
            from spicelab.distributed import DaskBackend
            backend = DaskBackend()
            self.results = await backend.run_monte_carlo(
                circuit_factory,
                n_iterations
            )
        else:
            # Local Monte Carlo
            self.results = await self._run_local_mc(circuit_factory, n_iterations)

        # Analyze yield for each spec
        self.yield_report = {}
        for spec in self.specs:
            yield_data = self._calculate_yield(spec)
            self.yield_report[spec.name or spec.measurement] = yield_data

        return self.yield_report

    def _calculate_yield(self, spec: YieldSpec) -> dict:
        """Calculate yield for a single specification."""
        values = self.results[spec.measurement].values

        # Count passing samples
        if spec.min_value is not None and spec.max_value is not None:
            passing = np.sum((values >= spec.min_value) & (values <= spec.max_value))
        elif spec.min_value is not None:
            passing = np.sum(values >= spec.min_value)
        elif spec.max_value is not None:
            passing = np.sum(values <= spec.max_value)
        else:
            passing = len(values)

        yield_pct = 100.0 * passing / len(values)

        # Statistical metrics
        return {
            "yield_percent": yield_pct,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "cp": self._process_capability(values, spec),  # Cp index
            "cpk": self._process_capability_index(values, spec),  # Cpk index
            "six_sigma": self._six_sigma_level(values, spec),
            "passing_samples": int(passing),
            "total_samples": len(values)
        }

    def _process_capability(self, values: np.ndarray, spec: YieldSpec) -> float:
        """Calculate Cp (process capability)."""
        if spec.min_value is None or spec.max_value is None:
            return np.nan

        sigma = np.std(values)
        if sigma == 0:
            return np.inf

        return (spec.max_value - spec.min_value) / (6 * sigma)

    def _process_capability_index(self, values: np.ndarray, spec: YieldSpec) -> float:
        """Calculate Cpk (process capability index)."""
        if spec.min_value is None or spec.max_value is None:
            return np.nan

        mean = np.mean(values)
        sigma = np.std(values)

        if sigma == 0:
            return np.inf

        cpu = (spec.max_value - mean) / (3 * sigma)
        cpl = (mean - spec.min_value) / (3 * sigma)

        return min(cpu, cpl)

    def _six_sigma_level(self, values: np.ndarray, spec: YieldSpec) -> float:
        """Calculate Six Sigma level (Z-score)."""
        cpk = self._process_capability_index(values, spec)
        return cpk * 3  # Convert Cpk to sigma level
```

### 2. Worst-Case Corner Analysis

```python
# spicelab/reliability/worst_case.py
from itertools import product

@dataclass
class WorstCaseCorner:
    """Definition of a worst-case corner."""
    process: str  # "tt", "ff", "ss", "fs", "sf"
    voltage: float  # Supply voltage
    temperature: float  # Operating temperature (°C)

class WorstCaseAnalyzer:
    """Automated worst-case corner analysis."""

    def __init__(self, pdk: 'PDK'):
        self.pdk = pdk

    def generate_corners(
        self,
        voltage_range: tuple[float, float] = (1.65, 1.95),
        temp_range: tuple[float, float] = (-40, 125),
        num_points: int = 3
    ) -> list[WorstCaseCorner]:
        """Generate all corner combinations."""

        process_corners = ["tt", "ff", "ss", "fs", "sf"]
        voltages = np.linspace(voltage_range[0], voltage_range[1], num_points)
        temperatures = np.linspace(temp_range[0], temp_range[1], num_points)

        corners = []
        for proc, volt, temp in product(process_corners, voltages, temperatures):
            corners.append(WorstCaseCorner(proc, volt, temp))

        return corners  # 5 * 3 * 3 = 45 corners

    async def run_worst_case_analysis(
        self,
        circuit: 'Circuit',
        corners: list[WorstCaseCorner],
        specs: list[YieldSpec]
    ) -> dict:
        """Run simulation across all worst-case corners."""

        results = {}
        for corner in corners:
            # Inject corner parameters
            circuit_corner = self._apply_corner(circuit, corner)

            # Run simulation
            result = await run_simulation(circuit_corner, ["ac", "tran"])
            results[self._corner_name(corner)] = result

        # Find worst-case corner for each spec
        worst_case_report = {}
        for spec in specs:
            worst = self._find_worst_corner(results, spec)
            worst_case_report[spec.name] = worst

        return worst_case_report

    def _find_worst_corner(
        self,
        results: dict[str, xr.Dataset],
        spec: YieldSpec
    ) -> dict:
        """Find corner that produces worst-case value."""

        values = {
            corner: result[spec.measurement].values
            for corner, result in results.items()
        }

        # Worst case is minimum if spec has min_value, maximum if max_value
        if spec.min_value is not None:
            worst_corner = min(values, key=lambda c: values[c].min())
            worst_value = values[worst_corner].min()
        else:
            worst_corner = max(values, key=lambda c: values[c].max())
            worst_value = values[worst_corner].max()

        return {
            "corner": worst_corner,
            "value": float(worst_value),
            "margin": self._calculate_margin(worst_value, spec)
        }
```

### 3. Reliability & Aging Models

```python
# spicelab/reliability/aging.py
from enum import Enum

class AgingMechanism(Enum):
    """Common aging mechanisms."""
    NBTI = "nbti"  # Negative Bias Temperature Instability
    PBTI = "pbti"  # Positive Bias Temperature Instability
    HCI = "hci"    # Hot Carrier Injection
    EM = "em"      # Electromigration
    TDDB = "tddb"  # Time-Dependent Dielectric Breakdown

class ReliabilityModel:
    """Reliability and aging analysis."""

    def __init__(self, temperature: float = 85.0):
        self.temperature = temperature  # Operating temperature (°C)

    def predict_nbti_degradation(
        self,
        stress_time: float,  # hours
        vdd: float,
        device: str = "pmos"
    ) -> float:
        """Predict NBTI-induced Vth shift."""

        # Simplified NBTI model: ΔVth = A * (time)^n * exp(-Ea/kT)
        A = 1e-3  # Pre-factor
        n = 0.25  # Time exponent
        Ea = 0.15  # Activation energy (eV)
        k = 8.617e-5  # Boltzmann constant (eV/K)

        T_kelvin = self.temperature + 273.15

        delta_vth = A * (stress_time ** n) * np.exp(-Ea / (k * T_kelvin))

        # Scale by voltage stress
        delta_vth *= (vdd / 1.8) ** 2

        return delta_vth  # Threshold voltage shift (V)

    def predict_hci_degradation(
        self,
        stress_time: float,
        vds: float,
        ids: float,
        device: str = "nmos"
    ) -> float:
        """Predict HCI-induced degradation."""

        # HCI primarily affects NMOS at high Vds
        if device != "nmos":
            return 0.0

        # Simplified HCI model
        A = 1e-4
        m = 3.5  # Voltage exponent
        n = 0.5  # Time exponent

        delta_vth = A * (vds ** m) * (stress_time ** n)

        return delta_vth

    def predict_em_lifetime(
        self,
        current_density: float,  # A/m²
        wire_width: float,  # m
        temperature: float | None = None
    ) -> float:
        """Predict electromigration lifetime (hours)."""

        # Black's equation: MTF = A * (J^-n) * exp(Ea / kT)
        A = 1e14
        n = 2.0
        Ea = 0.9  # eV (for Al interconnect)
        k = 8.617e-5

        T = (temperature or self.temperature) + 273.15

        MTF = A * (current_density ** -n) * np.exp(Ea / (k * T))

        return MTF / 3600  # Convert seconds to hours

    def calculate_fit_rate(
        self,
        failure_mechanisms: dict[AgingMechanism, float]
    ) -> float:
        """Calculate FIT (Failures In Time) rate."""

        # FIT = failures per billion device-hours
        total_failure_rate = sum(failure_mechanisms.values())

        fit_rate = total_failure_rate * 1e9

        return fit_rate

    def calculate_mtbf(self, fit_rate: float) -> float:
        """Calculate MTBF (Mean Time Between Failures)."""
        return 1e9 / fit_rate  # hours
```

### 4. Compliance Reporting (ISO 26262 Automotive)

```python
# spicelab/reliability/compliance.py
from datetime import datetime
from pathlib import Path

class ComplianceReporter:
    """Generate compliance reports for safety standards."""

    def __init__(self, standard: str = "ISO 26262"):
        self.standard = standard

    def generate_iso26262_report(
        self,
        circuit_name: str,
        yield_report: dict,
        worst_case_report: dict,
        reliability_report: dict,
        asil_level: str = "ASIL-B"
    ) -> str:
        """Generate ISO 26262 compliance report."""

        report = f"""
# ISO 26262 Functional Safety Report
**Circuit:** {circuit_name}
**ASIL Level:** {asil_level}
**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Standard:** ISO 26262:2018

## Executive Summary

This report documents the functional safety analysis of the {circuit_name}
circuit in accordance with ISO 26262:2018 Part 11 (Semiconductor).

## Yield Analysis (Part 11, Clause 8)

Manufacturing yield analysis with {yield_report.get('total_samples', 0)} Monte Carlo iterations:

"""
        for spec_name, data in yield_report.items():
            report += f"""
### {spec_name}
- **Yield:** {data['yield_percent']:.2f}%
- **Cpk:** {data['cpk']:.2f}
- **Six Sigma Level:** {data['six_sigma']:.2f}σ
- **Mean:** {data['mean']:.3e}
- **Std Dev:** {data['std']:.3e}

**Assessment:** {"PASS" if data['yield_percent'] >= 99.0 else "FAIL"}
(Target: ≥99% for {asil_level})
"""

        report += """
## Worst-Case Analysis (Part 11, Clause 9)

Process, voltage, and temperature corner analysis:

"""
        for spec_name, data in worst_case_report.items():
            report += f"""
### {spec_name}
- **Worst Corner:** {data['corner']}
- **Value:** {data['value']:.3e}
- **Safety Margin:** {data['margin']:.1f}%

**Assessment:** {"PASS" if data['margin'] >= 20.0 else "FAIL"}
(Target: ≥20% margin for {asil_level})
"""

        report += f"""
## Reliability Analysis (Part 11, Clause 10)

Long-term reliability prediction:

- **FIT Rate:** {reliability_report.get('fit_rate', 0):.1f} FIT
- **MTBF:** {reliability_report.get('mtbf', 0):.1e} hours
- **Mission Profile:** {reliability_report.get('mission_hours', 15000)} hours

**Assessment:** {"PASS" if reliability_report.get('fit_rate', float('inf')) < 100 else "FAIL"}
(Target: <100 FIT for {asil_level})

## Conclusion

Overall Assessment: {"COMPLIANT" if self._check_compliance(yield_report, worst_case_report, reliability_report) else "NON-COMPLIANT"}

---
*Report generated by SpiceLab Reliability Analysis Tool*
"""
        return report

    def generate_do254_report(self, circuit_name: str, **kwargs) -> str:
        """Generate DO-254 (Aerospace) compliance report."""
        # Similar structure for aerospace certification
        ...

    def _check_compliance(self, yield_report, worst_case_report, reliability_report) -> bool:
        """Check if all criteria are met."""
        # Implement compliance logic
        ...
```

## Implementation Plan

### Phase 1: Yield Analysis (Weeks 1-3)
- [ ] YieldAnalyzer class with statistical metrics (Cp, Cpk, Six Sigma)
- [ ] Large-scale Monte Carlo (10k+ iterations) with distributed backend
- [ ] Yield optimization (design centering)
- [ ] Yield visualization (histograms, probability plots)

### Phase 2: Worst-Case Analysis (Weeks 4-5)
- [ ] Automated corner generation (PVT combinations)
- [ ] Worst-case corner simulation
- [ ] Safety margin calculation
- [ ] Corner visualization (spider plots)

### Phase 3: Aging & Reliability (Weeks 6-8)
- [ ] NBTI/PBTI models
- [ ] HCI degradation model
- [ ] Electromigration lifetime prediction
- [ ] TDDB analysis
- [ ] FIT rate and MTBF calculation

### Phase 4: Stress Testing (Weeks 9-10)
- [ ] Overvoltage stress tests
- [ ] Overcurrent stress tests
- [ ] Temperature stress tests
- [ ] ESD stress simulation

### Phase 5: Compliance Reporting (Weeks 11-12)
- [ ] ISO 26262 report generator (automotive)
- [ ] DO-254 report generator (aerospace)
- [ ] IEC 61508 report (industrial)
- [ ] Customizable report templates
- [ ] PDF export with plots

## Success Metrics

### Must Have
- [ ] 10k+ Monte Carlo yield analysis
- [ ] Worst-case corner automation (45+ corners)
- [ ] Aging models (NBTI, HCI, EM)
- [ ] ISO 26262 compliance reports
- [ ] FIT/MTBF calculation

### Should Have
- [ ] Design centering optimization
- [ ] DO-254 aerospace reports
- [ ] Real-time stress test monitoring

### Nice to Have
- [ ] Machine learning for yield prediction
- [ ] Interactive compliance dashboard

## Dependencies

- M2 (Performance) - baseline metrics
- M9 (Optimization) - design centering
- M14 (PDK) - process corners
- M15 (Distributed) - large-scale Monte Carlo

## References

- [ISO 26262:2018](https://www.iso.org/standard/68383.html)
- [DO-254](https://en.wikipedia.org/wiki/DO-254)
- [Reliability Engineering](https://www.rel.com/)
- [Six Sigma Methodology](https://www.isixsigma.com/)
