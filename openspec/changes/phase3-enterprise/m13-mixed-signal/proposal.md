# M13: Mixed-Signal Support

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 10-12 weeks
**Dependencies:** M7 (measurement library), M8 (model management), M10 (I/O optimization)

## Problem Statement

SpiceLab currently focuses on analog/RF circuit simulation and lacks support for mixed-signal designs that combine analog and digital components. Modern SoC (System-on-Chip) designs, power management ICs, ADCs, DACs, and communication interfaces require co-simulation of analog and digital domains with proper timing, protocol, and behavioral modeling.

### Current Gaps
- ❌ No Verilog-AMS integration (digital + analog co-simulation)
- ❌ No VHDL-AMS support
- ❌ No digital timing models (setup/hold, propagation delays)
- ❌ No ADC/DAC behavioral models (SAR, Pipeline, Delta-Sigma topologies)
- ❌ No protocol analyzers (I2C, SPI, UART, CAN bus verification)
- ❌ No bus functional models (BFM) for IP verification
- ❌ Limited mixed-signal debugging capabilities

### Impact
- **Design Scope:** Cannot simulate modern mixed-signal ICs
- **Industry Relevance:** Missing critical capability for SoC/ASIC design
- **Competitiveness:** Commercial tools dominate mixed-signal space
- **User Adoption:** Power electronics and communication circuit designers cannot use SpiceLab

## Objectives

1. **Verilog-AMS integration** - Parser, co-simulation engine, analog/digital interface
2. **VHDL-AMS support** - Basic parser and simulation bridge
3. **Digital timing models** - Setup/hold checks, propagation delay models
4. **ADC/DAC behavioral models** - SAR, Pipeline, Delta-Sigma, Flash architectures
5. **Protocol analyzers** - I2C, SPI, UART, CAN bus monitors and checkers
6. **Bus functional models** - Generic BFM framework for testbench automation
7. **Mixed-signal debugging** - Waveform viewers, protocol decode, timing diagrams
8. **Target:** Enable realistic mixed-signal simulation for SoC designs

## Technical Design

### 1. Verilog-AMS Integration

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    SpiceLab Core                            │
├─────────────────────────────────────────────────────────────┤
│  Circuit Model (Analog + Digital)                           │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │ Analog       │ ←─A/D──→│ Digital      │                 │
│  │ Solver       │  Bridge  │ Simulator    │                 │
│  │ (NGSpice/    │         │ (Verilator/  │                 │
│  │  Xyce)       │         │  Icarus)     │                 │
│  └──────────────┘         └──────────────┘                 │
│         ↑                        ↑                          │
│         │                        │                          │
│  ┌──────┴────────┐        ┌─────┴───────┐                 │
│  │ Analog        │        │ Digital     │                 │
│  │ Components    │        │ Components  │                 │
│  │ (SPICE)       │        │ (Verilog)   │                 │
│  └───────────────┘        └─────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

#### Verilog-AMS Parser
```python
# spicelab/mixed_signal/vams/parser.py
from typing import Protocol
from dataclasses import dataclass
from enum import Enum

class VAMSNodeType(Enum):
    """Verilog-AMS node types."""
    ELECTRICAL = "electrical"
    VOLTAGE = "voltage"
    CURRENT = "current"
    DIGITAL = "logic"
    WIRE = "wire"
    REG = "reg"

@dataclass
class VAMSModule:
    """Parsed Verilog-AMS module."""
    name: str
    ports: list[str]
    port_types: dict[str, VAMSNodeType]
    analog_blocks: list[str]  # analog begin...end
    digital_blocks: list[str]  # always @...
    parameters: dict[str, float]

class VAMSParser:
    """Parse Verilog-AMS files into SpiceLab-compatible format."""

    def __init__(self):
        self.modules: dict[str, VAMSModule] = {}

    def parse_file(self, filepath: str) -> list[VAMSModule]:
        """Parse Verilog-AMS file and extract modules."""
        ...

    def parse_analog_block(self, block: str) -> str:
        """Convert analog block to SPICE netlist."""
        # Convert V(node) → node voltage
        # Convert I(branch) → branch current
        # Convert derivatives (ddt) to time-domain equations
        ...

    def extract_digital_logic(self, block: str) -> str:
        """Extract digital logic for digital simulator."""
        ...

    def identify_interfaces(self, module: VAMSModule) -> list[tuple[str, VAMSNodeType]]:
        """Find analog-digital interface points."""
        ...
```

#### Co-Simulation Engine
```python
# spicelab/mixed_signal/cosim.py
from dataclasses import dataclass
from typing import Protocol, Callable
import xarray as xr

@dataclass
class InterfaceSignal:
    """Signal crossing analog/digital boundary."""
    name: str
    direction: str  # "analog_to_digital" or "digital_to_analog"
    threshold_voltage: float = 1.4  # Logic threshold (V)
    hysteresis: float = 0.2  # Hysteresis window (V)

class AnalogDigitalBridge:
    """Bridge between analog and digital simulators."""

    def __init__(
        self,
        analog_step: float = 1e-9,  # 1ns
        digital_step: float = 1e-9,  # 1ns
        sync_tolerance: float = 1e-12  # 1ps
    ):
        self.analog_step = analog_step
        self.digital_step = digital_step
        self.sync_tolerance = sync_tolerance
        self.interfaces: list[InterfaceSignal] = []

    def add_interface(self, signal: InterfaceSignal):
        """Register an analog-digital interface signal."""
        self.interfaces.append(signal)

    def analog_to_digital(self, voltage: float, signal: InterfaceSignal) -> int:
        """Convert analog voltage to digital logic level."""
        # Apply Schmitt trigger logic with hysteresis
        if voltage > signal.threshold_voltage + signal.hysteresis / 2:
            return 1
        elif voltage < signal.threshold_voltage - signal.hysteresis / 2:
            return 0
        else:
            return -1  # Undefined (metastability)

    def digital_to_analog(self, logic_level: int, signal: InterfaceSignal) -> float:
        """Convert digital logic level to analog voltage."""
        # Simple ideal voltage source
        return 3.3 if logic_level else 0.0  # Assuming 3.3V logic

    def synchronize_time(self, analog_time: float, digital_time: float) -> float:
        """Synchronize analog and digital simulator time steps."""
        return min(analog_time, digital_time)

class MixedSignalSimulator:
    """Orchestrates analog and digital co-simulation."""

    def __init__(
        self,
        analog_circuit: 'Circuit',
        digital_module: VAMSModule,
        bridge: AnalogDigitalBridge
    ):
        self.analog_circuit = analog_circuit
        self.digital_module = digital_module
        self.bridge = bridge
        self.analog_results: xr.Dataset | None = None
        self.digital_results: dict = {}

    async def run_cosimulation(
        self,
        stop_time: float,
        callbacks: dict[str, Callable] | None = None
    ) -> tuple[xr.Dataset, dict]:
        """Run mixed-signal co-simulation."""
        current_time = 0.0

        while current_time < stop_time:
            # Run analog solver for one step
            analog_step_result = await self._step_analog(current_time)

            # Convert analog signals to digital
            for interface in self.bridge.interfaces:
                if interface.direction == "analog_to_digital":
                    analog_value = analog_step_result[interface.name]
                    digital_value = self.bridge.analog_to_digital(analog_value, interface)
                    self._inject_digital_signal(interface.name, digital_value)

            # Run digital simulator for one step
            digital_step_result = await self._step_digital(current_time)

            # Convert digital signals to analog
            for interface in self.bridge.interfaces:
                if interface.direction == "digital_to_analog":
                    digital_value = digital_step_result[interface.name]
                    analog_value = self.bridge.digital_to_analog(digital_value, interface)
                    self._inject_analog_signal(interface.name, analog_value)

            # Synchronize time
            current_time = self.bridge.synchronize_time(
                current_time + self.bridge.analog_step,
                current_time + self.bridge.digital_step
            )

            # Execute callbacks (for monitoring/debugging)
            if callbacks:
                for name, callback in callbacks.items():
                    callback(current_time, analog_step_result, digital_step_result)

        return self.analog_results, self.digital_results
```

### 2. VHDL-AMS Support

**Basic integration for VHDL-AMS analog extensions:**

```python
# spicelab/mixed_signal/vhdl_ams/parser.py
from dataclasses import dataclass

@dataclass
class VHDLEntity:
    """VHDL entity declaration."""
    name: str
    ports: dict[str, str]  # port_name → type
    generics: dict[str, float]

class VHDLAMSParser:
    """Basic VHDL-AMS parser (analog extensions)."""

    def parse_entity(self, vhdl_code: str) -> VHDLEntity:
        """Parse VHDL entity declaration."""
        ...

    def parse_analog_process(self, process: str) -> str:
        """Convert VHDL analog process to SPICE equations."""
        # QUANTITY → SPICE node voltage
        # :== → algebraic equation
        # == → differential equation (convert to .param)
        ...

    def convert_to_spice(self, entity: VHDLEntity) -> str:
        """Generate SPICE subcircuit from VHDL-AMS entity."""
        ...
```

### 3. Digital Timing Models

**Timing constraint checking:**

```python
# spicelab/mixed_signal/timing.py
from dataclasses import dataclass
import xarray as xr

@dataclass
class TimingConstraint:
    """Digital timing constraint."""
    name: str
    constraint_type: str  # "setup", "hold", "propagation", "clock_to_q"
    signal: str
    reference: str
    min_time: float | None = None  # ns
    max_time: float | None = None  # ns

class TimingAnalyzer:
    """Check digital timing constraints."""

    def __init__(self):
        self.constraints: list[TimingConstraint] = []
        self.violations: list[dict] = []

    def add_constraint(self, constraint: TimingConstraint):
        """Add timing constraint to check."""
        self.constraints.append(constraint)

    def check_setup_time(
        self,
        data_signal: xr.DataArray,
        clock_signal: xr.DataArray,
        setup_time: float
    ) -> bool:
        """Verify setup time constraint (data stable before clock edge)."""
        clock_edges = self._find_rising_edges(clock_signal)
        data_transitions = self._find_transitions(data_signal)

        for edge_time in clock_edges:
            # Find last data transition before clock edge
            recent_transitions = [t for t in data_transitions if t < edge_time]
            if recent_transitions:
                last_transition = max(recent_transitions)
                setup_margin = edge_time - last_transition

                if setup_margin < setup_time:
                    self.violations.append({
                        "type": "setup",
                        "time": edge_time,
                        "margin": setup_margin,
                        "required": setup_time
                    })
                    return False
        return True

    def check_hold_time(
        self,
        data_signal: xr.DataArray,
        clock_signal: xr.DataArray,
        hold_time: float
    ) -> bool:
        """Verify hold time constraint (data stable after clock edge)."""
        ...

    def measure_propagation_delay(
        self,
        input_signal: xr.DataArray,
        output_signal: xr.DataArray,
        threshold: float = 0.5  # 50% of Vdd
    ) -> float:
        """Measure propagation delay from input to output."""
        input_transition = self._find_threshold_crossing(input_signal, threshold)
        output_transition = self._find_threshold_crossing(output_signal, threshold)
        return output_transition - input_transition

    def _find_rising_edges(self, signal: xr.DataArray, threshold: float = 1.4) -> list[float]:
        """Find rising edge times in digital signal."""
        ...

    def _find_transitions(self, signal: xr.DataArray) -> list[float]:
        """Find all signal transition times."""
        ...
```

### 4. ADC/DAC Behavioral Models

**Common converter architectures:**

```python
# spicelab/mixed_signal/converters.py
from typing import Literal
import numpy as np
import xarray as xr

class ADCModel:
    """Behavioral ADC model base class."""

    def __init__(
        self,
        bits: int,
        vref: float,
        sample_rate: float,
        architecture: Literal["sar", "pipeline", "delta_sigma", "flash"]
    ):
        self.bits = bits
        self.vref = vref
        self.sample_rate = sample_rate
        self.architecture = architecture
        self.quantization_step = vref / (2 ** bits)

    def quantize(self, analog_value: float) -> int:
        """Quantize analog value to digital code."""
        return int(np.clip(analog_value / self.quantization_step, 0, 2**self.bits - 1))

    def add_noise(self, code: int, snr_db: float) -> int:
        """Add quantization and thermal noise."""
        noise_power = 10 ** (-snr_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_power) * self.quantization_step)
        return int(np.clip(code + noise / self.quantization_step, 0, 2**self.bits - 1))

class SAR_ADC(ADCModel):
    """Successive Approximation Register ADC model."""

    def __init__(self, bits: int, vref: float, sample_rate: float, settling_time: float):
        super().__init__(bits, vref, sample_rate, "sar")
        self.settling_time = settling_time  # DAC settling time

    def convert(self, analog_input: xr.DataArray) -> xr.DataArray:
        """Simulate SAR ADC conversion."""
        samples = []
        sample_times = []

        # Sample at sample_rate
        time = analog_input.time.values
        dt = 1 / self.sample_rate

        for t in np.arange(time[0], time[-1], dt):
            # Sample & Hold
            sampled_value = float(analog_input.interp(time=t))

            # Successive approximation algorithm
            code = 0
            test_value = self.vref / 2

            for bit in range(self.bits - 1, -1, -1):
                if sampled_value >= test_value:
                    code |= (1 << bit)
                    test_value += self.vref / (2 ** (self.bits - bit))
                else:
                    test_value -= self.vref / (2 ** (self.bits - bit))

            samples.append(code)
            sample_times.append(t)

        return xr.DataArray(
            samples,
            coords={"time": sample_times},
            dims=["time"],
            attrs={"bits": self.bits, "vref": self.vref}
        )

class DeltaSigmaADC(ADCModel):
    """Delta-Sigma (Sigma-Delta) ADC model."""

    def __init__(
        self,
        bits: int,
        vref: float,
        oversampling_ratio: int,
        order: int = 2
    ):
        # Effective sample rate after decimation
        sample_rate = oversampling_ratio * (2 ** bits)
        super().__init__(bits, vref, sample_rate, "delta_sigma")
        self.osr = oversampling_ratio
        self.order = order
        self.integrators = [0.0] * order

    def convert(self, analog_input: xr.DataArray) -> xr.DataArray:
        """Simulate Delta-Sigma modulation and decimation."""
        # High-speed 1-bit quantization
        modulator_output = []

        time = analog_input.time.values
        dt = 1 / (self.sample_rate * self.osr)

        for t in np.arange(time[0], time[-1], dt):
            input_val = float(analog_input.interp(time=t))

            # Modulator feedback
            feedback = self.vref if modulator_output and modulator_output[-1] else 0.0

            # Integrate error
            error = input_val - feedback
            for i in range(self.order):
                self.integrators[i] += error
                error = self.integrators[i]

            # 1-bit quantization
            bit_out = 1 if self.integrators[-1] > 0 else 0
            modulator_output.append(bit_out)

        # Decimation filter (sinc filter)
        decimated = self._sinc_filter(modulator_output, self.osr, self.order)

        return xr.DataArray(
            decimated,
            coords={"time": np.arange(0, len(decimated) * dt * self.osr, dt * self.osr)},
            dims=["time"]
        )

    def _sinc_filter(self, data: list[int], decimation: int, order: int) -> list[int]:
        """Apply sinc^N decimation filter."""
        ...

class DACModel:
    """Behavioral DAC model."""

    def __init__(
        self,
        bits: int,
        vref: float,
        update_rate: float,
        settling_time: float,
        inl_lsb: float = 0.5,  # Integral Nonlinearity
        dnl_lsb: float = 0.5   # Differential Nonlinearity
    ):
        self.bits = bits
        self.vref = vref
        self.update_rate = update_rate
        self.settling_time = settling_time
        self.inl_lsb = inl_lsb
        self.dnl_lsb = dnl_lsb
        self.step_size = vref / (2 ** bits)

    def convert(self, digital_codes: xr.DataArray) -> xr.DataArray:
        """Convert digital codes to analog voltage with non-idealities."""
        analog_output = []

        for code in digital_codes.values:
            # Ideal DAC output
            ideal_voltage = code * self.step_size

            # Add INL error
            inl_error = np.random.uniform(-self.inl_lsb, self.inl_lsb) * self.step_size

            # Add DNL error (affects step size)
            dnl_error = np.random.uniform(-self.dnl_lsb, self.dnl_lsb) * self.step_size

            actual_voltage = ideal_voltage + inl_error + dnl_error
            analog_output.append(actual_voltage)

        return xr.DataArray(
            analog_output,
            coords=digital_codes.coords,
            dims=digital_codes.dims
        )
```

### 5. Protocol Analyzers

**Digital bus protocol checkers:**

```python
# spicelab/mixed_signal/protocols.py
from dataclasses import dataclass
from enum import Enum
import xarray as xr

class I2CState(Enum):
    IDLE = "idle"
    START = "start"
    ADDRESS = "address"
    DATA = "data"
    ACK = "ack"
    STOP = "stop"

@dataclass
class I2CTransaction:
    """I2C bus transaction."""
    start_time: float
    address: int
    read_write: bool  # True = Read, False = Write
    data: list[int]
    acks: list[bool]
    stop_time: float

class I2CAnalyzer:
    """I2C protocol analyzer and checker."""

    def __init__(self, sda_signal: str, scl_signal: str):
        self.sda_signal = sda_signal
        self.scl_signal = scl_signal
        self.transactions: list[I2CTransaction] = []
        self.errors: list[dict] = []

    def decode(self, results: xr.Dataset) -> list[I2CTransaction]:
        """Decode I2C bus transactions from simulation results."""
        sda = results[self.sda_signal]
        scl = results[self.scl_signal]

        state = I2CState.IDLE
        current_byte = 0
        bit_count = 0
        transaction = None

        for i in range(1, len(sda)):
            sda_curr = sda[i].values > 1.4
            sda_prev = sda[i-1].values > 1.4
            scl_curr = scl[i].values > 1.4
            scl_prev = scl[i-1].values > 1.4

            # Detect START condition (SDA falling while SCL high)
            if scl_curr and scl_prev and (not sda_curr) and sda_prev:
                state = I2CState.START
                transaction = I2CTransaction(
                    start_time=float(sda.time[i]),
                    address=0,
                    read_write=False,
                    data=[],
                    acks=[],
                    stop_time=0.0
                )
                bit_count = 0
                current_byte = 0

            # Detect STOP condition (SDA rising while SCL high)
            elif scl_curr and scl_prev and sda_curr and (not sda_prev):
                if transaction:
                    transaction.stop_time = float(sda.time[i])
                    self.transactions.append(transaction)
                state = I2CState.IDLE

            # Sample data on SCL rising edge
            elif scl_curr and (not scl_prev):
                if state in [I2CState.START, I2CState.ADDRESS, I2CState.DATA]:
                    current_byte = (current_byte << 1) | (1 if sda_curr else 0)
                    bit_count += 1

                    if bit_count == 8:
                        if state == I2CState.START:
                            # First byte is address
                            transaction.address = current_byte >> 1
                            transaction.read_write = bool(current_byte & 1)
                            state = I2CState.ACK
                        elif state == I2CState.DATA:
                            transaction.data.append(current_byte)
                            state = I2CState.ACK

                        bit_count = 0
                        current_byte = 0

                elif state == I2CState.ACK:
                    # Check ACK bit
                    ack = not sda_curr  # ACK is low
                    if transaction:
                        transaction.acks.append(ack)
                    state = I2CState.DATA

        return self.transactions

    def check_timing(
        self,
        results: xr.Dataset,
        spec: dict  # {"t_su_dat": 100e-9, "t_hd_dat": 0, ...}
    ) -> list[dict]:
        """Check I2C timing specifications."""
        # Check setup time, hold time, clock frequency, etc.
        ...

class SPIAnalyzer:
    """SPI protocol analyzer."""

    def __init__(
        self,
        mosi: str,
        miso: str,
        sclk: str,
        cs: str,
        mode: int = 0  # SPI mode (0-3)
    ):
        self.mosi = mosi
        self.miso = miso
        self.sclk = sclk
        self.cs = cs
        self.mode = mode

    def decode(self, results: xr.Dataset) -> list[dict]:
        """Decode SPI transactions."""
        ...

class UARTAnalyzer:
    """UART protocol analyzer."""

    def __init__(
        self,
        tx_signal: str,
        baud_rate: int,
        data_bits: int = 8,
        parity: Literal["none", "even", "odd"] = "none",
        stop_bits: int = 1
    ):
        self.tx_signal = tx_signal
        self.baud_rate = baud_rate
        self.data_bits = data_bits
        self.parity = parity
        self.stop_bits = stop_bits

    def decode(self, results: xr.Dataset) -> list[dict]:
        """Decode UART frames."""
        bit_period = 1 / self.baud_rate
        tx = results[self.tx_signal]
        frames = []

        # Detect start bit (high-to-low transition)
        # Sample data bits at bit period intervals
        # Check parity
        # Detect stop bit
        ...

        return frames
```

### 6. Bus Functional Models (BFM)

**Testbench automation framework:**

```python
# spicelab/mixed_signal/bfm.py
from typing import Protocol, Callable
import asyncio

class BusFunctionalModel(Protocol):
    """Protocol for all BFMs."""

    async def write(self, address: int, data: int) -> None:
        """Write data to address."""
        ...

    async def read(self, address: int) -> int:
        """Read data from address."""
        ...

    async def wait_cycles(self, n: int) -> None:
        """Wait for N clock cycles."""
        ...

class I2C_BFM:
    """I2C Master Bus Functional Model."""

    def __init__(
        self,
        sda_control: Callable,  # Function to drive SDA
        scl_control: Callable,  # Function to drive SCL
        clock_freq: float = 100e3  # 100 kHz standard mode
    ):
        self.sda = sda_control
        self.scl = scl_control
        self.clock_freq = clock_freq
        self.bit_period = 1 / clock_freq

    async def start_condition(self):
        """Generate I2C START condition."""
        self.sda(True)
        self.scl(True)
        await asyncio.sleep(self.bit_period / 2)
        self.sda(False)  # SDA falls while SCL high
        await asyncio.sleep(self.bit_period / 2)
        self.scl(False)

    async def stop_condition(self):
        """Generate I2C STOP condition."""
        self.sda(False)
        self.scl(True)
        await asyncio.sleep(self.bit_period / 2)
        self.sda(True)  # SDA rises while SCL high
        await asyncio.sleep(self.bit_period / 2)

    async def write_byte(self, byte: int) -> bool:
        """Write byte and return ACK status."""
        for bit in range(7, -1, -1):
            self.sda(bool(byte & (1 << bit)))
            await asyncio.sleep(self.bit_period / 2)
            self.scl(True)
            await asyncio.sleep(self.bit_period / 2)
            self.scl(False)

        # Read ACK
        self.sda(True)  # Release SDA
        await asyncio.sleep(self.bit_period / 2)
        self.scl(True)
        # Sample ACK (would need feedback from simulation)
        ack = False  # Placeholder
        await asyncio.sleep(self.bit_period / 2)
        self.scl(False)

        return ack

    async def write(self, address: int, data: list[int]):
        """Write data bytes to I2C device."""
        await self.start_condition()
        await self.write_byte((address << 1) | 0)  # Write bit = 0
        for byte in data:
            await self.write_byte(byte)
        await self.stop_condition()

class SPI_BFM:
    """SPI Master Bus Functional Model."""

    def __init__(
        self,
        mosi_control: Callable,
        miso_read: Callable,
        sclk_control: Callable,
        cs_control: Callable,
        mode: int = 0,
        clock_freq: float = 1e6
    ):
        self.mosi = mosi_control
        self.miso = miso_read
        self.sclk = sclk_control
        self.cs = cs_control
        self.mode = mode
        self.clock_freq = clock_freq

    async def transfer(self, data_out: int, bits: int = 8) -> int:
        """SPI transfer (simultaneous TX/RX)."""
        data_in = 0

        self.cs(False)  # Assert chip select
        await asyncio.sleep(1 / self.clock_freq / 4)

        for bit in range(bits - 1, -1, -1):
            # Set MOSI
            self.mosi(bool(data_out & (1 << bit)))

            # Clock
            self.sclk(True)
            await asyncio.sleep(1 / self.clock_freq / 2)

            # Sample MISO
            if self.miso():
                data_in |= (1 << bit)

            self.sclk(False)
            await asyncio.sleep(1 / self.clock_freq / 2)

        self.cs(True)  # Deassert chip select
        return data_in
```

## Implementation Plan

### Phase 1: Verilog-AMS Foundation (Weeks 1-4)
- [ ] Implement Verilog-AMS parser (analog blocks, electrical nodes)
- [ ] Create analog-digital bridge interface
- [ ] Integrate with Verilator/Icarus Verilog for digital simulation
- [ ] Build co-simulation orchestrator
- [ ] Write 10+ mixed-signal test cases
- [ ] Document Verilog-AMS subset supported

### Phase 2: Digital Timing & Models (Weeks 5-6)
- [ ] Implement timing constraint checker (setup/hold)
- [ ] Add propagation delay measurement
- [ ] Create digital component library (gates, flip-flops)
- [ ] Build timing violation reporter
- [ ] Add 20+ timing test cases

### Phase 3: ADC/DAC Models (Weeks 7-8)
- [ ] Implement SAR ADC behavioral model
- [ ] Implement Pipeline ADC model
- [ ] Implement Delta-Sigma ADC model
- [ ] Implement Flash ADC model
- [ ] Create DAC models with INL/DNL
- [ ] Add ENOB (Effective Number of Bits) measurement
- [ ] Write converter tutorial examples

### Phase 4: Protocol Analyzers (Weeks 9-10)
- [ ] Implement I2C analyzer and decoder
- [ ] Implement SPI analyzer (all 4 modes)
- [ ] Implement UART decoder with parity checking
- [ ] Add CAN bus basic analyzer
- [ ] Create protocol violation checker
- [ ] Write protocol verification examples

### Phase 5: BFM & Integration (Weeks 11-12)
- [ ] Build I2C BFM (master and slave)
- [ ] Build SPI BFM
- [ ] Create generic BFM framework
- [ ] Add VHDL-AMS basic support
- [ ] Integration testing (complete SoC examples)
- [ ] Performance optimization
- [ ] Documentation and tutorials

## Success Metrics

### Functionality (Must Have)
- [ ] **Verilog-AMS co-simulation** working for 10+ test circuits
- [ ] **ADC/DAC models** match theoretical ENOB ±0.5 bits
- [ ] **Protocol analyzers** correctly decode 100+ transactions
- [ ] **Timing checks** detect 99%+ of violations
- [ ] **BFMs** automate testbench for 5+ protocols

### Performance
- [ ] Co-simulation overhead **<20%** vs pure analog
- [ ] Protocol decode time **<100ms** for 10k samples
- [ ] Timing checks **<1s** for 1M time points

### Documentation
- [ ] Mixed-signal tutorial (beginner to advanced)
- [ ] 15+ example circuits (ADC, DAC, SoC subsystems)
- [ ] API reference complete
- [ ] Migration guide from other tools (SystemVerilog, VHDL)

## Dependencies

- M7 (Measurement Library) - for ADC/DAC performance specs
- M8 (Model Management) - for storing Verilog-AMS models
- M10 (I/O Optimization) - for handling large digital waveforms

## References

- [Verilog-AMS Language Reference Manual](https://www.accellera.org/downloads/standards/v-ams)
- [VHDL-AMS IEEE Standard](https://standards.ieee.org/standard/1076_1-2017.html)
- [Verilator User Guide](https://verilator.org/guide/latest/)
- [Data Converter Architectures (Razavi)](https://www.springer.com/book/9780792375876)
