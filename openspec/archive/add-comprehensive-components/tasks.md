## 1. Semiconductors

- [x] 1.1 Implement `BJT` class (NPN/PNP) with collector, base, emitter ports (existing in library/transistors.py)
- [x] 1.2 Implement `MOSFET` class (NMOS/PMOS) with drain, gate, source, bulk ports (existing in library/transistors.py)
- [x] 1.3 Implement `JFET` class (NJF/PJF) with drain, gate, source ports
- [x] 1.4 Implement `ZenerDiode` class with model parameter support
- [x] 1.5 Add unit tests for all semiconductor components
- [x] 1.6 Add helper factories: `Q()`, `M()`, `JF()`, `DZ()`

## 2. Magnetic Components

- [x] 2.1 Implement `MutualInductance` (K) class for coupled inductors
- [x] 2.2 Implement `Transformer` class (ideal with coupled inductors)
- [x] 2.3 Add unit tests for magnetic components
- [x] 2.4 Add helper factories: `MK()`, `XFMR()`

## 3. Transmission Lines

- [x] 3.1 Implement `TLine` class (lossless transmission line - T element)
- [x] 3.2 Implement `TLineLossy` class (lossy line - O element)
- [x] 3.3 Implement `TLineRC` class (uniform RC line - U element)
- [x] 3.4 Add unit tests for transmission line components
- [x] 3.5 Add helper factories: `TLINE()`, `OLINE()`, `ULINE()`

## 4. Behavioral Sources

- [x] 4.1 Implement `BVoltage` class (arbitrary voltage expression)
- [x] 4.2 Implement `BCurrent` class (arbitrary current expression)
- [x] 4.3 Implement `VCVSExpr` class (E-source with expression) - alias for BVoltage
- [x] 4.4 Implement `VCCSExpr` class (G-source with expression) - alias for BCurrent
- [x] 4.5 Add unit tests for behavioral sources
- [x] 4.6 Add helper factories: `BV()`, `BI()`, `EExpr()`, `GExpr()`

## 5. Subcircuit Support

- [x] 5.1 Implement `SubcktInstance` class (X element)
- [x] 5.2 Support parameter passing to subcircuits
- [x] 5.3 Add unit tests for subcircuit instances
- [x] 5.4 Add helper factory: `XSUB()`

## 6. Probe Components

- [x] 6.1 Implement `VoltageProbe` class (for explicit voltage measurement)
- [x] 6.2 Implement `CurrentProbe` class (zero-volt source for current measurement)
- [x] 6.3 Add unit tests for probe components
- [x] 6.4 Add helper factories: `VPROBE()`, `IPROBE()`

## 7. ASC Parser Updates

- [x] 7.1 Update `ltspice_asc.py` to recognize BJT symbols (npn, pnp)
- [x] 7.2 Update `ltspice_asc.py` to recognize MOSFET symbols (nmos, nmos4, pmos, pmos4)
- [x] 7.3 Update `ltspice_asc.py` to recognize JFET (njf, pjf) and Zener symbols
- [x] 7.4 Add mapping from LTspice symbol names to SpiceLab classes (SYMBOL_LIBRARY, COMPONENT_TO_SYMBOL)
- [x] 7.5 Add integration tests with sample .asc files (tests/test_asc_transistors.py - 38 tests)

## 8. Documentation

- [x] 8.1 Add docstrings with examples for all new components
- [x] 8.2 Update API reference documentation (docs/components.md)
- [x] 8.3 Add tutorial notebook demonstrating new components (notebooks/07-advanced-components.ipynb)

## 9. Export Updates

- [x] 9.1 Update `spicelab/core/__init__.py` to export new components
- [x] 9.2 Ensure backward compatibility with existing code
