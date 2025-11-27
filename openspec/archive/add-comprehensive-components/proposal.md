## Why

SpiceLab currently supports only basic passive components (R, L, C), voltage/current sources, controlled sources (E, G, F, H), diodes, switches, and a simple op-amp ideal model. This limits the ability to:
1. Parse and simulate `.asc` files from LTspice that contain transistors, transformers, transmission lines, etc.
2. Build more complex circuits programmatically (amplifiers, power stages, RF circuits)
3. Support vendor component models from major manufacturers

## What Changes

### Phase 1: Ideal/Basic Components (This Proposal)
- **Semiconductors**: BJT (NPN/PNP), MOSFET (NMOS/PMOS), JFET (NJF/PJF), Zener diode
- **Magnetic**: Transformer (coupled inductors), mutual inductance (K)
- **Transmission Lines**: Lossless (T), lossy (O), uniform RC (U)
- **Behavioral**: B-source (arbitrary voltage/current), E/G with expressions
- **Special**: Subcircuit instance (X), voltage/current probes

### Phase 2: Vendor Models (Future Proposal)
- Op-amps with realistic models (TI, ADI, Microchip)
- Voltage regulators (LDO, switching)
- Comparators
- Reference voltages
- Common transistor families (2N2222, 2N7000, IRF series, etc.)

## Impact

- Affected specs: `components` (new capability)
- Affected code:
  - `spicelab/core/components.py` - Add new component classes
  - `spicelab/io/asc_reader.py` - Update parser to recognize new components
  - `spicelab/core/__init__.py` - Export new components
- New files:
  - `spicelab/library/semiconductors.py` - BJT, MOSFET, JFET classes
  - `spicelab/library/magnetic.py` - Transformer, MutualInductance
  - `spicelab/library/transmission.py` - Transmission line models
  - `spicelab/library/behavioral.py` - B-source behavioral elements
- Tests: New test files for each component category
