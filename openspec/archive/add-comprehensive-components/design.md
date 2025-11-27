## Context

SpiceLab needs comprehensive component support to:
1. Parse LTspice `.asc` schematics containing transistors, transformers, etc.
2. Allow users to build complex circuits programmatically
3. Provide a foundation for adding vendor-specific models later

Current state: Only R, L, C, V sources, I sources, controlled sources (E/G/F/H), diode, switches, and ideal op-amp are implemented.

## Goals / Non-Goals

**Goals:**
- Support all standard SPICE primitive elements
- Maintain consistent API with existing components (ports, spice_card, helpers)
- Enable parsing of ~90% of common LTspice schematics
- Keep ideal/simple models as default (no complex vendor models in Phase 1)

**Non-Goals:**
- Vendor-specific accurate models (deferred to Phase 2)
- GUI schematic capture
- Component parameter extraction from datasheets

## Decisions

### 1. Component Organization

**Decision:** Split components into category modules under `spicelab/library/`

```
spicelab/
├── core/
│   └── components.py          # Base Component class + basic passives (R, L, C)
└── library/
    ├── __init__.py            # Re-export all for convenience
    ├── sources.py             # V/I sources (move from core/components.py)
    ├── semiconductors.py      # BJT, MOSFET, JFET, Zener
    ├── magnetic.py            # K (mutual), Transformer
    ├── transmission.py        # T, O, U lines
    └── behavioral.py          # B-sources, expression-based E/G
```

**Rationale:** Keeps `core/components.py` focused on primitives while allowing growth in `library/`.

### 2. SPICE Card Format

**Decision:** Follow standard SPICE3/ngspice syntax as primary, with LTspice compatibility.

| Element | SPICE Card Format | Ports |
|---------|-------------------|-------|
| BJT | `Q<ref> C B E [S] <model>` | c, b, e, [s] |
| MOSFET | `M<ref> D G S B <model> [params]` | d, g, s, b |
| JFET | `J<ref> D G S <model>` | d, g, s |
| Zener | `D<ref> A C <model>` | a, c |
| Mutual | `K<ref> L1 L2 <coupling>` | (references) |
| TLine | `T<ref> P1+ P1- P2+ P2- Z0=<z> TD=<td>` | p1p, p1n, p2p, p2n |
| B-src | `B<ref> N+ N- V=<expr>` or `I=<expr>` | p, n |
| Subckt | `X<ref> <nodes...> <subckt_name> [params]` | dynamic |

### 3. Model Handling

**Decision:** Components reference model names (strings), models are added via `Circuit.add_directive()`.

```python
# User adds model
circuit.add_directive(".model 2N2222 NPN(BF=200 IS=1e-14)")

# Component references it
q1 = BJT("1", model="2N2222")
circuit.add(q1)
```

**Rationale:** Keeps component classes simple, allows flexible model definitions.

### 4. Port Naming Convention

**Decision:** Use lowercase single letters or short names matching SPICE conventions.

| Component | Ports |
|-----------|-------|
| BJT | c (collector), b (base), e (emitter), s (substrate, optional) |
| MOSFET | d (drain), g (gate), s (source), b (bulk/body) |
| JFET | d (drain), g (gate), s (source) |
| TLine | p1p, p1n (port 1 +/-), p2p, p2n (port 2 +/-) |

### 5. Helper Factory Naming

**Decision:** Single uppercase letter or short name matching SPICE element letter.

```python
Q(model="2N2222")           # BJT
M(model="NMOS_BASIC")       # MOSFET
J(model="2N5457")           # JFET
DZ(model="1N4733")          # Zener
K(l1, l2, coupling=0.99)    # Mutual inductance
BV(expr="V(in)*2")          # B-source voltage
BI(expr="I(Vin)*0.5")       # B-source current
X(name="OPAMP", nodes=...)  # Subcircuit instance
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Model compatibility across engines | Test with ngspice, LTspice, Xyce; document differences |
| Complex parameter handling for MOSFETs | Start with minimal params (W, L, model), extend later |
| Expression parsing for B-sources | Pass expressions as strings, let SPICE engine parse |
| Breaking changes to existing code | All new code, no modifications to existing component APIs |

## Migration Plan

1. Add new files in `spicelab/library/` - no breaking changes
2. Re-export from `spicelab.core` for convenience
3. Update `asc_reader.py` incrementally (one component type at a time)
4. Add deprecation warnings if any existing APIs change

## Open Questions

1. Should MOSFET default to 3-terminal (D, G, S) or 4-terminal (D, G, S, B)?
   - **Proposed:** 4-terminal with B defaulting to S if not connected

2. Should we support instance parameters (e.g., `M=2` for parallel devices)?
   - **Proposed:** Yes, via optional `mult` parameter

3. How to handle LTspice-specific extensions (e.g., `Rser`, `Cpar` on components)?
   - **Proposed:** Defer to Phase 2, focus on standard SPICE first
