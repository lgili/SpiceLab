from pathlib import Path

from spicelab.core.components import OpAmpIdeal
from spicelab.io.ltspice_asc import circuit_from_asc


def test_universal_opamp_imports_as_opampideal() -> None:
    p = Path("examples/PT1000_circuit_1.asc")
    assert p.exists(), "example ASC must exist for test"
    c = circuit_from_asc(p)
    # Find at least one OpAmpIdeal in components
    types = [type(comp) for comp in c._components]
    assert any(t is OpAmpIdeal for t in types), f"OpAmpIdeal not found in components: {types}"
