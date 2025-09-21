# Component Library

CAT provides a lightweight registry that lets you register custom component
factories and instantiate them later by name. This makes it easy to distribute
manufacturer-specific parts (for example, a diode with a recommended `.model`
card) or project-specific helpers.

```python
from cat.library import create_component, register_component
from cat.core.components import Resistor

register_component("custom.res", lambda ref, value: Resistor(ref, value))
part = create_component("custom.res", ref="R1", value="4k7")
```

## Built-in catalog

The `cat.library` package ships with a starter catalog. For example:

```python
from cat.library import create_component, get_component_spec

d = create_component("diode.1n4007", ref="D1")
spec = get_component_spec("diode.1n4007")
print(spec.metadata["model_card"])  # recommended .model statement
```

The catalog is organised by category. You can discover available entries via
`list_components()` or `list_components(category="diode")`.

## Creating reusable parts

A registered factory can accept any positional/keyword arguments. Metadata can
be attached during registration to capture links, model cards, etc.

```python
from cat.core.components import Diode
from cat.library import register_component

register_component(
    "diode.bat54",
    lambda ref, model="DBAT54": Diode(ref, model),
    category="diode",
    metadata={"datasheet": "https://www.onsemi.com/pdf/datasheet/bat54-d.pdf"},
)
```

Factories can also return subclasses of `Component`, enabling more elaborate
behaviour.

Remember to unregister temporary factories used inside tests by calling
`unregister_component(name)`.

Many entries expose a `model_card` metadata field containing a ready-to-use
`.model` or `.include` statement. For example::

    from cat.library import get_component_spec
    spec = get_component_spec('mosfet.bss138')
    circuit.add_directive(spec.metadata['model_card'])

You can discover parts programmatically using ``search_components``::

    from cat.library import search_components
    mosfets = search_components(metadata={'polarity': 'n-channel'})

This supports filtering by name substring, category, metadata key/value pairs,
and custom predicates.

To inject the recommended SPICE directives into a circuit once::

    from cat.core.circuit import Circuit
    from cat.library import get_component_spec, apply_metadata_to_circuit

    circuit = Circuit('demo')
    spec = get_component_spec('diode.1n4007')
    apply_metadata_to_circuit(circuit, spec)  # adds .model/.include only if absent

### Importing external catalogs

You can register many parts at once using the import helpers::

    from cat.library.importers import import_catalog_from_json
    import_catalog_from_json('my_components.json')

The JSON file should contain a list of entries with `name`, `factory` (or a
custom `factory_builder`), optional `category`, and metadata fields. Similar
helpers exist for CSV data (``import_catalog_from_csv``) and in-memory
structures (``import_catalog_from_mapping``).
