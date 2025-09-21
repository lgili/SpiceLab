"""Metadata schemas for component categories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldSchema:
    name: str
    types: tuple[type[Any], ...]
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class MetadataSchema:
    category: str
    fields: tuple[FieldSchema, ...]
    allow_extra: bool = False

    def validate(self, data: Mapping[str, Any]) -> None:
        missing = [field.name for field in self.fields if field.required and field.name not in data]
        if missing:
            raise ValueError(
                f"Metadata for category '{self.category}' missing required key(s): "
                f"{', '.join(missing)}"
            )
        if not self.allow_extra:
            allowed = {field.name for field in self.fields}
            extras = set(data.keys()) - allowed
            if extras:
                raise ValueError(
                    f"Metadata for category '{self.category}' contains unsupported key(s): "
                    f"{', '.join(sorted(extras))}"
                )
        for field in self.fields:
            if field.name not in data:
                continue
            value = data[field.name]
            if not isinstance(value, field.types):
                expected = " or ".join(t.__name__ for t in field.types)
                raise ValueError(
                    f"Metadata '{field.name}' for category '{self.category}' must be of type "
                    f"{expected}"
                )


_SCHEMAS: dict[str, MetadataSchema] = {}


def register_metadata_schema(schema: MetadataSchema, *, overwrite: bool = False) -> None:
    if not overwrite and schema.category in _SCHEMAS:
        raise ValueError(f"Schema already registered for category '{schema.category}'")
    _SCHEMAS[schema.category] = schema


def get_metadata_schema(category: str) -> MetadataSchema | None:
    return _SCHEMAS.get(category)


def validate_metadata(category: str, metadata: Mapping[str, Any]) -> None:
    schema = get_metadata_schema(category)
    if schema:
        schema.validate(metadata)


register_metadata_schema(
    MetadataSchema(
        category="resistor",
        fields=(
            FieldSchema("value", (str, float, int), required=True),
            FieldSchema("tolerance", (str,)),
            FieldSchema("power", (str,)),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="capacitor",
        fields=(
            FieldSchema("value", (str, float, int), required=True),
            FieldSchema("dielectric", (str,)),
            FieldSchema("voltage", (str,)),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="inductor",
        fields=(
            FieldSchema("value", (str, float, int), required=True),
            FieldSchema("current_rating", (str,)),
            FieldSchema("dcr", (str,)),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="diode",
        fields=(
            FieldSchema("model", (str,), required=True),
            FieldSchema("model_card", (str,), description="SPICE .model directive"),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="mosfet",
        fields=(
            FieldSchema("model", (str,), required=True),
            FieldSchema("polarity", (str,)),
            FieldSchema("default_params", (str,)),
            FieldSchema("model_card", (str,), description="SPICE .model directive"),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="bjt",
        fields=(
            FieldSchema("model", (str,), required=True),
            FieldSchema("type", (str,)),
            FieldSchema("model_card", (str,), description="SPICE .model directive"),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="opamp",
        fields=(
            FieldSchema("description", (str,), required=True),
            FieldSchema("variant", (str,), required=True),
            FieldSchema("gain", (str, float, int)),
            FieldSchema("subckt", (str,)),
            FieldSchema("include", (str, list, tuple)),
            FieldSchema("model_card", (str,)),
        ),
    ),
    overwrite=True,
)
register_metadata_schema(
    MetadataSchema(
        category="switch",
        fields=(
            FieldSchema("model", (str,), required=True),
            FieldSchema("model_card", (str,), description=".model directive"),
            FieldSchema("type", (str,)),
            FieldSchema("description", (str,)),
        ),
    ),
    overwrite=True,
)

__all__ = [
    "FieldSchema",
    "MetadataSchema",
    "register_metadata_schema",
    "get_metadata_schema",
    "validate_metadata",
]
