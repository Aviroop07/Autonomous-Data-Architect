"""The relational schema object model.

These types were in src/pipeline/stage2/models/ because Stage 2 is what
PRODUCES a schema. But 12 modules under src/util/ import them, so util/ -- which
is meant to be stage-agnostic -- could not be imported without pulling in Stage
2. Producing a type is not the same as owning it: Stages 2, 3 and 4, the
evaluation harness and the constraint model all speak in Schema/Table/Column, so
it belongs in shared code.

Import from the submodules (`.schema`, `.data_types`) or from here; both work.
"""

from src.util.schema_model.data_types import DataType
from src.util.schema_model.registry import TableFactRegistry
from src.util.schema_model.schema import (
    FORBIDDEN_TABLE_SUFFIXES,
    Column,
    CompositeUnique,
    ForeignKey,
    Schema,
    Table,
    looks_singular_noun,
    to_snake_case,
)

__all__ = [
    "Column",
    "CompositeUnique",
    "DataType",
    "FORBIDDEN_TABLE_SUFFIXES",
    "ForeignKey",
    "Schema",
    "Table",
    "TableFactRegistry",
    "looks_singular_noun",
    "to_snake_case",
]
