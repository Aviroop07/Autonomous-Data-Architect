"""derived_columns must be validated against the shard schema.

DerivedColumnConstraint carries no `on` tree, so it cannot go through
_canonicalize_list -- which is why it was the one output list checked by
nothing at all. A derivation naming a table the shard does not contain is
exactly the kind of hallucination the deterministic pass exists to catch
before it reaches the DOF graph.
"""

from __future__ import annotations

from src.pipeline.stage3.middleware.deterministic_checker import (
    DeterministicCheckerLoopAgent,
)
from src.pipeline.stage3.models.grain import _SchemaView
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, Schema, Table


class _Derived:
    """Minimal stand-in carrying only the fields the check reads."""

    def __init__(self, target_table, referenced_tables):
        self.target_table = target_table
        self.referenced_tables = referenced_tables


def _view() -> _SchemaView:
    schema = Schema(
        tables=[
            Table(
                name="ALPHA",
                columns=[Column(name="alpha_id", data_type=DataType.INTEGER)],
                primary_key=["alpha_id"],
            )
        ],
        relationships=[],
    )
    return _SchemaView.from_schema(schema)


def _check(items):
    return DeterministicCheckerLoopAgent()._check_derived_columns(items, _view())


def test_known_tables_produce_no_errors():
    assert _check([_Derived("ALPHA", ["ALPHA"])]) == []


def test_unknown_target_table_is_reported():
    errors = _check([_Derived("GHOST", ["ALPHA"])])
    assert len(errors) == 1
    assert "GHOST" in errors[0]


def test_unknown_referenced_table_is_reported():
    errors = _check([_Derived("ALPHA", ["ALPHA", "PHANTOM"])])
    assert len(errors) == 1
    assert "PHANTOM" in errors[0]


def test_the_item_index_is_included_so_the_model_can_locate_it():
    errors = _check([_Derived("ALPHA", ["ALPHA"]), _Derived("GHOST", ["ALPHA"])])
    assert "[1]" in errors[0]


def test_empty_list_is_fine():
    assert _check([]) == []


def test_missing_attributes_do_not_raise():
    """Defensive: the check must not itself become a crash path."""
    assert _check([_Derived(None, None)]) == []
