"""A primary key naming a column the table does not have must not crash.

Five call sites in the relational mapper used a bare
`next((c for c in ... if c.name == pk_c))` with no default. They relied on the
invariant that every name in a table's primary_key also appears in its columns
-- real at column-creation time, but broken later by the weak-entity pass and
by adjudicator-driven identifier rewrites. When it broke, the failure was a
bare StopIteration from deep inside a 492-line function, with nothing to
indicate which table or key was at fault.
"""

from __future__ import annotations

import logging

from src.pipeline.stage2.mapper.relational_mapper import _resolve_pk_column
from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column


def _cols() -> list[Column]:
    return [
        Column(name="alpha_id", data_type=DataType.INTEGER),
        Column(name="label", data_type=DataType.VARCHAR),
    ]


def test_present_column_is_returned():
    col = _resolve_pk_column(
        _cols(), "alpha_id", table_name="ALPHA", purpose="a foreign key"
    )
    assert col is not None
    assert col.data_type is DataType.INTEGER


def test_missing_column_returns_none_instead_of_raising():
    col = _resolve_pk_column(
        _cols(), "ghost_id", table_name="ALPHA", purpose="a foreign key"
    )
    assert col is None


def test_missing_column_names_the_table_and_key_in_the_warning(caplog):
    """The old StopIteration carried no context at all."""
    with caplog.at_level(logging.WARNING):
        _resolve_pk_column(
            _cols(), "ghost_id", table_name="ALPHA", purpose="a foreign key"
        )
    assert "ALPHA" in caplog.text
    assert "ghost_id" in caplog.text
    assert "foreign key" in caplog.text


def test_empty_column_list_is_handled():
    assert _resolve_pk_column([], "any", table_name="T", purpose="anything") is None
