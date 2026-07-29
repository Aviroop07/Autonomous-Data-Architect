"""A composite primary key must survive the `pk` -> `primary_key` coercion.

Found while wiring the first end-to-end run against the benchmark. The
ground-truth schemas use `pk`, a STRING for a single-column key and a LIST for a
composite one; `Table` uses `primary_key: List[str]`. The coercion wrapped
whatever it found unconditionally:

    data["primary_key"] = [pk_val] if pk_val else []

so a list became a NESTED list and then failed validation outright. Measured
before the fix: 103 tables across 58 of the benchmark's 150 cases carry a
list-valued `pk`, so 39% of the dataset could not be loaded into `Schema` at all
-- and with it, the entire schema-metric suite (structural, capacity, KDC) could
not run on those cases. The ones lost were disproportionately the junction-heavy
schemas, which are exactly where composite keys live and exactly the interesting
cases to measure.

The same coercion existed verbatim in schema_patch.SimplifiedTable, so a patch
proposing a composite key broke identically. Both are covered here.
"""

from __future__ import annotations

from src.util.schema_model.schema import Schema, Table
from src.util.schema_ops.schema_patch import SimplifiedTable

_COLS = [
    {"name": "student_id", "data_type": "INTEGER"},
    {"name": "course_id", "data_type": "INTEGER"},
    {"name": "grade", "data_type": "VARCHAR"},
]


def test_a_string_pk_becomes_a_single_element_list() -> None:
    """The case that always worked -- kept so the fix cannot regress it."""
    table = Table.model_validate(
        {"name": "STUDENT", "columns": _COLS, "pk": "student_id"}
    )
    assert table.primary_key == ["student_id"]


def test_a_list_pk_is_preserved_not_nested() -> None:
    """The defect. This raised a ValidationError before the fix."""
    table = Table.model_validate(
        {"name": "ENROLMENT", "columns": _COLS, "pk": ["student_id", "course_id"]}
    )
    assert table.primary_key == ["student_id", "course_id"]


def test_an_empty_pk_is_an_empty_list() -> None:
    for empty in ("", [], None):
        table = Table.model_validate({"name": "T", "columns": _COLS, "pk": empty})
        assert table.primary_key == [], f"pk={empty!r}"


def test_an_explicit_primary_key_still_wins_over_pk() -> None:
    """`primary_key` is the real field; `pk` is only the inbound alias, so an
    explicit value must not be overwritten."""
    table = Table.model_validate(
        {
            "name": "ENROLMENT",
            "columns": _COLS,
            "primary_key": ["student_id", "course_id"],
            "pk": "student_id",
        }
    )
    assert table.primary_key == ["student_id", "course_id"]


def test_a_string_primary_key_is_also_wrapped() -> None:
    table = Table.model_validate(
        {"name": "STUDENT", "columns": _COLS, "primary_key": "student_id"}
    )
    assert table.primary_key == ["student_id"]


def test_the_pk_convenience_property_reports_the_first_column() -> None:
    """Documented lossy shortcut: `.pk` returns one column, so anything that
    needs the whole key must read `primary_key`. Pinned so the shortcut is not
    mistaken for the key itself."""
    table = Table.model_validate(
        {"name": "ENROLMENT", "columns": _COLS, "pk": ["student_id", "course_id"]}
    )
    assert table.pk == "student_id"
    assert len(table.primary_key) == 2


def test_a_whole_schema_with_a_composite_key_loads() -> None:
    schema = Schema.model_validate(
        {
            "tables": [
                {"name": "STUDENT", "columns": _COLS, "pk": "student_id"},
                {
                    "name": "ENROLMENT",
                    "columns": _COLS,
                    "pk": ["student_id", "course_id"],
                },
            ],
            "relationships": [],
        }
    )
    by_name = {t.name: t for t in schema.tables}
    assert by_name["ENROLMENT"].primary_key == ["student_id", "course_id"]
    assert by_name["STUDENT"].primary_key == ["student_id"]


def test_the_patch_model_coerces_identically() -> None:
    """schema_patch carried the same code, so an ADD_TABLE proposing a composite
    key failed the same way."""
    table = SimplifiedTable.model_validate(
        {"name": "ENROLMENT", "columns": _COLS, "pk": ["student_id", "course_id"]}
    )
    assert table.primary_key == ["student_id", "course_id"]
