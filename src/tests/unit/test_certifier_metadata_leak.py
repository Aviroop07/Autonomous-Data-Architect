"""The compliance certifier invented a column named `is_nullable`.

Found in a live Stage 2 run: cert_report carried
`ADD_COLUMN CLUB_MEMBERSHIP.is_nullable BOOLEAN` with empty source_fact_ids,
and it was applied. The agent had been handed `schema.model_dump_json()`, in
which EVERY column carries an `is_nullable` key, and it echoed one back as a
column name. Patch validation rejected five malformed ADD_RELATIONSHIP patches
in that same run but passed this one, because adding a column that does not yet
exist is structurally valid.

Two layers are pinned here:
  - the root cause: the agent is shown rendered text that never names a field
    of the schema model, so the vocabulary that produced this is simply absent;
  - a backstop for any other producer: a column named after a field of the
    schema model is metadata, not content.

The backstop deliberately does NOT reject on missing provenance.
schema_patch.py records a measured decision that punishing absent
source_fact_ids discards mostly-correct schema fixes, and that trade is
unchanged -- this rejects a name that cannot denote a domain column at all,
which is a different claim.
"""

from __future__ import annotations

from src.util.schema_model.data_types import DataType
from src.util.schema_model.render import schema_to_prompt_text
from src.util.schema_model.schema import (
    Column,
    CompositeUnique,
    ForeignKey,
    Schema,
    Table,
)
from src.util.schema_ops.schema_patch import (
    AddColumnPatch,
    _schema_model_field_names,
)


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="CLUB",
                columns=[
                    Column(name="club_id", data_type=DataType.INTEGER),
                    Column(name="name", data_type=DataType.VARCHAR),
                ],
                primary_key=["club_id"],
                unique=[CompositeUnique(columns=["name"])],
            ),
            Table(
                name="CLUB_MEMBERSHIP",
                columns=[
                    Column(name="club_id", data_type=DataType.INTEGER),
                    Column(name="date_left", data_type=DataType.DATE, is_nullable=True),
                ],
                primary_key=["club_id"],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="CLUB_MEMBERSHIP",
                referencing_column="club_id",
                referred_table="CLUB",
            )
        ],
    )


class TestRenderedSchemaHidesModelVocabulary:
    def test_rendering_never_names_a_field_of_the_model(self):
        """The vector itself. `is_nullable` and `source_fact_ids` appear on every
        column of a JSON dump and on none of the rendered text."""
        text = schema_to_prompt_text(_schema(), heading="## GLOBAL SCHEMA")
        for reserved in _schema_model_field_names():
            assert reserved not in text, (
                f"model field '{reserved}' leaked into the prompt"
            )

    def test_the_json_dump_does_contain_it(self):
        """Guards the premise: if a future model rename made this false, the
        test above would be passing for the wrong reason."""
        assert "is_nullable" in _schema().model_dump_json()

    def test_nullability_is_still_conveyed(self):
        """Hiding the field name must not hide the information."""
        text = schema_to_prompt_text(_schema())
        assert "date_left: DATE NULL" in text
        assert "club_id: INTEGER NOT NULL" in text

    def test_types_render_as_sql_names_not_python_enum_names(self):
        """`DataType` is a (str, Enum), which since Python 3.11 formats as
        `DataType.VARCHAR`. Every prompt built from this renderer -- Stage 3's
        included -- had been showing the model a Python class name."""
        text = schema_to_prompt_text(_schema())
        assert "DataType." not in text
        assert "VARCHAR" in text

    def test_unique_constraints_are_conveyed_when_asked_for(self):
        """The certifier emits UPSERT_UNIQUE/DELETE_UNIQUE, so it must be able
        to see the existing unique constraints."""
        text = schema_to_prompt_text(_schema(), include_unique=True)
        assert "Unique: (name)" in text

    def test_unique_is_omitted_by_default(self):
        """Stage 3 does not deal in uniqueness and its prompt must be unchanged."""
        assert "Unique:" not in schema_to_prompt_text(_schema())

    def test_heading_is_configurable(self):
        assert schema_to_prompt_text(_schema()).startswith("## SCHEMA SHARD")
        assert schema_to_prompt_text(_schema(), heading="## GLOBAL SCHEMA").startswith(
            "## GLOBAL SCHEMA"
        )


class TestAddColumnRejectsModelFieldNames:
    def _patch(self, column_name: str, **kw) -> AddColumnPatch:
        return AddColumnPatch(
            table_name="CLUB_MEMBERSHIP",
            column_name=column_name,
            data_type="BOOLEAN",
            reason="Mandatory schema adjustment.",
            **kw,
        )

    def test_the_observed_patch_is_now_refused(self):
        errors = self._patch("is_nullable")._validate(_schema())
        assert any("schema model" in e for e in errors)

    def test_a_real_domain_column_is_accepted(self):
        assert self._patch("role")._validate(_schema()) == []

    def test_name_is_not_treated_as_reserved(self):
        """`name` is a field of Column AND an ordinary column name. Refusing it
        would cost far more than it saves."""
        assert "name" not in _schema_model_field_names()
        assert self._patch("name")._validate(_schema()) == []

    def test_data_type_is_not_treated_as_reserved(self):
        """Likewise plausible as a real column, e.g. in a catalogue of readings."""
        assert "data_type" not in _schema_model_field_names()
        assert self._patch("data_type")._validate(_schema()) == []

    def test_reserved_set_is_derived_from_the_models(self):
        """Not a hand-maintained list: it must track the models automatically."""
        reserved = _schema_model_field_names()
        assert {"is_nullable", "source_fact_ids", "primary_key"} <= reserved

    def test_missing_provenance_alone_is_not_a_rejection(self):
        """Pins the deliberate NON-change: schema_patch.py records a measured
        decision that dropping patches for absent source_fact_ids loses
        mostly-correct fixes."""
        patch = self._patch("role", source_fact_ids=[])
        assert patch._validate(_schema()) == []
