"""Rendering a Schema as prompt text.

Lives here rather than beside either caller because two modules held
byte-identical copies of this function -- Stage 3's context builder and its
constraint_generator agent -- so a change to how a shard is described to the
model had to be made twice or the two prompts silently diverged.
"""

from __future__ import annotations

from typing import List, Optional

from src.util.schema_model.schema import Schema


def _type_name(data_type: object) -> str:
    """The bare SQL type name, e.g. `VARCHAR`.

    `DataType` is a `(str, Enum)`, and since Python 3.11 that formats as
    `DataType.VARCHAR` rather than `VARCHAR` -- so every prompt built from this
    renderer had been showing the model a Python enum class name. Same family of
    defect as the JSON dump this rendering exists to replace: implementation
    vocabulary reaching the model as though it were schema content.

    Written to tolerate a plain string as well, though `Column.data_type` is
    typed `DataType` and pydantic coerces a patch's string into the enum before
    it ever reaches here -- so that branch is defensive, not a live path.
    """
    return str(getattr(data_type, "value", data_type))


def schema_to_prompt_text(
    schema: Schema,
    stub_tables: Optional[List[str]] = None,
    *,
    heading: str = "## SCHEMA SHARD",
    include_unique: bool = False,
) -> str:
    """Describe a schema shard for an LLM prompt.

    `stub_tables` names tables that exist in other shards and are referenceable
    but whose columns are not available here.

    Prefer this over handing a model `schema.model_dump_json()`. A raw dump
    carries the MODEL'S OWN field names -- `is_nullable`, `source_fact_ids`,
    `data_type` -- on every column, and a model reading it cannot tell schema
    metadata from schema content. Observed live: the compliance certifier,
    which was shown the JSON dump, emitted
    `ADD_COLUMN CLUB_MEMBERSHIP.is_nullable BOOLEAN` -- round-tripping a
    metadata key back as a domain column. This rendering states the same
    information without ever naming a field of the model.

    `include_unique` is opt-in rather than default so that callers which
    deliberately do not deal in uniqueness keep a byte-identical prompt --
    Stage 3 does not extract uniqueness at all (it lives in
    Table.primary_key/Table.unique), so surfacing it there would only invite
    constraints it is specified not to emit.
    """
    lines: List[str] = [heading]
    for table in schema.tables:
        lines.append(f"### {table.name}")
        lines.append(f"  Primary key: {', '.join(table.primary_key)}")
        for col in table.columns:
            nullable = "NULL" if col.is_nullable else "NOT NULL"
            lines.append(f"  {col.name}: {_type_name(col.data_type)} {nullable}")
        if include_unique:
            for uniq in table.unique or []:
                lines.append(f"  Unique: ({', '.join(uniq.columns)})")
        for fk in schema.relationships or []:
            if fk.referencing_table == table.name:
                lines.append(
                    f"  FK: {fk.referencing_column} -> "
                    f"{fk.referred_table} (its primary key)"
                )

    if stub_tables:
        lines.append("\n## STUB TABLES (cross-shard, schema-only)")
        for stub in stub_tables:
            lines.append(f"### {stub} (stub)")
            lines.append("  (columns not available -- use for ON-tree references only)")

    return "\n".join(lines)
