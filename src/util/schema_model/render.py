"""Rendering a Schema as prompt text.

Lives here rather than beside either caller because two modules held
byte-identical copies of this function -- Stage 3's context builder and its
constraint_generator agent -- so a change to how a shard is described to the
model had to be made twice or the two prompts silently diverged.
"""

from __future__ import annotations

from typing import List, Optional

from src.util.schema_model.schema import Schema


def schema_to_prompt_text(
    schema: Schema, stub_tables: Optional[List[str]] = None
) -> str:
    """Describe a schema shard for an LLM prompt.

    `stub_tables` names tables that exist in other shards and are referenceable
    but whose columns are not available here.
    """
    lines: List[str] = ["## SCHEMA SHARD"]
    for table in schema.tables:
        lines.append(f"### {table.name}")
        lines.append(f"  Primary key: {', '.join(table.primary_key)}")
        for col in table.columns:
            nullable = "NULL" if col.is_nullable else "NOT NULL"
            lines.append(f"  {col.name}: {col.data_type} {nullable}")
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
