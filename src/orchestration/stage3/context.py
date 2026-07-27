"""Rendering schema/facts/constraints into the text an agent actually sees.

Pure formatting -- no orchestration, no LLM calls. Kept apart from entry.py
because a change to how a shard is described to the model should not sit in
the same file as the control flow that runs it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
from src.orchestration.stage3.state import _Merged


def _schema_to_text(schema: Schema, stub_tables: Optional[List[str]] = None) -> str:
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


def _facts_to_text(fact_ids: List[int], facts_map: Dict[int, AtomicFact]) -> str:
    lines: List[str] = ["## FACTS"]
    for fid in sorted(fact_ids):
        fact = facts_map.get(fid)
        if fact is not None:
            lines.append(f"- [id={fid}] {fact.fact}")
    return "\n".join(lines)


def _render_involved_constraints(fact_ids: List[int], merged: "_Merged") -> str:
    """Dump every extracted constraint that references any of fact_ids, so
    the reconciliation agent can see exactly what was extracted from the
    facts it's re-examining."""
    fact_id_set = set(fact_ids)

    def _matches(c: Any) -> bool:
        return any(fid in fact_id_set for fid in c.fact_references)

    lines: List[str] = []
    for label, items in (
        ("Distribution", merged.distributions),
        ("MomentTarget", merged.moment_targets),
        ("Correlation", merged.correlations),
        ("Structural", merged.structural),
        ("Logic", merged.logic),
        ("Derived", merged.derived),
        ("StateSequence", merged.state_sequences),
    ):
        for c in items:
            if _matches(c):
                lines.append(f"[{label}] {c.model_dump_json()}")
    return (
        "\n".join(lines)
        if lines
        else "(no extracted constraints reference these facts)"
    )


def _serialize_context(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    reconciliation_guidance: Optional[str] = None,
) -> str:
    schema_dict = json.loads(shard.model_dump_json())
    return json.dumps(
        {
            "schema": schema_dict,
            "fact_ids": fact_ids,
            "facts_map": {
                str(fid): {"id": fid, "fact": facts_map[fid].fact}
                for fid in fact_ids
                if fid in facts_map
            },
            "stub_tables": stub_tables,
            "reconciliation_guidance": reconciliation_guidance or "",
        }
    )
