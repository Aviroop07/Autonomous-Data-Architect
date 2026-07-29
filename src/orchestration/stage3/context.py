"""Rendering schema/facts/constraints into the text an agent actually sees.

Pure formatting -- no orchestration, no LLM calls. Kept apart from entry.py
because a change to how a shard is described to the model should not sit in
the same file as the control flow that runs it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage3.models.shard_context import Stage3ShardContext
from src.util.schema_model.schema import Schema
from src.orchestration.stage3.state import _Merged


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


def _build_shard_context(
    shard: Schema,
    fact_ids: List[int],
    facts_map: Dict[int, AtomicFact],
    stub_tables: List[str],
    reconciliation_guidance: Optional[str] = None,
) -> Stage3ShardContext:
    """Build a typed Stage3ShardContext for one shard's extraction loop.

    Replaces the previous JSON-serialized str round-trip. The object is
    passed as-is through LoopContext; serialization to text happens only
    where the LLM prompt is built (build_context boundary).
    """
    filtered_map: Dict[int, AtomicFact] = {}
    for fid in fact_ids:
        if fid in facts_map:
            filtered_map[fid] = facts_map[fid]
    return Stage3ShardContext(
        shard_schema=shard,
        fact_ids=sorted(fact_ids),
        facts_map=filtered_map,
        stub_tables=stub_tables,
        reconciliation_guidance=reconciliation_guidance or "",
    )
