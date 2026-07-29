"""Typed context payload for Stage 3 per-shard extraction loops.

Replaces the previous JSON-serialized str that forced three consumers to
re-parse and type-narrow the same data independently. The object is now
passed as-is through LoopContext; serialization to text happens only where
the LLM prompt is built (the build_context boundary).
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_model.schema import Schema


class Stage3ShardContext(BaseModel):
    """Fully-typed context for one shard's generator/det_checker/auditor loop.

    Every field is a real Python object -- no JSON strings, no bare dicts.
    Consumers (deterministic_checker, constraint_generator, constraint_auditor)
    read attributes directly.
    """

    schema: Schema
    fact_ids: List[int]
    facts_map: Dict[int, AtomicFact] = Field(
        default_factory=dict,
        description="fact_id -> AtomicFact for every fact allocated to this shard",
    )
    stub_tables: List[str] = Field(default_factory=list)
    reconciliation_guidance: str = ""
