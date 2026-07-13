"""Allocates Stage 1's AtomicFacts to Stage 3 Phase 1's schema shards.

Per sharding_ilp.py's actual formulation, FK closure between two tables is
a soft reward term in the ILP objective, not a hard guarantee -- once
max_tables_per_shard or other capacity penalties bind, a fact can end up
with no single shard containing all the tables it references. This module
must therefore treat cross-shard facts as the normal case, not an edge
case: every allocation below is a best-effort placement, and stub_tables
(computed in step 4) is what lets a split-table fact still get extracted
correctly -- the extraction agent for a shard gets schema-only context
(table + column names, no data) for any table its allocated facts mention
but that isn't in the shard's own projection.
"""

from __future__ import annotations

import difflib
import re

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.registry import TableFactRegistry


class SemanticSimilarity:
    def get_score(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()


class ShardFactAllocation(BaseModel):
    """One shard's fact-allocation result."""

    fact_ids: list[int] = Field(default_factory=list)
    stub_tables: list[str] = Field(
        default_factory=list,
        description="Tables referenced by this shard's allocated facts but "
        "absent from the shard's own table set -- inject as schema-only "
        "stubs (table + column names, no data) into the extraction prompt.",
    )


def _table_name_variants(table_name: str) -> list[str]:
    """Lowercase natural-language variants of an UPPER_SNAKE_CASE table
    name to check for mentions in fact text, e.g. 'LINE_ITEM' ->
    ['line_item', 'line item', 'line items'].
    Handles irregular plurals (consonant+y -> -ies, -us -> -i, etc.)."""
    lower = table_name.lower()
    spaced = lower.replace("_", " ")
    variants = {lower, spaced}

    # Regular plural
    if not spaced.endswith("s"):
        variants.add(f"{spaced}s")

    # Irregular plurals
    if spaced.endswith("y") and len(spaced) > 1 and spaced[-2] not in "aeiou":
        # consonant+y -> -ies (e.g. category -> categories, entity -> entities)
        variants.add(f"{spaced[:-1]}ies")
    elif spaced.endswith("us"):
        # -us -> -i (e.g. status -> statuses is actually standard, but
        # nucleus -> nuclei, stimulus -> stimuli are real irregulars)
        variants.add(f"{spaced[:-2]}i")
    elif spaced.endswith("is"):
        # -is -> -es (e.g. analysis -> analyses, basis -> bases)
        variants.add(f"{spaced[:-2]}es")
    elif spaced.endswith("on"):
        # -on -> -a (e.g. phenomenon -> phenomena, criterion -> criteria)
        variants.add(f"{spaced[:-2]}a")
    elif (
        spaced.endswith("x")
        or spaced.endswith("ch")
        or spaced.endswith("sh")
        or spaced.endswith("ss")
        or spaced.endswith("z")
    ):
        # -x/-ch/-sh/-ss/-z -> add -es
        variants.add(f"{spaced}es")
    elif spaced.endswith("f"):
        # -f -> -ves (e.g. leaf -> leaves, knife -> knives)
        variants.add(f"{spaced[:-1]}ves")
    elif spaced.endswith("fe"):
        # -fe -> -ves (e.g. knife -> knives, life -> lives)
        variants.add(f"{spaced[:-2]}ves")

    return list(variants)


def find_mentioned_tables(text: str, table_names: list[str]) -> set[str]:
    """Deterministic word-boundary keyword scan: does this fact's raw text
    mention a known table name (or a natural-language variant)?
    Uses \\b word boundaries to prevent false positives (e.g. 'RATE' inside
    'corporate', 'AGE' inside 'storage')."""
    text_lower = text.lower()
    mentioned: set[str] = set()
    for table_name in table_names:
        for variant in _table_name_variants(table_name):
            pattern = r"\b" + re.escape(variant) + r"\b"
            if re.search(pattern, text_lower):
                mentioned.add(table_name)
                break
    return mentioned


def allocate_facts_to_shards(
    all_facts: list[AtomicFact],
    shard_table_sets: list[set[str]],
    registry: TableFactRegistry,
    similarity_threshold: float = 0.5,
) -> list[ShardFactAllocation]:
    """Allocates facts to shards using the global similarity engine, then
    computes each shard's stub-table requirement.

    1. Base Allocation: Registry lookup.
    2. Similarity Expansion: Context-based expansion.
    3. Orphan Recovery: table-mention-aware placement first, falling back
       to pure fact-to-fact text similarity only when a fact mentions no
       known table at all.
    4. Stub computation: for every allocated fact, any table it mentions
       that isn't in its shard's own table set becomes a stub_tables entry.
    """
    similarity_engine = SemanticSimilarity()
    fact_map = {f.id: f for f in all_facts}
    global_fact_ids = [f.id for f in all_facts]
    all_table_names = sorted({t for tables in shard_table_sets for t in tables})

    # 1. Base Allocation
    print("[Stage 3] Performing Base Allocation from Registry...")
    shard_allocations: list[set[int]] = []
    for table_names in shard_table_sets:
        base_fids = registry.get_facts_for_tables(list(table_names))
        shard_allocations.append(base_fids)

    # 2. Similarity Expansion (Context)
    print(
        f"[Stage 3] Expanding Shards via Global Similarity (Threshold: {similarity_threshold})..."
    )
    for i, table_names in enumerate(shard_table_sets):
        current_fids = shard_allocations[i]
        if not current_fids:
            continue

        base_texts = [fact_map[fid].fact for fid in current_fids if fid in fact_map]
        unincluded_fids = [fid for fid in global_fact_ids if fid not in current_fids]

        for fid in unincluded_fids:
            orphan_text = fact_map[fid].fact
            max_sim = 0.0
            for b_text in base_texts:
                sim = similarity_engine.get_score(orphan_text, b_text)
                if sim > max_sim:
                    max_sim = sim

            if max_sim >= similarity_threshold:
                current_fids.add(fid)

    # 3. Orphan Recovery
    allocated_globally: set[int] = set()
    for allocation in shard_allocations:
        allocated_globally.update(allocation)

    orphans = [fid for fid in global_fact_ids if fid not in allocated_globally]
    if orphans:
        print(f"[Stage 3] Recovering {len(orphans)} orphaned facts...")
        for fid in orphans:
            orphan_text = fact_map[fid].fact
            mentioned = find_mentioned_tables(orphan_text, all_table_names)

            best_shard_idx = None
            if mentioned:
                # Table-mention-aware placement: the shard covering the MOST
                # of this fact's mentioned tables -- a real table-identity
                # signal, not fuzzy similarity to unrelated facts' text.
                best_coverage = 0
                for s_idx, table_names in enumerate(shard_table_sets):
                    coverage = len(mentioned & table_names)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_shard_idx = s_idx

            if best_shard_idx is None:
                # Last-resort fallback: no known table mentioned at all (or
                # none of the mentioned tables landed in any shard) -- fuzzy
                # similarity to other facts' text is the only signal left.
                print(
                    f"[Stage 3] Fact {fid}: no table mention detected, "
                    f"falling back to text similarity."
                )
                best_shard_idx = 0
                best_max_sim = -1.0
                for s_idx, allocation in enumerate(shard_allocations):
                    if not allocation:
                        continue
                    shard_texts = [
                        fact_map[afid].fact for afid in allocation if afid in fact_map
                    ]
                    max_sim = 0.0
                    for s_text in shard_texts:
                        sim = similarity_engine.get_score(orphan_text, s_text)
                        if sim > max_sim:
                            max_sim = sim
                    if max_sim > best_max_sim:
                        best_max_sim = max_sim
                        best_shard_idx = s_idx

            shard_allocations[best_shard_idx].add(fid)

    # 4. Stub computation
    results: list[ShardFactAllocation] = []
    for table_names, allocation in zip(shard_table_sets, shard_allocations):
        stub_tables: set[str] = set()
        for fid in allocation:
            if fid not in fact_map:
                continue
            mentioned = find_mentioned_tables(fact_map[fid].fact, all_table_names)
            stub_tables.update(mentioned - table_names)
        results.append(
            ShardFactAllocation(
                fact_ids=sorted(allocation),
                stub_tables=sorted(stub_tables),
            )
        )

    return results
