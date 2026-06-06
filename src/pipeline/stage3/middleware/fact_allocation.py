from typing import List, Dict, Set, Tuple
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage2.middleware.schema_merging.similarity import TokenSimilarity

def allocate_facts_to_shards(
    all_facts: List[AtomicFact],
    shard_table_sets: List[Set[str]],
    registry: TableFactRegistry,
    similarity_threshold: float = 0.5
) -> List[List[int]]:
    """
    Allocates facts to shards using the global similarity engine.
    1. Base Allocation: Registry lookup.
    2. Similarity Expansion: Context-based expansion.
    3. Orphan Recovery: Global safety net.
    """
    similarity_engine = TokenSimilarity()
    fact_map = {f.id: f for f in all_facts}
    global_fact_ids = [f.id for f in all_facts]

    # 1. Base Allocation
    print(f"[Stage 3] Performing Base Allocation from Registry...")
    shard_allocations: List[Set[int]] = []
    for table_names in shard_table_sets:
        base_fids = registry.get_facts_for_tables(list(table_names))
        shard_allocations.append(base_fids)

    # 2. Similarity Expansion (Context)
    print(f"[Stage 3] Expanding Shards via Global Similarity (Threshold: {similarity_threshold})...")
    for i, table_names in enumerate(shard_table_sets):
        current_fids = shard_allocations[i]
        if not current_fids:
            continue

        base_texts = [fact_map[fid].fact for fid in current_fids if fid in fact_map]
        unincluded_fids = [fid for fid in global_fact_ids if fid not in current_fids]

        for fid in unincluded_fids:
            orphan_text = fact_map[fid].fact
            # Calculate max similarity between orphan and base set using global engine
            max_sim = 0.0
            for b_text in base_texts:
                sim = similarity_engine.get_score(orphan_text, b_text)
                if sim > max_sim:
                    max_sim = sim

            if max_sim >= similarity_threshold:
                current_fids.add(fid)

    # 3. Orphan Recovery (Safety Check)
    allocated_globally: Set[int] = set()
    for allocation in shard_allocations:
        allocated_globally.update(allocation)

    orphans = [fid for fid in global_fact_ids if fid not in allocated_globally]
    if orphans:
        print(f"[Stage 3] Recovering {len(orphans)} orphaned facts via maximum similarity fallback...")
        for fid in orphans:
            orphan_text = fact_map[fid].fact
            best_shard_idx = 0 # Default to first shard if all scores 0
            best_max_sim = -1.0

            for s_idx, allocation in enumerate(shard_allocations):
                if not allocation: continue
                shard_texts = [fact_map[afid].fact for afid in allocation if afid in fact_map]

                max_sim = 0.0
                for s_text in shard_texts:
                    sim = similarity_engine.get_score(orphan_text, s_text)
                    if sim > max_sim:
                        max_sim = sim

                if max_sim > best_max_sim:
                    best_max_sim = max_sim
                    best_shard_idx = s_idx

            shard_allocations[best_shard_idx].add(fid)

    return [list(s) for s in shard_allocations]
