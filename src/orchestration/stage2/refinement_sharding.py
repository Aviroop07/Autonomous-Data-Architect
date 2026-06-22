from typing import List, Set
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.middleware.schema_merging.similarity import (
    get_similarity_score,
)


def get_table_degree(table_name: str, schema: Schema) -> int:
    """Calculates the sum of incoming and outgoing relationships for a table."""
    degree = 0
    if not schema.relationships:
        return 0

    for rel in schema.relationships:
        if rel.referencing_table.upper() == table_name.upper():
            degree += 1
        if rel.referred_table.upper() == table_name.upper():
            degree += 1
    return degree


def get_neighbors(table_name: str, schema: Schema) -> Set[str]:
    """Returns the set of 1-hop neighbor table names (referencing or referred)."""
    neighbors = set()
    if not schema.relationships:
        return neighbors

    for rel in schema.relationships:
        if rel.referencing_table.upper() == table_name.upper():
            neighbors.add(rel.referred_table)
        if rel.referred_table.upper() == table_name.upper():
            neighbors.add(rel.referencing_table)
    return neighbors


def get_deterministic_shards(schema: Schema) -> List[List[str]]:
    """
    Implements degree-based anchor selection and 1-hop neighborhood sharding.
    Returns a list of shards, where each shard is a list of table names.
    """
    tables = schema.tables
    if not tables:
        return []

    # 1. Calculate degrees
    table_degrees = {t.name: get_table_degree(t.name, schema) for t in tables}

    # 2. Sort tables by degree descending
    sorted_tables = sorted(tables, key=lambda t: table_degrees[t.name], reverse=True)

    # 3. Greedy Anchor Selection
    anchors = []
    visited = set()

    for table in sorted_tables:
        if table.name not in visited:
            anchors.append(table.name)
            visited.add(table.name)
            # Mark neighbors as visited too
            neighbors = get_neighbors(table.name, schema)
            visited.update(neighbors)

    # 4. Construct Shards starting from Anchors
    shards = []
    for anchor in anchors:
        neighborhood = {anchor}
        neighbors = get_neighbors(anchor, schema)
        neighborhood.update(neighbors)
        shards.append(list(neighborhood))

    return shards


def absorb_isolated_singletons(
    shards: List[List[str]], schema: Schema
) -> List[List[str]]:
    """
    Post-processes deterministic shards to absorb singleton (degree-0) tables into
    the most appropriate other shard, so the auditor sees related tables together.

    For each singleton shard {T}, targets are located in two stages:
      Stage 1 (FK-column convention): find any shard containing a table with a column
        named exactly '{t_name.lower()}_id', indicating an FK reference to T.
      Stage 2 (name-similarity fallback): find the shard containing the table with the
        highest name-similarity score to T.

    Edge case: if all shards are singletons, returns shards unchanged.
    """
    if len(shards) <= 1:
        return shards

    if not any(len(s) > 1 for s in shards):
        return shards

    table_map = {t.name: t for t in schema.tables}
    result = [list(s) for s in shards]

    # Process singletons from highest index downward so pop() doesn't shift earlier indices.
    singleton_indices = sorted(
        [i for i, s in enumerate(result) if len(s) == 1], reverse=True
    )

    for idx in singleton_indices:
        t_name = result[idx][0]
        fk_col_name = f"{t_name.lower()}_id"
        best_target: int | None = None

        # Stage 1: find a shard whose tables contain a column named {t_name_lower}_id
        for i, shard in enumerate(result):
            if i == idx:
                continue
            for other_name in shard:
                other_table = table_map.get(other_name)
                if other_table and any(
                    c.name == fk_col_name for c in other_table.columns
                ):
                    best_target = i
                    break
            if best_target is not None:
                break

        # Stage 2: name-similarity fallback across all non-self shards
        if best_target is None:
            best_score = -1.0
            for i, shard in enumerate(result):
                if i == idx:
                    continue
                for other_name in shard:
                    score = get_similarity_score(t_name, other_name)
                    if score > best_score:
                        best_score = score
                        best_target = i

        if best_target is not None:
            print(
                f"  [DetShard] Absorbing singleton '{t_name}' into shard {best_target + 1}"
                f" {result[best_target]}"
            )
            result[best_target].append(t_name)
            result.pop(idx)

    return result
