from typing import List, Set
from src.pipeline.stage2.models.schema import Schema

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
