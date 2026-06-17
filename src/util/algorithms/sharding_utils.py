import networkx as nx
from typing import List, Set, Tuple
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.middleware.schema_merging.similarity import get_similarity_score

def get_deterministic_shards(
    global_schema: Schema,
    registry: TableFactRegistry,
    all_facts: List[AtomicFact],
    threshold: float = 0.8
) -> List[Tuple[Schema, Set[int]]]:
    """
    Generates deterministic shards from a global schema.
    Returns a list of (Schema, set_of_fact_ids).
    """
    # 1. Build Schema Graph
    G = nx.Graph()
    for table in global_schema.tables:
        G.add_node(table.name)

    if global_schema.relationships:
        for rel in global_schema.relationships:
            G.add_edge(rel.referencing_table, rel.referred_table)

    # 2. Rank tables by degree
    degrees = dict(G.degree())
    sorted_tables = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    # 3. Shard Generation
    visited = set()
    shards: List[List[str]] = []

    for table_name, _ in sorted_tables:
        if table_name in visited:
            continue

        # Anchor + immediate neighbors
        shard_tables = [table_name]
        neighbors = list(G.neighbors(table_name))
        shard_tables.extend(neighbors)

        shards.append(shard_tables)
        visited.update(shard_tables)

    # 4. Map tables to Schemas and basic fact sets
    shard_results: List[Tuple[Schema, Set[int]]] = []
    fact_id_to_obj = {f.id: f for f in all_facts}

    all_assigned_fact_ids = set()

    for shard_table_names in shards:
        shard_tables = [t.model_copy(deep=True) for t in global_schema.tables if t.name in shard_table_names]
        shard_relationships = []
        if global_schema.relationships:
            for rel in global_schema.relationships:
                if rel.referencing_table in shard_table_names and rel.referred_table in shard_table_names:
                    shard_relationships.append(rel.model_copy(deep=True))

        shard_schema = Schema(tables=shard_tables, relationships=shard_relationships)

        # Initial fact set from registry
        shard_fact_ids = set()
        for t_name in shard_table_names:
            shard_fact_ids.update(registry.table_to_facts.get(t_name, set()))

        shard_results.append((shard_schema, shard_fact_ids))
        all_assigned_fact_ids.update(shard_fact_ids)

    # 5. Handle Untapped Facts
    untapped_facts = [f for f in all_facts if f.id not in all_assigned_fact_ids]

    if untapped_facts and shard_results:
        # Pre-calculate shard fact texts for similarity
        shard_fact_texts = []
        for _, f_ids in shard_results:
            texts = [fact_id_to_obj[fid].fact for fid in f_ids if fid in fact_id_to_obj]
            shard_fact_texts.append(texts)

        for fact in untapped_facts:
            max_similarities = []
            for texts in shard_fact_texts:
                if not texts:
                    max_similarities.append(0.0)
                    continue
                # Maximum similarity between the untapped fact and any fact in the shard's set
                sims = [get_similarity_score(fact.fact, t) for t in texts]
                max_similarities.append(max(sims))

            # Assignment based on threshold and max
            assigned = False
            best_idx = -1
            best_sim = -1.0

            for idx, sim in enumerate(max_similarities):
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx

                if sim >= threshold:
                    shard_results[idx][1].add(fact.id)
                    assigned = True

            if not assigned and best_idx != -1:
                # Assign to the best scoring set if none above threshold
                shard_results[best_idx][1].add(fact.id)

    return shard_results
