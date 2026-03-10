from typing import List, Set
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.schema import Schema
from src.util.invoke import get_response
from src.util.agent import get_agent_

def filter_distributional_facts(
    shard: Schema,
    all_facts: List[AtomicFact],
    modeling_facts_ids: Set[int],
    model: str = None
) -> List[AtomicFact]:
    """
    Tiered filtering of distributional facts for a specific shard.
    """
    # 1. Identity Modeling Facts vs Distributional
    distributional = [f for f in all_facts if f.id not in modeling_facts_ids]
    if not distributional:
        return []

    # 2. Tier 1: Keyword/Entity Match
    table_names = {t.name.lower() for t in shard.tables}
    column_names = {c.name.lower() for t in shard.tables for c in t.columns}
    entities = table_names.union(column_names)
    
    candidates = []
    for fact in distributional:
        fact_norm = fact.fact.lower()
        if any(entity in fact_norm for entity in entities):
            candidates.append(fact)
            
    if not candidates:
        return []

    # 3. Tier 2: Semantic Scoping (Lightweight LLM)
    # This ensures we only keep facts that actually provide modeling requirements
    agent = get_agent_(
        system_prompt=(
            "You are a Data Fact Scoper. Given a list of tables and a list of 'distributional' facts, "
            "identify which facts imply specific structural requirements (constraints, data types, precision, or relationships) "
            "for those tables.\n\n"
            "Respond ONLY with a comma-separated list of Fact IDs that are relevant. If none, respond 'NONE'."
        ),
        model=model,
        name="Fact Scoper"
    )
    
    table_str = ", ".join(table_names)
    fact_str = "\n".join([f"{f.id}: {f.fact}" for f in candidates])
    
    query = f"TABLES: {table_str}\n\nCANDIDATE FACTS:\n{fact_str}"
    
    response, _ = get_response(agent=agent, query=query, output_structure=None)
    
    if response.strip().upper() == "NONE":
        return []
        
    try:
        relevant_ids = {int(id_str.strip()) for id_str in response.split(",") if id_str.strip().isdigit()}
        return [f for f in candidates if f.id in relevant_ids]
    except:
        return []
