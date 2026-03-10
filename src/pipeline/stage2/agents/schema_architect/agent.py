from typing import List, Optional, Tuple
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

PROMPT_FILE_URL = "src/pipeline/stage2/agents/schema_architect/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=Schema,
        model=model,
        name='Schema Architect'
    )

def run_schema_architect(
    chunk_facts: List[AtomicFact],
    full_facts: Optional[List[AtomicFact]] = None,
    architect = None,
    model: Optional[str] = None
) -> Tuple[Schema, int]:
    """
    Invokes the Schema Architect agent for a single generation.
    Returns: (schema, tokens)
    """
    if not architect:
        architect = get_agent(model)

    query = f"TARGET CHUNK FACTS:\n" + "\n".join([f"- {f.fact} (Category: {f.tag})" for f in chunk_facts])
    if full_facts:
        query += "\n\nFULL ARCHITECTURAL CONTEXT (All Facts):\n" + "\n".join([f"- {f.fact}" for f in full_facts])

    schema, tokens = get_response(
        agent=architect,
        output_structure=Schema,
        query=query
    )
    assert isinstance(schema, Schema)
    
    return schema, tokens


