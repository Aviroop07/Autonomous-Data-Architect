from pathlib import Path
from typing import List, Optional, Tuple
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=Schema,
        model=model,
        name='schema_architect'
    )

async def run_schema_architect(
    chunk_facts: List[AtomicFact],
    base_schema: Optional[Schema] = None,
    errors: Optional[List[str]] = None,
    architect: Optional[AgentType] = None,
    model: Optional[str] = None
) -> Tuple[Schema, int]:
    """
    Invokes the Schema Architect agent for a single generation.
    Returns: (schema, tokens)
    """
    if not architect:
        architect = get_agent(model)

    query = "TARGET CHUNK FACTS:\n" + "\n".join([f"- {f.fact} (Tags: {', '.join(f.tags)})" for f in chunk_facts])

    if base_schema:
        query += f"\n\nCURRENT SCHEMA STATE (JSON):\n{base_schema.model_dump_json(indent=2)}"

    if errors:
        query += "\n\nCRITICAL STRUCTURAL ERRORS TO FIX:\n" + "\n".join([f"- {e}" for e in errors])
        query += "\n\nMISSION: Repair the schema to resolve all errors while preserving the intent of the original facts."

    schema, tokens = await get_response(
        agent=architect,
        output_structure=Schema,
        query=query
    )
    assert isinstance(schema, Schema)

    return schema, tokens
