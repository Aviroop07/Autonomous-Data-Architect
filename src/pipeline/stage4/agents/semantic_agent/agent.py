from pathlib import Path
from typing import Tuple, Optional, List, Dict
from pydantic import BaseModel, Field

from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from src.pipeline.stage4.models import SynthesisResult

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

class SemanticSnippet(BaseModel):
    table_infill: Dict[str, str] = Field(description="Dictionary mapping table names to the generated Python code snippet for semantic infilling.")
    reasoning: str = Field(description="The underlying logic used to decide the semantic values.")

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=SemanticSnippet,
        model=model,
        name='semantic_infiller'
    )

async def infill_semantics(
    skeleton_code: str,
    schema_json: str,
    semantic_agent: Optional[AgentType] = None,
    model: Optional[str] = None
) -> Tuple[Dict[str, str], int]:
    """
    Analyzes the skeleton and schema to generate the 'flesh' (semantic strings).
    Returns the infill code snippet only.
    """
    if not semantic_agent:
        semantic_agent = get_agent(model)

    query = f"### GLOBAL SCHEMA:\n{schema_json}\n"
    query += f"\n### SKELETON SCRIPT:\n{skeleton_code}\n"

    parsed, tokens = await get_response(
        agent=semantic_agent,
        output_structure=SemanticSnippet,
        query=query
    )
    assert isinstance(parsed, SemanticSnippet)

    return parsed.table_infill, tokens
