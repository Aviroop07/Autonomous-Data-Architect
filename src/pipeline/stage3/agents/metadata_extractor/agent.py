from pathlib import Path
from typing import Tuple, Optional, List, Dict

from src.util.core.agent import get_agent_, AgentType
from src.util.core.invoke import get_response
from src.pipeline.stage3.models.sql_models import LLMResponse

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=LLMResponse,
        model=model,
        name='algebraic_metadata_extractor'
    )

async def extract_metadata(
    table_name: str,
    shard_schema_json: str,
    grounded_facts: List[str],
    validator_report: Optional[str] = None,
    extractor: Optional[AgentType] = None,
    model: Optional[str] = None
) -> Tuple[LLMResponse, int]:
    """
    Extracts structured constraints for all tables in a shard cluster.
    """
    if not extractor:
        extractor = get_agent(model)

    query = f"### SHARD TABLES: {table_name}\n"
    query += f"\n### SHARD SCHEMA:\n{shard_schema_json}\n"
    query += f"\n### ID-TAGGED GROUNDED FACTS:\n"
    for f in grounded_facts:
        query += f"- {f}\n"

    if validator_report:
        query += "\n### FEASIBILITY VALIDATION REPORT (FOR CORRECTION):\n"
        query += f"{validator_report}\n"

    parsed, tokens = await get_response(
        agent=extractor,
        output_structure=LLMResponse,
        query=query
    )
    assert isinstance(parsed, LLMResponse)

    return parsed, tokens
