from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

from src.util.core.agent import AgentType
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response
from src.pipeline.stage4.models import ParameterManifest

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_agent(
    model: Optional[str] = None, provider: Optional[AgentProvider] = None
) -> AgentType:
    with PROMPT_PATH.open(encoding="utf-8") as f:
        system_prompt = f.read()

    return resolve_agent_provider(provider).build(
        system_prompt=system_prompt,
        output_structure=ParameterManifest,
        model=model,
        name="parameter_scaling_expert",
    )


async def derive_parameters(
    schema_json: str,
    business_facts: List[str],
    nullable_columns_map: Dict[str, List[str]],  # Added knowledge of nullable columns
    parameter_agent: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[ParameterManifest, int]:
    """
    Analyzes the schema and facts to derive seeds, fanouts, and sparsity (nullability).
    """
    if not parameter_agent:
        parameter_agent = get_agent(model)

    query = f"### GLOBAL SCHEMA:\n{schema_json}\n"
    query += f"\n### NULLABLE COLUMN CANDIDATES (Columns mentioned in NULL constraints):\n{json.dumps(nullable_columns_map, indent=2)}\n"
    query += "\n### BUSINESS FACTS:\n"
    for f in business_facts:
        query += f"- {f}\n"

    parsed, tokens = await get_response(
        agent=parameter_agent, output_structure=ParameterManifest, query=query
    )
    assert isinstance(parsed, ParameterManifest)

    return parsed, tokens
