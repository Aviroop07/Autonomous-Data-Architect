from pathlib import Path
from typing import Optional, Tuple
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response
from src.pipeline.stage2.models.conflicts import AdjudicatorResponse

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_agent(model: Optional[str] = None, provider: Optional[AgentProvider] = None):
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return resolve_agent_provider(provider).build(
        system_prompt=system_prompt,
        output_structure=AdjudicatorResponse,
        model=model,
        name="Adjudicator",
    )


async def resolve_conflicts(
    subgraph: str,
    facts: str,
    flags: str,
    agent=None,
    model: Optional[str] = None,
    provider: Optional[AgentProvider] = None,
) -> Tuple[AdjudicatorResponse, int]:
    if not agent:
        agent = get_agent(model, provider)

    query = f"## INPUT\n\n### Subgraph\n{subgraph}\n\n### Source Facts\n{facts}\n\n### Flags\n{flags}\n"

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=AdjudicatorResponse,
        query=query,
    )
    return parsed, tokens
