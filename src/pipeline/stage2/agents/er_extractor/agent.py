from pathlib import Path
from typing import Optional

from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_agent(model: Optional[str] = None, provider: Optional[AgentProvider] = None):
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return resolve_agent_provider(provider).build(
        system_prompt=system_prompt,
        output_structure=ConceptualModel,
        model=model,
        name="ER Extractor",
    )


async def extract_conceptual_model(
    facts: str,
    nl_query: str,
    agent=None,
    model: Optional[str] = None,
    provider: Optional[AgentProvider] = None,
) -> tuple[ConceptualModel, int]:
    if not agent:
        agent = get_agent(model, provider)

    query = f"## INPUT\nOriginal Description:\n{nl_query}\n\nFacts:\n{facts}"

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=ConceptualModel,
        query=query,
    )
    return parsed, tokens
