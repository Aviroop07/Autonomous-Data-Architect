from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from src.util.web_search import get_web_search_tool
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.pipeline.stage1.models.raw_fact import RawFact

PROMPT_PATH = Path(__file__).parent / "prompt.txt"
AgentFactory = Callable[..., AgentType]


def get_context_enricher_tools() -> List[dict[str, Any]]:
    return [get_web_search_tool()]


def build_context_enricher_agent(
    system_prompt: str,
    model: Optional[str] = None,
    agent_factory: AgentFactory = get_agent_,
) -> AgentType:
    return agent_factory(
        system_prompt=system_prompt,
        output_structure=FactList,
        tools=get_context_enricher_tools(),
        model=model,
        name='domain_specialist',
        use_responses_api=True,
    )

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return build_context_enricher_agent(system_prompt=system_prompt, model=model)

async def enrich_context(
    facts: List[RawFact],
    enricher: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[List[RawFact], int]:
    if not enricher:
        enricher = get_agent(model)

    facts_text = "\n".join([
        f"- id: {f.id}\n  fact: {f.fact}\n  origin: {f.origin if f.origin else '(none)'}"
        for f in facts
    ])

    query = f"## FACTS TO ENRICH\n{facts_text}"

    parsed, tokens = await get_response(
        agent=enricher,
        output_structure=FactList,
        query=query
    )
    assert isinstance(parsed, FactList)

    for fact in parsed.facts:
        fact.is_external = True

    return parsed.facts, tokens
