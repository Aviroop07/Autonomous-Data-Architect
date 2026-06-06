from pathlib import Path
from typing import Optional, Tuple
from src.pipeline.stage2.agents.domain_intelligence_extractor.model import DomainIntelligenceReport
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=DomainIntelligenceReport,
        model=model,
        name='domain_intelligence_extractor_stage2'
    )

async def run_domain_intelligence(
    domain: str,
    analytical_goal: str = "General Schema Design",
    agent: Optional[AgentType] = None,
    model: Optional[str] = None
) -> Tuple[DomainIntelligenceReport, int]:
    """
    Invokes the Domain Intelligence Extractor to research industry-standard patterns.
    """
    if not agent:
        agent = get_agent(model)

    query = f"Domain: {domain}\nAnalytical Goal: {analytical_goal}"

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=DomainIntelligenceReport,
        query=query
    )
    assert isinstance(parsed, DomainIntelligenceReport)

    return parsed, tokens
