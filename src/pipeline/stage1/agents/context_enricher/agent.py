from pathlib import Path
from typing import List, Optional, Tuple
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.pipeline.stage1.models.raw_fact import RawFact

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=FactList,
        model=model,
        name='domain_specialist'
    )

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
