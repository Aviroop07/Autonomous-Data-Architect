from pathlib import Path
from typing import List, Optional, Tuple
from src.util.core.agent import get_agent_, AgentType
from src.util.core.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import TaggerOutput, TaggedFact
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.atomic_fact import FactTag

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=TaggerOutput,
        model=model,
        name='information_architect'
    )

async def tag_facts(
    facts: List[RawFact],
    tagger: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[List[TaggedFact], int]:
    if not tagger:
        tagger = get_agent(model)

    facts_text = "\n".join([
        f"- id: {f.id}\n  fact: {f.fact}\n  origin: {f.segment_text if hasattr(f, 'segment_text') and f.segment_text else '(none)'}\n  is_external: {f.is_external}"
        for f in facts
    ])

    query = f"## FACTS TO TAG\n{facts_text}"

    parsed, tokens = await get_response(
        agent=tagger,
        output_structure=TaggerOutput,
        query=query
    )
    assert isinstance(parsed, TaggerOutput)

    return parsed.facts, tokens
