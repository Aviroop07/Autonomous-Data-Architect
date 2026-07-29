from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.util.core.agent import AgentType
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import TaggerOutput, TaggedFact
from src.pipeline.stage1.models.raw_fact import RawFact

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_agent(
    model: Optional[str] = None, provider: Optional[AgentProvider] = None
) -> AgentType:
    with PROMPT_PATH.open(encoding="utf-8") as f:
        system_prompt = f.read()

    return resolve_agent_provider(provider).build(
        system_prompt=system_prompt,
        output_structure=TaggerOutput,
        model=model,
        name="information_architect",
    )


async def tag_facts(
    facts: List[RawFact],
    tagger: Optional[AgentType] = None,
    model: Optional[str] = None,
    origins: Optional[Dict[int, str]] = None,
    provider: Optional[AgentProvider] = None,
) -> Tuple[List[TaggedFact], int]:
    if not tagger:
        tagger = get_agent(model, provider)

    # `origins` maps fact id -> source segment text. It must be supplied by the
    # caller: this agent runs BEFORE facts are converted to AtomicFact, so the
    # facts themselves carry no segment_text and the previous hasattr() read
    # rendered "(none)" for every fact, despite prompt.md documenting the field.
    # Enrichment facts legitimately have no origin and stay "(none)".
    origin_by_id: Dict[int, str] = origins or {}

    facts_text = "\n".join(
        [
            f"- id: {f.id}\n  fact: {f.fact}\n  origin: {origin_by_id.get(f.id) or '(none)'}\n  is_external: {f.is_external}"
            for f in facts
        ]
    )

    query = f"## FACTS TO TAG\n{facts_text}"

    parsed, tokens = await get_response(
        agent=tagger, output_structure=TaggerOutput, query=query
    )
    assert isinstance(parsed, TaggerOutput)

    return parsed.facts, tokens
