from pathlib import Path
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from typing import List, Optional, Tuple

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=ChunkedPlan,
        model=model,
        name='chunker_stage2'
    )

async def run_chunker(
    enriched_nl: List[AtomicFact],
    chunker: Optional[AgentType] = None,
    model: Optional[str] = None
) -> Tuple[ChunkedPlan, int]:
    """
    Invokes the Chunker agent to segment the NL description into a modular plan.
    """
    if not chunker:
        chunker = get_agent(model)

    # Format the list of AtomicFacts into a numbered list string with tags
    formatted_facts = "\n".join([f"{f.id}. {f.fact} [{', '.join(f.tags)}]" for f in enriched_nl])

    parsed, tokens = await get_response(
        agent=chunker,
        output_structure=ChunkedPlan,
        query=f"### ENRICHED NL DESCRIPTION (ATOMIC FACTS):\n{formatted_facts}"
    )
    assert isinstance(parsed, ChunkedPlan)

    return parsed, tokens
