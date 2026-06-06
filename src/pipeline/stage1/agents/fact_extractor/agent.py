from pathlib import Path
from typing import Optional, Tuple
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=RephrasedOutput,
        model=model,
        name='atomic_fact_extractor'
    )

async def extract_facts(
    nl_description: str,
    extractor: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[RephrasedOutput, int]:
    if not extractor:
        extractor = get_agent(model)

    query = f"Extract atomic facts from the following description:\n{nl_description}"

    parsed, tokens = await get_response(
        agent=extractor,
        output_structure=RephrasedOutput,
        query=query
    )
    assert isinstance(parsed, RephrasedOutput)

    return parsed, tokens
