from typing import Optional, Tuple
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.chunk import ChunkedPlan

PROMPT_FILE_URL = "src/pipeline/stage2/agents/chunker/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=ChunkedPlan,
        model=model,
        name='Chunker (Stage 2)'
    )

def run_chunker(
    enriched_nl: str, 
    chunker = None,
    model: Optional[str] = None
) -> Tuple[ChunkedPlan, int]:
    """
    Invokes the Chunker agent to segment the NL description into a modular plan.
    """
    if not chunker:
        chunker = get_agent(model)
        
    parsed, tokens = get_response(
        agent=chunker,
        output_structure=ChunkedPlan,
        query=f"### ENRICHED NL DESCRIPTION:\n{enriched_nl}"
    )
    
    return parsed, tokens


