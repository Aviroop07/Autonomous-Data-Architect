from typing import Tuple, List, Optional, Union
from pydantic import BaseModel
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import AtomicFact

PROMPT_FILE_URL = "src/pipeline/stage1/agents/fact_extractor/prompt.txt"

class FactList(BaseModel):
    facts: List[AtomicFact]

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=FactList,
        model=model,
        name='Atomic Fact Extractor'
    )

def extract_facts(
    input_data: Union[str, List[AtomicFact]],
    extractor = None,
    model: Optional[str] = None
) -> Tuple[List[AtomicFact], int]:
    """
    Extracts facts from a string, or returns the list directly if already processed.
    """
    if isinstance(input_data, list):
        return input_data, 0

    if not extractor:
        extractor = get_agent(model)
        
    parsed, tokens = get_response(
        agent=extractor,
        output_structure=FactList,
        query=input_data
    )
    assert isinstance(parsed, FactList)
    
    return parsed.facts, tokens
