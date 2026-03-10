from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.orchestration.stage4.models import ExplicitCardinality
from pydantic import BaseModel
from typing import List, Optional, Tuple

class CardinalityExtractionList(BaseModel):
    items: List[ExplicitCardinality]

PROMPT_FILE_URL = "g:/Personal Project/Autonomous_Data_Architect/v6/src/pipeline/stage4/agents/cardinality_extractor/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=CardinalityExtractionList,
        model=model,
        name='Cardinality Extractor (Stage 4 - Pass 1)'
    )

def extract_cardinalities(
    facts: List[AtomicFact],
    schema_json: str,
    agent = None,
    model: Optional[str] = None
) -> Tuple[List[ExplicitCardinality], int]:
    """
    Extracts explicitly mentioned row counts from facts and maps them to the schema.
    """
    if not agent:
        agent = get_agent(model)
        
    fact_str = "\n".join([f"{f.id}: {f.fact}" for f in facts])
    query = f"GLOBAL SCHEMA:\n{schema_json}\n\nATOMIC FACTS:\n{fact_str}"
    
    result, tokens = get_response(
        agent=agent,
        output_structure=CardinalityExtractionList,
        query=query
    )
    assert isinstance(result, CardinalityExtractionList)
    
    return result.items, tokens
