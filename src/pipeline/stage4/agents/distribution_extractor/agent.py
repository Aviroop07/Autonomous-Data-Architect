from pydantic import BaseModel
from typing import List, Optional, Tuple
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.orchestration.stage4.models import UnivariateKernel, StateTransition, EventProcess, ConditionalPolicy

class BehavioralDistributionMap(BaseModel):
    kernels: List[UnivariateKernel] = []
    transitions: List[StateTransition] = []
    processes: List[EventProcess] = []
    policies: List[ConditionalPolicy] = []

PROMPT_FILE_URL = "g:/Personal Project/Autonomous_Data_Architect/v6/src/pipeline/stage4/agents/distribution_extractor/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=BehavioralDistributionMap,
        model=model,
        name='Distribution Extractor (Stage 4 - Pass 2)'
    )

def extract_distributions(
    facts: List[AtomicFact],
    agent = None,
    model: Optional[str] = None
) -> Tuple[BehavioralDistributionMap, int]:
    """
    Extracts statistical kernels, state transitions, event processes, and 
    conditional policies from facts.
    """
    if not agent:
        agent = get_agent(model)
        
    fact_str = "\n".join([f"{f.id}: {f.fact}" for f in facts])
    query = f"ATOMIC FACTS:\n{fact_str}"
    
    result, tokens = get_response(
        agent=agent,
        output_structure=BehavioralDistributionMap,
        query=query
    )
    assert isinstance(result, BehavioralDistributionMap)
    
    return result, tokens
