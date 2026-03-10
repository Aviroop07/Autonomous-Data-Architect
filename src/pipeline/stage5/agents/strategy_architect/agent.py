from typing import Tuple, Optional
from src.util.agent import get_agent_
from src.orchestration.stage5.models import GenerationPlan
from src.util.invoke import get_response

PROMPT_FILE_URL = "g:/Personal Project/Autonomous_Data_Architect/v6/src/pipeline/stage5/agents/strategy_architect/prompt.txt"

def plan_generation(
    schema_json: str,
    registry_json: str,
    prompt: Optional[str] = None,
    agent = None,
    model: Optional[str] = None
) -> Tuple[GenerationPlan, int]:
    """Uses LLM to architect the generation sequence and strategy."""
    if not agent:
        if not prompt:
            with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
                prompt = f.read()
            
        agent = get_agent_(
            system_prompt=prompt,
            output_structure=GenerationPlan,
            model=model,
            name='Generation Strategy Architect (Stage 5)'
        )
    
    query = f"### GLOBAL SCHEMA\n{schema_json}\n\n### DISTRIBUTION REGISTRY\n{registry_json}"
    
    result, tokens = get_response(
        agent=agent,
        output_structure=GenerationPlan,
        query=query
    )
    
    return result, tokens
