from typing import Tuple, Optional
from src.util.agent import get_agent_
from src.orchestration.stage6.models import GeneratedGenerator
from src.util.invoke import get_response

PROMPT_FILE_URL = "g:/Personal Project/Autonomous_Data_Architect/v6/src/pipeline/stage6/agents/code_generator/prompt.txt"

def generate_code(
    schema_json: str,
    registry_json: str,
    plan_json: str,
    agent = None,
    model: Optional[str] = None
) -> Tuple[GeneratedGenerator, int]:
    """Uses LLM to synthesize the final data generator Python code."""
    if not agent:
        with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
            
        agent = get_agent_(
            system_prompt=system_prompt,
            output_structure=GeneratedGenerator,
            model=model,
            name='Code Synthesizer (Stage 6)'
        )
    
    query = (
        f"### GLOBAL SCHEMA\n{schema_json}\n\n"
        f"### DISTRIBUTION REGISTRY\n{registry_json}\n\n"
        f"### GENERATION PLAN\n{plan_json}"
    )
    
    result, tokens = get_response(
        agent=agent,
        output_structure=GeneratedGenerator,
        query=query
    )
    
    return result, tokens
