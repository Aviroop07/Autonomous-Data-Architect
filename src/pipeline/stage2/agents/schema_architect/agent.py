from typing import List, Optional, Tuple
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import SchemaSegment

from src.pipeline.stage2.agents.schema_corrector.agent import fix_schema, get_agent as get_corrector_agent

PROMPT_FILE_URL = "src/pipeline/stage2/agents/schema_architect/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=SchemaSegment,
        model=model,
        name='Schema Architect'
    )

def run_schema_architect(
    chunk_text: str,
    architect = None,
    corrector = None,
    model: Optional[str] = None,
    max_retries: int = 5
) -> Tuple[SchemaSegment, int, List[dict]]:
    """
    Invokes the Schema Architect agent and performs self-correction if validation fails.
    Returns: (final_schema, total_tokens, list_of_correction_logs)
    """
    if not architect:
        architect = get_agent(model)
    if not corrector:
        corrector = get_corrector_agent(model)

    schema, tokens = get_response(
        agent=architect,
        output_structure=SchemaSegment,
        query=chunk_text
    )
    
    # Delegate iterative correction to fix_schema
    final_schema, c_tokens, correction_history = fix_schema(
        schema=schema,
        corrector=corrector,
        model=model,
        retry_limit=max_retries
    )
    
    tokens += c_tokens
    return final_schema, tokens, correction_history


