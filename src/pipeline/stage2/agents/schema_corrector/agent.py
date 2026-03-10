from typing import List, Optional, Tuple
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.corrections import SchemaResolve, FixHistoryStep

PROMPT_FILE_URL = "src/pipeline/stage2/agents/schema_corrector/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=SchemaResolve,
        model=model,
        name='Schema Corrector'
    )

def fix_schema_step(
    schema: Schema,
    corrector = None,
    model: Optional[str] = None
) -> Tuple[Schema, int, FixHistoryStep]:
    """
    Performs a SINGLE iterative correction step using the Schema Corrector agent.
    Returns: (corrected_schema, tokens, FixHistoryStep)
    """
    if not corrector:
        corrector = get_agent(model)

    current_errors = schema._validate()
    if not current_errors:
        return schema, 0, FixHistoryStep(attempt=0, errors=[], corrections=[], fixed_schema=str(schema))

    query = f"Schema:\n{str(schema)}\n\nErrors:\n" + "\n".join([f"- {err}" for err in current_errors])

    resolve_obj, tokens = get_response(
        agent=corrector,
        output_structure=SchemaResolve,
        query=query
    )
    assert isinstance(resolve_obj, SchemaResolve)
    
    # Update schema
    corrected_schema = resolve_obj.corrected_schema
    
    history_step = FixHistoryStep(
        attempt=1, # This is a single step; orchestration will manage the overall count
        errors=current_errors,
        corrections=resolve_obj.corrections,
        fixed_schema=str(corrected_schema)
    )

    return corrected_schema, tokens, history_step
