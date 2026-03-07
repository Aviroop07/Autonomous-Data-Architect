from typing import List, Optional, Tuple, Dict, Any
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import SchemaSegment
from src.pipeline.stage2.models.corrections import SchemaResolve, CorrectionStatus

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

def fix_schema(
    schema: SchemaSegment,
    corrector = None,
    model: Optional[str] = None,
    retry_limit: int = 5
) -> Tuple[SchemaSegment, int, List[Dict[str, Any]]]:
    """
    Iteratively invokes the Schema Corrector agent to fix validation errors.
    Stops if all errors remain deferred or if retry limit is reached.
    """
    if not corrector:
        corrector = get_agent(model)

    total_tokens = 0
    history: List[Dict[str, Any]] = []
    previous_errors_set = set()

    for attempt in range(retry_limit):
        current_errors = schema._validate()
        if not current_errors:
            break

        query = f"Schema:\n{str(schema)}\n\nErrors:\n" + "\n".join([f"- {err}" for err in current_errors])

        resolve_obj, tokens = get_response(
            agent=corrector,
            output_structure=SchemaResolve,
            query=query
        )
        total_tokens += tokens
        
        # Update schema
        schema = resolve_obj.corrected_schema
        
        # Log attempt
        entry = {
            "attempt": attempt + 1,
            "errors": current_errors,
            "corrections": [c.model_dump() for c in resolve_obj.corrections],
            "fixed_schema": str(schema)
        }
        history.append(entry)

        # Subset logic for early exit
        current_errors_set = set(current_errors)
        
        # If all current corrections were deferred AND these errors were already seen (subset check)
        if resolve_obj.all_deferred():
            if current_errors_set.issubset(previous_errors_set):
                # No progress made on these specific errors, they are stuck in deferred state
                break
        
        # If there are NO 'not_fixed' errors and everything was fixed or deferred,
        # and validation passes now, we'll hit the 'if not current_errors: break' at top.
        
        previous_errors_set = current_errors_set

    return schema, total_tokens, history
