from typing import List, Optional
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage2.models.corrections import FixHistoryStep, CorrectionStatus
from src.pipeline.stage2.agents.schema_corrector.agent import fix_schema_step

def run_correction_loop(
    schema: Schema,
    corrector,
    retry_count: int,
    model: Optional[str] = None
) -> List[FixHistoryStep]:
    """
    Robust correction loop with early exit:
    Exits if errors are a subset of previous errors AND all LLM responses are FIXED or DEFERRED.
    """
    fix_history: List[FixHistoryStep] = []
    
    for attempt in range(retry_count):
        old_errors = set(schema._validate())
        if not old_errors:
            break
            
        # Agent step
        schema, tokens, step = fix_schema_step(
            schema=schema,
            corrector=corrector,
            model=model
        )
        fix_history.append(step)
        
        # Check subset and status
        new_errors = set(schema._validate())
        
        # Subset check: new_errors must contain only errors that were already in old_errors
        is_subset = new_errors.issubset(old_errors)
        
        # Status check: All corrections must be FIXED or DEFERRED
        all_ok = all(c.status in [CorrectionStatus.FIXED, CorrectionStatus.DEFERRED] for c in step.corrections)
        
        if is_subset and all_ok:
            # We are either stable or improving, and the agent didn't "fail" (NOT_FIXED)
            # This is an intentional stopping point.
            break
            
    return fix_history
