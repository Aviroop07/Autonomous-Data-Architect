from typing import Optional, Tuple, List
from src.orchestration.stage2.models import Output
from src.pipeline.stage2.agents.schema_architect.agent import run_schema_architect, get_agent as get_architect
from src.pipeline.stage2.agents.schema_corrector.agent import fix_schema_step, get_agent as get_corrector
from src.pipeline.stage2.agents.chunker.agent import run_chunker
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.orchestration.stage2.utils import run_correction_loop

def orchestrate(
    facts: List[AtomicFact],
    retry_count: int = 5,
    model: Optional[str] = None
) -> Output:
    """
    Orchestrates Stage 2: Structural Chunking & Generation.
    Manages iterative correction for each schema shard manually.
    """
    plan, _ = run_chunker(facts, model=model)
    architect = get_architect(model)
    corrector = get_corrector(model)
    
    segments: List[Schema] = []
    
    for chunk in plan.chunks:
        # Step 1: Initial Generation
        schema, _ = run_schema_architect(
            chunk_facts=chunk,
            full_facts=plan.core_modeling_facts,
            architect=architect,
            model=model
        )
        
        # Step 2: Robust Correction Loop
        run_correction_loop(schema, corrector, retry_count, model=model)
            
        segments.append(schema)
        
    return Output(
        segments=segments,
        plan=plan
    )
