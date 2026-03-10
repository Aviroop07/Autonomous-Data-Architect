from typing import Optional, Tuple, List
from src.orchestration.stage1.models import Output
from src.pipeline.stage1.agents.context_enricher.agent import enrich_description, get_agent as get_enricher
from src.pipeline.stage1.agents.fact_extractor.agent import extract_facts
from src.pipeline.stage1.agents.verifier.agent import verify_integrity
from src.pipeline.stage1.models.rephrased_nl import EnrichedNL, RephrasedOutput

def orchestrate(
    nl_description: str,
    retry_count: int = 5,
    model: Optional[str] = None
) -> Output:
    """
    Orchestrates Stage 1: Technical Enrichment.
    Manages the self-correction loop manually without file I/O.
    """
    enricher = get_enricher(model)
    iterations: List[EnrichedNL] = []
    
    # Initial Fact Extraction
    orig_facts, _ = extract_facts(nl_description, model=model)
    
    current_report = None
    final_rephrased: Optional[RephrasedOutput] = None
    
    for attempt in range(retry_count + 1):
        # Call simplified agent
        rephrased_obj, _ = enrich_description(
            nl_descr=nl_description,
            integrity_report=current_report,
            enricher=enricher,
            model=model
        )
        
        # Extract facts from new version
        rephr_facts, _ = extract_facts(rephrased_obj.rephrased_text, model=model)
        
        # Verify integrity
        report, _ = verify_integrity(orig_facts, rephr_facts, model=model)
        
        # Track iteration
        iterations.append(EnrichedNL(
            rephrased_output=rephrased_obj,
            original_facts=orig_facts,
            integrity_report=report
        ))
        
        final_rephrased = rephrased_obj
        current_report = report
        
        if report.is_safe:
            break

    assert final_rephrased is not None
    
    return Output(
        final_facts=final_rephrased.rephrased_text,
        domain=final_rephrased.domain,
        analytical_goal=final_rephrased.analytical_goal,
        iterations=iterations
    )
