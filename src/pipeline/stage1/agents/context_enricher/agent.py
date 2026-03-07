from typing import Tuple, Optional, List
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import EnrichedNL, RephrasedOutput
from src.pipeline.stage1.agents.fact_extractor.agent import extract_facts
from src.pipeline.stage1.agents.verifier.agent import verify_integrity

from src.util.web_search import get_search_tool

PROMPT_FILE_URL = "src/pipeline/stage1/agents/context_enricher/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=[get_search_tool()],
        output_structure=RephrasedOutput,
        model=model,
        name='Technical Editor (Stage 1)'
    )

def enrich_description(
    nl_descr: str,
    enricher = None,
    model: Optional[str] = None
) -> Tuple[List[EnrichedNL], int]:
    """
    Two-phase strategy for Stage 1:
    1. Extract original facts.
    2. Rephrase/Clarify (Technical Editor + Search).
    3. Extract rephrased facts.
    4. Verify integrity.
    """
    total_tokens = 0
    iterations = []

    # Step 1: Extract original facts
    print("  -> Phase 1: Extracting original facts...")
    orig_facts, tok1 = extract_facts(nl_descr, model=model)
    total_tokens += tok1

    MAX_RETRIES = 5
    
    query = nl_descr
    
    for attempt in range(MAX_RETRIES + 1):
        if attempt == 0:
            print("  -> Phase 2: Technical rephrasing (Editorial)...")
        else:
            print(f"  -> Phase 2: Technical rephrasing (Correction Attempt {attempt})...")
            
        if not enricher:
            enricher = get_agent(model)
            
        rephrased_obj, tok2 = get_response(
            agent=enricher, 
            output_structure=RephrasedOutput, 
            query=query
        )
        total_tokens += tok2

        # Step 3: Extract rephrased facts
        print("  -> Phase 3: Extracting rephrased facts...")
        rephr_facts, tok3 = extract_facts(rephrased_obj.rephrased_text, model=model)
        total_tokens += tok3

        # Step 4: Verify integrity
        print("  -> Phase 4: Verifying information integrity...")
        report, tok4 = verify_integrity(orig_facts, rephr_facts, model=model)
        total_tokens += tok4

        # Combine into EnrichedNL for this iteration
        current_iteration = EnrichedNL(
            rephrased_output=rephrased_obj,
            original_facts=orig_facts,
            rephrased_facts=rephr_facts,
            integrity_report=report
        )
        iterations.append(current_iteration)
        
        if report.is_safe:
            print("  -> Integrity check passed.")
            break
        else:
            print("  -> Integrity check failed. Triggering Self-Correction...")
            query = (
                f"### ORIGINAL TEXT:\n{nl_descr}\n\n"
                f"### YOUR PREVIOUS OUTPUT:\n{rephrased_obj.rephrased_text}\n\n"
                f"### INTEGRITY REPORT (Fix The Following Issues):\n"
                f"- Missing Information: {report.missing_information}\n"
                f"- Introduced Information: {report.introduced_information}\n"
                f"- Changed Constraints: {report.changed_constraints}\n\n"
                "Please fix the issues identified in the Integrity Report using the SELF-CORRECTION MODE. "
                "Ensure NO new information is invented and ALL original facts are preserved."
            )

    return iterations, total_tokens

