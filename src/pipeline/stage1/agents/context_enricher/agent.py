from typing import Tuple, Optional
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput, IntegrityReport

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
    integrity_report: Optional[IntegrityReport] = None,
    enricher = None,
    model: Optional[str] = None
) -> Tuple[RephrasedOutput, int]:
    """
    Single-call technical rephrasing (Editorial).
    If integrity_report is provided, it uses it for self-correction context.
    Returns: (RephrasedOutput, tokens)
    """
    if not enricher:
        enricher = get_agent(model)

    query = nl_descr
    if integrity_report:
        # Construct self-correction query
        missing = "\n".join([f"- {iss.description}" for iss in integrity_report.missing_information])
        introduced = "\n".join([f"- {iss.description}" for iss in integrity_report.introduced_information])
        changed = "\n".join([f"- {iss.description}" for iss in integrity_report.changed_constraints])
        
        query = (
            f"### ORIGINAL TEXT:\n{nl_descr}\n\n"
            f"### INTEGRITY REPORT (Fix The Following Issues):\n"
            f"#### Missing Information:\n{missing}\n"
            f"#### Introduced Information:\n{introduced}\n"
            f"#### Changed Constraints:\n{changed}\n\n"
            "Please fix the issues identified in the Integrity Report using the SELF-CORRECTION MODE. "
            "Ensure NO new information is invented and ALL original facts are preserved."
        )

    rephrased_obj, tokens = get_response(
        agent=enricher, 
        output_structure=RephrasedOutput, 
        query=query
    )
    assert isinstance(rephrased_obj, RephrasedOutput)
    
    return rephrased_obj, tokens

