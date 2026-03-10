from typing import Tuple, List, Optional
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage1.models.rephrased_nl import IntegrityReport, AtomicFact

PROMPT_FILE_URL = "src/pipeline/stage1/agents/verifier/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=IntegrityReport,
        model=model,
        name='Integrity Verifier'
    )

def verify_integrity(
    original_facts: List[AtomicFact],
    rephrased_facts: List[AtomicFact],
    verifier = None,
    model: Optional[str] = None
) -> Tuple[IntegrityReport, int]:
    """
    Compares two lists of atomic facts to detect information loss or additions.
    """
    if not verifier:
        verifier = get_agent(model)
        
    query = (
        "### ORIGINAL FACTS:\n" + "\n".join([f"{f.id}. {f.fact}" for f in original_facts]) + "\n\n"
        "### REPHRASED FACTS:\n" + "\n".join([f"{f.id}. {f.fact}" for f in rephrased_facts])
    )
    
    report, tokens = get_response(
        agent=verifier,
        output_structure=IntegrityReport,
        query=query
    )
    assert isinstance(report, IntegrityReport)
    
    return report, tokens
