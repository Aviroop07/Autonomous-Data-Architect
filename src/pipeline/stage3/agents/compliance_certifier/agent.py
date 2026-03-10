from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.patch import CritiqueReport
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.agent import get_agent_
from src.util.invoke import get_response
from typing import List, Tuple, Optional

PROMPT_FILE_URL = "src/pipeline/stage3/agents/compliance_certifier/prompt.txt"

def get_agent(goal: str, enriched_nl: List[AtomicFact], model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Format AtomicFacts
    formatted_facts = "\n".join([f"{f.id}. {f.fact} [{f.tag}]" for f in enriched_nl])
    system_prompt = template.format(goal=goal, enriched_nl=formatted_facts)
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=CritiqueReport,
        model=model,
        name='Compliance Certifier (Stage 3)'
    )

def certify_compliance(
    schema: Schema,
    goal: str,
    enriched_nl: List[AtomicFact],
    agent = None,
    model: Optional[str] = None
) -> Tuple[CritiqueReport, int]:
    """
    Audits the global schema for analytical utility and join-path integrity.
    """
    if not agent:
        agent = get_agent(goal, enriched_nl, model)
    
    # Format AtomicFacts
    formatted_facts = "\n".join([f"{f.id}. {f.fact} [{f.tag}]" for f in enriched_nl])
        
    query = f"GLOBAL SCHEMA (JSON):\n{schema.model_dump_json(indent=2)}\n\nSOURCE FACTS:\n{formatted_facts}"
    
    report, tokens = get_response(
        agent=agent,
        output_structure=CritiqueReport,
        query=query
    )
    assert isinstance(report, CritiqueReport)
    
    return report, tokens
