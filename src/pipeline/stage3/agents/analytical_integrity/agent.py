from typing import Tuple, Optional, List
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import SchemaSegment
from src.pipeline.stage3.models.patch import CritiqueReport, SchemaPatch

PROMPT_FILE_URL = "src/pipeline/stage3/agents/analytical_integrity/prompt.txt"

def get_agent(goal: str, enriched_nl: str, model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        template = f.read()
    
    system_prompt = template.format(goal=goal, enriched_nl=enriched_nl)
    
    return get_agent_(
        system_prompt=system_prompt,
        tools=None,
        output_structure=CritiqueReport,
        model=model,
        name='Analytical Integrity Auditor (Stage 3)'
    )

def audit_integrity(
    schema: SchemaSegment,
    goal: str,
    enriched_nl: str,
    agent = None,
    model: Optional[str] = None
) -> Tuple[CritiqueReport, int]:
    """
    Audits the global schema for analytical utility and join-path integrity.
    """
    if not agent:
        agent = get_agent(goal, enriched_nl, model)
        
    query = f"GLOBAL SCHEMA (JSON):\n{schema.model_dump_json(indent=2)}"
    
    report, tokens = get_response(
        agent=agent,
        output_structure=CritiqueReport,
        query=query
    )
    
    return report, tokens
