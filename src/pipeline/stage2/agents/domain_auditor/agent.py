from pathlib import Path
from typing import List, Optional, Tuple, Dict
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.schema_patch import CritiqueReport
from src.pipeline.stage2.agents.domain_intelligence_extractor.model import DomainIntelligenceReport
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=CritiqueReport,
        model=model,
        name='domain_auditor_stage2'
    )

async def audit_domain(
    schema: Schema,
    intelligence: DomainIntelligenceReport,
    fact_clusters: Dict[str, List[AtomicFact]],
    structural_errors: Optional[List[str]] = None,
    agent: Optional[AgentType] = None,
    model: Optional[str] = None
) -> Tuple[CritiqueReport, int]:
    """
    Invokes the Domain Auditor to suggest semantic refinements based on industry standards and facts.
    """
    if not agent:
        agent = get_agent(model)

    # Format clusters for prompt
    formatted_clusters = ""
    for entity, facts in fact_clusters.items():
        fact_list = "\n".join([f"- {f.fact}" for f in facts])
        formatted_clusters += f"### ENTITY: {entity}\n{fact_list}\n\n"

    error_feedback = "\n".join([f"- {e}" for e in structural_errors]) if structural_errors else "None. The previous state was structurally valid."

    query = (
        f"### GENERATED SCHEMA:\n{schema}\n\n"
        f"### DOMAIN INTELLIGENCE REPORT:\n{intelligence.model_dump_json(indent=2)}\n\n"
        f"### INITIAL FACT CLUSTERS:\n{formatted_clusters}\n\n"
        f"### STRUCTURAL ERRORS (FEEDBACK):\n{error_feedback}"
    )

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=CritiqueReport,
        query=query
    )
    assert isinstance(parsed, CritiqueReport)

    return parsed, tokens
