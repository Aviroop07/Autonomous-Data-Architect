from pathlib import Path
from src.pipeline.stage2.models.schema import Schema
from src.util.schema_ops.schema_patch import CritiqueReport
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.core.agent import get_agent_, AgentType
from src.util.core.invoke import get_response
from typing import List, Tuple, Optional

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def get_agent(
    goal: str, enriched_nl: List[AtomicFact], model: Optional[str] = None
) -> AgentType:
    with PROMPT_PATH.open(encoding="utf-8") as f:
        template = f.read()

    # Format AtomicFacts
    formatted_facts = "\n".join(
        [f"{f.id}. {f.fact} [{', '.join(f.tags)}]" for f in enriched_nl]
    )
    system_prompt = template.format(goal=goal, enriched_nl=formatted_facts)

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=CritiqueReport,
        model=model,
        name="compliance_certifier_stage2",
    )


async def certify_compliance(
    schema: Schema,
    goal: str,
    enriched_nl: List[AtomicFact],
    agent: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[CritiqueReport, int]:
    """
    Audits the global schema for analytical utility and join-path integrity.
    """
    if not agent:
        agent = get_agent(goal, enriched_nl, model)

    # Format AtomicFacts
    formatted_facts = "\n".join(
        [f"{f.id}. {f.fact} [{', '.join(f.tags)}]" for f in enriched_nl]
    )

    query = f"GLOBAL SCHEMA (JSON):\n{schema.model_dump_json(indent=2)}\n\nSOURCE FACTS:\n{formatted_facts}"

    report, tokens = await get_response(
        agent=agent, output_structure=CritiqueReport, query=query
    )
    assert isinstance(report, CritiqueReport)

    return report, tokens
