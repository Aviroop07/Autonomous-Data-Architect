from pathlib import Path
from src.util.schema_model.render import schema_to_prompt_text
from src.util.schema_model.schema import Schema
from src.util.schema_ops.schema_patch import CritiqueReport
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.util.core.agent import AgentType
from src.util.core.agent_provider import AgentProvider, resolve_agent_provider
from src.util.core.invoke import get_response
from typing import List, Tuple, Optional

PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_agent(
    goal: str,
    enriched_nl: List[AtomicFact],
    model: Optional[str] = None,
    provider: Optional[AgentProvider] = None,
) -> AgentType:
    with PROMPT_PATH.open(encoding="utf-8") as f:
        template = f.read()

    # Format AtomicFacts
    formatted_facts = "\n".join(
        [f"{f.id}. {f.fact} [{', '.join(f.tags)}]" for f in enriched_nl]
    )
    system_prompt = template.format(goal=goal, enriched_nl=formatted_facts)

    return resolve_agent_provider(provider).build(
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
    provider: Optional[AgentProvider] = None,
) -> Tuple[CritiqueReport, int]:
    """
    Audits the global schema for analytical utility and join-path integrity.
    """
    if not agent:
        agent = get_agent(goal, enriched_nl, model, provider)

    # Format AtomicFacts
    formatted_facts = "\n".join(
        [f"{f.id}. {f.fact} [{', '.join(f.tags)}]" for f in enriched_nl]
    )

    # Rendered, NOT model_dump_json: a raw dump names the model's own fields on
    # every column, and this agent proved it cannot tell those apart from schema
    # content -- it emitted `ADD_COLUMN CLUB_MEMBERSHIP.is_nullable BOOLEAN`,
    # inventing a domain column out of a metadata key it had just been shown.
    rendered_schema = schema_to_prompt_text(
        schema, heading="## GLOBAL SCHEMA", include_unique=True
    )
    query = f"{rendered_schema}\n\nSOURCE FACTS:\n{formatted_facts}"

    report, tokens = await get_response(
        agent=agent, output_structure=CritiqueReport, query=query
    )
    assert isinstance(report, CritiqueReport)

    return report, tokens
