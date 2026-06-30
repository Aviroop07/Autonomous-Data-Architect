from pathlib import Path
from typing import Optional

from src.util.core.agent import get_agent_
from src.util.core.invoke import get_response
from src.pipeline.stage2.models.conflicts import ConflictResolutionPlan

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None):
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=ConflictResolutionPlan,
        model=model,
        name="Conflict Resolver",
    )

async def resolve_conflicts(
    conflicts_json: str,
    domain: str,
    analytical_goal: str,
    agent=None,
    model: Optional[str] = None
) -> tuple[ConflictResolutionPlan, int]:
    if not agent:
        agent = get_agent(model)

    query = (
        f"## DOMAIN\n{domain}\n\n"
        f"## ANALYTICAL GOAL\n{analytical_goal}\n\n"
        f"## INPUT (CONFLICTS)\n{conflicts_json}"
    )

    plan, tokens = await get_response(
        agent=agent,
        output_structure=ConflictResolutionPlan,
        query=query,
    )
    return plan, tokens
