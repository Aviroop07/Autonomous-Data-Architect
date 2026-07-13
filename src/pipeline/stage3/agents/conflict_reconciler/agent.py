from pathlib import Path
from typing import Optional

from src.pipeline.stage3.models.probe import ConflictReconciliation
from src.util.core.agent import get_agent_
from src.util.core.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def get_agent(model: Optional[str] = None):
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=ConflictReconciliation,
        model=model,
        name="Conflict Reconciler",
    )


async def reconcile_conflict(
    conflict_ref: str,
    conflict_description: str,
    involved_facts: str,
    involved_constraints: str,
    schema_context: str,
    agent=None,
    model: Optional[str] = None,
) -> tuple[ConflictReconciliation, int]:
    """Single structured-output call -- not a retry loop. This is a
    judgment/classification task, not generative structured output that
    benefits from iterative self-correction the way extraction does."""
    if not agent:
        agent = get_agent(model)

    query = (
        f"## CONFLICT_REF\n{conflict_ref}\n\n"
        f"## WHAT WAS DETECTED\n{conflict_description}\n\n"
        f"## SCHEMA CONTEXT\n{schema_context}\n\n"
        f"## ORIGINAL NL FACTS INVOLVED\n{involved_facts}\n\n"
        f"## WHAT WAS EXTRACTED FROM THEM\n{involved_constraints}"
    )

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=ConflictReconciliation,
        query=query,
    )
    assert isinstance(parsed, ConflictReconciliation)
    return parsed, tokens
