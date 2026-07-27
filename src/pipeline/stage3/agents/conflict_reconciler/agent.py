from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.pipeline.stage3.models.probe import GroupReconciliation
from src.util.core.agent import get_agent_
from src.util.core.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


class ConflictItemForReconciliation(BaseModel):
    """One conflict's worth of context, as handed to the reconciler within
    a group request."""

    conflict_ref: str
    description: str
    involved_facts: str
    involved_constraints: str


def get_agent(model: Optional[str] = None):
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=GroupReconciliation,
        model=model,
        name="Conflict Reconciler",
    )


def _render_group_query(
    items: List[ConflictItemForReconciliation], schema_context: str
) -> str:
    parts = [f"## SCHEMA CONTEXT (shared across this group)\n{schema_context}"]
    for item in items:
        parts.append(
            f"## CONFLICT_REF\n{item.conflict_ref}\n\n"
            f"## WHAT WAS DETECTED\n{item.description}\n\n"
            f"## ORIGINAL NL FACTS INVOLVED\n{item.involved_facts}\n\n"
            f"## WHAT WAS EXTRACTED FROM THEM\n{item.involved_constraints}"
        )
    return "\n\n---\n\n".join(parts)


async def reconcile_conflict_group(
    items: List[ConflictItemForReconciliation],
    schema_context: str,
    agent=None,
    model: Optional[str] = None,
) -> tuple[GroupReconciliation, int]:
    """Single structured-output call over a WHOLE GROUP of related
    conflicts -- not a retry loop. This is a judgment/classification task,
    not generative structured output that benefits from iterative
    self-correction the way extraction does."""
    if not agent:
        agent = get_agent(model)

    query = _render_group_query(items, schema_context)

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=GroupReconciliation,
        query=query,
    )
    assert isinstance(parsed, GroupReconciliation)
    return parsed, tokens
