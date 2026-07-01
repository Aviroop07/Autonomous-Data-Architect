from pathlib import Path
from typing import Any, Optional, Tuple

from src.pipeline.stage1.models.coverage_report import CoverageReport
from src.pipeline.stage1.models.raw_fact import RawFact
from src.util.core.agent import get_agent_
from src.util.core.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def get_agent(model: Optional[str] = None) -> Any:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=CoverageReport,
        model=model,
        name="Spec Completeness Auditor",
    )


def _build_completeness_query(
    facts: list[RawFact],
    domain: str,
    analytical_goal: str,
    verifier_suggestions: list[str],
) -> str:
    parts = [
        f"Domain: {domain}",
        f"Analytical Goal: {analytical_goal}",
        "",
        "## VERIFIER SUGGESTIONS",
    ]
    if verifier_suggestions:
        for s in verifier_suggestions:
            parts.append(f"- {s}")
    else:
        parts.append("None.")

    parts.append("\n## EXTRACTED FACTS")
    for f in facts:
        parts.append(str(f))

    return "\n".join(parts)


async def audit_completeness(
    facts: list[RawFact],
    domain: str,
    analytical_goal: str,
    verifier_suggestions: Optional[list[str]] = None,
    agent: Optional[Any] = None,
    model: Optional[str] = None,
) -> Tuple[CoverageReport, int]:
    if not agent:
        agent = get_agent(model)

    input_data = _build_completeness_query(
        facts=facts,
        domain=domain,
        analytical_goal=analytical_goal,
        verifier_suggestions=verifier_suggestions or [],
    )
    query = f"## INPUT\n{input_data}"

    parsed, tokens = await get_response(
        agent=agent,
        output_structure=CoverageReport,
        query=query,
    )
    return parsed, tokens
