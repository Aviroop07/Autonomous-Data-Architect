from pathlib import Path
from typing import List, Optional, Tuple

from src.pipeline.stage3.models.sql_models import LLMResponse
from src.pipeline.stage3.models.validation import MathematicsValidationReport, Stage3Issue
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def get_agent(model: Optional[str] = None) -> AgentType:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=MathematicsValidationReport,
        model=model,
        name="mathematics_verifier_stage3",
    )


async def verify_mathematics(
    table_name: str,
    shard_schema_json: str,
    grounded_facts: List[str],
    extracted_metadata: LLMResponse,
    deterministic_issues: Optional[List[Stage3Issue]] = None,
    verifier: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[MathematicsValidationReport, int]:
    if verifier is None:
        verifier = get_agent(model)

    query = f"### SHARD TABLES\n{table_name}\n"
    query += f"\n### SHARD SCHEMA\n{shard_schema_json}\n"
    query += "\n### ID-TAGGED GROUNDED FACTS\n"
    for fact in grounded_facts:
        query += f"- {fact}\n"
    query += f"\n### EXTRACTED STAGE 3 METADATA\n{extracted_metadata.model_dump_json()}\n"
    if deterministic_issues:
        issue_json = [issue.model_dump() for issue in deterministic_issues]
        query += f"\n### DETERMINISTIC VALIDATION ISSUES\n{issue_json}\n"

    parsed, tokens = await get_response(
        agent=verifier,
        output_structure=MathematicsValidationReport,
        query=query,
    )
    assert isinstance(parsed, MathematicsValidationReport)
    return parsed, tokens
