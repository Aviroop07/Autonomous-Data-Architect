from pathlib import Path
from typing import List, Optional, Tuple

from src.pipeline.stage3.models.sql_models import LLMResponse
from src.pipeline.stage3.models.validation import MathematicsValidationReport, Stage3PatchPlan
from src.util.core.agent import AgentType, get_agent_
from src.util.core.invoke import get_response

PROMPT_PATH = Path(__file__).parent / "prompt.txt"


def get_agent(model: Optional[str] = None) -> AgentType:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    return get_agent_(
        system_prompt=system_prompt,
        output_structure=Stage3PatchPlan,
        model=model,
        name="constraint_patch_agent_stage3",
    )


async def patch_stage3_output(
    table_name: str,
    shard_schema_json: str,
    grounded_facts: List[str],
    extracted_metadata: LLMResponse,
    validation_report: MathematicsValidationReport,
    patcher: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[Stage3PatchPlan, int]:
    if patcher is None:
        patcher = get_agent(model)

    query = f"### SHARD TABLES\n{table_name}\n"
    query += f"\n### SHARD SCHEMA\n{shard_schema_json}\n"
    query += "\n### ID-TAGGED GROUNDED FACTS\n"
    for fact in grounded_facts:
        query += f"- {fact}\n"
    query += f"\n### ORIGINAL STAGE 3 METADATA\n{extracted_metadata.model_dump_json()}\n"
    query += f"\n### VALIDATION REPORT\n{validation_report.model_dump_json()}\n"

    parsed, tokens = await get_response(
        agent=patcher,
        output_structure=Stage3PatchPlan,
        query=query,
    )
    assert isinstance(parsed, Stage3PatchPlan)
    return parsed, tokens
