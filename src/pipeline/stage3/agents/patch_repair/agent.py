import json
from typing import List, Tuple, Optional
from src.util.agent import get_agent_
from src.util.invoke import get_response
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.patch import CritiqueReport, PatchValidationError

PROMPT_FILE_URL = "src/pipeline/stage3/agents/patch_repair/prompt.txt"

def get_agent(model: Optional[str] = None):
    with open(PROMPT_FILE_URL, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=CritiqueReport,
        model=model,
        name='Patch Repair Agent'
    )

def repair_patches(
    schema: Schema,
    report: CritiqueReport,
    errors: List[PatchValidationError],
    repair_agent = None,
    model: Optional[str] = None
) -> Tuple[CritiqueReport, int]:
    """
    Invokes the Patch Repair Agent to fix invalid patches.
    """
    if not repair_agent:
        repair_agent = get_agent(model)
        
    query = (
        f"CURRENT SCHEMA:\n{schema.model_dump_json(indent=2)}\n\n"
        f"ORIGINAL CRITIQUE REPORT:\n{report.model_dump_json(indent=2)}\n\n"
        f"VALIDATION ERRORS:\n{json.dumps([e.model_dump() for e in errors], indent=2)}"
    )

    repaired_report, tokens = get_response(
        agent=repair_agent,
        output_structure=CritiqueReport,
        query=query
    )
    assert isinstance(repaired_report, CritiqueReport)
    
    return repaired_report, tokens
