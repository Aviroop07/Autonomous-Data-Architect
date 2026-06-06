from pathlib import Path
from typing import List, Optional, Tuple
from src.util.agent import get_agent_, AgentType
from src.util.invoke import get_response
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.util.retry_loop import ValidationResult, ErrorType, ErrorRecord, Severity
from src.pipeline.stage1.models.raw_fact import RawFact

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def get_agent(model: Optional[str] = None) -> AgentType:
    with PROMPT_PATH.open(encoding='utf-8') as f:
        system_prompt = f.read()

    return get_agent_(
        system_prompt=system_prompt,
        output_structure=IntegrityReport,
        model=model,
        name='integrity_verifier'
    )

async def verify_integrity(
    nl_description: str,
    facts: List[RawFact],
    verifier: Optional[AgentType] = None,
    model: Optional[str] = None,
) -> Tuple[ValidationResult[IntegrityReport], int]:
    if not verifier:
        verifier = get_agent(model)

    facts_text = "\n".join([
        f"{f.id}. {f.fact}\n   [Origin: \"{f.origin}\"]" + (f" | External: {f.is_external}" if f.is_external else "")
        for f in facts
    ])

    query = f"""## ORIGINAL DESCRIPTION
{nl_description}

## EXTRACTED FACTS
{facts_text}"""

    report, tokens = await get_response(
        agent=verifier,
        output_structure=IntegrityReport,
        query=query
    )
    assert isinstance(report, IntegrityReport)

    errors = []
    for issue in report.missing_information:
        errors.append(ErrorRecord(
            iteration=0,
            error_type=ErrorType.MISSING,
            severity=Severity(issue.severity.value),
            description=issue.description,
            fact_id=issue.fact_id
        ))

    for issue in report.introduced_information:
        errors.append(ErrorRecord(
            iteration=0,
            error_type=ErrorType.INTRODUCED,
            severity=Severity(issue.severity.value),
            description=issue.description,
            fact_id=issue.fact_id
        ))

    for issue in report.changed_constraints:
        errors.append(ErrorRecord(
            iteration=0,
            error_type=ErrorType.CHANGED,
            severity=Severity(issue.severity.value),
            description=issue.description,
            fact_id=issue.fact_id
        ))

    return ValidationResult(
        is_valid=report.is_safe,
        errors=errors,
        validation_output=report
    ), tokens
