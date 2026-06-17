from typing import List, Set

from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.sql_models import LLMResponse
from src.pipeline.stage3.models.validation import MathematicsValidationReport, Stage3Issue


def collect_deterministic_math_issues(response: LLMResponse, schema: Schema) -> List[Stage3Issue]:
    issues: List[Stage3Issue] = []
    for idx, constraint in enumerate(response.logical_constraints):
        target = f"logical_constraints[{idx}]"
        for message in constraint._validate(schema):
            issues.append(Stage3Issue(
                code="INVALID_STATE_CONSTRAINT",
                severity="critical",
                target=target,
                message=message,
                suggested_action="Repair the state_query or binary predicate so it references emitted schema columns.",
                fact_references=constraint.fact_references,
            ))
    return issues


def combine_math_reports(
    agent_report: MathematicsValidationReport,
    deterministic_issues: List[Stage3Issue],
) -> MathematicsValidationReport:
    issues = [*deterministic_issues, *agent_report.issues]
    seen: Set[tuple[str, str, str]] = set()
    deduped: List[Stage3Issue] = []
    for issue in issues:
        key = (issue.code, issue.target, issue.message)
        if key not in seen:
            seen.add(key)
            deduped.append(issue)

    return MathematicsValidationReport(
        is_valid=agent_report.is_valid and not deterministic_issues,
        issues=deduped,
        reasoning=agent_report.reasoning,
    )


def format_math_report_for_retry(report: MathematicsValidationReport) -> str:
    if report.is_valid:
        return "MATHEMATICS VALIDATION PASSED"

    lines = ["MATHEMATICS VALIDATION FAILED"]
    for issue in report.issues:
        lines.append(
            f"- [{issue.severity.upper()}] {issue.code} at {issue.target}: {issue.message}"
        )
        if issue.suggested_action:
            lines.append(f"  Suggested action: {issue.suggested_action}")
        if issue.fact_references:
            lines.append(f"  Related facts: {issue.fact_references}")
    return "\n".join(lines)

