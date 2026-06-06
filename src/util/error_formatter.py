from typing import List
from src.util.retry_loop import ErrorRecord, ErrorType, Severity

def format_errors_for_llm(errors: List[ErrorRecord], iteration: int) -> str:
    if not errors:
        return ""

    lines = [f"## ERRORS TO FIX (Iteration {iteration}/5)", ""]

    by_type = {ErrorType.MISSING: [], ErrorType.INTRODUCED: [], ErrorType.CHANGED: [], ErrorType.DETERMINISTIC: []}

    for e in errors:
        if e.error_type in by_type:
            by_type[e.error_type].append(e)

    type_labels = {
        ErrorType.MISSING: "MISSING INFORMATION",
        ErrorType.INTRODUCED: "INTRODUCED INFORMATION",
        ErrorType.CHANGED: "CHANGED CONSTRAINTS",
        ErrorType.DETERMINISTIC: "DETERMINISTIC ERRORS"
    }

    for error_type, errs in by_type.items():
        if not errs:
            continue
        lines.append(f"### {type_labels[error_type]}")
        for e in errs:
            severity_marker = e.severity.value.upper()
            fact_info = f" [Fact #{e.fact_id}]" if e.fact_id else ""
            lines.append(f"- [{severity_marker}] {e.description}{fact_info}")
        lines.append("")

    return "\n".join(lines)
