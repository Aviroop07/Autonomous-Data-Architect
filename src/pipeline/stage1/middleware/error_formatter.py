from typing import List, Any, Optional
from src.util.retry_loop import ErrorRecord, ErrorType, Severity
from src.pipeline.stage1.models.raw_fact import RawFact

def format_errors_for_stage1(
    errors: List[ErrorRecord],
    iteration: int,
    output: Any,
    source_text: str
) -> str:
    all_facts = []
    if hasattr(output, 'facts'):
        all_facts = output.facts
    elif isinstance(output, list):
        all_facts = output

    if not errors:
        return ""

    lines = [f"## ERRORS TO FIX (Iteration {iteration}/5)", ""]
    lines.append("Review these errors carefully and fix the affected facts.")
    lines.append("")

    by_type = {ErrorType.MISSING: [], ErrorType.INTRODUCED: [], ErrorType.CHANGED: [], ErrorType.DETERMINISTIC: []}

    for e in errors:
        if e.error_type in by_type:
            by_type[e.error_type].append(e)

    for error_type, errs in by_type.items():
        if not errs:
            continue

        type_labels = {
            ErrorType.MISSING: "MISSING FACTS",
            ErrorType.INTRODUCED: "INTRODUCED (HALLUCINATED) FACTS",
            ErrorType.CHANGED: "CHANGED CONSTRAINTS",
            ErrorType.DETERMINISTIC: "VERBATIM SUBSTRING CHECK FAILURES"
        }

        lines.append(f"### {type_labels[error_type]}")

        for e in errs:
            severity_marker = e.severity.value.upper()

            if error_type == ErrorType.DETERMINISTIC:
                fact = next((f for f in all_facts if f.id == e.fact_id), None)
                if fact:
                    lines.append(f"- [{severity_marker}] CHECK FAILED: Origin must be a verbatim substring of source text")
                    lines.append(f"  Error: {e.description}")
                    lines.append(f"  Fact ID: {fact.id}")
                    lines.append(f"  Fact text: {fact.fact[:80]}...")
                    lines.append(f"  Your origin: '{fact.origin}'")
                    lines.append(f"  Source text: '...{_find_context(source_text, fact.origin)}...'")
                    lines.append(f"  Fix: Find the EXACT verbatim substring in the source that matches this fact")
                else:
                    lines.append(f"- [{severity_marker}] {e.description}")

            elif error_type == ErrorType.MISSING:
                lines.append(f"- [{severity_marker}] CHECK FAILED: Required fact is missing from extraction")
                lines.append(f"  Error: {e.description}")
                lines.append(f"  Fix: Check the source text and add this missing fact if it exists in the original")

            elif error_type == ErrorType.INTRODUCED:
                fact = next((f for f in all_facts if f.id == e.fact_id), None)
                if fact:
                    lines.append(f"- [{severity_marker}] CHECK FAILED: Fact appears to be invented/not in source")
                    lines.append(f"  Error: {e.description}")
                    lines.append(f"  Fact ID: {fact.id}")
                    lines.append(f"  Fact text: {fact.fact[:80]}...")
                    lines.append(f"  Origin: '{fact.origin}'")
                    lines.append(f"  Fix: Remove this fact or find its exact source in the original text")
                else:
                    lines.append(f"- [{severity_marker}] {e.description}")

            elif error_type == ErrorType.CHANGED:
                fact = next((f for f in all_facts if f.id == e.fact_id), None)
                if fact:
                    lines.append(f"- [{severity_marker}] CHECK FAILED: Constraint was changed from source")
                    lines.append(f"  Error: {e.description}")
                    lines.append(f"  Fact ID: {fact.id}")
                    lines.append(f"  Fact text: {fact.fact[:80]}...")
                    lines.append(f"  Fix: Ensure constraints match exactly what was stated in the source")
                else:
                    lines.append(f"- [{severity_marker}] {e.description}")

            lines.append("")

    return "\n".join(lines)

def _find_context(text: str, snippet: str, context_chars: int = 50) -> str:
    if not snippet or not text:
        return ""
    idx = text.find(snippet)
    if idx == -1:
        return snippet
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(snippet) + context_chars)
    return text[start:end]
