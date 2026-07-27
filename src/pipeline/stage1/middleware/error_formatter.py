from typing import List
from src.util.orchestration.retry_loop import ErrorRecord, ErrorType
from src.pipeline.stage1.models.raw_fact import RawFact

def format_errors_for_stage1(
    errors: List[ErrorRecord],
    iteration: int,
    output: object,
    source_text: str
) -> str:
    all_facts = []
    if hasattr(output, 'flat_facts'):
        all_facts = output.flat_facts
    elif hasattr(output, 'facts'):
        all_facts = output.facts
    elif isinstance(output, list):
        all_facts = output

    if not errors:
        return ""

    lines = [f"## ERRORS TO FIX (Iteration {iteration}/5)", ""]
    lines.append("Review these errors carefully and fix the affected facts.")
    lines.append("")

    repair_rules = _repair_rules_for_errors(errors)
    if repair_rules:
        lines.append("## DO NOT REPEAT")
        for rule in repair_rules:
            lines.append(f"- {rule}")
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
            ErrorType.DETERMINISTIC: "DETERMINISTIC VALIDATION FAILURES"
        }

        lines.append(f"### {type_labels[error_type]}")

        for e in errs:
            severity_marker = e.severity.value.upper()

            if error_type == ErrorType.DETERMINISTIC:
                fact = next((f for f in all_facts if f.id == e.fact_id), None)
                _format_deterministic_error(lines, e, fact, source_text, severity_marker)

            elif error_type == ErrorType.MISSING:
                lines.append(f"- [{severity_marker}] CHECK FAILED: Required fact is missing from extraction")
                lines.append(f"  Error: {e.description}")
                lines.append("  Fix: Check the source text and add this missing fact if it exists in the original")

            elif error_type == ErrorType.INTRODUCED:
                fact = next((f for f in all_facts if f.id == e.fact_id), None)
                if fact:
                    lines.append(f"- [{severity_marker}] CHECK FAILED: Fact appears to be invented/not in source")
                    lines.append(f"  Error: {e.description}")
                    lines.append(f"  Fact ID: {fact.id}")
                    lines.append(f"  Fact text: {fact.fact[:80]}...")
                    lines.append("  Fix: Remove this fact or find its exact source in the original text")
                else:
                    lines.append(f"- [{severity_marker}] {e.description}")

            elif error_type == ErrorType.CHANGED:
                fact = next((f for f in all_facts if f.id == e.fact_id), None)
                if fact:
                    lines.append(f"- [{severity_marker}] CHECK FAILED: Constraint was changed from source")
                    lines.append(f"  Error: {e.description}")
                    lines.append(f"  Fact ID: {fact.id}")
                    lines.append(f"  Fact text: {fact.fact[:80]}...")
                    lines.append("  Fix: Ensure constraints match exactly what was stated in the source")
                else:
                    lines.append(f"- [{severity_marker}] {e.description}")

            lines.append("")

    return "\n".join(lines)


def _format_deterministic_error(
    lines: List[str],
    error: ErrorRecord,
    fact: RawFact | None,
    source_text: str,
    severity_marker: str,
) -> None:
    signature = error.signature()
    if signature.startswith("origin_missing") or signature.startswith("origin_failed"):
        # Not reachable, origin errors handled by validator node and ctx.det_errors
        return

    if signature.startswith("missing_relationship"):
        lines.append(f"- [{severity_marker}] RELATIONSHIP FACT MISSING")
        lines.append(f"  Error: {error.description}")
        lines.append("  Fix: Add a standalone relationship fact only if the source text explicitly supports it.")
        lines.append("  Do not rely on *_id attributes alone for relationships.")
        return



    lines.append(f"- [{severity_marker}] {error.description}")


def _repair_rules_for_errors(errors: List[ErrorRecord]) -> List[str]:
    rules: List[str] = []
    for error in errors:
        lowered = error.description.lower()
        if "missing relationship fact" in lowered:
            _append_once(rules, "Do not rely on *_id attributes alone; add a standalone relationship fact when the source text supports the relationship.")
            _append_once(rules, "For routing/bridge entities, emit each relationship separately, e.g. VM instances are assigned to compute nodes and associated with tenants.")

        if "allowed values" in lowered:
            _append_once(rules, "Allowed-value facts must preserve the attribute and every listed value exactly; do not split enum values into separate equality facts.")
    return rules


def _append_once(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)

def _find_context(text: str, snippet: str, context_chars: int = 50) -> str:
    if not snippet or not text:
        return ""
    idx = text.find(snippet)
    if idx == -1:
        return snippet
    start = max(0, idx - context_chars)
    end = min(len(text), idx + len(snippet) + context_chars)
    return text[start:end]
