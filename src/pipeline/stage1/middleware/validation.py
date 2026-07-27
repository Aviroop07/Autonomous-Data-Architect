from typing import List
from src.util.orchestration.retry_loop import ErrorRecord, ErrorType, Severity
from src.util.algorithms.semantic_match import FactOriginMatcher


def check_verbatim_substring(segments: list, source_text: str) -> List[ErrorRecord]:
    matcher = FactOriginMatcher(source_text)
    errors = []

    for segment in segments:
        claimed_origin = segment.text if hasattr(segment, "text") else ""
        if not claimed_origin:
            # Maybe it's an external metadata segment with no text, we can skip
            continue

        result = matcher.verify_origin(claimed_origin, claimed_origin)

        if not result.is_valid or result.match_type != "verbatim":
            if not claimed_origin:
                errors.append(
                    ErrorRecord(
                        iteration=0,
                        error_type=ErrorType.DETERMINISTIC,
                        severity=Severity.CRITICAL,
                        description="Segment missing text - must have source snippet",
                        fact_id=None,
                        signature_key="origin_missing:segment",
                    )
                )
            else:
                is_short_origin = (
                    len(claimed_origin.split()) <= 3 or len(claimed_origin) < 15
                )
                if is_short_origin:
                    severity = Severity.LOW
                else:
                    severity = Severity.MEDIUM
                # Find facts in this segment to attach the error to one of them, or just use fact_id=None
                fact_id = (
                    segment.facts[0].id
                    if hasattr(segment, "facts") and segment.facts
                    else None
                )
                errors.append(
                    ErrorRecord(
                        iteration=0,
                        error_type=ErrorType.DETERMINISTIC,
                        severity=severity,
                        description=f"Segment text verification failed (not verbatim). Best match: '{result.best_span[:120]}' (score: {result.score:.2f})",
                        fact_id=fact_id,
                        signature_key=f"origin_failed:{fact_id or 'segment'}",
                    )
                )

    return errors
