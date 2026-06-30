from typing import List
from src.util.orchestration.retry_loop import ErrorRecord, ErrorType, Severity
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput
from src.util.algorithms.semantic_match import FactOriginMatcher

def normalize_references(facts: List[RawFact]) -> List[RawFact]:
    for fact in facts:
        if hasattr(fact, 'referenced_fact_ids') and fact.referenced_fact_ids:
            clean_refs: List[int] = []
            for ref_id in fact.referenced_fact_ids:
                if ref_id == fact.id:
                    continue
                if ref_id not in clean_refs:
                    clean_refs.append(ref_id)
            fact.referenced_fact_ids = clean_refs
    return facts

def check_invalid_references(facts: List[RawFact]) -> List[ErrorRecord]:
    valid_ids = {f.id for f in facts}
    errors = []
    for fact in facts:
        ref_ids = fact.referenced_fact_ids if hasattr(fact, 'referenced_fact_ids') else []
        for ref_id in ref_ids:
            if ref_id not in valid_ids:
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Fact #{fact.id}: References non-existent fact ID {ref_id}",
                    fact_id=fact.id,
                    signature_key=f"invalid_reference:{fact.id}:{ref_id}",
                ))
    return errors

def check_self_references(facts: List[RawFact]) -> List[ErrorRecord]:
    errors = []
    for fact in facts:
        ref_ids = fact.referenced_fact_ids if hasattr(fact, 'referenced_fact_ids') else []
        if fact.id in ref_ids:
            errors.append(ErrorRecord(
                iteration=0,
                error_type=ErrorType.DETERMINISTIC,
                severity=Severity.CRITICAL,
                description=f"Fact #{fact.id}: Fact references itself",
                fact_id=fact.id,
                signature_key=f"self_reference:{fact.id}",
            ))
    return errors

def check_cycles(facts: List[RawFact]) -> List[ErrorRecord]:
    adj = {f.id: f.referenced_fact_ids if hasattr(f, 'referenced_fact_ids') else [] for f in facts}
    errors = []
    visited = set()
    rec_stack = set()

    def has_cycle(node: int, path: List[int]) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, path + [neighbor]):
                    return True
            elif neighbor in rec_stack:
                cycle = path[path.index(neighbor):] + [neighbor]
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Fact #{node}: Cyclical reference detected: {cycle}",
                    fact_id=node,
                    signature_key=f"cycle:{'-'.join(str(item) for item in cycle)}",
                ))
                return True
        rec_stack.remove(node)
        return False

    for fact in facts:
        if fact.id not in visited:
            has_cycle(fact.id, [fact.id])

    return errors

def check_verbatim_substring(segments: list, source_text: str) -> List[ErrorRecord]:
    matcher = FactOriginMatcher(source_text)
    errors = []
    
    for segment in segments:
        claimed_origin = segment.text if hasattr(segment, 'text') else ""
        if not claimed_origin:
            # Maybe it's an external metadata segment with no text, we can skip
            continue
            
        result = matcher.verify_origin(claimed_origin, claimed_origin)

        if not result.is_valid or result.match_type != "verbatim":
            if not claimed_origin:
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Segment missing text - must have source snippet",
                    fact_id=None,
                    signature_key=f"origin_missing:segment",
                ))
            else:
                is_short_origin = len(claimed_origin.split()) <= 3 or len(claimed_origin) < 15
                if is_short_origin:
                    severity = Severity.LOW
                else:
                    severity = Severity.MEDIUM
                # Find facts in this segment to attach the error to one of them, or just use fact_id=None
                fact_id = segment.facts[0].id if hasattr(segment, 'facts') and segment.facts else None
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=severity,
                    description=f"Segment text verification failed (not verbatim). Best match: '{result.best_span[:120]}' (score: {result.score:.2f})",
                    fact_id=fact_id,
                    signature_key=f"origin_failed:{fact_id or 'segment'}",
                ))

    return errors

def check_external_references(facts: List[RawFact]) -> List[ErrorRecord]:
    errors = []
    for fact in facts:
        if hasattr(fact, 'is_external') and fact.is_external:
            ref_ids = fact.referenced_fact_ids if hasattr(fact, 'referenced_fact_ids') else []
            if not ref_ids:
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.MEDIUM,
                    description=f"Fact #{fact.id}: METADATA fact must have at least one referenced_fact_id",
                    fact_id=fact.id,
                    signature_key=f"external_missing_refs:{fact.id}",
                ))
    return errors

def deterministic_validator(output: RephrasedOutput, source_text: str) -> List[ErrorRecord]:
    errors = []
    flat_facts = output.flat_facts
    normalize_references(flat_facts)
    errors.extend(check_invalid_references(flat_facts))
    errors.extend(check_self_references(flat_facts))
    errors.extend(check_cycles(flat_facts))
    errors.extend(check_verbatim_substring(output.segments, source_text))
    errors.extend(check_external_references(flat_facts))
    return errors
