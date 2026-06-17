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

def check_verbatim_substring(facts: List[RawFact], source_text: str) -> List[ErrorRecord]:
    matcher = FactOriginMatcher(source_text)
    errors = []
    
    for fact in facts:
        if hasattr(fact, 'is_external') and fact.is_external:
            continue

        claimed_origin = fact.origin if hasattr(fact, 'origin') else ""
        result = matcher.verify_origin(fact.fact, claimed_origin)

        if not result.is_valid:
            if not claimed_origin:
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Fact #{fact.id}: Missing origin - must have source snippet",
                    fact_id=fact.id,
                    signature_key=f"origin_missing:{fact.id}",
                ))
            else:
                # Skeleton facts often have short origins (minimal noun-phrase).
                # If the claimed origin is very short (<= 3 words or < 15 chars)
                # and has no verbatim match, downgrade to LOW since the
                # semantic matcher cannot reliably score very short strings.
                is_short_origin = len(claimed_origin.split()) <= 3 or len(claimed_origin) < 15
                if is_short_origin:
                    severity = Severity.LOW
                else:
                    severity = Severity.LOW if result.match_type == "low" else Severity.MEDIUM
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=severity,
                    description=f"Fact #{fact.id}: Origin verification failed ({result.match_type}). {result.warning} Best match: '{result.best_span[:120]}' (score: {result.score:.2f})",
                    fact_id=fact.id,
                    signature_key=f"origin_failed:{fact.id}:{result.match_type}",
                ))
        else:
            fact.origin = result.best_span

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
    normalize_references(output.facts)
    errors.extend(check_invalid_references(output.facts))
    errors.extend(check_self_references(output.facts))
    errors.extend(check_cycles(output.facts))
    errors.extend(check_verbatim_substring(output.facts, source_text))
    errors.extend(check_external_references(output.facts))
    return errors
