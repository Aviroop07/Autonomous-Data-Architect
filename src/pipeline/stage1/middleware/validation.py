from typing import List, Set
from src.util.retry_loop import ErrorRecord, ErrorType, Severity
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput
from src.pipeline.stage1.middleware.text_matching import verify_facts_parallel

def normalize_references(facts: List[RawFact]) -> List[RawFact]:
    seen_ids: Set[int] = set()
    for fact in facts:
        if hasattr(fact, 'referenced_fact_ids') and fact.referenced_fact_ids:
            unique_refs = list(set(fact.referenced_fact_ids))
            unique_refs = [rid for rid in unique_refs if rid != fact.id]
            fact.referenced_fact_ids = unique_refs
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
                    fact_id=fact.id
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
                fact_id=fact.id
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
                    fact_id=node
                ))
                return True
        rec_stack.remove(node)
        return False

    for fact in facts:
        if fact.id not in visited:
            has_cycle(fact.id, [fact.id])

    return errors

def check_verbatim_substring(facts: List[RawFact], source_text: str) -> List[ErrorRecord]:
    results, stats = verify_facts_parallel(facts, source_text, jaccard_threshold=0.75)

    errors = []
    for fact, result in zip(facts, results):
        if hasattr(fact, 'is_external') and fact.is_external:
            continue

        if not result.is_valid:
            if "Missing origin" in (result.warning or ""):
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Fact #{fact.id}: Missing origin - must have verbatim source snippet",
                    fact_id=fact.id
                ))
            else:
                errors.append(ErrorRecord(
                    iteration=0,
                    error_type=ErrorType.DETERMINISTIC,
                    severity=Severity.CRITICAL,
                    description=f"Fact #{fact.id}: Origin verification failed - {result.warning}. Best match: '{result.normalized_match}' (score: {result.jaccard_score:.2f})",
                    fact_id=fact.id
                ))
        else:
            if result.match_type in ["exact", "fuzzy"]:
                fact.origin = result.original_segment

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
                    severity=Severity.CRITICAL,
                    description=f"Fact #{fact.id}: METADATA fact must have at least one referenced_fact_id",
                    fact_id=fact.id
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
