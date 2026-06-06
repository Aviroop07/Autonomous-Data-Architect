import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class MatchResult:
    is_valid: bool
    original_segment: str
    normalized_match: str
    jaccard_score: float
    match_type: str  # "exact", "fuzzy", "failed"
    warning: Optional[str] = None

def normalize_text(text: str) -> Tuple[str, List[int]]:
    normalized = text.lower()
    normalized = re.sub(r'_+', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip()
    normalized = re.sub(r'^[\s\.]+|[\s\.]+$', '', normalized)
    normalized = re.sub(r'[\'"`]', '', normalized)

    token_positions = []
    pos = 0
    for match in re.finditer(r'\S+', normalized):
        token_positions.append(match.start())

    return normalized, token_positions

def tokenize(text: str) -> set:
    return set(text.split())

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def find_best_match_sliding_window(
    normalized_origin: str,
    normalized_desc: str,
    min_length: int = 10
) -> Tuple[Optional[str], float, int, int]:
    origin_tokens = tokenize(normalized_origin)
    origin_len = len(normalized_origin)

    if not origin_tokens or not normalized_desc:
        return None, 0.0, 0, 0

    best_score = 0.0
    best_segment = None
    best_start, best_end = 0, 0

    desc_tokens = normalized_desc.split()
    origin_token_count = len(origin_tokens)

    window_sizes = [
        origin_token_count,
        origin_token_count - 1,
        origin_token_count + 1,
        origin_token_count + 2,
    ]

    for window_size in window_sizes:
        if window_size < 1 or window_size > len(desc_tokens):
            continue

        for i in range(len(desc_tokens) - window_size + 1):
            window_tokens = set(desc_tokens[i:i + window_size])
            window_text = ' '.join(desc_tokens[i:i + window_size])

            if len(window_text) < min_length:
                continue

            score = jaccard_similarity(origin_tokens, window_tokens)

            if score > best_score:
                best_score = score
                best_segment = window_text
                start_pos = normalized_desc.find(best_segment)
                end_pos = start_pos + len(best_segment) if start_pos != -1 else 0
                best_start, best_end = start_pos, end_pos

    return best_segment, best_score, best_start, best_end

def verify_origin(
    fact_id: int,
    fact_text: str,
    origin: str,
    nl_description: str,
    jaccard_threshold: float = 0.75,
    min_match_length: int = 10
) -> MatchResult:
    if not origin:
        return MatchResult(
            is_valid=False,
            original_segment="",
            normalized_match="",
            jaccard_score=0.0,
            match_type="failed",
            warning="Missing origin"
        )

    normalized_nl, _ = normalize_text(nl_description)
    normalized_origin, _ = normalize_text(origin)

    if len(normalized_origin) < min_match_length:
        return MatchResult(
            is_valid=False,
            original_segment=origin,
            normalized_match=normalized_origin,
            jaccard_score=0.0,
            match_type="failed",
            warning="Origin too short"
        )

    if normalized_origin in normalized_nl:
        return MatchResult(
            is_valid=True,
            original_segment=origin,
            normalized_match=normalized_origin,
            jaccard_score=1.0,
            match_type="exact"
        )

    best_segment, score, start_pos, end_pos = find_best_match_sliding_window(
        normalized_origin, normalized_nl, min_match_length
    )

    if score >= jaccard_threshold:
        if start_pos >= 0 and end_pos <= len(nl_description):
            original_segment = nl_description[start_pos:end_pos]
            original_segment = original_segment.strip().strip('.').strip()
        else:
            original_segment = origin.strip().strip('.').strip()

        return MatchResult(
            is_valid=True,
            original_segment=original_segment,
            normalized_match=best_segment or "",
            jaccard_score=score,
            match_type="fuzzy"
        )

    return MatchResult(
        is_valid=False,
        original_segment=origin,
        normalized_match=best_segment or "",
        jaccard_score=score,
        match_type="failed",
        warning=f"Jaccard score {score:.2f} below threshold {jaccard_threshold}"
    )

def verify_facts_parallel(
    facts: List,
    nl_description: str,
    jaccard_threshold: float = 0.75,
    min_match_length: int = 10
) -> Tuple[List[MatchResult], Dict]:
    from concurrent.futures import ThreadPoolExecutor

    results = []

    def process_fact(fact):
        if hasattr(fact, 'is_external') and fact.is_external:
            return MatchResult(
                is_valid=True,
                original_segment="",
                normalized_match="",
                jaccard_score=1.0,
                match_type="external"
            )

        origin = fact.origin if hasattr(fact, 'origin') else ""
        fact_text = fact.fact if hasattr(fact, 'fact') else ""
        fact_id = fact.id if hasattr(fact, 'id') else 0

        return verify_origin(
            fact_id=fact_id,
            fact_text=fact_text,
            origin=origin,
            nl_description=nl_description,
            jaccard_threshold=jaccard_threshold,
            min_match_length=min_match_length
        )

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_fact, facts))

    stats = {
        "total": len(results),
        "exact": sum(1 for r in results if r.match_type == "exact"),
        "fuzzy": sum(1 for r in results if r.match_type == "fuzzy"),
        "failed": sum(1 for r in results if r.match_type == "failed"),
        "external": sum(1 for r in results if r.match_type == "external"),
    }

    return results, stats
