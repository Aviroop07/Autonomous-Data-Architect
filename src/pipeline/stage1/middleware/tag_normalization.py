from typing import List

from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag


ENUM_MARKERS = [
    "allowed values",
    "can be one of",
    "one of:",
    "enum",
]


def normalize_stage1_tags(facts: List[AtomicFact]) -> List[AtomicFact]:
    for fact in facts:
        if fact.is_external:
            _add_tag(fact, FactTag.METADATA)
        if _is_enum_or_domain_fact(fact.fact):
            _add_tag(fact, FactTag.STRUCTURAL)
            _add_tag(fact, FactTag.LOGICAL)
    return facts


def _is_enum_or_domain_fact(fact_text: str) -> bool:
    lowered = fact_text.lower()
    return any(marker in lowered for marker in ENUM_MARKERS)


def _add_tag(fact: AtomicFact, tag: FactTag) -> None:
    if tag not in fact.tags:
        fact.tags.append(tag)
