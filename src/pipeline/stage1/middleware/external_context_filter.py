import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from src.pipeline.stage1.models.raw_fact import ExternalFactKind, RawFact


class ExternalFactRejectionCode(str, Enum):
    INVALID_REFERENCE = "INVALID_REFERENCE"
    SELF_REFERENCE = "SELF_REFERENCE"
    DUPLICATE_EXTERNAL_FACT = "DUPLICATE_EXTERNAL_FACT"


class ExternalFactRejection(BaseModel):
    fact: RawFact = Field(description="Rejected external fact.")
    code: ExternalFactRejectionCode = Field(description="Deterministic rejection reason code.")
    reason: str = Field(description="Human-readable rejection reason.")


class ExternalFactFilterResult(BaseModel):
    accepted_facts: List[RawFact] = Field(default_factory=list)
    rejected_facts: List[ExternalFactRejection] = Field(default_factory=list)


ARCHITECTURE_TERMS = [
    "oltp",
    "olap",
    "star schema",
    "snowflake",
    "ledger",
    "audit",
    "temporal",
    "event",
    "operational scheduling",
]

def filter_external_facts(
    external_facts: List[RawFact],
    original_facts: List[RawFact],
) -> ExternalFactFilterResult:
    result = ExternalFactFilterResult()
    seen_fact_texts: List[str] = []
    for fact in external_facts:
        normalized_fact = _normalize(fact.fact)
        if normalized_fact in seen_fact_texts:
            result.rejected_facts.append(ExternalFactRejection(
                fact=fact,
                code=ExternalFactRejectionCode.DUPLICATE_EXTERNAL_FACT,
                reason="Duplicate external fact text.",
            ))
            continue
        seen_fact_texts.append(normalized_fact)

        rejection = _reject_reason(fact, original_facts)
        if rejection:
            result.rejected_facts.append(rejection)
            continue

        kind = classify_external_fact(fact)
        fact.external_kind = kind
        fact.novelty_reason = _novelty_reason(kind) if kind else "Accepted by context auditor and passed deterministic reference checks."
        fact.is_external = True
        result.accepted_facts.append(fact)
    return result


def classify_external_fact(fact: RawFact) -> Optional[ExternalFactKind]:
    text = _normalize(fact.fact)
    if text.startswith("technical definition:") or "defined as" in text or "stands for" in text:
        return ExternalFactKind.TECHNICAL_DEFINITION
    if any(term in text for term in ARCHITECTURE_TERMS):
        return ExternalFactKind.ARCHITECTURE_PATTERN
    if any(term in text for term in ["domain pattern", "standard schema", "common tables", "typical entities", "industry standard"]):
        return ExternalFactKind.DOMAIN_PATTERN
    if any(term in text for term in ["constraint", "threshold", "budget", "pricing", "rate", "discount", "capacity", "limit", "cap"]):
        return ExternalFactKind.DOMAIN_CONSTRAINT_HINT
    return None


def _reject_reason(fact: RawFact, original_facts: List[RawFact]) -> Optional[ExternalFactRejection]:
    if fact.id in fact.referenced_fact_ids:
        return ExternalFactRejection(
            fact=fact,
            code=ExternalFactRejectionCode.SELF_REFERENCE,
            reason="External fact cannot reference itself.",
        )

    if not _references_original_facts(fact, original_facts):
        return ExternalFactRejection(
            fact=fact,
            code=ExternalFactRejectionCode.INVALID_REFERENCE,
            reason="External fact must reference at least one original non-external fact.",
        )

    return None


def _references_original_facts(fact: RawFact, original_facts: List[RawFact]) -> bool:
    original_ids = [original.id for original in original_facts if not original.is_external]
    return any(ref_id in original_ids for ref_id in fact.referenced_fact_ids)


def _novelty_reason(kind: ExternalFactKind) -> str:
    if kind == ExternalFactKind.TECHNICAL_DEFINITION:
        return "Defines a non-obvious domain or technical term referenced by original facts."
    if kind == ExternalFactKind.ARCHITECTURE_PATTERN:
        return "Provides a domain-specific architecture/modeling pattern not explicitly stated in the input."
    if kind == ExternalFactKind.DOMAIN_CONSTRAINT_HINT:
        return "Adds domain-specific constraint or metric context connected to original facts."
    if kind == ExternalFactKind.DOMAIN_PATTERN:
        return "Provides standard domain schema pattern for underspecified input."
    return "Adds domain-specific modeling context connected to original facts."


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())
