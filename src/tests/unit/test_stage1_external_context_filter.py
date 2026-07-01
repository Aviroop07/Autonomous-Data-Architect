from __future__ import annotations

from src.pipeline.stage1.middleware.external_context_filter import (
    ExternalFactRejectionCode,
    filter_external_facts,
)
from src.pipeline.stage1.models.raw_fact import ExternalFactKind, RawFact


def _original_facts() -> list[RawFact]:
    return [
        RawFact(
            id=1,
            fact="VM instances route tenant workloads onto compute nodes.",
        ),
        RawFact(
            id=2,
            fact="Billing ledgers have gross_charge, discount_applied, and net_bill.",
        ),
    ]


def _external_fact(fact_id: int, text: str, refs: list[int]) -> RawFact:
    return RawFact(id=fact_id, fact=text, referenced_fact_ids=refs, is_external=True)


def test_filter_accepts_auditor_approved_generic_text_if_references_are_valid():
    result = filter_external_facts(
        [
            _external_fact(
                10,
                "Schema Guideline: Use foreign keys to enforce referential integrity between related tables.",
                [1],
            ),
        ],
        _original_facts(),
    )

    assert len(result.accepted_facts) == 1
    assert result.rejected_facts == []


def test_filter_rejects_invalid_references():
    result = filter_external_facts(
        [
            _external_fact(
                10,
                "Domain Insight: Cloud billing ledgers separate gross charge, discount, and net bill for auditability.",
                [99],
            ),
        ],
        _original_facts(),
    )

    assert result.accepted_facts == []
    assert result.rejected_facts[0].code == ExternalFactRejectionCode.INVALID_REFERENCE


def test_filter_rejects_self_reference():
    result = filter_external_facts(
        [
            _external_fact(
                10,
                "Domain Insight: Cloud billing ledgers separate gross charge, discount, and net bill for auditability.",
                [10],
            ),
        ],
        _original_facts(),
    )

    assert result.accepted_facts == []
    assert result.rejected_facts[0].code == ExternalFactRejectionCode.SELF_REFERENCE


def test_filter_rejects_duplicate_external_fact_text():
    result = filter_external_facts(
        [
            _external_fact(
                10, "Domain Insight: Cloud billing ledgers preserve auditability.", [1]
            ),
            _external_fact(
                11, "Domain Insight: Cloud billing ledgers preserve auditability.", [1]
            ),
        ],
        _original_facts(),
    )

    assert len(result.accepted_facts) == 1
    assert len(result.rejected_facts) == 1
    assert (
        result.rejected_facts[0].code
        == ExternalFactRejectionCode.DUPLICATE_EXTERNAL_FACT
    )


def test_filter_preserves_enricher_external_kind():
    """The enricher's external_kind is the source of truth -- the filter must NOT
    overwrite it with the deterministic classifier."""
    fact = _external_fact(
        10, "A hypervisor allocates VMs onto physical compute nodes.", [1]
    )
    fact.external_kind = ExternalFactKind.ARCHITECTURE_PATTERN  # set by enricher
    result = filter_external_facts([fact], _original_facts())

    assert len(result.accepted_facts) == 1
    # Even though the text contains no architecture keywords, the enricher's label wins.
    assert (
        result.accepted_facts[0].external_kind == ExternalFactKind.ARCHITECTURE_PATTERN
    )


def test_filter_fallback_classifies_when_kind_missing():
    """When the enricher leaves external_kind unset, the prefix-independent
    fallback classifier fills it in."""
    result = filter_external_facts(
        [
            _external_fact(
                10,
                "In this domain, a VM is defined as a virtualized compute instance.",
                [1],
            ),
        ],
        _original_facts(),
    )

    assert len(result.accepted_facts) == 1
    assert (
        result.accepted_facts[0].external_kind == ExternalFactKind.TECHNICAL_DEFINITION
    )
