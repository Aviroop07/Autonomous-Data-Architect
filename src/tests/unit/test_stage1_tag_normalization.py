from __future__ import annotations

from src.pipeline.stage1.middleware.tag_normalization import normalize_stage1_tags
from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag


def test_enum_allowed_values_get_structural_and_logical_tags():
    facts = [
        AtomicFact(id=1, fact="The status of a VM instance has allowed values Active, Suspended, and Terminated.", tags=[FactTag.LOGICAL]),
    ]

    normalized = normalize_stage1_tags(facts)

    assert FactTag.LOGICAL in normalized[0].tags
    assert FactTag.STRUCTURAL in normalized[0].tags


def test_external_facts_keep_metadata_tag():
    facts = [
        AtomicFact(id=1, fact="Domain Insight: Cloud billing ledgers support auditability.", is_external=True, tags=[]),
    ]

    normalized = normalize_stage1_tags(facts)

    assert FactTag.METADATA in normalized[0].tags
