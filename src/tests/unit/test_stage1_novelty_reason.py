"""The enricher's own novelty_reason survives the filter.

_novelty_reason() used to overwrite the field unconditionally, so the model's
justification for why an external fact was non-redundant was discarded before
any consumer saw it, and every accepted fact of a given kind carried the same
canned sentence. external_kind on the line directly above already used the
keep-what-the-model-wrote-else-fall-back pattern; this makes the two agree.
"""

from __future__ import annotations

from src.pipeline.stage1.middleware.external_context_filter import (
    filter_external_facts,
)
from src.pipeline.stage1.models.raw_fact import ExternalFactKind, RawFact


def _external(reason: str | None) -> RawFact:
    return RawFact(
        id=100,
        fact="An external statement referencing fact 1.",
        is_external=True,
        external_kind=ExternalFactKind.TECHNICAL_DEFINITION,
        novelty_reason=reason,
        evidence_refs=["E1"],
        referenced_fact_ids=[1],
    )


def _original() -> RawFact:
    return RawFact(id=1, fact="An original statement.", is_external=False)


def _accepted(fact: RawFact):
    result = filter_external_facts([fact], [_original()])
    return result.accepted_facts


def test_model_supplied_reason_is_preserved():
    accepted = _accepted(_external("Defines a term the source uses but never explains."))
    assert accepted, "fact was rejected; the preservation check needs an accepted fact"
    assert accepted[0].novelty_reason == (
        "Defines a term the source uses but never explains."
    )


def test_empty_reason_falls_back_to_the_deterministic_one():
    accepted = _accepted(_external(None))
    assert accepted
    assert accepted[0].novelty_reason


def test_blank_string_also_falls_back():
    accepted = _accepted(_external(""))
    assert accepted
    assert accepted[0].novelty_reason
