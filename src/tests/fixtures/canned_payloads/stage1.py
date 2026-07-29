"""Canned Stage 1 agent outputs, plus the specification they describe.

The segment texts below MUST be verbatim substrings of `SPEC` -- Stage 1's real
extraction validator checks exactly that, and it runs unmodified here. If a
segment stops matching, the offline run fails the same way a live one would.
"""

from __future__ import annotations

from src.pipeline.stage1.models.coverage_report import CoverageReport
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.pipeline.stage1.models.raw_fact import RawFact, Segment
from src.pipeline.stage1.models.rephrased_nl import (
    RephrasedOutput,
    TaggedFact,
    TaggerOutput,
)

SPEC = (
    "Customers place orders. "
    "Each customer has a name, a credit score and an annual income. "
    "Each order has a total amount and a status."
)

SEG_ORDERS = "Customers place orders."
SEG_CUSTOMER_ATTRS = "Each customer has a name, a credit score and an annual income."
SEG_ORDER_ATTRS = "Each order has a total amount and a status."

#: The single-fact spec used by the degenerate-input test.
SINGLE_FACT_SPEC = "Each customer has a name."


def _segment(text: str, facts: list[RawFact]) -> Segment:
    start = SPEC.find(text)
    assert start != -1, f"segment {text!r} is not a verbatim substring of SPEC"
    return Segment(text=text, start_char=start, end_char=start + len(text), facts=facts)


def extraction() -> RephrasedOutput:
    """Six facts across three segments. Ids 1-6 are the identities every
    cross-stage invariant in the offline layer is stated in terms of."""
    return RephrasedOutput(
        domain="Retail",
        analytical_goal="Simulate customer ordering behaviour.",
        segments=[
            _segment(
                SEG_ORDERS,
                [RawFact(id=1, fact="A customer can place many orders.")],
            ),
            _segment(
                SEG_CUSTOMER_ATTRS,
                [
                    RawFact(id=2, fact="A customer has a name."),
                    RawFact(id=3, fact="A customer has a credit score."),
                    RawFact(id=4, fact="A customer has an annual income."),
                ],
            ),
            _segment(
                SEG_ORDER_ATTRS,
                [
                    RawFact(id=5, fact="An order has a total amount."),
                    RawFact(id=6, fact="An order has a status."),
                ],
            ),
        ],
    )


def clean_integrity_report() -> IntegrityReport:
    return IntegrityReport(is_safe=True)


def complete_coverage_report() -> CoverageReport:
    """No gaps, so the enrichment gate stays CLOSED.

    That is deliberate and it is what keeps this layer network-free without any
    interception: enrichment is the only Stage 1 path that reaches web search, so
    a spec the auditor judges complete never constructs a search at all.
    """
    return CoverageReport(
        detected_entities=["CUSTOMER", "ORDER"],
        detected_relationships=["CUSTOMER places ORDER"],
        gaps=[],
    )


def tagger_output() -> TaggerOutput:
    """Every fact carries a tag in Stage 2's `_REQUIRED_FACT_TAGS` set, so every
    one of them is inside the partition invariant's scope -- a fact tagged only
    METADATA would be excused from coverage and weaken the test."""
    return TaggerOutput(
        facts=[
            TaggedFact(id=1, tags=["STRUCTURAL"]),
            TaggedFact(id=2, tags=["STRUCTURAL"]),
            TaggedFact(id=3, tags=["STATISTICAL"]),
            TaggedFact(id=4, tags=["STATISTICAL"]),
            TaggedFact(id=5, tags=["LOGICAL"]),
            TaggedFact(id=6, tags=["LOGICAL"]),
        ]
    )


def single_fact_extraction() -> RephrasedOutput:
    text = SINGLE_FACT_SPEC
    return RephrasedOutput(
        domain="Retail",
        analytical_goal="Minimal single-fact specification.",
        segments=[
            Segment(
                text=text,
                start_char=0,
                end_char=len(text),
                facts=[RawFact(id=1, fact="A customer has a name.")],
            )
        ],
    )


def single_fact_tagger_output() -> TaggerOutput:
    return TaggerOutput(facts=[TaggedFact(id=1, tags=["STRUCTURAL"])])


def empty_extraction() -> RephrasedOutput:
    """What an extractor returns for an empty specification: no segments at all.
    Not an error -- there is genuinely nothing to extract."""
    return RephrasedOutput(domain="Unknown", analytical_goal="Unknown", segments=[])


def empty_tagger_output() -> TaggerOutput:
    return TaggerOutput(facts=[])
