"""Tests for FactExtractorLoopAgent._compute_segment_offsets.

Offsets must be in original-string coordinates regardless of Unicode casing.
str.lower() is NOT length-preserving for all codepoints (e.g. Turkish İ),
so searching a lowered copy produces corrupted indices.
"""

from __future__ import annotations


from src.pipeline.stage1.agents.fact_extractor.agent import FactExtractorLoopAgent
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput
from src.pipeline.stage1.models.raw_fact import RawFact, Segment


def _make_output(segments: list[Segment]) -> RephrasedOutput:
    return RephrasedOutput(segments=segments)


def test_unicode_lowering_does_not_corrupt_offsets():
    """Turkish İ (U+0130) makes str.lower() grow by one byte. Offsets must
    still point into the original string, not the lowered copy."""
    text = "\u0130stanbul branch data. The customer places orders monthly."
    agent = FactExtractorLoopAgent()
    seg = Segment(
        text="The customer places orders monthly.",
        start_char=-1,
        end_char=-1,
        facts=[RawFact(id=1, fact="Customers place orders monthly.")],
    )
    output = _make_output([seg])
    agent._compute_segment_offsets(output, text)
    assert seg.start_char == 22, f"expected 22, got {seg.start_char}"
    assert seg.end_char == 57, f"expected 57, got {seg.end_char}"
    assert text[seg.start_char : seg.end_char] == seg.text


def test_cursor_advances_across_segments():
    text = "Hello World. Hello World."
    agent = FactExtractorLoopAgent()
    seg1 = Segment(
        text="Hello World.",
        start_char=-1,
        end_char=-1,
        facts=[RawFact(id=1, fact="hello")],
    )
    seg2 = Segment(
        text="Hello World.",
        start_char=-1,
        end_char=-1,
        facts=[RawFact(id=2, fact="hello again")],
    )
    output = _make_output([seg1, seg2])
    agent._compute_segment_offsets(output, text)
    # seg1 matches the first occurrence
    assert seg1.start_char == 0
    assert text[seg1.start_char : seg1.end_char] == "Hello World."
    # seg2 matches the second occurrence (cursor advanced past first)
    assert seg2.start_char > seg1.end_char
    assert text[seg2.start_char : seg2.end_char] == "Hello World."


def test_case_insensitive_match_preserves_offsets():
    text = "Hello World. Goodbye World."
    agent = FactExtractorLoopAgent()
    seg = Segment(
        text="hello world.",
        start_char=-1,
        end_char=-1,
        facts=[RawFact(id=1, fact="hello")],
    )
    output = _make_output([seg])
    agent._compute_segment_offsets(output, text)
    assert text[seg.start_char : seg.end_char].lower() == "hello world."
    assert text[seg.start_char] == "H"


def test_non_ascii_lower_preserves_indices():
    text = "Stra\u00dfe means street. Stra\u00dfe is German."
    agent = FactExtractorLoopAgent()
    seg = Segment(
        text="stra\u00dfe means street.",
        start_char=-1,
        end_char=-1,
        facts=[RawFact(id=1, fact="Stra\u00dfe means street")],
    )
    output = _make_output([seg])
    agent._compute_segment_offsets(output, text)
    assert text[seg.start_char : seg.end_char].lower() == "stra\u00dfe means street."
    assert text[seg.start_char] == "S"
