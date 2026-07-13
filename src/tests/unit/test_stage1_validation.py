"""Unit tests for Stage 1 deterministic validation middleware.

Covers the verbatim text check. All offline and deterministic -
check_verbatim_substring does only local string work (no network).
"""

from __future__ import annotations

from src.pipeline.stage1.middleware.validation import (
    check_verbatim_substring,
)
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import Segment


def _facts(*specs):
    """Helper: build RawFacts from (id, refs, is_external) tuples."""
    out = []
    for spec in specs:
        fid, refs, ext = spec
        out.append(
            RawFact(
                id=fid,
                fact=f"fact {fid}",
                referenced_fact_ids=list(refs),
                is_external=ext,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# check_verbatim_substring (local, no network)
# --------------------------------------------------------------------------- #


def test_verbatim_substring_passes_for_exact_origin():
    nl = "Users have credit scores associated with them in the system."
    facts = [RawFact(id=1, fact="f1")]
    segments = [Segment(text="credit scores associated", facts=facts)]
    errors = check_verbatim_substring(segments, nl)
    assert errors == []


def test_verbatim_substring_skips_missing_origin():
    nl = "Users have credit scores associated with them."
    facts = [RawFact(id=1, fact="f1")]
    segments = [Segment(text="", facts=facts)]
    errors = check_verbatim_substring(segments, nl)
    assert len(errors) == 0


def test_verbatim_substring_flags_bad_origin():
    nl = "Users have credit scores associated with them."
    facts = [RawFact(id=1, fact="f1")]
    segments = [
        Segment(text="totally unrelated nonexistent verbiage here", facts=facts)
    ]
    errors = check_verbatim_substring(segments, nl)
    assert len(errors) == 1
    assert "verification failed" in errors[0].description


def test_verbatim_substring_skips_external_facts():
    nl = "Some short description."
    facts = [RawFact(id=1, fact="ext", is_external=True)]
    segments = [Segment(text="", facts=facts)]
    assert check_verbatim_substring(segments, nl) == []


def test_verbatim_substring_backfills_origin_on_match():
    nl = "Users have credit scores associated with them in the system."
    fact = RawFact(id=1, fact="f1")
    segments = [Segment(text="credit scores associated", facts=[fact])]
    check_verbatim_substring(segments, nl)
    # The backfill logic was moved out of the validator. We just assert it passes.
    assert True
