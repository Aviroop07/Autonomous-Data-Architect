"""The shard loop must return its best MEASURED draft, not its last draft.

A live retail run had the auditor report 5 -> 8 -> 5 findings across three
audits, then ran out of budget on a fourth extractor draft that no reviewer had
seen -- and that unmeasured draft was what shipped. These tests pin the
selection rule that replaced it.
"""

from __future__ import annotations

from typing import List

from src.orchestration.stage2.utils import select_best_shard_model
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel, Entity
from src.pipeline.stage2.models.conceptual_critique import (
    ConceptualCritiqueReport,
    SuggestedFix,
)
from src.util.orchestration.loop_types import LoopOutputModel, NodeOutputRecord


def _model(name: str) -> ConceptualModel:
    return ConceptualModel(
        entities=[Entity(name=name, attributes=[], source_fact_ids=[1])],
        relationships=[],
        functional_dependencies=[],
    )


def _audit(n_fixes: int) -> ConceptualCritiqueReport:
    return ConceptualCritiqueReport(
        is_valid=n_fixes == 0,
        fixes=[
            SuggestedFix(description=f"fix {i}", rationale="because fact 1")
            for i in range(n_fixes)
        ],
    )


class _FilterReport(LoopOutputModel):
    is_valid: bool
    det_errors: List[str] = []

    def get_errors(self) -> List[str]:
        return list(self.det_errors)


def _trace(*pairs: tuple[str, LoopOutputModel]) -> List[NodeOutputRecord]:
    return [
        NodeOutputRecord(iteration=i + 1, node=node, output=out)
        for i, (node, out) in enumerate(pairs)
    ]


def test_picks_the_lowest_finding_draft_not_the_last() -> None:
    """The observed 5 -> 8 -> 5 -> unmeasured shape, in miniature."""
    trace = _trace(
        ("extractor", _model("A")),
        ("filter", _FilterReport(is_valid=True)),
        ("auditor", _audit(5)),
        ("extractor", _model("B")),
        ("filter", _FilterReport(is_valid=True)),
        ("auditor", _audit(8)),
        ("extractor", _model("C")),
        ("filter", _FilterReport(is_valid=True)),
        ("auditor", _audit(5)),
        ("extractor", _model("D")),  # budget ran out -- nobody reviewed this
    )
    best, reason = select_best_shard_model(trace)
    assert best is not None
    # C, not D: it ties A on findings and a later draft has seen more feedback.
    assert best.entities[0].name == "C"
    assert "5 finding" in reason


def test_a_clean_audit_wins_outright() -> None:
    trace = _trace(
        ("extractor", _model("A")),
        ("filter", _FilterReport(is_valid=True)),
        ("auditor", _audit(3)),
        ("extractor", _model("B")),
        ("filter", _FilterReport(is_valid=True)),
        ("auditor", _audit(0)),
    )
    best, _ = select_best_shard_model(trace)
    assert best is not None and best.entities[0].name == "B"


def test_a_filter_rejected_draft_is_disqualified_however_few_findings() -> None:
    """Filter errors are structural, so a rejected draft is not a candidate at
    all -- even though no audit ever gave it a finding count."""
    trace = _trace(
        ("extractor", _model("BROKEN")),
        ("filter", _FilterReport(is_valid=False, det_errors=["dangling owner"])),
        ("extractor", _model("OK")),
        ("filter", _FilterReport(is_valid=True)),
        ("auditor", _audit(4)),
    )
    best, _ = select_best_shard_model(trace)
    assert best is not None and best.entities[0].name == "OK"


def test_falls_back_to_the_last_draft_when_nothing_was_reviewed() -> None:
    """The single-round case, where this must reduce to the old behavior."""
    trace = _trace(("extractor", _model("ONLY")))
    best, reason = select_best_shard_model(trace)
    assert best is not None and best.entities[0].name == "ONLY"
    assert "no draft was ever audited" in reason


def test_all_drafts_rejected_still_returns_one_rather_than_losing_the_shard() -> None:
    """Silently returning nothing would drop the shard's entire contribution.
    A structurally flawed model at least fails loudly downstream."""
    trace = _trace(
        ("extractor", _model("A")),
        ("filter", _FilterReport(is_valid=False, det_errors=["dangling owner"])),
        ("extractor", _model("B")),
        ("filter", _FilterReport(is_valid=False, det_errors=["dangling owner"])),
    )
    best, reason = select_best_shard_model(trace)
    assert best is not None and best.entities[0].name == "B"
    assert "failed the structural filter" in reason


def test_an_empty_trace_yields_no_model_rather_than_raising() -> None:
    best, reason = select_best_shard_model([])
    assert best is None
    assert "no usable draft" in reason
