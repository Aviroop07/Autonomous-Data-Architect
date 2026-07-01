"""Offline wiring tests for the gated, gap-driven context-enrichment loop.

Fully deterministic -- NO LLM, NO network. These target the machinery that the
happy-path live runs never exercised (the auditor always accepted on round 1):

  1. Enricher round-2 query evolution: when the auditor reports an unresolved gap
     with new directed queries, the enricher must switch to those NEW queries
     instead of re-running the round-1 gap queries. (Fixes plan defect 1.4.)
  2. Evidence resolution: the auditor must be shown the GENUINE cached snippet
     text (resolved from the shared EvidenceStore by tag), not text the enricher
     could have doctored.

The auditor's *judgment* (deciding to reject as UNGROUNDED) needs the model and
lives in the integration suite; here we only prove the plumbing around it.
"""

from __future__ import annotations

from src.pipeline.stage1.agents.context_auditor.agent import ContextAuditorLoopAgent
from src.pipeline.stage1.agents.context_enricher.agent import ContextEnricherLoopAgent
from src.pipeline.stage1.models.context_audit import ContextAuditReport
from src.pipeline.stage1.models.coverage_report import (
    GapDimension,
    GapSeverity,
    SpecGap,
)
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.util.core.search_tool import EvidenceSnippet, EvidenceStore
from src.util.orchestration.loop_types import LoopContext


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _gap(gid: int, query: str, severity: GapSeverity = GapSeverity.MAJOR) -> SpecGap:
    return SpecGap(
        id=gid,
        dimension=GapDimension.ATTRIBUTE,
        description=f"gap {gid}",
        severity=severity,
        search_query=query,
    )


def _ctx(node_outputs: dict) -> LoopContext:
    """Minimal LoopContext for driving build_context() directly."""
    return LoopContext(
        initial_context="",
        current_node="enricher",
        iteration=1,
        node_outputs=node_outputs,
        history=[],
        det_errors=[],
        det_errors_by_node={},
        ema_issues=[],
    )


# --------------------------------------------------------------------------- #
# 1. Enricher round-2 query evolution
# --------------------------------------------------------------------------- #


def test_enricher_round1_uses_gap_queries():
    """With no auditor feedback yet, queries come from the open gaps."""
    gaps = [_gap(1, "gap one query"), _gap(2, "gap two query")]
    enricher = ContextEnricherLoopAgent(
        original_facts=[RawFact(id=1, fact="Applicants submit loan applications.")],
        gaps=gaps,
        evidence_store=EvidenceStore(),
    )
    assert enricher._derive_search_queries() == ["gap one query", "gap two query"]


def test_enricher_round2_switches_to_auditor_queries():
    """When the auditor reports unresolved gaps + new queries, the enricher must
    search the NEW queries, not re-run the round-1 gap queries (defect 1.4 fix)."""
    gaps = [_gap(1, "gap one query"), _gap(2, "gap two query")]
    enricher = ContextEnricherLoopAgent(
        original_facts=[RawFact(id=1, fact="Applicants submit loan applications.")],
        gaps=gaps,
        evidence_store=EvidenceStore(),
    )

    audit = ContextAuditReport(
        is_acceptable=False,
        accepted_fact_ids=[],
        unresolved_gap_ids=[2],
        next_search_queries=["debt-to-income ratio loan approval threshold"],
        retry_instructions="Close gap 2.",
    )
    # prior enricher output must exist for the accumulate step; empty is fine here.
    enricher.build_context(_ctx({"auditor": audit, "enricher": FactList(facts=[])}))

    # State updated from auditor feedback...
    assert enricher._open_gap_ids == {2}
    assert enricher._auditor_next_queries == [
        "debt-to-income ratio loan approval threshold"
    ]
    # ...and the next search uses the NEW query, NOT the round-1 gap queries.
    assert enricher._derive_search_queries() == [
        "debt-to-income ratio loan approval threshold"
    ]


def test_enricher_gaps_to_close_section_reflects_open_gaps():
    """The prompt's GAPS TO CLOSE section should list only still-open gaps."""
    gaps = [_gap(1, "q1"), _gap(2, "q2")]
    enricher = ContextEnricherLoopAgent(
        original_facts=[RawFact(id=1, fact="Applicants submit loan applications.")],
        gaps=gaps,
        evidence_store=EvidenceStore(),
    )
    audit = ContextAuditReport(
        is_acceptable=False, unresolved_gap_ids=[2], next_search_queries=["q2b"]
    )
    query = enricher.build_context(
        _ctx({"auditor": audit, "enricher": FactList(facts=[])})
    )
    assert "## GAPS TO CLOSE" in query
    assert "[Gap 2]" in query
    assert "[Gap 1]" not in query  # gap 1 is closed -> not listed


def test_enricher_accumulates_accepted_facts_across_rounds():
    """Facts the auditor accepted in a prior round are carried forward."""
    enricher = ContextEnricherLoopAgent(
        original_facts=[RawFact(id=1, fact="Applicants submit loan applications.")],
        gaps=[_gap(1, "q1")],
        evidence_store=EvidenceStore(),
    )
    prior = FactList(
        facts=[
            RawFact(
                id=101, fact="A credit score predicts default risk.", is_external=True
            )
        ]
    )
    audit = ContextAuditReport(
        is_acceptable=False,
        accepted_fact_ids=[101],
        unresolved_gap_ids=[1],
        next_search_queries=["q1b"],
    )
    enricher.build_context(_ctx({"auditor": audit, "enricher": prior}))
    assert [f.id for f in enricher.accumulated_accepted] == [101]


# --------------------------------------------------------------------------- #
# 2. Evidence resolution: auditor sees GENUINE snippets
# --------------------------------------------------------------------------- #


def test_auditor_context_injects_genuine_evidence_by_tag():
    """The auditor's context must contain the real cached snippet text resolved
    from the shared EvidenceStore for tags the proposed facts cite."""
    store = EvidenceStore()
    store._by_tag["E1"] = EvidenceSnippet(
        tag="E1",
        query="credit score",
        title="Credit score",
        url="https://en.wikipedia.org/wiki/Credit_score",
        text="A credit score (300-850, FICO) predicts default risk.",
        source="wikipedia",
    )
    auditor = ContextAuditorLoopAgent(
        original_facts=[RawFact(id=1, fact="Applicants submit loan applications.")],
        gaps=[_gap(1, "q1")],
        evidence_store=store,
    )
    proposed = FactList(
        facts=[
            RawFact(
                id=101,
                fact="Credit scores range 300-850 (FICO).",
                is_external=True,
                addresses_gap=1,
                evidence_refs=["E1"],
            )
        ]
    )
    ctx = _ctx({"enricher": proposed})
    ctx.current_node = "auditor"
    text = auditor.build_context(ctx)

    assert "## EVIDENCE (as retrieved)" in text
    assert "E1" in text
    assert "300-850, FICO" in text  # the GENUINE snippet, not the fact's wording
    # The proposed fact's citation metadata is surfaced for the grounding check.
    assert "evidence_refs" in text and "addresses_gap" in text


def test_auditor_context_reports_no_evidence_when_nothing_cited():
    """A proposed fact with empty evidence_refs (an inference) yields no cited
    evidence -- the auditor must still get a well-formed section."""
    store = EvidenceStore()
    auditor = ContextAuditorLoopAgent(
        original_facts=[RawFact(id=1, fact="Applicants submit loan applications.")],
        gaps=[_gap(1, "q1")],
        evidence_store=store,
    )
    proposed = FactList(
        facts=[
            RawFact(
                id=101,
                fact="This is an OLTP-style domain.",
                is_external=True,
                addresses_gap=1,
                evidence_refs=[],
            )
        ]
    )
    ctx = _ctx({"enricher": proposed})
    ctx.current_node = "auditor"
    text = auditor.build_context(ctx)
    assert "## EVIDENCE (as retrieved)" in text
    assert "None cited." in text


def test_evidence_store_resolve_ignores_unknown_tags():
    """resolve() returns only known tags and silently drops unknown ones."""
    store = EvidenceStore()
    store._by_tag["E1"] = EvidenceSnippet("E1", "q", "t", "u", "body", "web")
    resolved = store.resolve(["E1", "E999"])
    assert [s.tag for s in resolved] == ["E1"]
