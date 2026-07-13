"""Offline tests for the deterministic context-filter loop node and the
enrichment loop's routing around it.

Fully deterministic -- NO LLM, NO network. Covers:
  1. ContextFilterLoopAgent policy: self-reference repair, invalid-reference
     flagging (-> get_errors, drives retry), duplicate drop, and the should_exit
     decision (structurally clean AND last auditor acceptable).
  2. The production loop wiring from make_enrichment_loop_config: the enricher
     skips the auditor on a structural-only retry, and the loop exits from the
     filter node.
"""

from __future__ import annotations

import asyncio

from src.orchestration.stage1.loop_config import make_enrichment_loop_config
from src.pipeline.stage1.middleware.context_filter_node import ContextFilterLoopAgent
from src.pipeline.stage1.models.context_audit import ContextAuditReport
from src.pipeline.stage1.models.coverage_report import (
    GapDimension,
    GapSeverity,
    SpecGap,
)
from src.pipeline.stage1.models.filter_report import FilterReport
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.util.orchestration.loop import AgentLoop
from src.util.orchestration.loop_types import (
    AgentRoleConfig,
    EdgeCondition,
    GraphEdge,
    HistoryEntry,
    LoopAgent,
    LoopConfig,
    LoopContext,
    LoopOutputModel,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _original_facts() -> list[RawFact]:
    return [
        RawFact(id=1, fact="Applicants submit loan applications."),
        RawFact(id=2, fact="Each application records a credit score."),
    ]


def _external(fact_id: int, text: str, refs: list[int]) -> RawFact:
    return RawFact(id=fact_id, fact=text, referenced_fact_ids=refs, is_external=True)


def _gap(gid: int, query: str) -> SpecGap:
    return SpecGap(
        id=gid,
        dimension=GapDimension.ATTRIBUTE,
        description=f"gap {gid}",
        severity=GapSeverity.MAJOR,
        search_query=query,
    )


def _run_filter(
    proposed: list[RawFact],
    *,
    auditor_acceptable: bool | None,
    originals: list[RawFact] | None = None,
) -> FilterReport:
    """Drive the filter node's build_context + invoke deterministically."""
    node_outputs: dict = {"enricher": FactList(facts=proposed)}
    if auditor_acceptable is not None:
        node_outputs["auditor"] = ContextAuditReport(is_acceptable=auditor_acceptable)
    ctx = LoopContext(
        initial_context="",
        current_node="filter",
        iteration=1,
        node_outputs=node_outputs,
        history=[],
        det_errors=[],
        det_errors_by_node={},
        ema_issues=[],
    )
    agent = ContextFilterLoopAgent(original_facts=originals or _original_facts())
    agent.build_context(ctx)
    report, tokens = asyncio.run(agent.invoke(""))
    assert tokens == 0
    assert isinstance(report, FilterReport)
    return report


# --------------------------------------------------------------------------- #
# 1. Filter policy
# --------------------------------------------------------------------------- #


def test_self_reference_is_repaired_not_rejected():
    """A fact referencing itself AND a real original is repaired in place and
    accepted -- the self-id is stripped, the valid reference remains."""
    fact = _external(10, "A credit score predicts default risk.", [10, 1])
    report = _run_filter([fact], auditor_acceptable=True)

    assert report.self_ref_repaired == 1
    assert report.invalid_reference_count == 0
    assert report.get_errors() == []
    assert [f.id for f in report.accepted_facts] == [10]
    assert fact.referenced_fact_ids == [1]  # repaired in place


def test_self_reference_only_becomes_invalid_reference():
    """A fact that referenced ONLY itself has no anchor after repair, so it is
    flagged as an invalid reference the enricher must fix."""
    fact = _external(10, "A credit score predicts default risk.", [10])
    report = _run_filter([fact], auditor_acceptable=True)

    assert report.self_ref_repaired == 1
    assert report.invalid_reference_count == 1
    assert report.get_errors()  # non-empty -> routed back to enricher
    assert report.accepted_facts == []
    assert report.should_exit is False  # structural error blocks exit


def test_invalid_reference_is_flagged_as_error():
    fact = _external(10, "Cloud billing separates gross and net charges.", [99])
    report = _run_filter([fact], auditor_acceptable=True)

    assert report.invalid_reference_count == 1
    assert len(report.get_errors()) == 1
    assert "10" in report.get_errors()[0]
    assert report.accepted_facts == []


def test_duplicate_is_dropped_silently_not_errored():
    facts = [
        _external(10, "Credit scores predict default risk.", [1]),
        _external(11, "Credit scores predict default risk.", [1]),
    ]
    report = _run_filter(facts, auditor_acceptable=True)

    assert report.duplicates_dropped == 1
    assert report.invalid_reference_count == 0
    assert report.get_errors() == []  # dedup is not a retry-driving error
    assert len(report.accepted_facts) == 1


def test_should_exit_true_when_clean_and_auditor_acceptable():
    fact = _external(10, "Credit scores predict default risk.", [1])
    report = _run_filter([fact], auditor_acceptable=True)
    assert report.should_exit is True


def test_should_exit_false_when_auditor_not_acceptable():
    """Structurally clean but the auditor still has open gaps -> keep looping."""
    fact = _external(10, "Credit scores predict default risk.", [1])
    report = _run_filter([fact], auditor_acceptable=False)
    assert report.should_exit is False


def test_should_exit_false_when_structural_error_even_if_auditor_acceptable():
    fact = _external(10, "Credit scores predict default risk.", [99])
    report = _run_filter([fact], auditor_acceptable=True)
    assert report.should_exit is False


# --------------------------------------------------------------------------- #
# 2. Production routing (real edges from make_enrichment_loop_config)
# --------------------------------------------------------------------------- #


def _enricher_ctx(node_outputs: dict) -> LoopContext:
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


def _edge(edges, frm, to):
    return next(e for e in edges if e.from_node == frm and e.to_node == to)


def test_routing_skips_auditor_on_structural_only_retry():
    """Last auditor acceptable -> enricher.needs_audit False -> the enricher->auditor
    edge must NOT match and the enricher->filter fallback must match."""
    config, enricher, _auditor, _filter = make_enrichment_loop_config(
        original_facts=_original_facts(), gaps=[_gap(1, "q1")]
    )
    enricher.build_context(
        _enricher_ctx(
            {
                "auditor": ContextAuditReport(is_acceptable=True),
                "enricher": FactList(facts=[]),
            }
        )
    )
    assert enricher.needs_audit is False

    edges = config.graph["edges"]
    dummy = FactList(facts=[])
    assert _edge(edges, "enricher", "auditor").matches(dummy) is False
    assert _edge(edges, "enricher", "filter").matches(dummy) is True


def test_routing_audits_when_gaps_remain():
    """Last auditor NOT acceptable (or first round) -> needs_audit True -> routes
    to the auditor."""
    config, enricher, _auditor, _filter = make_enrichment_loop_config(
        original_facts=_original_facts(), gaps=[_gap(1, "q1")]
    )
    enricher.build_context(
        _enricher_ctx(
            {
                "auditor": ContextAuditReport(
                    is_acceptable=False,
                    unresolved_gap_ids=[1],
                    next_search_queries=["q1b"],
                ),
                "enricher": FactList(facts=[]),
            }
        )
    )
    assert enricher.needs_audit is True
    assert (
        _edge(config.graph["edges"], "enricher", "auditor").matches(FactList(facts=[]))
        is True
    )


def test_routing_first_round_requires_audit():
    config, enricher, _auditor, _filter = make_enrichment_loop_config(
        original_facts=_original_facts(), gaps=[_gap(1, "q1")]
    )
    # No prior auditor output at all.
    enricher.build_context(_enricher_ctx({}))
    assert enricher.needs_audit is True


def test_enricher_injects_filter_feedback_into_query():
    """Structural errors routed from the filter (ctx.det_errors) must appear in
    the enricher's query so it can re-anchor the offending facts."""
    _config, enricher, _auditor, _filter = make_enrichment_loop_config(
        original_facts=_original_facts(), gaps=[_gap(1, "q1")]
    )
    ctx = _enricher_ctx(
        {
            "auditor": ContextAuditReport(is_acceptable=True),
            "enricher": FactList(facts=[]),
        }
    )
    ctx.det_errors = ["External fact 10 references no valid original fact."]
    query = enricher.build_context(ctx)
    assert "## STRUCTURAL VALIDATION FEEDBACK" in query
    assert "External fact 10" in query


def test_routing_loop_exits_from_filter():
    """filter.should_exit True -> filter->enricher must NOT match, filter->end must."""
    config, _enricher, _auditor, _filter = make_enrichment_loop_config(
        original_facts=_original_facts(), gaps=[_gap(1, "q1")]
    )
    edges = config.graph["edges"]
    exit_report = FilterReport(should_exit=True)
    retry_report = FilterReport(should_exit=False)

    assert _edge(edges, "filter", "enricher").matches(exit_report) is False
    assert _edge(edges, "filter", "end").matches(exit_report) is True
    assert _edge(edges, "filter", "enricher").matches(retry_report) is True


# --------------------------------------------------------------------------- #
# 3. End-to-end AgentLoop execution (stub agents, real engine + edge fns)
# --------------------------------------------------------------------------- #


class _StubEnricher(LoopAgent):
    def __init__(self) -> None:
        self.needs_audit = True
        self.calls = 0

    def build_context(self, ctx: LoopContext) -> str:
        auditor = ctx.node_outputs.get("auditor")
        self.needs_audit = not (
            isinstance(auditor, ContextAuditReport) and auditor.is_acceptable
        )
        return ""

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        self.calls += 1
        return FactList(facts=[_external(10, "credit score fact", [1])]), 0

    def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
        return HistoryEntry(
            round=round_num, node=node, changes_summary="", was_improvement=True
        )


class _StubAuditor(LoopAgent):
    def __init__(self) -> None:
        self.calls = 0

    def build_context(self, ctx: LoopContext) -> str:
        return ""

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        self.calls += 1
        return ContextAuditReport(is_acceptable=True), 0

    def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
        return HistoryEntry(
            round=round_num, node=node, changes_summary="", was_improvement=True
        )


class _StubFilter(LoopAgent):
    def __init__(self, exits: list[bool]) -> None:
        self._exits = exits
        self.calls = 0

    def build_context(self, ctx: LoopContext) -> str:
        return ""

    async def invoke(self, query: str) -> tuple[LoopOutputModel, int]:
        should_exit = self._exits[min(self.calls, len(self._exits) - 1)]
        self.calls += 1
        return FilterReport(should_exit=should_exit), 0

    def emit_history(self, output, prior, round_num, node) -> HistoryEntry:
        return HistoryEntry(
            round=round_num, node=node, changes_summary="", was_improvement=True
        )


def _stub_config(enricher, auditor, filt) -> LoopConfig:
    """The production topology, wired to stub agents (edge fn closes over the
    stub enricher, matching how make_enrichment_loop_config closes over the real one)."""
    return LoopConfig(
        agents={
            "enricher": AgentRoleConfig(
                agent_factory=lambda: enricher, det_error_sources=["filter"]
            ),
            "auditor": AgentRoleConfig(agent_factory=lambda: auditor),
            "filter": AgentRoleConfig(agent_factory=lambda: filt),
        },
        graph={
            "edges": [
                GraphEdge(
                    from_node="enricher",
                    to_node="auditor",
                    condition=EdgeCondition(fn=lambda _o: enricher.needs_audit),
                ),
                GraphEdge(from_node="enricher", to_node="filter"),
                GraphEdge(from_node="auditor", to_node="filter"),
                GraphEdge(
                    from_node="filter",
                    to_node="enricher",
                    condition=EdgeCondition(
                        fn=lambda o: not getattr(o, "should_exit", False)
                    ),
                ),
                GraphEdge(from_node="filter", to_node="end"),
            ]
        },
        start_node="enricher",
        max_iter=9,
    )


def test_end_to_end_skips_auditor_on_structural_retry_and_exits_from_filter():
    """Full engine run. Sequence:
      r1 enricher(need audit) -> r2 auditor(accept) -> r3 filter(no exit)
      -> r4 enricher(structural-only, skip auditor) -> r5 filter(exit) -> end.
    The auditor must be called exactly once (skipped on the second enricher visit)."""
    enricher = _StubEnricher()
    auditor = _StubAuditor()
    filt = _StubFilter(exits=[False, True])
    config = _stub_config(enricher, auditor, filt)

    result = asyncio.run(AgentLoop(config).run(""))

    assert result.final_node == "filter"
    assert enricher.calls == 2
    assert auditor.calls == 1  # skipped on the structural-only retry
    assert filt.calls == 2
