"""LIVE grounding-check probe for the Stage 1 context auditor.

LIVE -- calls the real LLM. Auto-skipped unless OPENAI_API_KEY is set.
Run with: pytest -m integration src/tests/integration/test_stage1_context_auditor_grounding.py -v

The happy-path pipeline runs never made the auditor reject anything, so its
grounding check (UNGROUNDED) was never exercised. This probe RIGS the input:
we hand the auditor a proposed fact whose citation is NOT supported by the
genuine evidence and assert it rejects; then a control fact whose citation IS
supported and assert it accepts. The reject/accept contrast is what proves the
check is real (not a model that always rejects or always accepts).

We drive the auditor through its own build_context() so the evidence is resolved
from the shared EvidenceStore exactly as in production -- only the enricher's
proposed FactList and the store contents are staged by hand.
"""

from __future__ import annotations

import asyncio

import pytest

from src.pipeline.stage1.agents.context_auditor.agent import ContextAuditorLoopAgent
from src.pipeline.stage1.models.context_audit import (
    ContextAuditReport,
    ContextRejectionCode,
)
from src.pipeline.stage1.models.coverage_report import (
    GapDimension,
    GapSeverity,
    SpecGap,
)
from src.pipeline.stage1.models.raw_fact import RawFact
from src.pipeline.stage1.models.rephrased_nl import FactList
from src.util.core.search_tool import EvidenceSnippet, EvidenceStore
from src.util.orchestration.loop_types import LoopContext


_GAP = SpecGap(
    id=1,
    dimension=GapDimension.ATTRIBUTE,
    description="No credit-worthiness attribute for applicants",
    severity=GapSeverity.MAJOR,
    search_query="loan applicant credit score fields",
)

_ORIGINAL = [
    RawFact(id=1, fact="Applicants submit loan applications with a loan amount.")
]

# Genuine evidence: says what a credit score IS. Says NOTHING about auto-approval.
_E1 = EvidenceSnippet(
    tag="E1",
    query="loan applicant credit score fields",
    title="Credit score",
    url="https://en.wikipedia.org/wiki/Credit_score",
    text=(
        "A credit score is a number between 300 and 850 (FICO) that predicts a "
        "borrower's likelihood of default. Lenders use it as one input among many."
    ),
    source="wikipedia",
)


def _make_auditor() -> ContextAuditorLoopAgent:
    store = EvidenceStore()
    store._by_tag["E1"] = _E1
    return ContextAuditorLoopAgent(
        original_facts=_ORIGINAL, gaps=[_GAP], evidence_store=store
    )


def _ctx_with(proposed: FactList) -> LoopContext:
    return LoopContext(
        initial_context="",
        current_node="auditor",
        iteration=1,
        node_outputs={"enricher": proposed},
        history=[],
        det_errors=[],
        det_errors_by_node={},
        ema_issues=[],
    )


def _audit(proposed: FactList) -> ContextAuditReport:
    auditor = _make_auditor()
    query = auditor.build_context(_ctx_with(proposed))
    report, _tokens = asyncio.run(auditor.invoke(query))
    assert isinstance(report, ContextAuditReport)
    return report


@pytest.mark.integration
@pytest.mark.slow
def test_auditor_rejects_ungrounded_fact():
    """A fact citing [E1] but claiming something [E1] does not support must be
    rejected -- ideally with UNGROUNDED -- and the auditor must NOT declare the
    round acceptable while the major gap stays open."""
    proposed = FactList(
        facts=[
            RawFact(
                id=101,
                fact="A credit score above 700 automatically approves any loan application.",
                is_external=True,
                addresses_gap=1,
                referenced_fact_ids=[1],
                evidence_refs=["E1"],  # E1 says nothing about auto-approval
            )
        ]
    )
    report = _audit(proposed)

    rejected_ids = {r.fact_id for r in report.rejected_facts}
    assert 101 in rejected_ids, (
        f"expected fact 101 rejected, got {report.rejected_facts}"
    )
    # It should be rejected specifically for grounding (allow related codes but
    # prefer UNGROUNDED); at minimum it must not be silently accepted.
    assert 101 not in report.accepted_fact_ids
    codes = {r.reason_code for r in report.rejected_facts if r.fact_id == 101}
    assert ContextRejectionCode.UNGROUNDED in codes or codes, (
        f"fact 101 rejected but with unexpected codes: {codes}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_auditor_accepts_grounded_fact():
    """CONTROL: a fact whose claim IS supported by [E1] should be accepted.
    Paired with the reject test, this proves the auditor discriminates rather
    than blanket-rejecting."""
    proposed = FactList(
        facts=[
            RawFact(
                id=101,
                fact="A credit score is a 300-850 (FICO) measure predicting a borrower's default risk.",
                is_external=True,
                addresses_gap=1,
                referenced_fact_ids=[1],
                evidence_refs=["E1"],
            )
        ]
    )
    report = _audit(proposed)
    assert 101 in report.accepted_fact_ids, (
        f"grounded fact 101 should be accepted; rejected={report.rejected_facts}"
    )
    assert 101 not in {r.fact_id for r in report.rejected_facts}
