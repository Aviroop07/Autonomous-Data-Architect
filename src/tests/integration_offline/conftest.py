"""Shared setup for the offline cross-stage layer.

Every test here runs REAL stage code -- `orchestrate()` down through AgentLoop,
every validator, the Beta-mixture merger, the relational mapper,
`canonicalize()`, the DOF graph and the conflict engine -- with exactly one thing
substituted, through the front door: the `AgentProvider` that builds LLM agents.
"""

from __future__ import annotations

import asyncio
from typing import List, Tuple

import pytest

from src.orchestration.stage1.models import Output as Stage1Output
from src.orchestration.stage2.models import Output as Stage2Output
from src.orchestration.stage3.state import Stage3Output
from src.pipeline.stage1.models.coverage_report import CoverageReport
from src.pipeline.stage1.models.integrity_report import IntegrityReport
from src.pipeline.stage1.models.rephrased_nl import RephrasedOutput, TaggerOutput
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel
from src.pipeline.stage2.models.conceptual_critique import ConceptualCritiqueReport
from src.pipeline.stage3.agents.extraction_outputs import AuditReport
from src.pipeline.stage3.models.cross_shard import UnifiedExtractionOutput
from src.pipeline.stage3.models.probe import GroupReconciliation
from src.tests.fixtures.canned_llm import CannedAgentProvider
from src.tests.fixtures.canned_payloads import stage1 as p1
from src.tests.fixtures.canned_payloads import stage2 as p2
from src.tests.fixtures.canned_payloads import stage3 as p3
from src.util.schema_model.registry import TableFactRegistry
from src.util.schema_ops.schema_patch import CritiqueReport

#: Applied to every test in this directory by `pytest_collection_modifyitems`.
OFFLINE_MARKER = "offline_integration"


def pytest_collection_modifyitems(config, items) -> None:
    """Mark everything in this directory `offline_integration`.

    Applied here rather than repeated as a decorator on ~10 tests: the marker is a
    property of the DIRECTORY (these are the cross-stage tests), so a new file
    added here cannot forget it and silently escape `-m "not
    offline_integration"`.
    """
    for item in items:
        if item.path.is_relative_to(
            config.rootpath / "src" / "tests" / "integration_offline"
        ):
            item.add_marker(pytest.mark.offline_integration)


def pin_context_window(window: int = 200_000) -> None:
    """Stop the context-window lookup from reaching the network.

    Stage 1 resolves the model's real context window twice -- for its NL length
    ceiling (`entry._max_nl_chars`) and for the chunker's token budget -- and both
    go through `get_context_window`, which issues an HTTP request. Both callers
    already fall back to a safe constant when the lookup raises, so a run without
    a provider key is offline anyway; but a developer machine WITH a key would
    make a real request here.

    This seeds `get_context_window`'s own process-lifetime cache, which it
    consults before any HTTP call. It is a cache prime rather than an
    interception: the production code path is unchanged and still the one that
    runs. That lookup is deliberately NOT dependency-injected -- a resolver
    parameter on `orchestrate()` would exist for no caller but this one.
    """
    from src.util.core import context_window as cw

    try:
        from src.util.core.agent import _detect_provider

        provider, _key, _base_url, model = _detect_provider()
    except Exception:
        # No provider configured at all: both callers already take their offline
        # fallback path, so there is nothing to prime.
        return
    cw._resolved_cache[(provider, model)] = window


def script_full_pipeline(provider: CannedAgentProvider) -> CannedAgentProvider:
    """Script every agent the happy path of stages 1-3 calls."""
    return (
        provider.script(RephrasedOutput, p1.extraction)
        .script(IntegrityReport, p1.clean_integrity_report)
        .script(CoverageReport, p1.complete_coverage_report)
        .script(TaggerOutput, p1.tagger_output)
        .script(ConceptualModel, p2.conceptual_model)
        .script(ConceptualCritiqueReport, p2.clean_conceptual_critique)
        .script(CritiqueReport, p2.clean_compliance_report)
        .script(UnifiedExtractionOutput, p3.full_extraction)
        .script(AuditReport, p3.clean_audit_report)
        .script(GroupReconciliation, p3.empty_reconciliation)
    )


async def _run_all_three(
    provider: CannedAgentProvider,
) -> Tuple[Stage1Output, Stage2Output, Stage3Output, TableFactRegistry, List[int]]:
    from src.orchestration.stage1.entry import orchestrate as stage1
    from src.orchestration.stage2.entry import orchestrate as stage2
    from src.orchestration.stage3.entry import orchestrate as stage3

    s1_out, s1_tokens = await stage1(p1.SPEC, provider=provider)
    s2_out, s2_tokens, registry = await stage2(
        plan=s1_out.plan,
        facts=s1_out.final_facts,
        domain=s1_out.domain,
        analytical_goal=s1_out.analytical_goal,
        nl_query=p1.SPEC,
        provider=provider,
    )
    schema = s2_out.final_global_schema
    assert schema is not None, "Stage 2 produced no global schema"
    # `shards` is supplied explicitly rather than left to `shard_schema_auto()`:
    # the ILP sizes shards against the target model's live-queried context
    # window, so the automatic path is both network-bound and CPU-heavy. One
    # shard holding the whole schema is the faithful shape for a 2-table schema
    # anyway -- the ILP returns exactly that for an input this size.
    s3_out, s3_tokens = await stage3(
        schema=schema,
        facts=s1_out.final_facts,
        shards=[schema],
        provider=provider,
    )
    return s1_out, s2_out, s3_out, registry, [s1_tokens, s2_tokens, s3_tokens]


class PipelineRun:
    """The result of one stage1 -> stage2 -> stage3 run, shared by every
    assertion in the main module."""

    def __init__(
        self,
        stage1: Stage1Output,
        stage2: Stage2Output,
        stage3: Stage3Output,
        registry: TableFactRegistry,
        tokens: List[int],
        provider: CannedAgentProvider,
    ) -> None:
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.registry = registry
        self.tokens = tokens
        self.provider = provider

    @property
    def schema(self):
        schema = self.stage2.final_global_schema
        assert schema is not None
        return schema


@pytest.fixture(scope="session")
def pipeline_run() -> PipelineRun:
    """Run the whole pipeline ONCE for the session.

    Session-scoped and synchronous on purpose. The run is the expensive part
    (Stage 2 loads a real sentence-transformer for the merger); every assertion
    below is cheap and reads the same immutable result, so re-running per test
    would multiply the cost for no additional coverage.
    """
    pin_context_window()
    provider = script_full_pipeline(CannedAgentProvider())
    s1, s2, s3, registry, tokens = asyncio.run(_run_all_three(provider))
    return PipelineRun(s1, s2, s3, registry, tokens, provider)
