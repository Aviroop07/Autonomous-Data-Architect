"""Stage 3 under partial failure: one shard's agent raises, the other must not
be collateral damage.

Uses a two-shard split (CUSTOMER alone, ORDER alone) with real Stage 3
orchestration -- `run_parallel_loops`, the real per-shard AgentLoop, the real
deterministic checker, the real merge. The only thing scripted is which shard's
generator call raises, which is expressed through the injected provider's
query-text router rather than by replacing anything inside the stage.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Tuple

import pytest
from pydantic import BaseModel

from src.orchestration.stage3.state import Stage3Output
from src.pipeline.stage3.agents.extraction_outputs import AuditReport
from src.pipeline.stage3.models.cross_shard import UnifiedExtractionOutput
from src.pipeline.stage3.models.probe import GroupReconciliation, LostShardReason
from src.tests.fixtures.canned_llm import CannedAgentProvider
from src.tests.fixtures.canned_payloads import stage1 as p1
from src.tests.fixtures.canned_payloads import stage3 as p3
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table


class ShardGeneratorError(RuntimeError):
    """Raised by the canned provider to stand in for the failure modes a live
    provider produces at this exact point: an unrecoverable API error, or a
    structured-output parse the retry budget could not repair."""


def _schema() -> Schema:
    """The same two-table schema Stage 2's mapper derives from the canned
    conceptual model, written out here so this module needs no Stage 2 run."""
    return Schema(
        tables=[
            Table(
                name="CUSTOMER",
                columns=[
                    Column(name="customer_id", data_type=DataType.INTEGER),
                    Column(name="name", data_type=DataType.VARCHAR),
                    Column(name="credit_score", data_type=DataType.INTEGER),
                    Column(name="annual_income", data_type=DataType.FLOAT),
                ],
                primary_key=["customer_id"],
                source_fact_ids=[1, 2, 3, 4],
            ),
            Table(
                name="ORDER",
                columns=[
                    Column(name="order_id", data_type=DataType.INTEGER),
                    Column(name="total_amount", data_type=DataType.FLOAT),
                    Column(name="status", data_type=DataType.VARCHAR),
                    Column(name="customer_id", data_type=DataType.INTEGER),
                ],
                primary_key=["order_id"],
                source_fact_ids=[5, 6],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="ORDER",
                referencing_column="customer_id",
                referred_table="CUSTOMER",
                source_fact_ids=[1],
            )
        ],
    )


def _single_table_shard(schema: Schema, name: str) -> Schema:
    table = next(t for t in schema.tables if t.name == name)
    return Schema(tables=[table], relationships=[])


def _facts():
    """Real `AtomicFact`s, produced by running the real Stage 1 conversion over
    the canned extraction -- not hand-built, so tags and ids match a real run."""
    from src.pipeline.stage1.models.rephrased_nl import convert_to_atomic

    extraction = p1.extraction()
    return convert_to_atomic(
        extraction.flat_facts,
        p1.tagger_output().facts,
        {
            f.id: (segment.text, segment.start_char, segment.end_char)
            for segment in extraction.segments
            for f in segment.facts
        },
    )


# The rendered shard prompt heads the shard's OWN tables with "### NAME" and
# cross-shard references with "### NAME (stub)". Both shards therefore mention
# both table names, so only the un-stubbed heading identifies a shard.
_OWNS_ORDER_RE = re.compile(r"### ORDER(?! \(stub\))")


def _order_shard_raises(query: str) -> Optional[BaseModel]:
    """Route the ORDER shard's generator call to a failure.

    The two shards run CONCURRENTLY and both ask for the same output model, so
    the rendered query is the only thing that tells them apart -- and both
    queries NAME both tables, since each lists its sibling under "STUB TABLES".
    The discriminator therefore has to be which table the shard actually OWNS.
    That it really does split the two is asserted in the test rather than
    assumed.
    """
    if _OWNS_ORDER_RE.search(query):
        raise ShardGeneratorError("simulated unrecoverable failure on the ORDER shard")
    return p3.customer_only_extraction()


async def _run_two_shards() -> Tuple[Stage3Output, CannedAgentProvider]:
    from src.orchestration.stage3.entry import orchestrate as stage3

    schema = _schema()
    provider = (
        CannedAgentProvider()
        .route(UnifiedExtractionOutput, _order_shard_raises)
        .script(UnifiedExtractionOutput, p3.customer_only_extraction)
        .script(AuditReport, p3.clean_audit_report)
        .script(GroupReconciliation, p3.empty_reconciliation)
    )
    output, _tokens = await stage3(
        schema=schema,
        facts=_facts(),
        shards=[
            _single_table_shard(schema, "CUSTOMER"),
            _single_table_shard(schema, "ORDER"),
        ],
        provider=provider,
    )
    return output, provider


@pytest.fixture(scope="module")
def two_shard_run() -> Tuple[Stage3Output, CannedAgentProvider]:
    return asyncio.run(_run_two_shards())


def test_the_surviving_shard_contributes_its_full_output(
    two_shard_run: Tuple[Stage3Output, CannedAgentProvider],
) -> None:
    """When one shard's generator raises, the other shard's constraints arrive
    COMPLETE -- not partially, not at all.

    This is the property `run_parallel_loops` exists for, and it cannot be
    stated by a unit test of that helper: the helper only promises "one entry per
    input, None on failure", whereas the thing that matters is that Stage 3's
    merge then turns that None into an empty contribution and keeps the sibling's
    real constraints all the way into `Stage3Output`.

    Mutation this catches: replace `run_parallel_loops` with a bare
    `asyncio.gather` (its documented pre-history) and the raising shard aborts
    the surviving shard's in-flight work, so this returns nothing at all rather
    than the CUSTOMER shard's two constraints. Equally, make
    `_extract_generator_output` re-raise on `result is None` and the whole run
    dies.
    """
    output, provider = two_shard_run

    # The router's discriminator has to be sound or this test proves nothing:
    # exactly one of the two concurrent generator queries may match it.
    generator_queries = provider.queries_for(UnifiedExtractionOutput)
    matching = [q for q in generator_queries if _OWNS_ORDER_RE.search(q)]
    assert len(generator_queries) >= 2, (
        f"expected at least one generator call per shard, got {len(generator_queries)}"
    )
    assert 0 < len(matching) < len(generator_queries), (
        "the owned-table query discriminator did not split the shards: "
        f"{len(matching)} of {len(generator_queries)} queries matched"
    )

    assert [d.column for d in output.distributions] == ["credit_score"], (
        f"the CUSTOMER shard's distribution is missing; got "
        f"{[d.column for d in output.distributions]}"
    )
    assert [sorted(c.columns) for c in output.correlations] == [
        ["annual_income", "credit_score"]
    ], f"the CUSTOMER shard's correlation is missing; got {output.correlations}"

    # And nothing was invented on behalf of the shard that failed.
    assert output.moment_targets == []
    assert output.logic_constraints == []
    assert output.state_sequences == []
    assert output.derived_columns == []


def test_a_failed_shard_is_reported_in_the_output_not_only_the_log(
    two_shard_run: Tuple[Stage3Output, CannedAgentProvider],
) -> None:
    """A lost shard must be visible to the CALLER, not just to whoever reads the
    log afterwards. Stage 4 consumes this object, not the log.

    Asserts the STRUCTURED record rather than substring-matching free text, so
    it pins which shard was lost, why, and which facts went unrepresented --
    a report that merely mentioned the word "shard" somewhere would not pass.
    """
    output, _provider = two_shard_run
    report = output.analysis_report

    assert report.lost_shards, (
        "nothing in Stage3AnalysisReport records that a shard's extraction "
        f"failed entirely; report.unsupported carries {report.unsupported}"
    )
    assert not report.is_complete, (
        "is_complete must be False when a shard was lost -- it is the flag a "
        "caller checks to know the constraint set is not the whole story"
    )
    lost = report.lost_shards[0]
    assert lost.reason is LostShardReason.EXTRACTION_FAILED
    assert lost.fact_references, (
        "a lost shard must name the facts it left unrepresented, otherwise a "
        "caller learns something was lost but not what"
    )
    # Also surfaced in the human-facing list, so a consumer reading only
    # `unsupported` still learns the run was incomplete.
    assert any("shard" in text.lower() for text in report.unsupported)


# FIXED: the deterministic checker now reports WHICH item it refused
# (RejectedItem), and extraction drops exactly those -- regardless of whether
# the retry budget was exhausted. Keying removal on `det_errors_exhausted` alone
# could not work here: the generator config sets error_refresh, so that flag
# reads bool(state.error_accumulator), which can be EMPTY on the final pass even
# though the checker rejected constraints every round.
def test_constraints_that_never_passed_the_deterministic_checker_are_not_shipped() -> (
    None
):
    """A shard whose deterministic checker never converged must not contribute
    unrepaired constraints downstream.

    The scripted extraction references `nonexistent_column`, so the real
    `canonicalize()` column-resolution step reports an error on every round until
    the retry budget is exhausted -- exactly the `det_errors_exhausted` case.
    """

    async def run() -> Stage3Output:
        from src.orchestration.stage3.entry import orchestrate as stage3

        schema = _schema()
        provider = (
            CannedAgentProvider()
            .script(UnifiedExtractionOutput, p3.unrepairable_extraction)
            .script(AuditReport, p3.clean_audit_report)
            .script(GroupReconciliation, p3.empty_reconciliation)
        )
        output, _tokens = await stage3(
            schema=schema,
            facts=_facts(),
            shards=[_single_table_shard(schema, "ORDER")],
            provider=provider,
        )
        return output

    output = asyncio.run(run())
    offending = [
        c
        for c in output.logic_constraints
        if "nonexistent_column" in c.condition.model_dump_json()
    ]
    assert offending == [], (
        "a constraint the deterministic checker rejected on every round reached "
        f"Stage 3's output anyway: {offending}"
    )
