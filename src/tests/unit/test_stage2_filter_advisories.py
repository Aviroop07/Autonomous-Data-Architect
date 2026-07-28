"""An advisory must never fail a conceptual model.

The FK-name heuristic used to set is_valid=False, so a legitimately-named
attribute bounced the model back to the extractor and spent a retry round. A
live retail shard lost the first two of its five rounds that way, before the
auditor had run even once.
"""

from __future__ import annotations

import pytest

from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
)
from src.pipeline.stage2.middleware.conceptual_filter_node import (
    ConceptualFilterLoopAgent,
    ConceptualFilterReport,
)
from src.util.orchestration.loop_types import LoopContext
from src.util.schema_model.data_types import DataType


def _ctx(model: ConceptualModel | None) -> LoopContext:
    return LoopContext(
        initial_context="",
        current_node="filter",
        iteration=1,
        node_outputs={"extractor": model} if model is not None else {},
        history=[],
        det_errors=[],
        det_errors_by_node={},
        ema_issues=[],
    )


def _entity(name: str, attrs: list[str], fact_ids: list[int]) -> Entity:
    return Entity(
        name=name,
        attributes=[
            CMAttribute(name=a, type=DataType.VARCHAR, source_fact_ids=fact_ids)
            for a in attrs
        ],
        identifier_attributes=[],
        source_fact_ids=fact_ids,
    )


async def _run(agent: ConceptualFilterLoopAgent, model) -> ConceptualFilterReport:
    agent.build_context(_ctx(model))
    report, tokens = await agent.invoke("")
    assert tokens == 0
    assert isinstance(report, ConceptualFilterReport)
    return report


@pytest.mark.asyncio
async def test_fk_name_heuristic_advises_but_does_not_invalidate() -> None:
    model = ConceptualModel(
        entities=[_entity("ORDER", ["order_number"], [1])],
        relationships=[],
        functional_dependencies=[],
    )
    report = await _run(ConceptualFilterLoopAgent([1]), model)

    assert report.is_valid, "a naming guess must not fail a structurally sound model"
    assert report.advisories, "but the generator should still be told about it"
    assert report.det_errors == []
    # get_errors feeds the loop's unresolved-issue accounting, so an advisory
    # must not appear there either.
    assert report.get_errors() == []


@pytest.mark.asyncio
async def test_unreferenced_fact_ids_are_named_exactly() -> None:
    model = ConceptualModel(
        entities=[_entity("ORDER", ["placed_at"], [1])],
        relationships=[],
        functional_dependencies=[],
    )
    report = await _run(ConceptualFilterLoopAgent([1, 2, 7]), model)

    joined = " ".join(report.advisories)
    assert "2, 7" in joined, joined
    assert report.is_valid, "an unreferenced fact is not proof of a structural defect"


@pytest.mark.asyncio
async def test_a_fully_cited_model_raises_no_provenance_advisory() -> None:
    model = ConceptualModel(
        entities=[_entity("ORDER", ["placed_at"], [1, 2])],
        relationships=[],
        functional_dependencies=[],
    )
    report = await _run(ConceptualFilterLoopAgent([1, 2]), model)
    assert not any("source_fact_ids" in a for a in report.advisories)


@pytest.mark.asyncio
async def test_a_missing_model_is_still_a_hard_error() -> None:
    report = await _run(ConceptualFilterLoopAgent([1]), None)
    assert not report.is_valid
    assert report.get_errors()


@pytest.mark.asyncio
async def test_no_fact_ids_supplied_disables_the_provenance_check() -> None:
    """Callers that do not pass facts must not get a spurious advisory."""
    model = ConceptualModel(
        entities=[_entity("ORDER", ["placed_at"], [])],
        relationships=[],
        functional_dependencies=[],
    )
    report = await _run(ConceptualFilterLoopAgent(), model)
    assert not any("source_fact_ids" in a for a in report.advisories)
