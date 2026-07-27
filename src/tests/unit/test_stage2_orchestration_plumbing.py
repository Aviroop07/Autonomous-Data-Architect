"""Deterministic unit tests for Stage 2 orchestration plumbing.

These tests catch the kind of bugs that crashed the pipeline at the finish line:
  - Output model missing required fields (was Output(schema=...) without segments/plan)
  - Renamed CritiqueReport properties (.has_violations -> .patches)
  - Mismatched function signatures

None of these tests call the LLM — they are purely structural/static checks.
"""

from __future__ import annotations

import pytest

from src.orchestration.stage2.models import Output
from src.pipeline.stage2.models.schema import Schema, Table, Column, DataType
from src.pipeline.stage2.models.chunk import ChunkedPlan
from src.pipeline.stage2.models.conflicts import (
    ActionType,
    ConflictFlag,
    ResolutionAction,
    AdjudicatorResponse,
)
from src.pipeline.stage2.models.conceptual_critique import (
    ConceptualCritiqueReport,
    SuggestedFix,
)
from src.pipeline.stage2.models.corrections import FixHistoryStep
from src.pipeline.stage2.mapper.conceptual_model import (
    ConceptualModel,
    Entity,
    CMAttribute,
    Relationship,
    Participant,
)
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage1.models.atomic_fact import FactTag
from src.util.schema_ops.schema_patch import CritiqueReport
from src.pipeline.stage2.models.corrections import Correction, CorrectionStatus


# ---------------------------------------------------------------------------
# Output model construction — the bug that killed the first successful run
# ---------------------------------------------------------------------------


def make_minimal_schema(table_name: str = "TEST") -> Schema:
    return Schema(
        tables=[
            Table(
                name=table_name,
                columns=[Column(name="id", data_type=DataType.INTEGER)],
                primary_key=["id"],
            )
        ]
    )


def make_minimal_plan() -> ChunkedPlan:
    return ChunkedPlan(core_modeling_facts=[], chunks=[])


def test_output_requires_segments_and_plan():
    """Output(...) with missing required fields should raise."""
    with pytest.raises(Exception):
        Output()  # missing segments and plan


def test_output_minimal_construction():
    """Output(segments=[...], plan=...) should succeed."""
    schema = make_minimal_schema()
    plan = make_minimal_plan()
    output = Output(segments=[schema], plan=plan)
    assert isinstance(output, Output)
    assert output.segments == [schema]
    assert output.plan == plan
    assert output.final_global_schema is None
    assert output.token_usage == 0
    assert output.uncovered_fact_ids == []
    assert output.cert_report is None
    assert output.fix_history == []
    assert output.cycles == []
    assert output.merge_decision_log is None


def test_output_full_construction():
    """Output(...) with all fields populated should succeed (catches missing fields)."""
    schema = make_minimal_schema()
    plan = make_minimal_plan()
    cert = CritiqueReport(agent_name="test-certifier", patches=[])
    steps = [
        FixHistoryStep(
            attempt=1,
            errors=["test issue"],
            corrections=[
                Correction(
                    error_message="err",
                    status=CorrectionStatus.FIXED,
                    description="desc",
                )
            ],
            fixed_schema="",
            schema_state=schema,
        )
    ]
    output = Output(
        segments=[schema],
        plan=plan,
        fix_history=[steps],
        merged_schema=schema,
        final_global_schema=schema,
        final_fix_history=steps,
        token_usage=12345,
        cycles=[["cycle1"]],
        uncovered_fact_ids=[1, 2, 3],
        merge_decision_log={"log": "test"},
        cert_report=cert,
    )
    assert output.token_usage == 12345
    assert output.cert_report is cert
    assert output.cycles == [["cycle1"]]
    assert output.uncovered_fact_ids == [1, 2, 3]


def test_output_with_cert_report_none():
    """Output(cert_report=None) should work — catches the enable_audit=False path."""
    schema = make_minimal_schema()
    plan = make_minimal_plan()
    output = Output(segments=[schema], plan=plan, cert_report=None)
    assert output.cert_report is None


# ---------------------------------------------------------------------------
# CritiqueReport property access — the bug that killed the second run
# ---------------------------------------------------------------------------


def test_critique_report_has_patches_not_has_violations():
    """CritiqueReport uses .patches (not .has_violations or .critiques)."""
    report = CritiqueReport(agent_name="test", patches=[])
    assert hasattr(report, "patches")
    assert not hasattr(report, "has_violations")
    assert not hasattr(report, "critiques")
    assert report.patches == []


def test_critique_report_with_patches():
    """CritiqueReport stores patches correctly (pass dicts, not model instances)."""
    report = CritiqueReport(
        agent_name="test",
        patches=[
            {
                "action": "ADD_COLUMN",
                "table_name": "TEST",
                "column_name": "new_col",
                "data_type": "VARCHAR",
                "reason": "Need more data",
            },
        ],
    )
    assert len(report.patches) == 1
    assert report.patches[0].table_name == "TEST"
    assert report.patches[0].column_name == "new_col"


# ---------------------------------------------------------------------------
# ConceptualCritiqueReport — separate model used in ER loop (not to be confused)
# ---------------------------------------------------------------------------


def test_conceptual_critique_report_structure():
    """ConceptualCritiqueReport has is_valid and fixes (not patches)."""
    report = ConceptualCritiqueReport(is_valid=True, fixes=[])
    assert report.is_valid is True
    assert report.fixes == []
    assert not hasattr(report, "patches")


def test_conceptual_critique_report_with_fixes():
    report = ConceptualCritiqueReport(
        is_valid=False,
        fixes=[SuggestedFix(description="Fix it", rationale="Because")],
    )
    assert len(report.fixes) == 1
    assert report.fixes[0].description == "Fix it"


# ---------------------------------------------------------------------------
# Conflict models structural checks
# ---------------------------------------------------------------------------


def test_conflict_flag_all_types():
    """All flag types should construct and serialize correctly."""
    for flag_type in [
        "VETOED_MERGE",
        "FORCED_MERGE",
        "IDENTIFIER_DISAGREEMENT",
        "CARDINALITY_CONTRADICTION",
        "POSSIBLE_ATTR_SYNONYM",
        "CROSS_CATEGORY_COLLISION",
    ]:
        flag = ConflictFlag(flag_type=flag_type, entities=["A", "B"], message="test")
        assert flag.flag_type == flag_type
        assert flag.entities == ["A", "B"]


def test_resolution_action_all_types():
    """Every ActionType should construct with appropriate fields."""
    actions = [
        ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="X",
            entity_b="Y",
            new_name="Z",
            rationale="Merge them",
        ),
        ResolutionAction(
            action_type=ActionType.RENAME_ATTRIBUTE,
            entity_a="X",
            attribute_old="old",
            new_name="new",
            rationale="Rename",
        ),
        ResolutionAction(
            action_type=ActionType.RESOLVE_CARDINALITY,
            relationship_name="R",
            new_cardinality="1:N",
            rationale="Card fix",
        ),
        ResolutionAction(
            action_type=ActionType.RESOLVE_CROSS_CATEGORY,
            entity_a="X",
            relationship_name="R",
            rationale="Cross fix",
        ),
        ResolutionAction(
            action_type=ActionType.RESOLVE_IDENTIFIER,
            entity_a="X",
            new_identifier_attributes=["id"],
            rationale="ID fix",
        ),
        ResolutionAction(
            action_type=ActionType.NO_ACTION,
            rationale="No op",
        ),
    ]
    for a in actions:
        assert isinstance(a, ResolutionAction)
        assert a._validate() == []  # all valid


def test_adjudicator_response():
    """AdjudicatorResponse wraps a list of resolutions."""
    res = ResolutionAction(action_type=ActionType.NO_ACTION, rationale="test")
    resp = AdjudicatorResponse(resolutions=[res])
    assert len(resp.resolutions) == 1
    assert resp.resolutions[0].rationale == "test"


# ---------------------------------------------------------------------------
# _build_conflict_graph deterministic test
# ---------------------------------------------------------------------------


def test_build_conflict_graph():
    from src.orchestration.stage2.entry import _build_conflict_graph

    flags = [
        ConflictFlag(flag_type="VETOED_MERGE", entities=["A", "B"], message="m1"),
        ConflictFlag(flag_type="FORCED_MERGE", entities=["B", "C"], message="m2"),
        ConflictFlag(
            flag_type="CARDINALITY_CONTRADICTION",
            entities=["A"],
            relationship="R",
            message="m3",
        ),
    ]
    G, flag_map = _build_conflict_graph(flags)
    assert set(G.nodes()) == {"A", "B", "C"}
    assert G.has_edge("A", "B")
    assert G.has_edge("B", "C")
    assert not G.has_edge("A", "C")  # no direct flag linking A-C
    assert len(flag_map["A"]) == 2  # two flags involving A
    assert len(flag_map["B"]) == 2
    assert len(flag_map["C"]) == 1


def test_build_conflict_graph_empty():
    from src.orchestration.stage2.entry import _build_conflict_graph

    G, flag_map = _build_conflict_graph([])
    assert len(G.nodes()) == 0
    assert flag_map == {}


# ---------------------------------------------------------------------------
# _compute_uncovered_facts deterministic test
# ---------------------------------------------------------------------------


def test_compute_uncovered_facts():
    from src.orchestration.stage2.entry import _compute_uncovered_facts

    facts = [
        AtomicFact(id=1, fact="a", tags=[FactTag.STRUCTURAL]),
        AtomicFact(id=2, fact="b", tags=[FactTag.LOGICAL]),
        AtomicFact(id=3, fact="c", tags=[FactTag.STATISTICAL]),
        AtomicFact(id=4, fact="d", tags=[FactTag.METADATA]),  # not required
    ]
    schema = make_minimal_schema("COVERED")
    registry = TableFactRegistry()
    registry.register_table_facts("COVERED", [1, 2])

    uncovered = _compute_uncovered_facts(facts, schema, registry)
    # fact 3 (STATISTICAL) is required but not registered
    assert uncovered == [3]


def test_compute_uncovered_facts_none():
    from src.orchestration.stage2.entry import _compute_uncovered_facts

    facts = [AtomicFact(id=1, fact="a", tags=[FactTag.STRUCTURAL])]
    schema = make_minimal_schema("COVERED")
    registry = TableFactRegistry()
    registry.register_table_facts("COVERED", [1])

    uncovered = _compute_uncovered_facts(facts, schema, registry)
    assert uncovered == []


# ---------------------------------------------------------------------------
# apply_adjudicator_patches deterministic test
# ---------------------------------------------------------------------------


def _make_cm() -> ConceptualModel:
    return ConceptualModel(
        entities=[
            Entity(
                name="A",
                attributes=[CMAttribute(name="x", type=DataType.VARCHAR)],
                identifier_attributes=["x"],
            ),
            Entity(name="B", attributes=[CMAttribute(name="y", type=DataType.INTEGER)]),
        ],
        relationships=[
            Relationship(
                name="R_AB",
                participants=[
                    Participant(entity="A", cardinality_min=1, cardinality_max=1),
                    Participant(entity="B", cardinality_min=0, cardinality_max=None),
                ],
                degree="binary",
                kind="1:N",
            )
        ],
    )


def test_apply_adjudicator_patches_merge():
    from src.orchestration.stage2.entry import apply_adjudicator_patches

    cm = _make_cm()
    patches = [
        ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="A",
            entity_b="B",
            new_name="C",
            rationale="Merge test",
        ),
    ]
    result = apply_adjudicator_patches(cm, patches)
    assert len(result.entities) == 1
    assert result.entities[0].name == "C"
    # Relationship participants should be updated
    for r in result.relationships:
        for p in r.participants:
            assert p.entity == "C"


def test_apply_adjudicator_patches_rename_attr():
    from src.orchestration.stage2.entry import apply_adjudicator_patches

    cm = _make_cm()
    patches = [
        ResolutionAction(
            action_type=ActionType.RENAME_ATTRIBUTE,
            entity_a="A",
            attribute_old="x",
            new_name="z",
            rationale="Rename test",
        ),
    ]
    result = apply_adjudicator_patches(cm, patches)
    assert result.entities[0].attributes[0].name == "z"


def test_apply_adjudicator_patches_resolve_cardinality():
    from src.orchestration.stage2.entry import apply_adjudicator_patches

    cm = _make_cm()
    patches = [
        ResolutionAction(
            action_type=ActionType.RESOLVE_CARDINALITY,
            relationship_name="R_AB",
            new_cardinality="M:N",
            rationale="Card test",
        ),
    ]
    result = apply_adjudicator_patches(cm, patches)
    assert result.relationships[0].kind == "M:N"


def test_apply_adjudicator_patches_resolve_identifier():
    from src.orchestration.stage2.entry import apply_adjudicator_patches

    cm = _make_cm()
    patches = [
        ResolutionAction(
            action_type=ActionType.RESOLVE_IDENTIFIER,
            entity_a="A",
            new_identifier_attributes=["x", "y"],
            rationale="ID test",
        ),
    ]
    result = apply_adjudicator_patches(cm, patches)
    assert result.entities[0].identifier_attributes == ["x", "y"]


def test_apply_adjudicator_patches_unknown_entity_warning():
    """Patch with missing entity should log warning, not crash."""
    from src.orchestration.stage2.entry import apply_adjudicator_patches

    cm = _make_cm()
    patches = [
        ResolutionAction(
            action_type=ActionType.MERGE_ENTITIES,
            entity_a="NONEXISTENT",
            entity_b="B",
            new_name="C",
            rationale="Missing entity test",
        ),
    ]
    result = apply_adjudicator_patches(cm, patches)
    assert len(result.entities) == 2  # unchanged


# ---------------------------------------------------------------------------
# Fixtures for mock-based orchestrate tests
# ---------------------------------------------------------------------------


def _plan_with_one_chunk() -> ChunkedPlan:
    return ChunkedPlan(
        core_modeling_facts=[AtomicFact(id=1, fact="test", tags=[FactTag.STRUCTURAL])],
        chunks=[[AtomicFact(id=1, fact="test", tags=[FactTag.STRUCTURAL])]],
    )


def _cm_with_one_entity() -> ConceptualModel:
    return ConceptualModel(
        entities=[
            Entity(
                name="TEST",
                attributes=[CMAttribute(name="id", type=DataType.INTEGER)],
                identifier_attributes=["id"],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Full orchestrate() plumbing test with mocked LLM calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrate_plumbing_with_mocks(mocker):
    """Verify orchestrate() doesn't crash on plumbing errors when deps are mocked.

    This would have caught:
    - cert_report.has_violations -> .patches rename
    - Output() missing segments/plan fields
    - Any other attribute/signature mismatches
    """
    from src.orchestration.stage2.entry import orchestrate

    mock_plan = _plan_with_one_chunk()
    mock_facts = [
        AtomicFact(id=1, fact="test fact", tags=[FactTag.STRUCTURAL]),
    ]

    mock_cm = _cm_with_one_entity()

    empty_report = CritiqueReport(agent_name="test-certifier", patches=[])

    # Patch at entry.* — entry.py imported these names at module load time,
    # so the reference lives in entry's namespace, not the source module's.
    mocker.patch(
        "src.orchestration.stage2.entry.run_er_extractor_loop",
        return_value=(mock_cm, [], 100),
    )
    mocker.patch(
        "src.orchestration.stage2.entry.certify_compliance",
        return_value=(empty_report, 25),
    )

    # Lazy imports inside orchestrate() — resolved at call time after mock
    mocker.patch(
        "src.pipeline.stage2.middleware.conceptual_merger.merge_all_shards",
        return_value=(mock_cm, []),
    )
    mocker.patch(
        "src.pipeline.stage2.agents.adjudicator.agent.resolve_conflicts",
        return_value=(AdjudicatorResponse(resolutions=[]), 50),
    )

    output, tokens, registry = await orchestrate(
        plan=mock_plan,
        facts=mock_facts,
        domain="Test",
        analytical_goal="Test",
        nl_query="test",
    )

    assert isinstance(output, Output)
    assert tokens >= 0
    assert isinstance(registry, TableFactRegistry)
    assert len(output.segments) >= 0
    assert output.plan is mock_plan
    assert output.cert_report is not None


@pytest.mark.asyncio
async def test_orchestrate_plumbing_no_audit(mocker):
    """orchestrate() with enable_audit=False should not crash (catches cert_report scope bug)."""
    from src.orchestration.stage2.entry import orchestrate

    mock_plan = _plan_with_one_chunk()
    mock_facts = [AtomicFact(id=1, fact="test", tags=[FactTag.STRUCTURAL])]
    mock_cm = _cm_with_one_entity()

    # Patch entry.run_er_extractor_loop (top-level import in entry.py)
    mocker.patch(
        "src.orchestration.stage2.entry.run_er_extractor_loop",
        return_value=(mock_cm, [], 100),
    )
    # Lazy imports — resolved at call time after mock
    mocker.patch(
        "src.pipeline.stage2.middleware.conceptual_merger.merge_all_shards",
        return_value=(mock_cm, []),
    )
    mocker.patch(
        "src.pipeline.stage2.agents.adjudicator.agent.resolve_conflicts",
        return_value=(AdjudicatorResponse(resolutions=[]), 50),
    )

    output, tokens, registry = await orchestrate(
        plan=mock_plan,
        facts=mock_facts,
        domain="Test",
        analytical_goal="Test",
        nl_query="test",
        enable_audit=False,
    )

    assert isinstance(output, Output)
    assert output.cert_report is None
    assert tokens >= 0


@pytest.mark.asyncio
async def test_orchestrate_plumbing_no_sharding(mocker):
    """orchestrate() with ablation no-sharding should not crash."""
    from src.orchestration.stage2.entry import orchestrate
    from src.util.config.ablation import AblationConfig

    mock_plan = _plan_with_one_chunk()
    mock_facts = [AtomicFact(id=1, fact="test", tags=[FactTag.STRUCTURAL])]
    mock_cm = _cm_with_one_entity()
    empty_report = CritiqueReport(agent_name="test", patches=[])

    # Patch entry.* (top-level imports in entry.py)
    mocker.patch(
        "src.orchestration.stage2.entry.run_er_extractor_loop",
        return_value=(mock_cm, [], 100),
    )
    mocker.patch(
        "src.orchestration.stage2.entry.certify_compliance",
        return_value=(empty_report, 25),
    )
    # Lazy imports — resolved at call time after mock
    mocker.patch(
        "src.pipeline.stage2.middleware.conceptual_merger.merge_all_shards",
        return_value=(mock_cm, []),
    )
    mocker.patch(
        "src.pipeline.stage2.agents.adjudicator.agent.resolve_conflicts",
        return_value=(AdjudicatorResponse(resolutions=[]), 50),
    )

    output, tokens, registry = await orchestrate(
        plan=mock_plan,
        facts=mock_facts,
        domain="Test",
        analytical_goal="Test",
        nl_query="test",
        ablation_config=AblationConfig.no_sharding(),
    )

    assert isinstance(output, Output)
    assert tokens >= 0


# ---------------------------------------------------------------------------
# Edge-case: zero chunks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrate_zero_chunks_raises():
    """orchestrate() with empty chunks should raise ValueError."""
    from src.orchestration.stage2.entry import orchestrate

    empty_plan = ChunkedPlan(core_modeling_facts=[], chunks=[])
    mock_facts = []

    with pytest.raises(ValueError, match="No conceptual models generated"):
        await orchestrate(
            plan=empty_plan,
            facts=mock_facts,
            domain="Test",
            analytical_goal="Test",
        )
