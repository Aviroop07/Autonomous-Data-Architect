"""Tests for the deterministic control flow in the redesigned Stage 3
orchestration (src/orchestration/stage3/entry.py): single-generator-loop
per shard, global bridge+evaluate+schema-locality-grouping, and the
per-constraint retry-counter reconciliation loop. LLM-calling boundaries
(reconcile_conflict_group, _rerun_shard) are monkeypatched -- no live LLM
calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.orchestration.stage3 import entry as stage3_entry
from src.orchestration.stage3.entry import (
    _ConflictItem,
    _conflict_items_from,
    _fact_to_tables,
    _group_by_schema_locality,
    _merge_all,
    _reconcile_and_apply,
    _ShardState,
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
from src.pipeline.stage3.models.cross_shard import Constraint, UnifiedExtractionOutput
from src.pipeline.stage3.models.on_nodes import ONBaseTable
from src.pipeline.stage3.models.probe import (
    ConflictReconciliation,
    GroupReconciliation,
    MisextractionFix,
    ReconciliationVerdict,
)
from src.util.constraint_model.conflicts.models import Conflict
from src.util.constraint_model.relation.nodes import BaseTable


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="ORDER",
                primary_key=["id"],
                columns=[
                    Column(name="id", data_type=DataType.INTEGER, is_nullable=False),
                    Column(name="total", data_type=DataType.FLOAT, is_nullable=False),
                ],
            )
        ],
        relationships=[],
    )


def _range_constraint(fact_ids: list[int]) -> Constraint:
    return Constraint(
        fact_references=fact_ids,
        on=ONBaseTable(name="ORDER"),
        condition=RComparison(
            op=">=", left=RColumnRef(name="total"), right=RLiteral(value=5)
        ),
        category="structural",
    )


class TestMergeAll:
    def test_merges_every_shards_single_output(self):
        ss0 = _ShardState(index=0, schema=_schema(), fact_ids=[1], stub_tables=[])
        ss0.output = UnifiedExtractionOutput(
            moment_targets=[_range_constraint([1])],
            structural_constraints=[_range_constraint([1])],
        )
        ss1 = _ShardState(index=1, schema=_schema(), fact_ids=[2], stub_tables=[])
        ss1.output = UnifiedExtractionOutput(logic_constraints=[_range_constraint([2])])

        merged = _merge_all([ss0, ss1])
        assert len(merged.moment_targets) == 1
        assert len(merged.structural) == 1
        assert len(merged.logic) == 1
        assert merged.distributions == []
        assert merged.correlations == []
        assert merged.derived == []


class TestFactToTables:
    def test_maps_fact_id_to_union_of_relation_tables(self):
        c1 = type(
            "C", (), {"relation": BaseTable(name="ORDER"), "fact_references": [1, 2]}
        )()
        c2 = type(
            "C", (), {"relation": BaseTable(name="CUSTOMER"), "fact_references": [2]}
        )()
        mapping = _fact_to_tables([c1, c2])
        assert mapping[1] == frozenset({"ORDER"})
        assert mapping[2] == frozenset({"ORDER", "CUSTOMER"})


class TestConflictItemsFrom:
    def test_conflict_ref_stable_and_tables_derived_from_fact_map(self):
        conflict = Conflict(
            kind="moment_value_mismatch",
            summary="Two facts disagree.",
            involved_fact_references=[1, 2],
            detail="1 vs 2 mismatch.",
            softenable=True,
        )
        items = _conflict_items_from(
            [conflict], [], {1: frozenset({"ORDER"}), 2: frozenset({"ORDER"})}
        )
        assert len(items) == 1
        assert items[0].conflict_ref == "moment_value_mismatch::1-2"
        assert items[0].tables == frozenset({"ORDER"})

    def test_fact_independent_conflict_gets_empty_table_set(self):
        conflict = Conflict(
            kind="structural_overconstrained",
            summary="Overconstrained.",
            involved_fact_references=[],
            detail="No facts behind this.",
            softenable=False,
        )
        items = _conflict_items_from([conflict], [], {})
        assert items[0].fact_ids == []
        assert items[0].tables == frozenset()


class TestGroupBySchemaLocality:
    def test_conflicts_sharing_a_table_merge_into_one_group(self):
        a = _ConflictItem("a", "desc a", [1], frozenset({"ORDER"}))
        b = _ConflictItem("b", "desc b", [2], frozenset({"ORDER", "CUSTOMER"}))
        groups = _group_by_schema_locality([a, b])
        assert len(groups) == 1
        assert {it.conflict_ref for it in groups[0]} == {"a", "b"}

    def test_conflicts_over_disjoint_tables_stay_separate(self):
        a = _ConflictItem("a", "desc a", [1], frozenset({"ORDER"}))
        b = _ConflictItem("b", "desc b", [2], frozenset({"PRODUCT"}))
        groups = _group_by_schema_locality([a, b])
        assert len(groups) == 2

    def test_three_way_transitive_overlap_merges_all_three(self):
        a = _ConflictItem("a", "d", [1], frozenset({"ORDER"}))
        b = _ConflictItem("b", "d", [2], frozenset({"ORDER", "CUSTOMER"}))
        c = _ConflictItem("c", "d", [3], frozenset({"CUSTOMER", "ADDRESS"}))
        groups = _group_by_schema_locality([a, b, c])
        assert len(groups) == 1
        assert {it.conflict_ref for it in groups[0]} == {"a", "b", "c"}

    def test_fact_independent_items_never_merge_with_anything(self):
        a = _ConflictItem("a", "d", [], frozenset())
        b = _ConflictItem("b", "d", [], frozenset())
        c = _ConflictItem("c", "d", [1], frozenset({"ORDER"}))
        groups = _group_by_schema_locality([a, b, c])
        assert len(groups) == 3


@pytest.fixture
def facts_map():
    return {1: AtomicFact(id=1, fact="Order total must be at least 5.")}


def _empty_report(**kwargs):
    from src.pipeline.stage3.models.probe import Stage3AnalysisReport

    return Stage3AnalysisReport(), {}


class TestReconcileAndApplyRouting:
    @pytest.mark.asyncio
    async def test_false_positive_is_dismissed_and_removed(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.output = UnifiedExtractionOutput(
            structural_constraints=[_range_constraint([1])]
        )

        call_count = {"n": 0}

        def fake_evaluate(bridged, sch):
            from src.util.constraint_model.conflicts.models import ConflictReport

            call_count["n"] += 1
            if call_count["n"] == 1:
                return ConflictReport(
                    conflicts=[
                        Conflict(
                            kind="moment_value_mismatch",
                            summary="mismatch",
                            involved_fact_references=[1],
                            detail="d",
                            softenable=True,
                        )
                    ]
                )
            return ConflictReport()

        monkeypatch.setattr(
            stage3_entry, "analyze_cross_shard_constraints", _empty_report
        )
        monkeypatch.setattr(stage3_entry, "evaluate_constraints", fake_evaluate)
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict_group",
            AsyncMock(
                return_value=(
                    GroupReconciliation(
                        verdicts=[
                            ConflictReconciliation(
                                conflict_ref="moment_value_mismatch::1",
                                verdict=ReconciliationVerdict.FALSE_POSITIVE,
                                reasoning="Both facts agree; not a real conflict.",
                            )
                        ]
                    ),
                    0,
                )
            ),
        )

        conflicts, dismissed, unsupported, tokens = await _reconcile_and_apply(
            [ss], schema, facts_map, {1: 0}, None, 5, 5, 3
        )
        assert conflicts == []
        assert len(dismissed) == 1
        assert dismissed[0].conflict_ref == "moment_value_mismatch::1"

    @pytest.mark.asyncio
    async def test_genuine_contradiction_is_kept(self, monkeypatch, facts_map):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.output = UnifiedExtractionOutput(
            structural_constraints=[_range_constraint([1])]
        )

        conflict = Conflict(
            kind="moment_value_mismatch",
            summary="mismatch",
            involved_fact_references=[1],
            detail="d",
            softenable=True,
        )

        def fake_evaluate(bridged, sch):
            from src.util.constraint_model.conflicts.models import ConflictReport

            return ConflictReport(conflicts=[conflict])

        monkeypatch.setattr(
            stage3_entry, "analyze_cross_shard_constraints", _empty_report
        )
        monkeypatch.setattr(stage3_entry, "evaluate_constraints", fake_evaluate)
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict_group",
            AsyncMock(
                return_value=(
                    GroupReconciliation(
                        verdicts=[
                            ConflictReconciliation(
                                conflict_ref="moment_value_mismatch::1",
                                verdict=ReconciliationVerdict.GENUINE_CONTRADICTION,
                                reasoning="Two facts genuinely disagree.",
                            )
                        ]
                    ),
                    0,
                )
            ),
        )

        conflicts, dismissed, unsupported, tokens = await _reconcile_and_apply(
            [ss], schema, facts_map, {1: 0}, None, 5, 2, 3
        )
        assert len(conflicts) == 1
        assert dismissed == []

    @pytest.mark.asyncio
    async def test_misextraction_reruns_owning_shard_and_converges(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.output = UnifiedExtractionOutput(
            structural_constraints=[_range_constraint([1])]
        )

        conflict = Conflict(
            kind="moment_value_mismatch",
            summary="mismatch",
            involved_fact_references=[1],
            detail="d",
            softenable=True,
        )
        calls = {"n": 0}

        def fake_evaluate(bridged, sch):
            from src.util.constraint_model.conflicts.models import ConflictReport

            calls["n"] += 1
            if calls["n"] == 1:
                return ConflictReport(conflicts=[conflict])
            return ConflictReport()

        monkeypatch.setattr(
            stage3_entry, "analyze_cross_shard_constraints", _empty_report
        )
        monkeypatch.setattr(stage3_entry, "evaluate_constraints", fake_evaluate)
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict_group",
            AsyncMock(
                return_value=(
                    GroupReconciliation(
                        verdicts=[
                            ConflictReconciliation(
                                conflict_ref="moment_value_mismatch::1",
                                verdict=ReconciliationVerdict.MISEXTRACTION,
                                reasoning="Generator dropped a condition.",
                                fixes=[
                                    MisextractionFix(
                                        fact_id=1, guidance="Re-check fact 1."
                                    )
                                ],
                            )
                        ]
                    ),
                    0,
                )
            ),
        )

        rerun_calls = []

        async def fake_rerun_shard(
            schema_, fact_ids, facts_map_, stub_tables, guidance, model, max_retries
        ):
            rerun_calls.append(guidance)
            return UnifiedExtractionOutput(), 0

        monkeypatch.setattr(stage3_entry, "_rerun_shard", fake_rerun_shard)

        conflicts, dismissed, unsupported, tokens = await _reconcile_and_apply(
            [ss], schema, facts_map, {1: 0}, None, 5, 5, 3
        )
        assert len(rerun_calls) == 1
        assert "fact 1" in rerun_calls[0]
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_retry_budget_exhaustion_auto_dismisses_without_looping_forever(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.output = UnifiedExtractionOutput(
            structural_constraints=[_range_constraint([1])]
        )

        conflict = Conflict(
            kind="moment_value_mismatch",
            summary="stuck",
            involved_fact_references=[1],
            detail="d",
            softenable=True,
        )

        def fake_evaluate(bridged, sch):
            from src.util.constraint_model.conflicts.models import ConflictReport

            return ConflictReport(conflicts=[conflict])

        monkeypatch.setattr(
            stage3_entry, "analyze_cross_shard_constraints", _empty_report
        )
        monkeypatch.setattr(stage3_entry, "evaluate_constraints", fake_evaluate)
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict_group",
            AsyncMock(
                return_value=(
                    GroupReconciliation(
                        verdicts=[
                            ConflictReconciliation(
                                conflict_ref="moment_value_mismatch::1",
                                verdict=ReconciliationVerdict.MISEXTRACTION,
                                reasoning="stuck",
                                fixes=[MisextractionFix(fact_id=1, guidance="retry")],
                            )
                        ]
                    ),
                    0,
                )
            ),
        )

        async def fake_rerun_shard(
            schema_, fact_ids, facts_map_, stub_tables, guidance, model, max_retries
        ):
            return UnifiedExtractionOutput(
                structural_constraints=[_range_constraint([1])]
            ), 0

        monkeypatch.setattr(stage3_entry, "_rerun_shard", fake_rerun_shard)

        conflicts, dismissed, unsupported, tokens = await _reconcile_and_apply(
            [ss], schema, facts_map, {1: 0}, None, 5, 10, 2
        )
        # After max_constraint_retries=2 reconciler calls with no resolution,
        # the conflict is auto-dismissed rather than looping to max_rounds=10.
        assert conflicts == []
        assert len(dismissed) == 1
        assert "Retry budget exhausted" in dismissed[0].reason
