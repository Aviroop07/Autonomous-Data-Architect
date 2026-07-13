"""Tests for the deterministic control flow in the Stage 3 3-tier
reconciliation orchestration (src/orchestration/stage3/entry.py).

These target the pure/deterministic pieces (conflict extraction, dismissal
filtering, merging) directly, plus the reconciliation-round loop's
verdict-routing logic (MISEXTRACTION -> re-extract, FALSE_POSITIVE -> drop
+ record, GENUINE_CONTRADICTION -> keep) with the LLM-calling boundaries
(reconcile_conflict, _run_family_loop) monkeypatched -- no live LLM calls.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock

import pytest

from src.orchestration.stage3 import entry as stage3_entry
from src.orchestration.stage3.entry import (
    _ShardState,
    _extract_conflict_items,
    _filter_dismissed,
    _merge_all,
    _reconciliation_loop,
    _render_involved_constraints,
)
from src.pipeline.stage1.models.rephrased_nl import AtomicFact
from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, Schema, Table
from src.pipeline.stage3.models.condition_nodes import RComparison, RLiteral, RColumnRef
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    LogicExtractionOutput,
    StatisticalExtractionOutput,
    StructuralExtractionOutput,
)
from src.pipeline.stage3.models.on_nodes import ONBaseTable
from src.pipeline.stage3.models.probe import (
    CycleIssue,
    MisextractionFix,
    ConflictReconciliation,
    ReconciliationVerdict,
    Stage3AnalysisReport,
)
from src.util.algorithms.dof_graph import OverconstrainedBlock


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


def _range_constraint(fact_ids: List[int]) -> Constraint:
    return Constraint(
        fact_references=fact_ids,
        on=ONBaseTable(name="ORDER"),
        condition=RComparison(
            op=">=", left=RColumnRef(name="total"), right=RLiteral(value=5)
        ),
        category="structural",
    )


# ---------------------------------------------------------------------------
# _extract_conflict_items
# ---------------------------------------------------------------------------


class TestExtractConflictItems:
    def test_overconstrained_block_traces_fact_ids_via_variable_fact_map(self):
        report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(
                    variables=["ORDER.total.mean"], constraints=["pin_1", "pin_2"]
                )
            ]
        )
        variable_fact_map = {"ORDER.total.mean": [1, 2]}
        items = _extract_conflict_items(report, variable_fact_map)
        assert len(items) == 1
        assert items[0].conflict_ref == "ORDER.total.mean"
        assert items[0].fact_ids == [1, 2]

    def test_cycle_conflict_uses_its_own_fact_references_not_the_variable_map(self):
        report = Stage3AnalysisReport(
            derived_cycle_conflicts=[
                CycleIssue(
                    description="x = x + 5", nodes=("T.x",), fact_references=(9,)
                )
            ]
        )
        items = _extract_conflict_items(report, {})
        assert len(items) == 1
        assert items[0].conflict_ref == "cycle::x = x + 5"
        assert items[0].fact_ids == [9]

    def test_empty_report_yields_no_items(self):
        assert _extract_conflict_items(Stage3AnalysisReport(), {}) == []


# ---------------------------------------------------------------------------
# _filter_dismissed
# ---------------------------------------------------------------------------


class TestFilterDismissed:
    def test_dismissed_block_removed_from_overconstrained_blocks(self):
        report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(variables=["A"], constraints=[]),
                OverconstrainedBlock(variables=["B"], constraints=[]),
            ]
        )
        filtered = _filter_dismissed(report, {"A"})
        assert [b.variables for b in filtered.overconstrained_blocks] == [["B"]]

    def test_dismissed_cycle_removed_from_derived_cycle_conflicts(self):
        report = Stage3AnalysisReport(
            derived_cycle_conflicts=[
                CycleIssue(description="cyc1", nodes=(), fact_references=()),
                CycleIssue(description="cyc2", nodes=(), fact_references=()),
            ]
        )
        filtered = _filter_dismissed(report, {"cycle::cyc1"})
        assert [c.description for c in filtered.derived_cycle_conflicts] == ["cyc2"]

    def test_nothing_dismissed_is_a_no_op(self):
        report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(variables=["A"], constraints=[])
            ]
        )
        filtered = _filter_dismissed(report, set())
        assert len(filtered.overconstrained_blocks) == 1


# ---------------------------------------------------------------------------
# _merge_all
# ---------------------------------------------------------------------------


class TestMergeAll:
    def test_merges_every_family_across_every_shard(self):
        ss0 = _ShardState(index=0, schema=_schema(), fact_ids=[1], stub_tables=[])
        ss0.outputs = {
            "statistical": StatisticalExtractionOutput(
                distributions=[],
                moment_targets=[_range_constraint([1])],
                correlations=[],
            ),
            "structural": StructuralExtractionOutput(
                constraints=[_range_constraint([1])]
            ),
            "logic": LogicExtractionOutput(constraints=[], derived=[]),
        }
        ss1 = _ShardState(index=1, schema=_schema(), fact_ids=[2], stub_tables=[])
        ss1.outputs = {
            "statistical": StatisticalExtractionOutput(),
            "structural": StructuralExtractionOutput(),
            "logic": LogicExtractionOutput(
                constraints=[_range_constraint([2])], derived=[]
            ),
        }
        distributions, moment_targets, correlations, structural, logic, derived = (
            _merge_all([ss0, ss1])
        )
        assert len(moment_targets) == 1
        assert len(structural) == 1
        assert len(logic) == 1
        assert distributions == []
        assert correlations == []
        assert derived == []


# ---------------------------------------------------------------------------
# _render_involved_constraints
# ---------------------------------------------------------------------------


class TestRenderInvolvedConstraints:
    def test_only_matching_fact_ids_are_rendered(self):
        c1 = _range_constraint([1])
        c2 = _range_constraint([2])
        text = _render_involved_constraints([1], [], [], [], [c1, c2], [], [])
        assert "[Structural]" in text
        assert text.count("[Structural]") == 1

    def test_no_matches_returns_placeholder_not_empty_string(self):
        text = _render_involved_constraints(
            [999], [], [], [], [_range_constraint([1])], [], []
        )
        assert text == "(no extracted constraints reference these facts)"


# ---------------------------------------------------------------------------
# _reconciliation_loop -- verdict routing, with reconcile_conflict and
# _run_family_loop monkeypatched (no live LLM calls).
# ---------------------------------------------------------------------------


@pytest.fixture
def facts_map():
    return {1: AtomicFact(id=1, fact="Order total must be at least 5.")}


class TestReconciliationLoopRouting:
    @pytest.mark.asyncio
    async def test_false_positive_is_dismissed_and_removed_from_report(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.outputs = {
            "statistical": StatisticalExtractionOutput(),
            "structural": StructuralExtractionOutput(
                constraints=[_range_constraint([1])]
            ),
            "logic": LogicExtractionOutput(),
        }

        fake_report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(variables=["ORDER.total"], constraints=["c1"])
            ]
        )
        call_count = {"n": 0}

        def fake_analyze(**kwargs):
            call_count["n"] += 1
            # First call finds the conflict; subsequent calls (post-dismissal
            # re-check) find nothing, since nothing was re-extracted.
            if call_count["n"] == 1:
                return fake_report, {"ORDER.total": [1]}
            return Stage3AnalysisReport(), {}

        monkeypatch.setattr(
            stage3_entry, "analyze_cross_shard_constraints", fake_analyze
        )
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict",
            AsyncMock(
                return_value=(
                    ConflictReconciliation(
                        conflict_ref="ORDER.total",
                        verdict=ReconciliationVerdict.FALSE_POSITIVE,
                        reasoning="Both facts agree; not a real conflict.",
                    ),
                    0,
                )
            ),
        )

        report, tokens = await _reconciliation_loop(
            [ss], schema, facts_map, {1: 0}, None, 5, 5, label="test"
        )

        assert report.overconstrained_blocks == []
        assert len(report.dismissed_conflicts) == 1
        assert report.dismissed_conflicts[0].conflict_ref == "ORDER.total"

    @pytest.mark.asyncio
    async def test_genuine_contradiction_is_kept_in_final_report(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.outputs = {
            "statistical": StatisticalExtractionOutput(),
            "structural": StructuralExtractionOutput(
                constraints=[_range_constraint([1])]
            ),
            "logic": LogicExtractionOutput(),
        }

        fake_report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(variables=["ORDER.total"], constraints=["c1"])
            ]
        )
        monkeypatch.setattr(
            stage3_entry,
            "analyze_cross_shard_constraints",
            lambda **kwargs: (fake_report, {"ORDER.total": [1]}),
        )
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict",
            AsyncMock(
                return_value=(
                    ConflictReconciliation(
                        conflict_ref="ORDER.total",
                        verdict=ReconciliationVerdict.GENUINE_CONTRADICTION,
                        reasoning="Two facts genuinely disagree.",
                    ),
                    0,
                )
            ),
        )

        report, tokens = await _reconciliation_loop(
            [ss], schema, facts_map, {1: 0}, None, 5, 5, label="test"
        )

        assert len(report.overconstrained_blocks) == 1
        assert report.dismissed_conflicts == []

    @pytest.mark.asyncio
    async def test_misextraction_triggers_reextraction_and_converges(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.outputs = {
            "statistical": StatisticalExtractionOutput(),
            "structural": StructuralExtractionOutput(
                constraints=[_range_constraint([1])]
            ),
            "logic": LogicExtractionOutput(),
        }

        conflicting_report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(variables=["ORDER.total"], constraints=["c1"])
            ]
        )
        clean_report = Stage3AnalysisReport()
        analyze_calls = {"n": 0}

        def fake_analyze(**kwargs):
            analyze_calls["n"] += 1
            # Round 1: conflict found. After the misextraction fix is
            # "applied" (re-extraction stub below), every subsequent
            # analysis call finds nothing.
            return conflicting_report if analyze_calls["n"] == 1 else clean_report

        monkeypatch.setattr(
            stage3_entry,
            "analyze_cross_shard_constraints",
            lambda **kwargs: (fake_analyze(**kwargs), {"ORDER.total": [1]}),
        )
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict",
            AsyncMock(
                return_value=(
                    ConflictReconciliation(
                        conflict_ref="ORDER.total",
                        verdict=ReconciliationVerdict.MISEXTRACTION,
                        reasoning="Structural extractor dropped a condition.",
                        fixes=[
                            MisextractionFix(
                                family="structural",
                                fact_id=1,
                                guidance="Re-check the condition on fact 1.",
                            )
                        ],
                    ),
                    0,
                )
            ),
        )

        rerun_calls = []

        async def fake_rerun_single_family(
            schema_,
            fact_ids,
            facts_map_,
            stub_tables,
            family,
            guidance,
            model,
            max_retries,
        ):
            rerun_calls.append((family, guidance))
            return StructuralExtractionOutput(constraints=[]), 0

        monkeypatch.setattr(
            stage3_entry, "_rerun_single_family", fake_rerun_single_family
        )

        report, tokens = await _reconciliation_loop(
            [ss], schema, facts_map, {1: 0}, None, 5, 5, label="test"
        )

        assert len(rerun_calls) == 1
        assert rerun_calls[0][0] == "structural"
        assert "fact 1" in rerun_calls[0][1]
        assert report.overconstrained_blocks == []
        # The re-extraction stub replaced the shard's structural output.
        assert ss.outputs["structural"].constraints == []

    @pytest.mark.asyncio
    async def test_hits_max_rounds_without_converging_and_does_not_raise(
        self, monkeypatch, facts_map
    ):
        schema = _schema()
        ss = _ShardState(index=0, schema=schema, fact_ids=[1], stub_tables=[])
        ss.outputs = {
            "statistical": StatisticalExtractionOutput(),
            "structural": StructuralExtractionOutput(
                constraints=[_range_constraint([1])]
            ),
            "logic": LogicExtractionOutput(),
        }

        # Every round finds the same conflict and the same misextraction
        # verdict -- this can never converge, exercising the round cap.
        stuck_report = Stage3AnalysisReport(
            overconstrained_blocks=[
                OverconstrainedBlock(variables=["ORDER.total"], constraints=["c1"])
            ]
        )
        monkeypatch.setattr(
            stage3_entry,
            "analyze_cross_shard_constraints",
            lambda **kwargs: (stuck_report, {"ORDER.total": [1]}),
        )
        monkeypatch.setattr(
            stage3_entry,
            "reconcile_conflict",
            AsyncMock(
                return_value=(
                    ConflictReconciliation(
                        conflict_ref="ORDER.total",
                        verdict=ReconciliationVerdict.MISEXTRACTION,
                        reasoning="stuck",
                        fixes=[
                            MisextractionFix(
                                family="structural", fact_id=1, guidance="retry"
                            )
                        ],
                    ),
                    0,
                )
            ),
        )

        async def fake_rerun_single_family(
            schema_,
            fact_ids,
            facts_map_,
            stub_tables,
            family,
            guidance,
            model,
            max_retries,
        ):
            return StructuralExtractionOutput(constraints=[_range_constraint([1])]), 0

        monkeypatch.setattr(
            stage3_entry, "_rerun_single_family", fake_rerun_single_family
        )

        report, tokens = await _reconciliation_loop(
            [ss], schema, facts_map, {1: 0}, None, 5, 3, label="test"
        )

        # Never converged -- the conflict is still there, but it must not
        # raise or hang past max_rounds=3.
        assert len(report.overconstrained_blocks) == 1
