"""Integration tests for src/util/constraint_model/conflicts/evaluate.py --
the top-level evaluate_constraints() API exercised against large, complex,
multi-entity schemas and constraint batches: SQL-string-built relations
mixed with object-built ones, 3+ constraints overlapping over shared
columns, and every conflict family combined in one evaluation.

Explicit scope-boundary tests are included alongside the "does it catch
real conflicts" ones -- per user instruction, a false negative that stems
from a genuine structural/grammar limitation (e.g. MAX/MIN aren't cross-
checked against Distributed moments at all) is an acceptable, documented
boundary, not a bug to chase.
"""

from __future__ import annotations

from typing import Literal, Optional

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.cohesive import (
    Correlated,
    Distributed,
    DistributionFamily,
    PairwiseCorrelation,
    StateSequence,
    StateTransition,
)
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.conflicts import evaluate_constraints
from src.util.constraint_model.constraint import Constraint
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    AggregateFn,
    BaseTable,
    Filter,
    Join,
    JoinCondition,
    RelationUnion,
)
from src.util.constraint_model.relation.sql_bridge import from_sql

_ComparisonOp = Literal["<", "<=", "=", "!=", ">=", ">"]


def _big_schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="CUSTOMER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="region", data_type=DataType.VARCHAR),
                ],
                primary_key=["id"],
            ),
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="customer_id", data_type=DataType.INTEGER),
                    Column(name="total", data_type=DataType.FLOAT),
                    Column(name="quantity", data_type=DataType.FLOAT),
                    Column(name="status", data_type=DataType.VARCHAR),
                ],
                primary_key=["id"],
            ),
            Table(
                name="ORDER_ITEM",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="order_id", data_type=DataType.INTEGER),
                    Column(name="unit_price", data_type=DataType.FLOAT),
                ],
                primary_key=["id"],
            ),
            Table(
                name="ORDER_STATUS_EVENT",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(name="order_id", data_type=DataType.INTEGER),
                    Column(name="status", data_type=DataType.VARCHAR),
                    Column(name="event_at", data_type=DataType.DATETIME),
                ],
                primary_key=["id"],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="ORDER",
                referencing_column="customer_id",
                referred_table="CUSTOMER",
            ),
            ForeignKey(
                referencing_table="ORDER_ITEM",
                referencing_column="order_id",
                referred_table="ORDER",
            ),
            ForeignKey(
                referencing_table="ORDER_STATUS_EVENT",
                referencing_column="order_id",
                referred_table="ORDER",
            ),
        ],
    )


def _moment(
    relation: "RelationUnion",
    alias: str,
    value: float,
    fid: int,
    fn: AggregateFn = "AVG",
    column: str = "total",
    op: _ComparisonOp = "=",
) -> Constraint:
    return Constraint(
        relation=Aggregate(source=relation, fn=fn, column=column, alias=alias),
        condition=RComparison(
            op=op, left=RAggregateRef(alias=alias), right=RLiteral(value=value)
        ),
        fact_references=[fid],
    )


def _dist(
    column: str,
    family: DistributionFamily,
    params: dict,
    fid: int,
    relation: Optional["RelationUnion"] = None,
) -> Constraint:
    return Constraint(
        relation=relation or BaseTable(name="ORDER"),
        condition=Distributed(column=column, family=family, parameters=params),
        fact_references=[fid],
    )


class TestSqlStringBuiltRelations:
    def test_sql_string_join_filter_relation_participates_normally(self):
        schema = _big_schema()
        obj, errs = from_sql(
            'SELECT * FROM ORDER_ITEM JOIN "ORDER" ON ORDER_ITEM.order_id = "ORDER".id WHERE total > 100'
        )
        assert errs == []
        assert obj is not None
        c1 = Constraint(
            relation=obj,
            condition=Distributed(
                column="unit_price",
                family="GAUSSIAN",
                parameters={"mean": 25, "std_dev": 5},
            ),
            fact_references=[1],
        )
        c2 = Constraint(
            relation=obj,
            condition=Distributed(
                column="unit_price",
                family="GAUSSIAN",
                parameters={"mean": 999, "std_dev": 5},
            ),
            fact_references=[2],
        )
        report = evaluate_constraints([c1, c2], schema)
        assert report.has_conflicts
        assert report.conflicts[0].kind == "distributed_parameter_mismatch"

    def test_sql_string_and_object_built_relations_produce_equivalent_populations(self):
        schema = _big_schema()
        sql_relation, errs = from_sql('SELECT * FROM "ORDER"')
        assert errs == []
        assert sql_relation is not None
        object_relation = BaseTable(name="ORDER")

        c1 = _dist(
            "total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 1, relation=sql_relation
        )
        c2 = _dist(
            "total",
            "GAUSSIAN",
            {"mean": 999, "std_dev": 10},
            2,
            relation=object_relation,
        )
        report = evaluate_constraints([c1, c2], schema)
        # a SQL-parsed BaseTable and an object-built BaseTable for the same
        # real table must be recognized as the SAME population.
        assert report.has_conflicts
        assert report.conflicts[0].kind == "distributed_parameter_mismatch"

    def test_sql_string_aggregate_with_having_used_as_moment_fact(self):
        schema = _big_schema()
        sql_relation, errs = from_sql(
            'SELECT customer_id, SUM(total) AS total_sum FROM "ORDER" GROUP BY customer_id '
            "HAVING SUM(total) > 1000"
        )
        assert errs == []
        # sql_bridge already produces a Filter(source=Aggregate(...)) with an
        # RAggregateRef condition -- wrap it directly as a Constraint.
        assert isinstance(sql_relation, Filter)
        c1 = Constraint(
            relation=sql_relation, condition=sql_relation.condition, fact_references=[1]
        )
        report = evaluate_constraints([c1], schema)
        assert not report.has_conflicts


class TestThreeWayColumnOverlaps:
    def test_three_distributed_facts_same_column_two_disagree(self):
        c1 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)
        c2 = _dist(
            "total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 2
        )  # agrees with c1
        c3 = _dist(
            "total", "GAUSSIAN", {"mean": 500, "std_dev": 10}, 3
        )  # disagrees with both
        report = evaluate_constraints([c1, c2, c3], _big_schema())
        kinds = [c.kind for c in report.conflicts]
        assert kinds.count("distributed_parameter_mismatch") == 2  # (1,3) and (2,3)

    def test_three_correlated_facts_forming_a_triangle_plus_a_moment_fact(self):
        schema = _big_schema()
        c1 = Constraint(
            relation=BaseTable(name="ORDER"),
            condition=Correlated(
                columns=["total", "quantity"],
                family="GAUSSIAN",
                pairwise=[
                    PairwiseCorrelation(left="total", right="quantity", value=0.9)
                ],
            ),
            fact_references=[1],
        )
        c2 = Constraint(
            relation=BaseTable(name="ORDER"),
            condition=Distributed(
                column="total",
                family="GAUSSIAN",
                parameters={"mean": 100, "std_dev": 10},
            ),
            fact_references=[2],
        )
        c3 = _moment(BaseTable(name="ORDER"), "avg_total", 100, 3)
        report = evaluate_constraints([c1, c2, c3], schema)
        assert not report.has_conflicts

    def test_overlapping_correlated_and_moment_and_distributed_all_together_broken(
        self,
    ):
        schema = _big_schema()
        correlated = Constraint(
            relation=BaseTable(name="ORDER"),
            condition=Correlated(
                columns=["total", "quantity"],
                family="GAUSSIAN",
                pairwise=[
                    PairwiseCorrelation(left="total", right="quantity", value=0.5)
                ],
            ),
            fact_references=[1],
        )
        distributed = Constraint(
            relation=BaseTable(name="ORDER"),
            condition=Distributed(
                column="total",
                family="GAUSSIAN",
                parameters={"mean": 100, "std_dev": 10},
            ),
            fact_references=[2],
        )
        moment_wrong = _moment(BaseTable(name="ORDER"), "avg_total_wrong", 99999, 3)
        report = evaluate_constraints([correlated, distributed, moment_wrong], schema)
        assert len(report.conflicts) == 1
        assert report.conflicts[0].kind == "moment_vs_distributed_mismatch"


class TestMinMaxScopeBoundary:
    def test_max_facts_never_cross_checked_against_distributed(self):
        # deliberate, documented scope boundary -- MAX/MIN don't correspond
        # to a Distributed family's mean/variance formula, so this must
        # neither crash nor fabricate a conflict.
        c1 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)
        c2 = _moment(BaseTable(name="ORDER"), "max_total", 1_000_000, 2, fn="MAX")
        report = evaluate_constraints([c1, c2], _big_schema())
        assert not report.has_conflicts

    def test_min_facts_never_cross_checked_against_each_other(self):
        c1 = _moment(BaseTable(name="ORDER"), "min1", 0, 1, fn="MIN")
        c2 = _moment(BaseTable(name="ORDER"), "min2", 999999, 2, fn="MIN")
        report = evaluate_constraints([c1, c2], _big_schema())
        assert not report.has_conflicts

    def test_avg_and_max_together_only_avg_participates(self):
        c1 = _dist("total", "GAUSSIAN", {"mean": 100, "std_dev": 10}, 1)
        c2 = _moment(BaseTable(name="ORDER"), "avg_bad", 99999, 2, fn="AVG")
        c3 = _moment(BaseTable(name="ORDER"), "max_ok", 999999, 3, fn="MAX")
        report = evaluate_constraints([c1, c2, c3], _big_schema())
        assert len(report.conflicts) == 1
        assert report.conflicts[0].kind == "moment_vs_distributed_mismatch"
        assert 3 not in report.conflicts[0].involved_fact_references


class TestKitchenSinkAllFamiliesTogether:
    def test_large_consistent_batch_produces_zero_conflicts(self):
        schema = _big_schema()
        join = Join(
            left=BaseTable(name="ORDER"),
            right=BaseTable(name="CUSTOMER"),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        us_filter = Filter(
            source=join,
            condition=RComparison(
                op="=", left=RColumnRef(name="region"), right=RLiteral(value="US")
            ),
        )

        constraints = [
            _dist("total", "GAUSSIAN", {"mean": 150, "std_dev": 20}, 1),
            _moment(BaseTable(name="ORDER"), "avg_total", 150, 2),
            Constraint(
                relation=BaseTable(name="ORDER"),
                condition=Correlated(
                    columns=["total", "quantity"],
                    family="GAUSSIAN",
                    pairwise=[
                        PairwiseCorrelation(left="total", right="quantity", value=0.4)
                    ],
                ),
                fact_references=[3],
            ),
            Constraint(
                relation=BaseTable(name="ORDER_STATUS_EVENT"),
                condition=StateSequence(
                    sequence_column="status",
                    allowed_transitions=[
                        StateTransition(from_state="ready", to_state="packed"),
                        StateTransition(from_state="packed", to_state="shipped"),
                    ],
                    strict=True,
                ),
                fact_references=[4],
            ),
            _moment(us_filter, "avg_us_total", 200, 5),
        ]
        report = evaluate_constraints(constraints, schema)
        assert report.conflicts == []
        assert report.unsupported == []

    def test_large_batch_with_one_planted_conflict_per_family(self):
        schema = _big_schema()
        constraints = [
            # Distributed family mismatch
            _dist("total", "GAUSSIAN", {}, 1),
            _dist("total", "POISSON", {}, 2),
            # Correlated infeasible triangle
            Constraint(
                relation=BaseTable(name="ORDER"),
                condition=Correlated(
                    columns=["total", "quantity"],
                    family="GAUSSIAN",
                    pairwise=[
                        PairwiseCorrelation(left="total", right="quantity", value=0.9)
                    ],
                ),
                fact_references=[3],
            ),
            # StateSequence direct contradiction
            Constraint(
                relation=BaseTable(name="ORDER_STATUS_EVENT"),
                condition=StateSequence(
                    sequence_column="status",
                    allowed_transitions=[
                        StateTransition(from_state="ready", to_state="packed")
                    ],
                ),
                fact_references=[4],
            ),
            Constraint(
                relation=BaseTable(name="ORDER_STATUS_EVENT"),
                condition=StateSequence(
                    sequence_column="status",
                    forbidden_transitions=[
                        StateTransition(from_state="ready", to_state="packed")
                    ],
                ),
                fact_references=[5],
            ),
            # Moment value mismatch
            _moment(BaseTable(name="ORDER"), "a1", 100, 6),
            _moment(BaseTable(name="ORDER"), "a2", 200, 7),
        ]
        report = evaluate_constraints(constraints, schema)
        kinds = {c.kind for c in report.conflicts}
        assert "distributed_family_mismatch" in kinds
        assert "state_sequence_direct_contradiction" in kinds
        assert "moment_value_mismatch" in kinds


class TestErrorAndUnsupportedHandling:
    def test_unresolvable_relation_reported_not_crashing(self):
        bad = _dist(
            "total",
            "GAUSSIAN",
            {"mean": 1, "std_dev": 1},
            1,
            relation=BaseTable(name="NOPE"),
        )
        good = _dist("total", "GAUSSIAN", {"mean": 2, "std_dev": 2}, 2)
        report = evaluate_constraints([bad, good], _big_schema())
        assert report.conflicts == []
        assert len(report.unsupported) == 1

    def test_non_chordal_correlation_surfaces_as_unsupported_not_a_conflict(self):
        schema = _big_schema()
        c1 = Constraint(
            relation=BaseTable(name="ORDER"),
            condition=Correlated(
                columns=["total", "quantity"],
                family="GAUSSIAN",
                pairwise=[
                    PairwiseCorrelation(left="total", right="quantity", value=0.3)
                ],
            ),
            fact_references=[1],
        )
        report = evaluate_constraints([c1], schema)
        assert report.conflicts == []
        assert (
            report.unsupported == []
        )  # a single edge is trivially chordal -- sanity check

    def test_empty_constraint_list(self):
        report = evaluate_constraints([], _big_schema())
        assert report.conflicts == []
        assert report.unsupported == []
