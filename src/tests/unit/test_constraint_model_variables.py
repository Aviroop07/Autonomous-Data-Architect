"""Tests for src/util/constraint_model/variables.py."""

from __future__ import annotations

import pytest

from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.expressions import RColumnRef, RLiteral
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Filter,
    Join,
    JoinCondition,
)
from src.util.constraint_model.relation.schema import RowCountVar
from src.util.constraint_model.variables import (
    build_dof_graph,
    collect_row_count_vars,
    latent_threshold_variables,
    row_count_constraint,
    row_count_variable,
    selectivity_variable,
)


def _schema() -> Schema:
    return Schema(
        tables=[
            Table(
                name="CUSTOMER",
                columns=[Column(name="id", data_type=DataType.INTEGER)],
                primary_key=["id"],
            ),
            Table(
                name="ORDER",
                columns=[
                    Column(name="id", data_type=DataType.INTEGER),
                    Column(
                        name="customer_id", data_type=DataType.INTEGER, is_nullable=True
                    ),
                    Column(name="total", data_type=DataType.FLOAT),
                ],
                primary_key=["id"],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="ORDER",
                referencing_column="customer_id",
                referred_table="CUSTOMER",
            )
        ],
    )


def _order() -> BaseTable:
    return BaseTable(name="ORDER")


def _customer() -> BaseTable:
    return BaseTable(name="CUSTOMER")


class TestRowCountVariable:
    def test_free_kind_has_no_lower_bound(self):
        rc = RowCountVar(name="ORDER.row_count", kind="free")
        var = row_count_variable(rc)
        assert var.name == "ORDER.row_count"
        assert var.lower_bound is None

    def test_grouped_kind_has_lower_bound_one(self):
        rc = RowCountVar(name="agg.row_count", kind="grouped", source="ORDER.row_count")
        var = row_count_variable(rc)
        assert var.lower_bound == 1.0

    def test_fact_references_pass_through(self):
        rc = RowCountVar(name="ORDER.row_count", kind="free")
        var = row_count_variable(rc, fact_references=[7])
        assert var.fact_references == [7]


class TestSelectivityVariable:
    def test_bounded_zero_to_one(self):
        rc = RowCountVar(
            name="f.row_count",
            kind="filtered",
            source="ORDER.row_count",
            selectivity="f.selectivity",
        )
        var = selectivity_variable(rc)
        assert var.name == "f.selectivity"
        assert var.lower_bound == 0.0
        assert var.upper_bound == 1.0

    def test_non_filtered_kind_rejected(self):
        rc = RowCountVar(name="ORDER.row_count", kind="free")
        with pytest.raises(ValueError):
            selectivity_variable(rc)


class TestRowCountConstraint:
    def test_free_kind_mints_no_constraint(self):
        rc = RowCountVar(name="ORDER.row_count", kind="free")
        assert row_count_constraint(rc) is None

    def test_grouped_kind_mints_no_constraint(self):
        rc = RowCountVar(name="agg.row_count", kind="grouped", source="ORDER.row_count")
        assert row_count_constraint(rc) is None

    def test_identity_kind_ties_to_equals(self):
        rc = RowCountVar(name="j.row_count", kind="identity", equals="ORDER.row_count")
        con = row_count_constraint(rc)
        assert con is not None
        assert set(con.variables) == {"j.row_count", "ORDER.row_count"}
        assert con.fact_references == []

    def test_filtered_kind_ties_to_source_and_selectivity(self):
        rc = RowCountVar(
            name="f.row_count",
            kind="filtered",
            source="j.row_count",
            selectivity="f.selectivity",
        )
        con = row_count_constraint(rc)
        assert con is not None
        assert set(con.variables) == {"f.row_count", "j.row_count", "f.selectivity"}


class TestLatentThresholdVariables:
    def test_three_categories_yields_two_thresholds(self):
        variables = latent_threshold_variables("status", 3)
        assert [v.name for v in variables] == [
            "status.threshold[0]",
            "status.threshold[1]",
        ]

    def test_fewer_than_two_categories_rejected(self):
        with pytest.raises(ValueError):
            latent_threshold_variables("status", 1)


class TestCollectRowCountVars:
    def test_base_table_yields_one_row_count(self):
        row_counts, errors = collect_row_count_vars(_order(), _schema())
        assert errors == []
        assert len(row_counts) == 1
        assert row_counts[0].name == "ORDER.row_count"

    def test_join_and_filter_names_are_mutually_consistent(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        f = Filter(
            source=j,
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        row_counts, errors = collect_row_count_vars(f, _schema())
        assert errors == []
        names = {rc.name for rc in row_counts}
        # the Filter's own row_count.source and the Join's own row_count.name
        # must be the SAME string -- this is exactly the bug the shared-namer
        # fix in relation/schema.py's synthesize_schema_tree() addresses.
        filter_rc = next(rc for rc in row_counts if rc.kind == "filtered")
        join_rc = next(rc for rc in row_counts if rc.kind == "identity")
        assert filter_rc.source == join_rc.name
        assert join_rc.name in names

    def test_propagates_synthesis_errors(self):
        row_counts, errors = collect_row_count_vars(BaseTable(name="NOPE"), _schema())
        assert row_counts == []
        assert len(errors) == 1

    def test_fanout_also_yields_its_parent_tables_own_row_count(self):
        """Regression: Fanout's own RowCountVar is kind='identity' with
        equals=f"{parent_table}.row_count" -- but parent_table is a plain
        string, not a child RelationUnion node, so it was never visited by
        _synth_collecting()'s recursion, leaving that identity's target
        variable never minted (build_dof_graph then rejected the fanout's
        constraint as referencing an undefined variable)."""
        fan = Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        )
        row_counts, errors = collect_row_count_vars(fan, _schema())
        assert errors == []
        names = {rc.name for rc in row_counts}
        assert "CUSTOMER.row_count" in names
        fanout_rc = next(rc for rc in row_counts if rc.kind == "identity")
        assert fanout_rc.equals == "CUSTOMER.row_count"
        # build_dof_graph must not raise "references undefined variable(s)".
        graph = build_dof_graph(row_counts)
        assert graph.classify() is not None


class TestBuildDofGraph:
    def test_join_chain_builds_a_valid_graph_with_no_dangling_references(self):
        j = Join(
            left=_order(),
            right=_customer(),
            on=[JoinCondition(left="ORDER.customer_id", right="CUSTOMER.id")],
        )
        f = Filter(
            source=j,
            condition=RComparison(
                op=">", left=RColumnRef(name="total"), right=RLiteral(value=100)
            ),
        )
        row_counts, errors = collect_row_count_vars(f, _schema())
        assert errors == []
        graph = build_dof_graph(row_counts)
        result = graph.classify()
        # nothing pinned anywhere in this bridge alone -- every row-count/
        # selectivity variable is a genuine free/loose DOF target.
        assert result.square_variables == []
        expected_loose = {rc.name for rc in row_counts} | {
            rc.selectivity for rc in row_counts if rc.selectivity
        }
        assert set(result.loose_variables) == expected_loose
        assert result.overconstrained_blocks == []

    def test_aggregate_grouped_variable_has_lower_bound(self):
        agg = Aggregate(
            source=_order(), fn="COUNT", column="*", group_by=["customer_id"], alias="n"
        )
        row_counts, errors = collect_row_count_vars(agg, _schema())
        assert errors == []
        graph = build_dof_graph(row_counts)
        grouped_var = next(
            v
            for v in graph.variables
            if v.name.endswith("n.row_count") or v.name == "n.row_count"
        )
        assert grouped_var.lower_bound == 1.0

    def test_extra_variables_and_constraints_are_included(self):
        from src.util.algorithms.dof_graph import Variable as DOFVariable

        row_counts, _ = collect_row_count_vars(_order(), _schema())
        extra = DOFVariable(name="mu_total")
        graph = build_dof_graph(row_counts, extra_variables=[extra])
        assert "mu_total" in {v.name for v in graph.variables}
