"""Regression tests for every confirmed bug in ISSUES.md (project root).

Each test reproduces the ORIGINAL failure mode directly against the real,
currently-fixed code -- not a happy-path test that merely exercises the
fix without proving the original bug is gone. See ISSUES.md for the full
narrative of how each was found and confirmed.
"""

from __future__ import annotations

import pytest

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.middleware.cycles import detect_derived_cycles
from src.pipeline.stage3.middleware.fact_allocation import find_mentioned_tables
from src.pipeline.stage3.middleware.fork_registry import (
    BranchCondition,
    ForkKey,
    ForkKeyRegistry,
    Operator,
    Unresolved,
)
from src.pipeline.stage3.models.condition_nodes import RArithmetic, RColumnRef, RLiteral
from src.pipeline.stage3.models.cross_shard import DerivedColumnConstraint
from src.pipeline.stage3.models.shard import SchemaShard
from src.util.algorithms.sharding_ilp import ILPSharder


# ---------------------------------------------------------------------------
# Item 1 -- detect_all_conflicts() crashes on any non-linear cyclic derived
# column (bare `None` return unconditionally 2-tuple-unpacked by its caller)
# ---------------------------------------------------------------------------


class TestNonLinearCycleCrash:
    def test_non_linear_cycle_reports_unverifiable_instead_of_crashing(self):
        """x depends on y*y (non-linear); y depends on x. Before the fix,
        _linear_coeff's multiplication branch returned a bare None when
        BOTH operands were non-constant, and its caller unconditionally
        unpacked the result as a 2-tuple -- an unhandled TypeError, not a
        graceful bail. Must not raise; must report as unverifiable."""
        dc_x = DerivedColumnConstraint(
            fact_references=[1],
            target_table="T",
            target_column="x",
            expression=RArithmetic(
                op="*", left=RColumnRef(name="y"), right=RColumnRef(name="y")
            ),
            referenced_tables=["T"],
        )
        dc_y = DerivedColumnConstraint(
            fact_references=[2],
            target_table="T",
            target_column="y",
            expression=RColumnRef(name="x"),
            referenced_tables=["T"],
        )
        issues = detect_derived_cycles([dc_x, dc_y])  # must not raise
        assert len(issues) == 1
        assert "non-linear" in issues[0].description.lower()
        # fact_references must trace back to BOTH originating facts -- this
        # is what lets the conflict-reconciliation agent find the right NL
        # text to re-examine.
        assert set(issues[0].fact_references) == {1, 2}


# ---------------------------------------------------------------------------
# Item 2 -- fork-key category loss: _discover_fork broke after the FIRST
# matching CategoricalDistribution, so a second shard's additional
# categories never reached the registry's union logic.
# ---------------------------------------------------------------------------


class TestForkKeyCategoryUnion:
    def test_categories_from_multiple_shards_all_survive(self):
        """Two independent CategoricalDistribution facts (as if extracted
        from two different shards) register Bronze/Silver and Gold/Platinum
        separately -- the registry must union both, not keep only the
        first-registered pair."""
        registry = ForkKeyRegistry()
        fork_key = ForkKey(table_name="CUSTOMER", column_name="loyalty_tier")

        registry.register_fork(fork_key, ["Bronze", "Silver"])
        registry.register_fork(fork_key, ["Gold", "Platinum"])

        assert set(registry.forks[fork_key]) == {
            "Bronze",
            "Silver",
            "Gold",
            "Platinum",
        }


# ---------------------------------------------------------------------------
# Item 3 -- SchemaShard._validate() crashed on any real schema (referenced
# Column.is_primary_key / ForeignKey.referencing_columns/referred_columns,
# none of which exist on the real models).
# ---------------------------------------------------------------------------


class TestSchemaShardValidateDoesNotCrash:
    def test_validate_runs_against_a_real_schema_without_attribute_error(self):
        schema = Schema(
            tables=[
                Table(
                    name="CUSTOMER",
                    primary_key=["customer_id"],
                    columns=[
                        Column(name="customer_id", data_type=DataType.INTEGER),
                        Column(name="name", data_type=DataType.VARCHAR),
                    ],
                ),
                Table(
                    name="ORDER",
                    primary_key=["order_id"],
                    columns=[
                        Column(name="order_id", data_type=DataType.INTEGER),
                        Column(name="customer_id", data_type=DataType.INTEGER),
                    ],
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
        shard = SchemaShard(
            shard_index=0,
            tables=["ORDER"],
            projections={"ORDER": ["order_id", "customer_id"]},
            allocated_fact_ids=[1],
        )
        # Must not raise AttributeError -- and correctly flags the real FK
        # closure violation (ORDER.customer_id present, CUSTOMER absent).
        errors = shard._validate(schema)
        assert any("FK Closure Violation" in e for e in errors)

    def test_validate_passes_when_fk_closure_and_pk_are_satisfied(self):
        schema = Schema(
            tables=[
                Table(
                    name="ORDER",
                    primary_key=["order_id"],
                    columns=[Column(name="order_id", data_type=DataType.INTEGER)],
                ),
            ],
        )
        shard = SchemaShard(
            shard_index=0,
            tables=["ORDER"],
            projections={"ORDER": ["order_id"]},
            allocated_fact_ids=[1],
        )
        assert shard._validate(schema) == []


# ---------------------------------------------------------------------------
# Item 4 -- the "skeleton shard" loophole: a fact with an empty or
# schema-mismatched column list left its containment variable unconstrained,
# letting the solver satisfy non-emptiness for free.
# ---------------------------------------------------------------------------


class TestSkeletonShardLoophole:
    def test_fact_with_no_matching_columns_cannot_satisfy_non_emptiness_for_free(self):
        """A fact whose column list matches nothing in the schema (e.g. a
        malformed extraction) must never let an otherwise-empty shard count
        as non-empty just to satisfy HC9 -- it should be excluded from
        valid_fact_ids entirely, not silently pinned to 1."""
        sharder = ILPSharder(max_shards=2, max_tables_per_shard=2)
        tables = ["ORDER"]
        columns_by_table = {"ORDER": ["order_id", "total"]}
        pks_by_table = {"ORDER": ["order_id"]}
        fks: list = []
        facts = {
            "f1": [("NONEXISTENT_TABLE", "bogus_column")],
            "f2": [("ORDER", "order_id")],
        }
        shards, shard_facts = sharder.shard_schema(
            tables, columns_by_table, pks_by_table, fks, facts
        )
        assert shards is not None
        assert shard_facts is not None
        # The bogus fact must never appear as contained in any shard --
        # only a real ORDER-table fact may.
        for facts_in_shard in shard_facts:
            assert "f1" not in facts_in_shard


# ---------------------------------------------------------------------------
# Item 5 -- NEQ mislabeling: get_branches_for_condition returned the
# EXCLUDED value as if it were the branch itself.
# ---------------------------------------------------------------------------


class TestNeqResolution:
    def test_neq_with_unknown_category_list_is_unresolved_not_guessed(self):
        """Before the fix, an NEQ condition against an unregistered fork key
        returned the excluded value(s) AS the branch (backwards) instead of
        admitting it doesn't know the full category list yet."""
        registry = ForkKeyRegistry()
        fork_key = ForkKey(table_name="CUSTOMER", column_name="tier")
        condition = BranchCondition(
            fork_key=fork_key, operator=Operator.NEQ, values=["Platinum"]
        )
        result = registry.get_branches_for_condition(condition)
        assert isinstance(result, Unresolved)

    def test_neq_resolves_correctly_once_categories_are_known(self):
        registry = ForkKeyRegistry()
        fork_key = ForkKey(table_name="CUSTOMER", column_name="tier")
        registry.register_fork(fork_key, ["Bronze", "Silver", "Gold", "Platinum"])
        condition = BranchCondition(
            fork_key=fork_key, operator=Operator.NEQ, values=["Platinum"]
        )
        result = registry.get_branches_for_condition(condition)
        assert set(result) == {"Bronze", "Silver", "Gold"}


# ---------------------------------------------------------------------------
# Item 6 -- table-mention detection was both too loose (substring matches:
# RATE inside "corporate") and too strict (naive +s pluralization missed
# CATEGORY -> categories).
# ---------------------------------------------------------------------------


class TestTableMentionDetection:
    def test_substring_false_positive_is_not_a_match(self):
        """'RATE' must not match inside 'corporate', 'AGE' must not match
        inside 'storage' -- the original bug had no word-boundary check."""
        mentioned = find_mentioned_tables(
            "The corporate storage policy applies here.", ["RATE", "AGE"]
        )
        assert mentioned == set()

    def test_irregular_plural_is_detected(self):
        """CATEGORY -> categories (consonant+y -> -ies) was previously
        undetectable -- the naive rule only ever appended a bare 's'."""
        mentioned = find_mentioned_tables(
            "Products are grouped into several categories.", ["CATEGORY"]
        )
        assert mentioned == {"CATEGORY"}

    def test_regular_plural_still_detected(self):
        mentioned = find_mentioned_tables(
            "Each order has several line items attached.", ["LINE_ITEM"]
        )
        assert mentioned == {"LINE_ITEM"}


# ---------------------------------------------------------------------------
# Item 7 -- cross-table derived-column cycles were invisible: the
# dependency graph always assumed a referenced column lived on the SAME
# table as the column being computed.
# ---------------------------------------------------------------------------


class TestCrossTableCycleVisibility:
    def test_two_table_cycle_is_detected(self):
        """ORDER.total depends on TAX.rate; TAX.rate depends on ORDER.total
        -- a genuine cross-table circular definition. Before the fix, the
        dependency graph resolution always assumed a referenced column
        lived on the formula's OWN table, making this invisible."""
        dc_order = DerivedColumnConstraint(
            fact_references=[1],
            target_table="ORDER",
            target_column="total",
            expression=RArithmetic(
                op="+", left=RColumnRef(name="rate"), right=RLiteral(value=1)
            ),
            referenced_tables=["TAX"],
        )
        dc_tax = DerivedColumnConstraint(
            fact_references=[2],
            target_table="TAX",
            target_column="rate",
            expression=RArithmetic(
                op="+", left=RColumnRef(name="total"), right=RLiteral(value=1)
            ),
            referenced_tables=["ORDER"],
        )
        issues = detect_derived_cycles([dc_order, dc_tax])
        assert len(issues) == 1
        assert "no solution" in issues[0].description
        assert set(issues[0].fact_references) == {1, 2}


# ---------------------------------------------------------------------------
# Item 8 -- an entire tested subsystem never actually ran in production
# (values_compatible/distribution_support/conditions_overlap in the old
# conflict_detection.py had zero live callers). Resolved by deletion --
# see STAGE3_HANDOFF_REVIEW.md -- confirmed here by asserting the module
# no longer exists, so it can never again silently drift from what's live.
# ---------------------------------------------------------------------------


class TestDeadSubsystemRemoved:
    def test_conflict_detection_module_no_longer_exists(self):
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module("src.pipeline.stage3.middleware.conflict_detection")
