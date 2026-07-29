"""Canned Stage 3 agent outputs.

Every constraint below names columns that exist in the schema Stage 2's REAL
mapper derives from `canned_payloads.stage2.conceptual_model()` -- CUSTOMER
(customer_id, name, credit_score, annual_income) and ORDER (order_id,
total_amount, status, customer_id), with ORDER.customer_id -> CUSTOMER the only
foreign key. None of that is asserted here; it is what the offline tests check.

All seven of `UnifiedExtractionOutput`'s constraint families are populated on
purpose: the column-resolution invariant is only as strong as the number of
distinct column-bearing shapes it actually visits, and each family carries its
columns in a different field.
"""

from __future__ import annotations

from src.pipeline.stage3.agents.extraction_outputs import AuditReport
from src.pipeline.stage3.models.cross_shard import (
    Constraint,
    CorrelatedConstraint,
    DerivedColumnConstraint,
    DistributionConstraint,
    PairwiseCorrelationSpec,
    StateSequenceConstraint,
    StateTransitionSpec,
    UnifiedExtractionOutput,
)
from src.pipeline.stage3.models.probe import GroupReconciliation
from src.util.constraint_model.condition.expressions import (
    RAggregateRef,
    RArithmetic,
    RColumnRef,
    RLiteral,
)
from src.util.constraint_model.condition.predicates import RComparison
from src.util.constraint_model.relation.nodes import Aggregate, BaseTable, Fanout


# ---------------------------------------------------------------------------
# Individual constraints, one factory each so a shard can take a subset
# ---------------------------------------------------------------------------


def credit_score_distribution() -> DistributionConstraint:
    return DistributionConstraint(
        fact_references=[3],
        on=BaseTable(name="CUSTOMER"),
        column="credit_score",
        family="GAUSSIAN",
        parameters={"mean": 700.0, "std_dev": 50.0},
    )


def income_credit_correlation() -> CorrelatedConstraint:
    return CorrelatedConstraint(
        fact_references=[3, 4],
        on=BaseTable(name="CUSTOMER"),
        columns=["credit_score", "annual_income"],
        family="GAUSSIAN",
        pairwise=[
            PairwiseCorrelationSpec(
                left="credit_score", right="annual_income", value=0.6
            )
        ],
    )


def average_order_total_moment() -> Constraint:
    """A moment target over an Aggregate ON tree -- the only shape whose
    condition references an aggregate ALIAS rather than a base column."""
    return Constraint(
        fact_references=[5],
        on=Aggregate(
            source=BaseTable(name="ORDER"),
            fn="AVG",
            column="total_amount",
            alias="avg_total_amount",
        ),
        condition=RComparison(
            op="=",
            left=RAggregateRef(alias="avg_total_amount"),
            right=RLiteral(value=250.0),
        ),
        category="statistical",
    )


def every_customer_has_an_order_fanout() -> Constraint:
    """A Fanout ON tree, whose `child_count` is a SYNTHETIC column: it exists on
    the grain but on neither table, so it exercises the one case where naive
    column resolution against the schema would wrongly fail.

    The bound is `>= 1`, not `>= 0`: a count is never negative, so `>= 0` is
    vacuous and the real deterministic checker rejects it.
    """
    return Constraint(
        fact_references=[1],
        on=Fanout(
            parent_table="CUSTOMER", child_table="ORDER", fk_column="customer_id"
        ),
        condition=RComparison(
            op=">=", left=RColumnRef(name="child_count"), right=RLiteral(value=1)
        ),
        category="structural",
    )


def positive_order_total_logic() -> Constraint:
    return Constraint(
        fact_references=[5],
        on=BaseTable(name="ORDER"),
        condition=RComparison(
            op=">", left=RColumnRef(name="total_amount"), right=RLiteral(value=0.0)
        ),
        category="logic",
    )


def order_status_lifecycle() -> StateSequenceConstraint:
    return StateSequenceConstraint(
        fact_references=[6],
        on=BaseTable(name="ORDER"),
        sequence_column="status",
        allowed_transitions=[
            StateTransitionSpec(from_state="pending", to_state="shipped"),
            StateTransitionSpec(from_state="shipped", to_state="delivered"),
        ],
        strict=True,
    )


def total_with_tax_derived() -> DerivedColumnConstraint:
    return DerivedColumnConstraint(
        fact_references=[5],
        target_table="ORDER",
        target_column="total_with_tax",
        expression=RArithmetic(
            op="*", left=RColumnRef(name="total_amount"), right=RLiteral(value=1.08)
        ),
        referenced_tables=["ORDER"],
    )


# ---------------------------------------------------------------------------
# Whole-shard outputs
# ---------------------------------------------------------------------------


def full_extraction() -> UnifiedExtractionOutput:
    """Everything, for a single shard holding the whole schema."""
    return UnifiedExtractionOutput(
        distributions=[credit_score_distribution()],
        moment_targets=[average_order_total_moment()],
        correlations=[income_credit_correlation()],
        structural_constraints=[every_customer_has_an_order_fanout()],
        logic_constraints=[positive_order_total_logic()],
        derived_columns=[total_with_tax_derived()],
        state_sequences=[order_status_lifecycle()],
    )


def customer_only_extraction() -> UnifiedExtractionOutput:
    """What a CUSTOMER-only shard can legitimately extract: nothing here names
    ORDER, so it canonicalizes against a shard schema containing CUSTOMER
    alone."""
    return UnifiedExtractionOutput(
        distributions=[credit_score_distribution()],
        correlations=[income_credit_correlation()],
    )


def order_only_extraction() -> UnifiedExtractionOutput:
    """The ORDER-only counterpart."""
    return UnifiedExtractionOutput(
        moment_targets=[average_order_total_moment()],
        logic_constraints=[positive_order_total_logic()],
        derived_columns=[total_with_tax_derived()],
        state_sequences=[order_status_lifecycle()],
    )


def unrepairable_extraction() -> UnifiedExtractionOutput:
    """A constraint that can NEVER pass the deterministic checker: `nonexistent`
    is not a column of any table, so `canonicalize()`'s column-resolution step
    reports an error every round until the retry budget is exhausted.

    Used to drive the `det_errors_exhausted` case -- see the xfail in
    test_offline_stage3_failure_isolation.py.
    """
    return UnifiedExtractionOutput(
        logic_constraints=[
            Constraint(
                fact_references=[5],
                on=BaseTable(name="ORDER"),
                condition=RComparison(
                    op=">",
                    left=RColumnRef(name="nonexistent_column"),
                    right=RLiteral(value=0.0),
                ),
                category="logic",
            )
        ]
    )


# ---------------------------------------------------------------------------
# The other two Stage 3 agents
# ---------------------------------------------------------------------------


def clean_audit_report() -> AuditReport:
    return AuditReport(is_valid=True, issues=[], reasoning="Faithful extraction.")


def empty_reconciliation() -> GroupReconciliation:
    """No verdicts. Reached only if the conflict engine finds something; on a
    clean run the reconciler is never called at all, which is itself asserted."""
    return GroupReconciliation(verdicts=[])
