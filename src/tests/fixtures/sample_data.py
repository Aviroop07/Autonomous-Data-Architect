"""Sample-data builders shared across unit and integration tests.

Every function returns a freshly built object so tests may mutate results
without leaking state into other tests. Constructed strictly against the
verified model APIs:
  - Column(name, data_type="VARCHAR")          # no `type`/`pk` kwargs
  - Table(name, columns, pk, unique=None)
  - ForeignKey(referencing_table, referencing_column, referred_table)  # no referred_column
  - Schema(tables, relationships=None, ...)
  - AtomicFact(id, fact, origin="", referenced_fact_ids=[], is_external=False, tags=[])
  - TableFactRegistry().register_table_facts(name, [ids])
"""
from __future__ import annotations

from typing import List

from src.pipeline.stage1.models.atomic_fact import AtomicFact, FactTag
from src.pipeline.stage2.models.schema import (
    Schema,
    Table,
    Column,
    ForeignKey,
)
from src.pipeline.stage2.models.registry import TableFactRegistry


# --------------------------------------------------------------------------- #
# Fintech / credit-risk domain (salvaged from stage1/example + stage2/example)
# --------------------------------------------------------------------------- #

FINTECH_TITLE = "Fintech Lending Product - Initial Brainstorm"

FINTECH_NL = (
    "Hey, we're trying to build this simulator for our new credit product. "
    "Basically, we've got users who apply, and they've got all the usual stuff "
    "like scores and what they earn. We're seeing this weird thing where the "
    "'Approval Alpha' is super high for people in certain zip codes, maybe "
    "because of the legacy 'Tier-1' rules? Also, the 'Duration Drift' is "
    "starting to look like a bell curve but shifted. And obviously, we can't "
    "have 'double-dipping' in the drawdown phase. Oh, and make sure the "
    "'Spread' calculation respects the 'Basel-III' floor if they're "
    "institutional. We also track 'Maturity' and 'Yield' but sometimes they "
    "don't match up. The system should handle the 'LTV' calculation on the "
    "fly, and if the 'Haircut' is too aggressive, we need to flag it."
)

FINTECH_DOMAIN = "Fintech / Credit Risk"
FINTECH_GOAL = (
    "Simulate a new credit product with rigorous regulatory (Basel-III) and "
    "behavioral constraints."
)


def fintech_facts() -> List[AtomicFact]:
    """A representative atomic-fact list for the fintech lending domain."""
    return [
        AtomicFact(id=1, fact="A system is being developed to simulate a new credit product.", tags=[FactTag.METADATA]),
        AtomicFact(id=2, fact="Users of the system have credit scores associated with them.", tags=[FactTag.STRUCTURAL]),
        AtomicFact(id=3, fact="Users of the system have earnings information associated with them.", tags=[FactTag.STRUCTURAL]),
        AtomicFact(id=4, fact="'Approval Alpha' (a regional approval rate metric) is observed to be high for certain zip codes.", tags=[FactTag.STATISTICAL]),
        AtomicFact(id=5, fact="The high 'Approval Alpha' might be caused by legacy 'Tier-1' rules.", tags=[FactTag.STATISTICAL]),
        AtomicFact(id=6, fact="The 'Duration Drift' is observed to resemble a shifted bell curve.", tags=[FactTag.STATISTICAL]),
        AtomicFact(id=7, fact="Double-dipping (concurrent drawdowns) is not permissible during the drawdown phase.", tags=[FactTag.LOGICAL]),
        AtomicFact(id=8, fact="'Spread' calculation must respect the 'Basel-III' floor for institutional cases.", tags=[FactTag.LOGICAL]),
        AtomicFact(id=9, fact="The system tracks 'Maturity' and 'Yield' for each credit product.", tags=[FactTag.STRUCTURAL]),
        AtomicFact(id=10, fact="There are occasions when 'Maturity' and 'Yield' do not match, requiring logging.", tags=[FactTag.STATISTICAL]),
        AtomicFact(id=11, fact="The system should calculate 'LTV' (Loan-to-Value) on the fly.", tags=[FactTag.STRUCTURAL]),
        AtomicFact(id=12, fact="An aggressive 'Haircut' (collateral reduction) must be flagged by the system.", tags=[FactTag.LOGICAL]),
        AtomicFact(id=13, fact="'Approval Alpha' refers to region-specific trends in approval rates.", tags=[FactTag.METADATA]),
        AtomicFact(id=14, fact="'Basel-III' represents international banking regulations for capital and risk.", tags=[FactTag.METADATA]),
        AtomicFact(id=15, fact="'Duration Drift' is a deviation from the expected duration of a financial instrument.", tags=[FactTag.METADATA]),
    ]


def fintech_schema() -> Schema:
    """A small, valid two-table fintech schema (USER 1--* CREDIT_PRODUCT)."""
    return Schema(
        tables=[
            Table(
                name="USER",
                pk="user_id",
                columns=[
                    Column(name="user_id", data_type="INTEGER"),
                    Column(name="credit_score", data_type="INTEGER"),
                    Column(name="earnings_information", data_type="FLOAT"),
                    Column(name="is_institutional", data_type="BOOLEAN"),
                ],
            ),
            Table(
                name="CREDIT_PRODUCT",
                pk="credit_product_id",
                columns=[
                    Column(name="credit_product_id", data_type="INTEGER"),
                    Column(name="maturity", data_type="INTEGER"),
                    Column(name="yield_value", data_type="FLOAT"),
                    Column(name="user_id", data_type="INTEGER"),
                    Column(name="spread", data_type="FLOAT"),
                    Column(name="haircut", data_type="FLOAT"),
                    Column(name="ltv", data_type="FLOAT"),
                    Column(name="drawdown_phase", data_type="BOOLEAN"),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="CREDIT_PRODUCT",
                referencing_column="user_id",
                referred_table="USER",
            )
        ],
    )


def fintech_registry() -> TableFactRegistry:
    """A registry mapping the fintech tables to contributing fact IDs."""
    reg = TableFactRegistry()
    reg.register_table_facts("USER", [2, 3, 4, 5, 13])
    reg.register_table_facts("CREDIT_PRODUCT", [7, 8, 9, 10, 11, 12])
    return reg


# --------------------------------------------------------------------------- #
# Tiny generic two-table schema (handy for merger / matching / sharding tests)
# --------------------------------------------------------------------------- #

def simple_two_table_schema() -> Schema:
    """ALPHA (parent) <-- BETA (child via alpha_id). Minimal but valid."""
    return Schema(
        tables=[
            Table(
                name="ALPHA",
                pk="alpha_id",
                columns=[
                    Column(name="alpha_id", data_type="INTEGER"),
                    Column(name="label", data_type="VARCHAR"),
                ],
            ),
            Table(
                name="BETA",
                pk="beta_id",
                columns=[
                    Column(name="beta_id", data_type="INTEGER"),
                    Column(name="alpha_id", data_type="INTEGER"),
                    Column(name="description", data_type="VARCHAR"),
                ],
            ),
        ],
        relationships=[
            ForeignKey(
                referencing_table="BETA",
                referencing_column="alpha_id",
                referred_table="ALPHA",
            )
        ],
    )


def simple_facts() -> List[AtomicFact]:
    """Three plain facts for sharding / allocation tests."""
    return [
        AtomicFact(id=1, fact="Definition of the ALPHA entity and its label."),
        AtomicFact(id=2, fact="Definition of the BETA entity referencing ALPHA."),
        AtomicFact(id=3, fact="A fact relevant to both ALPHA and BETA entities."),
    ]
