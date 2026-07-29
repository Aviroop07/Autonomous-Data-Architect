"""Canned Stage 2 agent outputs.

The conceptual model here is the ONLY thing Stage 2 is told; the relational
schema the offline tests assert against is whatever the REAL deterministic mapper
derives from it (table names, FK column names, junction tables, nullability).
Nothing about the schema is hard-coded on this side, which is what makes the
schema-resolution tests meaningful rather than circular.
"""

from __future__ import annotations

from src.pipeline.stage2.mapper.conceptual_model import (
    CMAttribute,
    ConceptualModel,
    Entity,
    Participant,
    Relationship,
)
from src.pipeline.stage2.models.conceptual_critique import ConceptualCritiqueReport
from src.util.schema_model.data_types import DataType
from src.util.schema_ops.schema_patch import CritiqueReport


def conceptual_model() -> ConceptualModel:
    """CUSTOMER 1--N ORDER, with every attribute traced to a Stage 1 fact id.

    Every one of facts 1-6 appears in some `source_fact_ids` here, which is what
    lets the partition invariant distinguish "Stage 2 covered it" from "Stage 2
    dropped it" rather than just counting.
    """
    return ConceptualModel(
        entities=[
            Entity(
                name="CUSTOMER",
                attributes=[
                    CMAttribute(
                        name="customer_id", type=DataType.INTEGER, source_fact_ids=[1]
                    ),
                    CMAttribute(
                        name="name", type=DataType.VARCHAR, source_fact_ids=[2]
                    ),
                    CMAttribute(
                        name="credit_score", type=DataType.INTEGER, source_fact_ids=[3]
                    ),
                    CMAttribute(
                        name="annual_income", type=DataType.FLOAT, source_fact_ids=[4]
                    ),
                ],
                identifier_attributes=["customer_id"],
                source_fact_ids=[1, 2, 3, 4],
            ),
            Entity(
                name="ORDER",
                attributes=[
                    CMAttribute(
                        name="order_id", type=DataType.INTEGER, source_fact_ids=[5]
                    ),
                    CMAttribute(
                        name="total_amount", type=DataType.FLOAT, source_fact_ids=[5]
                    ),
                    CMAttribute(
                        name="status", type=DataType.VARCHAR, source_fact_ids=[6]
                    ),
                ],
                identifier_attributes=["order_id"],
                source_fact_ids=[5, 6],
            ),
        ],
        relationships=[
            Relationship(
                name="places",
                degree="binary",
                kind="1:N",
                participants=[
                    # The FK-holding child is the participant whose
                    # cardinality_max is not 1 -- see relational_mapper.py's
                    # 1:N branch. Getting this backwards silently reverses the
                    # foreign key, so it is stated explicitly rather than
                    # left to ordering.
                    Participant(
                        entity="ORDER", cardinality_min=1, cardinality_max=None
                    ),
                    Participant(
                        entity="CUSTOMER", cardinality_min=0, cardinality_max=1
                    ),
                ],
                source_fact_ids=[1],
            )
        ],
        functional_dependencies=[],
    )


def clean_conceptual_critique() -> ConceptualCritiqueReport:
    return ConceptualCritiqueReport(is_valid=True, fixes=[])


def clean_compliance_report() -> CritiqueReport:
    """No patches, so the certifier leaves the mapped schema exactly as the
    mapper produced it -- the schema under test stays the mapper's output."""
    return CritiqueReport(agent_name="compliance_certifier", patches=[])
