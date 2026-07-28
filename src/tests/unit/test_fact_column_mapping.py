"""Tests for fact_column_mapping.py's build_fact_column_map() -- the
replacement for the old "grab every column of every mentioned table"
approach. Covers all three layered signals (per-column provenance, FK
relationship provenance, FK-path fallback) independently and combined,
using the exact real-world case that motivated this module: a fact like
"each order contains 1-15 order items" should map to ONLY order_id on
both sides, not every column of ORDER.
"""

from __future__ import annotations

from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.util.schema_model.data_types import DataType
from src.util.schema_model.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.middleware.fact_column_mapping import build_fact_column_map


def _fact(fid: int, text: str) -> AtomicFact:
    return AtomicFact(id=fid, fact=text)


def _col(name: str, source_fact_ids=None) -> Column:
    return Column(
        name=name, data_type=DataType.INTEGER, source_fact_ids=source_fact_ids or []
    )


class TestPerColumnProvenanceSignal:
    def test_uses_column_level_source_fact_ids_directly(self):
        schema = Schema(
            tables=[
                Table(
                    name="CUSTOMER",
                    primary_key=["email"],
                    columns=[
                        _col("email"),
                        _col("loyalty_tier", source_fact_ids=[3, 4, 20]),
                    ],
                ),
                Table(
                    name="ORDER",
                    primary_key=["order_id"],
                    columns=[
                        _col("order_id"),
                        _col("total_amount", source_fact_ids=[16, 20]),
                    ],
                ),
            ],
            relationships=[],
        )
        facts = [_fact(20, "The average order total for Platinum customers is $450.")]
        result = build_fact_column_map(schema, facts)
        assert 20 in result
        assert ("CUSTOMER", "loyalty_tier") in result[20]
        assert ("ORDER", "total_amount") in result[20]

    def test_table_level_only_fact_anchors_on_pk(self):
        schema = Schema(
            tables=[
                Table(
                    name="ORDER",
                    primary_key=["order_id"],
                    columns=[_col("order_id"), _col("status")],
                    source_fact_ids=[21],  # table-level only, no column has it
                ),
            ],
            relationships=[],
        )
        facts = [_fact(21, "Order status follows a strict lifecycle.")]
        result = build_fact_column_map(schema, facts)
        assert result[21] == [("ORDER", "order_id")]


class TestFkRelationshipProvenanceSignal:
    def test_uses_fk_source_fact_ids(self):
        schema = Schema(
            tables=[
                Table(
                    name="ORDER", primary_key=["order_id"], columns=[_col("order_id")]
                ),
                Table(
                    name="ORDER_ITEM",
                    primary_key=["order_item_id"],
                    columns=[_col("order_item_id"), _col("order_id")],
                ),
            ],
            relationships=[
                ForeignKey(
                    referencing_table="ORDER_ITEM",
                    referencing_column="order_id",
                    referred_table="ORDER",
                    source_fact_ids=[32],
                )
            ],
        )
        facts = [_fact(32, "Each order contains between 1 and 15 order items.")]
        result = build_fact_column_map(schema, facts)
        assert set(result[32]) == {("ORDER_ITEM", "order_id"), ("ORDER", "order_id")}
        # Critically: does NOT include unrelated columns like ORDER.status
        # or ORDER_ITEM.order_item_id -- the exact bug this module fixes.


class TestFkPathFallback:
    def test_directly_connected_tables_use_the_fk_edge(self):
        schema = Schema(
            tables=[
                Table(
                    name="ORDER", primary_key=["order_id"], columns=[_col("order_id")]
                ),
                Table(
                    name="ORDER_ITEM",
                    primary_key=["order_item_id"],
                    columns=[_col("order_item_id"), _col("order_id")],
                ),
            ],
            relationships=[
                ForeignKey(
                    referencing_table="ORDER_ITEM",
                    referencing_column="order_id",
                    referred_table="ORDER",
                )
            ],
        )
        # No provenance anywhere -- must fall back to text detection + FK path
        facts = [_fact(99, "Each order contains between 1 and 15 order items.")]
        result = build_fact_column_map(schema, facts)
        assert set(result[99]) == {("ORDER_ITEM", "order_id"), ("ORDER", "order_id")}

    def test_indirectly_connected_tables_include_the_hop_table(self):
        # A -> B -> C chain; a fact mentioning only A and C (never B) must
        # still include B's PK/FK columns, since the join can't skip it.
        schema = Schema(
            tables=[
                Table(name="A", primary_key=["a_id"], columns=[_col("a_id")]),
                Table(
                    name="B",
                    primary_key=["b_id"],
                    columns=[_col("b_id"), _col("a_id")],
                ),
                Table(
                    name="C",
                    primary_key=["c_id"],
                    columns=[_col("c_id"), _col("b_id")],
                ),
            ],
            relationships=[
                ForeignKey(
                    referencing_table="B", referencing_column="a_id", referred_table="A"
                ),
                ForeignKey(
                    referencing_table="C", referencing_column="b_id", referred_table="B"
                ),
            ],
        )
        facts = [_fact(1, "Every a relates to every c through the chain.")]
        result = build_fact_column_map(schema, facts)
        pairs = set(result[1])
        assert ("A", "a_id") in pairs
        assert ("B", "a_id") in pairs
        assert ("B", "b_id") in pairs
        assert ("C", "b_id") in pairs

    def test_single_table_fact_uses_its_pk(self):
        schema = Schema(
            tables=[
                Table(
                    name="PRODUCT",
                    primary_key=["product_id"],
                    columns=[_col("product_id"), _col("category")],
                )
            ],
            relationships=[],
        )
        facts = [_fact(1, "Products have a category.")]
        result = build_fact_column_map(schema, facts)
        assert result[1] == [("PRODUCT", "product_id")]

    def test_no_mention_yields_no_entry(self):
        schema = Schema(
            tables=[
                Table(name="CUSTOMER", primary_key=["email"], columns=[_col("email")]),
                Table(
                    name="PRODUCT",
                    primary_key=["product_id"],
                    columns=[_col("product_id")],
                ),
            ],
            relationships=[],
        )
        facts = [_fact(1, "Something entirely unrelated to any known table.")]
        result = build_fact_column_map(schema, facts)
        assert 1 not in result

    def test_disconnected_mentioned_tables_yield_no_pairs(self):
        # CUSTOMER and PRODUCT are both mentioned by name but have no FK
        # path between them -- no pairs get added for the disconnected
        # combination. The fact may still surface as a key with an empty
        # list (harmless: functionally identical to absent for any caller
        # using .get(fid, [])), so check the resolved columns, not key
        # presence.
        schema = Schema(
            tables=[
                Table(name="CUSTOMER", primary_key=["email"], columns=[_col("email")]),
                Table(
                    name="PRODUCT",
                    primary_key=["product_id"],
                    columns=[_col("product_id")],
                ),
            ],
            relationships=[],  # disconnected -- no FK path between them
        )
        facts = [
            _fact(2, "Customers and products are both mentioned but disconnected.")
        ]
        result = build_fact_column_map(schema, facts)
        assert result.get(2, []) == []


class TestSignalPriority:
    def test_real_provenance_takes_priority_over_fallback(self):
        # If a fact already has real provenance (signal 1/2), the fallback
        # (signal 3) must not also run and add noisy extra columns.
        schema = Schema(
            tables=[
                Table(
                    name="ORDER",
                    primary_key=["order_id"],
                    columns=[
                        _col("order_id"),
                        _col("status"),
                        _col("total_amount", source_fact_ids=[16]),
                    ],
                ),
                Table(
                    name="ORDER_ITEM",
                    primary_key=["order_item_id"],
                    columns=[_col("order_item_id"), _col("order_id")],
                ),
            ],
            relationships=[
                ForeignKey(
                    referencing_table="ORDER_ITEM",
                    referencing_column="order_id",
                    referred_table="ORDER",
                )
            ],
        )
        facts = [_fact(16, "Orders have a total amount, tracked per order item.")]
        result = build_fact_column_map(schema, facts)
        # Real provenance says this fact is ONLY about ORDER.total_amount --
        # the fallback's table-mention detection would also see ORDER_ITEM,
        # but must not run since signal 1 already resolved it.
        assert result[16] == [("ORDER", "total_amount")]
