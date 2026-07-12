"""Unit tests for fact_allocation.py, per the finding that the ILP
sharder's FK-closure is a soft reward, not a guarantee (sharding_ilp.py):
a fact can end up with no single shard containing all its referenced
tables. These tests cover the table-mention-aware orphan recovery and the
stub_tables computation this drove, not just the pre-existing base/
similarity allocation paths.
"""

from __future__ import annotations

from src.pipeline.stage1.models.atomic_fact import AtomicFact
from src.pipeline.stage2.models.registry import TableFactRegistry
from src.pipeline.stage3.middleware.fact_allocation import (
    allocate_facts_to_shards,
    find_mentioned_tables,
)


def _fact(fid: int, text: str) -> AtomicFact:
    return AtomicFact(id=fid, fact=text)


class TestFindMentionedTables:
    def test_matches_exact_and_natural_language_variants(self):
        assert find_mentioned_tables(
            "Each line_item belongs to exactly one order.", ["LINE_ITEM", "ORDER"]
        ) == {"LINE_ITEM", "ORDER"}
        assert find_mentioned_tables(
            "Line items reference a single order.", ["LINE_ITEM", "ORDER"]
        ) == {"LINE_ITEM", "ORDER"}

    def test_no_match_returns_empty_set(self):
        assert (
            find_mentioned_tables("Customers have a loyalty tier.", ["ORDER"]) == set()
        )


class TestOrphanRecovery:
    def test_table_mention_routes_to_the_shard_covering_the_most_tables(self):
        """A fact registry never associated with any table (a genuine
        orphan) but whose text explicitly names two tables split across
        different shards must route to the shard covering MORE of them,
        not to an unrelated shard picked by fact-to-fact text similarity."""
        facts = [
            _fact(1, "Order totals are the sum of each line item's subtotal."),
            _fact(2, "Customers have a name and an email address."),
        ]
        registry = TableFactRegistry()
        # Fact 1 deliberately NOT registered under any table -- a true orphan.
        registry.register_table_facts("CUSTOMER", [2])

        shard_table_sets = [{"CUSTOMER"}, {"ORDER", "LINE_ITEM"}]
        result = allocate_facts_to_shards(facts, shard_table_sets, registry)

        assert 1 in result[1].fact_ids
        assert 1 not in result[0].fact_ids

    def test_falls_back_to_similarity_when_no_table_is_mentioned(self):
        """An orphan whose text names no known table at all must still be
        placed via the pre-existing fact-to-fact similarity fallback,
        never dropped."""
        facts = [
            _fact(1, "Every entity must have exactly one owner."),  # no table names
            _fact(2, "Customers have a name and an email address."),
        ]
        registry = TableFactRegistry()
        registry.register_table_facts("CUSTOMER", [2])

        shard_table_sets = [{"CUSTOMER"}, {"ORDER"}]
        result = allocate_facts_to_shards(facts, shard_table_sets, registry)

        all_allocated = {fid for r in result for fid in r.fact_ids}
        assert 1 in all_allocated


class TestStubTables:
    def test_cross_table_fact_produces_a_stub_for_the_foreign_table(self):
        """A fact allocated to a shard that mentions a table OUTSIDE that
        shard's own projection must surface that table as a stub -- the
        extraction agent needs schema-only context for it."""
        facts = [
            _fact(
                1,
                "Order totals are the sum of each line item's subtotal.",
            )
        ]
        registry = TableFactRegistry()
        registry.register_table_facts("ORDER", [1])

        shard_table_sets = [{"ORDER"}, {"LINE_ITEM"}]
        result = allocate_facts_to_shards(facts, shard_table_sets, registry)

        order_shard = result[0]
        assert 1 in order_shard.fact_ids
        assert order_shard.stub_tables == ["LINE_ITEM"]

    def test_no_stub_when_all_mentioned_tables_are_local(self):
        facts = [_fact(1, "Customers have a name and an email address.")]
        registry = TableFactRegistry()
        registry.register_table_facts("CUSTOMER", [1])

        shard_table_sets = [{"CUSTOMER"}]
        result = allocate_facts_to_shards(facts, shard_table_sets, registry)

        assert result[0].stub_tables == []
