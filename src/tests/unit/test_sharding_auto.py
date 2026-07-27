"""Tests for src/util/algorithms/sharding_ilp.py's shard_schema_auto() and
its derivation helpers -- the automatic entry point that exposes NO
manually-set max_shards/max_tables_per_shard/weight hyperparameters to
callers. Everything is derived from the schema, the fact set, and the
target provider/model's real (mocked here) context window.
"""

from __future__ import annotations

from unittest.mock import patch

from src.util.algorithms.sharding_ilp import (
    ILPSharder,
    SearchSpaceSeeder,
    _derive_max_shards,
    _derive_max_tables_per_shard,
    _estimate_prompt_tokens_per_table,
    shard_schema_auto,
)


class TestEstimatePromptTokensPerTable:
    def test_empty_schema_returns_fallback(self):
        assert _estimate_prompt_tokens_per_table({}) == 40.0

    def test_more_columns_means_more_tokens(self):
        small = _estimate_prompt_tokens_per_table({"T": ["a", "b"]})
        large = _estimate_prompt_tokens_per_table({"T": [f"c{i}" for i in range(20)]})
        assert large > small
        assert small > 0


class TestDeriveMaxShards:
    def test_bounded_by_valid_fact_count(self):
        tables = [f"T{i}" for i in range(8)]
        facts = {"1": [("T0", "a")], "2": [("T1", "b")], "3": []}
        # 2 valid (non-empty) facts, 8 tables -> min(8, 2) = 2
        assert _derive_max_shards(tables, facts) == 2

    def test_bounded_by_table_count(self):
        tables = ["T0", "T1", "T2"]
        facts = {str(i): [("T0", "a")] for i in range(10)}
        # 10 valid facts, 3 tables -> min(3, 10) = 3
        assert _derive_max_shards(tables, facts) == 3

    def test_no_valid_facts_falls_back_to_table_count(self):
        tables = ["T0", "T1"]
        facts = {"1": [], "2": []}
        assert _derive_max_shards(tables, facts) == 2

    def test_never_zero(self):
        assert _derive_max_shards([], {}) == 1


class TestDeriveMaxTablesPerShard:
    def test_uses_context_window_budget(self):
        tables = [f"T{i}" for i in range(20)]
        columns_by_table = {t: ["id", "name", "value"] for t in tables}
        with patch(
            "src.util.core.context_window.get_context_window", return_value=100_000
        ):
            result = _derive_max_tables_per_shard(
                tables,
                columns_by_table,
                provider="deepseek",
                model="deepseek-v4-flash",
                api_key="",
                fixed_prompt_overhead_tokens=1000,
                context_window_safety_margin=0.5,
            )
        assert 1 <= result <= len(tables)

    def test_degrades_to_table_count_when_budget_non_positive(self):
        tables = [f"T{i}" for i in range(5)]
        columns_by_table = {t: ["id"] for t in tables}
        with patch("src.util.core.context_window.get_context_window", return_value=100):
            result = _derive_max_tables_per_shard(
                tables,
                columns_by_table,
                provider="deepseek",
                model="deepseek-v4-flash",
                api_key="",
                fixed_prompt_overhead_tokens=1_000_000,  # dwarfs the context window
                context_window_safety_margin=0.5,
            )
        assert result == len(tables)


class TestSearchSpaceSeederWeightRanges:
    def test_weight_ranges_has_exactly_three_swept_dimensions(self):
        # w_shard is fixed at 1 (the numeraire, added separately in
        # generate_grid()) -- only w_size/w_cohesion/w_facts are swept.
        seeder = SearchSpaceSeeder()
        ranges = seeder.weight_ranges()
        assert set(ranges.keys()) == {"w_size", "w_cohesion", "w_facts"}
        assert all(len(v) > 0 for v in ranges.values())

    def test_generate_grid_is_the_cartesian_product_of_the_ranges(self):
        seeder = SearchSpaceSeeder()
        ranges = seeder.weight_ranges()
        expected_size = 1
        for v in ranges.values():
            expected_size *= len(v)

        grid = seeder.generate_grid()

        assert len(grid) == expected_size
        assert all(
            set(combo.keys()) == {"w_size", "w_shard", "w_cohesion", "w_facts"}
            for combo in grid
        )
        # w_shard is the fixed numeraire, never swept.
        assert all(combo["w_shard"] == 1 for combo in grid)


class TestNoSpuriousDuplication:
    """Regression coverage for a real bug found via live validation: for a
    fully FK-connected schema (every table transitively required together
    via FK closure), the OLD soft objective could duplicate the ENTIRE
    schema across every available shard slot for zero additional cost,
    because (a) the co-occurrence reward summed per-(pair, shard) instead
    of being capped once per pair, and (b) the size penalty was a makespan
    (bounds only the single worst shard) rather than a sum across active
    shards, so it stopped growing entirely once every shard already hit
    its capacity ceiling -- confirmed live on a real 9-table, fully
    FK-chained e-commerce schema, which produced 9 shards that were each a
    full duplicate copy of everything rather than a real partition, when
    max_shards/max_tables_per_shard were both generously large (exactly
    the situation this fixture reproduces on a tiny 3-table chain)."""

    def test_fully_fk_connected_schema_does_not_duplicate_across_shards(self):
        tables = ["A", "B", "C"]
        cols = {"A": ["a_id"], "B": ["b_id", "a_fk"], "C": ["c_id", "b_fk"]}
        pks = {"A": ["a_id"], "B": ["b_id"], "C": ["c_id"]}
        fks = [("B", "a_fk", "A", "a_id"), ("C", "b_fk", "B", "b_id")]
        facts = {
            "f1": [("A", "a_id"), ("B", "b_id")],
            "f2": [("B", "b_id"), ("C", "c_id")],
        }

        # Generous ceilings, mirroring the live conditions that exposed
        # the bug: max_shards well above what's structurally needed, and
        # max_tables_per_shard large enough to hold the whole schema.
        sharder = ILPSharder(
            max_shards=5,
            max_tables_per_shard=10,
            w_size=10,
            w_shard=20,
            w_cohesion=30,
        )
        shards, shard_facts = sharder.shard_schema(tables, cols, pks, fks, facts)

        assert shards is not None
        full_schema_copies = [s for s in shards if set(s.keys()) == {"A", "B", "C"}]
        assert len(full_schema_copies) == 1, (
            f"Expected exactly one shard containing the full FK-connected "
            f"schema, got {len(full_schema_copies)} duplicate copies "
            f"across {len(shards)} total active shard(s) -- the "
            f"co-occurrence reward is rewarding duplication instead of "
            f"being capped once per pair."
        )
        assert len(shards) == 1


class TestShardSchemaAutoIntegration:
    def test_produces_valid_shards_with_no_manual_hyperparameters(self):
        tables = ["CUSTOMERS", "ORDERS", "PRODUCTS"]
        cols = {
            "CUSTOMERS": ["C_CUSTKEY", "C_NAME"],
            "ORDERS": ["O_ORDERKEY", "O_CUSTKEY", "O_TOTALPRICE"],
            "PRODUCTS": ["P_PRODUCTKEY", "P_NAME"],
        }
        pks = {
            "CUSTOMERS": ["C_CUSTKEY"],
            "ORDERS": ["O_ORDERKEY"],
            "PRODUCTS": ["P_PRODUCTKEY"],
        }
        fks = [("ORDERS", "O_CUSTKEY", "CUSTOMERS", "C_CUSTKEY")]
        facts = {"fact_1": [("CUSTOMERS", "C_NAME"), ("ORDERS", "O_TOTALPRICE")]}

        with patch(
            "src.util.core.context_window.get_context_window", return_value=1_000_000
        ):
            shards, shard_facts = shard_schema_auto(
                tables,
                cols,
                pks,
                fks,
                facts,
                provider="deepseek",
                model="deepseek-v4-flash",
            )

        assert shards is not None
        found_joint = False
        for shard in shards:
            if "CUSTOMERS" in shard and "ORDERS" in shard:
                found_joint = True
                assert "C_CUSTKEY" in shard["CUSTOMERS"]
                assert "O_ORDERKEY" in shard["ORDERS"]
        assert found_joint, (
            "shard_schema_auto failed to group tables required by fact_1"
        )
