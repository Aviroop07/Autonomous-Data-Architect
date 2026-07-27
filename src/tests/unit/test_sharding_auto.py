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


class TestSingleShardFastPath:
    """When every table fits inside one shard, the single-shard assignment is a
    global optimum for EVERY positive weight vector, so the 216-config sweep is
    guaranteed to return it and is skipped.

    Why it is optimal, term by term (see shard_schema_auto's own comment): one
    active shard is the strict minimum for the w_shard term; total_size is
    minimal because HC3/HC4 already force every table and column into at least
    one shard and extra shards can only duplicate; every fact touches exactly
    one shard, minimising the w_facts term; and every co-occurring pair is
    co-located, maximising the subtracted cohesion reward.

    Verified against the real solver on a live 11-table schema: all six corner
    and centre weight vectors of the 0.1-30 cube returned one shard, matching
    this path. Without the skip that check cost 30s per config; the sweep is
    216 of them.
    """

    TABLES = ["CUSTOMERS", "ORDERS", "PRODUCTS"]
    COLS = {
        "CUSTOMERS": ["C_CUSTKEY", "C_NAME"],
        "ORDERS": ["O_ORDERKEY", "O_CUSTKEY"],
        "PRODUCTS": ["P_PRODUCTKEY"],
    }
    PKS = {
        "CUSTOMERS": ["C_CUSTKEY"],
        "ORDERS": ["O_ORDERKEY"],
        "PRODUCTS": ["P_PRODUCTKEY"],
    }
    FKS = [("ORDERS", "O_CUSTKEY", "CUSTOMERS", "C_CUSTKEY")]

    def _run(self, facts, context_window=1_000_000):
        with patch(
            "src.util.core.context_window.get_context_window",
            return_value=context_window,
        ):
            return shard_schema_auto(
                self.TABLES,
                self.COLS,
                self.PKS,
                self.FKS,
                facts,
                provider="deepseek",
                model="deepseek-v4-flash",
            )

    def test_returns_exactly_one_shard_holding_every_table_and_column(self):
        shards, _ = self._run({"f1": [("CUSTOMERS", "C_NAME")]})
        assert shards is not None
        assert len(shards) == 1
        assert set(shards[0]) == set(self.TABLES)
        for table, columns in self.COLS.items():
            assert sorted(shards[0][table]) == sorted(columns)

    def test_all_resolvable_facts_are_reported_as_contained(self):
        facts = {
            "f1": [("CUSTOMERS", "C_NAME")],
            "f2": [("ORDERS", "O_ORDERKEY"), ("PRODUCTS", "P_PRODUCTKEY")],
        }
        _, shard_facts = self._run(facts)
        assert shard_facts is not None
        assert sorted(shard_facts[0]) == ["f1", "f2"]

    def test_unresolvable_facts_are_excluded_matching_HC7a(self):
        """A fact whose column list resolves to nothing is forced h=0 by HC7a
        in the solver, so the fast path must exclude it too rather than
        claiming containment the ILP would have denied."""
        facts = {"real": [("CUSTOMERS", "C_NAME")], "bogus": []}
        _, shard_facts = self._run(facts)
        assert shard_facts is not None
        assert shard_facts[0] == ["real"]

    def test_skips_the_sweep_entirely(self):
        """The point of the fast path. If run_stability_sweep is reached at
        all, the skip did not happen."""
        with patch("src.util.algorithms.sharding_ilp.run_stability_sweep") as swept:
            shards, _ = self._run({"f1": [("CUSTOMERS", "C_NAME")]})
        swept.assert_not_called()
        assert shards is not None and len(shards) == 1

    def test_no_resolvable_fact_falls_through_to_the_sweep(self):
        """HC9 requires every active shard to hold at least one valid fact, so
        the one-shard assignment is not necessarily feasible when none exists
        -- that case must still go to the solver."""
        with patch(
            "src.util.algorithms.sharding_ilp.run_stability_sweep",
            return_value=None,
        ) as swept:
            self._run({"bogus": []})
        swept.assert_called_once()

    def test_schema_too_large_for_one_shard_falls_through_to_the_sweep(self):
        """When the schema genuinely does not fit, HC5 fails, the fast path's
        precondition is false, and the real optimisation must run.

        This needs WIDE tables, not merely many of them, and not a small
        context window. 40 narrow tables really do fit in 50k tokens, and a
        window small enough to make the budget negative degrades to
        "no meaningful cap" by design. 40 tables of 200 columns (~1752 tokens
        each) against a 50k window is a case where the cap genuinely binds:
        max_tables_per_shard comes out at 13.
        """
        wide_cols = {
            f"T{i:02d}": [f"column_name_{j}" for j in range(200)] for i in range(40)
        }
        wide_pks = {t: [f"{t}_id"] for t in wide_cols}
        with patch(
            "src.util.algorithms.sharding_ilp.run_stability_sweep",
            return_value=None,
        ) as swept:
            with patch(
                "src.util.core.context_window.get_context_window",
                return_value=50_000,
            ):
                shard_schema_auto(
                    list(wide_cols),
                    wide_cols,
                    wide_pks,
                    [],
                    {"f1": [("T00", "column_name_0")]},
                    provider="deepseek",
                    model="deepseek-v4-flash",
                )
        swept.assert_called_once()
