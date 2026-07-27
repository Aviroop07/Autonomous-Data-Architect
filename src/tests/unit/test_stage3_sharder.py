from src.util.algorithms.sharding_ilp import ILPSharder, run_stability_sweep


def test_ilp_sharder_basic():
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

    # Fact demanding customer names and their order prices
    facts = {"fact_1": [("CUSTOMERS", "C_NAME"), ("ORDERS", "O_TOTALPRICE")]}

    sharder = ILPSharder(
        max_shards=2, max_tables_per_shard=3, w_size=10, w_shard=20, w_cohesion=30
    )
    shards, shard_facts = sharder.shard_schema(tables, cols, pks, fks, facts)

    assert shards is not None
    # Ensure FK closure and facts are satisfied
    # At least one shard must contain BOTH CUSTOMERS and ORDERS due to fact_1
    found_joint = False
    for shard in shards:
        if "CUSTOMERS" in shard and "ORDERS" in shard:
            found_joint = True
            # Verify primary keys are included
            assert "C_CUSTKEY" in shard["CUSTOMERS"]
            assert "O_ORDERKEY" in shard["ORDERS"]

    assert found_joint, "ILP failed to group tables required by fact_1"


def test_column_level_affinity_splits_a_single_table_into_projections():
    """The core motivation for upgrading SO1 to column granularity: a
    single table's DIFFERENT columns should be able to land in DIFFERENT
    shards based on which other table each column's facts actually pull
    it toward, with the PK constant everywhere the table appears --
    genuine vertical fragmentation, not just whole-table grouping."""
    tables = ["A", "B", "C"]
    cols = {"A": ["a_id", "a1", "a2"], "B": ["b_id", "b1"], "C": ["c_id", "c1"]}
    pks = {"A": ["a_id"], "B": ["b_id"], "C": ["c_id"]}
    fks = []
    # a1 belongs with B; a2 belongs with C -- nothing ties a1 and a2
    # together, and nothing ties B and C together.
    facts = {
        "f1": [("A", "a1"), ("B", "b1")],
        "f2": [("A", "a2"), ("C", "c1")],
    }

    sharder = ILPSharder(
        max_shards=2,
        max_tables_per_shard=2,
        w_size=1,
        w_shard=1,
        w_cohesion=30,
        w_facts=1,
    )
    shards, shard_facts = sharder.shard_schema(tables, cols, pks, fks, facts)

    assert shards is not None
    assert len(shards) == 2

    a_projections = [set(s["A"]) for s in shards if "A" in s]
    assert len(a_projections) == 2, "table A should appear in both shards"

    b_shard = next(s for s in shards if "B" in s)
    c_shard = next(s for s in shards if "C" in s)

    # A's PK travels everywhere A appears (the vertical-fragmentation
    # invariant), but its two non-key columns split by real affinity.
    assert "a_id" in b_shard["A"]
    assert "a1" in b_shard["A"]
    assert "a2" not in b_shard["A"]

    assert "a_id" in c_shard["A"]
    assert "a2" in c_shard["A"]
    assert "a1" not in c_shard["A"]


def test_stability_sweeper_mocked():
    tables = ["CUSTOMERS", "ORDERS"]
    cols = {"CUSTOMERS": ["C_CUSTKEY", "C_NAME"], "ORDERS": ["O_ORDERKEY", "O_CUSTKEY"]}
    pks = {"CUSTOMERS": ["C_CUSTKEY"], "ORDERS": ["O_ORDERKEY"]}
    fks = [("ORDERS", "O_CUSTKEY", "CUSTOMERS", "C_CUSTKEY")]
    facts = {"fact_1": [("CUSTOMERS", "C_NAME"), ("ORDERS", "O_CUSTKEY")]}

    plateau = run_stability_sweep(
        tables, cols, pks, fks, facts, max_shards=2, max_tables_per_shard=2
    )

    assert plateau is not None
    assert len(plateau.structure) > 0
    assert plateau.frequency > 0
