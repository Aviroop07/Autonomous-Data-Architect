import pytest
from src.util.algorithms.sharding_ilp import ILPSharder, SearchSpaceSeeder, StabilitySelector, run_stability_sweep

def test_ilp_sharder_basic():
    tables = ["CUSTOMERS", "ORDERS", "PRODUCTS"]
    cols = {
        "CUSTOMERS": ["C_CUSTKEY", "C_NAME"],
        "ORDERS": ["O_ORDERKEY", "O_CUSTKEY", "O_TOTALPRICE"],
        "PRODUCTS": ["P_PRODUCTKEY", "P_NAME"]
    }
    pks = {
        "CUSTOMERS": ["C_CUSTKEY"],
        "ORDERS": ["O_ORDERKEY"],
        "PRODUCTS": ["P_PRODUCTKEY"]
    }
    fks = [("ORDERS", "O_CUSTKEY", "CUSTOMERS", "C_CUSTKEY")]
    
    # Fact demanding customer names and their order prices
    facts = {
        "fact_1": [("CUSTOMERS", "C_NAME"), ("ORDERS", "O_TOTALPRICE")]
    }

    sharder = ILPSharder(max_shards=2, max_tables_per_shard=3, w_cap=10, w_shard=20, w_fact=50, w_fk=30, w_col=5)
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

def test_stability_sweeper_mocked():
    tables = ["CUSTOMERS", "ORDERS"]
    cols = {
        "CUSTOMERS": ["C_CUSTKEY", "C_NAME"],
        "ORDERS": ["O_ORDERKEY", "O_CUSTKEY"]
    }
    pks = {
        "CUSTOMERS": ["C_CUSTKEY"],
        "ORDERS": ["O_ORDERKEY"]
    }
    fks = [("ORDERS", "O_CUSTKEY", "CUSTOMERS", "C_CUSTKEY")]
    facts = {
        "fact_1": [("CUSTOMERS", "C_NAME"), ("ORDERS", "O_CUSTKEY")]
    }
    
    plateau = run_stability_sweep(tables, cols, pks, fks, facts)
    
    assert plateau is not None
    assert len(plateau.structure) > 0
    assert plateau.frequency > 0