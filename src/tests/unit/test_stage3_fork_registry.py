import pytest
from src.pipeline.stage3.middleware.fork_registry import ForkKey, BranchCondition, ForkKeyRegistry, parse_if_condition

def test_parse_if_condition():
    cond = parse_if_condition("CUSTOMER.loyalty_tier = 'Platinum'")
    assert cond is not None
    assert cond.fork_key.table_name == "CUSTOMER"
    assert cond.fork_key.column_name == "loyalty_tier"
    assert cond.operator == "EQ"
    assert cond.values == ["Platinum"]
    
    cond2 = parse_if_condition("ORDER.status != 'Shipped'")
    assert cond2 is not None
    assert cond2.fork_key.table_name == "ORDER"
    assert cond2.fork_key.column_name == "status"
    assert cond2.operator == "NEQ"
    assert cond2.values == ["Shipped"]
    
    assert parse_if_condition("1 = 1") is None
    assert parse_if_condition("invalid sql") is None

def test_fork_registry():
    registry = ForkKeyRegistry()
    fk = ForkKey(table_name="CUSTOMER", column_name="loyalty_tier")
    registry.register_fork(fk, ["Bronze", "Silver", "Gold", "Platinum"])
    
    registry.register_fork(fk, ["Bronze", "Silver", "Gold", "Platinum"])
    assert len(registry.forks) == 1
    
    cond1 = parse_if_condition("CUSTOMER.loyalty_tier = 'Platinum'")
    branches1 = registry.get_branches_for_condition(cond1)
    assert branches1 == ["Platinum"]
    
    cond2 = parse_if_condition("CUSTOMER.loyalty_tier != 'Platinum'")
    branches2 = registry.get_branches_for_condition(cond2)
    assert set(branches2) == {"Bronze", "Silver", "Gold"}
    
    cond_unknown = parse_if_condition("UNKNOWN.col = 'Val'")
    branches3 = registry.get_branches_for_condition(cond_unknown)
    assert branches3 == ["Val"]
