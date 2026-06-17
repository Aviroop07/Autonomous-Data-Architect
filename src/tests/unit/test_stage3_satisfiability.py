from __future__ import annotations

from src.orchestration.stage3.entry import _build_global_table_manifests
from src.orchestration.stage3.models import ShardMetadata
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.pipeline.stage3.middleware.satisfiability import (
    check_state_constraint_satisfiability,
    check_structural_satisfiability,
)
from src.pipeline.stage3.models.manifest import TableConstraintManifest
from src.pipeline.stage3.models.sql_models import (
    BinaryOperator,
    BinaryOperand,
    CardinalityConstraint,
    FanoutConstraint,
    SQLGroundedConstraint,
)


def _schema() -> Schema:
    return Schema(tables=[
        Table(
            name="ORDER_ITEM",
            pk="order_item_id",
            columns=[
                Column(name="order_item_id", data_type="INTEGER"),
                Column(name="quantity", data_type="INTEGER"),
                Column(name="unit_price", data_type="DECIMAL"),
                Column(name="status", data_type="VARCHAR"),
                Column(name="processed_at", data_type="TIMESTAMP"),
            ],
        )
    ])


def _constraint(
    left_operand: str,
    operator: BinaryOperator,
    right_operand: BinaryOperand | None = None,
) -> SQLGroundedConstraint:
    return SQLGroundedConstraint(
        state_query="SELECT quantity, unit_price, status, processed_at FROM ORDER_ITEM",
        left_operand=left_operand,
        operator=operator,
        right_operand=right_operand,
        fact_references=[1],
    )


def _commerce_schema() -> Schema:
    return Schema(tables=[
        Table(
            name="PRODUCT",
            pk="product_id",
            columns=[
                Column(name="product_id", data_type="INTEGER"),
                Column(name="stock_quantity", data_type="INTEGER"),
            ],
        ),
        Table(
            name="ORDER_ITEM",
            pk="order_item_id",
            columns=[
                Column(name="order_item_id", data_type="INTEGER"),
                Column(name="product_id", data_type="INTEGER"),
                Column(name="quantity", data_type="INTEGER"),
                Column(name="unit_price", data_type="DECIMAL"),
            ],
        ),
        Table(
            name="PAYMENT",
            pk="payment_id",
            columns=[
                Column(name="payment_id", data_type="INTEGER"),
                Column(name="status", data_type="VARCHAR"),
                Column(name="processed_at", data_type="TIMESTAMP"),
            ],
        ),
    ], relationships=[
        ForeignKey(
            referencing_table="ORDER_ITEM",
            referencing_column="product_id",
            referred_table="PRODUCT",
        )
    ])


def _order_schema() -> Schema:
    return Schema(tables=[
        Table(
            name="ORDER",
            pk="order_id",
            columns=[Column(name="order_id", data_type="INTEGER")],
        ),
        Table(
            name="ORDER_ITEM",
            pk="order_item_id",
            columns=[
                Column(name="order_item_id", data_type="INTEGER"),
                Column(name="order_id", data_type="INTEGER"),
            ],
        ),
    ], relationships=[
        ForeignKey(
            referencing_table="ORDER_ITEM",
            referencing_column="order_id",
            referred_table="ORDER",
        )
    ])


def _cloud_schema() -> Schema:
    return Schema(tables=[
        Table(
            name="DATA_CENTERS",
            pk="region_id",
            columns=[
                Column(name="region_id", data_type="INTEGER"),
                Column(name="region_name", data_type="VARCHAR"),
                Column(name="power_cost_per_hour", data_type="DECIMAL"),
            ],
        ),
        Table(
            name="COMPUTE_NODES",
            pk="node_id",
            columns=[
                Column(name="node_id", data_type="INTEGER"),
                Column(name="region_id", data_type="INTEGER"),
                Column(name="max_cpu_cores", data_type="INTEGER"),
                Column(name="max_ram_gb", data_type="INTEGER"),
                Column(name="is_active", data_type="INTEGER"),
            ],
        ),
        Table(
            name="TENANT_PROFILES",
            pk="tenant_id",
            columns=[
                Column(name="tenant_id", data_type="INTEGER"),
                Column(name="subscription_tier", data_type="VARCHAR"),
                Column(name="monthly_budget_cap", data_type="DECIMAL"),
            ],
        ),
        Table(
            name="VM_INSTANCES",
            pk="instance_id",
            columns=[
                Column(name="instance_id", data_type="INTEGER"),
                Column(name="tenant_id", data_type="INTEGER"),
                Column(name="node_id", data_type="INTEGER"),
                Column(name="allocated_cpu", data_type="INTEGER"),
                Column(name="allocated_ram", data_type="INTEGER"),
                Column(name="runtime_hours", data_type="INTEGER"),
                Column(name="status", data_type="VARCHAR"),
            ],
        ),
        Table(
            name="BILLING_LEDGERS",
            pk="ledger_id",
            columns=[
                Column(name="ledger_id", data_type="INTEGER"),
                Column(name="tenant_id", data_type="INTEGER"),
                Column(name="gross_charge", data_type="DECIMAL"),
                Column(name="discount_applied", data_type="DECIMAL"),
                Column(name="net_bill", data_type="DECIMAL"),
            ],
        ),
    ], relationships=[
        ForeignKey(
            referencing_table="COMPUTE_NODES",
            referencing_column="region_id",
            referred_table="DATA_CENTERS",
        ),
        ForeignKey(
            referencing_table="VM_INSTANCES",
            referencing_column="tenant_id",
            referred_table="TENANT_PROFILES",
        ),
        ForeignKey(
            referencing_table="VM_INSTANCES",
            referencing_column="node_id",
            referred_table="COMPUTE_NODES",
        ),
        ForeignKey(
            referencing_table="BILLING_LEDGERS",
            referencing_column="tenant_id",
            referred_table="TENANT_PROFILES",
        ),
    ])


def _cloud_capacity_query() -> str:
    return (
        "SELECT COMPUTE_NODES.node_id, "
        "SUM(VM_INSTANCES.allocated_cpu) AS total_cpu, "
        "SUM(VM_INSTANCES.allocated_ram) AS total_ram, "
        "COMPUTE_NODES.max_cpu_cores, COMPUTE_NODES.max_ram_gb "
        "FROM COMPUTE_NODES JOIN VM_INSTANCES "
        "ON COMPUTE_NODES.node_id = VM_INSTANCES.node_id "
        "WHERE VM_INSTANCES.status = 'Active' "
        "GROUP BY COMPUTE_NODES.node_id, COMPUTE_NODES.max_cpu_cores, COMPUTE_NODES.max_ram_gb"
    )


def _cloud_pricing_query() -> str:
    return (
        "SELECT VM_INSTANCES.tenant_id, "
        "SUM(VM_INSTANCES.runtime_hours * VM_INSTANCES.allocated_cpu * "
        "CASE "
        "WHEN TENANT_PROFILES.subscription_tier = 'Enterprise' THEN 0.04 "
        "WHEN TENANT_PROFILES.subscription_tier = 'Premium' THEN 0.06 "
        "ELSE 0.09 END) AS total_gross, "
        "BILLING_LEDGERS.gross_charge "
        "FROM VM_INSTANCES JOIN TENANT_PROFILES "
        "ON VM_INSTANCES.tenant_id = TENANT_PROFILES.tenant_id "
        "JOIN BILLING_LEDGERS ON BILLING_LEDGERS.tenant_id = TENANT_PROFILES.tenant_id "
        "GROUP BY VM_INSTANCES.tenant_id, BILLING_LEDGERS.gross_charge"
    )


def _cloud_case_rate_query() -> str:
    return (
        "SELECT TENANT_PROFILES.subscription_tier, "
        "CASE "
        "WHEN TENANT_PROFILES.subscription_tier = 'Enterprise' THEN 0.04 "
        "WHEN TENANT_PROFILES.subscription_tier = 'Premium' THEN 0.06 "
        "ELSE 0.09 END AS computed_rate "
        "FROM TENANT_PROFILES"
    )


def _cloud_discount_query() -> str:
    return (
        "SELECT BILLING_LEDGERS.tenant_id, BILLING_LEDGERS.gross_charge, "
        "BILLING_LEDGERS.discount_applied, BILLING_LEDGERS.net_bill, "
        "TENANT_PROFILES.monthly_budget_cap, "
        "CASE WHEN BILLING_LEDGERS.gross_charge > 5000.00 "
        "THEN MIN(BILLING_LEDGERS.gross_charge * 0.15, 2000.00) "
        "ELSE 0.00 END AS expected_discount, "
        "(BILLING_LEDGERS.gross_charge - BILLING_LEDGERS.discount_applied) AS expected_net_bill "
        "FROM BILLING_LEDGERS JOIN TENANT_PROFILES "
        "ON BILLING_LEDGERS.tenant_id = TENANT_PROFILES.tenant_id"
    )


def _cloud_regional_profit_query() -> str:
    return (
        "SELECT power_costs.region_id, revenue.total_regional_revenue, "
        "power_costs.total_power_cost * 1.5 AS required_revenue "
        "FROM (SELECT COMPUTE_NODES.region_id, SUM(DATA_CENTERS.power_cost_per_hour) * 720 AS total_power_cost "
        "FROM COMPUTE_NODES JOIN DATA_CENTERS ON COMPUTE_NODES.region_id = DATA_CENTERS.region_id "
        "WHERE COMPUTE_NODES.is_active = 1 GROUP BY COMPUTE_NODES.region_id"
        ") AS power_costs JOIN (SELECT COMPUTE_NODES.region_id, "
        "SUM(VM_INSTANCES.runtime_hours * VM_INSTANCES.allocated_cpu * 0.06) AS total_regional_revenue "
        "FROM VM_INSTANCES JOIN COMPUTE_NODES ON VM_INSTANCES.node_id = COMPUTE_NODES.node_id "
        "GROUP BY COMPUTE_NODES.region_id"
        ") AS revenue ON power_costs.region_id = revenue.region_id"
    )


def _cloud_constraint(
    query: str,
    left_operand: str,
    operator: BinaryOperator,
    right_operand: BinaryOperand,
    fact_id: int,
) -> SQLGroundedConstraint:
    return SQLGroundedConstraint(
        state_query=query,
        left_operand=left_operand,
        operator=operator,
        right_operand=right_operand,
        fact_references=[fact_id],
    )


def _inventory_state_query(extra_spaces: bool = False) -> str:
    if extra_spaces:
        return (
            " SELECT  PRODUCT.product_id,  SUM(ORDER_ITEM.quantity) AS total_demanded_quantity, "
            " PRODUCT.stock_quantity FROM ORDER_ITEM JOIN PRODUCT "
            " ON ORDER_ITEM.product_id = PRODUCT.product_id "
            " GROUP BY PRODUCT.product_id, PRODUCT.stock_quantity "
        )
    return (
        "SELECT PRODUCT.product_id, SUM(ORDER_ITEM.quantity) AS total_demanded_quantity, "
        "PRODUCT.stock_quantity FROM ORDER_ITEM JOIN PRODUCT "
        "ON ORDER_ITEM.product_id = PRODUCT.product_id "
        "GROUP BY PRODUCT.product_id, PRODUCT.stock_quantity"
    )


def _inventory_constraint(
    left_operand: str,
    operator: BinaryOperator,
    right_operand: BinaryOperand,
    fact_id: int,
    extra_spaces: bool = False,
) -> SQLGroundedConstraint:
    return SQLGroundedConstraint(
        state_query=_inventory_state_query(extra_spaces=extra_spaces),
        left_operand=left_operand,
        operator=operator,
        right_operand=right_operand,
        fact_references=[fact_id],
    )


def test_numeric_bounds_sat():
    issues = check_state_constraint_satisfiability([
        _constraint("quantity", "GTE", BinaryOperand(kind="literal", value=1)),
        _constraint("quantity", "LTE", BinaryOperand(kind="literal", value=5)),
    ], _schema())

    assert issues == []


def test_numeric_bounds_unsat():
    issues = check_state_constraint_satisfiability([
        _constraint("quantity", "GTE", BinaryOperand(kind="literal", value=10)),
        _constraint("quantity", "LTE", BinaryOperand(kind="literal", value=5)),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_LINEAR_CONSTRAINTS"
    assert issues[0].target == "state_constraints[0]"


def test_text_ordering_is_unsupported():
    issues = check_state_constraint_satisfiability([
        _constraint("status", "GT", BinaryOperand(kind="literal", value="Completed")),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSUPPORTED_NON_NUMERIC_ORDERING"


def test_null_conflict_unsat():
    issues = check_state_constraint_satisfiability([
        _constraint("processed_at", "IS_NULL"),
        _constraint("processed_at", "IS_NOT_NULL"),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_NULL_CONFLICT"


def test_literal_equality_conflict_unsat():
    issues = check_state_constraint_satisfiability([
        _constraint("status", "EQUALS", BinaryOperand(kind="literal", value="Completed")),
        _constraint("status", "EQUALS", BinaryOperand(kind="literal", value="Cancelled")),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_LITERAL_CONFLICT"


def test_column_equality_and_strict_ordering_unsat():
    issues = check_state_constraint_satisfiability([
        _constraint("quantity", "EQUALS", BinaryOperand(kind="column", value="unit_price")),
        _constraint("quantity", "GT", BinaryOperand(kind="column", value="unit_price")),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_LINEAR_CONSTRAINTS"


def test_strict_inequality_cycle_unsat():
    issues = check_state_constraint_satisfiability([
        _constraint("quantity", "GT", BinaryOperand(kind="column", value="unit_price")),
        _constraint("unit_price", "GT", BinaryOperand(kind="column", value="quantity")),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_LINEAR_CONSTRAINTS"


def test_qualified_self_inequality_unsat():
    issues = check_state_constraint_satisfiability([
        _constraint("quantity", "NOT_EQUALS", BinaryOperand(kind="column", value="ORDER_ITEM.quantity")),
    ], _schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_SELF_INEQUALITY"


def test_inventory_aggregate_constraints_across_simulated_shards_unsat():
    query_constraints = [
        _inventory_constraint(
            "total_demanded_quantity",
            "LTE",
            BinaryOperand(kind="column", value="stock_quantity"),
            fact_id=1,
        ),
        _inventory_constraint(
            "total_demanded_quantity",
            "GTE",
            BinaryOperand(kind="literal", value=100),
            fact_id=2,
            extra_spaces=True,
        ),
        _inventory_constraint(
            "stock_quantity",
            "LTE",
            BinaryOperand(kind="literal", value=90),
            fact_id=3,
        ),
    ]
    shard_results = [
        ShardMetadata(
            shard_index=0,
            table_names=["ORDER_ITEM", "PRODUCT"],
            manifests=[
                TableConstraintManifest(
                    table_name="ORDER_ITEM",
                    state_constraints=[query_constraints[0]],
                )
            ],
        ),
        ShardMetadata(
            shard_index=1,
            table_names=["PRODUCT", "ORDER_ITEM"],
            manifests=[
                TableConstraintManifest(
                    table_name="PRODUCT",
                    state_constraints=query_constraints[1:],
                )
            ],
        ),
    ]

    global_manifests = _build_global_table_manifests(shard_results)
    unified_constraints = []
    for manifest in global_manifests:
        unified_constraints.extend(manifest.state_constraints)

    issues = check_state_constraint_satisfiability(unified_constraints, _commerce_schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_LINEAR_CONSTRAINTS"
    assert issues[0].fact_references == [1, 2, 3]


def test_independent_state_tables_report_multiple_issue_types():
    constraints = [
        _inventory_constraint(
            "total_demanded_quantity",
            "GTE",
            BinaryOperand(kind="literal", value=100),
            fact_id=1,
        ),
        _inventory_constraint(
            "total_demanded_quantity",
            "LTE",
            BinaryOperand(kind="literal", value=50),
            fact_id=2,
        ),
        SQLGroundedConstraint(
            state_query="SELECT status, processed_at FROM PAYMENT",
            left_operand="status",
            operator="GT",
            right_operand=BinaryOperand(kind="literal", value="Completed"),
            fact_references=[3],
        ),
        SQLGroundedConstraint(
            state_query="SELECT status, processed_at FROM PAYMENT",
            left_operand="processed_at",
            operator="IS_NULL",
            fact_references=[4],
        ),
        SQLGroundedConstraint(
            state_query="SELECT status, processed_at FROM PAYMENT",
            left_operand="processed_at",
            operator="IS_NOT_NULL",
            fact_references=[5],
        ),
    ]

    issues = check_state_constraint_satisfiability(constraints, _commerce_schema())
    issue_codes = {issue.code for issue in issues}

    assert issue_codes == {
        "UNSAT_LINEAR_CONSTRAINTS",
        "UNSUPPORTED_NON_NUMERIC_ORDERING",
        "UNSAT_NULL_CONFLICT",
    }


def test_structural_cardinality_conflict_unsat():
    issues, knobs = check_structural_satisfiability([
        CardinalityConstraint(table_name="ORDER", min_rows=10, fact_references=[1]),
        CardinalityConstraint(table_name="ORDER", max_rows=5, fact_references=[2]),
    ], [], _order_schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_CARDINALITY_BOUNDS"
    assert knobs


def test_structural_fanout_cardinality_milp_unsat():
    issues, _knobs = check_structural_satisfiability([
        CardinalityConstraint(table_name="ORDER", exact_rows=10, fact_references=[1]),
        CardinalityConstraint(table_name="ORDER_ITEM", max_rows=15, fact_references=[2]),
    ], [
        FanoutConstraint(
            parent_table="ORDER",
            child_table="ORDER_ITEM",
            min_children_per_parent=2,
            fact_references=[3],
        )
    ], _order_schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_STRUCTURAL_CONSTRAINTS"
    assert issues[0].fact_references == [1, 2, 3]


def test_deterministic_knobs_for_unconstrained_root_and_child_fanout():
    issues, knobs = check_structural_satisfiability([], [], _order_schema())

    assert issues == []
    knob_names = {knob.name for knob in knobs}
    assert "order_row_count" in knob_names
    assert "order_to_order_item_fanout" in knob_names


def test_bounded_fanout_knob_when_fact_gives_range_not_exact_value():
    issues, knobs = check_structural_satisfiability([], [
        FanoutConstraint(
            parent_table="ORDER",
            child_table="ORDER_ITEM",
            min_children_per_parent=1,
            max_children_per_parent=5,
            fact_references=[1],
        )
    ], _order_schema())

    assert issues == []
    fanout_knob = next(knob for knob in knobs if knob.kind == "relationship_fanout")
    assert fanout_knob.min_value == 1
    assert fanout_knob.max_value == 5
    assert fanout_knob.source == "bounded"


def test_exact_cardinality_and_exact_fanout_remove_independent_knobs():
    issues, knobs = check_structural_satisfiability([
        CardinalityConstraint(table_name="ORDER", exact_rows=10, fact_references=[1]),
    ], [
        FanoutConstraint(
            parent_table="ORDER",
            child_table="ORDER_ITEM",
            min_children_per_parent=2,
            max_children_per_parent=2,
            fact_references=[2],
        )
    ], _order_schema())

    assert issues == []
    assert knobs == []


def test_cloud_capacity_constraints_are_supported_linear_state_tables():
    issues = check_state_constraint_satisfiability([
        _cloud_constraint(
            _cloud_capacity_query(),
            "total_cpu",
            "LTE",
            BinaryOperand(kind="column", value="max_cpu_cores"),
            fact_id=1,
        ),
        _cloud_constraint(
            _cloud_capacity_query(),
            "total_ram",
            "LTE",
            BinaryOperand(kind="column", value="max_ram_gb"),
            fact_id=2,
        ),
    ], _cloud_schema())

    assert issues == []


def test_cloud_capacity_gridlock_detects_cross_metric_unsat():
    issues = check_state_constraint_satisfiability([
        _cloud_constraint(
            _cloud_capacity_query(),
            "total_cpu",
            "LTE",
            BinaryOperand(kind="column", value="max_cpu_cores"),
            fact_id=1,
        ),
        _cloud_constraint(
            _cloud_capacity_query(),
            "total_cpu",
            "GTE",
            BinaryOperand(kind="literal", value=256),
            fact_id=2,
        ),
        _cloud_constraint(
            _cloud_capacity_query(),
            "max_cpu_cores",
            "LTE",
            BinaryOperand(kind="literal", value=128),
            fact_id=3,
        ),
    ], _cloud_schema())

    assert len(issues) == 1
    assert issues[0].code == "UNSAT_LINEAR_CONSTRAINTS"
    assert issues[0].fact_references == [1, 2, 3]


def test_cloud_piecewise_case_rate_bounds_are_supported():
    issues = check_state_constraint_satisfiability([
        _cloud_constraint(
            _cloud_case_rate_query(),
            "computed_rate",
            "GTE",
            BinaryOperand(kind="literal", value=0.04),
            fact_id=4,
        ),
        _cloud_constraint(
            _cloud_case_rate_query(),
            "computed_rate",
            "LTE",
            BinaryOperand(kind="literal", value=0.09),
            fact_id=5,
        ),
    ], _cloud_schema())

    assert issues == []


def test_cloud_pricing_discount_and_regional_profit_are_explicitly_unsupported_v1():
    constraints = [
        _cloud_constraint(
            _cloud_pricing_query(),
            "total_gross",
            "EQUALS",
            BinaryOperand(kind="column", value="gross_charge"),
            fact_id=10,
        ),
        _cloud_constraint(
            _cloud_discount_query(),
            "discount_applied",
            "EQUALS",
            BinaryOperand(kind="column", value="expected_discount"),
            fact_id=11,
        ),
        _cloud_constraint(
            _cloud_regional_profit_query(),
            "total_regional_revenue",
            "GTE",
            BinaryOperand(kind="column", value="required_revenue"),
            fact_id=12,
        ),
    ]

    issues = check_state_constraint_satisfiability(constraints, _cloud_schema())

    assert len(issues) == 3
    assert {issue.code for issue in issues} == {"UNSUPPORTED_STATE_QUERY_EXPRESSION"}


def test_cloud_structural_knobs_match_expected_independent_parameters():
    issues, knobs = check_structural_satisfiability([], [
        FanoutConstraint(
            parent_table="COMPUTE_NODES",
            child_table="VM_INSTANCES",
            min_children_per_parent=1,
            max_children_per_parent=20,
            fact_references=[20],
        ),
        FanoutConstraint(
            parent_table="TENANT_PROFILES",
            child_table="BILLING_LEDGERS",
            min_children_per_parent=1,
            max_children_per_parent=1,
            fact_references=[21],
        ),
    ], _cloud_schema())

    assert issues == []
    knob_names = {knob.name for knob in knobs}
    assert knob_names == {
        "data_centers_row_count",
        "compute_nodes_row_count",
        "tenant_profiles_row_count",
        "compute_nodes_to_vm_instances_fanout",
    }
    assert "vm_instances_row_count" not in knob_names
    assert "billing_ledgers_row_count" not in knob_names
