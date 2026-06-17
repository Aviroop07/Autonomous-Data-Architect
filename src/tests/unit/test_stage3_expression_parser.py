from __future__ import annotations

from src.pipeline.stage3.middleware.expression_parser import classify_expression, parse_expression


def test_parse_column_reference_with_table_qualifier():
    expression = parse_expression("VM_INSTANCES.allocated_cpu")

    assert expression.kind == "column"
    assert expression.table_name == "VM_INSTANCES"
    assert expression.column_name == "allocated_cpu"


def test_constant_multiplication_is_linear():
    classification = classify_expression("power_costs.total_power_cost * 1.5")

    assert classification.solver_tier == "linear"
    assert classification.variable_factor_count == 1
    assert "arithmetic_product" in classification.features


def test_runtime_cpu_rate_product_is_nonlinear_when_two_factors_are_variables():
    classification = classify_expression("VM_INSTANCES.runtime_hours * VM_INSTANCES.allocated_cpu * 0.06")

    assert classification.solver_tier == "nonlinear_product"
    assert classification.variable_factor_count == 2
    assert "Multiplicative chain contains more than one variable factor." in classification.unsupported_reasons


def test_subscription_tier_case_is_piecewise_linear():
    classification = classify_expression(
        "CASE "
        "WHEN TENANT_PROFILES.subscription_tier = 'Enterprise' THEN 0.04 "
        "WHEN TENANT_PROFILES.subscription_tier = 'Premium' THEN 0.06 "
        "ELSE 0.09 END"
    )

    assert classification.expression.kind == "case_when"
    assert classification.solver_tier == "piecewise_linear"
    assert len(classification.expression.case_branches) == 2
    assert classification.expression.else_result is not None


def test_discount_min_gate_is_big_m_gate_when_arguments_are_linear():
    classification = classify_expression("MIN(BILLING_LEDGERS.gross_charge * 0.15, 2000.00)")

    assert classification.expression.kind == "min"
    assert classification.solver_tier == "big_m_gate"
    assert classification.variable_factor_count == 1


def test_net_bill_subtraction_is_linear():
    classification = classify_expression("BILLING_LEDGERS.gross_charge - BILLING_LEDGERS.discount_applied")

    assert classification.expression.kind == "sub"
    assert classification.solver_tier == "linear"
