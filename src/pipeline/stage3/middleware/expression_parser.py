from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.pipeline.stage3.models.expressions import (
    CaseBranch,
    ExpressionClassification,
    ExpressionNode,
    PredicateNode,
    SolverTier,
)
from src.pipeline.stage3.models.sql_models import BinaryOperator, OperandValue


def parse_expression(expression_text: str) -> ExpressionNode:
    expression = _strip_outer_parentheses(expression_text.strip())
    if not expression:
        return ExpressionNode(kind="unsupported", text=expression_text)

    if _starts_with_keyword(expression, "CASE"):
        return _parse_case_expression(expression)

    function_node = _parse_min_max_function(expression)
    if function_node:
        return function_node

    additive_split = _split_top_level_binary(expression, ["+", "-"])
    if additive_split:
        left_text, operator, right_text = additive_split
        return ExpressionNode(
            kind="add" if operator == "+" else "sub",
            text=expression,
            left=parse_expression(left_text),
            right=parse_expression(right_text),
        )

    multiplicative_split = _split_top_level_binary(expression, ["*", "/"])
    if multiplicative_split:
        left_text, operator, right_text = multiplicative_split
        return ExpressionNode(
            kind="mul" if operator == "*" else "div",
            text=expression,
            left=parse_expression(left_text),
            right=parse_expression(right_text),
        )

    literal_value = _parse_literal(expression)
    if literal_value is not None or expression.upper() == "NULL":
        return ExpressionNode(kind="literal", text=expression, literal_value=literal_value)

    column = _parse_column_ref(expression)
    if column:
        table_name, column_name = column
        return ExpressionNode(kind="column", text=expression, table_name=table_name, column_name=column_name)

    return ExpressionNode(kind="unsupported", text=expression)


def classify_expression(expression_text: str) -> ExpressionClassification:
    expression = parse_expression(expression_text)
    features: List[str] = []
    unsupported_reasons: List[str] = []
    _collect_features(expression, features)
    variable_factor_count = _max_variable_factor_count(expression)
    solver_tier = _classify_solver_tier(expression, variable_factor_count, unsupported_reasons)
    return ExpressionClassification(
        expression=expression,
        solver_tier=solver_tier,
        variable_factor_count=variable_factor_count,
        features=features,
        unsupported_reasons=unsupported_reasons,
    )


def expression_is_supported_linear_or_constant(expression_text: str) -> bool:
    classification = classify_expression(expression_text)
    return classification.solver_tier == "linear"


def expression_requires_unsupported_v1(expression_text: str) -> List[str]:
    classification = classify_expression(expression_text)
    reasons: List[str] = []
    if classification.solver_tier in {"piecewise_linear", "big_m_gate", "groundable_product"}:
        reasons.append(f"{classification.solver_tier} expression requires a later solver tier")
    reasons.extend(classification.unsupported_reasons)
    return reasons


def _parse_case_expression(expression: str) -> ExpressionNode:
    body = expression.strip()[4:].strip()
    if body.upper().endswith("END"):
        body = body[:-3].strip()

    branches: List[CaseBranch] = []
    else_result: Optional[ExpressionNode] = None
    remaining = body
    while _starts_with_keyword(remaining, "WHEN"):
        then_idx = _find_keyword_top_level(remaining, "THEN")
        if then_idx is None:
            return ExpressionNode(kind="unsupported", text=expression)
        condition_text = remaining[4:then_idx].strip()
        after_then = remaining[then_idx + 4:].strip()
        next_when = _find_keyword_top_level(after_then, "WHEN")
        next_else = _find_keyword_top_level(after_then, "ELSE")
        boundary = _first_boundary(next_when, next_else)
        if boundary is None:
            result_text = after_then.strip()
            remaining = ""
        else:
            result_text = after_then[:boundary].strip()
            remaining = after_then[boundary:].strip()

        condition = _parse_predicate(condition_text)
        if condition is None:
            return ExpressionNode(kind="unsupported", text=expression)
        branches.append(CaseBranch(condition=condition, result=parse_expression(result_text)))

    if _starts_with_keyword(remaining, "ELSE"):
        else_result = parse_expression(remaining[4:].strip())
    elif remaining:
        return ExpressionNode(kind="unsupported", text=expression)

    return ExpressionNode(
        kind="case_when",
        text=expression,
        case_branches=branches,
        else_result=else_result,
    )


def _parse_min_max_function(expression: str) -> Optional[ExpressionNode]:
    upper = expression.upper()
    for function_name, kind in [("MIN", "min"), ("MAX", "max")]:
        prefix = f"{function_name}("
        if upper.startswith(prefix) and expression.endswith(")"):
            inner = expression[len(prefix):-1]
            args = [_arg.strip() for _arg in _split_top_level_commas(inner)]
            return ExpressionNode(
                kind=kind,  # type: ignore[arg-type]
                text=expression,
                arguments=[parse_expression(arg) for arg in args if arg],
            )
    return None


def _parse_predicate(condition_text: str) -> Optional[PredicateNode]:
    predicate_split = _split_top_level_predicate(condition_text)
    if predicate_split is None:
        return None
    left_text, operator, right_text = predicate_split
    return PredicateNode(
        left=parse_expression(left_text),
        operator=operator,
        right=parse_expression(right_text),
    )


def _split_top_level_predicate(condition_text: str) -> Optional[Tuple[str, BinaryOperator, str]]:
    operators = [">=", "<=", "!=", "==", "=", ">", "<"]
    for operator in operators:
        split_result = _split_top_level_token(condition_text, operator)
        if split_result:
            left, right = split_result
            return left, _map_predicate_operator(operator), right
    return None


def _map_predicate_operator(operator: str) -> BinaryOperator:
    if operator in {"=", "=="}:
        return "EQUALS"
    if operator == "!=":
        return "NOT_EQUALS"
    if operator == ">":
        return "GT"
    if operator == ">=":
        return "GTE"
    if operator == "<":
        return "LT"
    return "LTE"


def _split_top_level_binary(expression: str, operators: List[str]) -> Optional[Tuple[str, str, str]]:
    depth = 0
    in_quote = False
    for idx in range(len(expression) - 1, -1, -1):
        char = expression[idx]
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == ")":
            depth += 1
            continue
        if char == "(":
            depth -= 1
            continue
        if depth == 0 and char in operators and idx > 0:
            previous = expression[idx - 1]
            if char in {"+", "-"} and previous in {"*", "/", "+", "-", "("}:
                continue
            return expression[:idx].strip(), char, expression[idx + 1:].strip()
    return None


def _split_top_level_token(expression: str, token: str) -> Optional[Tuple[str, str]]:
    depth = 0
    in_quote = False
    idx = 0
    while idx <= len(expression) - len(token):
        char = expression[idx]
        if char == "'":
            in_quote = not in_quote
            idx += 1
            continue
        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and expression[idx:idx + len(token)] == token:
                return expression[:idx].strip(), expression[idx + len(token):].strip()
        idx += 1
    return None


def _split_top_level_commas(expression: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    in_quote = False
    start = 0
    for idx, char in enumerate(expression):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(expression[start:idx])
            start = idx + 1
    parts.append(expression[start:])
    return parts


def _strip_outer_parentheses(expression: str) -> str:
    current = expression.strip()
    while current.startswith("(") and current.endswith(")") and _outer_parentheses_wrap_all(current):
        current = current[1:-1].strip()
    return current


def _outer_parentheses_wrap_all(expression: str) -> bool:
    depth = 0
    in_quote = False
    for idx, char in enumerate(expression):
        if char == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and idx != len(expression) - 1:
                return False
    return depth == 0


def _find_keyword_top_level(expression: str, keyword: str) -> Optional[int]:
    depth = 0
    in_quote = False
    upper = expression.upper()
    keyword_upper = keyword.upper()
    idx = 0
    while idx <= len(expression) - len(keyword):
        char = expression[idx]
        if char == "'":
            in_quote = not in_quote
            idx += 1
            continue
        if not in_quote:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif depth == 0 and upper[idx:idx + len(keyword)] == keyword_upper:
                before_ok = idx == 0 or not upper[idx - 1].isalnum()
                after_idx = idx + len(keyword)
                after_ok = after_idx == len(upper) or not upper[after_idx].isalnum()
                if before_ok and after_ok:
                    return idx
        idx += 1
    return None


def _starts_with_keyword(expression: str, keyword: str) -> bool:
    stripped = expression.strip()
    upper = stripped.upper()
    keyword_upper = keyword.upper()
    if not upper.startswith(keyword_upper):
        return False
    return len(upper) == len(keyword_upper) or not upper[len(keyword_upper)].isalnum()


def _first_boundary(first: Optional[int], second: Optional[int]) -> Optional[int]:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _parse_literal(expression: str) -> Optional[OperandValue]:
    stripped = expression.strip()
    upper = stripped.upper()
    if upper == "NULL":
        return None
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if len(stripped) >= 2 and stripped.startswith("'") and stripped.endswith("'"):
        return stripped[1:-1]
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return None


def _parse_column_ref(expression: str) -> Optional[Tuple[Optional[str], str]]:
    stripped = expression.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", stripped):
        return None
    if "." in stripped:
        table_name, column_name = stripped.split(".", 1)
        return table_name, column_name
    return None, stripped


def _collect_features(expression: ExpressionNode, features: List[str]) -> None:
    _append_feature(features, expression.kind)
    if expression.kind in {"mul", "div"}:
        _append_feature(features, "arithmetic_product" if expression.kind == "mul" else "arithmetic_division")
    if expression.kind == "case_when":
        _append_feature(features, "case_when")
    if expression.kind in {"min", "max"}:
        _append_feature(features, "scalar_min_max")
    if expression.left:
        _collect_features(expression.left, features)
    if expression.right:
        _collect_features(expression.right, features)
    for argument in expression.arguments:
        _collect_features(argument, features)
    for branch in expression.case_branches:
        _collect_features(branch.condition.left, features)
        _collect_features(branch.condition.right, features)
        _collect_features(branch.result, features)
    if expression.else_result:
        _collect_features(expression.else_result, features)


def _append_feature(features: List[str], feature: str) -> None:
    if feature not in features:
        features.append(feature)


def _max_variable_factor_count(expression: ExpressionNode) -> int:
    if expression.kind == "mul":
        factors = _flatten_multiplication(expression)
        count = sum(1 for factor in factors if factor.kind != "literal")
        nested_max = max((_max_variable_factor_count(factor) for factor in factors), default=0)
        return max(count, nested_max)
    child_counts: List[int] = []
    if expression.left:
        child_counts.append(_max_variable_factor_count(expression.left))
    if expression.right:
        child_counts.append(_max_variable_factor_count(expression.right))
    for argument in expression.arguments:
        child_counts.append(_max_variable_factor_count(argument))
    for branch in expression.case_branches:
        child_counts.append(_max_variable_factor_count(branch.condition.left))
        child_counts.append(_max_variable_factor_count(branch.condition.right))
        child_counts.append(_max_variable_factor_count(branch.result))
    if expression.else_result:
        child_counts.append(_max_variable_factor_count(expression.else_result))
    return max(child_counts, default=0)


def _flatten_multiplication(expression: ExpressionNode) -> List[ExpressionNode]:
    if expression.kind != "mul":
        return [expression]
    factors: List[ExpressionNode] = []
    if expression.left:
        factors.extend(_flatten_multiplication(expression.left))
    if expression.right:
        factors.extend(_flatten_multiplication(expression.right))
    return factors


def _classify_solver_tier(
    expression: ExpressionNode,
    variable_factor_count: int,
    unsupported_reasons: List[str],
) -> SolverTier:
    if expression.kind == "unsupported":
        unsupported_reasons.append("Expression could not be parsed into the supported IR.")
        return "unsupported"
    if variable_factor_count > 1:
        unsupported_reasons.append("Multiplicative chain contains more than one variable factor.")
        return "nonlinear_product"
    if expression.kind == "case_when" or _contains_kind(expression, "case_when"):
        if _case_results_are_linear(expression):
            return "piecewise_linear"
        unsupported_reasons.append("CASE expression has non-linear branch results.")
        return "unsupported"
    if expression.kind in {"min", "max"} or _contains_min_max(expression):
        if _min_max_arguments_are_linear(expression):
            return "big_m_gate"
        unsupported_reasons.append("MIN/MAX expression has non-linear arguments.")
        return "unsupported"
    if _contains_kind(expression, "div") and not _division_is_by_literal(expression):
        unsupported_reasons.append("Division by a non-literal expression is not linear.")
        return "unsupported"
    return "linear"


def _contains_kind(expression: ExpressionNode, kind: str) -> bool:
    if expression.kind == kind:
        return True
    children = _expression_children(expression)
    return any(_contains_kind(child, kind) for child in children)


def _contains_min_max(expression: ExpressionNode) -> bool:
    return _contains_kind(expression, "min") or _contains_kind(expression, "max")


def _expression_children(expression: ExpressionNode) -> List[ExpressionNode]:
    children: List[ExpressionNode] = []
    if expression.left:
        children.append(expression.left)
    if expression.right:
        children.append(expression.right)
    children.extend(expression.arguments)
    for branch in expression.case_branches:
        children.append(branch.condition.left)
        children.append(branch.condition.right)
        children.append(branch.result)
    if expression.else_result:
        children.append(expression.else_result)
    return children


def _case_results_are_linear(expression: ExpressionNode) -> bool:
    if expression.kind == "case_when":
        for branch in expression.case_branches:
            if classify_expression(branch.result.text).solver_tier != "linear":
                return False
        if expression.else_result and classify_expression(expression.else_result.text).solver_tier != "linear":
            return False
    return all(_case_results_are_linear(child) for child in _expression_children(expression))


def _min_max_arguments_are_linear(expression: ExpressionNode) -> bool:
    if expression.kind in {"min", "max"}:
        for argument in expression.arguments:
            if classify_expression(argument.text).solver_tier != "linear":
                return False
    return all(_min_max_arguments_are_linear(child) for child in _expression_children(expression))


def _division_is_by_literal(expression: ExpressionNode) -> bool:
    if expression.kind == "div" and expression.right and expression.right.kind != "literal":
        return False
    return all(_division_is_by_literal(child) for child in _expression_children(expression))
