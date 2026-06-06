import re
from typing import List, Optional, Union, Any, Dict, Set
from ..models.nodes import (
    LogicalNode, IfNode, ConditionPair, ColumnNode, ConstNode, TableNode, CombinationNode
)

class LogicParsingError(Exception):
    def __init__(self, message: str, fragment: str = None, hint: str = None):
        self.message = message
        self.fragment = fragment
        self.hint = hint
        error_msg = f"Parsing Error: {message}"
        if fragment: error_msg += f"\n  At: '{fragment}'"
        if hint: error_msg += f"\n  Hint: {hint}"
        super().__init__(error_msg)

def parse_logic_rule(rule_str: str, anchor_table: TableNode, available_columns: Set[str], alias_map: Dict[str, str] = None) -> Union[LogicalNode, IfNode, CombinationNode]:
    if alias_map is None: alias_map = {}
    rule_str = rule_str.strip()

    # 1. Check for IF...THEN pattern
    if rule_str.upper().startswith("IF "):
        match = re.search(r"IF\s+(.*?)\s+THEN\s+(.*)", rule_str, re.IGNORECASE)
        if not match:
             raise LogicParsingError("Malformed 'IF...THEN' statement.", rule_str)

        cond_str, res_str = match.groups()
        cond = _parse_logic_expression(cond_str, anchor_table, available_columns, alias_map)
        res = _parse_logic_expression(res_str, anchor_table, available_columns, alias_map)
        return IfNode(anchor_table=anchor_table, pairs=[ConditionPair(condition=cond, result=res)])

    # 2. Parse as a complex expression
    return _parse_logic_expression(rule_str, anchor_table, available_columns, alias_map)

def _parse_logic_expression(expr_str: str, table: TableNode, cols: Set[str], aliases: Dict[str, str]) -> Union[LogicalNode, CombinationNode]:
    """Handles OR combinations."""
    # Split by OR (case-insensitive, whole word)
    parts = re.split(r"\bOR\b", expr_str, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        operands = [_parse_conjunction(p, table, cols, aliases) for p in parts]
        return CombinationNode(operator="OR", operands=operands, table=table)

    return _parse_conjunction(parts[0], table, cols, aliases)

def _parse_conjunction(conj_str: str, table: TableNode, cols: Set[str], aliases: Dict[str, str]) -> Union[LogicalNode, CombinationNode]:
    """Handles AND combinations."""
    parts = re.split(r"\bAND\b", conj_str, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        operands = [_parse_atom(p, table, cols, aliases) for p in parts]
        return CombinationNode(operator="AND", operands=operands, table=table)

    return _parse_atom(parts[0], table, cols, aliases)

def _parse_atom(atom_str: str, table: TableNode, columns: Set[str], alias_map: Dict[str, str]) -> Union[LogicalNode, CombinationNode]:
    atom_str = atom_str.strip()
    upper_cols = {c.upper() for c in columns}

    # 1. Handle IS NOT NULL / IS NULL
    null_match = re.match(r"([\w\.]+)\s+(IS\s+NOT\s+NULL|IS\s+NULL)", atom_str, re.IGNORECASE)
    if null_match:
        col_raw, op_type = null_match.groups()
        col_name = col_raw.split('.')[-1]
        op = "IS_NOT_NULL" if "NOT" in op_type.upper() else "IS_NULL"

        if col_name.upper() not in upper_cols:
            raise LogicParsingError(f"Column '{col_name}' not found.", col_name)
        canonical_name = next(c for c in columns if c.upper() == col_name.upper())

        return LogicalNode(
            operator=op, table=table,
            column_1=ColumnNode(name=canonical_name),
            column_2=ConstNode(value=None)
        )

    # 2. Handle IN ('val1', 'val2', ...)
    in_match = re.match(r"([\w\.]+)\s+IN\s*\((.*?)\)", atom_str, re.IGNORECASE)
    if in_match:
        col_raw, vals_raw = in_match.groups()
        col_name = col_raw.split('.')[-1]
        if col_name.upper() not in upper_cols:
            raise LogicParsingError(f"Column '{col_name}' not found.", col_name)
        canonical_name = next(c for c in columns if c.upper() == col_name.upper())

        # Split by comma but respect quotes (very simple for now)
        vals = [v.strip().strip("'").strip('"') for v in vals_raw.split(",")]

        operands = []
        for v in vals:
            try:
                v_val = float(v) if '.' in v else int(v)
            except ValueError:
                v_val = v
            operands.append(LogicalNode(
                operator="EQUALS", table=table,
                column_1=ColumnNode(name=canonical_name),
                column_2=ConstNode(value=v_val)
            ))

        if len(operands) == 1: return operands[0]
        return CombinationNode(operator="OR", operands=operands, table=table)

    # 3. Handle BETWEEN X AND Y
    between_match = re.match(r"([\w\.]+)\s+BETWEEN\s+(.*?)\s+AND\s+(.*)", atom_str, re.IGNORECASE)
    if between_match:
        col_raw, val1_str, val2_str = between_match.groups()
        col_name = col_raw.split('.')[-1]
        if col_name.upper() not in upper_cols:
            raise LogicParsingError(f"Column '{col_name}' not found.", col_name)
        canonical_name = next(c for c in columns if c.upper() == col_name.upper())

        def _parse_val(v):
            v = v.strip().strip("'").strip('"')
            try: return float(v) if '.' in v else int(v)
            except: return v

        val1 = _parse_val(val1_str)
        val2 = _parse_val(val2_str)

        return CombinationNode(operator="AND", table=table, operands=[
            LogicalNode(operator="GTE", table=table, column_1=ColumnNode(name=canonical_name), column_2=ConstNode(value=val1)),
            LogicalNode(operator="LTE", table=table, column_1=ColumnNode(name=canonical_name), column_2=ConstNode(value=val2))
        ])

    # 4. General Comparison Operators (Updated to support words and constants)
    ops_pattern = r"==|!=|>=|<=|>|<|=|EQUALS|NOT_EQUALS|GTE|LTE|GT|LT"
    match = re.search(f"^(.*?)\\s+({ops_pattern})\\s+(.*)$", atom_str, re.IGNORECASE)
    if not match:
        # Fallback for symbols without mandatory spaces
        match = re.search(r"^(.*?)(==|!=|>=|<=|>|<|=)(.*)$", atom_str)

    if not match:
        raise LogicParsingError("Expression does not match a recognizable predicate.", atom_str)

    lhs_raw, op_raw, rhs_raw = match.groups()
    lhs_raw = lhs_raw.strip()
    op_raw = op_raw.strip().upper()
    rhs_raw = rhs_raw.strip().strip("'").strip('"')

    op_map = {
        "=": "EQUALS", "==": "EQUALS", "EQUALS": "EQUALS",
        "!=": "NOT_EQUALS", "NOT_EQUALS": "NOT_EQUALS",
        ">": "GT", "GT": "GT",
        "<": "LT", "LT": "LT",
        ">=": "GTE", "GTE": "GTE",
        "<=": "LTE", "LTE": "LTE"
    }
    op = op_map.get(op_raw)

    # Process LHS
    lhs_clean = lhs_raw.split('.')[-1]
    if lhs_clean.upper() in upper_cols:
        canonical_lhs = next(c for c in columns if c.upper() == lhs_clean.upper())
        lhs_node = ColumnNode(name=canonical_lhs)
    else:
        try:
            val_lhs = float(lhs_raw) if '.' in lhs_raw else int(lhs_raw)
        except ValueError:
            val_lhs = lhs_raw
        lhs_node = ConstNode(value=val_lhs)

    # Process RHS
    rhs_clean = rhs_raw.split('.')[-1]
    if rhs_clean.upper() in upper_cols:
        canonical_rhs = next(c for c in columns if c.upper() == rhs_clean.upper())
        rhs_node = ColumnNode(name=canonical_rhs)
    else:
        try:
            val_rhs = float(rhs_raw) if '.' in rhs_raw else int(rhs_raw)
        except ValueError:
            val_rhs = rhs_raw
        rhs_node = ConstNode(value=val_rhs)

    return LogicalNode(
        operator=op, table=table,
        column_1=lhs_node,
        column_2=rhs_node
    )
