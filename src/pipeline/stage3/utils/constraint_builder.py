import re
from typing import Tuple, List, Optional, Union, Dict
from ..models.nodes import (
    BaseTableNode, JoinNode, AggNode, ColumnNode, TableNode, LogicalNode, IfNode
)
from src.pipeline.stage2.models.schema import Schema
from .sql_validator import validate_sql_against_schema
from .logic_parser import parse_logic_rule, LogicParsingError
from .sql_parser_utils import find_balanced_parentheses, resolve_table_aliases

class SQLParsingError(Exception):
    pass

def build_from_llm_input(
    sql_str: str,
    rule_str: str,
    schema: Schema
) -> Tuple[Optional[Union[LogicalNode, IfNode]], Optional[str]]:
    """
    Parses LLM-generated SQL and Rule strings into a structured Algebraic Node.
    Returns: (ConstraintNode, Error_Message)
    """
    # 1. Validate SQL against Mock SQLite
    # This provides immediate feedback on column/table hallucinations
    is_valid, result_columns, sql_error = validate_sql_against_schema(sql_str, schema)
    if not is_valid:
        return None, f"SQL Validation Error: {sql_error}"

    # 2. Extract Relational Context (Decompile SQL to Nodes)
    try:
        table_context, alias_map = _decompile_sql_to_algebraic_context(sql_str, schema)
    except SQLParsingError as e:
        return None, f"Could not map SQL to relational nodes: {str(e)}"
    except Exception as e:
        return None, f"An unexpected error occurred during SQL decompilation: {str(e)}"

    # 3. Parse Logic Rule anchored to the context table
    try:
        logic_node = parse_logic_rule(rule_str, table_context, set(result_columns), alias_map)
        return logic_node, None
    except LogicParsingError as e:
        # Pass back the specific, helpful error with hint
        return None, str(e)
    except Exception as e:
        return None, f"Relational binding failed: {str(e)}"

def _decompile_sql_to_algebraic_context(sql: str, schema: Schema) -> Tuple[TableNode, Dict[str, str]]:
    """
    Recursive decompiler for SQL structures.
    Supports: Base Tables, Joins, and Nested Subqueries.
    Erases: Table Aliases (AS).
    """
    # 1. Normalize SQL
    sql_clean = sql.strip().replace("\n", " ")
    if not sql_clean.upper().startswith("FROM"):
        if not sql_clean.upper().startswith("SELECT"):
            sql_clean = "FROM " + sql_clean
        else:
            # It starts with SELECT, which is fine
            pass

    alias_map = resolve_table_aliases(sql_clean)

    # 2. Handle Subqueries (Top-Down Balanced Matching)
    sub_results = find_balanced_parentheses(sql_clean)
    sq_map = {}
    modified_sql = sql_clean

    # Replace subqueries with internal markers for easier parsing
    for i, (start, end, content) in enumerate(reversed(sub_results)):
        inner_sql = content[1:-1].strip() # Strip outer ( )
        # Recursively build the virtual TableNode for this subquery
        virtual_node, _ = _decompile_sql_to_algebraic_context(inner_sql, schema)

        # Check if this subquery was aliased (e.g., ) AS sub)
        # resolve_table_aliases already has VIRTUAL_SUB in its map
        marker = f"__SQ_{i}__"
        sq_map[marker] = virtual_node

        # We need to find the alias following this subquery (if any)
        # e.g., ) AS sub JOIN ...
        suffix = modified_sql[end:].strip()
        alias_match = re.match(r"(?:AS\s+)?(\w+)\b", suffix, re.IGNORECASE)
        if alias_match:
             alias = alias_match.group(1).upper()
             # Link the alias to our marker
             # This lets us resolve 'sub.col' later back to the marker's columns
             alias_map[alias] = marker
             # We consume the alias so it doesn't confuse the later parser
             offset = alias_match.end()
             # We leave a space to separate from JOIN
             modified_sql = modified_sql[:start] + marker + " " + suffix[offset:]
        else:
             modified_sql = modified_sql[:start] + marker + modified_sql[end:]

    # 3. Parse the flattened SQL text (which now has markers like __SQ_i__)
    # a. Identify FROM
    from_match = re.search(r"FROM\s+(\w+)", modified_sql, re.IGNORECASE)
    if not from_match:
        raise SQLParsingError("The SQL query must contain a 'FROM' clause.")

    root_name = from_match.group(1).upper()
    current_context = _resolve_node(root_name, alias_map, sq_map, schema)

    # b. Identify JOIN Chains
    # Pattern: JOIN table2 [AS alias] ON table1.col1 = table2.col2
    joins = re.findall(
        r"JOIN\s+(\w+)(?:\s+AS\s+\w+)?\s+ON\s+([\w\.]+)\s*=\s*([\w\.]+)",
        modified_sql,
        re.IGNORECASE
    )

    for i, (next_table_name, p1, p2) in enumerate(joins):
        next_name = next_table_name.upper()
        next_node = _resolve_node(next_name, alias_map, sq_map, schema)

        # Extract Local Column Names
        c1 = p1.split('.')[-1]
        c2 = p2.split('.')[-1]

        current_context = JoinNode(
            name=f"INTERNAL_STEP_{i+1}",
            l_table=current_context,
            l_column=ColumnNode(name=c1),
            r_table=next_node,
            r_column=ColumnNode(name=c2)
        )

    return current_context, alias_map

def _resolve_node(name: str, alias_map: dict, sq_map: dict, schema: Schema) -> TableNode:
    """Helper to resolve a name to a physical table OR a pre-parsed subquery."""
    # 1. Check if it's an alias
    actual_name = alias_map.get(name, name)

    # 2. Check if it's a subquery marker
    if actual_name in sq_map:
        return sq_map[actual_name]

    # 3. Assume it's a base table
    return BaseTableNode(name=actual_name)
