import re
from typing import Dict, List, Tuple

def find_balanced_parentheses(text: str) -> List[Tuple[int, int, str]]:
    """
    Finds all top-level balanced parenthesized expressions.
    Returns: List of (start_index, end_index, content)
    """
    results = []
    stack = []
    for i, char in enumerate(text):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start = stack.pop()
                if not stack: # Only top-level
                    results.append((start, i + 1, text[start:i+1]))
    return results

def resolve_table_aliases(sql: str) -> Dict[str, str]:
    """
    Scans FROM and JOIN clauses for 'Table AS Alias' or 'Table Alias'.
    Returns: Dict mapping Alias -> OriginalName
    """
    aliases = {}

    # 1. Standard pattern: FROM/JOIN <Table> [AS] <Alias>
    # We ignore standard SQL keywords to avoid false positives
    keywords = {"ON", "WHERE", "GROUP", "ORDER", "JOIN", "LIMIT", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "NATURAL", "FULL", "USING"}

    # Regex for Table + Alias (optional AS)
    # Matches: FROM Table Alias, JOIN Table AS Alias, etc.
    matches = re.finditer(r"\b(FROM|JOIN)\s+(\w+)(?:\s+AS)?\s+(\w+)\b", sql, re.IGNORECASE)
    for m in matches:
        full_match = m.group(0)
        table = m.group(2).upper()
        alias = m.group(3).upper()

        if alias not in keywords:
            aliases[alias] = table

    # 2. Subquery Aliases: (SELECT ...) [AS] Alias
    # Matches at the end of a parenthesized block
    sq_matches = re.finditer(r"\)\s*(?:AS\s+)?(\w+)\b", sql, re.IGNORECASE)
    for m in sq_matches:
        alias = m.group(1).upper()
        if alias not in keywords:
             # We don't know the table yet (it's virtual), but we map the alias to a marker later
             aliases[alias] = f"VIRTUAL_{alias}"

    return aliases
