from __future__ import annotations

from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from enum import Enum
import sqlglot
from sqlglot import exp


class Operator(str, Enum):
    EQ = "EQ"
    NEQ = "NEQ"
    IN = "IN"


class ForkKey(BaseModel):
    table_name: str
    column_name: str

    def __hash__(self):
        return hash((self.table_name, self.column_name))

    def to_string(self) -> str:
        return f"{self.table_name}.{self.column_name}"


class BranchCondition(BaseModel):
    fork_key: ForkKey
    operator: Operator
    values: List[str]


class Unresolved(BaseModel):
    """Explicit 'we don't know yet' signal -- never silently guessed."""

    reason: str


class ForkKeyRegistry:
    def __init__(self):
        self.forks: Dict[ForkKey, List[str]] = {}

    def register_fork(self, fork_key: ForkKey, categories: List[str]) -> None:
        """Union categories from ONE source -- deduplicated, order-preserving."""
        existing = self.forks.setdefault(fork_key, [])
        seen = set(existing)
        for cat in categories:
            if cat not in seen:
                existing.append(cat)
                seen.add(cat)

    def scan_and_register_all(
        self, categorical_facts: list[tuple[ForkKey, list[str]]]
    ) -> None:
        """Pass 1: union categories from EVERY matching fact -- no break, no
        first-match-wins.  Call this ONCE with all available facts BEFORE
        calling resolve() on anything."""
        for fork_key, categories in categorical_facts:
            self.register_fork(fork_key, categories)

    def is_fully_known(self, fork_key: ForkKey) -> bool:
        return fork_key in self.forks

    def get_branches_for_condition(
        self, condition: BranchCondition
    ) -> Union[List[str], Unresolved]:
        """Resolve a condition against the registry.

        EQ/IN: the literal values named ARE the branch, independent of
        whether the full category list is known -- always safe to answer.

        NEQ: requires the full category list to compute 'everything except
        these values'; returns Unresolved when the list isn't known yet
        (never guess -- the old code returned the excluded values as if
        they were the branch, which is backwards).

        Unknown operator: Unresolved.
        """
        if condition.operator in (Operator.EQ, Operator.IN):
            if not self.is_fully_known(condition.fork_key):
                return list(condition.values)
            all_cats = self.forks[condition.fork_key]
            return [v for v in all_cats if v in condition.values]

        if condition.operator == Operator.NEQ:
            if not self.is_fully_known(condition.fork_key):
                return Unresolved(
                    reason=(
                        f"NEQ condition on {condition.fork_key.table_name}."
                        f"{condition.fork_key.column_name} needs the full "
                        "category list to compute 'everything except these "
                        "values', and no CategoricalDistribution fact for "
                        "this column has been registered yet."
                    )
                )
            all_cats = self.forks[condition.fork_key]
            return [v for v in all_cats if v not in condition.values]

        return Unresolved(reason=f"Unknown operator: {condition.operator}")


def parse_if_condition(condition_str: str) -> Optional[BranchCondition]:
    try:
        parsed = sqlglot.parse_one(condition_str)
    except Exception:
        return None

    if isinstance(parsed, (exp.EQ, exp.NEQ)):
        left = parsed.left
        right = parsed.right

        if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
            table = left.args.get("table")
            col = left.args.get("this")
            if not table or not col:
                return None

            table_name = table.name.upper()
            col_name = col.name.lower()
            val = right.name

            op = Operator.EQ if isinstance(parsed, exp.EQ) else Operator.NEQ
            return BranchCondition(
                fork_key=ForkKey(table_name=table_name, column_name=col_name),
                operator=op,
                values=[val],
            )
    elif isinstance(parsed, exp.In):
        left = parsed.this
        expressions = parsed.expressions
        if isinstance(left, exp.Column) and all(
            isinstance(e, exp.Literal) for e in expressions
        ):
            table = left.args.get("table")
            col = left.args.get("this")
            if not table or not col:
                return None

            table_name = table.name.upper()
            col_name = col.name.lower()
            values = [e.name for e in expressions]

            return BranchCondition(
                fork_key=ForkKey(table_name=table_name, column_name=col_name),
                operator=Operator.IN,
                values=values,
            )
    return None
