from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union
from src.pipeline.stage2.models.schema import Schema
from src.util.schema_ops.sql_utils import check_sql_queryability
from src.pipeline.stage3.utils.sql_validator import validate_sql_against_schema


BinaryOperator = Literal[
    "EQUALS",
    "NOT_EQUALS",
    "GT",
    "GTE",
    "LT",
    "LTE",
    "IS_NULL",
    "IS_NOT_NULL",
]

OperandValue = Union[str, int, float, bool, None]


class BinaryOperand(BaseModel):
    """Right-hand side of a state-table binary predicate."""

    kind: Literal["column", "literal"] = Field(
        description="Whether value names a state-query output column or is a literal constant."
    )
    value: OperandValue = Field(description="Column name or literal value used by the predicate.")


def _normalize_result_columns(columns: List[str]) -> set[str]:
    normalized = {column.lower() for column in columns}
    normalized.update(column.split(".")[-1].lower() for column in columns)
    return normalized

class SQLGroundedConstraint(BaseModel):
    """
    A SQL state table plus one binary predicate over that state table.
    """

    state_query: str = Field(description="SQL SELECT statement that materializes the state table for this constraint.")
    left_operand: str = Field(description="Output column from state_query used as the left operand.")
    operator: BinaryOperator = Field(description="Binary operator applied to the state table operands.")
    right_operand: Optional[BinaryOperand] = Field(
        default=None,
        description="Right operand. Required except for IS_NULL and IS_NOT_NULL.",
    )
    fact_references: List[int] = Field(default_factory=list, description="IDs of the business facts supporting this constraint.")

    def _validate(self, schema: Schema) -> List[str]:
        errors = []
        if not check_sql_queryability(schema, self.state_query):
            errors.append(f"SQL execution failed for state_query: {self.state_query}")
            return errors

        is_valid, result_cols, err_msg = validate_sql_against_schema(self.state_query, schema)
        if not is_valid:
            errors.append(f"SQL reference error in state_query: {err_msg}")
            return errors

        result_cols_lower = _normalize_result_columns(result_cols)
        left = self.left_operand.split(".")[-1].lower()
        if left not in result_cols_lower:
            errors.append(
                f"Left operand column '{self.left_operand}' was not found in state_query output. "
                f"Available columns: {', '.join(result_cols)}"
            )

        if self.operator in {"IS_NULL", "IS_NOT_NULL"}:
            return errors

        if self.right_operand is None:
            errors.append(f"Operator '{self.operator}' requires a right_operand.")
            return errors

        if self.right_operand.kind == "column":
            if not isinstance(self.right_operand.value, str):
                errors.append("Column right_operand must be a string column name.")
            else:
                right = self.right_operand.value.split(".")[-1].lower()
                if right not in result_cols_lower:
                    errors.append(
                        f"Right operand column '{self.right_operand.value}' was not found in state_query output. "
                        f"Available columns: {', '.join(result_cols)}"
                    )

        return errors

    def get_signature(self) -> str:
        right = "NULL" if self.right_operand is None else f"{self.right_operand.kind}:{self.right_operand.value}"
        return f"StateConstraint:{self.state_query}:{self.left_operand}:{self.operator}:{right}"


class CardinalityConstraint(BaseModel):
    """Bounded table cardinality requirement for structural satisfiability."""

    table_name: str = Field(description="Table whose row count is constrained.")
    min_rows: Optional[int] = Field(default=None, description="Inclusive lower bound on row count.")
    max_rows: Optional[int] = Field(default=None, description="Inclusive upper bound on row count.")
    exact_rows: Optional[int] = Field(default=None, description="Exact row count if known.")
    fact_references: List[int] = Field(default_factory=list, description="IDs of the business facts supporting this structural constraint.")

    def _validate(self, schema: Schema) -> List[str]:
        errors = []
        table_names = {table.name.upper() for table in schema.tables}
        if self.table_name.upper() not in table_names:
            errors.append(f"Table '{self.table_name}' not found in schema.")

        for field_name, value in (
            ("min_rows", self.min_rows),
            ("max_rows", self.max_rows),
            ("exact_rows", self.exact_rows),
        ):
            if value is not None and value < 0:
                errors.append(f"{field_name} for table '{self.table_name}' must be non-negative.")

        lower, upper = self.bounds()
        if lower is not None and upper is not None and lower > upper:
            errors.append(
                f"Cardinality bounds for table '{self.table_name}' are inconsistent: min {lower} > max {upper}."
            )
        return errors

    def bounds(self) -> tuple[Optional[int], Optional[int]]:
        lower = self.min_rows
        upper = self.max_rows
        if self.exact_rows is not None:
            lower = self.exact_rows if lower is None else max(lower, self.exact_rows)
            upper = self.exact_rows if upper is None else min(upper, self.exact_rows)
        return lower, upper

    def get_signature(self) -> str:
        return f"Cardinality:{self.table_name}:{self.min_rows}:{self.max_rows}:{self.exact_rows}"


class FanoutConstraint(BaseModel):
    """Bounded child-per-parent structural routing requirement."""

    parent_table: str = Field(description="Parent table on the one-side of the relationship.")
    child_table: str = Field(description="Child table containing the foreign key.")
    min_children_per_parent: Optional[int] = Field(default=None, description="Inclusive lower bound per parent row.")
    max_children_per_parent: Optional[int] = Field(default=None, description="Inclusive upper bound per parent row.")
    fact_references: List[int] = Field(default_factory=list, description="IDs of the business facts supporting this structural constraint.")

    def _validate(self, schema: Schema) -> List[str]:
        errors = []
        table_names = {table.name.upper() for table in schema.tables}
        parent = self.parent_table.upper()
        child = self.child_table.upper()
        if parent not in table_names:
            errors.append(f"Parent table '{self.parent_table}' not found in schema.")
        if child not in table_names:
            errors.append(f"Child table '{self.child_table}' not found in schema.")

        if parent in table_names and child in table_names:
            has_fk = any(
                rel.referencing_table.upper() == child and rel.referred_table.upper() == parent
                for rel in (schema.relationships or [])
            )
            if not has_fk:
                errors.append(
                    f"No foreign key relationship found from child '{self.child_table}' to parent '{self.parent_table}'."
                )

        for field_name, value in (
            ("min_children_per_parent", self.min_children_per_parent),
            ("max_children_per_parent", self.max_children_per_parent),
        ):
            if value is not None and value < 0:
                errors.append(f"{field_name} for {self.parent_table}->{self.child_table} must be non-negative.")

        if (
            self.min_children_per_parent is not None
            and self.max_children_per_parent is not None
            and self.min_children_per_parent > self.max_children_per_parent
        ):
            errors.append(
                f"Fanout bounds for {self.parent_table}->{self.child_table} are inconsistent: "
                f"min {self.min_children_per_parent} > max {self.max_children_per_parent}."
            )

        if self.min_children_per_parent is None and self.max_children_per_parent is None:
            errors.append(
                f"Fanout constraint for {self.parent_table}->{self.child_table} must define at least one bound."
            )
        return errors

    def get_signature(self) -> str:
        return (
            f"Fanout:{self.parent_table}:{self.child_table}:"
            f"{self.min_children_per_parent}:{self.max_children_per_parent}"
        )


class StructuralKnob(BaseModel):
    """A structural degree of freedom left unspecified by facts."""

    kind: Literal["table_cardinality", "relationship_fanout"] = Field(
        description="Whether this knob controls table row count or FK fanout."
    )
    name: str = Field(description="Stable human-readable knob name.")
    table_name: Optional[str] = Field(default=None, description="Table controlled by a table-cardinality knob.")
    parent_table: Optional[str] = Field(default=None, description="Parent table for a fanout knob.")
    child_table: Optional[str] = Field(default=None, description="Child table for a fanout knob.")
    min_value: Optional[int] = Field(default=None, description="Optional lower bound for the knob.")
    max_value: Optional[int] = Field(default=None, description="Optional upper bound for the knob.")
    default_value: Optional[int] = Field(default=None, description="Suggested default value for generation.")
    source: Literal["unconstrained", "bounded"] = Field(
        description="Whether the knob is fully unconstrained or bounded by extracted facts."
    )
    reason: str = Field(description="Why this knob is independent/tunable.")

    def get_signature(self) -> str:
        if self.kind == "table_cardinality":
            return f"Knob:TableCardinality:{self.table_name}"
        return f"Knob:Fanout:{self.parent_table}:{self.child_table}"

class LLMResponse(BaseModel):
    """
    The unified response format from the Stage 3 Extraction Agent.
    """
    logical_constraints: List[SQLGroundedConstraint] = Field(
        default_factory=list,
        description="All deterministic logical constraints as SQL state tables plus binary predicates.",
    )
    cardinality_constraints: List[CardinalityConstraint] = Field(
        default_factory=list,
        description="Explicit bounded table cardinality constraints extracted from facts.",
    )
    fanout_constraints: List[FanoutConstraint] = Field(
        default_factory=list,
        description="Explicit bounded parent-child fanout constraints extracted from facts.",
    )
