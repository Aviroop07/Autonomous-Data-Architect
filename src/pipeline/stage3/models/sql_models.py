import re
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from src.util.sql_utils import check_sql_queryability
from src.pipeline.stage3.utils.sql_validator import validate_sql_against_schema
from src.pipeline.stage3.models.distributions import UnivariateDist

class SQLGroundedConstraint(BaseModel):
    """
    A simplified constraint format for LLM generation.
    """
    on: str = Field(description="SQL DML statement defining the table context (e.g. 'SELECT * FROM SALE JOIN CAR ON ...')")
    condition: str = Field(description="SQL boolean expression or 'IF...THEN' logic.")
    fact_references: List[int] = Field(default_factory=list, description="IDs of the business facts supporting this constraint.")

    def _validate(self, schema: Any) -> List[str]:
        errors = []
        # 1. Check SQL Queryability of 'on'
        if not check_sql_queryability(schema, self.on):
            errors.append(f"SQL execution failed for 'on' statement: {self.on}")
            return errors

        # 2. Extract Resulting Columns from 'on'
        is_valid, result_cols, err_msg = validate_sql_against_schema(self.on, schema)
        if not is_valid:
            errors.append(f"SQL reference error in 'on' statement: {err_msg}")
            return errors

        # 3. Parse 'condition' for column names using regex
        # We look for words that match snake_case column patterns
        # Simple extraction: find all word-boundary tokens that are not reserved SQL keywords
        keywords = {"AND", "OR", "NOT", "IS", "NULL", "IN", "BETWEEN", "LIKE", "SELECT", "FROM", "JOIN", "ON", "WHERE", "GROUP", "BY", "IF", "THEN", "ELSE", "CASE", "WHEN", "END", "AS", "EQUALS", "NOT_EQUALS", "GT", "LT", "GTE", "LTE"}
        # 3. Parse 'condition' for column names using regex
        # Strip string literals to avoid picking up values as columns
        stripped_condition = re.sub(r"'[^']*'", "", self.condition)
        # Use negative lookahead to avoid picking up table names in 'table.column' format
        found_tokens = re.findall(r"\b([a-z][a-z0-9_]*)\b(?!\.)", stripped_condition.lower())
        found_cols = {t for t in found_tokens if t.upper() not in keywords}

        # 4. Check if found columns are in the result set of 'on'
        result_cols_lower = {c.lower() for c in result_cols}
        for col in found_cols:
            if col not in result_cols_lower:
                errors.append(f"Column '{col}' used in condition was not found in the result set of the 'on' statement. Available columns: {', '.join(result_cols)}")

        return errors

class LLMResponse(BaseModel):
    """
    The unified response format from the Stage 3 Extraction Agent.
    """
    logical_constraints: List[SQLGroundedConstraint] = Field(default_factory=list, description="All cross-column or conditional business rules.")
    distributions: List[UnivariateDist] = Field(default_factory=list, description="All statistical distributions and static numeric/date ranges.")
