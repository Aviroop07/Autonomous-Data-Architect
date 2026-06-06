from src.pipeline.stage3.utils.sql_validator import validate_sql_against_schema
from src.pipeline.stage2.models.schema import Schema

def check_sql_queryability(schema: Schema, sql_dml: str) -> bool:
    """
    Executes a dry-run of the SQL DML against a mock schema to check for syntax and reference errors.
    Returns True if the SQL is executable, False otherwise.
    """
    is_valid, _, _ = validate_sql_against_schema(sql_dml, schema)
    return is_valid
