import sqlite3
from src.pipeline.stage2.models.schema import Schema
from typing import Tuple, List, Optional

def validate_sql_against_schema(sql: str, schema: Schema) -> Tuple[bool, List[str], Optional[str]]:
    """
    Creates an in-memory SQLite database, initializes tables from the schema,
    and attempts to explain the query plan to validate syntax and references.

    Returns: (is_valid, result_columns, error_message)
    """
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    try:
        # 1. Create Mock Tables
        for table in schema.tables:
            cols_str = ", ".join([f"{col.name} TEXT" for col in table.columns])
            create_sql = f"CREATE TABLE {table.name.upper()} ({cols_str})"
            cursor.execute(create_sql)

        # 2. Extract Resulting Columns (using a VIEW to dry-run the SELECT)
        # LLMs often use lowercase or mixed case; we normalize for validation
        clean_sql = sql.strip()
        if not clean_sql.upper().startswith("SELECT"):
            if clean_sql.upper().startswith("FROM"):
                clean_sql = f"SELECT * {clean_sql}"
            else:
                clean_sql = f"SELECT * FROM {clean_sql}"

        cursor.execute(f"CREATE VIEW tmp_constraint_view AS {clean_sql}")
        cursor.execute("PRAGMA table_info(tmp_constraint_view)")
        columns = [row[1] for row in cursor.fetchall()]
        return True, columns, None

    except sqlite3.OperationalError as e:
        return False, [], str(e)
    finally:
        conn.close()
