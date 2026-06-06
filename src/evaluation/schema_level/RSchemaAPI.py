import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.pipeline.stage2.models.schema import Schema, Table, Column, ForeignKey

def map_rschema_to_pydantic(rschema_dict: Dict[str, Any]) -> Schema:
    """
    Maps an RSchema 'answer' dictionary to a Pydantic Schema object.
    """
    tables = []
    relationships = []

    for table_name, table_info in rschema_dict.items():
        # 1. Create columns
        columns = []
        for attr in table_info.get("Attributes", []):
            columns.append(Column(name=attr.replace(" ", "_").lower()))

        # 2. Determine Primary Key
        # The Table model expects a single string for pk.
        # RSchema provides a list. We'll take the first one or join if composite.
        pk_list = table_info.get("Primary key", [])
        pk_str = pk_list[0] if pk_list else ""
        if len(pk_list) > 1:
            pk_str = pk_list[0]
        pk_str = pk_str.replace(" ", "_").lower()

        # 3. Create Table
        tables.append(Table(
            name=table_name.replace(" ", "_").upper(),
            columns=columns,
            pk=pk_str
        ))

        # 4. Extract Foreign Keys
        # "Foreign key": {"LocalCol": {"RemoteTable": "RemoteCol"}}
        fk_dict = table_info.get("Foreign key", {})
        for local_col, remote_info in fk_dict.items():
            for remote_table, remote_col in remote_info.items():
                relationships.append(ForeignKey(
                    referencing_table=table_name.replace(" ", "_").upper(),
                    referencing_column=local_col.replace(" ", "_").lower(),
                    referred_table=remote_table.replace(" ", "_").upper()
                ))

    return Schema(tables=tables, relationships=relationships)

def get_rschema_case_by_idx(file_path: str, line_idx: int) -> Tuple[str, Schema]:
    """
    Loads the test case at the specified index from the jsonl file.
    Returns (NL description, Schema object).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == line_idx:
                data = json.loads(line)
                nl_description = data.get("question", "")
                rschema_data = data.get("answer", {})
                schema_obj = map_rschema_to_pydantic(rschema_data)
                return nl_description, schema_obj
    raise IndexError(f"Line index {line_idx} out of range for {file_path}")

if __name__ == "__main__":
    # Quick demonstration
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset/RSchema/annotation.jsonl")

    if os.path.exists(DATASET_PATH):
        try:
            nl, schema = get_rschema_case_by_idx(DATASET_PATH, 0)
            print("--- NL Description ---")
            print(nl)
            print("\n--- Pydantic Schema ---")
            print(schema)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Dataset not found at {DATASET_PATH}")
