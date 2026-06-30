import json
import os
import sys
from typing import Dict, Any, Tuple

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
        pk_list = table_info.get("Primary key", [])
        norm_pk_list = [pk.replace(" ", "_").lower() for pk in pk_list]

        # 3. Create Table
        tables.append(
            Table(name=table_name.replace(" ", "_").upper(), columns=columns, primary_key=norm_pk_list)
        )

        # 4. Extract Foreign Keys
        # "Foreign key": {"LocalCol": {"RemoteTable": "RemoteCol"}}
        fk_dict = table_info.get("Foreign key", {})
        for local_col, remote_info in fk_dict.items():
            for remote_table, remote_col in remote_info.items():
                relationships.append(
                    ForeignKey(
                        referencing_table=table_name.replace(" ", "_").upper(),
                        referencing_column=local_col.replace(" ", "_").lower(),
                        referred_table=remote_table.replace(" ", "_").upper(),
                    )
                )

    return Schema(tables=tables, relationships=relationships)


def load_golden_schemas(jsonl_path: str) -> Dict[str, Schema]:
    """
    Loads all golden schemas from annotation.jsonl.
    Returns {id -> Schema} for every line.
    """
    result: Dict[str, Schema] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", ""))
            answer = row.get("answer", {})
            if not answer:
                continue
            result[sample_id] = map_rschema_to_pydantic(answer)
    return result


def load_llm4db_predictions(jsonl_path: str) -> Dict[str, Schema]:
    """
    Loads predicted schemas from a LLM4DBdesign output JSONL file.
    Supports base_direct / base_cot output format where the schema dict
    lives under row["answer"]["schema"].
    Returns {id -> Schema}; skips lines with empty or unparseable schemas.
    """
    result: Dict[str, Schema] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", ""))
            answer = row.get("answer", {})
            # base_direct / base_cot: schema is nested under "schema" key
            schema_dict = (
                answer.get("schema", answer) if isinstance(answer, dict) else {}
            )
            if not schema_dict:
                continue
            try:
                result[sample_id] = map_rschema_to_pydantic(schema_dict)
            except Exception:
                continue
    return result


def get_rschema_case_by_idx(file_path: str, line_idx: int) -> Tuple[str, Schema]:
    """
    Loads the test case at the specified index from the jsonl file.
    Returns (NL description, Schema object).
    """
    with open(file_path, "r", encoding="utf-8") as f:
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
