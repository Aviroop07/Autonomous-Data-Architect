from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum

class ActionTag(str, Enum):
    ADD_COLUMN = "ADD_COLUMN"
    RENAME_COLUMN = "RENAME_COLUMN"
    DELETE_COLUMN = "DELETE_COLUMN"
    ADD_TABLE = "ADD_TABLE"
    MERGE_TABLES = "MERGE_TABLES"
    ADD_RELATIONSHIP = "ADD_RELATIONSHIP"
    DELETE_RELATIONSHIP = "DELETE_RELATIONSHIP"
    UPDATE_PK = "UPDATE_PK"
    UPSERT_UNIQUE = "UPSERT_UNIQUE"

class SchemaPatch(BaseModel):
    action: ActionTag
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    new_name: Optional[str] = None
    source_table: Optional[str] = None
    target_table: Optional[str] = None
    data_type: Optional[str] = None
    fk_definition: Optional[Dict[str, Any]] = None
    table_definition: Optional[Dict[str, Any]] = None
    unique_definition: Optional[Dict[str, Any]] = None
    reason: str

class CritiqueReport(BaseModel):
    agent_name: str
    patches: List[SchemaPatch]
    observations: Optional[str] = None
