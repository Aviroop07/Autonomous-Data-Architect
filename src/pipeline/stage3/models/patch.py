from typing import List, Optional, Any, Dict, Union, Annotated, Literal
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
    DELETE_TABLE = "DELETE_TABLE"

class BasePatch(BaseModel):
    action: ActionTag
    reason: str = Field(description="TECHNICAL RATIONALE: Why this change is necessary.")

class AddColumnPatch(BasePatch):
    action: Literal[ActionTag.ADD_COLUMN] = ActionTag.ADD_COLUMN
    table_name: str
    column_name: str
    data_type: Optional[str] = None

class RenameColumnPatch(BasePatch):
    action: Literal[ActionTag.RENAME_COLUMN] = ActionTag.RENAME_COLUMN
    table_name: str
    column_name: str
    new_name: str

class DeleteColumnPatch(BasePatch):
    action: Literal[ActionTag.DELETE_COLUMN] = ActionTag.DELETE_COLUMN
    table_name: str
    column_name: str

class SimplifiedTable(BaseModel):
    name: str = Field(description="Table name in UPPER_SNAKE_CASE.")
    columns: List[str] = Field(description="List of column names in lowercase snake_case.")
    pk: str = Field(description="Primary key column name (usually table_name_id).")
    unique: Optional[List[List[str]]] = Field(default=None, description="Optional list of composite unique keys, where each item is a list of column names.")

class RelationshipDefinition(BaseModel):
    referencing_table: str
    referencing_column: str
    referred_table: str

class AddTablePatch(BasePatch):
    action: Literal[ActionTag.ADD_TABLE] = ActionTag.ADD_TABLE
    table_definition: SimplifiedTable

class MergeTablesPatch(BasePatch):
    action: Literal[ActionTag.MERGE_TABLES] = ActionTag.MERGE_TABLES
    source_table: str
    target_table: str

class AddRelationshipPatch(BasePatch):
    action: Literal[ActionTag.ADD_RELATIONSHIP] = ActionTag.ADD_RELATIONSHIP
    fk_definition: RelationshipDefinition

class DeleteRelationshipPatch(BasePatch):
    action: Literal[ActionTag.DELETE_RELATIONSHIP] = ActionTag.DELETE_RELATIONSHIP
    fk_definition: RelationshipDefinition

class UpdatePKPatch(BasePatch):
    action: Literal[ActionTag.UPDATE_PK] = ActionTag.UPDATE_PK
    table_name: str
    column_name: str

class UpsertUniquePatch(BasePatch):
    action: Literal[ActionTag.UPSERT_UNIQUE] = ActionTag.UPSERT_UNIQUE
    table_name: str
    unique_definition: Dict[str, Any] = Field(description="CompositeUnique definition: {columns}")

class DeleteTablePatch(BasePatch):
    action: Literal[ActionTag.DELETE_TABLE] = ActionTag.DELETE_TABLE
    table_name: str

SchemaPatch = Annotated[
    Union[
        AddColumnPatch,
        RenameColumnPatch,
        DeleteColumnPatch,
        AddTablePatch,
        MergeTablesPatch,
        AddRelationshipPatch,
        DeleteRelationshipPatch,
        UpdatePKPatch,
        UpsertUniquePatch,
        DeleteTablePatch
    ],
    Field(discriminator='action')
]

class CritiqueReport(BaseModel):
    agent_name: str
    observations: Optional[str] = None
    patches: List[SchemaPatch] = Field(default_factory=list)
