from typing import List, Optional, Any, Dict, Union, Annotated, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from enum import Enum

if TYPE_CHECKING:
    from src.pipeline.stage2.models.schema import Schema

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
    # action: ActionTag
    reason: str = Field(description="TECHNICAL RATIONALE: Why this change is necessary.")

    def _validate(self, schema: 'Schema') -> List[str]:
        return []

class ColumnPatch(BasePatch):
    table_name: str
    column_name: str

class AddColumnPatch(ColumnPatch):
    action: Literal[ActionTag.ADD_COLUMN] = ActionTag.ADD_COLUMN

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for ADD_COLUMN.")
        elif any(c.name == self.column_name for c in table_map[self.table_name].columns):
            errors.append(f"Column '{self.column_name}' already exists in table '{self.table_name}'.")
        return errors

class RenameColumnPatch(ColumnPatch):
    action: Literal[ActionTag.RENAME_COLUMN] = ActionTag.RENAME_COLUMN
    new_name: str

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for RENAME_COLUMN.")
        else:
            table = table_map[self.table_name]
            if not any(c.name == self.column_name for c in table.columns):
                errors.append(f"Source column '{self.column_name}' not found in table '{self.table_name}'.")
            if any(c.name == self.new_name for c in table.columns):
                errors.append(f"New column name '{self.new_name}' already exists in table '{self.table_name}'.")
        return errors

class DeleteColumnPatch(ColumnPatch):
    action: Literal[ActionTag.DELETE_COLUMN] = ActionTag.DELETE_COLUMN

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for DELETE_COLUMN.")
        elif not any(c.name == self.column_name for c in table_map[self.table_name].columns):
            errors.append(f"Column '{self.column_name}' not found in table '{self.table_name}' for deletion.")
        return errors

class SimplifiedUnique(BaseModel):
    columns: List[str]

class SimplifiedTable(BaseModel):
    name: str = Field(description="Table name in UPPER_SNAKE_CASE.")
    columns: List[str] = Field(description="List of column names in lowercase snake_case.")
    pk: str = Field(description="Primary key column name (usually table_name_id).")
    unique: Optional[List[SimplifiedUnique]] = Field(default=None, description="Optional list of composite unique keys.")

class RelationshipDefinition(BaseModel):
    referencing_table: str
    referencing_column: str
    referred_table: str

class AddTablePatch(BasePatch):
    action: Literal[ActionTag.ADD_TABLE] = ActionTag.ADD_TABLE
    table_definition: SimplifiedTable

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        if any(t.name == self.table_definition.name for t in schema.tables):
            errors.append(f"Table '{self.table_definition.name}' already exists.")
        return errors

class MergeTablesPatch(BasePatch):
    action: Literal[ActionTag.MERGE_TABLES] = ActionTag.MERGE_TABLES
    source_table: str
    target_table: str

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        if self.source_table not in table_map:
            errors.append(f"Source table '{self.source_table}' does not exist for merge.")
        if self.target_table not in table_map:
            errors.append(f"Target table '{self.target_table}' does not exist for merge.")
        return errors

class AddRelationshipPatch(BasePatch):
    action: Literal[ActionTag.ADD_RELATIONSHIP] = ActionTag.ADD_RELATIONSHIP
    fk_definition: RelationshipDefinition

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        defn = self.fk_definition
        if defn.referencing_table not in table_map:
            errors.append(f"Referencing table '{defn.referencing_table}' does not exist.")
        elif not any(c.name == defn.referencing_column for c in table_map[defn.referencing_table].columns):
            errors.append(f"Referencing column '{defn.referencing_column}' not found in table '{defn.referencing_table}'.")
        
        if defn.referred_table not in table_map:
            errors.append(f"Referred table '{defn.referred_table}' does not exist.")
        return errors

class DeleteRelationshipPatch(BasePatch):
    action: Literal[ActionTag.DELETE_RELATIONSHIP] = ActionTag.DELETE_RELATIONSHIP
    fk_definition: RelationshipDefinition

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        # Check if relationship exists
        if not schema.relationships:
            errors.append("No relationships exist in schema to delete.")
            return errors
        
        defn = self.fk_definition
        exists = any(
            r.referencing_table == defn.referencing_table and 
            r.referencing_column == defn.referencing_column and 
            r.referred_table == defn.referred_table
            for r in schema.relationships
        )
        if not exists:
            errors.append(f"Relationship {defn.referencing_table}.{defn.referencing_column} -> {defn.referred_table} not found.")
        return errors

class UpdatePKPatch(BasePatch):
    action: Literal[ActionTag.UPDATE_PK] = ActionTag.UPDATE_PK
    table_name: str
    column_name: str

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for PK update.")
        elif not any(c.name == self.column_name for c in table_map[self.table_name].columns):
            errors.append(f"Column '{self.column_name}' not found in table '{self.table_name}' for PK.")
        return errors

class UpsertUniquePatch(BasePatch):
    action: Literal[ActionTag.UPSERT_UNIQUE] = ActionTag.UPSERT_UNIQUE
    table_name: str
    unique_definition: SimplifiedUnique = Field(description="Strict composite unique definition with columns list.")

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for UNIQUE upsert.")
        else:
            table_cols = {c.name for c in table_map[self.table_name].columns}
            for col in self.unique_definition.columns:
                if col not in table_cols:
                    errors.append(f"Column '{col}' not found in table '{self.table_name}' for UNIQUE constraint.")
        return errors

class DeleteTablePatch(BasePatch):
    action: Literal[ActionTag.DELETE_TABLE] = ActionTag.DELETE_TABLE
    table_name: str

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = []
        if not any(t.name == self.table_name for t in schema.tables):
            errors.append(f"Table '{self.table_name}' does not exist for deletion.")
        return errors

class PatchValidationError(BaseModel):
    patch_index: int
    action: ActionTag
    errors: List[str]

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

import re as _re

def _normalize_action_tag(raw: str) -> str:
    """
    Normalise a raw action string into the canonical UPPER_SNAKE_CASE ActionTag value.
    Handles variants like 'ADDCOLUMN', 'addColumn', 'Add_Column', 'add-column' etc.
    Also handles common LLM truncations like 'UPSER' or 'UPSER Unique'.
    """
    # 1. Basic cleaning: remove non-alphanumeric, upper case, collapse underscores
    s = _re.sub(r'([a-z])([A-Z])', r'\1_\2', raw) # CamelCase
    s = _re.sub(r'[^A-Z0-9a-z]', '_', s)
    s = s.upper()
    s = _re.sub(r'_+', '_', s).strip('_')

    # 2. Fuzzy/Prefix matching for common typos
    if (s.startswith('UPSER') or s.startswith('UPSET')) and 'UNIQUE' in s:
        return "UPSERT_UNIQUE"
    if (s.startswith('UPSER') or s.startswith('UPSET')) and not s.endswith('UNIQUE'):
        return "UPSERT_UNIQUE" # Most likely intended
    if 'UNIQUE' in s and (s.startswith('ADD') or s.startswith('CREATE')):
        return "UPSERT_UNIQUE"
    if 'CONSTRAINT' in s:
        return "UPSERT_UNIQUE" # Most likely intended
    if s.startswith('ADD_COL'): return "ADD_COLUMN"
    if s.startswith('RENAME_COL'): return "RENAME_COLUMN"
    if s.startswith('DELETE_COL'): return "DELETE_COLUMN"
    
    return s

_KNOWN_TAGS = {tag.value for tag in ActionTag}

class CritiqueReport(BaseModel):
    agent_name: str
    observations: Optional[str] = None
    patches: List[SchemaPatch] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, obj, *, strict=None, from_attributes=None, context=None, experimental_allow_partial=None):
        # Normalise action strings in raw dicts before discriminated union parsing
        if isinstance(obj, dict) and 'patches' in obj and isinstance(obj['patches'], list):
            normalised_patches = []
            for patch in obj['patches']:
                if isinstance(patch, dict) and 'action' in patch:
                    raw_action = patch['action']
                    normalised = _normalize_action_tag(str(raw_action))
                    if normalised in _KNOWN_TAGS:
                        patch = {**patch, 'action': normalised}
                normalised_patches.append(patch)
            obj = {**obj, 'patches': normalised_patches}
        return super().model_validate(obj, strict=strict, from_attributes=from_attributes, context=context, experimental_allow_partial=experimental_allow_partial)

    def _validate(self, schema: 'Schema') -> List[PatchValidationError]:
        """
        Validates all patches in the report and returns a list of error details.
        """
        validation_results = []
        for i, patch in enumerate(self.patches):
            errors = patch._validate(schema)
            if errors:
                validation_results.append(PatchValidationError(
                    patch_index=i,
                    action=patch.action,
                    errors=errors
                ))
        return validation_results
