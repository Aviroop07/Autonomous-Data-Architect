from typing import List, Optional, Any, Union, Annotated, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator
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
    DELETE_UNIQUE = "DELETE_UNIQUE"
    RENAME_TABLE = "RENAME_TABLE"
    UPDATE_COLUMN_TYPE = "UPDATE_COLUMN_TYPE"

class BasePatch(BaseModel):
    # action: ActionTag
    reason: str = Field(description="TECHNICAL RATIONALE: Why this change is necessary.")

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        return errors

    def _check_consistency(self) -> List[str]:
        """
        Heuristic check for contradictions between reasoning and action.
        """
        reason_lower = self.reason.lower()
        errors = []

        # Action is DELETE_COLUMN but reason suggests keeping it
        if hasattr(self, 'action') and self.action == ActionTag.DELETE_COLUMN:
            keep_keywords = ["keep", "no change", "correctly", "already exists", "preserve", "needed", "essential"]
            if any(k in reason_lower for k in keep_keywords):
                # Check for "not" to avoid false positives like "should NOT keep"
                # (Simple heuristic)
                if "not" not in reason_lower or reason_lower.find("not") > reason_lower.find(next(k for k in keep_keywords if k in reason_lower)):
                     errors.append(f"CONSISTENCY_ERROR: Reasoning suggests keeping/correctness ('{self.reason}'), but action is DELETE_COLUMN.")

        # Action is ADD_COLUMN but reason suggests it exists
        if hasattr(self, 'action') and self.action == ActionTag.ADD_COLUMN:
            exists_keywords = ["already exists", "already present", "already has"]
            if any(k in reason_lower for k in exists_keywords):
                errors.append(f"CONSISTENCY_ERROR: Reasoning suggests column already exists ('{self.reason}'), but action is ADD_COLUMN.")

        return errors

class ColumnPatch(BasePatch):
    table_name: str = Field(description="The table target for the patch.")
    column_name: str = Field(description="The column target for the patch.")

class AddColumnPatch(ColumnPatch):
    action: Literal[ActionTag.ADD_COLUMN] = ActionTag.ADD_COLUMN
    data_type: str = "VARCHAR"

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for ADD_COLUMN.")
        elif any(c.name == self.column_name for c in table_map[self.table_name].columns):
            errors.append(f"Column '{self.column_name}' already exists in table '{self.table_name}'.")
        return errors

    def __str__(self) -> str:
        return f"ADD_COLUMN: Adding '{self.column_name}' to table '{self.table_name}'. (Reason: {self.reason})"

class RenameColumnPatch(ColumnPatch):
    action: Literal[ActionTag.RENAME_COLUMN] = ActionTag.RENAME_COLUMN
    new_name: str

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
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

    def __str__(self) -> str:
        return f"RENAME_COLUMN: Renaming '{self.table_name}.{self.column_name}' to '{self.new_name}'. (Reason: {self.reason})"

class DeleteColumnPatch(ColumnPatch):
    action: Literal[ActionTag.DELETE_COLUMN] = ActionTag.DELETE_COLUMN

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for DELETE_COLUMN.")
        elif not any(c.name == self.column_name for c in table_map[self.table_name].columns):
            errors.append(f"Column '{self.column_name}' not found in table '{self.table_name}' for deletion.")
        return errors

    def __str__(self) -> str:
        return f"DELETE_COLUMN: Removing '{self.column_name}' from table '{self.table_name}'. (Reason: {self.reason})"

class SimplifiedUnique(BaseModel):
    columns: List[str] = Field(description="List of columns that form the unique constraint.")

class SimplifiedColumn(BaseModel):
    name: str = Field(description="The column name.")
    data_type: str = "VARCHAR"

class SimplifiedTable(BaseModel):
    name: str = Field(description="Table name in UPPER_SNAKE_CASE.")
    columns: List[SimplifiedColumn] = Field(description="List of column definitions (name and type).")
    pk: str = Field(description="Primary key column name (usually table_name_id).")
    unique: Optional[List[SimplifiedUnique]] = Field(default=None, description="Optional list of composite unique keys.")

class RelationshipDefinition(BaseModel):
    referencing_table: Optional[str] = Field(default=None, description="The child/source table.")
    referencing_column: Optional[str] = Field(default=None, description="The foreign key column in the child table.")
    referred_table: Optional[str] = Field(default=None, description="The parent/referred table.")

    @model_validator(mode='before')
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalise table names
            if 'referenced_table' in data and 'referred_table' not in data:
                data['referred_table'] = data.pop('referenced_table')
            if 'target_table' in data and 'referred_table' not in data:
                data['referred_table'] = data.pop('target_table')
            if 'parent_table' in data and 'referred_table' not in data:
                data['referred_table'] = data.pop('parent_table')
            if 'to_table' in data and 'referred_table' not in data:
                data['referred_table'] = data.pop('to_table')

            if 'source_table' in data and 'referencing_table' not in data:
                data['referencing_table'] = data.pop('source_table')
            if 'child_table' in data and 'referencing_table' not in data:
                data['referencing_table'] = data.pop('child_table')
            if 'from_table' in data and 'referencing_table' not in data:
                data['referencing_table'] = data.pop('from_table')

            # Normalise column names
            if 'referenced_column' in data and 'referencing_column' not in data:
                data['referencing_column'] = data.pop('referenced_column')
            if 'source_column' in data and 'referencing_column' not in data:
                data['referencing_column'] = data.pop('source_column')
            if 'child_column' in data and 'referencing_column' not in data:
                data['referencing_column'] = data.pop('child_column')
            if 'from_column' in data and 'referencing_column' not in data:
                data['referencing_column'] = data.pop('from_column')
        return data

class AddTablePatch(BasePatch):
    action: Literal[ActionTag.ADD_TABLE] = ActionTag.ADD_TABLE
    table_definition: SimplifiedTable

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        if any(t.name == self.table_definition.name for t in schema.tables):
            errors.append(f"Table '{self.table_definition.name}' already exists.")
        return errors

    def __str__(self) -> str:
        cols = ", ".join([f"{c.name}({c.data_type})" for c in self.table_definition.columns])
        return f"ADD_TABLE: Creating table '{self.table_definition.name}' with columns ({cols}). (Reason: {self.reason})"

class MergeTablesPatch(BasePatch):
    action: Literal[ActionTag.MERGE_TABLES] = ActionTag.MERGE_TABLES
    source_table: str
    target_table: str

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.source_table not in table_map:
            errors.append(f"Source table '{self.source_table}' does not exist for merge.")
        if self.target_table not in table_map:
            errors.append(f"Target table '{self.target_table}' does not exist for merge.")
        return errors

    def __str__(self) -> str:
        return f"MERGE_TABLES: Merging '{self.source_table}' into '{self.target_table}'. (Reason: {self.reason})"

class AddRelationshipPatch(BasePatch):
    action: Literal[ActionTag.ADD_RELATIONSHIP] = ActionTag.ADD_RELATIONSHIP
    fk_definition: RelationshipDefinition

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        defn = self.fk_definition

        if not defn.referencing_table:
            errors.append("MISSING_DATA: 'referencing_table' is required for ADD_RELATIONSHIP.")
        elif defn.referencing_table not in table_map:
            errors.append(f"Referencing table '{defn.referencing_table}' does not exist.")
        elif not defn.referencing_column:
            errors.append("MISSING_DATA: 'referencing_column' is required for ADD_RELATIONSHIP.")
        elif not any(c.name == defn.referencing_column for c in table_map[defn.referencing_table].columns):
            errors.append(f"Referencing column '{defn.referencing_column}' not found in table '{defn.referencing_table}'.")

        if not defn.referred_table:
            errors.append("MISSING_DATA: 'referred_table' is required for ADD_RELATIONSHIP.")
        elif defn.referred_table not in table_map:
            errors.append(f"Referred table '{defn.referred_table}' does not exist.")
        return errors

    def __str__(self) -> str:
        defn = self.fk_definition
        return f"ADD_RELATIONSHIP: Linking '{defn.referencing_table}.{defn.referencing_column}' -> '{defn.referred_table}'. (Reason: {self.reason})"

class DeleteRelationshipPatch(BasePatch):
    action: Literal[ActionTag.DELETE_RELATIONSHIP] = ActionTag.DELETE_RELATIONSHIP
    fk_definition: Optional[RelationshipDefinition] = None

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        if not self.fk_definition:
            errors.append("MISSING_DATA: 'fk_definition' is required even for DELETE_RELATIONSHIP to identify the target link.")
            return errors

        defn = self.fk_definition
        if not defn.referencing_table or not defn.referencing_column or not defn.referred_table:
            errors.append(f"INCOMPLETE_DATA: 'referencing_table', 'referencing_column', and 'referred_table' are all required to identify the link to delete. Got: {defn.referencing_table}.{defn.referencing_column} -> {defn.referred_table}")
            return errors

        # Check if relationship exists
        if not schema.relationships:
            errors.append("No relationships exist in schema to delete.")
            return errors

        exists = any(
            r.referencing_table == defn.referencing_table and
            r.referencing_column == defn.referencing_column and
            r.referred_table == defn.referred_table
            for r in schema.relationships
        )
        if not exists:
            errors.append(f"Relationship {defn.referencing_table}.{defn.referencing_column} -> {defn.referred_table} not found.")
        return errors

    def __str__(self) -> str:
        if not self.fk_definition:
            return f"DELETE_RELATIONSHIP: [MISSING FK DEFINITION] (Reason: {self.reason})"
        defn = self.fk_definition
        return f"DELETE_RELATIONSHIP: Removing link '{defn.referencing_table}.{defn.referencing_column}' -> '{defn.referred_table}'. (Reason: {self.reason})"

class UpdatePKPatch(BasePatch):
    action: Literal[ActionTag.UPDATE_PK] = ActionTag.UPDATE_PK
    table_name: str
    column_name: str

class UpdateColumnTypePatch(ColumnPatch):
    action: Literal[ActionTag.UPDATE_COLUMN_TYPE] = ActionTag.UPDATE_COLUMN_TYPE
    new_type: str = Field(description="The target data type for the column (e.g. INTEGER, FLOAT).")

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for type update.")
        elif not any(c.name == self.column_name for c in table_map[self.table_name].columns):
            errors.append(f"Column '{self.column_name}' not found in table '{self.table_name}' for type update.")
        return errors

    def __str__(self) -> str:
        return f"UPDATE_COLUMN_TYPE: Setting type of '{self.table_name}.{self.column_name}' to '{self.new_type}'. (Reason: {self.reason})"

class UpsertUniquePatch(BasePatch):
    action: Literal[ActionTag.UPSERT_UNIQUE] = ActionTag.UPSERT_UNIQUE
    table_name: str
    unique_definition: SimplifiedUnique = Field(description="Strict composite unique definition with columns list.")

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for UNIQUE upsert.")
        else:
            table_cols = {c.name for c in table_map[self.table_name].columns}
            for col in self.unique_definition.columns:
                if col not in table_cols:
                    errors.append(f"Column '{col}' not found in table '{self.table_name}' for UNIQUE constraint.")
        return errors

    def __str__(self) -> str:
        cols = ", ".join(self.unique_definition.columns)
        return f"UPSERT_UNIQUE: Setting UNIQUE constraint on '{self.table_name}'({cols}). (Reason: {self.reason})"

class DeleteUniquePatch(BasePatch):
    action: Literal[ActionTag.DELETE_UNIQUE] = ActionTag.DELETE_UNIQUE
    table_name: str
    unique_definition: SimplifiedUnique

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Table '{self.table_name}' does not exist for UNIQUE deletion.")
        else:
            table = table_map[self.table_name]
            # Check if such a unique constraint exists
            exists = False
            if table.unique:
                for uc in table.unique:
                    if set(uc.columns) == set(self.unique_definition.columns):
                        exists = True
                        break
            if not exists:
                errors.append(f"UNIQUE constraint on ({', '.join(self.unique_definition.columns)}) not found in table '{self.table_name}'.")
        return errors

    def __str__(self) -> str:
        cols = ", ".join(self.unique_definition.columns)
        return f"DELETE_UNIQUE: Removing UNIQUE constraint on '{self.table_name}'({cols}). (Reason: {self.reason})"

class DeleteTablePatch(BasePatch):
    action: Literal[ActionTag.DELETE_TABLE] = ActionTag.DELETE_TABLE
    table_name: str = Field(description="The table name to delete.")

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        if not any(t.name == self.table_name for t in schema.tables):
            errors.append(f"Table '{self.table_name}' does not exist for deletion.")
        return errors

    def __str__(self) -> str:
        return f"DELETE_TABLE: Deleting table '{self.table_name}'. (Reason: {self.reason})"

class RenameTablePatch(BasePatch):
    action: Literal[ActionTag.RENAME_TABLE] = ActionTag.RENAME_TABLE
    table_name: str = Field(description="The current table name.")
    new_name: str = Field(description="The target table name.")

    def _validate(self, schema: 'Schema') -> List[str]:
        errors = self._check_consistency()
        table_map = schema.get_table_map()
        if self.table_name not in table_map:
            errors.append(f"Source table '{self.table_name}' does not exist for RENAME_TABLE.")
        if self.new_name in table_map:
            errors.append(f"Target table name '{self.new_name}' already exists.")
        return errors

    def __str__(self) -> str:
        return f"RENAME_TABLE: Renaming '{self.table_name}' to '{self.new_name}'. (Reason: {self.reason})"

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
        UpdateColumnTypePatch,
        UpsertUniquePatch,
        DeleteUniquePatch,
        DeleteTablePatch,
        RenameTablePatch
    ],
    Field(discriminator='action')
]

import re as _re

def _normalize_action_tag(raw: str) -> str:
    """
    Normalise a raw action string into the canonical UPPER_SNAKE_CASE ActionTag value.
    """
    s = _re.sub(r'([a-z])([A-Z])', r'\1_\2', raw) # CamelCase
    s = _re.sub(r'[^A-Z0-9a-z]', '_', s)
    s = s.upper()
    s = _re.sub(r'_+', '_', s).strip('_')

    if s.endswith('_PATCH'): s = s[:-6]
    if s.endswith('PATCH') and not s.endswith('_PATCH'): s = s[:-5]
    s = s.strip('_')

    if (s.startswith('UPSER') or s.startswith('UPSET')) and 'UNIQUE' in s: return "UPSERT_UNIQUE"
    if 'CONSTRAINT' in s: return "UPSERT_UNIQUE"
    if s.startswith('ADD_COL'): return "ADD_COLUMN"
    if s.startswith('RENAME_COL'): return "RENAME_COLUMN"
    if s.startswith('DELETE_COL'): return "DELETE_COLUMN"

    return s

_KNOWN_TAGS = {tag.value for tag in ActionTag}

class CritiqueReport(BaseModel):
    agent_name: str = Field(description="The AI name or persona assessing the schema.")
    observations: Optional[str] = Field(default=None, description="General strategic observations regarding the current schema architecture.")
    patches: List[SchemaPatch] = Field(default_factory=list, description="List of proposed atomic modifications (patches) to the schema.")

    def __str__(self) -> str:
        lines = [f"### Critique by: {self.agent_name}"]
        if self.observations:
            lines.append(f"\n**Observations**:\n{self.observations}\n")

        if self.patches:
            lines.append("**Suggested Patches**:")
            for p in self.patches: lines.append(str(p))
        else:
            lines.append("*No patches suggested.*")
        return "\n".join(lines)

    @model_validator(mode='before')
    @classmethod
    def preprocess_action_tags(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'patches' in data and isinstance(data['patches'], list):
            normalised_patches = []
            for patch in data['patches']:
                if patch is None or not isinstance(patch, dict): continue

                reason_key = next((k for k in patch.keys() if k.lower() in ('reason', 'rationale', 'explanation')), 'reason')
                if reason_key != 'reason': patch['reason'] = patch.pop(reason_key)
                if not patch.get('reason'): patch['reason'] = "Mandatory schema adjustment."

                action_key = next((k for k in patch.keys() if k.lower() == 'action'), None)
                if action_key:
                    raw_action = patch[action_key]
                    if action_key != 'action': patch['action'] = patch.pop(action_key)

                    normalised = _normalize_action_tag(str(raw_action))
                    if normalised in _KNOWN_TAGS:
                        if normalised == "UPSERT_UNIQUE":
                            if 'unique_definition' not in patch:
                                if 'columns' in patch: patch['unique_definition'] = {'columns': patch.pop('columns')}

                        if normalised in ("ADD_RELATIONSHIP", "DELETE_RELATIONSHIP"):
                            if 'fk_definition' not in patch:
                                potential_defn = {}
                                for f_key in ['referencing_table', 'referencing_column', 'referred_table', 'referenced_table']:
                                    if f_key in patch: potential_defn[f_key] = patch.pop(f_key)
                                if potential_defn: patch['fk_definition'] = potential_defn

                        patch = {**patch, 'action': normalised}
                        normalised_patches.append(patch)
            data = {**data, 'patches': normalised_patches}
        return data

    def _validate(self, schema: Optional['Schema'] = None) -> List[PatchValidationError]:
        if not schema: return []
        validation_results = []
        for i, patch in enumerate(self.patches):
            errors = patch._validate(schema)
            if errors:
                validation_results.append(PatchValidationError(patch_index=i, action=patch.action, errors=errors))
        return validation_results
