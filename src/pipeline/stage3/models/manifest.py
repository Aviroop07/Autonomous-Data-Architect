from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union, Any, Dict, Set, Tuple
from src.pipeline.stage3.models.nodes import IfNode, LogicalNode, CombinationNode
from src.pipeline.stage3.models.distributions import UnivariateDist, NumericRange

class TableConstraintManifest(BaseModel):
    table_name: str = Field(description="The formal table name.")

    numeric_bounds: Dict[str, NumericRange] = Field(default_factory=dict, description="Numeric bounds as {col: {min: X, max: Y}} object.")
    distributions: Dict[str, UnivariateDist] = Field(default_factory=dict, description="Distribution mapping per column.")
    logical_rules: List[IfNode] = Field(default_factory=list, description="Logical IF-THEN constraints.")
    nullable_columns: List[str] = Field(default_factory=list, description="Columns flagged as nullable based on constraints.")

    def _validate(self, schema: Any) -> List[str]:
        errors = []

        # SPECIAL CASE: GLOBAL RULES BUCKET
        if self.table_name.upper() == "__GLOBAL__":
            # Still validate the rules themselves
            for rule in self.logical_rules:
                errors.extend(rule._validate(schema))
            return errors

        # Find table in schema
        t_map = {t.name.upper(): t for t in schema.tables}
        if self.table_name.upper() not in t_map:
            return [f"Table '{self.table_name}' not found in schema."]

        table = t_map[self.table_name.upper()]
        pk_col = getattr(table, 'pk', 'id').upper()

        # 1. Prohibit distributions on PK
        for col in self.distributions.keys():
            if col.upper() == pk_col:
                errors.append(f"CRITICAL: Distribution assigned to Primary Key '{col}' in table '{self.table_name}'.")

            # Validate the distribution itself
            errors.extend(self.distributions[col]._validate(schema))

        # 2. Validate numeric bounds
        for col, r in self.numeric_bounds.items():
            errors.extend(r._validate())

        # 3. Validate logical rules
        for rule in self.logical_rules:
            errors.extend(rule._validate(schema))

        return errors


class AlgebraicManifest(BaseModel):
    """The unified project-level artifact passed to the compiler."""
    table_manifests: Dict[str, TableConstraintManifest]
    global_rules: List[Union[IfNode, LogicalNode, CombinationNode]] = Field(default_factory=list)

    def _validate(self, schema: Any) -> List[str]:
        errors = []
        for t_name, manifest in self.table_manifests.items():
            errors.extend(manifest._validate(schema))
        return errors
