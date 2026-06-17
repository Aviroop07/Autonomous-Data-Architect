from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from src.pipeline.stage2.models.schema import Schema
from src.pipeline.stage3.models.sql_models import (
    CardinalityConstraint,
    FanoutConstraint,
    SQLGroundedConstraint,
    StructuralKnob,
)

class TableConstraintManifest(BaseModel):
    table_name: str = Field(description="The formal table name.")

    state_constraints: List[SQLGroundedConstraint] = Field(default_factory=list, description="SQL state-table constraints with binary predicates.")
    cardinality_constraints: List[CardinalityConstraint] = Field(default_factory=list, description="Table cardinality constraints anchored to this table.")
    fanout_constraints: List[FanoutConstraint] = Field(default_factory=list, description="Relationship fanout constraints anchored to this table.")
    nullable_columns: List[str] = Field(default_factory=list, description="Columns flagged as nullable based on constraints.")

    def _validate(self, schema: Schema) -> List[str]:
        errors = []

        # SPECIAL CASE: GLOBAL RULES BUCKET
        if self.table_name.upper() == "__GLOBAL__":
            for constraint in self.state_constraints:
                errors.extend(constraint._validate(schema))
            for constraint in self.cardinality_constraints:
                errors.extend(constraint._validate(schema))
            for constraint in self.fanout_constraints:
                errors.extend(constraint._validate(schema))
            return errors

        # Find table in schema
        t_map = {t.name.upper(): t for t in schema.tables}
        if self.table_name.upper() not in t_map:
            return [f"Table '{self.table_name}' not found in schema."]

        for constraint in self.state_constraints:
            errors.extend(constraint._validate(schema))
        for constraint in self.cardinality_constraints:
            errors.extend(constraint._validate(schema))
        for constraint in self.fanout_constraints:
            errors.extend(constraint._validate(schema))

        return errors


class AlgebraicManifest(BaseModel):
    """Stage 3 project-level constraint artifact."""
    table_manifests: List[TableConstraintManifest]
    global_state_constraints: List[SQLGroundedConstraint] = Field(default_factory=list)
    global_cardinality_constraints: List[CardinalityConstraint] = Field(default_factory=list)
    global_fanout_constraints: List[FanoutConstraint] = Field(default_factory=list)
    tunable_knobs: List[StructuralKnob] = Field(default_factory=list)

    def get_table_manifest(self, table_name: str) -> Optional[TableConstraintManifest]:
        for manifest in self.table_manifests:
            if manifest.table_name.upper() == table_name.upper():
                return manifest
        return None

    def table_manifest_pairs(self) -> List[Tuple[str, TableConstraintManifest]]:
        return [(manifest.table_name, manifest) for manifest in self.table_manifests]

    def _validate(self, schema: Schema) -> List[str]:
        errors = []
        for manifest in self.table_manifests:
            errors.extend(manifest._validate(schema))
        return errors
