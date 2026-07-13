from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from src.pipeline.stage2.models.schema import Schema


class SchemaShard(BaseModel):
    shard_index: int = Field(description="The 0-indexed ID of the shard.")
    tables: List[str] = Field(description="List of table names included in this shard.")
    projections: Dict[str, List[str]] = Field(
        description="Mapping of table name to the exact list of columns projected into this shard."
    )
    allocated_fact_ids: List[int] = Field(
        description="List of fact IDs that are fully contained within this shard."
    )

    def _validate(self, global_schema: Schema) -> List[str]:
        errors = []
        global_pks: Dict[str, List[str]] = {}
        global_fks: List[tuple[str, str, str]] = []

        for t in global_schema.tables:
            global_pks[t.name.upper()] = list(t.primary_key)
        if global_schema.relationships:
            for r in global_schema.relationships:
                global_fks.append(
                    (
                        r.referencing_table.upper(),
                        r.referencing_column.upper(),
                        r.referred_table.upper(),
                    )
                )

        # 1. PK Inclusion
        for table, cols in self.projections.items():
            t_upper = table.upper()
            if t_upper in global_pks:
                for pk in global_pks[t_upper]:
                    if pk.lower() not in [c.lower() for c in cols]:
                        errors.append(
                            f"Table {table} in shard {self.shard_index} is missing its Primary Key '{pk}'."
                        )

        # 2. FK Closure
        for table, cols in self.projections.items():
            t_upper = table.upper()
            c_upper = [c.upper() for c in cols]
            for t_fk, c_fk, t_ref in global_fks:
                if t_fk == t_upper and c_fk in c_upper:
                    # If this shard includes a column that acts as an FK, it MUST include the referenced table
                    if t_ref not in [t.upper() for t in self.tables]:
                        errors.append(
                            f"FK Closure Violation: Shard {self.shard_index} contains {table}.{c_fk} but missing referenced table {t_ref}."
                        )

        return errors
