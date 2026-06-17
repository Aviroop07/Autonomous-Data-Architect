from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Dict, Set, Any
from pydantic import BaseModel, Field, model_validator
from src.util.orchestration.loop_types import LoopOutputModel

if TYPE_CHECKING:
    from src.pipeline.stage2.models.registry import TableFactRegistry

UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
LOWER_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")

FORBIDDEN_TABLE_SUFFIXES = {"FACT", "DIM", "ID", "ATTR", "TABLE"}
ALLOWED_PK_TYPES = {"INTEGER", "VARCHAR"}
SINGULAR_S_SUFFIXES = ("SS", "IS", "US")
SINGULAR_S_EXCEPTIONS = {"NEWS", "SERIES", "SPECIES"}


def is_upper_snake(value: str) -> bool:
    return bool(UPPER_SNAKE.fullmatch(value))


def is_lower_snake(value: str) -> bool:
    return bool(LOWER_SNAKE.fullmatch(value))


def to_snake_case(s: str) -> str:
    """
    Converts a string to snake_case.
    """
    s = s.strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


class Column(BaseModel):
    name: str = Field(description="Column name in lowercase snake_case.")
    data_type: Optional[str] = Field(
        default="VARCHAR",
        description="The data type of the column (e.g., INT, FLOAT, VARCHAR).",
    )

    def _validate(self) -> List[str]:
        errors = []
        if not is_lower_snake(self.name):
            errors.append(f"Column must be lowercase snake_case: {self.name}")

        # [NEW] data_type validation (optional but good for consistency)
        if self.data_type and self.data_type.upper() not in {
            "INT",
            "INTEGER",
            "VARCHAR",
            "FLOAT",
            "NUMERIC",
            "DECIMAL",
            "BOOLEAN",
            "DATE",
            "DATETIME",
            "TIMESTAMP",
            "JSON",
            "TEXT",
            "UUID",
        }:
            # We don't report as hard error yet to be flexible, but we could.
            pass

        return errors

    def __str__(self) -> str:
        return self.name


class CompositeUnique(BaseModel):
    columns: List[str] = Field(
        description="List of column names that form a composite unique constraint."
    )

    def __str__(self) -> str:
        cols = ", ".join(self.columns)
        return f"UNIQUE({cols})"


class Table(BaseModel):
    name: str = Field(description="Table name in UPPER_SNAKE_CASE.")
    columns: List[Column] = Field(
        description="List of column definitions for the table."
    )
    pk: str = Field(description="Primary key column name (usually table_name_id).")
    unique: Optional[List[CompositeUnique]] = Field(
        default=None, description="Optional list of composite unique constraints."
    )

    def rename_column(self, old_name: str, new_name: str) -> None:
        """
        Renames a column within the table.
        """
        for col in self.columns:
            if col.name == old_name:
                col.name = new_name

        if self.pk == old_name:
            self.pk = new_name

        if self.unique:
            for uq in self.unique:
                uq.columns = [new_name if c == old_name else c for c in uq.columns]

    def normalize(self, registry: Optional[TableFactRegistry] = None) -> None:
        """
        Normalizes table name and column names, and scrubs redundant unique constraints.
        """
        old_table_name = self.name
        self.name = to_snake_case(self.name).upper()

        if registry and old_table_name != self.name:
            registry.rename_table(old_table_name, self.name)

        # Normalize columns
        for col in self.columns:
            old_col_name = col.name
            new_col_name = to_snake_case(col.name).lower()
            if old_col_name != new_col_name:
                self.rename_column(old_col_name, new_col_name)

        # Ensure pk is normalized
        self.pk = to_snake_case(self.pk).lower()

        # [HARDENING] Scrub redundant unique constraints
        if self.unique:
            # 1. Remove PK from any composite unique constraint
            # 2. Filter out constraints that become empty or singleton-PK
            new_uniques = []
            for uq in self.unique:
                scrubbed_cols = [c for c in uq.columns if c != self.pk]
                if scrubbed_cols:
                    new_uniques.append(CompositeUnique(columns=scrubbed_cols))

            # 3. Identify and remove subset redundancies
            # If UNIQUE(A) exists, UNIQUE(A, B) is redundant.
            unique_sets = [set(uq.columns) for uq in new_uniques]
            final_uniques = []
            for i, set_a in enumerate(unique_sets):
                is_redundant = False
                for j, set_b in enumerate(unique_sets):
                    # If set_b is a proper subset of set_a, then set_a is redundant
                    if i != j and set_b.issubset(set_a) and set_b != set_a:
                        is_redundant = True
                        break
                    # If they are identical and i > j, remove duplicate
                    if i > j and set_a == set_b:
                        is_redundant = True
                        break
                if not is_redundant:
                    final_uniques.append(new_uniques[i])

            self.unique = final_uniques if final_uniques else None

    def _validate(self) -> List[str]:
        errors = []

        # Table name format
        if not is_upper_snake(self.name):
            errors.append(f"Table name must be UPPER_SNAKE_CASE: {self.name}")

        normalized_tokens = set(self.name.split("_"))
        for suffix in FORBIDDEN_TABLE_SUFFIXES:
            if suffix in normalized_tokens:
                errors.append(
                    f"Forbidden word/suffix in table name: {self.name} ('{suffix}')"
                )

        # Column validation
        if not self.columns:
            errors.append(f"Table {self.name} must have at least one column")

        column_names = set()
        numeric_keywords = {
            "amount",
            "rate",
            "price",
            "balance",
            "principal",
            "salary",
            "income",
            "ratio",
            "score",
            "weight",
            "cost",
            "budget",
        }
        for col in self.columns:
            errors.extend(col._validate())
            if col.name in column_names:
                errors.append(f"Duplicate column in {self.name}: {col.name}")

            # [HARDENING] Prevent VARCHAR for numeric-sounding columns
            c_name_lower = col.name.lower()
            if col.data_type and col.data_type.upper() == "VARCHAR":
                found_keyword = next(
                    (k for k in numeric_keywords if k in c_name_lower), None
                )
                if found_keyword:
                    errors.append(
                        f"Technical Error: Column '{col.name}' in {self.name} contains numeric keyword '{found_keyword}' but is typed as VARCHAR. You MUST use FLOAT, DECIMAL, or INTEGER for quantifiable attributes."
                    )

            column_names.add(col.name)

        # Primary key validation
        if not self.pk:
            errors.append(f"Table {self.name} must have a primary key")
        else:
            pk_col = next((c for c in self.columns if c.name == self.pk), None)
            if not pk_col:
                errors.append(
                    f"Primary key '{self.pk}' not found in columns for table {self.name}"
                )
            else:
                # [STRICT DATA TYPE CHECK]
                if (
                    pk_col.data_type
                    and pk_col.data_type.upper() not in ALLOWED_PK_TYPES
                ):
                    errors.append(
                        f"Primary key '{self.pk}' in table {self.name} must be of type {', '.join(ALLOWED_PK_TYPES)} (found {pk_col.data_type})."
                    )

            # Singular check (heuristic)
            if self.name.endswith("S") and not self.name.endswith(SINGULAR_S_SUFFIXES):
                if not any(
                    token in SINGULAR_S_EXCEPTIONS for token in normalized_tokens
                ):
                    errors.append(f"Table name should be singular: {self.name}")

        # Unique constraints validation
        unique_sets = []
        for unique_constraint in self.unique or []:
            curr_set = set(unique_constraint.columns)
            unique_sets.append(curr_set)

            # Check for Primary Key inclusion
            if self.pk in curr_set:
                if len(curr_set) == 1:
                    errors.append(
                        f"Redundant unique singleton: Column '{self.pk}' is already the Primary Key of {self.name}."
                    )
                else:
                    cols_str = ", ".join(unique_constraint.columns)
                    errors.append(
                        f"Redundant unique composite containing PK: '{self.pk}' in ({cols_str}) for table {self.name}."
                    )

            # Check for unknown columns
            for col_name in unique_constraint.columns:
                if col_name not in column_names:
                    errors.append(
                        f"Unique constraint references unknown column '{col_name}' in table {self.name}"
                    )

        # Subset redundancy check
        for i, set_a in enumerate(unique_sets):
            for j, set_b in enumerate(unique_sets):
                if i != j:
                    if set_a.issubset(set_b):
                        cols_a = ", ".join(sorted(list(set_a)))
                        cols_b = ", ".join(sorted(list(set_b)))
                        errors.append(
                            f"Subset redundancy in unique constraints for {self.name}: ({cols_a}) is a subset of ({cols_b}). The larger constraint is redundant and must be removed."
                        )

        return errors

    def __str__(self) -> str:
        lines = [f"TABLE {self.name} ("]
        for col in self.columns:
            if col.name == self.pk:
                lines.append(f"    {col.name} PRIMARY KEY,")
            else:
                lines.append(f"    {col.name},")
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        if self.unique:
            lines.append("")
            for uq in self.unique:
                lines.append(f"    {uq}")
        lines.append(")")
        return "\n".join(lines)


class ForeignKey(BaseModel):
    referencing_table: str = Field(
        description="The table that contains the foreign key."
    )
    referencing_column: str = Field(
        description="The column in the referencing table that points to the referred table."
    )
    referred_table: str = Field(description="The table that is being referenced.")

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "referenced_table" in data and "referred_table" not in data:
                data["referred_table"] = data.pop("referenced_table")
            if "referenced_column" in data and "referencing_column" not in data:
                data["referencing_column"] = data.pop("referenced_column")
        return data

    def _validate(self, tables_map) -> List[str]:
        errors = []
        if self.referencing_table not in tables_map:
            errors.append(
                f"FK error: Referencing table '{self.referencing_table}' does not exist."
            )
        else:
            ref_table = tables_map[self.referencing_table]
            if not any(c.name == self.referencing_column for c in ref_table.columns):
                errors.append(
                    f"FK error: Column '{self.referencing_column}' not found in table '{self.referencing_table}'."
                )

        if self.referred_table not in tables_map:
            errors.append(
                f"FK error: Referred table '{self.referred_table}' does not exist."
            )

        if self.referencing_table in tables_map:
            ref_table = tables_map[self.referencing_table]
            if ref_table.pk == self.referencing_column:
                errors.append(
                    f"FK error: Column '{self.referencing_column}' is the Primary Key of '{self.referencing_table}'. Using a PK as an FK is prohibited."
                )

            # [STRICT TYPE MATCH CHECK]
            if self.referred_table in tables_map:
                target_table = tables_map[self.referred_table]
                ref_col = next(
                    (c for c in ref_table.columns if c.name == self.referencing_column),
                    None,
                )
                target_pk_col = next(
                    (c for c in target_table.columns if c.name == target_table.pk), None
                )

                if ref_col and target_pk_col:
                    if (
                        ref_col.data_type
                        and target_pk_col.data_type
                        and ref_col.data_type.upper() != target_pk_col.data_type.upper()
                    ):
                        errors.append(
                            f"Type mismatch in Foreign Key: {self.referencing_table}.{self.referencing_column} "
                            f"(type: {ref_col.data_type}) must match referred table {self.referred_table} PK "
                            f"'{target_table.pk}' (type: {target_pk_col.data_type})."
                        )

        # [HARDENING] Discourage bridge-table targets unless necessary
        if self.referred_table in tables_map:
            target_table = tables_map[self.referred_table]
            target_cols = {c.name for c in target_table.columns}
            # Heuristic: If target table looks like a bridge (no descriptive non-PK/FK columns)
            # but has more than 3 columns, it's likely a complex entity; otherwise, flag potential inaccuracy.
            if len(target_cols) <= 3 and target_table.pk in target_cols:
                # If there is another table with exactly the same PK name as this target table's PK,
                # but with more metadata, then referencing the bridge might be a mistake.
                pass

        return errors

    def __str__(self) -> str:
        return (
            f"FOREIGN KEY ({self.referencing_table}.{self.referencing_column}) "
            f"REFERENCES {self.referred_table}"
        )


class Schema(LoopOutputModel):
    chunk_title: Optional[str] = Field(
        default=None, description="A descriptive title for this schema subset."
    )
    tables: List[Table] = Field(description="List of all tables in the schema.")
    relationships: Optional[List[ForeignKey]] = Field(
        default=None, description="List of foreign key relationships between tables."
    )

    def get_errors(self) -> list[str]:
        return []

    def get_table_map(self) -> Dict[str, Table]:
        return {t.name: t for t in self.tables}

    def rename_table(
        self, old_name: str, new_name: str, registry: Optional[TableFactRegistry] = None
    ) -> None:
        """
        Renames a table and updates all its relationship occurrences.
        """
        for table in self.tables:
            if table.name == old_name:
                table.name = new_name

        if self.relationships:
            for rel in self.relationships:
                if rel.referencing_table == old_name:
                    rel.referencing_table = new_name
                if rel.referred_table == old_name:
                    rel.referred_table = new_name

        if registry:
            registry.rename_table(old_name, new_name)

    def rename_column(
        self, table_name: str, old_col_name: str, new_col_name: str
    ) -> None:
        """
        Renames a column in a specific table and updates all its relationship occurrences.
        """
        for table in self.tables:
            if table.name == table_name:
                table.rename_column(old_col_name, new_col_name)

        if self.relationships:
            for rel in self.relationships:
                if (
                    rel.referencing_table == table_name
                    and rel.referencing_column == old_col_name
                ):
                    rel.referencing_column = new_col_name

    def normalize(self, registry: Optional[TableFactRegistry] = None) -> None:
        """
        Normalizes the entire schema.
        """
        # First ensure all names are trimmed and follow snake_case
        for table in self.tables:
            old_table_name = table.name
            table.normalize(registry=registry)
            new_table_name = table.name

            # If table name changed, update relationships
            if old_table_name != new_table_name:
                if self.relationships:
                    for rel in self.relationships:
                        if rel.referencing_table == old_table_name:
                            rel.referencing_table = new_table_name
                        if rel.referred_table == old_table_name:
                            rel.referred_table = new_table_name

    def align_fk_column_types(self) -> None:
        """
        Align referencing column types to match referred table PK types.
        """
        if not self.relationships:
            return
        table_map = self.get_table_map()
        for rel in self.relationships:
            if (
                rel.referencing_table not in table_map
                or rel.referred_table not in table_map
            ):
                continue
            ref_table = table_map[rel.referencing_table]
            target_table = table_map[rel.referred_table]
            ref_col = next(
                (c for c in ref_table.columns if c.name == rel.referencing_column), None
            )
            target_pk_col = next(
                (c for c in target_table.columns if c.name == target_table.pk), None
            )
            if (
                ref_col
                and target_pk_col
                and ref_col.data_type
                and target_pk_col.data_type
            ):
                if ref_col.data_type.upper() != target_pk_col.data_type.upper():
                    ref_col.data_type = target_pk_col.data_type

    def _validate(self) -> List[str]:
        errors = []
        if not self.tables:
            errors.append("Schema must contain at least one table.")

        table_map = self.get_table_map()
        for table in self.tables:
            errors.extend(table._validate())

        seen_tables: Set[str] = set()
        for table in self.tables:
            if table.name in seen_tables:
                errors.append(f"Duplicate table name in schema: {table.name}")
            seen_tables.add(table.name)

        adj: Dict[str, Set[str]] = {t.name.upper(): set() for t in self.tables}
        for fk in self.relationships or []:
            errors.extend(fk._validate(table_map))
            t1 = fk.referencing_table.upper()
            t2 = fk.referred_table.upper()
            if t1 in adj and t2 in adj:
                adj[t1].add(t2)
                adj[t2].add(t1)

        # BFS to find connected components
        visited = set()
        components = []
        for table_name in adj:
            if table_name not in visited:
                component = set()
                queue = [table_name]
                visited.add(table_name)
                while queue:
                    curr = queue.pop(0)
                    component.add(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)

        SKELETON_KEYWORDS = {
            "ORDER",
            "PAYMENT",
            "TRANSACTION",
            "METRIC",
            "TRACE",
            "LOG",
            "SPAN",
            "USER",
            "CUSTOMER",
            "PRODUCT",
            "MERCHANT",
            "FULFILLMENT",
            "SHIPMENT",
            "GAUGE",
            "COUNTER",
        }

        if len(components) > 1:
            for component in components:
                if len(component) == 1:
                    t_name = list(component)[0]
                    is_skeleton = any(k in t_name.upper() for k in SKELETON_KEYWORDS)
                    # Isolation is permitted for skeleton tables or single-table shards
                    if not is_skeleton and len(self.tables) > 1:
                        errors.append(
                            f"Table '{t_name}' is strictly isolated (no relationships)."
                        )

        # Cycle detection
        cycles = self.detect_cycles()
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            # [REFINEMENT] Column-level cycles are the true hard dependencies.
            # However, with PK != FK, column-level cycles are technically impossible.
            # We report them if they somehow occur.
            errors.append(
                f"FK error: Circular dependency detected (column-level): {cycle_str}"
            )

        return errors

    def detect_cycles(self) -> List[List[str]]:
        """
        Detects cycles in the Foreign Key relationship graph at the COLUMN level.
        Returns a list of cycles, where each cycle is a list of "table.column" strings.
        """
        if not self.relationships:
            return []

        table_map = self.get_table_map()

        # Build adjacency list: "table.column" -> "referred_table.pk"
        adj: Dict[str, Set[str]] = {}
        all_nodes = set()

        for rel in self.relationships:
            source_node = f"{rel.referencing_table}.{rel.referencing_column}"
            # Fetch the PK of the target table
            target_pk = "id"  # Default fallback
            if rel.referred_table in table_map:
                target_pk = table_map[rel.referred_table].pk
            target_node = f"{rel.referred_table}.{target_pk}"

            if source_node not in adj:
                adj[source_node] = set()
            adj[source_node].add(target_node)
            all_nodes.add(source_node)
            all_nodes.add(target_node)

        visited: set = set()
        stack: set = set()
        path: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            visited.add(node)
            stack.add(node)
            path.append(node)

            if node in adj:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in stack:
                        try:
                            idx = path.index(neighbor)
                            cycles.append(path[idx:] + [neighbor])
                        except ValueError:
                            pass

            stack.remove(node)
            path.pop()

        for node in sorted(list(all_nodes)):
            if node not in visited:
                dfs(node)

        return cycles

    def __str__(self) -> str:
        lines = []
        if self.chunk_title:
            lines.append(f"SEGMENT: {self.chunk_title}")

        for table in self.tables:
            lines.append(str(table))
        if self.relationships:
            lines.append("\nRELATIONSHIPS:")
            for rel in self.relationships:
                lines.append(f"    {rel}")
        return "\n".join(lines)
