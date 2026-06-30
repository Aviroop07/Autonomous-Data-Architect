from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Dict, Set, Any
from pydantic import BaseModel, Field, model_validator, computed_field
from src.util.orchestration.loop_types import LoopOutputModel
from src.pipeline.stage2.models.data_types import DataType

if TYPE_CHECKING:
    from src.pipeline.stage2.models.registry import TableFactRegistry

UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
LOWER_SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")

FORBIDDEN_TABLE_SUFFIXES = {"FACT", "DIM", "ID", "ATTR", "TABLE"}

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


def looks_singular_noun(name: str) -> bool:
    """Heuristic: True unless `name` looks like a plural.

    A name looks plural when it ends in 'S' but not one of SINGULAR_S_SUFFIXES
    (SS/IS/US) and contains no SINGULAR_S_EXCEPTIONS token. Shared by the table-name
    style advisory and the mapper's junction-name acceptability check.
    """
    upper = name.upper()
    if not upper.endswith("S") or upper.endswith(SINGULAR_S_SUFFIXES):
        return True
    tokens = set(upper.split("_"))
    return any(token in SINGULAR_S_EXCEPTIONS for token in tokens)


class Column(BaseModel):
    name: str = Field(description="Column name in lowercase snake_case.")
    data_type: DataType = Field(
        description="The data type of the column.",
    )

    def _validate(self) -> List[str]:
        errors = []
        if not is_lower_snake(self.name):
            errors.append(f"Column must be lowercase snake_case: {self.name}")

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
    primary_key: List[str] = Field(
        default_factory=list, description="Primary key column name(s)."
    )
    unique: Optional[List[CompositeUnique]] = Field(
        default=None, description="Optional list of composite unique constraints."
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_pk(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "pk" in data and "primary_key" not in data:
                pk_val = data.pop("pk")
                data["primary_key"] = [pk_val] if pk_val else []
            elif "primary_key" in data and isinstance(data["primary_key"], str):
                data["primary_key"] = (
                    [data["primary_key"]] if data["primary_key"] else []
                )
        return data

    @computed_field
    @property
    def pk(self) -> str:
        return self.primary_key[0] if self.primary_key else ""

    @property
    def pk_set(self) -> Set[str]:
        return set(self.primary_key)

    @property
    def is_composite_pk(self) -> bool:
        return len(self.primary_key) > 1

    def rename_column(self, old_name: str, new_name: str) -> None:
        """
        Renames a column within the table.
        """
        for col in self.columns:
            if col.name == old_name:
                col.name = new_name

        self.primary_key = [new_name if c == old_name else c for c in self.primary_key]

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
        self.primary_key = [to_snake_case(c).lower() for c in self.primary_key]

        # [HARDENING] Scrub redundant unique constraints
        if self.unique:
            # 1. Remove PK from any composite unique constraint
            # 2. Filter out constraints that become empty or singleton-PK
            new_uniques = []
            for uq in self.unique:
                scrubbed_cols = [c for c in uq.columns if c not in self.pk_set]
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

        for col in self.columns:
            errors.extend(col._validate())
            if col.name in column_names:
                errors.append(f"Duplicate column in {self.name}: {col.name}")

            column_names.add(col.name)

        # Primary key validation
        if not self.primary_key:
            errors.append(f"Table {self.name} must have a primary key")
        else:
            for pk_member in self.primary_key:
                pk_col = next((c for c in self.columns if c.name == pk_member), None)
                if not pk_col:
                    errors.append(
                        f"Primary key '{pk_member}' not found in columns for table {self.name}"
                    )
                else:
                    # [STRICT DATA TYPE CHECK]
                    if (
                        pk_col.data_type
                        and pk_col.data_type not in {DataType.INTEGER, DataType.VARCHAR, DataType.UUID}
                    ):
                        errors.append(
                            f"Primary key '{pk_member}' in table {self.name} must be of type INTEGER, VARCHAR, or UUID (found {pk_col.data_type})."
                        )

        # NOTE: the singular-noun check is intentionally NOT a hard error -- it is a
        # naming-style advisory surfaced via _style_warnings() so it never crashes the
        # mapper's structural postcondition. Junction names are fixed at the source
        # (participant-based naming in the relational mapper).

        # Unique constraints validation
        unique_sets = []
        for unique_constraint in self.unique or []:
            curr_set = set(unique_constraint.columns)
            unique_sets.append(curr_set)

            # Check for Primary Key inclusion
            if self.pk_set.issubset(curr_set):
                if len(curr_set) == len(self.pk_set):
                    errors.append(
                        f"Redundant unique constraint: matches the Primary Key of {self.name}."
                    )
                else:
                    cols_str = ", ".join(unique_constraint.columns)
                    errors.append(
                        f"Redundant unique composite containing PK: in ({cols_str}) for table {self.name}."
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

    def _style_warnings(self) -> List[str]:
        """Non-blocking naming-style advisories (never returned by _validate()).

        These flag quality issues that must NOT crash the structural pipeline.
        Currently: a table name that looks plural rather than singular.
        """
        warnings: List[str] = []
        if not looks_singular_noun(self.name):
            warnings.append(f"Table name should be singular: {self.name}")
        return warnings

    def __str__(self) -> str:
        lines = [f"TABLE {self.name} ("]
        for col in self.columns:
            if col.name in self.pk_set:
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
                        and ref_col.data_type != target_pk_col.data_type
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

        # [HARDENING] Strip single-column UNIQUE constraints on FK columns of junction
        # tables. A junction table has 2+ FK columns; individually unique FK columns
        # forbid value reuse across rows, which is always wrong for bridge entities.
        if self.relationships:
            fk_cols_by_table: Dict[str, Set[str]] = {}
            for rel in self.relationships:
                fk_cols_by_table.setdefault(rel.referencing_table, set()).add(
                    rel.referencing_column
                )
            for table in self.tables:
                fk_cols = fk_cols_by_table.get(table.name, set())
                if len(fk_cols) < 2 or not table.unique:
                    continue
                cleaned = [
                    uq
                    for uq in table.unique
                    if not (len(uq.columns) == 1 and uq.columns[0] in fk_cols)
                ]
                table.unique = cleaned if cleaned else None

    def wire_orphan_fk_columns(self) -> None:
        """
        Declares missing FK relationships for columns that follow the naming convention
        {referenced_table_lower_snake}_id where the referenced table exists in the schema
        but no FK has been declared for that column.

        This repairs the gap where architects include FK columns (e.g. department_id) but
        omit the explicit FK declaration. Only fires when the column already exists, the
        target table exists, the column is not the table's own PK, and no FK is already
        declared for this (table, column) pair.
        """
        if not self.tables:
            return

        table_names = {t.name for t in self.tables}
        existing_fk_pairs: set[tuple[str, str]] = {
            (r.referencing_table, r.referencing_column)
            for r in (self.relationships or [])
        }

        new_fks: list[ForeignKey] = []
        for table in self.tables:
            for col in table.columns:
                if col.name in table.pk_set:
                    continue
                if not col.name.endswith("_id"):
                    continue
                # e.g. department_id -> DEPARTMENT, faculty_member_id -> FACULTY_MEMBER
                candidate = col.name[:-3].upper()
                if candidate not in table_names:
                    continue
                if candidate == table.name:
                    continue
                if (table.name, col.name) in existing_fk_pairs:
                    continue
                new_fks.append(
                    ForeignKey(
                        referencing_table=table.name,
                        referencing_column=col.name,
                        referred_table=candidate,
                    )
                )
                existing_fk_pairs.add((table.name, col.name))

        if new_fks:
            if self.relationships is None:
                self.relationships = []
            for fk in new_fks:
                print(
                    f"  [Wire] Auto-declared FK: {fk.referencing_table}.{fk.referencing_column} -> {fk.referred_table}"
                )
            self.relationships.extend(new_fks)

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
                if ref_col.data_type != target_pk_col.data_type:
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

        # Hollow table check: a table with only its PK and no descriptive columns is
        # evidence of incorrect entity separation. Guard on len > 1 to avoid false
        # positives during single-entity shard generation. EXEMPT tables that are an
        # FK target (referred_table): a PK-only table that other tables reference is a
        # legitimate parent/lookup entity, NOT a collapse candidate -- dropping it would
        # orphan the referencing FKs.
        if len(self.tables) > 1:
            referred = {fk.referred_table for fk in (self.relationships or [])}
            for table in self.tables:
                if table.is_composite_pk or table.name in referred:
                    continue
                non_pk_cols = [c for c in table.columns if c.name not in table.pk_set]
                if not non_pk_cols:
                    errors.append(
                        f"Table '{table.name}' has only PK column(s) {', '.join(table.pk_set)} with no "
                        f"descriptive attributes. Either add columns supported by the facts, "
                        f"or collapse this entity into a VARCHAR attribute of its referencing table."
                    )

        for fk in self.relationships or []:
            errors.extend(fk._validate(table_map))

        # NOTE: table isolation (a non-skeleton table with no relationships) is a
        # naming/quality ADVISORY, not a structural error -- see _style_warnings().
        # It must never be a hard error: doing so previously forced a destructive
        # "prune to largest component" repair that silently deleted legitimately
        # extracted entities. Isolated tables are valid SQL.

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

    def _style_warnings(self) -> List[str]:
        """Non-blocking naming/quality advisories (never returned by _validate()).

        Aggregates per-table advisories (e.g. plural names) and a schema-level
        isolation advisory: a non-skeleton table that participates in no relationship.
        Isolation is advisory, not a hard error, so it can never force a destructive
        repair -- an isolated table is valid SQL and may be a legitimate standalone entity.
        """
        warnings: List[str] = [w for t in self.tables for w in t._style_warnings()]

        if len(self.tables) > 1:
            connected: Set[str] = set()
            for fk in self.relationships or []:
                connected.add(fk.referencing_table.upper())
                connected.add(fk.referred_table.upper())
            for table in self.tables:
                upper = table.name.upper()
                if upper in connected:
                    continue
                warnings.append(
                    f"Table '{table.name}' is strictly isolated (no relationships)."
                )

        return warnings

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
