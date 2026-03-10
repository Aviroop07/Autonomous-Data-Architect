import re
import string
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

UPPER_SNAKE = re.compile(r"^[A-Z]+(?:_[A-Z0-9]+)*$")
LOWER_SNAKE = re.compile(r"^[a-z]+(?:_[a-z0-9]+)*$")

FORBIDDEN_TABLE_SUFFIXES = {"FACT", "DIM", "ID", "ATTR", "TABLE"}

def is_upper_snake(value: str) -> bool:
    return bool(UPPER_SNAKE.fullmatch(value))

def is_lower_snake(value: str) -> bool:
    return bool(LOWER_SNAKE.fullmatch(value))

def to_snake_case(s: str) -> str:
    """
    Converts a string to snake_case.
    """
    s = s.strip()
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
    return re.sub(r'_+', '_', s).strip('_')

def expected_pk_name(table_name: str) -> str:
    """
    Standard surrogate key rule: table_name.lower()_id
    """
    return table_name.lower() + "_id"

class Column(BaseModel):
    name: str = Field(description="Column name in lowercase snake_case.")

    def _validate(self) -> List[str]:
        errors = []
        if not is_lower_snake(self.name):
            errors.append(f"Column must be lowercase snake_case: {self.name}")
        return errors

    def __str__(self) -> str:
        return self.name

class CompositeUnique(BaseModel):
    columns: List[str] = Field(description="List of column names that form a composite unique constraint.")

    def __str__(self) -> str:
        cols = ", ".join(self.columns)
        return f"UNIQUE({cols})"

class Table(BaseModel):
    name: str = Field(description="Table name in UPPER_SNAKE_CASE.")
    columns: List[Column]
    pk: str = Field(description="Primary key column name (usually table_name_id).")
    unique: Optional[List[CompositeUnique]] = None

    def rename_column(self, old_name: str, new_name: str):
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

    def normalize(self):
        """
        Normalizes table name and column names, and scrubs redundant unique constraints.
        """
        old_name = self.name
        self.name = to_snake_case(self.name).upper()
        
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
            # Remove any unique constraint that only contains the PK column
            self.unique = [
                uq for uq in self.unique 
                if not (len(uq.columns) == 1 and uq.columns[0] == self.pk)
            ]
            if not self.unique:
                self.unique = None

    def _validate(self) -> List[str]:
        errors = []

        # Table name format
        if not is_upper_snake(self.name):
            errors.append(f"Table name must be UPPER_SNAKE_CASE: {self.name}")

        normalized_tokens = set(self.name.split("_"))
        for suffix in FORBIDDEN_TABLE_SUFFIXES:
            if suffix in normalized_tokens:
                errors.append(f"Forbidden word/suffix in table name: {self.name} ('{suffix}')")

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
        if not self.pk:
            errors.append(f"Table {self.name} must have a primary key")
        else:
            if self.pk not in column_names:
                errors.append(f"Primary key '{self.pk}' not found in columns for table {self.name}")
            
            # Singular check (heuristic)
            if self.name.endswith("S") and not self.name.endswith("SS"):
                 # This is a bit risky but we have a rule for singular names.
                 # Let's check for most common plural patterns
                 if self.name not in ["STATUS", "ACCESS", "PROCESS"]:
                    errors.append(f"Table name should be singular: {self.name}")

        # Unique constraints validation
        for unique_constraint in (self.unique or []):
            if self.pk in unique_constraint.columns:
                if len(unique_constraint.columns) == 1:
                    errors.append(f"Redundant unique singleton: Column '{self.pk}' is already the Primary Key of {self.name}.")
                else:
                    cols_str = ", ".join(unique_constraint.columns)
                    errors.append(f"Redundant unique composite containing PK: '{self.pk}' in ({cols_str}) for table {self.name}.")
                
            for col_name in unique_constraint.columns:
                if col_name not in column_names:
                    errors.append(f"Unique constraint references unknown column '{col_name}' in table {self.name}")

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
    referencing_table: str
    referencing_column: str
    referred_table: str

    def _validate(self, tables_map) -> List[str]:
        errors = []
        if self.referencing_table not in tables_map:
            errors.append(f"FK error: Referencing table '{self.referencing_table}' does not exist.")
        else:
            ref_table = tables_map[self.referencing_table]
            if not any(c.name == self.referencing_column for c in ref_table.columns):
                errors.append(f"FK error: Column '{self.referencing_column}' not found in table '{self.referencing_table}'.")

        if self.referred_table not in tables_map:
            errors.append(f"FK error: Referred table '{self.referred_table}' does not exist.")
        
        if self.referencing_table in tables_map:
            ref_table = tables_map[self.referencing_table]
            if ref_table.pk == self.referencing_column:
                errors.append(f"FK error: Column '{self.referencing_column}' is the Primary Key of '{self.referencing_table}'. Using a PK as an FK is prohibited.")

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

class Schema(BaseModel):
    version: Optional[str] = "1.0"
    chunk_title: Optional[str] = None
    tables: List[Table]
    relationships: Optional[List[ForeignKey]] = None

    def get_table_map(self) -> Dict[str, Table]:
        return {t.name: t for t in self.tables}

    def rename_table(self, old_name: str, new_name: str):
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

    def rename_column(self, table_name: str, old_col_name: str, new_col_name: str):
        """
        Renames a column in a specific table and updates all its relationship occurrences.
        """
        for table in self.tables:
            if table.name == table_name:
                table.rename_column(old_col_name, new_col_name)
                
        if self.relationships:
            for rel in self.relationships:
                if (rel.referencing_table == table_name and 
                    rel.referencing_column == old_col_name):
                    rel.referencing_column = new_col_name

    def normalize(self):
        """
        Normalizes the entire schema.
        """
        # First ensure all names are trimmed and follow snake_case
        for table in self.tables:
            old_table_name = table.name
            table.normalize()
            new_table_name = table.name
            
            # If table name changed, update relationships
            if old_table_name != new_table_name:
                if self.relationships:
                    for rel in self.relationships:
                        if rel.referencing_table == old_table_name:
                            rel.referencing_table = new_table_name
                        if rel.referred_table == old_table_name:
                            rel.referred_table = new_table_name

    def _validate(self) -> List[str]:
        errors = []
        if not self.tables:
            errors.append("Schema must contain at least one table.")

        table_map = self.get_table_map()
        for table in self.tables:
            errors.extend(table._validate())
            # Duplicate check handled by get_table_map in a way, but let's be explicit
            
        seen_tables = set()
        for table in self.tables:
            if table.name in seen_tables:
                errors.append(f"Duplicate table name in schema: {table.name}")
            seen_tables.add(table.name)

        relationship_counts = {t.name: 0 for t in self.tables}
        for fk in (self.relationships or []):
            fk_errors = fk._validate(table_map)
            errors.extend(fk_errors)
            
            if fk.referencing_table in relationship_counts:
                relationship_counts[fk.referencing_table] += 1
            if fk.referred_table in relationship_counts:
                relationship_counts[fk.referred_table] += 1

        # Check for isolated tables if there are multiple tables
        if len(self.tables) > 1:
            for t_name, count in relationship_counts.items():
                if count == 0:
                    errors.append(f"Table '{t_name}' is isolated (no relationships).")

        # Cycle detection
        cycles = self.detect_cycles()
        for cycle in cycles:
            cycle_str = " -> ".join(cycle)
            # [REFINEMENT] Column-level cycles are the true hard dependencies.
            # However, with PK != FK, column-level cycles are technically impossible.
            # We report them if they somehow occur.
            errors.append(f"FK error: Circular dependency detected (column-level): {cycle_str}")

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
        adj: Dict[str, set] = {}
        all_nodes = set()
        
        for rel in self.relationships:
            source_node = f"{rel.referencing_table}.{rel.referencing_column}"
            # Fetch the PK of the target table
            target_pk = "id" # Default fallback
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
        
        def dfs(node: str):
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
        else:
            lines.append("=== GLOBAL SCHEMA ===")
            
        for table in self.tables:
            lines.append(str(table))
        if self.relationships:
            lines.append("\nRELATIONSHIPS:")
            for rel in self.relationships:
                lines.append(f"    {rel}")
        return "\n".join(lines)
