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
        Normalizes table name and column names.
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
        
        return errors

    def __str__(self) -> str:
        return (
            f"FOREIGN KEY ({self.referencing_table}.{self.referencing_column}) "
            f"REFERENCES {self.referred_table}"
        )

class SchemaSegment(BaseModel):
    chunk_title: str
    tables: List[Table]
    relationships: Optional[List[ForeignKey]] = None

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
        Normalizes the entire schema segment.
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
            errors.append("Schema segment must contain at least one table.")

        table_map = {}
        for table in self.tables:
            errors.extend(table._validate())
            if table.name in table_map:
                errors.append(f"Duplicate table name in segment: {table.name}")
            table_map[table.name] = table

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

        return errors

    def __str__(self) -> str:
        lines = [f"SEGMENT: {self.chunk_title}"]
        for table in self.tables:
            lines.append(str(table))
        if self.relationships:
            lines.append("\nRELATIONSHIPS:")
            for rel in self.relationships:
                lines.append(f"    {rel}")
        return "\n".join(lines)
