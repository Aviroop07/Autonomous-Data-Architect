from typing import List
from src.pipeline.stage2.models.schema import SchemaSegment, Table, Column, ForeignKey, CompositeUnique
from src.pipeline.stage3.models.patch import (
    SchemaPatch, ActionTag, AddColumnPatch, RenameColumnPatch, 
    DeleteColumnPatch, AddTablePatch, MergeTablesPatch, AddRelationshipPatch,
    DeleteRelationshipPatch, UpdatePKPatch, UpsertUniquePatch, DeleteTablePatch
)

def apply_patches(schema: SchemaSegment, patches: List[SchemaPatch]):
    """
    Deterministically applies a list of patches to a SchemaSegment.
    """
    for patch in patches:
        if isinstance(patch, AddColumnPatch):
            _add_column(schema, patch)
        elif isinstance(patch, RenameColumnPatch):
            _rename_column(schema, patch)
        elif isinstance(patch, DeleteColumnPatch):
            _delete_column(schema, patch)
        elif isinstance(patch, AddTablePatch):
            _add_table(schema, patch)
        elif isinstance(patch, MergeTablesPatch):
            _merge_tables(schema, patch)
        elif isinstance(patch, AddRelationshipPatch):
            _add_relationship(schema, patch)
        elif isinstance(patch, DeleteRelationshipPatch):
            _delete_relationship(schema, patch)
        elif isinstance(patch, UpdatePKPatch):
            _update_pk(schema, patch)
        elif isinstance(patch, UpsertUniquePatch):
            _upsert_unique(schema, patch)
        elif isinstance(patch, DeleteTablePatch):
            _delete_table(schema, patch)
    
    # Final global check to scrub orphans and duplicates
    _cleanup_relationships(schema)

def _add_column(schema: SchemaSegment, patch: AddColumnPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            if not any(c.name == patch.column_name for c in table.columns):
                table.columns.append(Column(name=patch.column_name, data_type=patch.data_type or "TEXT"))

def _rename_column(schema: SchemaSegment, patch: RenameColumnPatch):
    schema.rename_column(patch.table_name, patch.column_name, patch.new_name)

def _delete_column(schema: SchemaSegment, patch: DeleteColumnPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.columns = [c for c in table.columns if c.name != patch.column_name]
            if table.pk == patch.column_name:
                table.pk = "" # Warning: PK deleted
            
            # Clean up relationships referencing this deleted column
            if schema.relationships:
                schema.relationships = [
                    r for r in schema.relationships
                    if not (r.referencing_table == patch.table_name and r.referencing_column == patch.column_name)
                ]

def _add_table(schema: SchemaSegment, patch: AddTablePatch):
    defn = patch.table_definition
    
    # Convert string columns to Column objects
    cols = [Column(name=c, data_type="TEXT") for c in defn.columns]
    
    # Convert nested lists to CompositeUnique objects
    uniques = []
    if defn.unique:
        for u_cols in defn.unique:
            uniques.append(CompositeUnique(columns=u_cols))
            
    new_table = Table(
        name=defn.name,
        columns=cols,
        pk=defn.pk,
        unique=uniques if uniques else None
    )
    
    if not any(t.name == new_table.name for t in schema.tables):
        schema.tables.append(new_table)

def _merge_tables(schema: SchemaSegment, patch: MergeTablesPatch):
    source_name = patch.source_table
    target_name = patch.target_table
    
    source_table = None
    target_table = None
    for t in schema.tables:
        if t.name == source_name: source_table = t
        if t.name == target_name: target_table = t
        
    if source_table and target_table:
        # Move columns
        existing_cols = {c.name for c in target_table.columns}
        for col in source_table.columns:
            if col.name not in existing_cols:
                target_table.columns.append(col.model_copy(deep=True))
        
        # Merge Uniques
        if source_table.unique:
            if target_table.unique is None: target_table.unique = []
            target_table.unique.extend([u.model_copy(deep=True) for u in source_table.unique])
            
        # Update relationships in schema
        if schema.relationships:
            for rel in schema.relationships:
                if rel.referencing_table == source_name: rel.referencing_table = target_name
                if rel.referred_table == source_name: rel.referred_table = target_name
            
            # Remove any self-references that were created by the merge (e.g., A->B becomes A->A)
            # UNLESS they are intended (we'll be conservative and just deduplicate for now)
            schema.relationships = _deduplicate_relationships(schema.relationships)
        
        # Remove source table
        schema.tables = [t for t in schema.tables if t.name != source_name]

def _add_relationship(schema: SchemaSegment, patch: AddRelationshipPatch):
    defn = patch.fk_definition
    new_rel = ForeignKey(
        referencing_table=defn.referencing_table,
        referencing_column=defn.referencing_column,
        referred_table=defn.referred_table
    )
    
    if schema.relationships is None:
        schema.relationships = []
        
    # Check for duplication
    exists = False
    for r in schema.relationships:
        if (r.referencing_table == new_rel.referencing_table and 
            r.referencing_column == new_rel.referencing_column and 
            r.referred_table == new_rel.referred_table):
            exists = True
            break
    if not exists:
        schema.relationships.append(new_rel)

def _delete_relationship(schema: SchemaSegment, patch: DeleteRelationshipPatch):
    if schema.relationships:
        defn = patch.fk_definition
        ref_t = defn.referencing_table
        ref_c = defn.referencing_column
        referred_t = defn.referred_table
        
        schema.relationships = [
            r for r in schema.relationships 
            if not (r.referencing_table == ref_t and r.referencing_column == ref_c and r.referred_table == referred_t)
        ]

def _update_pk(schema: SchemaSegment, patch: UpdatePKPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.pk = patch.column_name

def _upsert_unique(schema: SchemaSegment, patch: UpsertUniquePatch):
    if patch.unique_definition:
        for table in schema.tables:
            if table.name == patch.table_name:
                new_uq = CompositeUnique(**patch.unique_definition)
                if table.unique is None: table.unique = []
                # Simple check for exists
                exists = False
                for u in table.unique:
                    if set(u.columns) == set(new_uq.columns):
                        exists = True
                        break
                if not exists:
                    table.unique.append(new_uq)

def _delete_table(schema: SchemaSegment, patch: DeleteTablePatch):
    schema.tables = [t for t in schema.tables if t.name != patch.table_name]
    # Also clean up relationships
    if schema.relationships:
        schema.relationships = [
            r for r in schema.relationships 
            if r.referencing_table != patch.table_name and r.referred_table != patch.table_name
        ]

def _cleanup_relationships(schema: SchemaSegment):
    """
    Final pass to ensure referential integrity:
    1. Removes FKs referring to non-existent tables.
    2. Removes FKs referring to non-existent columns.
    3. Removes self-referencing FKs (strictly enforcing non-circular logic).
    4. Deduplicates identical FKs.
    """
    if not schema.relationships:
        return

    valid_tables = {t.name: {c.name for c in t.columns} for t in schema.tables}
    new_rels = []
    
    for r in schema.relationships:
        # Check table existence
        if r.referencing_table not in valid_tables or r.referred_table not in valid_tables:
            continue
            
        # Check column existence
        if r.referencing_column not in valid_tables[r.referencing_table]:
            continue
            
        # Prevent self-referencing cycles (Strict Rule)
        if r.referencing_table == r.referred_table:
            continue
            
        new_rels.append(r)
        
    schema.relationships = _deduplicate_relationships(new_rels)

def _deduplicate_relationships(rels: List[ForeignKey]) -> List[ForeignKey]:
    seen = set()
    unique_rels = []
    for r in rels:
        key = (r.referencing_table, r.referencing_column, r.referred_table)
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)
    return unique_rels
