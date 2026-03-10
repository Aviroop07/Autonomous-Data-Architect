from typing import List
from src.pipeline.stage2.models.schema import Schema, Table, Column, ForeignKey, CompositeUnique
from src.pipeline.stage3.models.patch import (
    SchemaPatch, ActionTag, AddColumnPatch, RenameColumnPatch, 
    DeleteColumnPatch, AddTablePatch, MergeTablesPatch, AddRelationshipPatch,
    DeleteRelationshipPatch, UpdatePKPatch, UpsertUniquePatch, DeleteTablePatch
)

def apply_patches(schema: Schema, patches: List[SchemaPatch]):
    """
    Deterministically applies a list of patches to a Schema.
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

def _add_column(schema: Schema, patch: AddColumnPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            if not any(c.name == patch.column_name for c in table.columns):
                table.columns.append(Column(name=patch.column_name))

def _rename_column(schema: Schema, patch: RenameColumnPatch):
    schema.rename_column(patch.table_name, patch.column_name, patch.new_name)

def _delete_column(schema: Schema, patch: DeleteColumnPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            # 1. Remove column from list
            table.columns = [c for c in table.columns if c.name != patch.column_name]
            
            # 2. Update PK if it was deleted
            if table.pk == patch.column_name:
                table.pk = "" 
            
            # 3. CRITICAL: Clean up unique constraints containing this column
            if table.unique:
                new_uniques = []
                for uq in table.unique:
                    if patch.column_name in uq.columns:
                        # Remove the column from the composite list
                        uq.columns = [c for c in uq.columns if c != patch.column_name]
                    
                    # Only keep constraint if it still has columns
                    if uq.columns:
                        new_uniques.append(uq)
                table.unique = new_uniques if new_uniques else None

            # 4. Clean up relationships referencing this deleted column
            if schema.relationships:
                schema.relationships = [
                    r for r in schema.relationships
                    if not (r.referencing_table == patch.table_name and r.referencing_column == patch.column_name)
                ]

def _add_table(schema: Schema, patch: AddTablePatch):
    defn = patch.table_definition
    cols = [Column(name=c) for c in defn.columns]
    uniques = [CompositeUnique(columns=u_defn.columns) for u_defn in (defn.unique or [])]
            
    new_table = Table(
        name=defn.name,
        columns=cols,
        pk=defn.pk,
        unique=uniques if uniques else None
    )
    
    if not any(t.name == new_table.name for t in schema.tables):
        schema.tables.append(new_table)

def _merge_tables(schema: Schema, patch: MergeTablesPatch):
    source_name = patch.source_table
    target_name = patch.target_table
    
    source_table = None
    target_table = None
    for t in schema.tables:
        if t.name == source_name: source_table = t
        if t.name == target_name: target_table = t
        
    if source_table and target_table:
        # 1. Move columns (deduplicate by name)
        existing_cols = {c.name for c in target_table.columns}
        for col in source_table.columns:
            if col.name not in existing_cols:
                target_table.columns.append(col.model_copy(deep=True))
        
        # 2. Merge Uniques (deduplicate by column set)
        if source_table.unique:
            if target_table.unique is None: target_table.unique = []
            for s_uq in source_table.unique:
                source_cols = set(s_uq.columns)
                if not any(set(t_uq.columns) == source_cols for t_uq in target_table.unique):
                    target_table.unique.append(s_uq.model_copy(deep=True))
            
        # 3. Update relationships in schema
        if schema.relationships:
            for rel in schema.relationships:
                if rel.referencing_table == source_name: rel.referencing_table = target_name
                if rel.referred_table == source_name: rel.referred_table = target_name
            
            # Mirror/Loop scrubbing is handled at the end of apply_patches via _cleanup_relationships
        
        # 4. Remove source table
        schema.tables = [t for t in schema.tables if t.name != source_name]

def _add_relationship(schema: Schema, patch: AddRelationshipPatch):
    defn = patch.fk_definition
    new_rel = ForeignKey(
        referencing_table=defn.referencing_table,
        referencing_column=defn.referencing_column,
        referred_table=defn.referred_table
    )
    
    if schema.relationships is None: schema.relationships = []
    
    # Deduplicate identical
    exists = any(
        r.referencing_table == new_rel.referencing_table and 
        r.referencing_column == new_rel.referencing_column and 
        r.referred_table == new_rel.referred_table 
        for r in schema.relationships
    )
    if not exists:
        schema.relationships.append(new_rel)

def _delete_relationship(schema: Schema, patch: DeleteRelationshipPatch):
    if schema.relationships:
        defn = patch.fk_definition
        schema.relationships = [
            r for r in schema.relationships 
            if not (r.referencing_table == defn.referencing_table and 
                    r.referencing_column == defn.referencing_column and 
                    r.referred_table == defn.referred_table)
        ]

def _update_pk(schema: Schema, patch: UpdatePKPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.pk = patch.column_name

def _upsert_unique(schema: Schema, patch: UpsertUniquePatch):
    if patch.unique_definition:
        for table in schema.tables:
            if table.name == patch.table_name:
                new_uq = CompositeUnique(columns=patch.unique_definition.columns)
                if table.unique is None: table.unique = []
                
                # Deduplicate by column set
                new_cols = set(new_uq.columns)
                if not any(set(u.columns) == new_cols for u in table.unique):
                    table.unique.append(new_uq)

def _delete_table(schema: Schema, patch: DeleteTablePatch):
    schema.tables = [t for t in schema.tables if t.name != patch.table_name]
    if schema.relationships:
        schema.relationships = [
            r for r in schema.relationships 
            if r.referencing_table != patch.table_name and r.referred_table != patch.table_name
        ]

def _cleanup_relationships(schema: Schema):
    """
    Final pass to ensure referential integrity.
    """
    if not schema.relationships:
        return

    valid_tables = {t.name: {c.name for c in t.columns} for t in schema.tables}
    new_rels = []
    
    for r in schema.relationships:
        # 1. Table existence
        if r.referencing_table not in valid_tables or r.referred_table not in valid_tables:
            continue
            
        # 2. Column existence
        if r.referencing_column not in valid_tables[r.referencing_table]:
            continue
            
        # 3. Prevent self-referencing unless authorized (hierarchies)
        # (Self-referencing is now permitted to represent reporting hierarchies)
        # if r.referencing_table == r.referred_table:
        #     continue

        new_rels.append(r)
        
    # 4. Deduplicate and scrub mirrored links deterministically
    schema.relationships = _deduplicate_and_scrub_mirrors(new_rels)

def _deduplicate_and_scrub_mirrors(rels: List[ForeignKey]) -> List[ForeignKey]:
    seen = set()
    cleaned_rels = []
    
    # Sort for determinism based on table and column names
    sorted_rels = sorted(rels, key=lambda x: (x.referencing_table, x.referencing_column, x.referred_table))
    
    for r in sorted_rels:
        # Identity check
        core = (r.referencing_table, r.referencing_column, r.referred_table)
        if core in seen:
            continue
        
        # Mirror check: if we already have B -> A, we only add A -> B if it's on a different column
        # and we want to be strict here to avoid cycles. 
        # Deterministic rule: if A < B alphabetically, A -> B is preferred over B -> A if a cycle is risk.
        has_mirror = False
        if r.referencing_table != r.referred_table:
            has_mirror = any(
                sr.referencing_table == r.referred_table and sr.referred_table == r.referencing_table
                for sr in cleaned_rels
            )
        
        if not has_mirror:
            seen.add(core)
            cleaned_rels.append(r)
        else:
            # We already have a link in the opposite direction. 
            # In a strict DAG schema, we should skip this to prevent cycles.
            pass
            
    return cleaned_rels
