from typing import List
from src.pipeline.stage2.models.schema import SchemaSegment, Table, Column, ForeignKey, CompositeUnique
from src.pipeline.stage3.models.patch import SchemaPatch, ActionTag

def apply_patches(schema: SchemaSegment, patches: List[SchemaPatch]):
    """
    Deterministically applies a list of patches to a SchemaSegment.
    """
    for patch in patches:
        if patch.action == ActionTag.ADD_COLUMN:
            _add_column(schema, patch)
        elif patch.action == ActionTag.RENAME_COLUMN:
            _rename_column(schema, patch)
        elif patch.action == ActionTag.DELETE_COLUMN:
            _delete_column(schema, patch)
        elif patch.action == ActionTag.ADD_TABLE:
            _add_table(schema, patch)
        elif patch.action == ActionTag.MERGE_TABLES:
            _merge_tables(schema, patch)
        elif patch.action == ActionTag.ADD_RELATIONSHIP:
            _add_relationship(schema, patch)
        elif patch.action == ActionTag.DELETE_RELATIONSHIP:
            _delete_relationship(schema, patch)
        elif patch.action == ActionTag.UPDATE_PK:
            _update_pk(schema, patch)
        elif patch.action == ActionTag.UPSERT_UNIQUE:
            _upsert_unique(schema, patch)

def _add_column(schema: SchemaSegment, patch: SchemaPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            if not any(c.name == patch.column_name for c in table.columns):
                table.columns.append(Column(name=patch.column_name))

def _rename_column(schema: SchemaSegment, patch: SchemaPatch):
    schema.rename_column(patch.table_name, patch.column_name, patch.new_name)

def _delete_column(schema: SchemaSegment, patch: SchemaPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.columns = [c for c in table.columns if c.name != patch.column_name]
            if table.pk == patch.column_name:
                table.pk = "" # Warning: PK deleted

def _add_table(schema: SchemaSegment, patch: SchemaPatch):
    if patch.table_definition:
        table_data = patch.table_definition.copy()
        # Robustly handle missing 'pk'
        if 'pk' not in table_data:
            name = table_data.get('name', 'UNKNOWN')
            inferred_pk = name.lower() + "_id"
            cols = table_data.get('columns', [])
            col_names = []
            for c in cols:
                if isinstance(c, dict): col_names.append(c.get('name'))
                elif isinstance(c, str): col_names.append(c)
            
            if inferred_pk in col_names:
                table_data['pk'] = inferred_pk
            elif col_names:
                table_data['pk'] = col_names[0]
            else:
                table_data['pk'] = "id"
                
        new_table = Table(**table_data)
        if not any(t.name == new_table.name for t in schema.tables):
            schema.tables.append(new_table)

def _merge_tables(schema: SchemaSegment, patch: SchemaPatch):
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
        
        # Remove source table
        schema.tables = [t for t in schema.tables if t.name != source_name]

def _add_relationship(schema: SchemaSegment, patch: SchemaPatch):
    if patch.fk_definition:
        # Robustly handle LLM hallucinations
        fk_data = patch.fk_definition.copy()
        if 'source_column' in fk_data and 'referencing_column' not in fk_data:
            fk_data['referencing_column'] = fk_data.pop('source_column')
        if 'source_table' in fk_data and 'referencing_table' not in fk_data:
            fk_data['referencing_table'] = fk_data.pop('source_table')
        if 'target_table' in fk_data and 'referred_table' not in fk_data:
            fk_data['referred_table'] = fk_data.pop('target_table')
        if 'target_column' in fk_data:
            fk_data.pop('target_column') # ForeignKey model doesn't use referred_column (assumes PK)

        new_rel = ForeignKey(**fk_data)
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

def _delete_relationship(schema: SchemaSegment, patch: SchemaPatch):
    if schema.relationships and patch.fk_definition:
        fk_data = patch.fk_definition
        ref_t = fk_data.get('referencing_table') or fk_data.get('source_table')
        ref_c = fk_data.get('referencing_column') or fk_data.get('source_column')
        referred_t = fk_data.get('referred_table') or fk_data.get('target_table')
        
        schema.relationships = [
            r for r in schema.relationships 
            if not (r.referencing_table == ref_t and r.referencing_column == ref_c and r.referred_table == referred_t)
        ]

def _update_pk(schema: SchemaSegment, patch: SchemaPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.pk = patch.column_name

def _upsert_unique(schema: SchemaSegment, patch: SchemaPatch):
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
