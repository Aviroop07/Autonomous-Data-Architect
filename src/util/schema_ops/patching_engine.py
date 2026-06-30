from typing import List, Optional
from src.pipeline.stage2.models.schema import (
    Schema,
    Table,
    Column,
    ForeignKey,
    CompositeUnique,
)
from src.util.schema_ops.schema_patch import (
    SchemaPatch,
    AddColumnPatch,
    RenameColumnPatch,
    DeleteColumnPatch,
    AddTablePatch,
    MergeTablesPatch,
    AddRelationshipPatch,
    DeleteRelationshipPatch,
    UpdatePKPatch,
    UpsertUniquePatch,
    DeleteTablePatch,
    RenameTablePatch,
    UpdateColumnTypePatch,
)
from src.pipeline.stage2.models.registry import TableFactRegistry


def apply_patches(
    schema: Schema,
    patches: List[SchemaPatch],
    registry: Optional[TableFactRegistry] = None,
    owner_fact_ids: List[int] = [],
):
    """
    Deterministically applies a list of patches to a Schema.
    If a registry is provided, it is kept in sync with structural changes.
    """
    for patch in patches:
        if isinstance(patch, AddColumnPatch):
            _add_column(schema, patch)
        elif isinstance(patch, RenameColumnPatch):
            _rename_column(schema, patch)
        elif isinstance(patch, DeleteColumnPatch):
            _delete_column(schema, patch)
        elif isinstance(patch, AddTablePatch):
            _add_table(schema, patch, registry, owner_fact_ids)
        elif isinstance(patch, MergeTablesPatch):
            _merge_tables(schema, patch, registry)
        elif isinstance(patch, AddRelationshipPatch):
            _add_relationship(schema, patch)
        elif isinstance(patch, DeleteRelationshipPatch):
            _delete_relationship(schema, patch)
        elif isinstance(patch, UpdatePKPatch):
            _update_pk(schema, patch)
        elif isinstance(patch, UpsertUniquePatch):
            _upsert_unique(schema, patch)
        elif isinstance(patch, DeleteTablePatch):
            _delete_table(schema, patch, registry)
        elif isinstance(patch, RenameTablePatch):
            _rename_table(schema, patch, registry)
        elif isinstance(patch, UpdateColumnTypePatch):
            _update_column_type(schema, patch)

    _cleanup_relationships(schema)


def _add_column(schema: Schema, patch: AddColumnPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            if not any(c.name == patch.column_name for c in table.columns):
                table.columns.append(
                    Column(name=patch.column_name, data_type=patch.data_type)
                )


def _rename_column(schema: Schema, patch: RenameColumnPatch):
    schema.rename_column(patch.table_name, patch.column_name, patch.new_name)


def _delete_column(schema: Schema, patch: DeleteColumnPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.columns = [c for c in table.columns if c.name != patch.column_name]
            table.primary_key = [c for c in table.primary_key if c != patch.column_name]
            if table.unique:
                new_uniques = []
                for uq in table.unique:
                    uq.columns = [c for c in uq.columns if c != patch.column_name]
                    if uq.columns:
                        new_uniques.append(uq)
                table.unique = new_uniques if new_uniques else None
            if schema.relationships:
                schema.relationships = [
                    r
                    for r in schema.relationships
                    if not (
                        r.referencing_table == patch.table_name
                        and r.referencing_column == patch.column_name
                    )
                ]


def _add_table(
    schema: Schema,
    patch: AddTablePatch,
    registry: Optional[TableFactRegistry] = None,
    owner_fact_ids: List[int] = [],
):
    defn = patch.table_definition
    cols = [Column(name=c.name, data_type=c.data_type) for c in defn.columns]
    uniques = [
        CompositeUnique(columns=u_defn.columns) for u_defn in (defn.unique or [])
    ]
    new_table = Table(
        name=defn.name, columns=cols, primary_key=defn.primary_key, unique=uniques if uniques else None
    )
    if not any(t.name == new_table.name for t in schema.tables):
        schema.tables.append(new_table)
        if registry:
            registry.register_table_facts(new_table.name, owner_fact_ids)


def _merge_tables(
    schema: Schema,
    patch: MergeTablesPatch,
    registry: Optional[TableFactRegistry] = None,
):
    source_name = patch.source_table
    target_name = patch.target_table
    if registry:
        registry.merge_tables(source_name, target_name)
    source_table = None
    target_table = None
    for t in schema.tables:
        if t.name == source_name:
            source_table = t
        if t.name == target_name:
            target_table = t

    if source_table and target_table:
        existing_cols = {c.name for c in target_table.columns}
        for col in source_table.columns:
            if col.name not in existing_cols:
                target_table.columns.append(col.model_copy(deep=True))
        if source_table.unique:
            if target_table.unique is None:
                target_table.unique = []
            for s_uq in source_table.unique:
                if not any(
                    set(t_uq.columns) == set(s_uq.columns)
                    for t_uq in target_table.unique
                ):
                    target_table.unique.append(s_uq.model_copy(deep=True))
        if schema.relationships:
            for rel in schema.relationships:
                if rel.referencing_table == source_name:
                    rel.referencing_table = target_name
                if rel.referred_table == source_name:
                    rel.referred_table = target_name
        schema.tables = [t for t in schema.tables if t.name != source_name]


def _add_relationship(schema: Schema, patch: AddRelationshipPatch):
    defn = patch.fk_definition
    # Defensive check: Only apply if all required fields are present to avoid Pydantic validation crashes
    if not (defn.referencing_table and defn.referencing_column and defn.referred_table):
        return

    # If the FK column doesn't exist yet, create it — inferring the type from the
    # referred table's PK so the schema stays type-consistent without a separate patch.
    table_map = {t.name: t for t in schema.tables}
    ref_table = table_map.get(defn.referencing_table)
    target_table = table_map.get(defn.referred_table)
    if (
        ref_table
        and target_table
        and not any(c.name == defn.referencing_column for c in ref_table.columns)
    ):
        target_pk_col = next(
            (c for c in target_table.columns if c.name == target_table.pk), None
        )
        col_type = (
            target_pk_col.data_type
            if target_pk_col and target_pk_col.data_type
            else "INTEGER"
        )
        print(
            f"  [PatchEngine] ADD_RELATIONSHIP: auto-creating {defn.referencing_table}.{defn.referencing_column}"
            f" ({col_type}) — column was absent, inferred from {defn.referred_table} PK"
        )
        ref_table.columns.append(
            Column(name=defn.referencing_column, data_type=col_type)
        )

    new_rel = ForeignKey(
        referencing_table=defn.referencing_table,
        referencing_column=defn.referencing_column,
        referred_table=defn.referred_table,
    )
    if schema.relationships is None:
        schema.relationships = []

    # Avoid duplicates
    if not any(
        r.referencing_table == new_rel.referencing_table
        and r.referencing_column == new_rel.referencing_column
        and r.referred_table == new_rel.referred_table
        for r in schema.relationships
    ):
        schema.relationships.append(new_rel)


def _delete_relationship(schema: Schema, patch: DeleteRelationshipPatch):
    if schema.relationships and patch.fk_definition:
        defn = patch.fk_definition
        if not (
            defn.referencing_table and defn.referencing_column and defn.referred_table
        ):
            return

        schema.relationships = [
            r
            for r in schema.relationships
            if not (
                r.referencing_table == defn.referencing_table
                and r.referencing_column == defn.referencing_column
                and r.referred_table == defn.referred_table
            )
        ]


def _update_pk(schema: Schema, patch: UpdatePKPatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            table.primary_key = patch.column_name


def _update_column_type(schema: Schema, patch: UpdateColumnTypePatch):
    for table in schema.tables:
        if table.name == patch.table_name:
            for col in table.columns:
                if col.name == patch.column_name:
                    col.data_type = patch.new_type


def _upsert_unique(schema: Schema, patch: UpsertUniquePatch):
    if patch.unique_definition:
        for table in schema.tables:
            if table.name == patch.table_name:
                new_uq = CompositeUnique(columns=patch.unique_definition.columns)
                if table.unique is None:
                    table.unique = []
                if not any(set(u.columns) == set(new_uq.columns) for u in table.unique):
                    table.unique.append(new_uq)


def _delete_table(
    schema: Schema,
    patch: DeleteTablePatch,
    registry: Optional[TableFactRegistry] = None,
):
    schema.tables = [t for t in schema.tables if t.name != patch.table_name]
    if registry:
        registry.delete_table(patch.table_name)
    if schema.relationships:
        schema.relationships = [
            r
            for r in schema.relationships
            if r.referencing_table != patch.table_name
            and r.referred_table != patch.table_name
        ]


def _rename_table(
    schema: Schema,
    patch: RenameTablePatch,
    registry: Optional[TableFactRegistry] = None,
):
    old_name = patch.table_name
    new_name = patch.new_name
    schema.rename_table(old_name, new_name, registry=registry)


def _cleanup_relationships(schema: Schema):
    if not schema.relationships:
        return
    valid_tables = {t.name: {c.name for c in t.columns} for t in schema.tables}
    new_rels = []
    for r in schema.relationships:
        if r.referencing_table in valid_tables and r.referred_table in valid_tables:
            if r.referencing_column in valid_tables[r.referencing_table]:
                new_rels.append(r)

    seen = set()
    deduped = []
    for r in sorted(
        new_rels,
        key=lambda x: (x.referencing_table, x.referencing_column, x.referred_table),
    ):
        core = (r.referencing_table, r.referencing_column, r.referred_table)
        if core not in seen:
            seen.add(core)
            deduped.append(r)
    schema.relationships = deduped
