from typing import Dict, List, Set
from src.pipeline.stage2.models.schema import (
    Column,
    CompositeUnique,
    FORBIDDEN_TABLE_SUFFIXES,
    ForeignKey,
    Schema,
    Table,
    to_snake_case,
)
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel, Relationship
from src.pipeline.stage2.models.data_types import DataType


def looks_singular_noun(name: str) -> bool:
    """Basic heuristic to avoid verb/plural relationships."""
    candidate = name.upper()
    if candidate.endswith("S") and candidate not in {
        "STATUS",
        "ACCESS",
        "PROCESS",
        "DIAGNOSIS",
        "TV_SERIES",
    }:
        return False
    return True


def _derive_junction_name(
    rel: Relationship,
    participant_tables: List[Table],
    used_names: Set[str],
) -> str:
    """Deterministically name an M:N / n-ary junction table.

    Prefer the relationship's own name when it is a clean singular noun (the extractor
    sometimes emits good associative-entity names like ENROLLMENT); otherwise compose a
    noun name from the participant entities (e.g. FLIGHT_CREW, TASK_CARRIER), matching the
    golden convention and avoiding verb/plural names like OPERATES/RECORDS. Disambiguates
    on collision so two distinct M:N relationships over the same entities do not clash.
    """
    candidate = to_snake_case(rel.name).upper()
    tokens = set(candidate.split("_"))
    is_clean_noun = (
        bool(candidate)
        and looks_singular_noun(candidate)
        and not (tokens & FORBIDDEN_TABLE_SUFFIXES)
    )

    if not is_clean_noun:
        # Compose from participant entity table names (sorted + deduped for determinism).
        parts = sorted({t.name for t in participant_tables})
        composed = "_".join(parts)
        candidate = composed or candidate

    if candidate not in used_names:
        return candidate

    # Collision: append the relationship name, then a numeric suffix, until unique.
    rel_suffix = to_snake_case(rel.name).upper()
    if rel_suffix and f"{candidate}_{rel_suffix}" not in used_names:
        return f"{candidate}_{rel_suffix}"
    i = 2
    while f"{candidate}_{i}" in used_names:
        i += 1
    return f"{candidate}_{i}"


def map_conceptual_to_relational(cm: ConceptualModel) -> Schema:
    tables: List[Table] = []
    relationships_to_add: List[ForeignKey] = []

    entity_tables: Dict[str, Table] = {}

    # 1. Entities to tables
    for entity in cm.entities:
        t_name = to_snake_case(entity.name).upper()
        columns = []
        pk_cols = []

        mva_attributes = []

        # 2. Attributes
        for attr in entity.attributes:
            if attr.is_derived:
                continue
            if attr.is_multivalued:
                mva_attributes.append(attr)
                continue

            c_name = to_snake_case(attr.name).lower()
            if not any(c.name == c_name for c in columns):
                columns.append(Column(name=c_name, data_type=attr.type))

        # 3. PK Selection
        if entity.identifier_attributes:
            pk_cols = [to_snake_case(a).lower() for a in entity.identifier_attributes]
        else:
            entity_attr_names = {
                to_snake_case(a.name).lower()
                for a in entity.attributes
                if not a.is_multivalued and not a.is_derived
            }

            candidate_fd_det = None
            for fd in cm.functional_dependencies:
                det_cols = []
                valid_fd = True
                for det in fd.determinant:
                    if "." in det:
                        e_name, a_name = det.split(".", 1)
                        if e_name.lower() != entity.name.lower():
                            valid_fd = False
                            break
                        det_cols.append(to_snake_case(a_name).lower())
                    else:
                        valid_fd = False
                        break

                if valid_fd:
                    dep_cols = [
                        to_snake_case(dep.split(".", 1)[1]).lower()
                        for dep in fd.dependent
                        if "." in dep
                        and dep.split(".", 1)[0].lower() == entity.name.lower()
                    ]
                    if set(det_cols).union(set(dep_cols)) == entity_attr_names:
                        candidate_fd_det = det_cols
                        break

            if candidate_fd_det:
                pk_cols = candidate_fd_det
            else:
                pk_col = f"{to_snake_case(entity.name).lower()}_id"
                pk_cols = [pk_col]
                if not any(c.name == pk_col for c in columns):
                    columns.append(Column(name=pk_col, data_type=DataType.INTEGER))

        # Ensure every chosen PK member has a column.
        for pk_name in pk_cols:
            if not any(c.name == pk_name for c in columns):
                columns.append(Column(name=pk_name, data_type=DataType.INTEGER))

        # A primary key must be key-eligible (INTEGER/VARCHAR/UUID). If the chosen natural
        # key (from identifier_attributes or an FD) has a non-eligible member -- e.g. a DATE
        # like 'start_date' the extractor picked for an entity with no real identifier --
        # keep those columns as plain attributes and synthesize a surrogate INTEGER PK.
        key_eligible = {DataType.INTEGER, DataType.VARCHAR, DataType.UUID}
        pk_ok = all(
            (c := next((col for col in columns if col.name == pk_name), None))
            is not None
            and c.data_type in key_eligible
            for pk_name in pk_cols
        )
        if not pk_ok:
            surrogate = f"{to_snake_case(entity.name).lower()}_id"
            if not any(c.name == surrogate for c in columns):
                columns.append(Column(name=surrogate, data_type=DataType.INTEGER))
            pk_cols = [surrogate]

        table = Table(name=t_name, primary_key=pk_cols, columns=columns)
        tables.append(table)
        entity_tables[entity.name.lower()] = table

        # 8. Multivalued attributes
        for mva in mva_attributes:
            mva_t_name = f"{t_name}_{to_snake_case(mva.name).upper()}"
            mva_col_name = to_snake_case(mva.name).lower()
            mva_cols = [
                Column(
                    name=mva_col_name,
                    data_type=mva.type,
                )
            ]
            mva_pk_cols = [mva_col_name]

            for pk_c in pk_cols:
                parent_col = next((c for c in columns if c.name == pk_c))
                mva_cols.append(Column(name=pk_c, data_type=parent_col.data_type))
                mva_pk_cols.append(pk_c)
                relationships_to_add.append(
                    ForeignKey(
                        referencing_table=mva_t_name,
                        referencing_column=pk_c,
                        referred_table=t_name,
                    )
                )

            tables.append(
                Table(name=mva_t_name, primary_key=mva_pk_cols, columns=mva_cols)
            )

    # Weak entity pass
    for entity in cm.entities:
        if entity.is_weak and entity.owner:
            owner_t = entity_tables.get(entity.owner.lower())
            child_t = entity_tables.get(entity.name.lower())
            if owner_t and child_t:
                for pk_c in owner_t.primary_key:
                    if not any(c.name == pk_c for c in child_t.columns):
                        owner_col = next((c for c in owner_t.columns if c.name == pk_c))
                        child_t.columns.append(
                            Column(name=pk_c, data_type=owner_col.data_type)
                        )
                    if pk_c not in child_t.primary_key:
                        child_t.primary_key.append(pk_c)
                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=child_t.name,
                            referencing_column=pk_c,
                            referred_table=owner_t.name,
                        )
                    )

    # 4, 5, 6, 7. Relationships
    for rel in cm.relationships:
        if not rel.participants:
            continue

        if rel.degree == "n-ary" or rel.kind == "M:N":
            # Resolve participant tables first -- needed for both naming and FK columns.
            participant_tables: List[Table] = []
            for p in rel.participants:
                p_t = entity_tables.get(p.entity.lower())
                if p_t and p_t not in participant_tables:
                    participant_tables.append(p_t)
            if not participant_tables:
                continue

            # Deterministic, noun-based junction name (avoids verb/plural names like OPERATES).
            t_name = _derive_junction_name(
                rel, participant_tables, {t.name for t in tables}
            )
            columns = []
            pk_cols = []

            for attr in rel.attributes:
                if attr.is_derived or attr.is_multivalued:
                    continue
                c_name = to_snake_case(attr.name).lower()
                if not any(c.name == c_name for c in columns):
                    columns.append(Column(name=c_name, data_type=attr.type))

            for p in rel.participants:
                p_t = entity_tables.get(p.entity.lower())
                if not p_t:
                    continue
                role_prefix = f"{to_snake_case(p.role).lower()}_" if p.role else ""

                for pk_c in p_t.primary_key:
                    fk_col_name = f"{role_prefix}{pk_c}"
                    parent_col = next((c for c in p_t.columns if c.name == pk_c))

                    if not any(c.name == fk_col_name for c in columns):
                        columns.append(
                            Column(name=fk_col_name, data_type=parent_col.data_type)
                        )

                    if fk_col_name not in pk_cols:
                        pk_cols.append(fk_col_name)

                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=t_name,
                            referencing_column=fk_col_name,
                            referred_table=p_t.name,
                        )
                    )

            tables.append(Table(name=t_name, primary_key=pk_cols, columns=columns))

        elif rel.kind == "1:N" and len(rel.participants) == 2:
            p1, p2 = rel.participants
            if p1.cardinality_max != 1:
                child_p, parent_p = p1, p2
            else:
                child_p, parent_p = p2, p1

            child_t = entity_tables.get(child_p.entity.lower())
            parent_t = entity_tables.get(parent_p.entity.lower())

            if child_t and parent_t:
                if child_t.name == parent_t.name:
                    role_prefix = (
                        f"{to_snake_case(parent_p.role).lower()}_"
                        if parent_p.role
                        else f"{to_snake_case(rel.name).lower()}_"
                    )
                else:
                    role_prefix = (
                        f"{to_snake_case(parent_p.role).lower()}_"
                        if parent_p.role
                        else ""
                    )

                for pk_c in parent_t.primary_key:
                    fk_col_name = f"{role_prefix}{pk_c}"
                    parent_col = next((c for c in parent_t.columns if c.name == pk_c))
                    if not any(c.name == fk_col_name for c in child_t.columns):
                        child_t.columns.append(
                            Column(name=fk_col_name, data_type=parent_col.data_type)
                        )

                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=child_t.name,
                            referencing_column=fk_col_name,
                            referred_table=parent_t.name,
                        )
                    )

        elif rel.kind == "1:1" and len(rel.participants) == 2:
            p1, p2 = rel.participants
            if p1.cardinality_min == 1 and p2.cardinality_min != 1:
                child_p, parent_p = p1, p2
            elif p2.cardinality_min == 1 and p1.cardinality_min != 1:
                child_p, parent_p = p2, p1
            else:
                if p1.entity.lower() > p2.entity.lower():
                    child_p, parent_p = p1, p2
                else:
                    child_p, parent_p = p2, p1

            child_t = entity_tables.get(child_p.entity.lower())
            parent_t = entity_tables.get(parent_p.entity.lower())

            if child_t and parent_t:
                fk_cols_added = []
                for pk_c in parent_t.primary_key:
                    fk_col_name = pk_c
                    if child_t.name == parent_t.name:
                        role_prefix = (
                            f"{to_snake_case(parent_p.role).lower()}_"
                            if parent_p.role
                            else f"{to_snake_case(rel.name).lower()}_"
                        )
                        fk_col_name = f"{role_prefix}{pk_c}"
                    else:
                        role_prefix = (
                            f"{to_snake_case(parent_p.role).lower()}_"
                            if parent_p.role
                            else ""
                        )
                        fk_col_name = f"{role_prefix}{pk_c}"

                    parent_col = next((c for c in parent_t.columns if c.name == pk_c))
                    if not any(c.name == fk_col_name for c in child_t.columns):
                        child_t.columns.append(
                            Column(name=fk_col_name, data_type=parent_col.data_type)
                        )

                    fk_cols_added.append(fk_col_name)
                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=child_t.name,
                            referencing_column=fk_col_name,
                            referred_table=parent_t.name,
                        )
                    )

                if fk_cols_added:
                    if child_t.unique is None:
                        child_t.unique = []
                    child_t.unique.append(CompositeUnique(columns=fk_cols_added))

    schema = Schema(tables=tables, relationships=relationships_to_add)
    schema.normalize()
    schema.wire_orphan_fk_columns()
    schema.align_fk_column_types()

    # A2: Enforce validation postcondition
    errors = schema._validate()
    if errors:
        print(
            f"  [Mapper] WARNING: Generated schema failed validation with {len(errors)} errors:"
        )
        for e in errors[:5]:
            print(f"    - {e}")

        # Bounded deterministic repair loop
        for _ in range(3):
            if not schema._validate():
                break

            # FK-target tables are legitimate parent/lookup entities -- never drop them
            # as "hollow" even if they only have a PK (dropping orphans referencing FKs).
            referred = {r.referred_table for r in (schema.relationships or [])}

            seen_t = set()
            unique_tables = []
            for t in schema.tables:
                if not t.columns or not t.primary_key:
                    continue

                # Identify hollow tables (PK-only), exempting composite-PK junctions and
                # FK-target tables.
                non_pk_cols = [c for c in t.columns if c.name not in t.pk_set]
                if (
                    not t.is_composite_pk
                    and t.name not in referred
                    and not non_pk_cols
                    and len(schema.tables) > 1
                ):
                    continue

                if t.name not in seen_t:
                    seen_t.add(t.name)
                    unique_tables.append(t)

            schema.tables = unique_tables

            valid_t_names = {t.name for t in schema.tables}
            seen_r = set()
            unique_rels = []
            for r in schema.relationships or []:
                r_key = (r.referencing_table, r.referencing_column, r.referred_table)
                if (
                    r_key not in seen_r
                    and r.referencing_table in valid_t_names
                    and r.referred_table in valid_t_names
                ):
                    ref_table = next(
                        (t for t in schema.tables if t.name == r.referencing_table),
                        None,
                    )
                    if ref_table and any(
                        c.name == r.referencing_column for c in ref_table.columns
                    ):
                        seen_r.add(r_key)
                        unique_rels.append(r)
            schema.relationships = unique_rels
            schema.normalize()
            # Re-run FK type alignment inside the repair loop: dropping/deduping tables
            # and columns above can leave a referencing column whose type no longer matches
            # the referred PK. align is idempotent and must run every iteration so the
            # type-mismatch postcondition can actually converge (it was previously only
            # run once before the loop, so a surviving mismatch could never be repaired).
            schema.align_fk_column_types()
            # NOTE: table isolation is a non-blocking advisory (see Schema._style_warnings),
            # NOT a structural error -- we deliberately do NOT prune isolated tables here,
            # because that silently deleted legitimately extracted entities.

        final_errors = schema._validate()
        if final_errors:
            raise ValueError(
                f"RelationalMapper failed to repair schema. Remaining errors: {final_errors}"
            )

    # Non-blocking naming/quality advisories (plural names, isolated tables): surfaced, never fatal.
    for w in schema._style_warnings():
        print(f"  [Mapper] STYLE: {w}")

    return schema
