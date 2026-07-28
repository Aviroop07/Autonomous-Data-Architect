import logging
from typing import Dict, List, Optional, Set
from src.util.schema_model.schema import (
    Column,
    CompositeUnique,
    FORBIDDEN_TABLE_SUFFIXES,
    ForeignKey,
    Schema,
    Table,
    looks_singular_noun,
    to_snake_case,
)
from src.pipeline.stage2.mapper.conceptual_model import ConceptualModel, Relationship
from src.util.schema_model.data_types import DataType

logger = logging.getLogger(__name__)


def _resolve_pk_column(
    columns: List[Column], pk_name: str, *, table_name: str, purpose: str
) -> Optional[Column]:
    """Find a primary-key column by name, or return None with a warning.

    Five call sites used a bare `next((c for c in ... if c.name == pk_c))` with
    no default, each relying on the invariant that every name in a table's
    primary_key also appears in its columns. That invariant is real at column
    creation time but is broken later by the weak-entity pass and by
    adjudicator-driven identifier rewrites, and when it broke the failure was a
    bare StopIteration from deep inside the mapper with no indication of which
    table or key was at fault.

    Returning None rather than synthesising a column is deliberate: a foreign
    key pointing at a primary key that does not exist would produce a schema
    that fails its own validation, so the caller skips that key instead.
    """
    column = next((c for c in columns if c.name == pk_name), None)
    if column is None:
        logger.warning(
            "  [Mapper] Table '%s' declares primary-key column '%s' which is not "
            "among its columns; skipping %s that depends on it.",
            table_name,
            pk_name,
            purpose,
        )
    return column


# looks_singular_noun is imported from util.schema_model.schema, which has always
# claimed in its own docstring to be "shared by ... the mapper's junction-name
# acceptability check" -- while this module shadowed it with a weaker copy that
# matched a hardcoded domain word set against the WHOLE name instead of
# tokenizing. The two disagreed on half the names tested, and always in the same
# direction: the local copy called a singular noun plural (ORDER_STATUS,
# PATIENT_DIAGNOSIS, ADDRESS, ANALYSIS, CAMPUS, BONUS), so the mapper rejected
# perfectly good relationship names for junction tables and fell back to
# composing one from the participants. The canonical version also carries no
# domain vocabulary -- its exceptions are English morphology (NEWS/SERIES/
# SPECIES) plus the SS/IS/US suffixes.


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

    # Collision. A SELF-REFERENTIAL relationship reaches here by construction:
    # its participants dedupe to one table, so the composed name IS that table's
    # name, which already exists. Participant roles are the right
    # disambiguator -- they are what distinguishes the two ends of a
    # self-reference, and the extractor is instructed to supply them precisely
    # for this case.
    role_parts = sorted(
        {to_snake_case(p.role).upper() for p in rel.participants if p.role}
    )
    for suffix in ("_".join(role_parts), *role_parts):
        if suffix and _is_new_token(suffix, candidate):
            proposed = f"{candidate}_{suffix}"
            if proposed not in used_names:
                return proposed

    # Then the relationship's own name -- but only if it contributes a token the
    # candidate does not already have. Without that guard a relationship named
    # after its own participant produced a doubled name (an observed live run
    # emitted a junction called <TABLE>_<TABLE>), which says nothing about what
    # the table is for.
    rel_suffix = to_snake_case(rel.name).upper()
    if (
        rel_suffix
        and _is_new_token(rel_suffix, candidate)
        and f"{candidate}_{rel_suffix}" not in used_names
    ):
        return f"{candidate}_{rel_suffix}"

    i = 2
    while f"{candidate}_{i}" in used_names:
        i += 1
    logger.info(
        "  [Mapper] Junction table for relationship '%s' fell back to the "
        "numeric name '%s_%d'; neither participant roles nor the relationship "
        "name added anything distinguishing.",
        rel.name,
        candidate,
        i,
    )
    return f"{candidate}_{i}"


def _is_new_token(suffix: str, candidate: str) -> bool:
    """Does `suffix` contribute at least one token `candidate` lacks?

    Guards against appending a word the name already contains, which produces
    duplicated names like <TABLE>_<TABLE> that carry no information.
    """
    return bool(set(suffix.split("_")) - set(candidate.split("_")))


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
                columns.append(
                    Column(
                        name=c_name,
                        data_type=attr.type,
                        is_nullable=attr.is_nullable,
                        source_fact_ids=attr.source_fact_ids,
                    )
                )

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
                    columns.append(
                        Column(
                            name=pk_col,
                            data_type=DataType.INTEGER,
                            source_fact_ids=entity.source_fact_ids,
                        )
                    )

        # Ensure every chosen PK member has a column.
        for pk_name in pk_cols:
            if not any(c.name == pk_name for c in columns):
                columns.append(
                    Column(
                        name=pk_name,
                        data_type=DataType.INTEGER,
                        source_fact_ids=entity.source_fact_ids,
                    )
                )

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
                columns.append(
                    Column(
                        name=surrogate,
                        data_type=DataType.INTEGER,
                        source_fact_ids=entity.source_fact_ids,
                    )
                )
            pk_cols = [surrogate]

        table = Table(
            name=t_name,
            primary_key=pk_cols,
            columns=columns,
            source_fact_ids=entity.source_fact_ids,
        )
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
                    source_fact_ids=mva.source_fact_ids,
                )
            ]

            key_eligible = {DataType.INTEGER, DataType.VARCHAR, DataType.UUID}
            if mva.type in key_eligible:
                mva_pk_cols = [mva_col_name]
            else:
                surrogate = f"{mva_col_name}_id"
                mva_cols.append(
                    Column(
                        name=surrogate,
                        data_type=DataType.INTEGER,
                        source_fact_ids=mva.source_fact_ids,
                    )
                )
                mva_pk_cols = [surrogate]

            for pk_c in pk_cols:
                parent_col = _resolve_pk_column(
                    columns,
                    pk_c,
                    table_name=t_name,
                    purpose="a multi-valued-attribute table",
                )
                if parent_col is None:
                    continue
                mva_cols.append(
                    Column(
                        name=pk_c,
                        data_type=parent_col.data_type,
                        source_fact_ids=mva.source_fact_ids,
                    )
                )
                mva_pk_cols.append(pk_c)
                relationships_to_add.append(
                    ForeignKey(
                        referencing_table=mva_t_name,
                        referencing_column=pk_c,
                        referred_table=t_name,
                        source_fact_ids=mva.source_fact_ids,
                    )
                )

            tables.append(
                Table(
                    name=mva_t_name,
                    primary_key=mva_pk_cols,
                    columns=mva_cols,
                    source_fact_ids=mva.source_fact_ids,
                )
            )

    # Weak entity pass
    for entity in cm.entities:
        if entity.is_weak and entity.owner:
            owner_t = entity_tables.get(entity.owner.lower())
            child_t = entity_tables.get(entity.name.lower())
            if owner_t and child_t:
                for pk_c in owner_t.primary_key:
                    if not any(c.name == pk_c for c in child_t.columns):
                        owner_col = _resolve_pk_column(
                            owner_t.columns,
                            pk_c,
                            table_name=owner_t.name,
                            purpose="a weak-entity identifying key",
                        )
                        if owner_col is None:
                            continue
                        child_t.columns.append(
                            Column(
                                name=pk_c,
                                data_type=owner_col.data_type,
                                source_fact_ids=entity.source_fact_ids,
                            )
                        )
                    if pk_c not in child_t.primary_key:
                        child_t.primary_key.append(pk_c)
                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=child_t.name,
                            referencing_column=pk_c,
                            referred_table=owner_t.name,
                            source_fact_ids=entity.source_fact_ids,
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
                    columns.append(
                        Column(
                            name=c_name,
                            data_type=attr.type,
                            source_fact_ids=attr.source_fact_ids,
                        )
                    )

            for position, p in enumerate(rel.participants, start=1):
                p_t = entity_tables.get(p.entity.lower())
                if not p_t:
                    continue
                role_prefix = f"{to_snake_case(p.role).lower()}_" if p.role else ""

                for pk_c in p_t.primary_key:
                    fk_col_name = f"{role_prefix}{pk_c}"
                    # A SELF-REFERENTIAL relationship without roles sends both
                    # ends through here with the identical column name, so the
                    # dedup check below collapsed them into ONE foreign key. The
                    # resulting single-FK junction was then classified hollow and
                    # dropped, deleting the relationship outright -- verified on a
                    # self-referencing M:N, which produced zero foreign keys.
                    # These are common (prerequisite, supersedes, manager-of,
                    # part-of), so losing them silently is expensive.
                    #
                    # Roles are the proper disambiguator and are used when
                    # present. Falling back to the participant's position is
                    # deterministic and domain-free; it keeps both ends, which
                    # matters far more than the column being prettily named.
                    # Participants are the outer loop and this table's primary-key
                    # members the inner one, so a name already present can only
                    # have come from a DIFFERENT participant -- i.e. exactly the
                    # self-reference case.
                    if any(c.name == fk_col_name for c in columns):
                        fk_col_name = f"{pk_c}_{position}"
                        logger.info(
                            "  [Mapper] Junction '%s' has two participants resolving "
                            "to table '%s' with no distinguishing roles; naming this "
                            "end's foreign key '%s' by position. Supplying roles on "
                            "the relationship would give it a meaningful name.",
                            t_name,
                            p_t.name,
                            fk_col_name,
                        )
                    parent_col = _resolve_pk_column(
                        p_t.columns,
                        pk_c,
                        table_name=p_t.name,
                        purpose="a junction-table FK",
                    )
                    if parent_col is None:
                        continue

                    if not any(c.name == fk_col_name for c in columns):
                        columns.append(
                            Column(
                                name=fk_col_name,
                                data_type=parent_col.data_type,
                                source_fact_ids=rel.source_fact_ids,
                            )
                        )

                    if fk_col_name not in pk_cols:
                        pk_cols.append(fk_col_name)

                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=t_name,
                            referencing_column=fk_col_name,
                            referred_table=p_t.name,
                            source_fact_ids=rel.source_fact_ids,
                        )
                    )

            tables.append(
                Table(
                    name=t_name,
                    primary_key=pk_cols,
                    columns=columns,
                    source_fact_ids=rel.source_fact_ids,
                )
            )

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

                # child_p.cardinality_min == 0 means an instance of the child
                # entity need not participate in this relationship at all
                # (e.g. a PATIENT need not have INSURANCE) -- the synthesized
                # FK is nullable in exactly that case, never inferred from
                # naming. Unspecified (None) defaults to required (False),
                # the existing safe default for a schema with no cardinality
                # info at all.
                fk_is_nullable = child_p.cardinality_min == 0
                for pk_c in parent_t.primary_key:
                    fk_col_name = f"{role_prefix}{pk_c}"
                    parent_col = _resolve_pk_column(
                        parent_t.columns,
                        pk_c,
                        table_name=parent_t.name,
                        purpose="a foreign key",
                    )
                    if parent_col is None:
                        continue
                    if not any(c.name == fk_col_name for c in child_t.columns):
                        child_t.columns.append(
                            Column(
                                name=fk_col_name,
                                data_type=parent_col.data_type,
                                is_nullable=fk_is_nullable,
                                source_fact_ids=rel.source_fact_ids,
                            )
                        )

                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=child_t.name,
                            referencing_column=fk_col_name,
                            referred_table=parent_t.name,
                            source_fact_ids=rel.source_fact_ids,
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
                # Same rule as the 1:N branch above -- cardinality_min == 0 on
                # whichever participant ended up as child_p (including via the
                # alphabetical tiebreak, when cardinality info didn't clearly
                # pick a side) means that side's participation is optional, so
                # its FK is nullable. Unspecified/None or 1 both default to
                # required (False), matching the tiebreak's own "assume
                # required unless told otherwise" stance.
                fk_is_nullable = child_p.cardinality_min == 0
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

                    parent_col = _resolve_pk_column(
                        parent_t.columns,
                        pk_c,
                        table_name=parent_t.name,
                        purpose="a foreign key",
                    )
                    if parent_col is None:
                        continue
                    if not any(c.name == fk_col_name for c in child_t.columns):
                        child_t.columns.append(
                            Column(
                                name=fk_col_name,
                                data_type=parent_col.data_type,
                                is_nullable=fk_is_nullable,
                                source_fact_ids=rel.source_fact_ids,
                            )
                        )

                    fk_cols_added.append(fk_col_name)
                    relationships_to_add.append(
                        ForeignKey(
                            referencing_table=child_t.name,
                            referencing_column=fk_col_name,
                            referred_table=parent_t.name,
                            source_fact_ids=rel.source_fact_ids,
                        )
                    )

                if fk_cols_added:
                    if child_t.unique is None:
                        child_t.unique = []
                    child_t.unique.append(CompositeUnique(columns=fk_cols_added))

        else:
            # The three branches above cover n-ary/M:N, binary 1:N and binary
            # 1:1. Anything else fell off the end of the chain and vanished
            # with no trace. Two ways to get here, both reachable:
            #   - a `kind` outside the Literal, which the adjudicator could
            #     previously write (now blocked at its source, but the mapper
            #     should not depend on that being the only writer);
            #   - a NON-n-ary relationship declared 1:N or 1:1 with a
            #     participant count other than two, which no branch matches.
            # Logged rather than repaired: guessing a cardinality would invent
            # a foreign key the specification never stated.
            logger.warning(
                "  [Mapper] Relationship '%s' matched no mapping rule "
                "(kind=%r, degree=%r, %d participant(s)); no foreign key was "
                "generated for it. Facts %s are unrepresented.",
                rel.name,
                rel.kind,
                rel.degree,
                len(rel.participants),
                rel.source_fact_ids or "[]",
            )

    schema = Schema(tables=tables, relationships=relationships_to_add)
    schema.normalize()
    schema.wire_orphan_fk_columns()
    schema.align_fk_column_types()

    # A2: Enforce validation postcondition
    errors = schema._validate()
    if errors:
        logger.warning(
            "  [Mapper] Generated schema failed validation with %d error(s):",
            len(errors),
        )
        for e in errors[:5]:
            logger.warning("    - %s", e)

        # Bounded deterministic repair loop
        for _ in range(3):
            if not schema._validate():
                break

            # FK-target tables are legitimate parent/lookup entities -- never drop them
            # as "hollow" even if they only have a PK (dropping orphans referencing FKs).
            referred = {r.referred_table for r in (schema.relationships or [])}

            # Facts that some OTHER table also carries. A hollow table whose facts
            # all appear elsewhere is genuinely redundant; one holding the only
            # copy of a fact is not, and dropping it deletes that fact from the
            # schema entirely.
            #
            # Measured on a live run: the extractor emitted a PACKAGE entity with
            # no attributes and no identifier, from two facts -- one asserting the
            # entity exists, one asserting a parent contains several of them --
            # but did NOT emit the containing relationship, so nothing referenced
            # PACKAGE and the FK-target exemption above did not apply. PACKAGE was
            # dropped, and Stage 3 then could not extract the fanout that second
            # fact states. A Stage 2 cleanup silently cost a Stage 3 constraint.
            #
            # The root cause is upstream (the relationship should have been
            # extracted), but this is the deterministic backstop for the whole
            # class: never let a cleanup step be the reason a fact vanishes.
            facts_held_elsewhere: Dict[str, Set[int]] = {}
            for t in schema.tables:
                own = set(t.source_fact_ids or [])
                for c in t.columns:
                    own.update(c.source_fact_ids or [])
                facts_held_elsewhere[t.name] = own
            all_fact_ids: Set[int] = set()
            for ids in facts_held_elsewhere.values():
                all_fact_ids |= ids

            seen_t = set()
            unique_tables = []
            for t in schema.tables:
                # Every drop below is logged. These were bare `continue`s, so a
                # table extracted from the spec could disappear between the
                # conceptual model and the shipped schema with no trace, and
                # the fact registry was never told -- leaving FK provenance and
                # uncovered_fact_ids describing tables that no longer exist.
                if not t.columns or not t.primary_key:
                    logger.warning(
                        "  [Mapper] Dropping table '%s': %s. Its source facts are "
                        "no longer represented in the schema.",
                        t.name,
                        "no columns" if not t.columns else "no primary key",
                    )
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
                    own_facts = facts_held_elsewhere.get(t.name, set())
                    elsewhere: Set[int] = set()
                    for other_name, ids in facts_held_elsewhere.items():
                        if other_name != t.name:
                            elsewhere |= ids
                    exclusive = own_facts - elsewhere
                    if exclusive:
                        # Deliberately still dropped, and deliberately noisy about
                        # it. KEEPING it was tried and is worse: Schema._validate()
                        # rejects a primary-key-only table outright, so this drop
                        # is what SATISFIES validation -- retaining the table turns
                        # a silent constraint loss into a hard mapper failure.
                        # Fixing it here is the wrong layer; the real fix is
                        # upstream, where the relationship that would have given
                        # this table a foreign-key column should have been
                        # extracted. Until then, say exactly which facts are being
                        # deleted so the loss is diagnosable in one log line
                        # instead of surfacing as a missing Stage 3 constraint.
                        logger.warning(
                            "  [Mapper] Dropping hollow table '%s' DESTROYS the only "
                            "representation of fact(s) %s -- no other table carries "
                            "them, and no constraint referencing this table can be "
                            "extracted downstream. It has only a primary key because "
                            "the extraction gave it neither attributes nor a "
                            "relationship.",
                            t.name,
                            sorted(exclusive),
                        )
                    else:
                        logger.warning(
                            "  [Mapper] Dropping hollow table '%s': primary key only, "
                            "no other columns, nothing references it, and its fact(s) "
                            "%s are carried by other tables.",
                            t.name,
                            sorted(own_facts) or "[]",
                        )
                    continue

                if t.name in seen_t:
                    logger.warning(
                        "  [Mapper] Dropping duplicate table '%s'; keeping the first "
                        "occurrence. Columns unique to this copy are lost.",
                        t.name,
                    )
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
        logger.info("  [Mapper] STYLE: %s", w)

    return schema
