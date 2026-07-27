"""Bottom-up effective-schema synthesis for a Relation tree (Section 4).

Each operator's validity requires its operand(s) already valid; on success
it synthesizes an EffectiveSchema for its parent (columns/types/nullability,
effective PK, column-level FK/PK provenance, and a row-count variable
descriptor). Synthesis follows the project's non-raising validator
convention: failure is reported as a List[str], never an exception -- a
None EffectiveSchema paired with a non-empty error list means "this operator
is not valid, nothing to build on top of it."

Deliberately NOT this module's job (see relation/validate.py, task #27):
enforcing that a Join's FK-PK direction is legitimate beyond what's needed
to synthesize nullability/PK here, PK-never-dropped-by-Project as a hard
rejection, and alias-collision/ambiguous-qualifier rejection beyond the bare
minimum needed to resolve a JoinCondition's own qualifiers.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.pipeline.stage2.models.data_types import DataType
from src.pipeline.stage2.models.schema import Column, ForeignKey, Schema, Table
from src.util.constraint_model.condition.expressions import (
    RColumnRef,
    TypeMismatch,
    infer_type,
)
from src.util.constraint_model.condition.predicates import (
    RAnd,
    RBetween,
    RComparison,
    RInSet,
    RNot,
    RNotInSet,
    RPredicateUnion,
    extract_columns,
)
from src.util.constraint_model.relation.nodes import (
    Aggregate,
    BaseTable,
    Fanout,
    Filter,
    Join,
    Project,
    RawSQL,
    RelationUnion,
    extract_base_tables,
)

_COUNT_LIKE_FNS = frozenset({"COUNT", "COUNT_DISTINCT"})
_TYPE_PRESERVING_FNS = frozenset({"MAX", "MIN", "MEDIAN", "PERCENTILE", "MODE"})
FANOUT_CHILD_COUNT_COLUMN = "child_count"


class ColumnProvenance(BaseModel):
    """Lineage back to the real, schema-declared FK/PK a column traces to.

    `is_primary_key` and `referred_table` are independent: a weak-entity /
    identifying-relationship key column (e.g. a junction table whose
    composite PK is made up of its parents' FKs) is legitimately both a PK
    member AND a foreign key at once. Modeling them as a single `kind`
    enum would force an either/or that such columns violate by definition."""

    is_primary_key: bool = False
    table: str = Field(description="The base table this key belongs to.")
    column: str = Field(description="The real column name on that base table.")
    referred_table: Optional[str] = Field(
        default=None,
        description="Set when this column is a foreign key: the table it references.",
    )


class EffectiveColumn(BaseModel):
    data_type: DataType
    nullable: bool
    provenance: Optional[ColumnProvenance] = None
    ambiguous: bool = Field(
        default=False,
        description=(
            "True when two same-named columns merged from both sides of a Join "
            "without a distinguishing alias -- the name is permanently "
            "unresolvable unqualified until renamed (known limitation, matches "
            "the old grain.py precedent; not a synthesis-blocking error here)."
        ),
    )


class RowCountVar(BaseModel):
    """A structural descriptor of how this Relation's row count relates to
    its operand(s) (Section 4.2) -- NOT a DOF variable itself. Turning this
    into a real, solvable variable is variables.py's job (task #31); this is
    deliberately just the structural shape, since the object model only
    needs the math settled at this layer (Section 4.2's own explicit
    deferral)."""

    name: str = Field(description="Canonical name for this row-count quantity.")
    kind: Literal["free", "identity", "filtered", "grouped"]
    equals: Optional[str] = Field(
        default=None, description="kind='identity': the operand row-count var name."
    )
    source: Optional[str] = Field(
        default=None,
        description="kind='filtered'/'grouped': the source row-count var name.",
    )
    selectivity: Optional[str] = Field(
        default=None,
        description="kind='filtered': the selectivity-factor variable name.",
    )


class EffectiveSchema(BaseModel):
    columns: Dict[str, EffectiveColumn]
    primary_key: List[str]
    row_count: RowCountVar


class NodeNamer:
    """Assigns deterministic, unique names to anonymous (alias-less)
    derived Relation nodes, for row-count variable naming. BaseTable and
    any node with an explicit alias just use that name directly.

    Memoized by node object identity: asking for the same node object's
    name twice within one NodeNamer's lifetime returns the same name
    rather than consuming another counter tick. This matters whenever the
    same node is (re-)synthesized more than once against the SAME namer
    instance (see synthesize_schema_tree below) -- without memoization,
    the second pass would silently mint different names for identical
    nodes, breaking every 'identity'/'filtered' RowCountVar's equals/
    source string reference."""

    def __init__(self) -> None:
        self._counter = 0
        self._cache: dict[int, str] = {}

    def name_for(self, node: "RelationUnion") -> str:
        alias = getattr(node, "alias", None)
        if alias:
            return alias
        if isinstance(node, BaseTable):
            return node.name
        key = id(node)
        if key in self._cache:
            return self._cache[key]
        self._counter += 1
        name = f"_{type(node).__name__.lower()}_{self._counter}"
        self._cache[key] = name
        return name


def _base_provenance(
    table: Table, column: Column, fks: List[ForeignKey]
) -> Optional[ColumnProvenance]:
    is_pk = column.name in table.pk_set
    referred_table = next(
        (
            fk.referred_table
            for fk in fks
            if fk.referencing_table == table.name
            and fk.referencing_column == column.name
        ),
        None,
    )
    if not is_pk and referred_table is None:
        return None
    return ColumnProvenance(
        is_primary_key=is_pk,
        referred_table=referred_table,
        table=table.name,
        column=column.name,
    )


def _relation_qualifiers(node: "RelationUnion") -> set[str]:
    """Valid table-qualifiers a JoinCondition may use to refer to `node`:
    its own alias if set (SQL rule: once aliased, only the alias is valid),
    else any base table name reachable from it."""
    alias = getattr(node, "alias", None)
    if alias:
        return {alias}
    return extract_base_tables(node)


def _fk_pk_direction(
    left_col: str,
    left_eff: EffectiveSchema,
    right_col: str,
    right_eff: EffectiveSchema,
) -> Optional[str]:
    """Returns 'left_is_child' or 'right_is_child' if column provenance
    resolves one side as a genuine FK to the other side's real PK table,
    else None (full legitimacy enforcement is relation/validate.py's job;
    this is the bare minimum schema synthesis needs to proceed)."""
    lp = left_eff.columns[left_col].provenance
    rp = right_eff.columns[right_col].provenance
    if (
        lp is not None
        and lp.referred_table is not None
        and rp is not None
        and rp.is_primary_key
        and lp.referred_table == rp.table
    ):
        return "left_is_child"
    if (
        rp is not None
        and rp.referred_table is not None
        and lp is not None
        and lp.is_primary_key
        and rp.referred_table == lp.table
    ):
        return "right_is_child"
    return None


def _narrowed_columns(node: "RPredicateUnion") -> set[str]:
    """Columns guaranteed non-null after this Filter's condition passes,
    per the full three-valued-logic rule (Section 4.4): propagates through
    RAnd, a single RComparison/RBetween/RInSet/RNotInSet, and RNot -- NOT
    through ROr or RIfThen (neither guarantees anything unconditionally)."""
    if isinstance(node, RAnd):
        cols: set[str] = set()
        for op in node.operands:
            cols |= _narrowed_columns(op)
        return cols
    if isinstance(node, RNot):
        return _narrowed_columns(node.operand)
    if isinstance(node, (RComparison, RBetween, RInSet, RNotInSet)):
        return extract_columns(node)
    return set()


def synthesize_schema(
    node: "RelationUnion", schema: Schema
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    """Public entry point: bottom-up effective-schema synthesis for `node`
    against the real, schema-declared `schema` (tables + FKs)."""
    return _synth(node, schema, NodeNamer())


def _relation_children(node: "RelationUnion") -> List["RelationUnion"]:
    if isinstance(node, Join):
        return [node.left, node.right]
    if isinstance(node, Fanout):
        # parent_table/child_table are plain strings, not child RelationUnion
        # nodes -- but their own base-table RowCountVars (kind='free') must
        # still be synthesized, since Fanout's own RowCountVar is
        # kind='identity' with equals=f"{parent_table}.row_count": without
        # visiting the parent as a BaseTable here, that identity equation
        # references a variable that was never minted (build_dof_graph then
        # rejects it as undefined).
        return [BaseTable(name=node.parent_table), BaseTable(name=node.child_table)]
    source = getattr(node, "source", None)
    return [source] if source is not None else []


def synthesize_schema_tree(
    node: "RelationUnion", schema: Schema, namer: Optional[NodeNamer] = None
) -> Tuple[Optional[EffectiveSchema], List[RowCountVar], List[str]]:
    """Like synthesize_schema, but ALSO returns every intermediate node's
    own RowCountVar, collected from ONE shared NodeNamer used to synthesize
    `node` itself -- needed by variables.py's DOF-graph bridge, which must
    build a Variable for every name an 'identity'/'filtered' RowCountVar's
    equals/source string references. Calling synthesize_schema()
    independently per node (rather than sharing one namer across the whole
    tree) would mint a DIFFERENT anonymous-node name each time, since the
    counter restarts at each call -- this function exists specifically to
    avoid that trap.

    `namer` defaults to a fresh one (the common case: one relation tree at
    a time). Pass an EXTERNALLY-SHARED namer when synthesizing MULTIPLE
    separate relation trees that must be merged into one combined DOF
    graph (e.g. conflicts/evaluate.py's structural-overconstrained check
    across many Constraints) -- otherwise two unrelated, independently-
    alias-less nodes (e.g. an anonymous Join in Constraint A and an
    unrelated anonymous Join in Constraint B) would coincidentally both
    mint the identical synthetic name (each call's own counter starts back
    at 1), silently merging two DIFFERENT quantities into one DOF
    variable. A shared namer keeps synthetic names globally unique across
    calls while still correctly reusing real table names/aliases (which
    bypass the counter entirely and are SUPPOSED to be shared)."""
    namer = namer if namer is not None else NodeNamer()
    row_counts: List[RowCountVar] = []
    eff, errors = _synth_collecting(node, schema, namer, row_counts)
    return eff, row_counts, errors


def _synth_collecting(
    node: "RelationUnion", schema: Schema, namer: NodeNamer, out: List[RowCountVar]
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    eff, errors = _synth(node, schema, namer)
    if eff is None:
        return None, errors
    out.append(eff.row_count)
    for child in _relation_children(node):
        _, child_errors = _synth_collecting(child, schema, namer, out)
        errors.extend(child_errors)
    return eff, errors


def _synth(
    node: "RelationUnion", schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    if isinstance(node, BaseTable):
        return _synth_base_table(node, schema, namer)
    if isinstance(node, Join):
        return _synth_join(node, schema, namer)
    if isinstance(node, Aggregate):
        return _synth_aggregate(node, schema, namer)
    if isinstance(node, Filter):
        return _synth_filter(node, schema, namer)
    if isinstance(node, Project):
        return _synth_project(node, schema, namer)
    if isinstance(node, Fanout):
        return _synth_fanout(node, schema, namer)
    if isinstance(node, RawSQL):
        return None, [
            "RawSQL nodes cannot be schema-synthesized until relation/sql_bridge.py "
            "normalizes them into structured nodes (not yet built)."
        ]
    return None, [f"Unknown Relation node type: {type(node).__name__}"]


def _synth_base_table(
    node: BaseTable, schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    table = schema.get_table_map().get(node.name)
    if table is None:
        return None, [f"BaseTable: table '{node.name}' not found in schema."]
    fks = schema.relationships or []
    columns = {
        c.name: EffectiveColumn(
            data_type=c.data_type,
            nullable=c.is_nullable,
            provenance=_base_provenance(table, c, fks),
        )
        for c in table.columns
    }
    row_count = RowCountVar(name=f"{namer.name_for(node)}.row_count", kind="free")
    return EffectiveSchema(
        columns=columns, primary_key=list(table.primary_key), row_count=row_count
    ), []


def resolve_join_child(
    node: Join, left_eff: EffectiveSchema, right_eff: EffectiveSchema
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Given the already-synthesized effective schemas of a Join's two
    operands, resolves which side is the FK-holding "child" side (Section
    5's companion rule: the join's effective PK is always the child's own
    PK). Returns (direction, child_fk_column, errors) where direction is
    'left_is_child'/'right_is_child', or (None, None, errors) if it can't
    be resolved. Factored out of _synth_join so population.py's operation-
    history tracking can reuse the exact same resolution logic rather than
    re-deriving it."""
    if len(node.on) != 1:
        return (
            None,
            None,
            [
                "Join.on: composite join conditions (more than one JoinCondition) are "
                "not supported -- composite FKs are an explicit non-goal."
            ],
        )

    cond = node.on[0]
    left_qual, _, left_col = cond.left.partition(".")
    right_qual, _, right_col = cond.right.partition(".")
    left_names = _relation_qualifiers(node.left)
    right_names = _relation_qualifiers(node.right)

    if left_qual in left_names and right_qual in right_names:
        pass
    elif right_qual in left_names and left_qual in right_names:
        left_col, right_col = right_col, left_col
    else:
        return (
            None,
            None,
            [
                f"Join.on[0]: qualifiers '{left_qual}'/'{right_qual}' do not "
                "unambiguously resolve to Join.left/Join.right -- add an explicit alias."
            ],
        )

    errors: List[str] = []
    if left_col not in left_eff.columns:
        errors.append(f"Join.on[0]: column '{left_col}' not found on Join.left.")
    if right_col not in right_eff.columns:
        errors.append(f"Join.on[0]: column '{right_col}' not found on Join.right.")
    if errors:
        return None, None, errors

    direction = _fk_pk_direction(left_col, left_eff, right_col, right_eff)
    if direction is None:
        return (
            None,
            None,
            [
                "Join.on[0]: does not resolve to a genuine FK-PK relationship via "
                "column provenance -- full join legitimacy is enforced in "
                "relation/validate.py, but schema synthesis needs a resolvable "
                "direction to compute nullability/effective PK."
            ],
        )
    return direction, (left_col if direction == "left_is_child" else right_col), []


def _synth_join(
    node: Join, schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    name = namer.name_for(node)
    left_eff, left_errs = _synth(node.left, schema, namer)
    right_eff, right_errs = _synth(node.right, schema, namer)
    errors = list(left_errs) + list(right_errs)
    if left_eff is None or right_eff is None:
        return None, errors

    direction, child_fk_col, direction_errors = resolve_join_child(
        node, left_eff, right_eff
    )
    if direction is None or child_fk_col is None:
        errors.extend(direction_errors)
        return None, errors

    if direction == "left_is_child":
        child_eff, parent_eff = left_eff, right_eff
    else:
        child_eff, parent_eff = right_eff, left_eff

    fk_nullable = child_eff.columns[child_fk_col].nullable
    columns: Dict[str, EffectiveColumn] = dict(child_eff.columns)
    for cname, col in parent_eff.columns.items():
        if cname in columns:
            columns[cname] = EffectiveColumn(
                data_type=columns[cname].data_type,
                nullable=columns[cname].nullable,
                provenance=None,
                ambiguous=True,
            )
            continue
        columns[cname] = col.model_copy(
            update={"nullable": col.nullable or fk_nullable}
        )

    row_count = RowCountVar(
        name=f"{name}.row_count", kind="identity", equals=child_eff.row_count.name
    )
    return EffectiveSchema(
        columns=columns, primary_key=list(child_eff.primary_key), row_count=row_count
    ), []


def _synth_aggregate(
    node: Aggregate, schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    name = namer.name_for(node)
    src_eff, errs = _synth(node.source, schema, namer)
    if src_eff is None:
        return None, errs

    errors: List[str] = []
    group_by = node.group_by or []
    columns: Dict[str, EffectiveColumn] = {}
    for gb in group_by:
        src_col = src_eff.columns.get(gb)
        if src_col is None:
            errors.append(f"Aggregate.group_by: column '{gb}' not found in source.")
            continue
        columns[gb] = src_col

    result_col: Optional[EffectiveColumn] = None
    if node.column == "*":
        result_col = EffectiveColumn(data_type=DataType.INTEGER, nullable=False)
    else:
        src_col = src_eff.columns.get(node.column)
        if src_col is None:
            errors.append(
                f"Aggregate.column: column '{node.column}' not found in source."
            )
        elif node.fn in _COUNT_LIKE_FNS:
            result_col = EffectiveColumn(data_type=DataType.INTEGER, nullable=False)
        elif node.fn in _TYPE_PRESERVING_FNS:
            result_col = EffectiveColumn(data_type=src_col.data_type, nullable=True)
        else:  # SUM, AVG, STDDEV, VARIANCE
            result_col = EffectiveColumn(data_type=DataType.FLOAT, nullable=True)

    if errors:
        return None, errors
    if node.alias in columns:
        return None, [
            f"Aggregate.alias '{node.alias}' collides with a group_by column name."
        ]
    assert result_col is not None
    columns[node.alias] = result_col

    row_count = RowCountVar(
        name=f"{name}.row_count", kind="grouped", source=src_eff.row_count.name
    )
    return EffectiveSchema(
        columns=columns, primary_key=list(group_by), row_count=row_count
    ), []


def _synth_filter(
    node: Filter, schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    name = namer.name_for(node)
    src_eff, errs = _synth(node.source, schema, namer)
    if src_eff is None:
        return None, errs

    narrowed = _narrowed_columns(node.condition)
    columns: Dict[str, EffectiveColumn] = {}
    for cname, col in src_eff.columns.items():
        if cname in narrowed and col.nullable:
            columns[cname] = col.model_copy(update={"nullable": False})
        else:
            columns[cname] = col

    row_count = RowCountVar(
        name=f"{name}.row_count",
        kind="filtered",
        source=src_eff.row_count.name,
        selectivity=f"{name}.selectivity",
    )
    return EffectiveSchema(
        columns=columns, primary_key=list(src_eff.primary_key), row_count=row_count
    ), []


def _synth_project(
    node: Project, schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    name = namer.name_for(node)
    src_eff, errs = _synth(node.source, schema, namer)
    if src_eff is None:
        return None, errs

    errors: List[str] = []
    columns: Dict[str, EffectiveColumn] = {}
    rename_map: Dict[str, str] = {}
    for entry in node.columns:
        out_name = entry.output_name()
        if isinstance(entry.expr, RColumnRef):
            src_col = src_eff.columns.get(entry.expr.name)
            if src_col is None:
                errors.append(
                    f"Project: column '{entry.expr.name}' not found in source."
                )
                continue
            columns[out_name] = src_col
            rename_map[entry.expr.name] = out_name
        else:
            try:
                dtype = infer_type(
                    entry.expr, {n: c.data_type for n, c in src_eff.columns.items()}
                )
            except TypeMismatch as e:
                errors.append(f"Project: {e}")
                continue
            columns[out_name] = EffectiveColumn(data_type=dtype, nullable=True)
    if errors:
        return None, errors

    primary_key = [rename_map[c] for c in src_eff.primary_key if c in rename_map]
    row_count = RowCountVar(
        name=f"{name}.row_count", kind="identity", equals=src_eff.row_count.name
    )
    return EffectiveSchema(
        columns=columns, primary_key=primary_key, row_count=row_count
    ), []


def _synth_fanout(
    node: Fanout, schema: Schema, namer: NodeNamer
) -> Tuple[Optional[EffectiveSchema], List[str]]:
    table_map = schema.get_table_map()
    errors: List[str] = []
    parent = table_map.get(node.parent_table)
    child = table_map.get(node.child_table)
    if parent is None:
        errors.append(
            f"Fanout.parent_table: table '{node.parent_table}' not found in schema."
        )
    if child is None:
        errors.append(
            f"Fanout.child_table: table '{node.child_table}' not found in schema."
        )
    if errors:
        return None, errors
    assert parent is not None and child is not None

    if not any(c.name == node.fk_column for c in child.columns):
        return None, [
            f"Fanout.fk_column: column '{node.fk_column}' not found on table '{node.child_table}'."
        ]

    fks = schema.relationships or []
    columns = {
        c.name: EffectiveColumn(
            data_type=c.data_type,
            nullable=c.is_nullable,
            provenance=_base_provenance(parent, c, fks),
        )
        for c in parent.columns
    }
    columns[FANOUT_CHILD_COUNT_COLUMN] = EffectiveColumn(
        data_type=DataType.INTEGER, nullable=False
    )

    name = namer.name_for(node)
    row_count = RowCountVar(
        name=f"{name}.row_count", kind="identity", equals=f"{parent.name}.row_count"
    )
    return EffectiveSchema(
        columns=columns, primary_key=list(parent.primary_key), row_count=row_count
    ), []
