"""Key and Dependency Correctness, and an internal-consistency diagnostic.

TWO MODES, and they are NOT interchangeable. The difference is where the
functional dependencies come from, and it decides whether the number is a
quality metric or merely a diagnostic.

`source="ground_truth"` -- METRIC.
    Dependencies authored from the specification, independent of the pipeline.
    This measures normalisation quality against a standard: are the keys laid
    out so the dependencies the domain actually has are enforced? This is the
    check a database reviewer weights above all others, because a partial
    dependency does not announce itself at review time -- it surfaces later as
    an update anomaly, where one logical edit must touch many rows and
    eventually touches only some of them.

`source="pipeline"` -- DIAGNOSTIC ONLY, NOT A METRIC.
    Dependencies the pipeline derived itself, in Stage 2's conceptual model.
    Scoring the pipeline against its own assertions is CIRCULAR: the extractor
    can reach a perfect score by asserting no dependencies at all, or by
    asserting only ones the mapper already satisfies. Nothing here constrains
    it toward the truth. What it does detect honestly is DISAGREEMENT between
    the extractor and the mapper -- the extractor asserted a dependency and the
    mapper laid out keys that cannot enforce it -- which is a real bug class and
    worth surfacing. It is reported under `internal_fd_consistency`, never
    under `kdc`, so a circular number can never be mistaken for a measured one.

Ground-truth dependencies do not yet exist in cases.jsonl, so today only the
diagnostic runs. Adding them is a contract change, noted in
docs/design/EVALUATION_METRICS.md.

Violations, kept separate because they are not equally serious:

  `unenforced`     the determinant is not a superkey, so the database cannot
                   enforce the dependency at all -- the schema permits states
                   the specification forbids. The most serious.
  `partial_2nf`    a non-key attribute depends on part of a composite key. The
                   classic update-anomaly generator.
  `transitive_3nf` a non-key attribute depends on another non-key attribute.
                   Redundancy that drifts out of agreement with itself.
  `cross_table`    the dependency spans two tables, so no single table's keys
                   can enforce it. Usually a legitimate consequence of
                   decomposition, so it is REPORTED and NOT counted a defect.

`n_checked` is always reported, so a vacuous 1.0 on a schema with nothing to
check is never mistaken for an earned one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.util.schema_model.schema import Schema

Violation = Tuple[str, str]  # (kind, human-readable detail)


@dataclass
class KDCResult:
    """`source` records where the dependencies came from -- see the module
    docstring. It decides which key the score is reported under."""

    kdc: Optional[float]
    source: str = "pipeline"
    unenforced: List[str] = field(default_factory=list)
    partial_2nf: List[str] = field(default_factory=list)
    transitive_3nf: List[str] = field(default_factory=list)
    cross_table: List[str] = field(default_factory=list)
    tables_without_key: List[str] = field(default_factory=list)
    unresolved_tables: List[str] = field(default_factory=list)
    n_checked: int = 0

    @property
    def n_violations(self) -> int:
        # cross_table is deliberately excluded: it is usually a legitimate
        # consequence of decomposition, not an error.
        return len(self.unenforced) + len(self.partial_2nf) + len(self.transitive_3nf)

    def as_dict(self) -> Dict[str, Optional[float]]:
        # A circular score must never be published under the metric's name.
        key = "kdc" if self.source == "ground_truth" else "internal_fd_consistency"
        return {
            key: self.kdc,
            "kdc_unenforced": float(len(self.unenforced)),
            "kdc_partial_2nf": float(len(self.partial_2nf)),
            "kdc_transitive_3nf": float(len(self.transitive_3nf)),
            "kdc_tables_without_key": float(len(self.tables_without_key)),
            "kdc_unresolved_tables": float(len(self.unresolved_tables)),
            "kdc_n_checked": float(self.n_checked),
        }


def _split_qualified(ref: str) -> Tuple[str, str] | None:
    """ "TABLE.column" -> (TABLE, column); None if not qualified that way."""
    if not isinstance(ref, str) or ref.count(".") != 1:
        return None
    table, column = ref.split(".", 1)
    return (table, column) if table and column else None


def _normalise_fd(fd: object) -> Tuple[List[str], List[str]] | None:
    det = getattr(fd, "determinant", None)
    dep = getattr(fd, "dependent", None)
    if det is None and isinstance(fd, dict):
        det, dep = fd.get("determinant"), fd.get("dependent")
    if not det or not dep:
        return None
    return [str(x) for x in det], [str(x) for x in dep]


def _table_index(schema: Schema) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Return {table: columns} and {table: primary key columns}, upper-cased keys."""
    cols: Dict[str, Set[str]] = {}
    pks: Dict[str, Set[str]] = {}
    for table in schema.tables:
        cols[table.name.upper()] = {c.name.lower() for c in table.columns}
        pks[table.name.upper()] = {p.lower() for p in (table.primary_key or [])}
    return cols, pks


def _resolve_table(
    name: str,
    cols: Dict[str, Set[str]],
    name_map: Optional[Mapping[str, str]],
) -> Optional[str]:
    """Find the schema table a reference dependency is talking about.

    Direct hit first, then the caller-supplied structural alignment, which is
    what makes this name-blind like the rest of the suite.
    """
    if name in cols:
        return name
    if name_map:
        mapped = name_map.get(name) or name_map.get(name.upper())
        if mapped and mapped.upper() in cols:
            return mapped.upper()
    return None


def evaluate_kdc(
    schema: Schema,
    fds: Iterable[object],
    source: str = "pipeline",
    name_map: Optional[Mapping[str, str]] = None,
) -> KDCResult:
    """Check each functional dependency against the schema's key structure.

    `fds` are the conceptual model's dependencies, whose determinants and
    dependents are qualified `ENTITY.attribute`. Entity names become table names
    upper-cased, which is the mapper's own convention, so a dependency naming an
    entity that did not survive mapping is reported rather than silently skipped.
    """
    cols, pks = _table_index(schema)

    unenforced: List[str] = []
    partial: List[str] = []
    transitive: List[str] = []
    cross: List[str] = []
    unresolved: List[str] = []
    checked = 0

    for raw in fds:
        parsed = _normalise_fd(raw)
        if parsed is None:
            continue
        det_refs, dep_refs = parsed

        det_pairs = [p for p in (_split_qualified(r) for r in det_refs) if p]
        dep_pairs = [p for p in (_split_qualified(r) for r in dep_refs) if p]
        if not det_pairs or not dep_pairs:
            continue

        det_tables = {t.upper() for t, _ in det_pairs}
        dep_tables = {t.upper() for t, _ in dep_pairs}
        label = f"{'+'.join(det_refs)} -> {'+'.join(dep_refs)}"

        if len(det_tables | dep_tables) > 1:
            cross.append(label)
            continue

        table = _resolve_table(next(iter(det_tables)), cols, name_map)
        if table is None:
            # NOT the same thing as "the entity did not survive". A table that
            # was merely RENAMED is still there, and renaming is exactly what
            # Stage 2 does: a benchmark run produced SITE and FAULT_RECORD where
            # the reference says SOLAR_FARM and FAULT_EVENT. Every other metric
            # in this suite is name-blind for that reason; this one was not, so
            # its dependencies silently left the denominator and the score came
            # back a flattering 1.000 having checked NOTHING.
            #
            # `name_map` carries the structural alignment the caller already
            # computed, so a rename resolves. What remains unresolved is
            # recorded rather than dropped.
            unresolved.append(label)
            continue

        det_cols = {c.lower() for _, c in det_pairs}
        dep_cols = {c.lower() for _, c in dep_pairs}
        pk = pks.get(table, set())
        table_cols = cols[table]

        # Only columns that actually exist can participate.
        det_cols &= table_cols
        dep_cols &= table_cols
        if not det_cols or not dep_cols:
            # COUNTED AS UNRESOLVED, NOT AS CHECKED. The table was found but its
            # COLUMNS were not, which happens because the alignment this metric
            # borrows is table-level -- there is no column-level alignment to
            # borrow. Case 104's reference dependency names fault_code and
            # fault_wording where the generated table has code and
            # code_description.
            #
            # `checked` used to be incremented ABOVE this point, so four
            # dependencies none of which could be evaluated produced
            # n_checked=4 and kdc=1.0 -- a number that looks earned and is not.
            # The counter now means what it says: evaluated.
            unresolved.append(label)
            continue

        checked += 1

        non_key_dep = dep_cols - pk
        if not non_key_dep:
            # Determining part of the key is not a normalisation problem.
            continue

        if pk and det_cols >= pk:
            # Determinant is a superkey: the dependency is enforced. Nothing else
            # to check -- 2NF and 3NF violations are by definition about
            # determinants that are NOT superkeys.
            continue

        if pk and det_cols < pk:
            partial.append(f"{table}: {label} (determinant is part of the key)")
        elif pk and not (det_cols & pk):
            transitive.append(f"{table}: {label} (non-key determines non-key)")
        else:
            unenforced.append(f"{table}: {label} (determinant is not a superkey)")

    tables_without_key = sorted(t for t, pk in pks.items() if not pk)

    result = KDCResult(
        kdc=0.0,
        source=source,
        unenforced=sorted(unenforced),
        partial_2nf=sorted(partial),
        transitive_3nf=sorted(transitive),
        cross_table=sorted(cross),
        tables_without_key=tables_without_key,
        unresolved_tables=sorted(unresolved),
        n_checked=checked,
    )
    # UNDEFINED, not 1.0, when nothing was checked. "1 - violations/0" has no
    # value, and reporting a perfect score for an empty check is the single most
    # misleading thing this module could do -- it is how a case with a real 3NF
    # violation in its reference dependencies scored 1.000.
    result.kdc = (1.0 - result.n_violations / checked) if checked else None
    return result


def evaluate_internal_fd_consistency(
    schema: Schema, conceptual: object
) -> KDCResult:
    """DIAGNOSTIC: does the mapper's key layout honour the extractor's own FDs?

    Explicitly named for what it is. This is not KDC -- see the module docstring
    on why scoring the pipeline against its own assertions is circular.
    """
    fds: Sequence[object] = getattr(conceptual, "functional_dependencies", None) or []
    return evaluate_kdc(schema, fds, source="pipeline")
