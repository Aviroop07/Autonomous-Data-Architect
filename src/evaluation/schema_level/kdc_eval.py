"""Key and Dependency Correctness: is the schema actually normalised?

This is the check a database reviewer weights above everything else, because a
partial dependency does not announce itself at review time -- it shows up much
later as an update anomaly in production, where one logical edit has to touch
many rows and eventually touches only some of them.

It needs functional dependencies, and the useful realisation is that the pipeline
already derives them: Stage 2's conceptual model carries
`functional_dependencies`, and they now survive the merge. So this asks a
self-consistency question that requires NO ground-truth schema --

    the extractor asserted these dependencies from the text; did the mapper then
    lay out keys such that they actually hold?

-- which is the same property that makes IC-Recall usable on any spec someone
writes rather than only on authored benchmark cases.

Four violations, kept separate because they are not equally serious:

  `unenforced`   the determinant is not a superkey, so the database cannot
                 enforce the dependency at all. The most serious: the schema
                 permits states the specification forbids.
  `partial_2nf`  a non-key attribute depends on part of a composite key. The
                 classic update-anomaly generator.
  `transitive_3nf` a non-key attribute depends on another non-key attribute.
                 Redundancy that drifts out of agreement with itself.
  `cross_table`  the dependency spans two tables, so no single table's key
                 structure can enforce it. Often legitimate after
                 decomposition, so it is reported and NOT counted as a defect.

A schema with no dependencies to check scores 1.0 vacuously, which is honest --
there is nothing to get wrong -- but `n_checked` is reported so a vacuous 1.0 is
never mistaken for a demonstrated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from src.util.schema_model.schema import Schema

Violation = Tuple[str, str]  # (kind, human-readable detail)


@dataclass
class KDCResult:
    kdc: float
    unenforced: List[str] = field(default_factory=list)
    partial_2nf: List[str] = field(default_factory=list)
    transitive_3nf: List[str] = field(default_factory=list)
    cross_table: List[str] = field(default_factory=list)
    tables_without_key: List[str] = field(default_factory=list)
    n_checked: int = 0

    @property
    def n_violations(self) -> int:
        # cross_table is deliberately excluded: it is usually a legitimate
        # consequence of decomposition, not an error.
        return len(self.unenforced) + len(self.partial_2nf) + len(self.transitive_3nf)

    def as_dict(self) -> Dict[str, float]:
        return {
            "kdc": self.kdc,
            "kdc_unenforced": float(len(self.unenforced)),
            "kdc_partial_2nf": float(len(self.partial_2nf)),
            "kdc_transitive_3nf": float(len(self.transitive_3nf)),
            "kdc_tables_without_key": float(len(self.tables_without_key)),
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


def evaluate_kdc(schema: Schema, fds: Iterable[object]) -> KDCResult:
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

        table = next(iter(det_tables))
        if table not in cols:
            # The entity did not survive mapping; that is a capacity problem,
            # reported by IC, not a normalisation one. Do not count it here.
            continue

        checked += 1
        det_cols = {c.lower() for _, c in det_pairs}
        dep_cols = {c.lower() for _, c in dep_pairs}
        pk = pks.get(table, set())
        table_cols = cols[table]

        # Only columns that actually exist can participate.
        det_cols &= table_cols
        dep_cols &= table_cols
        if not det_cols or not dep_cols:
            continue

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
        unenforced=sorted(unenforced),
        partial_2nf=sorted(partial),
        transitive_3nf=sorted(transitive),
        cross_table=sorted(cross),
        tables_without_key=tables_without_key,
        n_checked=checked,
    )
    result.kdc = 1.0 - (result.n_violations / checked) if checked else 1.0
    return result


def evaluate_kdc_from_conceptual(schema: Schema, conceptual: object) -> KDCResult:
    """Convenience wrapper: pull the dependencies off a ConceptualModel."""
    fds: Sequence[object] = getattr(conceptual, "functional_dependencies", None) or []
    return evaluate_kdc(schema, fds)
