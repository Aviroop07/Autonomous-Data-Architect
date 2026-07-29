"""Name-blind structural schema evaluation.

The metrics this replaces were all keyed on NAMES: Table F1 matched table names,
Attr F1 matched column names, and PK/FK/DT accuracy were computed only over pairs
that had already matched by name. So the most arbitrary property of a schema decides
every number, and the fuzzy name matcher that props this up demonstrably cannot
separate synonyms from unrelated words -- measured on its own matcher,
DRUG/MEDICATION scores 0.839 and matches while MEMBER/PATRON scores 0.505 and
PRODUCT/STYLE 0.382 are counted as misses, yet the unrelated DRUG/PATIENT scores
0.483, above one of those real synonyms. No threshold separates the classes
because their ranges overlap.

This module answers a different question: does the predicted schema have the same
SHAPE as the ground truth? Tables are aligned by their structural role in the
foreign-key graph rather than by what they are called, and the metrics then score
topology and column composition over that alignment. Two consequences worth
stating:

  - Renaming every table cannot change any score here.
  - A different but equally valid normalisation is not automatically punished.
    A live hospital run produced 12 tables in one configuration and 11 in
    another with IDENTICAL 100% fact coverage -- a junction decomposed
    differently. Name-set F1 read that as a regression; structural alignment
    charges only for the topology that actually differs.

This REPLACES the name-based metrics rather than supplementing them; they have
been deleted. See docs/design/EVALUATION_METRICS.md for the whole metric set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.util.schema_model.schema import Schema

# Lifted from the deleted name-based evaluator: coarsening a declared type to a
# family is the one part of it that never read a name.
COARSE_DT_MAP: Dict[str, str] = {
    "INT": "NUMERIC",
    "INTEGER": "NUMERIC",
    "BIGINT": "NUMERIC",
    "SMALLINT": "NUMERIC",
    "TINYINT": "NUMERIC",
    "FLOAT": "NUMERIC",
    "DOUBLE": "NUMERIC",
    "REAL": "NUMERIC",
    "DECIMAL": "NUMERIC",
    "NUMERIC": "NUMERIC",
    "NUMBER": "NUMERIC",
    "VARCHAR": "TEXT",
    "CHAR": "TEXT",
    "TEXT": "TEXT",
    "STRING": "TEXT",
    "NVARCHAR": "TEXT",
    "DATE": "DATETIME",
    "DATETIME": "DATETIME",
    "TIMESTAMP": "DATETIME",
    "TIME": "DATETIME",
    "BLOB": "BINARY",
    "BINARY": "BINARY",
    "BYTEA": "BINARY",
    "VARBINARY": "BINARY",
    "BOOLEAN": "BOOL",
    "BOOL": "BOOL",
    "BIT": "BOOL",
}


def coarsen_dt(dt: Optional[str]) -> str:
    if not dt:
        return "TEXT"
    base = str(dt).upper().split("(")[0].strip()
    return COARSE_DT_MAP.get(base, "TEXT")


# Feature weights for the alignment cost. Topology is weighted above size
# because a table's position in the FK graph identifies it far more reliably
# than how many columns someone gave it.
_W_OUT_DEGREE = 3.0
_W_IN_DEGREE = 3.0
_W_PK_ARITY = 2.0
_W_TYPES = 2.0
_W_N_COLUMNS = 1.0

# A ground-truth table counts as recovered only if its aligned partner is at
# least this structurally similar. Without a floor, the assignment always pairs
# everything with something and recall becomes meaningless.
_ALIGNMENT_FLOOR = 0.5


@dataclass(frozen=True)
class TableSignature:
    """What a table looks like with its name removed."""

    n_columns: int
    pk_arity: int
    out_degree: int  # foreign keys this table declares
    in_degree: int  # foreign keys pointing at this table
    type_counts: Tuple[Tuple[str, int], ...]  # coarse data type -> count

    @property
    def type_map(self) -> Dict[str, int]:
        return dict(self.type_counts)


@dataclass
class StructuralResult:
    fk_topology_f1: float
    fk_topology_precision: float
    fk_topology_recall: float
    table_structural_recall: float
    column_type_agreement: float
    aligned_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    unaligned_gt: List[str] = field(default_factory=list)
    unaligned_pred: List[str] = field(default_factory=list)

    @property
    def structural_score(self) -> float:
        """Headline figure: topology agreement DISCOUNTED by whether the aligned
        tables actually correspond.

        Topology F1 alone is gameable, and not hypothetically -- reversing the
        one foreign key in a two-table schema turns it into its own mirror image,
        so the optimal alignment simply swaps the two tables and topology scores
        a perfect 1.0. The schema really is isomorphic; it is just not the same
        schema. Column composition is what exposes the swap (0.2 in that case),
        so the product is the honest summary: an alignment earns credit only to
        the extent that the tables it paired resemble each other at all.

        Still entirely name-free -- column TYPE composition, never column names.
        """
        return self.fk_topology_f1 * self.column_type_agreement

    def as_dict(self) -> Dict[str, float]:
        return {
            "structural_score": self.structural_score,
            "fk_topology_f1": self.fk_topology_f1,
            "fk_topology_precision": self.fk_topology_precision,
            "fk_topology_recall": self.fk_topology_recall,
            "table_structural_recall": self.table_structural_recall,
            "column_type_agreement": self.column_type_agreement,
        }


def _signatures(schema: Schema) -> Dict[str, TableSignature]:
    out_deg: Dict[str, int] = {}
    in_deg: Dict[str, int] = {}
    for table in schema.tables:
        out_deg.setdefault(table.name, 0)
        in_deg.setdefault(table.name, 0)
    for fk in schema.relationships or []:
        if fk.referencing_table in out_deg:
            out_deg[fk.referencing_table] += 1
        if fk.referred_table in in_deg:
            in_deg[fk.referred_table] += 1

    sigs: Dict[str, TableSignature] = {}
    for table in schema.tables:
        counts: Dict[str, int] = {}
        for col in table.columns:
            coarse = coarsen_dt(
                col.data_type.value
                if hasattr(col.data_type, "value")
                else str(col.data_type)
            )
            counts[coarse] = counts.get(coarse, 0) + 1
        pk = table.primary_key or []
        sigs[table.name] = TableSignature(
            n_columns=len(table.columns),
            pk_arity=len(pk),
            out_degree=out_deg.get(table.name, 0),
            in_degree=in_deg.get(table.name, 0),
            type_counts=tuple(sorted(counts.items())),
        )
    return sigs


def _ratio(a: int, b: int) -> float:
    """1.0 when equal, decaying to 0.0 as they diverge. Scale-free."""
    if a == b:
        return 1.0
    hi = max(abs(a), abs(b))
    return 1.0 - (abs(a - b) / hi) if hi else 1.0


def _type_similarity(a: TableSignature, b: TableSignature) -> float:
    """Multiset Jaccard over coarse column types."""
    am, bm = a.type_map, b.type_map
    keys = set(am) | set(bm)
    if not keys:
        return 1.0
    inter = sum(min(am.get(k, 0), bm.get(k, 0)) for k in keys)
    union = sum(max(am.get(k, 0), bm.get(k, 0)) for k in keys)
    return inter / union if union else 1.0


def structural_similarity(a: TableSignature, b: TableSignature) -> float:
    """Name-free similarity of two tables in [0, 1]."""
    total = _W_OUT_DEGREE + _W_IN_DEGREE + _W_PK_ARITY + _W_TYPES + _W_N_COLUMNS
    score = (
        _W_OUT_DEGREE * _ratio(a.out_degree, b.out_degree)
        + _W_IN_DEGREE * _ratio(a.in_degree, b.in_degree)
        + _W_PK_ARITY * _ratio(a.pk_arity, b.pk_arity)
        + _W_TYPES * _type_similarity(a, b)
        + _W_N_COLUMNS * _ratio(a.n_columns, b.n_columns)
    )
    return score / total


def align_tables(
    pred: Schema,
    gt: Schema,
    name_tiebreak: bool = True,
) -> Tuple[Dict[str, str], List[Tuple[str, str, float]]]:
    """Map ground-truth table name -> predicted table name, structurally.

    Uses optimal assignment (Hungarian) rather than a greedy pass, so the result
    depends only on the two schemas and never on iteration order -- the same
    reproducibility property the name-based evaluator had to be fixed for.

    `name_tiebreak` adds a tiny bonus for identical names. It cannot change which
    tables are structurally comparable; it only decides between candidates that
    are otherwise indistinguishable, which is common in schemas full of
    similarly-shaped lookup tables.
    """
    pred_sigs = _signatures(pred)
    gt_sigs = _signatures(gt)
    pred_names = sorted(pred_sigs)
    gt_names = sorted(gt_sigs)
    if not pred_names or not gt_names:
        return {}, []

    sim = np.zeros((len(gt_names), len(pred_names)), dtype=float)
    for i, g in enumerate(gt_names):
        for j, p in enumerate(pred_names):
            s = structural_similarity(gt_sigs[g], pred_sigs[p])
            if name_tiebreak and g.lower() == p.lower():
                s += 1e-6
            sim[i, j] = s

    rows, cols = linear_sum_assignment(-sim)
    mapping: Dict[str, str] = {}
    pairs: List[Tuple[str, str, float]] = []
    for i, j in zip(rows, cols):
        score = float(sim[i, j])
        pairs.append((gt_names[i], pred_names[j], score))
        if score >= _ALIGNMENT_FLOOR:
            mapping[gt_names[i]] = pred_names[j]
    return mapping, sorted(pairs, key=lambda t: t[0])


def _fk_edges(schema: Schema) -> Set[Tuple[str, str]]:
    return {
        (fk.referencing_table, fk.referred_table) for fk in (schema.relationships or [])
    }


def evaluate_structural(pred: Schema, gt: Schema) -> StructuralResult:
    """Score a predicted schema's shape against a ground-truth schema's shape."""
    mapping, pairs = align_tables(pred, gt)
    gt_names = {t.name for t in gt.tables}
    pred_names = {t.name for t in pred.tables}

    table_recall = len(mapping) / len(gt_names) if gt_names else 0.0

    # FK topology, translated into predicted-schema names through the alignment.
    gt_edges = _fk_edges(gt)
    pred_edges = _fk_edges(pred)
    translated: Set[Tuple[str, str]] = set()
    untranslatable = 0
    for a, b in gt_edges:
        if a in mapping and b in mapping:
            translated.add((mapping[a], mapping[b]))
        else:
            untranslatable += 1

    hits = len(translated & pred_edges)
    # An edge whose endpoints did not align cannot be credited, so it counts
    # against recall rather than vanishing from the denominator.
    recall_denom = len(translated) + untranslatable
    precision = hits / len(pred_edges) if pred_edges else (1.0 if not gt_edges else 0.0)
    recall = hits / recall_denom if recall_denom else (1.0 if not pred_edges else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    pred_sigs = _signatures(pred)
    gt_sigs = _signatures(gt)
    agreements = [
        _type_similarity(gt_sigs[g], pred_sigs[p]) for g, p in mapping.items()
    ]
    col_agreement = float(np.mean(agreements)) if agreements else 0.0

    return StructuralResult(
        fk_topology_f1=f1,
        fk_topology_precision=precision,
        fk_topology_recall=recall,
        table_structural_recall=table_recall,
        column_type_agreement=col_agreement,
        aligned_pairs=pairs,
        unaligned_gt=sorted(gt_names - set(mapping)),
        unaligned_pred=sorted(pred_names - set(mapping.values())),
    )
